import numpy as np


class TransportDynamics:
    """
    Physical tape transport simulation.

    Reel inertia model
    ------------------
    A real tape reel is a spinning mass. Its angular momentum resists speed changes.
    When the motor changes voltage the capstan speed does NOT change instantly —
    it accelerates/decelerates according to:

        I * dω/dt = T_motor - T_friction - T_load

    We model this as a first-order lag on the target speed whose time constant
    is proportional to the effective reel moment of inertia, which itself changes
    as the tape transfers from supply to take-up (more tape on a reel = more inertia).

    The 'pitch_speed' parameter acts like a direct capstan drive voltage increase —
    it raises the target speed above 1.0, making the tape run fast (pitch/speed up).
    Torque load ('motor_drag') lowers it below 1.0, making it slow down.
    Both changes are subject to the same inertial lag, so speed changes are never
    instantaneous — they ramp up/down over time exactly like a real machine.
    """

    # NAB 10.5" reel geometry
    REEL_RADIUS_FULL = 0.133   # m
    REEL_RADIUS_HUB  = 0.025   # m
    TAPE_THICKNESS   = 12e-6   # m
    # Reel + flange mass approximately 0.35 kg, r_gyration ≈ 0.08 m
    # I_empty ≈ 0.35 * 0.08^2 ≈ 0.00224 kg·m²
    # Full reel adds ~0.5 kg tape at mean r ≈ 0.08 m → I_tape ≈ 0.0032
    I_REEL_EMPTY = 0.00224
    I_REEL_FULL  = 0.00224 + 0.0032   # = 0.00544 kg·m²

    def __init__(self):
        self.current_motor_speed = 1.0
        self.surge_state         = 0.0
        self.sticky_drag         = 0.0
        self.last_instant_speed  = 1.0
        self.last_conflict        = 0.0
        self.dropout_count        = 0
        self.last_dropout_time    = -999.0

        self._roller_phase = np.random.uniform(0, 2 * np.pi)
        self._roller_ecc   = np.random.uniform(0.0002, 0.0008)
        self._supply_phase = np.random.uniform(0, 2 * np.pi)
        self._tension_arm  = 0.5
        self._tension_vel  = 0.0

        self._next_dropout  = 0.0   # fire immediately on first block, schedule from there
        self._dropout_timer = 0.0

    def reset(self):
        self.current_motor_speed = 1.0
        self.surge_state         = 0.0
        self.sticky_drag         = 0.0
        self.last_instant_speed  = 1.0
        self._tension_arm        = 0.5
        self._tension_vel        = 0.0
        self._next_dropout       = 0.0
        self._dropout_timer      = 0.0
        self.dropout_count       = 0
        self.last_dropout_time   = -999.0

    def _schedule_dropout(self, rate=0.1):
        # Mean interval shrinks as rate increases:
        # rate=0.01 → avg every ~100s,  rate=1.0 → avg every ~1s
        mean_interval = max(0.3, 8.0 / max(rate, 0.001))
        return np.random.exponential(scale=mean_interval)

    def _reel_inertia(self, progress):
        """
        Effective moment of inertia of the combined supply+take-up reel system
        as a function of tape progress (0=start, 1=end).
        Supply reel loses mass, take-up gains it; total inertia has a slight
        minimum near the midpoint (both reels half-full).
        """
        p = np.clip(progress, 0.0, 1.0)
        # Supply: starts full (I_FULL), ends empty (I_EMPTY)
        I_supply = self.I_REEL_EMPTY + (self.I_REEL_FULL - self.I_REEL_EMPTY) * (1.0 - p)
        # Take-up: starts empty, ends full
        I_takeup = self.I_REEL_EMPTY + (self.I_REEL_FULL - self.I_REEL_EMPTY) * p
        return I_supply + I_takeup

    def _reel_geometry(self, progress):
        p          = np.clip(progress, 0.0, 1.0)
        full_area  = np.pi * (self.REEL_RADIUS_FULL**2 - self.REEL_RADIUS_HUB**2)
        supply_r   = np.sqrt(full_area * (1.0 - p) / np.pi + self.REEL_RADIUS_HUB**2)
        takeup_r   = np.sqrt(full_area * p           / np.pi + self.REEL_RADIUS_HUB**2)
        return supply_r, takeup_r

    def process_speed(self, frames, current_time, play_head, total_samples, params):
        SR       = 44100.0
        t_arr    = current_time + np.arange(frames) / SR
        progress = float(play_head) / max(total_samples, 1)

        ips = params.get('ips_base', 15.0)

        # --- Reel geometry --------------------------------------------------
        supply_r, takeup_r = self._reel_geometry(progress)

        # --- Inertia-based speed lag -----------------------------------------
        # Time constant τ = I / (k_motor * r_capstan²)
        # We normalise so that at default load τ ≈ 0.3 s (perceptible but not annoying)
        I_eff         = self._reel_inertia(progress)
        # Normalise against I at 50% progress so τ_base is consistent
        I_ref         = self._reel_inertia(0.5)
        inertia_ratio = I_eff / I_ref
        # Base lag coefficient: lower = slower response (more inertia)
        # 0.0005 per sample at 44100 Hz ≈ τ ≈ 0.045 s at I_ref
        # Multiplied by inertia_ratio: heavier reels respond slower
        lag_coeff = 0.0005 / inertia_ratio   # slows down when reels are heavy

        # --- Target speed ---------------------------------------------------
        # motor_boost > 0 raises target above 1.0 (mirror of motor_drag)
        # motor_drag  > 0 lowers target below 1.0
        # Both are subject to the same inertial lag — no instant pitch jump
        boost      = params.get('motor_boost', 0.0)
        drag       = params.get('motor_drag', 0.0)

        # Sticky shed builds up cumulative drag
        # Sticky drag approaches a maximum determined by sticky_shed intensity.
        # Cap at 0.015 to prevent the azimuth cutoff going negative (which causes
        # coefficient changes every block → pops). The electronics module uses
        # (1 - sticky_drag * 50), so 0.02 = complete blockage; we stay just under.
        shed = params.get('sticky_shed', 0.0)
        target_sticky = shed * 0.015          # max drag at full shed = 0.015
        self.sticky_drag += (target_sticky - self.sticky_drag) * 0.0002

        # Back-tension from take-up reel (increases as reel fills)
        back_tension = (takeup_r / self.REEL_RADIUS_FULL) * params.get('tension_load', 0.05) * 0.5

        # Tension arm spring-mass oscillator
        tension_k = 0.3
        tension_b = 0.05
        tension_f = back_tension - self._tension_arm * tension_k - self._tension_vel * tension_b
        self._tension_vel += tension_f * (frames / SR)
        self._tension_arm  = np.clip(self._tension_arm + self._tension_vel * (frames / SR), 0, 1)
        tension_mod        = self._tension_arm * params.get('tension_load', 0.05)

        # --- Fighting speed conflict ----------------------------------------
        # When both motor_boost and motor_drag are active simultaneously,
        # the competing forces create mechanical instability: stick-slip lurches,
        # rapid irregular speed oscillation, and amplitude modulation from
        # alternating tape tension.
        conflict = min(boost, drag)           # the overlapping fighting force
        if conflict > 0.05:
            # Stick-slip: random lurches — occasional speed spikes and drops
            if not hasattr(self, '_fight_phase'):
                self._fight_phase = 0.0
                self._fight_lurch_timer = 0.0
                self._fight_lurch_mag = 0.0

            # Irregular fast oscillation — 2-8 Hz, amplitude scales with conflict
            fight_freq = 3.5 + np.sin(self._fight_phase * 0.3) * 2.0
            self._fight_phase += fight_freq * frames / SR
            fight_osc = np.sin(self._fight_phase) * conflict * 0.4

            # Lurch events: periodic sudden speed jumps
            self._fight_lurch_timer += frames / SR
            lurch_interval = max(0.3, 1.5 / (conflict + 0.1))
            if self._fight_lurch_timer > lurch_interval:
                self._fight_lurch_mag = np.random.choice([-1, 1]) * conflict * 0.8
                self._fight_lurch_timer = 0.0
            # Decay the lurch
            self._fight_lurch_mag *= 0.85 ** (frames / (SR * 0.1))
        else:
            fight_osc = 0.0
            if hasattr(self, '_fight_lurch_mag'):
                self._fight_lurch_mag = 0.0

        lurch_add = getattr(self, '_fight_lurch_mag', 0.0)

        target_speed = 1.0 + boost - drag - tension_mod - self.sticky_drag + fight_osc + lurch_add
        target_speed = max(0.02, target_speed)

        # --- Oscillations (wow/flutter) -------------------------------------
        roller_radius  = 0.025
        roller_freq    = (ips * 0.0254) / (2 * np.pi * roller_radius)
        roller_wow     = (params.get('wow_dep', 0.2) / 100.0) * self._roller_ecc * 50 \
                         * np.sin(2 * np.pi * roller_freq * t_arr + self._roller_phase)

        reel_freq      = (ips * 0.0254) / (2 * np.pi * max(supply_r, 0.001))
        supply_flutter = (params.get('wow_dep', 0.2) / 200.0) * (progress * 0.5 + 0.1) \
                         * np.sin(2 * np.pi * reel_freq * t_arr + self._supply_phase)

        flutter_base   = params.get('flutter_dep', 0.05) / 150.0
        flutter        = flutter_base * np.sin(2 * np.pi * 15.0 * t_arr)
        flutter       += flutter_base * 0.3 * np.sin(2 * np.pi * 7.3 * t_arr + 0.7)

        # Scrape flutter is NOT modelled as a speed variation — it's a direct
        # amplitude/phase modulation applied to the audio signal in the engine.
        # Keeping a zero array here for the speed assembly loop below.
        scrape         = np.zeros(frames)

        # --- Brownian voltage drift -----------------------------------------
        raw_noise   = np.random.normal(0, 0.01, size=frames)
        drift_array = np.zeros(frames)
        health      = params.get('motor_health', 0.5)
        for i in range(frames):
            self.surge_state += (raw_noise[i] * health * 0.1) - (self.surge_state * 0.01)
            drift_array[i]    = self.surge_state

        # --- Dropout events -------------------------------------------------
        # dropout_mask is applied to the AUDIO signal in dsp_process.
        # Rate controls frequency: rate=0.1 → ~1 event/80s, rate=1.0 → ~1/8s.
        dropout_mask        = np.ones(frames)
        rate                = params.get('dropout_rate', 0.0)
        self._dropout_timer += frames / SR
        if rate > 0:
            while self._dropout_timer >= self._next_dropout:
                # How far into the past did this event fire?
                # overshoot=0 → event fires right at start of block (s=0)
                # overshoot large → event fired long ago, already past this block
                overshoot = self._dropout_timer - self._next_dropout
                s = int(np.clip(overshoot * SR, 0, frames - 1))
                # Duration: 1ms–20ms, longer at higher rates (more binder damage)
                max_dur  = 0.005 + rate * 0.040
                dur_samp = int(np.random.uniform(0.001, max_dur) * SR)
                e        = min(frames, s + dur_samp)
                # Depth: always near-complete silence (real dropouts kill the signal)
                depth    = np.random.uniform(0.7, 1.0) * np.clip(rate, 0, 1)
                if s < e:
                    seg_len = e - s
                    env     = np.ones(seg_len)
                    fade    = min(32, seg_len // 3)
                    if fade > 0:
                        env[:fade]  = np.linspace(1.0, 1.0 - depth, fade)
                        env[-fade:] = np.linspace(1.0 - depth, 1.0, fade)
                    if seg_len > 2 * fade:
                        env[fade:-fade] = 1.0 - depth
                    dropout_mask[s:e] = np.minimum(dropout_mask[s:e], env)
                    self.last_dropout_time = self._dropout_timer
                    self.dropout_count    += 1
                self._next_dropout += self._schedule_dropout(rate)

        # --- Assemble final speed array -------------------------------------
        # Motor speed approaches target_speed with inertia-based lag
        final_speeds = np.zeros(frames)
        for i in range(frames):
            self.current_motor_speed += (target_speed - self.current_motor_speed) * lag_coeff
            speed = (self.current_motor_speed
                     + roller_wow[i]
                     + supply_flutter[i]
                     + flutter[i]
                     + scrape[i]
                     + drift_array[i])
            final_speeds[i] = max(0.01, speed)

        self.last_instant_speed = final_speeds[-1]
        self.last_ips           = float(ips)
        self.last_conflict       = float(min(
            params.get('motor_boost', 0.0), params.get('motor_drag', 0.0)))
        return final_speeds, self.sticky_drag, dropout_mask
