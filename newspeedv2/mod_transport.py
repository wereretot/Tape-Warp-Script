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

    # Capstan spin-up / spin-down physics
    # A real capstan motor takes ~0.3–1.5 s to reach full speed from rest,
    # and coasts for a similar duration after the brake is applied.
    # We model this as a first-order lag on a "motor_engage" state:
    #   engage → 1  when transport is running  (motor energised)
    #   engage → 0  when transport is stopped  (motor de-energised / braking)
    # The lag coefficient determines how fast speed changes: smaller = slower ramp.
    # SPINUP_TAU  ≈ 0.6 s  (time to reach 63% of target from rest)
    # SPINDOWN_TAU ≈ 0.9 s (coasting to stop is slightly slower than spin-up)
    SPINUP_COEFF   = 1.0 / (0.6  * 44100)   # per-sample coefficient for spin-up
    SPINDOWN_COEFF = 1.0 / (0.9  * 44100)   # per-sample coefficient for spin-down

    def __init__(self, seed=None):
        self.current_motor_speed = 0.0   # starts at rest; spins up to 1.0
        self.sticky_drag         = 0.0
        self.last_instant_speed  = 0.0
        self.last_conflict        = 0.0
        self.dropout_count        = 0
        self.last_dropout_time    = -999.0

        # Start/stop inertia state
        self.motor_engage        = 0.0   # 0 = stopped, 1 = fully running
        self.is_stopping         = False

        # Oscillator phases — use a seeded RNG so every TapeEngine instance
        # that shares the same seed produces identical wow/flutter LFO phases.
        # This matters for multi-thread rendering: all worker engines must have
        # the same oscillator personality so their speed modulation is continuous
        # across slice boundaries (the warm-up blocks handle IIR filter state;
        # matching phases ensures no pitch step at the join).
        rng = np.random.default_rng(seed)
        self._roller_phase = rng.uniform(0, 2 * np.pi)
        self._roller_ecc   = rng.uniform(0.0002, 0.0008)
        self._supply_phase = rng.uniform(0, 2 * np.pi)

        self._next_dropout  = 0.0   # fire immediately on first block, schedule from there
        self._dropout_timer = 0.0
        self._dropout_rem   = 0     # samples remaining from a dropout that crossed block boundary
        self._dropout_depth = 0.0   # depth of carried-over dropout

    def reset(self):
        self.current_motor_speed = 0.0   # reset to rest — will spin up on next play
        self.sticky_drag         = 0.0
        self.last_instant_speed  = 0.0
        self.motor_engage        = 0.0
        self.is_stopping         = False
        self._next_dropout       = 0.0
        self._dropout_timer      = 0.0
        self.dropout_count       = 0
        self.last_dropout_time   = -999.0
        self._dropout_rem        = 0
        self._dropout_depth      = 0.0

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

        # Tension arm — modelled as a low-frequency sinusoid driven by reel geometry.
        # The spring-mass integrator (T≈11s) cannot be warmed up by the short
        # pre-roll used in multi-thread rendering, causing a pitch step at every
        # chunk boundary.  Instead we use a deterministic function of absolute
        # time so all worker engines produce identical values at the same instant.
        # Physical basis: real tension arms oscillate at 0.05–0.3 Hz driven by
        # reel torque variations and the arm's spring return.
        tension_load = params.get('tension_load', 0.05)
        t0 = current_time   # scalar — start of this block in seconds
        tension_osc = (
            np.sin(2 * np.pi * 0.11 * t0 + self._roller_phase * 0.7) * 0.35
            + np.sin(2 * np.pi * 0.07 * t0 + self._supply_phase * 0.5) * 0.20
            + np.sin(2 * np.pi * 0.19 * t0 + 1.3) * 0.15
        )
        # Bias toward the reel-fill position so it slowly drifts as tape transfers
        tension_dc   = back_tension * 2.0
        tension_mod  = np.clip(tension_dc + tension_osc * tension_load, 0.0, tension_load)

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

        # --- Motor voltage drift (replaces Brownian random walk) -------------
        # The random-walk surge_state diverges between parallel workers because
        # each runs independent np.random calls.  Replacing it with a seeded
        # deterministic LFO sum gives the same perceptual "unstable motor" effect
        # while being fully reproducible from absolute time position.
        # Three slow sinusoids at irrational frequency ratios produce a
        # pseudo-random but continuous drift pattern.
        health      = params.get('motor_health', 0.5)
        drift_amp   = health * 0.008
        drift_slow  = drift_amp * np.sin(2 * np.pi * 0.031 * t_arr + self._roller_phase * 1.3)
        drift_mid   = drift_amp * 0.5 * np.sin(2 * np.pi * 0.073 * t_arr + self._supply_phase)
        drift_fast  = drift_amp * 0.25 * np.sin(2 * np.pi * 0.157 * t_arr + 2.1)
        drift_array = drift_slow + drift_mid + drift_fast

        # --- Dropout events -------------------------------------------------
        # Dropouts are carried across block boundaries — a dropout that starts
        # near the end of a block continues into the next block with no hard edge.
        # Fade length is 4ms (176 samples) for click-free transitions.
        FADE  = 176   # 4ms at 44100Hz — minimum perceptible pop threshold
        dropout_mask        = np.ones(frames)
        rate                = params.get('dropout_rate', 0.0)
        self._dropout_timer += frames / SR

        # Apply carry-over from previous block first
        if self._dropout_rem > 0:
            carry   = min(self._dropout_rem, frames)
            depth   = self._dropout_depth
            env     = np.ones(carry)
            fade_out = min(FADE, carry)
            env[:carry - fade_out] = 1.0 - depth
            env[carry - fade_out:] = np.linspace(1.0 - depth, 1.0, fade_out)
            dropout_mask[:carry] = np.minimum(dropout_mask[:carry], env)
            self._dropout_rem -= carry
            if self._dropout_rem <= 0:
                self._dropout_rem   = 0
                self._dropout_depth = 0.0

        if rate > 0:
            while self._dropout_timer >= self._next_dropout:
                overshoot = self._dropout_timer - self._next_dropout
                s = int(np.clip(overshoot * SR, 0, frames - 1))
                max_dur  = 0.005 + rate * 0.040
                dur_samp = int(np.random.uniform(0.001, max_dur) * SR)
                depth    = np.random.uniform(0.7, 1.0) * np.clip(rate, 0, 1)

                # Build full envelope then slice to this block, carrying remainder
                full_env    = np.ones(dur_samp)
                fade        = min(FADE, dur_samp // 3)
                if fade > 0:
                    full_env[:fade]  = np.linspace(1.0, 1.0 - depth, fade)
                    full_env[-fade:] = np.linspace(1.0 - depth, 1.0, fade)
                if dur_samp > 2 * fade:
                    full_env[fade:-fade] = 1.0 - depth

                in_block = min(dur_samp, frames - s)
                if in_block > 0:
                    dropout_mask[s:s + in_block] = np.minimum(
                        dropout_mask[s:s + in_block], full_env[:in_block])

                # Carry remaining samples to next block
                if dur_samp > in_block:
                    self._dropout_rem   = dur_samp - in_block
                    self._dropout_depth = depth

                self.last_dropout_time = self._dropout_timer
                self.dropout_count    += 1
                self._next_dropout += self._schedule_dropout(rate)

        # --- Start/stop inertia ---------------------------------------------
        # motor_engage: 0 = stopped, 1 = fully running.
        # The engine sets params['motor_engage'] = 1.0 during playback.
        # When stopping, it sets it to 0.0 and the capstan coasts to a halt.
        # The ramp is applied as a gain on target_speed so all wow/flutter
        # also ramps up/down naturally — you hear the pitch rise on start
        # and fall on stop, exactly like a real Studer or Ampex.
        engage_target = float(params.get('motor_engage', 1.0))

        # --- Assemble final speed array -------------------------------------
        # Motor speed approaches target_speed with inertia-based lag,
        # then scaled by the start/stop engage ramp.
        final_speeds = np.zeros(frames)
        for i in range(frames):
            # Spin-up is faster than spin-down (brake assist vs coasting)
            if self.motor_engage < engage_target:
                self.motor_engage = min(engage_target,
                                        self.motor_engage + self.SPINUP_COEFF)
            elif self.motor_engage > engage_target:
                self.motor_engage = max(engage_target,
                                        self.motor_engage - self.SPINDOWN_COEFF)

            self.current_motor_speed += (target_speed - self.current_motor_speed) * lag_coeff
            speed = (self.current_motor_speed
                     + roller_wow[i]
                     + supply_flutter[i]
                     + flutter[i]
                     + scrape[i]
                     + drift_array[i])
            # Scale by engage ramp: speed falls to 0 as motor disengages,
            # rises from 0 as motor spins up.  Clamp to 0.001 so we never
            # produce a negative read-index step.
            final_speeds[i] = max(0.001, speed * self.motor_engage)

        self.last_instant_speed = final_speeds[-1]
        self.last_ips           = float(ips)
        self.last_conflict       = float(min(
            params.get('motor_boost', 0.0), params.get('motor_drag', 0.0)))
        return final_speeds, self.sticky_drag, dropout_mask
