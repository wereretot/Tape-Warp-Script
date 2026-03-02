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

        self._roller_phase = np.random.uniform(0, 2 * np.pi)
        self._roller_ecc   = np.random.uniform(0.0002, 0.0008)
        self._supply_phase = np.random.uniform(0, 2 * np.pi)
        self._tension_arm  = 0.5
        self._tension_vel  = 0.0

        self._next_dropout  = self._schedule_dropout()
        self._dropout_timer = 0.0

    def reset(self):
        self.current_motor_speed = 1.0
        self.surge_state         = 0.0
        self.sticky_drag         = 0.0
        self.last_instant_speed  = 1.0
        self._tension_arm        = 0.5
        self._tension_vel        = 0.0
        self._next_dropout       = self._schedule_dropout()
        self._dropout_timer      = 0.0

    def _schedule_dropout(self):
        return np.random.exponential(scale=30.0)

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
        self.sticky_drag += params.get('sticky_shed', 0.0) * 0.00001

        # Back-tension from take-up reel (increases as reel fills)
        back_tension = (takeup_r / self.REEL_RADIUS_FULL) * params.get('tension_load', 0.05) * 0.5

        # Tension arm spring-mass oscillator
        tension_k = 0.3
        tension_b = 0.05
        tension_f = back_tension - self._tension_arm * tension_k - self._tension_vel * tension_b
        self._tension_vel += tension_f * (frames / SR)
        self._tension_arm  = np.clip(self._tension_arm + self._tension_vel * (frames / SR), 0, 1)
        tension_mod        = self._tension_arm * params.get('tension_load', 0.05)

        target_speed = 1.0 + boost - drag - tension_mod - self.sticky_drag
        target_speed = max(0.05, target_speed)

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

        scrape_freq    = np.clip(3200.0 * (ips / 15.0), 800, 12000)
        scrape         = (params.get('scrape_flutter', 0.1) / 800.0) \
                         * np.sin(2 * np.pi * scrape_freq * t_arr)

        # --- Brownian voltage drift -----------------------------------------
        raw_noise   = np.random.normal(0, 0.01, size=frames)
        drift_array = np.zeros(frames)
        health      = params.get('motor_health', 0.5)
        for i in range(frames):
            self.surge_state += (raw_noise[i] * health * 0.1) - (self.surge_state * 0.01)
            drift_array[i]    = self.surge_state

        # --- Dropout events -------------------------------------------------
        dropout_mask        = np.ones(frames)
        self._dropout_timer += frames / SR
        if params.get('dropout_rate', 0.0) > 0:
            while self._dropout_timer > self._next_dropout:
                evt_pos  = int((self._dropout_timer - self._next_dropout) * SR)
                dur_samp = int(np.random.uniform(0.002, 0.020) * SR)
                depth    = np.random.uniform(0.3, 1.0) * params.get('dropout_rate', 0.0)
                s = max(0, frames - evt_pos)
                e = min(frames, s + dur_samp)
                if s < e:
                    env = np.ones(e - s)
                    fade = min(20, (e - s) // 2)
                    env[:fade]    *= np.linspace(1, 1 - depth, fade)
                    env[-fade:]   *= np.linspace(1 - depth, 1, fade)
                    env[fade:-fade] *= (1 - depth)
                    dropout_mask[s:e] = env
                self._next_dropout += self._schedule_dropout()

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

        final_speeds          *= dropout_mask
        self.last_instant_speed = final_speeds[-1]
        return final_speeds, self.sticky_drag
