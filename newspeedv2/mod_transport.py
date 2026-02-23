import numpy as np

class TransportDynamics:
    """Simulates the physical movement and motor speed of the tape transport."""
    def __init__(self):
        self.current_motor_speed = 1.0
        self.surge_state = 0.0          # Brownian Voltage Drift
        self.sticky_drag = 0.0          # Cumulative sticky shed friction
        self.last_instant_speed = 1.0

    def reset(self):
        self.current_motor_speed = 1.0
        self.surge_state = 0.0
        self.sticky_drag = 0.0
        self.last_instant_speed = 1.0

    def process_speed(self, frames, current_time, play_head, total_samples, params):
        t_arr = current_time + np.arange(frames) / 44100.0

        # Mechanical Oscillations
        wow = (params.get('wow_dep', 0.2) / 100.0) * np.sin(2 * np.pi * 0.5 * t_arr)
        flutter = (params.get('flutter_dep', 0.05) / 150.0) * np.sin(2 * np.pi * 15.0 * t_arr)
        scrape = (params.get('scrape_flutter', 0.1) / 800.0) * np.sin(2 * np.pi * 3200.0 * t_arr)
        
        # Brownian Voltage Drift
        raw_noise = np.random.normal(0, 0.01, size=frames)
        drift_array = np.zeros(frames)
        health = params.get('motor_health', 0.5)
        for i in range(frames):
            self.surge_state += (raw_noise[i] * health * 0.1) - (self.surge_state * 0.01)
            drift_array[i] = self.surge_state

        # Reel Tension (increases as supply reel empties)
        progress = play_head / total_samples if total_samples > 0 else 0
        tension = progress * params.get('tension_load', 0.05)
        
        # Sticky Shed: Tape binder degrades and physically gums up the capstan over time
        self.sticky_drag += (params.get('sticky_shed', 0.0) * 0.00001)

        final_speeds = np.zeros(frames)
        target_base = 1.0 - params.get('motor_drag', 0.0) - tension - self.sticky_drag
        
        for i in range(frames):
            self.current_motor_speed += (target_base - self.current_motor_speed) * 0.0005
            # Combine all physics, ensuring speed never goes in reverse (<0)
            final_speeds[i] = max(0.01, self.current_motor_speed + wow[i] + flutter[i] + scrape[i] + drift_array[i])
        
        self.last_instant_speed = final_speeds[-1]
        return final_speeds, self.sticky_drag