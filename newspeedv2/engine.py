import numpy as np
import threading
from pydub import AudioSegment
from scipy.signal import butter, lfilter

class TapeEngine:
    def __init__(self):
        self.audio_data = None
        self.total_samples = 0
        self.play_head = 0.0
        self.current_time = 0.0
        self.is_playing = False
        self.lock = threading.Lock()
        
        # Physics State
        self.filter_zi = None
        self.hiss_zi = None
        self.current_motor_speed = 1.0
        self.surge_state = 0.0        # Voltage drift accumulator
        self.belt_slip_val = 0.0
        self.last_instant_speed = 1.0 # For UI Telemetry
        
        self.params = {
            "base_ips": 15.0,
            "motor_health": 0.5,      # Voltage Drift intensity
            "motor_drag": 0.0,
            "wow_hz": 0.5, "wow_dep": 0.1, 
            "flutter_hz": 15.0, "flutter_dep": 0.05,
            "hiss": 0.001, "hiss_grit": 3000, "drive": 1.2, "bias": 1.0,
            "cutoff_base": 18000, "print_through": 0.0
        }

    def load_file(self, path):
        audio = AudioSegment.from_file(path).set_frame_rate(44100).set_channels(2).set_sample_width(2)
        audio = audio.apply_gain(-audio.max_dBFS - 1.0)
        samples = np.array(audio.get_array_of_samples()).reshape((-1, 2))
        with self.lock:
            self.audio_data = samples.astype(np.float32) / 32768.0
            self.total_samples = len(self.audio_data)
            self.play_head = 0.0; self.current_time = 0.0

    def dsp_process(self, frames):
        with self.lock:
            if self.audio_data is None or self.play_head >= self.total_samples - 2:
                return None

            p = self.params
            speed_factor = p['base_ips'] / 15.0
            
            # 1. MECHANICAL SPEED MODULATORS
            t_arr = self.current_time + np.arange(frames) / 44100.0
            
            # Wow and Flutter (Sine-based)
            wow = (p['wow_dep'] / 100.0) * np.sin(2 * np.pi * p['wow_hz'] * t_arr)
            flutter = (p['flutter_dep'] / 150.0) * np.sin(2 * np.pi * p['flutter_hz'] * t_arr)
            
            # Voltage Drift (Stochastic "Brownian" movement)
            # We generate random noise and integrate it to get 'drift'
            raw_noise = np.random.normal(0, 0.01, size=frames)
            drift_array = np.zeros(frames)
            for i in range(frames):
                # Surge state drifts slowly based on motor_health
                self.surge_state += (raw_noise[i] * p['motor_health'] * 0.1) - (self.surge_state * 0.01)
                drift_array[i] = self.surge_state

            # 2. SPEED CALCULATION
            final_speeds = np.zeros(frames)
            target_base = 1.0 - p['motor_drag']
            
            for i in range(frames):
                # Motor acceleration/inertia
                self.current_motor_speed += (target_base - self.current_motor_speed) * 0.0005
                # Combine all factors
                final_speeds[i] = self.current_motor_speed + wow[i] + flutter[i] + drift_array[i]
            
            # Update telemetry for UI (use the last sample of the block)
            self.last_instant_speed = final_speeds[-1]

            # 3. SAMPLE EXTRACTION
            read_indices = self.play_head + np.cumsum(final_speeds)
            valid_mask = read_indices < self.total_samples - 1
            if not valid_mask.any(): return None
            n_valid = int(np.nonzero(valid_mask)[0][-1]) + 1
            ri = read_indices[:n_valid]
            
            i0 = np.floor(ri).astype(np.int32)
            frac = ri - i0
            out_v = self.audio_data[i0] + (self.audio_data[i0+1] - self.audio_data[i0]) * frac[:, None]

            # 4. SIGNAL CONDITIONING (IPS Dependent)
            dynamic_cutoff = min(21000, p['cutoff_base'] * speed_factor)
            dynamic_hiss = p['hiss'] / (speed_factor ** 0.5)

            proc = np.tanh(out_v * p['drive'] * p['bias']) / (p['drive'] * 0.5 + 0.5)
            
            # Hiss
            raw_hiss = np.random.normal(0, dynamic_hiss, (n_valid, 2))
            b_h, a_h = butter(1, max(0.001, p['hiss_grit']/22050), btype='low')
            if self.hiss_zi is None: self.hiss_zi = np.zeros((1, 2))
            hiss_f, self.hiss_zi = lfilter(b_h, a_h, raw_hiss, axis=0, zi=self.hiss_zi)
            proc += hiss_f

            # HF Roll-off
            b, a = butter(2, max(0.001, dynamic_cutoff/22050), btype='low')
            if self.filter_zi is None: self.filter_zi = np.zeros((2, 2))
            proc, self.filter_zi = lfilter(b, a, proc, axis=0, zi=self.filter_zi)

            out = np.zeros((frames, 2), dtype=np.float32)
            out[:n_valid] = proc
            self.play_head = float(ri[-1])
            self.current_time += frames / 44100.0
            return np.clip(out, -1.0, 1.0)