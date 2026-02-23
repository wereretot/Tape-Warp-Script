import numpy as np
from scipy.signal import butter, lfilter

class ElectronicComponents:
    """Simulates the analog circuits, EQ curves, and noise floors of the playback unit."""
    def __init__(self):
        self.bump_zi = None
        self.filter_zi = None
        self.azimuth_state = 0.0

    def reset(self):
        self.bump_zi = None
        self.filter_zi = None
        self.azimuth_state = 0.0

    def process(self, proc, frames, current_time, speed_factor, sticky_drag, params):
        t_arr = current_time + np.arange(frames) / 44100.0

        # 1. HEAD BUMP (LF Resonance)
        bump_amt = params.get('head_bump', 0.0)
        if bump_amt > 0:
            # Clamp bump frequency to prevent filter collapse at ultra-low speeds
            bump_f = np.clip(50.0 * speed_factor, 20, 500)
            # Normalize and clamp Wn for bandpass
            low = np.clip(bump_f * 0.8 / 22050, 0.0001, 0.9999)
            high = np.clip(bump_f * 1.2 / 22050, 0.0002, 0.9999)
            
            b_b, a_b = butter(2, [low, high], btype='band')
            if self.bump_zi is None or self.bump_zi.shape != (len(a_b)-1, 2): 
                self.bump_zi = np.zeros((len(a_b)-1, 2))
            bump_sig, self.bump_zi = lfilter(b_b, a_b, proc, axis=0, zi=self.bump_zi)
            proc += bump_sig * bump_amt

        # 2. ELECTRONIC NOISE & HUM
        dynamic_hiss = params.get('hiss', 0.0) / (speed_factor ** 0.5) if speed_factor > 0.01 else 0
        hiss_sig = np.random.normal(0, dynamic_hiss, proc.shape)
        
        hum_amt = params.get('mains_hum', 0.0)
        if hum_amt > 0:
            hum_sig = (np.sin(2 * np.pi * 60.0 * t_arr) * hum_amt)[:, None]
            proc += hiss_sig + hum_sig
        else:
            proc += hiss_sig

        # 3. AZIMUTH CUTOFF & STICKY SHED LOSS (The Fix)
        cutoff = params.get('cutoff_base', 18000)
        # Calculate frequency based on speed and sticky buildup
        raw_cut = cutoff * speed_factor * (1.0 - sticky_drag * 50)
        
        # SAFETY CLAMP: Ensure frequency is always between 20Hz and 22kHz
        # This prevents the ValueError: Digital filter critical frequencies must be 0 < Wn < 1
        safe_cut = np.clip(raw_cut, 20, 22000)
        wn = safe_cut / 22050
        
        b_a, a_a = butter(2, wn, btype='low')
        if self.filter_zi is None or self.filter_zi.shape != (len(a_a)-1, 2):
            self.filter_zi = np.zeros((len(a_a)-1, 2))
        proc, self.filter_zi = lfilter(b_a, a_a, proc, axis=0, zi=self.filter_zi)

        # 4. AZIMUTH PHASE WANDER
        az_drift = params.get('azimuth_drift', 0.0)
        if az_drift > 0:
            self.azimuth_state += np.random.normal(0, 0.01)
            drift_amt = int(np.sin(self.azimuth_state) * az_drift * 5)
            # Ensure roll doesn't break if drift_amt is 0
            if drift_amt != 0:
                proc[:, 1] = np.roll(proc[:, 1], drift_amt)

        return proc