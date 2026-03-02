import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class ElectronicComponents:
    def __init__(self):
        self._bump_f            = None
        self._az_f              = None
        self.azimuth_state      = 0.0
        self._azimuth_delay_buf = np.zeros((128, 2))  # bigger buffer = more shift headroom
        self._diff_last         = np.zeros(2)

    def reset(self):
        self._bump_f            = None
        self._az_f              = None
        self.azimuth_state      = 0.0
        self._azimuth_delay_buf = np.zeros((128, 2))
        self._diff_last         = np.zeros(2)

    def _apply_filter(self, fkey, new_b, new_a, signal):
        new_b   = np.asarray(new_b, dtype=float)
        new_a   = np.asarray(new_a, dtype=float)
        stored  = getattr(self, fkey)
        coeff_match = (
            stored is not None
            and np.array_equal(stored['b'], new_b)
            and np.array_equal(stored['a'], new_a)
        )
        if coeff_match:
            zi = stored['zi']
        else:
            zi_base = lfilter_zi(new_b, new_a)
            dc      = float(np.mean(signal[:4]))
            zi      = zi_base[:, None] * np.array([[dc, dc]])
        out, zi_new = lfilter(new_b, new_a, signal, axis=0, zi=zi)
        setattr(self, fkey, {'b': new_b, 'a': new_a, 'zi': zi_new})
        return out

    def process(self, proc, frames, current_time, speed_factor, sticky_drag, params):
        t_arr = current_time + np.arange(frames) / 44100.0

        # 1. HEAD BUMP
        bump_amt = params.get('head_bump', 0.0)
        if bump_amt > 0:
            bump_f = float(np.clip(50.0 * speed_factor, 20, 500))
            low    = float(np.clip(bump_f * 0.7 / 22050, 1e-4, 0.499))
            high   = float(np.clip(bump_f * 1.3 / 22050, low + 1e-4, 0.4999))
            b_b, a_b = butter(2, [low, high], btype='band')
            bump_sig = self._apply_filter('_bump_f', b_b, a_b, proc)
            proc = proc + bump_sig * bump_amt

        # 2. AZIMUTH LOWPASS + STICKY SHED
        cutoff   = params.get('cutoff_base', 18000)
        raw_cut  = cutoff * speed_factor * (1.0 - sticky_drag * 50)
        safe_cut = float(np.clip(raw_cut, 20, 20000))
        wn       = float(np.clip(safe_cut / 22050, 1e-4, 0.4999))
        b_a, a_a = butter(2, wn, btype='low')
        proc     = self._apply_filter('_az_f', b_a, a_a, proc)

        # 3. AZIMUTH PHASE WANDER
        #    FIX: clamp shift so combined[start:start+frames] is always exactly `frames` long
        az_drift = params.get('azimuth_drift', 0.0)
        if az_drift > 0:
            self.azimuth_state += np.random.normal(0, 0.002)
            buf_len = self._azimuth_delay_buf.shape[0]
            # Max shift is buf_len-1 so slice always has `frames` samples available
            max_shift    = buf_len - 1
            delay_samp   = np.sin(self.azimuth_state) * az_drift * float(max_shift) * 0.5
            shift        = int(np.clip(delay_samp, -(max_shift), 0))  # only negative = delay
            frac         = delay_samp - int(delay_samp)

            combined     = np.vstack([self._azimuth_delay_buf, proc])  # (buf_len+frames, 2)
            start        = buf_len + shift   # shift<=0, so start<=buf_len, always enough room
            right        = combined[start: start + frames, 1].copy()

            if abs(frac) > 0.001:
                right_next = combined[start + 1: start + frames + 1, 1]
                if len(right_next) == frames:
                    right = right * (1.0 - abs(frac)) + right_next * abs(frac)

            proc_out       = proc.copy()
            proc_out[:, 1] = right
            proc           = proc_out
            self._azimuth_delay_buf = proc[-buf_len:].copy()

        # 4. NOISE & HUM
        dynamic_hiss = params.get('hiss', 0.0) / (speed_factor ** 0.5) if speed_factor > 0.01 else 0.0
        if dynamic_hiss > 0:
            proc = proc + np.random.normal(0, dynamic_hiss, proc.shape)

        hum_amt = params.get('mains_hum', 0.0)
        if hum_amt > 0:
            hum = (
                np.sin(2 * np.pi * 60.0  * t_arr) * hum_amt
                + np.sin(2 * np.pi * 180.0 * t_arr) * hum_amt * 0.3
            )[:, None]
            proc = proc + hum

        # 5. PINK HISS TILT
        hiss_color = params.get('hiss_color', 0.0)
        if hiss_color > 0 and dynamic_hiss > 0:
            pink = np.cumsum(np.random.normal(0, dynamic_hiss * 0.3, proc.shape), axis=0)
            pink -= np.mean(pink, axis=0)
            pink  = np.clip(pink, -dynamic_hiss * 8, dynamic_hiss * 8)
            proc  = proc + pink * hiss_color

        # 6. SPEED-DEPENDENT OUTPUT LEVEL
        proc = proc * float(np.clip(speed_factor ** 0.25, 0.5, 1.5))

        return proc
