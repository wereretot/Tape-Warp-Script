import numpy as np

class MagneticPath:
    OXIDE_PRESETS = {
        'Fe2O3': {'Hc': 250,  'Ms': 1.0, 'bias_trim': 1.0 },
        'CrO2':  {'Hc': 480,  'Ms': 1.2, 'bias_trim': 1.35},
        'Metal': {'Hc': 1400, 'Ms': 1.8, 'bias_trim': 1.7 },
        'FeCo':  {'Hc': 700,  'Ms': 1.4, 'bias_trim': 1.4 },
    }

    def __init__(self):
        self._last_proc  = np.zeros(2)   # last output sample — for cross-block diff
        self._last_bark  = np.zeros(2)   # last proc sample — for barkhausen diff
        self._demag_last = np.zeros(2)   # last sample — for cross-block demag filter

    def reset(self):
        self._last_proc  = np.zeros(2)
        self._last_bark  = np.zeros(2)
        self._demag_last = np.zeros(2)

    def process(self, audio_chunk, full_audio_data, read_indices, params):
        out_v = audio_chunk.copy()
        n     = len(out_v)

        oxide_name = params.get('oxide_type', 'Fe2O3')
        oxide      = self.OXIDE_PRESETS.get(oxide_name, self.OXIDE_PRESETS['Fe2O3'])
        drive      = params.get('drive', 1.2)
        bias       = params.get('bias', 1.0) * oxide['bias_trim']

        # 1. PRINT-THROUGH
        print_amt = params.get('print_through', 0.0)
        if print_amt > 0 and full_audio_data is not None:
            la  = int(44100 * 1.5)
            lb  = int(44100 * 1.5)
            gi_fwd  = np.clip(np.floor(read_indices).astype(np.int32) + la, 0, len(full_audio_data)-1)
            gi_back = np.clip(np.floor(read_indices).astype(np.int32) - lb, 0, len(full_audio_data)-1)
            out_v += full_audio_data[gi_fwd]  * print_amt * 0.7
            out_v += full_audio_data[gi_back] * print_amt * 0.3

        # 2. STEREO CROSSTALK
        cross = params.get('crosstalk', 0.0)
        if cross > 0:
            l, r = out_v[:, 0].copy(), out_v[:, 1].copy()
            out_v[:, 0] = l * (1.0 - cross) + r * cross
            out_v[:, 1] = r * (1.0 - cross) + l * cross

        # 3. MAGNETIC SATURATION
        softness = np.clip(bias, 0.5, 3.0)
        h_in = out_v * drive
        proc = np.tanh(h_in / softness) * softness / max(drive * 0.5 + 0.5, 0.1)

        # 4. REPLAY HEAD DIFFERENTIATION — cross-block continuous
        replay_diff = params.get('replay_diff', 0.3)
        if replay_diff > 0 and n > 1:
            proc_with_prev = np.vstack([self._last_proc[None, :], proc])
            diff = np.diff(proc_with_prev, axis=0)
            proc = proc * (1.0 - replay_diff) + diff * replay_diff
        self._last_proc = proc[-1].copy()

        # 5. BARKHAUSEN NOISE — cross-block continuous diff
        bark = params.get('barkhausen', 0.0)
        if bark > 0:
            proc_with_prev = np.vstack([self._last_bark[None, :], proc])
            dm    = np.abs(np.diff(proc_with_prev, axis=0))
            burst = np.random.normal(0, 1, proc.shape) * dm
            proc += burst * bark * 0.05
        self._last_bark = proc[-1].copy()

        # 6. ASPERITIES
        asp = params.get('asperities', 0.0)
        if asp > 0:
            proc += np.random.normal(0, asp * 0.005, proc.shape) * np.abs(proc)

        # 7. DEMAGNETISATION — cross-block continuous 1-pole IIR
        demag = params.get('demagnetization', 0.0)
        if demag > 0:
            alpha = np.clip(demag * 0.8, 0, 0.99)
            # Prepend last sample of previous block so first sample is blended correctly
            proc_with_prev = np.vstack([self._demag_last[None, :], proc])
            for i in range(1, n + 1):
                proc_with_prev[i] = proc_with_prev[i] * (1 - alpha) + proc_with_prev[i-1] * alpha
            proc = proc_with_prev[1:]
        self._demag_last = proc[-1].copy()

        # 8. OXIDE SHEDDING
        shedding = params.get('oxide_shedding', 0.0)
        if shedding > 0:
            grain = np.random.binomial(1, shedding * 0.001, proc.shape).astype(float)
            proc *= (1.0 - grain * np.random.uniform(0.1, 0.5, proc.shape))

        return proc
