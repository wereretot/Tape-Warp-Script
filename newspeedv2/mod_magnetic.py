import numpy as np


class MagneticPath:
    """
    Magnetic tape path simulation.

    Oxide type affects three things physically:
      Hc (coercivity)  — how sharply the tape saturates. High Hc (Metal) = harder
                         knee, more odd-harmonic distortion at the same drive level.
      Ms (saturation)  — maximum flux density, scales the output ceiling.
      bias_trim        — optimal bias point shifts with coercivity; CrO2/Metal need
                         more bias current to reach their optimal operating point.
    """

    OXIDE_PRESETS = {
        'Fe2O3': {'Hc': 250,   'Ms': 1.0,  'bias_trim': 1.0  },
        'CrO2':  {'Hc': 480,   'Ms': 1.2,  'bias_trim': 1.35 },
        'Metal': {'Hc': 1400,  'Ms': 1.8,  'bias_trim': 1.7  },
        'FeCo':  {'Hc': 700,   'Ms': 1.4,  'bias_trim': 1.4  },
    }
    # Normalise Hc so Fe2O3 = 1.0 (reference)
    _Hc_REF = 250.0

    def __init__(self):
        self._last_proc  = np.zeros(2)
        self._last_bark  = np.zeros(2)
        self._demag_last = np.zeros(2)

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

        # Oxide physics factors
        # hc_ratio > 1 = harder oxide = sharper saturation knee
        hc_ratio = oxide['Hc'] / self._Hc_REF
        Ms       = oxide['Ms']

        # 1. PRINT-THROUGH
        print_amt = params.get('print_through', 0.0)
        if print_amt > 0 and full_audio_data is not None:
            la      = int(44100 * 1.5)
            lb      = int(44100 * 1.5)
            gi_fwd  = np.clip(np.floor(read_indices).astype(np.int32) + la,
                              0, len(full_audio_data) - 1)
            gi_back = np.clip(np.floor(read_indices).astype(np.int32) - lb,
                              0, len(full_audio_data) - 1)
            out_v += full_audio_data[gi_fwd]  * print_amt * 0.7
            out_v += full_audio_data[gi_back] * print_amt * 0.3

        # 2. STEREO CROSSTALK
        cross = params.get('crosstalk', 0.0)
        if cross > 0:
            l, r = out_v[:, 0].copy(), out_v[:, 1].copy()
            out_v[:, 0] = l * (1.0 - cross) + r * cross
            out_v[:, 1] = r * (1.0 - cross) + l * cross

        # 3. MAGNETIC SATURATION — oxide-aware
        #
        # Two independent physical properties:
        #
        # Hc (coercivity) — controls the saturation KNEE SHARPNESS.
        #   High Hc (Metal) = harder knee = more odd-harmonic distortion at the
        #   same relative drive level. Low Hc (Fe2O3) = gentle, warm rounding.
        #   Modelled by scaling the tanh argument: hc_ratio > 1 = harder clip.
        #
        # Ms (saturation magnetisation) — controls the OUTPUT CEILING.
        #   Metal tape (Ms=1.8) can hold 1.8× the flux of Fe2O3 (Ms=1.0),
        #   so it produces a proportionally higher output level at the same drive.
        #   Modelled as a direct post-saturation scale factor.
        #
        # bias shifts the optimal operating point for each oxide type (via bias_trim).
        #   Over-bias → lower THD, compressed HF. Under-bias → more 2nd harmonic.

        # softness of the tanh knee — high Hc = harder (smaller softness)
        softness = np.clip(bias, 0.25, 4.0)
        # sharpness factor: Metal (hc_ratio=5.6) clips ~5× harder than Fe2O3
        # use a compressed scale so the difference is audible but not extreme
        knee_scale = 1.0 / np.clip(hc_ratio ** 0.5, 0.5, 4.0)

        h_in    = out_v * drive
        raw_sat = np.tanh(h_in * knee_scale / softness) * softness / knee_scale
        # Ms scales the output ceiling: Metal produces more output for same input
        # normalise by drive so unity-gain at low levels regardless of oxide
        ceiling = Ms / max(drive * 0.5 + 0.5, 0.1)
        proc    = raw_sat * ceiling

        # 4. REPLAY HEAD DIFFERENTIATION — cross-block continuous
        replay_diff = params.get('replay_diff', 0.3)
        if replay_diff > 0 and n > 1:
            proc_with_prev = np.vstack([self._last_proc[None, :], proc])
            diff  = np.diff(proc_with_prev, axis=0)
            proc  = proc * (1.0 - replay_diff) + diff * replay_diff
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
            proc_with_prev = np.vstack([self._demag_last[None, :], proc])
            for i in range(1, n + 1):
                proc_with_prev[i] = (proc_with_prev[i] * (1 - alpha)
                                     + proc_with_prev[i - 1] * alpha)
            proc = proc_with_prev[1:]
        self._demag_last = proc[-1].copy()

        # 8. OXIDE SHEDDING
        shedding = params.get('oxide_shedding', 0.0)
        if shedding > 0:
            grain = np.random.binomial(1, shedding * 0.001, proc.shape).astype(float)
            proc *= (1.0 - grain * np.random.uniform(0.1, 0.5, proc.shape))

        return proc
