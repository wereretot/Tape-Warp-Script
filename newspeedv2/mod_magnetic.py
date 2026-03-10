import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


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
        self._demag_f    = None   # scipy lfilter state for 2-pole demagnetization

    def reset(self):
        self._last_proc  = np.zeros(2)
        self._last_bark  = np.zeros(2)
        self._demag_last = np.zeros(2)
        self._demag_f    = None

    def _apply_filter(self, fkey, new_b, new_a, signal):
        """Stateful IIR filter — preserves zi across blocks to avoid boundary clicks."""
        new_b  = np.asarray(new_b, dtype=float)
        new_a  = np.asarray(new_a, dtype=float)
        stored = getattr(self, fkey)
        if (stored is not None
                and np.array_equal(stored['b'], new_b)
                and np.array_equal(stored['a'], new_a)):
            zi = stored['zi']
        else:
            zi_base = lfilter_zi(new_b, new_a)
            dc      = float(np.mean(signal[:4]))
            zi      = zi_base[:, None] * np.array([[dc, dc]])
        out, zi_new = lfilter(new_b, new_a, signal, axis=0, zi=zi)
        setattr(self, fkey, {'b': new_b, 'a': new_a, 'zi': zi_new})
        return out

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
        # Layer-to-layer magnetic bleed between adjacent tape winds.
        # The stronger ghost (70%) comes from the layer wound ON TOP of the current
        # position — which in a forward-wound reel is the content recorded LATER.
        # In forward play: "later content" = higher read_indices → gi_ahead = +offset.
        # In reverse play: tape traversed in opposite direction, so "later" in winding
        # order is at LOWER read_indices in the reversed array → gi_ahead = -offset.
        # We zero any ghost whose index was boundary-clamped (would add same-content artefact).
        print_amt = params.get('print_through', 0.0)
        if print_amt > 0 and full_audio_data is not None:
            # Layer spacing is NOT fixed: adjacent tape layers are separated by
            # one reel circumference of tape, which shrinks as the supply empties.
            # We approximate: at the current read position, supply reel radius is
            # estimated from progress; adjacent-layer offset = 2π r_supply / v_tape.
            #
            # REEL_RADIUS constants match TransportDynamics geometry.
            _REEL_FULL = 0.133; _REEL_HUB = 0.025
            _full_area  = np.pi * (_REEL_FULL**2 - _REEL_HUB**2)
            N           = len(full_audio_data)
            progress    = float(np.mean(read_indices)) / max(N - 1, 1)
            progress    = float(np.clip(progress, 0.0, 1.0))
            supply_r    = float(np.sqrt(_full_area * (1.0 - progress) / np.pi + _REEL_HUB**2))
            ips         = float(params.get('ips_base', 15.0))
            v_tape      = ips * 0.0254           # m/s
            layer_sep_s = (2.0 * np.pi * supply_r) / v_tape   # seconds between layers
            layer_sep_s = float(np.clip(layer_sep_s, 0.2, 4.0))

            la      = int(44100 * layer_sep_s)
            lb      = int(44100 * layer_sep_s)
            raw_idx = np.floor(read_indices).astype(np.int32)
            is_rev  = params.get('is_reversed', False)

            if not is_rev:
                # Forward: the next layer wound on top = content recorded LATER = +offset
                pre_raw  = raw_idx + la
                post_raw = raw_idx - lb
                w_pre, w_post = 0.70, 0.30
            else:
                pre_raw  = raw_idx - la
                post_raw = raw_idx + lb
                w_pre, w_post = 0.70, 0.30

            pre_clamped  = np.clip(pre_raw,  0, N - 1)
            post_clamped = np.clip(post_raw, 0, N - 1)

            pre_mask  = (pre_raw  == pre_clamped ).astype(float)[:, None]
            post_mask = (post_raw == post_clamped).astype(float)[:, None]

            out_v += full_audio_data[pre_clamped]  * (print_amt * w_pre)  * pre_mask
            out_v += full_audio_data[post_clamped] * (print_amt * w_post) * post_mask

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

        # ── Magnetic saturation — oxide-specific harmonic character ─────────────
        #
        # Two independent physical properties control saturation character:
        #
        # Hc (coercivity): controls knee sharpness.
        #   High Hc (Metal, FeCo) = hard knee, prominent odd harmonics, "aggressive".
        #   Low Hc (Fe2O3) = soft knee, gentle warmth, low THD.
        #   Modelled via knee_scale: higher Hc = sharper drive into tanh.
        #
        # Ms (saturation magnetisation): controls output ceiling.
        #   Metal (Ms=1.8) holds 80% more flux than Fe2O3 (Ms=1.0).
        #
        # Bias: optimal recording bias is oxide-specific (set via bias_trim).
        #   Under-bias shifts the operating point off-centre → 2nd harmonic rises.
        #   Over-bias introduces pre-ringing and compresses HF.
        #   Here a small DC offset models the operating-point asymmetry so under-
        #   biased recordings show the characteristic warm 2nd-harmonic colouration.
        #
        # knee_scale: compressed (^0.45) so Metal clips audibly harder than Fe2O3
        # but not unrealistically so.
        knee_scale = 1.0 / np.clip(hc_ratio ** 0.45, 0.4, 3.5)

        # Bias asymmetry: shifts the input operating point.
        # bias=1.0 (nominal) → dc_offset=0 (symmetric, odd harmonics only).
        # bias=0.7 (under) → dc_offset≈ -0.024 → 2nd harmonic content rises.
        # bias=1.4 (over)  → dc_offset≈ +0.032 → slight HF softening.
        dc_offset = (bias - 1.0) * 0.08

        if params.get('_presaturated', False):
            # Signal already saturated externally (oversampled path) — skip tanh.
            proc = out_v
        else:
            h_in    = out_v * drive + dc_offset
            # tanh saturation with oxide knee + normalised output level
            raw_sat = np.tanh(h_in * knee_scale) / knee_scale
            # Ms ceiling: Metal tape outputs more for same input drive level
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

        # 7. DEMAGNETISATION — 2-pole Butterworth lowpass, causal in tape direction.
        #
        # A demagnetised playback head has increased effective gap length, which
        # creates a sinc-like HF rolloff (gap loss formula).  A 2-pole Butterworth
        # (12 dB/octave) approximates this better than the previous 1-pole (6 dB/oct).
        #
        # -3 dB frequency mapping (realistic head gap physics):
        #   demag = 0.10  →  fc ≈ 14 kHz  (clean, barely perceptible)
        #   demag = 0.30  →  fc ≈  8 kHz  (slight air loss)
        #   demag = 0.50  →  fc ≈  4 kHz  (clearly muffled HF)
        #   demag = 0.70  →  fc ≈  1.5 kHz (heavy loss of presence)
        #   demag = 0.90  →  fc ≈  400 Hz  (extremely degraded)
        #
        # Applied causally in the playback direction (forward or reverse) so
        # temporal smearing matches the physical head gap integration direction.
        demag = params.get('demagnetization', 0.0)
        if demag > 0:
            SR_mag = 44100.0
            fc_hz  = float(np.clip(300.0 + 17500.0 * (1.0 - demag) ** 2.2, 300.0, 20000.0))
            wn     = float(np.clip(fc_hz / (SR_mag * 0.5), 1e-4, 0.4999))
            b_d, a_d = butter(2, wn, btype='low')
            is_rev = params.get('is_reversed', False)
            if is_rev:
                proc = self._apply_filter('_demag_f', b_d, a_d, proc[::-1].copy())[::-1]
            else:
                proc = self._apply_filter('_demag_f', b_d, a_d, proc)
        else:
            self._demag_f = None   # reset state when demag is off
        self._demag_last = proc[-1].copy()

        # 8. OXIDE SHEDDING
        shedding = params.get('oxide_shedding', 0.0)
        if shedding > 0:
            grain = np.random.binomial(1, shedding * 0.001, proc.shape).astype(float)
            proc *= (1.0 - grain * np.random.uniform(0.1, 0.5, proc.shape))

        return proc
