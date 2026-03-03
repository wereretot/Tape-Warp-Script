import numpy as np
import threading
from pydub import AudioSegment

from mod_transport import TransportDynamics
from mod_magnetic import MagneticPath
from mod_electronics import ElectronicComponents

class TapeEngine:
    def __init__(self):
        self.audio_data    = None
        self.total_samples = 0
        self.play_head     = 0.0
        self.current_time  = 0.0
        self.is_playing    = False
        self.is_reversed   = False
        self.lock          = threading.Lock()

        self.transport   = TransportDynamics()
        self.magnetic    = MagneticPath()
        self.electronics = ElectronicComponents()

        self.params       = {}
        self._scrape_phase = 0.0
        self._scrape_state = np.zeros(2)   # IIR state for scrape coloring

    def load_file(self, path):
        raw = AudioSegment.from_file(path)
        # pydub only supports mono<->stereo in set_channels, so collapse >2 ch to mono first
        if raw.channels > 2:
            raw = raw.set_channels(1)
        audio = raw.set_frame_rate(44100).set_channels(2).set_sample_width(2)
        audio = audio.apply_gain(-audio.max_dBFS - 1.0)
        samples = np.array(audio.get_array_of_samples()).reshape((-1, 2))
        with self.lock:
            self.audio_data    = samples.astype(np.float32) / 32768.0
            self.total_samples = len(self.audio_data)
            self.is_reversed   = False
            self.reset_state()

    def set_reverse(self, want_reverse, force_reset=False):
        """Flip audio_data in place. Only resets DSP state when direction actually changes."""
        with self.lock:
            if want_reverse != self.is_reversed:
                self.audio_data  = self.audio_data[::-1].copy()
                self.is_reversed = want_reverse
                self.reset_state()
            elif force_reset:
                self.reset_state()

    def reset_state(self):
        self.play_head    = 0.0
        self.current_time = 0.0
        self.transport.reset()
        self.electronics.reset()
        self.magnetic.reset()

    def dsp_process(self, frames, oversample=1):
        """
        oversample=1  : realtime quality (1024 frames → 1024 output samples)
        oversample=2  : 2× oversampling of nonlinear stages, then decimate
        oversample=4  : 4× oversampling — catches high-order intermod in saturation
        oversample=8  : 8× — diminishing returns but theoretically cleanest
        """
        with self.lock:
            if self.audio_data is None:
                return None
            if self.play_head >= self.total_samples - frames:
                return None

            p            = self.params
            speed_factor = p.get('ips_base', 15.0) / 15.0

            _result = self.transport.process_speed(
                frames, self.current_time, self.play_head, self.total_samples, p
            )
            # Support both old (2-value) and new (3-value) transport return
            if len(_result) == 3:
                final_speeds, sticky_drag, dropout_mask = _result
            else:
                final_speeds, sticky_drag = _result
                dropout_mask = np.ones(frames)

            read_indices = self.play_head + np.cumsum(final_speeds)
            if read_indices[-1] >= self.total_samples - 1:
                return None

            i0    = np.floor(read_indices).astype(np.int32)
            frac  = read_indices - i0
            out_v = self.audio_data[i0] + (self.audio_data[i0+1] - self.audio_data[i0]) * frac[:, None]

            if oversample > 1:
                # Oversampling the nonlinear saturation stage prevents tanh aliasing.
                # Strategy: run the full magnetic chain at native rate, then compute
                # what the saturation contributed natively vs oversampled, and swap it.
                from scipy.signal import resample_poly

                oxide_name = p.get('oxide_type', 'Fe2O3')
                oxide      = self.magnetic.OXIDE_PRESETS.get(
                                 oxide_name, self.magnetic.OXIDE_PRESETS['Fe2O3'])
                drive      = p.get('drive', 1.2)
                bias       = p.get('bias', 1.0) * oxide['bias_trim']
                hc_ratio   = oxide['Hc'] / self.magnetic._Hc_REF
                Ms         = oxide['Ms']
                softness   = float(np.clip(bias, 0.25, 4.0))
                knee_scale = 1.0 / float(np.clip(hc_ratio ** 0.5, 0.5, 4.0))
                ceiling    = Ms / max(drive * 0.5 + 0.5, 0.1)

                # Native saturation of out_v (what magnetic.process does internally)
                h_in_nat   = out_v * drive
                sat_native = np.tanh(h_in_nat * knee_scale / softness) * softness / knee_scale * ceiling

                # Oversampled saturation
                up         = resample_poly(out_v, oversample, 1, axis=0)
                h_in_up    = up * drive
                sat_up     = np.tanh(h_in_up * knee_scale / softness) * softness / knee_scale * ceiling
                sat_down   = resample_poly(sat_up, 1, oversample, axis=0)[:frames]

                # Correction: how much does OS saturation differ from native?
                sat_correction = sat_down - sat_native

                # Full native chain (correct read_indices, all stages)
                proc_native = self.magnetic.process(out_v, self.audio_data, read_indices, p)

                # Add the OS correction to the native output
                proc = proc_native + sat_correction
                proc = self.electronics.process(proc, frames, self.current_time,
                                                speed_factor, sticky_drag, p)
            else:
                proc = self.magnetic.process(out_v, self.audio_data, read_indices, p)
                proc = self.electronics.process(proc, frames, self.current_time,
                                                speed_factor, sticky_drag, p)

            # ── Scrape flutter — audio-domain stick-slip modulation ──────────
            # Real scrape flutter: the tape micro-sticks and releases against the
            # stationary erase/guide heads, creating rapid irregular amplitude
            # and phase bursts correlated with high-frequency content.
            # Modelled as: band-limited noise envelope × signal, at a fundamental
            # frequency set by IPS (faster tape → higher scrape frequency).
            scrape_amt = p.get('scrape_flutter', 0.0)
            if scrape_amt > 0.001:
                SR_f  = 44100.0
                ips   = p.get('ips_base', 15.0)
                # Fundamental scrape frequency scales with tape speed
                scrape_hz  = float(np.clip(2800.0 * (ips / 15.0), 600, 14000))
                phase_inc  = scrape_hz / SR_f
                phases     = self._scrape_phase + np.arange(frames) * phase_inc
                self._scrape_phase = float(phases[-1] % 1.0)

                # Irregular envelope: sum of detuned oscillators + noise bursts
                env = (np.sin(2 * np.pi * phases)
                     + np.sin(2 * np.pi * phases * 1.031 + 0.7) * 0.6
                     + np.sin(2 * np.pi * phases * 0.973 + 1.3) * 0.4
                     + np.random.normal(0, 0.3, frames))
                # Rectify so it's always a positive amplitude multiplier
                env = np.abs(env)
                # Normalise to [0, 1]
                env_max = np.max(env) + 1e-9
                env = env / env_max

                # IIR colouring: scrape has more effect on high-frequency content.
                # Apply a simple first-order HP to make it transient-focused.
                alpha = float(np.clip(0.85 + scrape_amt * 0.1, 0.85, 0.97))
                hf = np.zeros(frames)
                s  = self._scrape_state[0]
                for i in range(frames):
                    s = alpha * s + alpha * (proc[i, 0] - (s if i > 0 else proc[i, 0]))
                    hf[i] = proc[i, 0] - s
                self._scrape_state[0] = s

                # Depth: at full scrape_amt=1.0, modulates up to ±40% of HF content
                depth = float(np.clip(scrape_amt * 0.4, 0.0, 0.45))
                # Modulate only the high-frequency portion of the signal
                mod_l = hf * env * depth
                mod_r = mod_l * 0.85   # slight stereo decorrelation

                proc[:, 0] += mod_l
                proc[:, 1] += mod_r

            # Apply dropout mask — silences during oxide dropout events
            proc *= dropout_mask[:, None]

            # Fighting speed: when boost and drag conflict, add tape tension AM.
            # Use getattr so this works even if mod_transport is an older version.
            conflict = getattr(self.transport, 'last_conflict', 0.0)
            if conflict > 0.05:
                t_arr = self.current_time + np.arange(frames) / 44100.0
                am_freq  = 2.0 + conflict * 3.0
                am_depth = np.clip(conflict * 0.7, 0.0, 0.8)
                am_env   = 1.0 - am_depth * (0.5 + 0.5 * np.sin(2 * np.pi * am_freq * t_arr))
                proc    *= am_env[:, None]

            self.play_head     = float(read_indices[-1])
            self.current_time += frames / 44100.0
            return np.clip(proc, -1.0, 1.0)
