import numpy as np
import threading
from pydub import AudioSegment

from mod_transport import TransportDynamics
from mod_magnetic import MagneticPath
from mod_electronics import ElectronicComponents

class TapeEngine:
    def __init__(self, seed=None):
        self.audio_data    = None
        self.total_samples = 0
        self.play_head     = 0.0
        self.current_time  = 0.0
        self.is_playing    = False
        self.is_reversed   = False
        self.lock          = threading.Lock()

        self.transport   = TransportDynamics(seed=seed)
        self.magnetic    = MagneticPath()
        self.electronics = ElectronicComponents()

        self.params        = {}
        self._scrape_phase = 0.0
        self._scrape_state = np.zeros(4)   # [y_prev, x_prev] per channel for HP IIR
        self._os_context   = None          # last N samples before upsample (FIR overlap context)
        self._fade_in      = 0             # samples remaining in post-preset-change crossfade
        self._last_out     = np.zeros(2, dtype=np.float32)  # last output sample for DC crossfade

    def make_worker_engine(self, start_sample, block_size, oversample,
                           shared_seed, n_warmup_blocks=16):
        """
        Create a worker TapeEngine pre-warmed to render from start_sample.

        The engine is initialised with the same oscillator seed as the main
        engine so wow/flutter LFO phases are continuous across slice boundaries.
        It then runs n_warmup_blocks of actual audio (from before start_sample)
        through the full DSP chain and discards the output.  This settles all
        IIR filter states (head bump, azimuth lowpass, pink noise, scrape flutter,
        azimuth delay buffer) to the values they would have had had the engine
        been running continuously from the beginning of the file.

        Worker 0 starts at sample 0 so its warm-up window is empty — it just
        uses the cold-started engine directly (same as single-thread).
        """
        eng = TapeEngine(seed=shared_seed)
        with self.lock:
            eng.audio_data    = self.audio_data
            eng.total_samples = self.total_samples
            eng.params        = dict(self.params)
            eng.is_reversed   = self.is_reversed

        # Warm-up: start engine BEFORE the slice boundary so IIR filters settle.
        warmup_samples = n_warmup_blocks * block_size
        warmup_start   = max(0, start_sample - warmup_samples)
        eng.play_head    = float(warmup_start)
        eng.current_time = float(warmup_start) / 44100.0

        # Motor is fully running mid-tape — no spin-up ramp.
        eng.transport.motor_engage        = 1.0
        eng.transport.current_motor_speed = 1.0

        # Run warm-up blocks and discard their output.
        while eng.play_head < start_sample:
            blk = eng.dsp_process(block_size, oversample=oversample)
            if blk is None:
                break

        # After warm-up, play_head may have overshot start_sample slightly
        # (by up to block_size samples).  Reset it exactly so the worker
        # begins its output at the correct position.
        eng.play_head    = float(start_sample)
        eng.current_time = float(start_sample) / 44100.0
        # Keep _last_out from warm-up so the DC crossfade blends correctly.

        return eng

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
        self._os_context  = None
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
            p            = dict(p, is_reversed=self.is_reversed,
                                   motor_engage=1.0)   # fully engaged during playback
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
                # Oversampled saturation: upsample → saturate → decimate.
                # resample_poly uses a linear-phase FIR filter. Without context
                # samples from the previous block, the filter cold-starts at
                # block boundaries and produces a transient of up to 0.22 amplitude
                # — audible as a regular click every block_size/44100 seconds.
                # Fix: prepend _os_context (last PAD samples) before upsampling,
                # strip after downsampling so output length stays == frames.
                from scipy.signal import resample_poly
                OS_PAD = 64   # 1.45ms context — enough for all supported OS ratios

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

                # Pad with previous-block context
                has_context = self._os_context is not None
                if has_context:
                    padded = np.vstack([self._os_context, out_v])
                else:
                    padded = out_v   # first block: cold start is inaudible (silence→signal)
                self._os_context = out_v[-OS_PAD:].copy()

                up      = resample_poly(padded, oversample, 1, axis=0)
                h_up    = up * drive
                sat_up  = np.tanh(h_up * knee_scale / softness) * softness / knee_scale * ceiling
                sat_dec = resample_poly(sat_up, 1, oversample, axis=0)

                # Strip the padded prefix so output length == frames
                pad_out = OS_PAD if has_context else 0
                sat_dec = sat_dec[pad_out : pad_out + frames]

                # Strip the padded prefix (OS_PAD input samples → OS_PAD output samples)
                pad_out = 0 if padded is out_v else OS_PAD
                sat_dec = sat_dec[pad_out : pad_out + frames]

                p_os = dict(p, _presaturated=True)
                proc = self.magnetic.process(sat_dec, self.audio_data, read_indices, p_os)
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
            #
            # IMPORTANT: after oversampling, proc may be shorter than `frames`
            # (the last block of a pre-roll silence buffer or a speed->1 read
            # can produce fewer output samples).  All array sizes below are
            # derived from len(proc) — never from the `frames` parameter —
            # so we never index out of bounds.
            n_out = len(proc)
            scrape_amt = p.get('scrape_flutter', 0.0)
            if scrape_amt > 0.001:
                SR_f  = 44100.0
                ips   = p.get('ips_base', 15.0)
                scrape_hz  = float(np.clip(2800.0 * (ips / 15.0), 600, 14000))
                phase_inc  = scrape_hz / SR_f
                phases     = self._scrape_phase + np.arange(n_out) * phase_inc
                self._scrape_phase = float(phases[-1] % 1.0)

                env = (np.sin(2 * np.pi * phases)
                     + np.sin(2 * np.pi * phases * 1.031 + 0.7) * 0.6
                     + np.sin(2 * np.pi * phases * 0.973 + 1.3) * 0.4
                     + np.random.normal(0, 0.3, n_out))
                env = np.abs(env) / (np.max(np.abs(env)) + 1e-9)

                alpha  = float(np.clip(0.85 + scrape_amt * 0.1, 0.85, 0.97))
                hf     = np.zeros(n_out)
                y_prev = self._scrape_state[0]
                x_prev = self._scrape_state[1]
                for i in range(n_out):
                    x_cur  = proc[i, 0]
                    y_cur  = alpha * (y_prev + x_cur - x_prev)
                    hf[i]  = y_cur
                    x_prev = x_cur
                    y_prev = y_cur
                self._scrape_state[0] = y_prev
                self._scrape_state[1] = x_prev

                depth = float(np.clip(scrape_amt * 0.4, 0.0, 0.45))
                mod_l = hf * env * depth
                proc[:, 0] += mod_l
                proc[:, 1] += mod_l * 0.85

            # Apply dropout mask — trim to n_out in case transport generated
            # a full-length mask but proc is shorter after OS decimation.
            proc *= dropout_mask[:n_out, None]

            # Fighting speed: when boost and drag conflict, add tape tension AM.
            conflict = getattr(self.transport, 'last_conflict', 0.0)
            if conflict > 0.05:
                t_arr = self.current_time + np.arange(n_out) / 44100.0
                am_freq  = 2.0 + conflict * 3.0
                am_depth = np.clip(conflict * 0.7, 0.0, 0.8)
                am_env   = 1.0 - am_depth * (0.5 + 0.5 * np.sin(2 * np.pi * am_freq * t_arr))
                proc    *= am_env[:, None]

            self.play_head     = float(read_indices[-1])
            self.current_time += frames / 44100.0

            # ── Analog-style output limiter ───────────────────────────────────
            KNEE = 0.97
            proc = np.tanh(proc / KNEE) * KNEE

            # ── Post-preset-change DC crossfade ───────────────────────────────
            # When params change (preset switch or slider move), IIR filter states
            # are tuned to the old preset. The new block's first sample can jump
            # discontinuously. We crossfade from _last_out (the true last output
            # sample) to the new signal over _FADE_SAMPLES samples:
            #   output[n] = new[n]*t + last_out*(1-t)  where t: 0→1
            # At n=0: output = last_out exactly → zero jump across the boundary.
            _FADE_SAMPLES = 512  # ~12ms — covers any IIR settling transient
            if self._fade_in > 0:
                fade_len  = min(self._fade_in, n_out)
                pos_start = _FADE_SAMPLES - self._fade_in
                t_ramp    = np.linspace(pos_start / _FADE_SAMPLES,
                                        (pos_start + fade_len) / _FADE_SAMPLES,
                                        fade_len, dtype=np.float32)
                proc[:fade_len] = (proc[:fade_len] * t_ramp[:, None]
                                   + self._last_out[None, :] * (1.0 - t_ramp[:, None]))
                self._fade_in -= fade_len

            self._last_out = proc[-1].copy()
            return proc
