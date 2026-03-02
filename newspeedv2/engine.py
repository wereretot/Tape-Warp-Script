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
        self.is_reversed   = False   # True when audio_data is currently flipped
        self.lock          = threading.Lock()

        self.transport   = TransportDynamics()
        self.magnetic    = MagneticPath()
        self.electronics = ElectronicComponents()

        self.params = {}

    def load_file(self, path):
        audio = AudioSegment.from_file(path).set_frame_rate(44100).set_channels(2).set_sample_width(2)
        audio = audio.apply_gain(-audio.max_dBFS - 1.0)
        samples = np.array(audio.get_array_of_samples()).reshape((-1, 2))
        with self.lock:
            self.audio_data    = samples.astype(np.float32) / 32768.0
            self.total_samples = len(self.audio_data)
            self.is_reversed   = False
            self.reset_state()

    def set_reverse(self, want_reverse):
        """Flip audio_data in place and reset DSP state. Call only when stopped."""
        with self.lock:
            if want_reverse != self.is_reversed:
                self.audio_data  = self.audio_data[::-1].copy()
                self.is_reversed = want_reverse
            self.reset_state()

    def reset_state(self):
        self.play_head    = 0.0
        self.current_time = 0.0
        self.transport.reset()
        self.electronics.reset()
        self.magnetic.reset()

    def dsp_process(self, frames):
        with self.lock:
            if self.audio_data is None:
                return None
            if self.play_head >= self.total_samples - frames:
                return None

            p            = self.params
            speed_factor = p.get('ips_base', 15.0) / 15.0

            final_speeds, sticky_drag = self.transport.process_speed(
                frames, self.current_time, self.play_head, self.total_samples, p
            )

            read_indices = self.play_head + np.cumsum(final_speeds)
            if read_indices[-1] >= self.total_samples - 1:
                return None

            i0    = np.floor(read_indices).astype(np.int32)
            frac  = read_indices - i0
            out_v = self.audio_data[i0] + (self.audio_data[i0+1] - self.audio_data[i0]) * frac[:, None]

            proc = self.magnetic.process(out_v, self.audio_data, read_indices, p)
            proc = self.electronics.process(proc, frames, self.current_time, speed_factor, sticky_drag, p)

            self.play_head     = float(read_indices[-1])
            self.current_time += frames / 44100.0
            return np.clip(proc, -1.0, 1.0)
