import numpy as np

class MagneticPath:
    """Handles the translation of audio into magnetic flux and physical oxide interactions."""
    def process(self, audio_chunk, full_audio_data, read_indices, params):
        out_v = audio_chunk.copy()
        
        # 1. PRINT-THROUGH (The Pre-Echo Ghost)
        # Magnetic fields bleed through the layers of tape on the reel
        print_amt = params.get('print_through', 0.0)
        if print_amt > 0 and full_audio_data is not None:
            look_ahead = int(44100 * 1.5) # Look 1.5 seconds into the future
            ghost_indices = np.clip(np.floor(read_indices).astype(np.int32) + look_ahead, 0, len(full_audio_data)-1)
            out_v += full_audio_data[ghost_indices] * print_amt

        # 2. STEREO CROSSTALK (Head Bleed)
        cross = params.get('crosstalk', 0.0)
        if cross > 0:
            l, r = out_v[:, 0], out_v[:, 1]
            out_v[:, 0] = l * (1.0 - cross) + r * cross
            out_v[:, 1] = r * (1.0 - cross) + l * cross

        # 3. MAGNETIC SATURATION & BIAS
        drive = params.get('drive', 1.2)
        bias = params.get('bias', 1.0)
        proc = np.tanh(out_v * drive * bias) / (drive * 0.5 + 0.5)

        # 4. BARKHAUSEN NOISE (Magnetic Domain Snapping)
        bark = params.get('barkhausen', 0.0)
        if bark > 0:
            proc += np.random.normal(0, bark * 0.01, proc.shape) * (np.abs(proc)**2)

        # 5. ASPERITIES (Signal-Modulated Noise)
        asp = params.get('asperities', 0.0)
        if asp > 0:
            envelope = np.abs(proc)
            proc += np.random.normal(0, asp * 0.005, proc.shape) * envelope

        return proc