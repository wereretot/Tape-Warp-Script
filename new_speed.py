import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import os
from scipy.interpolate import interp1d

# Function to apply sine wave pitch modulation

def sine_wave_pitch_modulation(audio, sample_rate, min_speed, max_speed, modulation_rate):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    t = np.linspace(0, len(samples) / sample_rate, num=len(samples), endpoint=False)

    modulation = np.sin(2 * np.pi * modulation_rate * t)
    speed_factors = min_speed + (max_speed - min_speed) * (modulation + 1) / 2

    new_sample_positions = np.cumsum(speed_factors)
    new_sample_positions = (new_sample_positions / new_sample_positions[-1]) * (len(samples) - 1)

    interpolator = interp1d(np.arange(len(samples)), samples, kind='linear', bounds_error=False, fill_value='extrapolate')
    modulated_samples = interpolator(new_sample_positions)

    # Smooth the modulated samples to remove the phaser effect
    kernel_size = 5  # Length of the smoothing kernel
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_samples = np.convolve(modulated_samples, kernel, mode='same')

    smoothed_samples = np.clip(smoothed_samples, -32768, 32767).astype(np.int16)

    # Create a new audio segment from the smoothed samples
    modulated_audio = AudioSegment(
        smoothed_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    return modulated_audio

# Function to normalize audio levels for consistent output
def normalize_audio(audio):
    return audio.apply_gain(-audio.max_dBFS)

# Function to load and process audio file
def process_audio(file_path, min_speed, max_speed, modulation_rate):
    try:
        audio = AudioSegment.from_file(file_path)
        sample_rate = audio.frame_rate

        # Normalize audio for .opus files to prevent distortion
        if file_path.lower().endswith(".opus"):
            audio = normalize_audio(audio)

        modulated_audio = sine_wave_pitch_modulation(audio, sample_rate, min_speed, max_speed, modulation_rate)

        script_directory = os.path.dirname(os.path.realpath(__file__))
        file_extension = ".mp3"
        output_file = os.path.join(script_directory, os.path.splitext(os.path.basename(file_path))[0] + "_modulated" + file_extension)
        modulated_audio.export(output_file, format=file_extension[1:])

        play(modulated_audio)
        messagebox.showinfo("Success", f"Audio processed and saved to: {output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI for the application
def main():
    def browse_file():
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.opus")])
        if file_path:
            file_path_var.set(file_path)

    def start_processing():
        file_path = file_path_var.get()
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Invalid file path")
            return

        try:
            min_speed = float(min_speed_var.get())
            max_speed = float(max_speed_var.get())
            modulation_rate = float(modulation_rate_var.get())

            if min_speed <= 0 or max_speed <= 0 or modulation_rate <= 0:
                raise ValueError("Speed and rate values must be positive.")

            if min_speed >= max_speed:
                raise ValueError("Min speed must be less than max speed.")

            process_audio(file_path, min_speed, max_speed, modulation_rate)

        except ValueError as ve:
            messagebox.showerror("Error", str(ve))

    root = tk.Tk()
    root.title("Sine Wave Pitch Modulation")

    tk.Label(root, text="Audio File:").grid(row=0, column=0, padx=10, pady=10)
    file_path_var = tk.StringVar()
    tk.Entry(root, textvariable=file_path_var, width=50).grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=10, pady=10)

    tk.Label(root, text="Min Speed:").grid(row=1, column=0, padx=10, pady=10)
    min_speed_var = tk.StringVar(value="0.98")
    tk.Entry(root, textvariable=min_speed_var).grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Max Speed:").grid(row=2, column=0, padx=10, pady=10)
    max_speed_var = tk.StringVar(value="1.0")
    tk.Entry(root, textvariable=max_speed_var).grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Modulation Rate (Hz):").grid(row=3, column=0, padx=10, pady=10)
    modulation_rate_var = tk.StringVar(value="0.9")
    tk.Entry(root, textvariable=modulation_rate_var).grid(row=3, column=1, padx=10, pady=10)

    tk.Button(root, text="Start", command=start_processing).grid(row=4, column=0, columnspan=3, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
