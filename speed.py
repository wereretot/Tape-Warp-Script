import os
import random
import threading
from pydub import AudioSegment
from tqdm import tqdm
from configparser import ConfigParser
import math
import tkinter as tk
from tkinter import filedialog

class LinearWarpThread(threading.Thread):
    def __init__(self, audio, start_time, end_time, speed_ratio, pitch_ratio, temperature, wobble_factor):
        threading.Thread.__init__(self)
        self.audio = audio
        self.start_time = start_time
        self.end_time = end_time
        self.speed_ratio = speed_ratio
        self.pitch_ratio = pitch_ratio
        self.temperature = temperature
        self.wobble_factor = wobble_factor

    def run(self):
        chunk = self.audio[self.start_time:self.end_time]
        chunk_with_altered_speed = self.warp_speed(chunk)
        chunk_with_altered_pitch = self.warp_pitch(chunk_with_altered_speed)
        self.chunk = chunk_with_altered_pitch

    def warp_speed(self, chunk):
        speed = self.speed_ratio

        speed = random.uniform(0.98, 1.01) - (self.temperature / 1000)

        return chunk._spawn(chunk.raw_data, overrides={"frame_rate": int(chunk.frame_rate / speed)})

    def warp_pitch(self, chunk):
        pitch = self.pitch_ratio
        pitch = random.uniform(0.98, 1.01) - (self.temperature / 1000)

        pitch_factor = pitch # Convert pitch to pitch factor

        return chunk._spawn(chunk.raw_data, overrides={"frame_rate": int(chunk.frame_rate / pitch_factor)})

    def apply_wobble(self, chunk):
        if len(chunk) < 100:
            return chunk

            wobble_speed = random.uniform(0.98, 1.01) * self.wobble_factor / (self.temperature / 500)
            wobble_pitch = random.uniform(0.98, 1.01) * self.wobble_factor / (self.temperature / 500)
            chunk = chunk.speedup(playback_speed=wobble_speed)
            chunk = chunk._spawn(chunk.raw_data, overrides={"frame_rate": int(chunk.frame_rate * wobble_pitch)})

        return chunk


def ensure_directory_exists(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_settings():
    config = ConfigParser()
    config.read("config.ini")

    settings = {}
    settings['speed_min'] = float(config.get('Settings', 'speed_min'))
    settings['speed_max'] = float(config.get('Settings', 'speed_max'))
    settings['ambient_temperature'] = float(config.get('Settings', 'ambient_temperature'))
    settings['wobble_factor'] = float(config.get('Settings', 'wobble_factor'))

    return settings


def change_speed_and_pitch(filename, output_filename, speed_range, ambient_temperature, wobble_factor):
    audio = AudioSegment.from_file(filename)

    duration = len(audio)
    progress_bar = tqdm(total=duration, unit='ms', ncols=60)

    threads = []

    current_time = 0
    while current_time < duration:
        speed = random.uniform(speed_range[0], speed_range[1])
        pitch = random.uniform(speed_range[0], speed_range[1])

        next_time = min(current_time + 100, duration)
        thread = LinearWarpThread(audio, current_time, next_time, speed, pitch, ambient_temperature, wobble_factor)
        threads.append(thread)
        thread.start()

        current_time += 100
        progress_bar.update(100)

    progress_bar.close()

    new_audio = AudioSegment.empty()
    for thread in threads:
        thread.join()
        new_chunk = thread.apply_wobble(thread.chunk)
        new_audio += new_chunk

    ensure_directory_exists(output_filename)
    new_audio.export(output_filename, format='wav')


def convert_to_wav(filename):
    base, ext = os.path.splitext(filename)
    if ext.lower() == ".wav":
        return filename

    output_filename = base + "temp" + ".wav"
    audio = AudioSegment.from_file(filename)
    ensure_directory_exists(output_filename)
    audio.export(output_filename, format='wav')
    return output_filename


def select_input_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Input File",
        filetypes=(("Audio files", "*.wav *.mp3 *.opus *.ogg *.flac *.webm"), ("All files", "*.*"))
    )
    return file_path


def main():
    input_filename = select_input_file()
    if not input_filename:
        print("No input file selected.")
        return

    output_filename = "tape_warp/output_file.wav"
    input_filename = convert_to_wav(input_filename)
    settings = load_settings()
    change_speed_and_pitch(
        input_filename,
        output_filename,
        (settings['speed_min'], settings['speed_max']),
        settings['ambient_temperature'],
        settings['wobble_factor']
    )
    print("Audio processing complete.")


if __name__ == "__main__":
    main()
