import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np
import os
import random
import sounddevice as sd
from pydub import AudioSegment
from scipy.signal import butter, lfilter

class ForensicTapeStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Analog Forensics: Polished Master Suite")
        self.root.geometry("1000x980")
        
        # Audio Engine State
        self.audio_data = None
        self.total_samples = 0
        self.play_head = 0.0
        self.current_time = 0.0
        self.stream = None
        self.is_playing = False
        self.lock = threading.Lock()
        
        # --- Physical State (Persistent across blocks) ---
        self.filter_zi = None
        self.hiss_zi = None
        self.current_motor_speed = 1.0  
        self.belt_slip_val = 0.0      
        self.surge_state = 0.0          # Smoothed voltage jitter
        
        self.controls = {} 
        self.setup_gui()
        self.apply_builtin_preset("Studio Reel (15ips)")

    def setup_gui(self):
        # --- Top Header ---
        top_frame = tk.Frame(self.root, bg="#0a0a0a", pady=15)
        top_frame.pack(fill="x")
        
        tk.Button(top_frame, text="IMPORT TAPE", command=self.load_file, 
                  bg="#1b5e20", fg="white", font=("Courier", 10, "bold")).pack(side="left", padx=20)
        
        self.lbl_file = tk.Label(top_frame, text="READY...", bg="#0a0a0a", fg="#00ff00", font=("Courier", 10))
        self.lbl_file.pack(side="left")

        preset_ctrl = tk.Frame(top_frame, bg="#0a0a0a")
        preset_ctrl.pack(side="right", padx=20)
        
        self.preset_var = tk.StringVar()
        builtins = [
            "Studio Reel (15ips)", "Radio Broadcast 1970", "Factory Fresh",
            "Chrome Type II (Clean)", "Ferric Type I (Cheapo)", "Walkman (Low Battery)",
            "VHS Standard Play", "VHS Long Play (EP)", "Answering Machine",
            "Worn Drive Belt", "Sticky Shed Syndrome", "Capstan Slip",
            "Garage Sale Find", "Sun-Warped Reel", "Basement Flood (Moldy)",
            "Sand in the Gears", "Burned Motor", "Melted Plastic", "End of Life"
        ]
        self.cb_presets = ttk.Combobox(preset_ctrl, textvariable=self.preset_var, values=builtins, state="readonly", width=25)
        self.cb_presets.pack(side="left", padx=10)
        self.cb_presets.bind("<<ComboboxSelected>>", lambda e: self.apply_builtin_preset(self.preset_var.get()))
        
        tk.Button(preset_ctrl, text="PROCEDURAL AGE", command=self.procedural_age_tape, bg="#d84315", fg="white").pack(side="left", padx=5)

        # --- Transport Bar ---
        trans_frame = tk.Frame(self.root, pady=10)
        trans_frame.pack()
        self.btn_play = tk.Button(trans_frame, text="[ ENGAGE DRIVE ]", command=self.toggle_playback, 
                                  width=25, state="disabled", font=("Arial", 10, "bold"))
        self.btn_play.pack(side="left", padx=10)
        self.btn_export = tk.Button(trans_frame, text="[ BOUNCE TO DISK ]", command=self.export_audio, width=20, state="disabled")
        self.btn_export.pack(side="left", padx=10)

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=15, pady=10)

        tab_mech = tk.Frame(tabs)
        tabs.add(tab_mech, text=" Mechanical Health ")
        self.motor_health = self.create_complex_slider(tab_mech, "Motor Surge (Voltage Chaos)", 0.0, 10.0, 0.1)
        self.motor_drag = self.create_complex_slider(tab_mech, "Motor Torque Loss (Slowdown)", 0.0, 1.0, 0.0)
        self.belt_slip = self.create_complex_slider(tab_mech, "Belt Slippage (Friction Dips)", 0.0, 20.0, 0.2)
        self.wow_hz = self.create_complex_slider(tab_mech, "Wow Rate (Cycle Speed)", 0.1, 8.0, 0.5)
        self.wow_dep = self.create_complex_slider(tab_mech, "Wow Depth (Pitch Sway)", 0.0, 45.0, 0.5)

        tab_elec = tk.Frame(tabs)
        tabs.add(tab_elec, text=" Signal & Noise ")
        self.hiss = self.create_complex_slider(tab_elec, "Organic Hiss Floor", 0.0, 0.01, 0.001)
        self.hiss_grit = self.create_complex_slider(tab_elec, "Hiss Texture (Filter)", 100, 8000, 3000)
        self.drive = self.create_complex_slider(tab_elec, "Saturation Drive", 1.0, 50.0, 1.2)
        self.cutoff = self.create_complex_slider(tab_elec, "High-Frequency Loss", 200, 20000, 18000)

        self.controls = {
            "motor_health": self.motor_health, "motor_drag": self.motor_drag, "belt_slip": self.belt_slip,
            "wow_hz": self.wow_hz, "wow_dep": self.wow_dep, "hiss": self.hiss, "hiss_grit": self.hiss_grit,
            "drive": self.drive, "cutoff": self.cutoff
        }

    def create_complex_slider(self, parent, label, mn, mx, df):
        frame = tk.Frame(parent, pady=12)
        frame.pack(fill="x", padx=20)
        lbl_frame = tk.Frame(frame)
        lbl_frame.pack(fill="x")
        tk.Label(lbl_frame, text=label, font=("Arial", 9, "bold"), width=30, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=df)
        entry = tk.Entry(lbl_frame, width=10, justify="center")
        entry.insert(0, f"{df:.4f}")
        entry.pack(side="right")
        canvas = tk.Canvas(frame, height=10, bg="#222", highlightthickness=0)
        canvas.pack(fill="x", pady=(5,0))
        self.draw_gradient(canvas)
        slider = tk.Scale(frame, variable=var, from_=mn, to=mx, resolution=0.0001, orient="horizontal", showvalue=False)
        slider.pack(fill="x")
        def update_ui(*args):
            entry.delete(0, tk.END)
            entry.insert(0, f"{var.get():.4f}")
        var.trace_add("write", update_ui)
        return var

    def draw_gradient(self, canvas):
        canvas.update()
        w = canvas.winfo_width() if canvas.winfo_width() > 1 else 600
        for i in range(w):
            rel = i/w
            r = int(255 * rel) if rel > 0.5 else int(510 * rel)
            g = int(255 * (1-rel)) if rel < 0.5 else int(510 * (1-rel))
            canvas.create_line(i, 0, i, 10, fill=f'#{min(255,r):02x}{min(255,g):02x}00')

    def apply_builtin_preset(self, name):
        p = {
            "Studio Reel (15ips)": {"motor_health":0.01, "motor_drag":0.0, "belt_slip":0.0, "wow_hz":0.2, "wow_dep":0.02, "hiss":0.00005, "hiss_grit":8000, "drive":1.05, "cutoff":22000},
            "Radio Broadcast 1970": {"motor_health":0.2, "motor_drag":0.0, "belt_slip":0.1, "wow_hz":0.5, "wow_dep":0.15, "hiss":0.001, "hiss_grit":4500, "drive":2.1, "cutoff":14000},
            "Factory Fresh": {"motor_health":0.05, "motor_drag":0.0, "belt_slip":0.05, "wow_hz":0.4, "wow_dep":0.05, "hiss":0.0001, "hiss_grit":6000, "drive":1.1, "cutoff":20000},
            "Walkman (Low Battery)": {"motor_health":1.5, "motor_drag":0.35, "belt_slip":1.2, "wow_hz":0.4, "wow_dep":3.5, "hiss":0.005, "hiss_grit":2800, "drive":1.8, "cutoff":9000},
            "End of Life": {"motor_health":10.0, "motor_drag":0.95, "belt_slip":15.0, "wow_hz":6.0, "wow_dep":30.0, "hiss":0.009, "hiss_grit":800, "drive":10.0, "cutoff":2500},
        }
        if name in p:
            for k, v in p[name].items(): self.controls[k].set(v)

    def procedural_age_tape(self):
        sev = random.uniform(0.1, 1.0)
        self.controls['motor_health'].set(random.uniform(0, sev * 10))
        self.controls['motor_drag'].set(random.uniform(0, sev * 0.9))
        self.controls['belt_slip'].set(random.uniform(0, sev * 20))
        self.controls['wow_dep'].set(random.uniform(0, sev * 45))
        self.controls['hiss'].set(random.uniform(0.0001, 0.01 * sev))
        self.controls['cutoff'].set(20000 - (sev * 18000))

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.opus *.wav *.mp3 *.ogg")])
        if not path: return
        self.lbl_file.config(text="LOADING MAGNETIC DATA...", fg="orange")
        def _load():
            try:
                audio = AudioSegment.from_file(path).set_frame_rate(44100).set_channels(2).set_sample_width(2)
                audio = audio.apply_gain(-audio.max_dBFS - 1.0)
                samples = np.array(audio.get_array_of_samples()).reshape((-1, 2))
                with self.lock:
                    self.audio_data = samples.astype(np.float32) / 32768.0
                    self.total_samples = len(self.audio_data)
                    self.play_head = 0.0
                    self.current_time = 0.0
                    self.current_motor_speed = 1.0
                    self.belt_slip_val = 0.0
                self.root.after(0, lambda: [self.btn_play.config(state="normal"), self.btn_export.config(state="normal"),
                                           self.lbl_file.config(text=f"TAPE READY: {os.path.basename(path)}", fg="#00ff00")])
            except Exception as e: self.root.after(0, lambda: messagebox.showerror("System Error", str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def dsp_process(self, frames):
        """Polished Physics Engine: Sample-accurate speed transitions to prevent clicks."""
        with self.lock:
            if self.audio_data is None:
                return None

            # Hard EOF stop for both realtime and offline render
            if self.play_head >= self.total_samples - 2:
                self.is_playing = False
                return None

            c = {k: v.get() for k, v in self.controls.items()}
            
            # --- Vectorized Physics Generation ---
            # 1. Torque & Inertia: Smoothly move current_motor_speed toward target
            target_base = 1.0 - c['motor_drag']
            inertia = 0.0005 
            
            # 2. Voltage Surge: Smoothed jitter (Flutter)
            raw_surge = np.random.normal(0, 1.0, size=frames)
            # Use 1-pole filter to keep jitter organic and click-free
            surge_samples = np.zeros(frames)
            for i in range(frames):
                self.surge_state += (raw_surge[i] - self.surge_state) * 0.01
                surge_samples[i] = self.surge_state * (c['motor_health'] / 180.0)

            # 3. Belt Slip: Trigger based on probability
            if np.random.random() < (c['belt_slip'] / 1500.0):
                self.belt_slip_target = - (c['belt_slip'] / 40.0)
            else:
                if not hasattr(self, 'belt_slip_target'): self.belt_slip_target = 0.0
                self.belt_slip_target *= 0.995 # Recovery decay

            # 4. Composite Speed Array (Sample-accurate)
            t_arr = self.current_time + np.arange(frames) / 44100.0
            wow_arr = (c['wow_dep'] / 150.0) * np.sin(2 * np.pi * c['wow_hz'] * t_arr)
            
            final_speeds = np.zeros(frames)
            for i in range(frames):
                # Smooth speed transitions prevent click artifacts
                self.current_motor_speed += (target_base - self.current_motor_speed) * inertia
                self.belt_slip_val += (self.belt_slip_target - self.belt_slip_val) * 0.05
                final_speeds[i] = self.current_motor_speed + surge_samples[i] + self.belt_slip_val + wow_arr[i]
            
            final_speeds = np.clip(final_speeds, 0.01, 3.0)
            # Continuous fractional phase integration (click-free)
            read_indices = self.play_head + np.cumsum(final_speeds) - final_speeds[0]

            
            # EOF Handling
            if read_indices[0] >= self.total_samples - 2:
                self.is_playing = False
                return None
            valid_mask = read_indices < self.total_samples
            n_valid = int(np.nonzero(valid_mask)[0][-1]) + 1 if valid_mask.any() else 0
            if n_valid == 0: return None
            
            # Audio Fetch & Interp (phase-stable)
            out = np.zeros((frames, 2), dtype=np.float32)

            global_idx = np.arange(self.total_samples)

            out_v = np.zeros((n_valid, 2), dtype=np.float32)
            out_v[:, 0] = np.interp(read_indices[:n_valid], global_idx, self.audio_data[:, 0])
            out_v[:, 1] = np.interp(read_indices[:n_valid], global_idx, self.audio_data[:, 1])

            # Processing valid portion
            proc = np.tanh(out_v * c['drive']) / (c['drive'] * 0.1 + 0.9)
            
            # Filtered Noise
            raw_hiss = np.random.normal(0, c['hiss'], (n_valid, 2))
            b_h, a_h = butter(1, max(0.001, c['hiss_grit']/22050), btype='low')
            if self.hiss_zi is None or self.hiss_zi.shape[0] != len(a_h)-1:
                self.hiss_zi = np.zeros((len(a_h)-1, 2))
            hiss_f, self.hiss_zi = lfilter(b_h, a_h, raw_hiss, axis=0, zi=self.hiss_zi)
            proc += hiss_f * (np.abs(proc) * 0.2 + 0.8)

            # High Cut
            if c['cutoff'] < 19800:
                b, a = butter(2, max(0.001, c['cutoff']/22050), btype='low')
                if self.filter_zi is None or self.filter_zi.shape[0] != len(a)-1:
                    self.filter_zi = np.zeros((len(a)-1, 2))
                proc, self.filter_zi = lfilter(b, a, proc, axis=0, zi=self.filter_zi)

            # End of file smoothing (Anti-click fade)
            if n_valid < frames:
                fade_len = min(n_valid, 100)
                proc[n_valid-fade_len:n_valid] *= np.linspace(1, 0, fade_len)[:, None]
                self.is_playing = False

            out[:n_valid] = proc
            self.play_head = float(read_indices[n_valid - 1])
            self.current_time += frames / 44100.0
            return np.clip(out, -1.0, 1.0)

    def audio_callback(self, outdata, frames, time_info, status):
        if not self.is_playing: raise sd.CallbackStop
        processed = self.dsp_process(frames)
        if processed is None: 
            self.is_playing = False
            outdata.fill(0)
            raise sd.CallbackStop
        outdata[:] = processed

    def toggle_playback(self):
        if self.is_playing:
            self.is_playing = False
            if self.stream: self.stream.stop(); self.stream.close()
            self.btn_play.config(text="[ ENGAGE DRIVE ]")
        else:
            self.is_playing = True
            self.stream = sd.OutputStream(samplerate=44100, channels=2, callback=self.audio_callback, blocksize=2048)
            self.stream.start()
            self.btn_play.config(text="[ STOP ENGINE ]")

    def export_audio(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not path: return
        self.btn_export.config(text="RENDERING...", state="disabled")
        def _export():
            with self.lock:
                old_h, old_t, old_s = self.play_head, self.current_time, self.current_motor_speed
                self.play_head = 0.0; self.current_time = 0.0; self.current_motor_speed = 1.0
                self.filter_zi = None; self.hiss_zi = None
                self.surge_state = 0.0; self.belt_slip_val = 0.0
                self.is_playing = True

            rendered_chunks = []
            chunk_frames = 44100 * 2 # Process in 2s chunks
            while True:
                chunk = self.dsp_process(chunk_frames)
                if chunk is None: break
                rendered_chunks.append(chunk)
                progress = (self.play_head / self.total_samples) * 100 if self.total_samples > 0 else 0
                self.root.after(0, lambda p=progress: self.btn_export.config(text=f"RENDERING {p:.1f}%"))

            with self.lock:
                self.play_head, self.current_time, self.current_motor_speed = old_h, old_t, old_s

            if rendered_chunks:
                out = np.vstack(rendered_chunks)
                out_int = (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16)
                AudioSegment(out_int.tobytes(), frame_rate=44100, sample_width=2, channels=2).export(path, format="wav")
                self.root.after(0, lambda: messagebox.showinfo("Success", "Tape Rendered."))
            self.root.after(0, lambda: self.btn_export.config(text="[ BOUNCE TO DISK ]", state="normal"))
        threading.Thread(target=_export, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ForensicTapeStudio(root)
    root.mainloop()