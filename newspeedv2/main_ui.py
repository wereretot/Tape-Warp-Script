import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sounddevice as sd
from engine import TapeEngine

T_BG = "#0D0D0E"
T_ACCENT = "#FFB000" 

class ForensicIPSApp:
    def __init__(self, root):
        self.root = root
        self.engine = TapeEngine()
        
        self.root.title("NAGRA-V ANALOG FORENSICS : : IPS TELEMETRY")
        self.root.geometry("1100x900")
        self.root.configure(bg=T_BG)
        
        self.setup_ui()
        self.update_telemetry()

    def setup_ui(self):
        # --- TECHNICAL HEADER ---
        header = tk.Frame(self.root, bg="#000", height=100, bd=1, relief="solid")
        header.pack(fill="x", side="top", padx=10, pady=10)
        
        # The IPS Readout (Live)
        self.lbl_ips = tk.Label(header, text="IPS: 00.000", bg="#000", fg=T_ACCENT, 
                                font=("Consolas", 32, "bold"), width=12)
        self.lbl_ips.pack(side="right", padx=20)
        
        # Machine Status Readout
        self.lbl_status = tk.Label(header, text="SYSTEM: IDLE\nVOLTAGE: STABLE", bg="#000", fg="#444", 
                                   font=("Consolas", 10), justify="left")
        self.lbl_status.pack(side="left", padx=20)

        # --- SPEED SELECTOR ---
        top_bar = tk.Frame(self.root, bg=T_BG)
        top_bar.pack(fill="x", padx=20, pady=5)

        tk.Label(top_bar, text="CAPSTAN REFERENCE (IPS):", bg=T_BG, fg="white", font=("Consolas", 10)).pack(side="left")
        self.ips_var = tk.DoubleVar(value=15.0)
        ips_selector = ttk.Combobox(top_bar, textvariable=self.ips_var, values=[30.0, 15.0, 7.5, 3.75, 1.875], width=10)
        ips_selector.pack(side="left", padx=10)
        ips_selector.bind("<<ComboboxSelected>>", lambda e: self.sync())

        # --- RACK CONTROLS ---
        rack = tk.Frame(self.root, bg=T_BG)
        rack.pack(fill="both", expand=True, padx=20, pady=10)

        m1 = self.create_rack(rack, "POWER & MOTOR DYNAMICS")
        self.ctrls = {
            "motor_health": self.add_slider(m1, "VOLTAGE DRIFT (SURGE)", 0, 10, 0.5),
            "motor_drag": self.add_slider(m1, "TORQUE LOAD (DRAG)", 0, 0.9, 0),
            "wow_hz": self.add_slider(m1, "WOW FREQUENCY", 0.1, 5, 0.5),
            "wow_dep": self.add_slider(m1, "WOW INTENSITY", 0, 30, 0.2),
            "flutter_hz": self.add_slider(m1, "FLUTTER FREQUENCY", 10, 100, 15),
            "flutter_dep": self.add_slider(m1, "FLUTTER INTENSITY", 0, 10, 0.05),
        }

        m2 = self.create_rack(rack, "MAGNETIC FLUX PROPERTIES")
        self.ctrls.update({
            "drive": self.add_slider(m2, "HEAD SATURATION", 1, 15, 1.2),
            "bias": self.add_slider(m2, "AC BIAS TUNING", 0.5, 2.5, 1.0),
            "hiss": self.add_slider(m2, "NOISE FLOOR", 0, 0.02, 0.001),
            "cutoff_base": self.add_slider(m2, "AZIMUTH MASTER (Hz)", 500, 20000, 18000),
        })

        # --- FOOTER ---
        footer = tk.Frame(self.root, bg="#111", height=100)
        footer.pack(fill="x", side="bottom")

        self.btn_play = tk.Button(footer, text="ENGAGE TAPE", command=self.toggle_play,
                                  bg="#222", fg=T_ACCENT, font=("Consolas", 18), width=15)
        self.btn_play.pack(side="left", padx=20, pady=20)

        tk.Button(footer, text="LOAD MEDIA", command=self.load_file, bg="#333", fg="white", width=12).pack(side="left")
        tk.Button(footer, text="RENDER TO DISK", command=self.export, bg=T_ACCENT, fg="black", font=("Consolas", 12, "bold")).pack(side="right", padx=20)

    def create_rack(self, parent, title):
        f = tk.LabelFrame(parent, text=f" {title} ", bg="#181819", fg=T_ACCENT, font=("Consolas", 10, "bold"), bd=1, padx=10, pady=10)
        f.pack(fill="x", pady=5)
        return f

    def add_slider(self, parent, label, mn, mx, df):
        row = tk.Frame(parent, bg="#181819")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg="#181819", fg="#888", font=("Consolas", 9), width=22, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=df)
        tk.Scale(row, variable=v, from_=mn, to=mx, resolution=0.001, orient="horizontal", showvalue=False, 
                 bg="#181819", highlightthickness=0, troughcolor="#000", command=lambda x: self.sync()).pack(side="left", fill="x", expand=True)
        return v

    def sync(self):
        self.engine.params["base_ips"] = self.ips_var.get()
        for k, v in self.ctrls.items(): self.engine.params[k] = v.get()

    def update_telemetry(self):
        if self.engine.is_playing:
            # Multiply master IPS by the current real-time physics factor
            actual_ips = self.engine.params["base_ips"] * self.engine.last_instant_speed
            self.lbl_ips.config(text=f"IPS: {actual_ips:06.3f}")
            
            # Change status text based on drift
            if abs(self.engine.last_instant_speed - 1.0) > 0.05:
                self.lbl_status.config(text="SYSTEM: RUNNING\nVOLTAGE: UNSTABLE", fg="#FF5555")
            else:
                self.lbl_status.config(text="SYSTEM: RUNNING\nVOLTAGE: NOMINAL", fg=T_ACCENT)
        else:
            self.lbl_ips.config(text="IPS: 00.000")
            self.lbl_status.config(text="SYSTEM: IDLE\nVOLTAGE: STABLE", fg="#444")
            
        self.root.after(40, self.update_telemetry)

    def toggle_play(self):
        if self.engine.is_playing:
            self.engine.is_playing = False
            self.btn_play.config(text="ENGAGE TAPE", fg=T_ACCENT)
        elif self.engine.audio_data is not None:
            self.engine.is_playing = True
            self.btn_play.config(text="STOP TRANSPORT", fg="#FF5555")
            threading.Thread(target=self.audio_thread, daemon=True).start()

    def audio_thread(self):
        def cb(o, f, t, s):
            d = self.engine.dsp_process(f)
            if d is None: raise sd.CallbackStop
            o[:] = d
        with sd.OutputStream(samplerate=44100, channels=2, callback=cb):
            while self.engine.is_playing: sd.sleep(100)
        self.engine.is_playing = False
        self.root.after(0, lambda: self.btn_play.config(text="ENGAGE TAPE", fg=T_ACCENT))

    def load_file(self):
        p = filedialog.askopenfilename()
        if p: self.engine.load_file(p)

    def export(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav")
        if not path: return
        self.engine.play_head = 0
        chunks = []
        while True:
            c = self.engine.dsp_process(44100)
            if c is None: break
            chunks.append(c)
        import numpy as np
        from pydub import AudioSegment
        out = (np.vstack(chunks) * 32767).astype(np.int16)
        AudioSegment(out.tobytes(), frame_rate=44100, sample_width=2, channels=2).export(path, format="wav")
        messagebox.showinfo("Export", "Offline Forensic Render Complete")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TCombobox", fieldbackground="#222", background="#444", foreground="white")
    app = ForensicIPSApp(root)
    root.mainloop()