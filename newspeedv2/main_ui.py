import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sounddevice as sd
import numpy as np
from pydub import AudioSegment

# Import our modular engine
from engine import TapeEngine

class ForensicTapeStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("NAGRA-V ANALOG FORENSICS : : MODULAR ARCHITECTURE")
        self.root.geometry("1100x1000")
        self.root.configure(bg="#0D0D0E")
        
        self.engine = TapeEngine()
        self.controls = {}
        
        self.setup_gui()
        self.apply_builtin_preset("Master Studio Reel (30ips)")
        self.update_telemetry()

    def setup_gui(self):
        header = tk.Frame(self.root, bg="#000", height=100, bd=1, relief="solid")
        header.pack(fill="x", side="top", padx=10, pady=10)
        
        self.lbl_ips = tk.Label(header, text="IPS: 00.000", bg="#000", fg="#FFB000", font=("Consolas", 32, "bold"), width=12)
        self.lbl_ips.pack(side="right", padx=20)
        
        self.lbl_status = tk.Label(header, text="SYSTEM: IDLE\nVOLTAGE: STABLE", bg="#000", fg="#444", font=("Consolas", 10), justify="left")
        self.lbl_status.pack(side="left", padx=20)

        top_bar = tk.Frame(self.root, bg="#0D0D0E")
        top_bar.pack(fill="x", padx=20, pady=5)

        tk.Button(top_bar, text="LOAD TAPE", command=self.load_file, bg="#1b5e20", fg="white").pack(side="left", padx=5)
        
        tk.Label(top_bar, text="PRESET:", bg="#0D0D0E", fg="white", padx=10).pack(side="left")
        self.preset_var = tk.StringVar()
        presets = ["Master Studio Reel (30ips)", "Standard Tape (15ips)", "Consumer Cassette (1.875ips)", 
                   "VHS Linear Audio", "Sticky Shed Disaster", "Abandoned Attic Find"]
        cb = ttk.Combobox(top_bar, textvariable=self.preset_var, values=presets, state="readonly", width=25)
        cb.pack(side="left", padx=5)
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_builtin_preset(self.preset_var.get()))

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=15, pady=10)

        # Build UI Tabs mapped to modules
        t_mech = tk.Frame(self.tabs, bg="#181819")
        self.tabs.add(t_mech, text=" Transport Mechanics ")
        self.ctrl("ips_base", t_mech, "CAPSTAN SPEED (IPS)", 0.5, 30.0, 15.0)
        self.ctrl("motor_health", t_mech, "VOLTAGE DRIFT (JITTER)", 0.0, 10.0, 0.5)
        self.ctrl("motor_drag", t_mech, "TORQUE LOAD", 0.0, 0.9, 0.0)
        self.ctrl("wow_dep", t_mech, "WOW INTENSITY", 0.0, 30.0, 0.2)
        self.ctrl("flutter_dep", t_mech, "FLUTTER INTENSITY", 0.0, 10.0, 0.05)
        self.ctrl("scrape_flutter", t_mech, "SCRAPE FLUTTER (3kHz)", 0.0, 1.0, 0.1)
        self.ctrl("tension_load", t_mech, "REEL TENSION DYNAMICS", 0.0, 0.5, 0.05)

        t_mag = tk.Frame(self.tabs, bg="#181819")
        self.tabs.add(t_mag, text=" Magnetic Flux ")
        self.ctrl("drive", t_mag, "HEAD SATURATION", 1.0, 20.0, 1.2)
        self.ctrl("bias", t_mag, "AC BIAS TUNING", 0.5, 3.0, 1.0)
        self.ctrl("asperities", t_mag, "ASPERITIES (MOD NOISE)", 0.0, 0.5, 0.05)
        self.ctrl("barkhausen", t_mag, "BARKHAUSEN GRAIN", 0.0, 0.1, 0.01)
        self.ctrl("crosstalk", t_mag, "STEREO CROSSTALK", 0.0, 0.5, 0.02)
        self.ctrl("print_through", t_mag, "PRINT-THROUGH (GHOST)", 0.0, 0.1, 0.0)

        t_elec = tk.Frame(self.tabs, bg="#181819")
        self.tabs.add(t_elec, text=" Electronics & Wear ")
        self.ctrl("hiss", t_elec, "NOISE FLOOR", 0.0, 0.02, 0.001)
        self.ctrl("mains_hum", t_elec, "60Hz MAINS HUM", 0.0, 0.05, 0.001)
        self.ctrl("cutoff_base", t_elec, "AZIMUTH CUTOFF (Hz)", 500, 22000, 18000)
        self.ctrl("head_bump", t_elec, "HEAD BUMP (LF EQ)", 0.0, 5.0, 0.5)
        self.ctrl("azimuth_drift", t_elec, "AZIMUTH PHASE DRIFT", 0.0, 1.0, 0.05)
        self.ctrl("sticky_shed", t_elec, "STICKY SHED INTENSITY", 0.0, 1.0, 0.0)

        # Footer
        footer = tk.Frame(self.root, bg="#111", height=100)
        footer.pack(fill="x", side="bottom")

        self.btn_play = tk.Button(footer, text="ENGAGE TAPE", command=self.toggle_playback, bg="#222", fg="#FFB000", font=("Consolas", 18), width=15)
        self.btn_play.pack(side="left", padx=20, pady=20)
        
        self.btn_export = tk.Button(footer, text="FORENSIC RENDER", command=self.export_audio, bg="#FFB000", fg="black", font=("Consolas", 12, "bold"))
        self.btn_export.pack(side="right", padx=20)

    def ctrl(self, key, parent, label, mn, mx, df):
        row = tk.Frame(parent, bg="#181819", pady=5)
        row.pack(fill="x", padx=20)
        tk.Label(row, text=label, bg="#181819", fg="#888", font=("Consolas", 9), width=25, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=df)
        s = tk.Scale(row, variable=v, from_=mn, to=mx, resolution=0.0001, orient="horizontal", showvalue=False, bg="#181819", highlightthickness=0, troughcolor="#000")
        s.pack(side="left", fill="x", expand=True)
        v.trace_add("write", lambda *a: self.sync())
        self.controls[key] = v

    def sync(self):
        for k, v in self.controls.items():
            self.engine.params[k] = v.get()

    def apply_builtin_preset(self, name):
        p = {
            "Master Studio Reel (30ips)": {"ips_base":30.0, "motor_health":0.01, "wow_dep":0.01, "drive":1.05, "hiss":0.00005, "cutoff_base":22000, "head_bump":0.2, "sticky_shed":0.0, "print_through": 0.02},
            "Consumer Cassette (1.875ips)": {"ips_base":1.875, "motor_health":0.8, "wow_dep":1.2, "drive":2.5, "hiss":0.004, "cutoff_base":12000, "crosstalk":0.2},
            "Sticky Shed Disaster": {"ips_base":7.5, "sticky_shed":1.0, "motor_health":3.0, "wow_dep":5.0, "cutoff_base":4000, "hiss":0.01},
        }
        if name in p:
            for k, v in p[name].items():
                if k in self.controls: self.controls[k].set(v)
        self.sync()

    def update_telemetry(self):
        if self.engine.is_playing:
            actual_ips = self.controls['ips_base'].get() * self.engine.transport.last_instant_speed
            self.lbl_ips.config(text=f"IPS: {actual_ips:06.3f}")
            if abs(self.engine.transport.last_instant_speed - 1.0) > 0.1:
                self.lbl_status.config(text="SYSTEM: CRITICAL\nVOLTAGE: UNSTABLE", fg="#FF5555")
            else:
                self.lbl_status.config(text="SYSTEM: RUNNING\nVOLTAGE: NOMINAL", fg="#FFB000")
        else:
            self.lbl_ips.config(text="IPS: 00.000")
            self.lbl_status.config(text="SYSTEM: IDLE\nVOLTAGE: STABLE", fg="#444")
        self.root.after(40, self.update_telemetry)

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            threading.Thread(target=self.engine.load_file, args=(path,), daemon=True).start()

    def toggle_playback(self):
        if self.engine.is_playing:
            self.engine.is_playing = False
            self.btn_play.config(text="ENGAGE TAPE", fg="#FFB000")
        elif self.engine.audio_data is not None:
            self.engine.is_playing = True
            self.btn_play.config(text="STOP TRANSPORT", fg="#FF5555")
            threading.Thread(target=self.audio_thread, daemon=True).start()

    def audio_thread(self):
        def cb(o, f, t, s):
            d = self.engine.dsp_process(f)
            if d is None: raise sd.CallbackStop
            o[:] = d
        with sd.OutputStream(samplerate=44100, channels=2, callback=cb, blocksize=1024):
            while self.engine.is_playing: sd.sleep(100)
        self.engine.is_playing = False
        self.root.after(0, lambda: self.btn_play.config(text="ENGAGE TAPE", fg="#FFB000"))

    def export_audio(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav")
        if not path: return
        self.btn_export.config(text="RENDERING...", state="disabled")
        def _run():
            with self.engine.lock:
                old_h, old_t = self.engine.play_head, self.engine.current_time
                self.engine.reset_state()
            chunks = []
            while True:
                c = self.engine.dsp_process(44100)
                if c is None: break
                chunks.append(c)
            out = (np.vstack(chunks) * 32767).astype(np.int16)
            AudioSegment(out.tobytes(), frame_rate=44100, sample_width=2, channels=2).export(path, format="wav")
            with self.engine.lock: self.engine.play_head, self.engine.current_time = old_h, old_t
            self.root.after(0, lambda: [messagebox.showinfo("Export", "Success"), self.btn_export.config(text="FORENSIC RENDER", state="normal")])
        threading.Thread(target=_run, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TNotebook", background="#0D0D0E", borderwidth=0)
    style.configure("TNotebook.Tab", background="#222", foreground="#888", padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", "#181819")], foreground=[("selected", "#FFB000")])
    app = ForensicTapeStudio(root)
    root.mainloop()