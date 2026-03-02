import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sounddevice as sd
import numpy as np
from pydub import AudioSegment

from engine import TapeEngine

class ForensicTapeStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("NAGRA-V ANALOG FORENSICS : : MODULAR ARCHITECTURE")
        self.root.geometry("1100x1060")
        self.root.configure(bg="#0D0D0E")

        self.engine = TapeEngine()
        self.controls = {}
        self.is_rewinding = False

        self.setup_gui()
        self.apply_builtin_preset("Master Studio Reel (30ips)")
        self.update_telemetry()

    # ------------------------------------------------------------------
    def setup_gui(self):
        header = tk.Frame(self.root, bg="#000", height=100, bd=1, relief="solid")
        header.pack(fill="x", side="top", padx=10, pady=10)

        self.lbl_ips = tk.Label(header, text="IPS: 00.000", bg="#000", fg="#FFB000",
                                font=("Consolas", 32, "bold"), width=12)
        self.lbl_ips.pack(side="right", padx=20)

        self.lbl_reel = tk.Label(header, text="SUPPLY: ●●●●●●●●  TAKEUP: ○○○○○○○○",
                                 bg="#000", fg="#555", font=("Consolas", 9))
        self.lbl_reel.pack(side="right", padx=10)

        self.lbl_status = tk.Label(header, text="SYSTEM: IDLE\nVOLTAGE: STABLE",
                                   bg="#000", fg="#444", font=("Consolas", 10), justify="left")
        self.lbl_status.pack(side="left", padx=20)

        top_bar = tk.Frame(self.root, bg="#0D0D0E")
        top_bar.pack(fill="x", padx=20, pady=5)

        tk.Button(top_bar, text="LOAD TAPE", command=self.load_file,
                  bg="#1b5e20", fg="white").pack(side="left", padx=5)

        tk.Label(top_bar, text="PRESET:", bg="#0D0D0E", fg="white", padx=10).pack(side="left")
        self.preset_var = tk.StringVar()
        presets = [
            "Master Studio Reel (30ips)", "Standard Tape (15ips)",
            "Consumer Cassette (1.875ips)", "Chrome Cassette (1.875ips)",
            "Metal Tape (1.875ips)", "VHS Linear Audio",
            "Sticky Shed Disaster", "Abandoned Attic Find", "Demagnetised Archive",
        ]
        cb = ttk.Combobox(top_bar, textvariable=self.preset_var, values=presets,
                          state="readonly", width=28)
        cb.pack(side="left", padx=5)
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_builtin_preset(self.preset_var.get()))

        tk.Label(top_bar, text="OXIDE:", bg="#0D0D0E", fg="white", padx=6).pack(side="left")
        self.oxide_var = tk.StringVar(value="Fe2O3")
        oxide_cb = ttk.Combobox(top_bar, textvariable=self.oxide_var,
                                values=["Fe2O3", "CrO2", "Metal", "FeCo"],
                                state="readonly", width=8)
        oxide_cb.pack(side="left", padx=2)
        oxide_cb.bind("<<ComboboxSelected>>", lambda e: self.sync())

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=15, pady=10)

        t_mech = tk.Frame(self.tabs, bg="#181819")
        self.tabs.add(t_mech, text=" Transport Mechanics ")
        self.ctrl("ips_base",       t_mech, "CAPSTAN SPEED (IPS)",        0.5,  30.0,  15.0)
        self.ctrl("motor_health",   t_mech, "VOLTAGE DRIFT (JITTER)",     0.0,  10.0,   0.5)
        self.ctrl("motor_drag",     t_mech, "TORQUE LOAD",                    0.0,   0.9,   0.0)
        self.ctrl("motor_boost",    t_mech, "MOTOR BOOST (SPEED UP)",          0.0,   0.9,   0.0)
        self.ctrl("wow_dep",        t_mech, "WOW INTENSITY",              0.0,  30.0,   0.2)
        self.ctrl("flutter_dep",    t_mech, "FLUTTER INTENSITY",          0.0,  10.0,  0.05)
        self.ctrl("scrape_flutter", t_mech, "SCRAPE FLUTTER (3kHz)",      0.0,   1.0,   0.1)
        self.ctrl("tension_load",   t_mech, "REEL TENSION DYNAMICS",      0.0,   0.5,  0.05)
        self.ctrl("dropout_rate",   t_mech, "OXIDE DROPOUT RATE",         0.0,   1.0,   0.0)

        t_mag = tk.Frame(self.tabs, bg="#181819")
        self.tabs.add(t_mag, text=" Magnetic Flux ")
        self.ctrl("drive",           t_mag, "HEAD SATURATION",            1.0,  20.0,   1.2)
        self.ctrl("bias",            t_mag, "AC BIAS TUNING",             0.5,   3.0,   1.0)
        self.ctrl("replay_diff",     t_mag, "REPLAY HEAD DIFFERENTIATION",0.0,   1.0,   0.3)
        self.ctrl("asperities",      t_mag, "ASPERITIES (MOD NOISE)",     0.0,   0.5,  0.05)
        self.ctrl("barkhausen",      t_mag, "BARKHAUSEN GRAIN",           0.0,   0.1,  0.01)
        self.ctrl("crosstalk",       t_mag, "STEREO CROSSTALK",           0.0,   0.5,  0.02)
        self.ctrl("print_through",   t_mag, "PRINT-THROUGH (GHOST)",      0.0,   0.1,   0.0)
        self.ctrl("demagnetization", t_mag, "DEMAGNETISATION (HF LOSS)",  0.0,  0.99,   0.0)
        self.ctrl("oxide_shedding",  t_mag, "OXIDE SHEDDING TEXTURE",     0.0,   1.0,   0.0)

        t_elec = tk.Frame(self.tabs, bg="#181819")
        self.tabs.add(t_elec, text=" Electronics & Wear ")
        self.ctrl("hiss",          t_elec, "NOISE FLOOR",                 0.0, 0.02,  0.001)
        self.ctrl("hiss_color",    t_elec, "HISS COLOUR (PINK TILT)",     0.0,  1.0,   0.0)
        self.ctrl("mains_hum",     t_elec, "60Hz MAINS HUM",              0.0, 0.05,  0.001)
        self.ctrl("cutoff_base",   t_elec, "AZIMUTH CUTOFF (Hz)",         500, 22000, 18000)
        self.ctrl("head_bump",     t_elec, "HEAD BUMP (LF EQ)",           0.0,  5.0,   0.5)
        self.ctrl("azimuth_drift", t_elec, "AZIMUTH PHASE DRIFT",         0.0,  1.0,  0.05)
        self.ctrl("sticky_shed",   t_elec, "STICKY SHED INTENSITY",       0.0,  1.0,   0.0)

        # ------------------------------------------------------------------
        # Footer — transport controls
        # ------------------------------------------------------------------
        footer = tk.Frame(self.root, bg="#111")
        footer.pack(fill="x", side="bottom")

        # Row 1: main transport buttons
        btn_row = tk.Frame(footer, bg="#111")
        btn_row.pack(fill="x", padx=20, pady=(15, 5))

        self.btn_play = tk.Button(btn_row, text="▶ PLAY", command=self.start_forward,
                                  bg="#222", fg="#FFB000", font=("Consolas", 14), width=12)
        self.btn_play.pack(side="left", padx=4)

        self.btn_rev_play = tk.Button(btn_row, text="◀ PLAY REVERSE", command=self.start_reverse,
                                      bg="#222", fg="#FF8C00", font=("Consolas", 14), width=16)
        self.btn_rev_play.pack(side="left", padx=4)

        self.btn_stop = tk.Button(btn_row, text="■ STOP", command=self.stop_transport,
                                  bg="#222", fg="#FF5555", font=("Consolas", 14), width=10)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_rewind = tk.Button(btn_row, text="◀◀ REWIND", command=self.toggle_rewind,
                                    bg="#222", fg="#00BFFF", font=("Consolas", 14), width=12)
        self.btn_rewind.pack(side="left", padx=4)

        self.btn_export = tk.Button(btn_row, text="FORENSIC RENDER", command=self.export_audio,
                                    bg="#FFB000", fg="black", font=("Consolas", 12, "bold"))
        self.btn_export.pack(side="right", padx=4)



    # ------------------------------------------------------------------
    def ctrl(self, key, parent, label, mn, mx, df):
        row = tk.Frame(parent, bg="#181819", pady=5)
        row.pack(fill="x", padx=20)
        tk.Label(row, text=label, bg="#181819", fg="#888", font=("Consolas", 9),
                 width=30, anchor="w").pack(side="left")
        v = tk.DoubleVar(value=df)
        tk.Scale(row, variable=v, from_=mn, to=mx, resolution=0.0001,
                 orient="horizontal", showvalue=False, bg="#181819",
                 highlightthickness=0, troughcolor="#000").pack(side="left", fill="x", expand=True)
        v.trace_add("write", lambda *a: self.sync())
        self.controls[key] = v

    def sync(self):
        for k, v in self.controls.items():
            self.engine.params[k] = v.get()
        self.engine.params['oxide_type'] = self.oxide_var.get()

    # ------------------------------------------------------------------
    def apply_builtin_preset(self, name):
        p = {
            "Master Studio Reel (30ips)": {
                "ips_base": 30.0, "motor_health": 0.01, "wow_dep": 0.01,
                "drive": 1.05, "hiss": 0.00005, "cutoff_base": 22000,
                "head_bump": 0.2, "sticky_shed": 0.0, "print_through": 0.02,
                "demagnetization": 0.0, "replay_diff": 0.3,
            },
            "Standard Tape (15ips)": {
                "ips_base": 15.0, "motor_health": 0.1, "wow_dep": 0.05,
                "drive": 1.2, "hiss": 0.0003, "cutoff_base": 18000,
                "head_bump": 0.5, "print_through": 0.01, "replay_diff": 0.3,
            },
            "Consumer Cassette (1.875ips)": {
                "ips_base": 1.875, "motor_health": 0.8, "wow_dep": 1.2,
                "drive": 2.5, "hiss": 0.004, "cutoff_base": 12000,
                "crosstalk": 0.2, "head_bump": 1.5, "replay_diff": 0.2,
            },
            "Chrome Cassette (1.875ips)": {
                "ips_base": 1.875, "motor_health": 0.5, "wow_dep": 0.8,
                "drive": 1.8, "bias": 1.35, "hiss": 0.002, "cutoff_base": 15000,
                "crosstalk": 0.1, "head_bump": 1.0, "replay_diff": 0.25,
            },
            "Metal Tape (1.875ips)": {
                "ips_base": 1.875, "motor_health": 0.3, "wow_dep": 0.5,
                "drive": 1.5, "bias": 1.7, "hiss": 0.0008, "cutoff_base": 18000,
                "crosstalk": 0.05, "head_bump": 0.8, "replay_diff": 0.3,
            },
            "Sticky Shed Disaster": {
                "ips_base": 7.5, "sticky_shed": 1.0, "motor_health": 3.0,
                "wow_dep": 5.0, "cutoff_base": 4000, "hiss": 0.01,
                "dropout_rate": 0.4, "oxide_shedding": 0.5,
            },
            "Abandoned Attic Find": {
                "ips_base": 7.5, "motor_health": 2.0, "wow_dep": 3.0,
                "drive": 3.0, "hiss": 0.008, "cutoff_base": 6000,
                "head_bump": 2.0, "demagnetization": 0.4,
                "print_through": 0.06, "dropout_rate": 0.2,
            },
            "Demagnetised Archive": {
                "ips_base": 15.0, "motor_health": 0.5, "wow_dep": 0.3,
                "drive": 1.5, "hiss": 0.002, "cutoff_base": 8000,
                "demagnetization": 0.75, "head_bump": 1.0, "print_through": 0.05,
            },
            "VHS Linear Audio": {
                "ips_base": 1.3125, "motor_health": 1.5, "wow_dep": 2.0,
                "drive": 2.0, "hiss": 0.007, "cutoff_base": 8000,
                "crosstalk": 0.3, "head_bump": 2.0, "mains_hum": 0.005,
            },
        }
        oxide_map = {
            "Chrome Cassette (1.875ips)": "CrO2",
            "Metal Tape (1.875ips)":      "Metal",
        }
        self.oxide_var.set(oxide_map.get(name, "Fe2O3"))
        if name in p:
            for k, v in p[name].items():
                if k in self.controls:
                    self.controls[k].set(v)
        self.sync()

    # ------------------------------------------------------------------
    def update_telemetry(self):
        if self.engine.total_samples > 0:
            # play_head in the reversed buffer means position from end-of-tape
            # For display, convert to forward-tape progress
            ph = self.engine.play_head
            ts = self.engine.total_samples
            fwd_progress = (1.0 - ph / ts) if self.engine.is_reversed else (ph / ts)
            fwd_progress  = float(np.clip(fwd_progress, 0, 1))
            filled    = int(fwd_progress * 8)
            supply_s  = "●" * (8 - filled) + "○" * filled
            takeup_s  = "○" * (8 - filled) + "●" * filled
        else:
            supply_s = "●●●●●●●●"
            takeup_s = "○○○○○○○○"

        if self.is_rewinding:
            tick = int(self.engine.current_time * 12) % 2
            spin = ["◑", "◐"]
            self.lbl_reel.config(
                text=f"SUPPLY: {supply_s}  {spin[tick]} REWINDING {spin[1-tick]}", fg="#00BFFF")
            self.lbl_ips.config(text=f"IPS: {self.controls['ips_base'].get() * 40.0:06.1f}")
            self.lbl_status.config(text="SYSTEM: REWIND\nVOLTAGE: NOMINAL", fg="#00BFFF")

        elif self.engine.is_playing:
            actual_ips = self.controls['ips_base'].get() * self.engine.transport.last_instant_speed
            dir_arrow  = "◀" if self.engine.is_reversed else "▶"
            self.lbl_ips.config(text=f"{dir_arrow} {actual_ips:05.2f}")
            reel_fg = "#FF8C00" if self.engine.is_reversed else "#FFB000"
            self.lbl_reel.config(text=f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}", fg=reel_fg)
            if abs(self.engine.transport.last_instant_speed - 1.0) > 0.1:
                self.lbl_status.config(text="SYSTEM: CRITICAL\nVOLTAGE: UNSTABLE", fg="#FF5555")
            elif self.engine.is_reversed:
                self.lbl_status.config(text="SYSTEM: REVERSE\nVOLTAGE: NOMINAL", fg="#FF8C00")
            else:
                self.lbl_status.config(text="SYSTEM: RUNNING\nVOLTAGE: NOMINAL", fg="#FFB000")

        else:
            self.lbl_ips.config(text="IPS: 00.000")
            self.lbl_reel.config(text=f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}", fg="#555")
            self.lbl_status.config(text="SYSTEM: IDLE\nVOLTAGE: STABLE", fg="#444")

        self.root.after(40, self.update_telemetry)

    # ------------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------------
    def _stop_all(self):
        """Stop any active playback or rewind without resetting the head position."""
        self.engine.is_playing = False
        self.is_rewinding      = False
        self.btn_play.config(text="▶ PLAY", fg="#FFB000", state="normal")
        self.btn_rev_play.config(text="◀ PLAY REVERSE", fg="#FF8C00", state="normal")
        self.btn_rewind.config(text="◀◀ REWIND", fg="#00BFFF")

    def start_forward(self):
        """Start forward playback from the beginning."""
        if self.engine.audio_data is None:
            return
        self._stop_all()
        # Ensure audio is in forward orientation and head is at start
        self.engine.set_reverse(False)
        self.engine.is_playing = True
        self.btn_play.config(text="▶ PLAYING", fg="#FF5555")
        threading.Thread(target=self._audio_thread, daemon=True).start()

    def start_reverse(self):
        """Start reverse playback from the end of the tape."""
        if self.engine.audio_data is None:
            return
        self._stop_all()
        # Flip audio in place and reset DSP state — play_head starts at 0 = end of tape
        self.engine.set_reverse(True)
        self.engine.is_playing = True
        self.btn_rev_play.config(text="◀ REVERSING", fg="#FF5555")
        threading.Thread(target=self._audio_thread, daemon=True).start()

    def stop_transport(self):
        self._stop_all()
        # Restore forward orientation if we were reversed
        if self.engine.is_reversed:
            threading.Thread(target=lambda: self.engine.set_reverse(False), daemon=True).start()

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            self._stop_all()
            threading.Thread(target=self.engine.load_file, args=(path,), daemon=True).start()

    # ------------------------------------------------------------------
    def _audio_thread(self):
        """Single audio thread used by both forward and reverse playback."""
        def cb(out, frames, time_info, status):
            d = self.engine.dsp_process(frames)
            if d is None:
                raise sd.CallbackStop
            out[:] = d

        with sd.OutputStream(samplerate=44100, channels=2, callback=cb, blocksize=1024):
            while self.engine.is_playing:
                sd.sleep(50)

        self.engine.is_playing = False
        # Restore forward orientation when playback ends
        if self.engine.is_reversed:
            self.engine.set_reverse(False)
        self.root.after(0, lambda: [
            self.btn_play.config(text="▶ PLAY", fg="#FFB000"),
            self.btn_rev_play.config(text="◀ PLAY REVERSE", fg="#FF8C00"),
        ])

    # ------------------------------------------------------------------
    def toggle_rewind(self):
        if self.engine.audio_data is None:
            return
        if self.is_rewinding:
            self.is_rewinding = False
            self.btn_rewind.config(text="◀◀ REWIND", fg="#00BFFF")
        else:
            self._stop_all()
            self.is_rewinding = True
            self.btn_rewind.config(text="■ STOP RWD", fg="#FF5555")
            threading.Thread(target=self._rewind_thread, daemon=True).start()

    def _rewind_thread(self):
        SR         = 44100
        block      = 512
        base_ips   = self.controls['ips_base'].get()
        squeal_phase = 0.0

        def cb(out, frames, time_info, status):
            nonlocal squeal_phase
            if not self.is_rewinding:
                raise sd.CallbackStop

            with self.engine.lock:
                pos   = self.engine.play_head
                total = self.engine.total_samples

            if pos <= 0:
                with self.engine.lock:
                    self.engine.play_head    = 0.0
                    self.engine.current_time = 0.0
                self.is_rewinding = False
                self.root.after(0, lambda: self.btn_rewind.config(text="◀◀ REWIND", fg="#00BFFF"))
                raise sd.CallbackStop

            step    = (base_ips / 15.0) * SR * 40.0 * frames / SR
            new_pos = max(0.0, pos - step)
            with self.engine.lock:
                self.engine.play_head    = new_pos
                self.engine.current_time = new_pos / SR

            progress    = float(new_pos / max(total, 1))
            reel_factor = 1.0 + (1.0 - progress) * 1.5
            squeal_hz   = float(np.clip(1800.0 * (base_ips / 15.0) * reel_factor, 200, 18000))
            phase_inc   = 2 * np.pi * squeal_hz / SR
            phases      = squeal_phase + np.arange(frames) * phase_inc
            squeal_sig  = (np.sin(phases) * 0.50 + np.sin(phases * 3) * 0.15
                           + np.sin(phases * 5) * 0.07 + np.sin(phases * 7) * 0.03)
            squeal_phase = (squeal_phase + frames * phase_inc) % (2 * np.pi)
            amp         = 0.18 * np.sin(np.pi * progress) + 0.04
            out[:]      = (squeal_sig * amp)[:, None] * np.array([[1.0, 0.92]])

        with sd.OutputStream(samplerate=SR, channels=2, callback=cb,
                             blocksize=block, dtype='float32'):
            while self.is_rewinding:
                sd.sleep(50)

        self.is_rewinding = False
        self.root.after(0, lambda: self.btn_rewind.config(text="◀◀ REWIND", fg="#00BFFF"))

    # ------------------------------------------------------------------
    def export_audio(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav")
        if not path:
            return
        self.btn_export.config(text="RENDERING...", state="disabled")

        def _run():
            with self.engine.lock:
                old_h, old_t = self.engine.play_head, self.engine.current_time
                self.engine.reset_state()
            chunks = []
            while True:
                c = self.engine.dsp_process(1024)
                if c is None:
                    break
                chunks.append(c)
            out = (np.vstack(chunks) * 32767).astype(np.int16)
            AudioSegment(out.tobytes(), frame_rate=44100, sample_width=2,
                         channels=2).export(path, format="wav")
            with self.engine.lock:
                self.engine.play_head    = old_h
                self.engine.current_time = old_t
            self.root.after(0, lambda: [
                messagebox.showinfo("Export", "Success"),
                self.btn_export.config(text="FORENSIC RENDER", state="normal"),
            ])

        threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TNotebook", background="#0D0D0E", borderwidth=0)
    style.configure("TNotebook.Tab", background="#222", foreground="#888", padding=[10, 5])
    style.map("TNotebook.Tab",
              background=[("selected", "#181819")],
              foreground=[("selected", "#FFB000")])
    app = ForensicTapeStudio(root)
    root.mainloop()
