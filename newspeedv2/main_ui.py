import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import logging
import sys
import time

from engine import TapeEngine

# ── Logging setup ─────────────────────────────────────────────────────────────
class ColorFormatter(logging.Formatter):
    ANSI = {
        logging.DEBUG:    "\033[38;5;240m",   # dark grey
        logging.INFO:     "\033[38;5;75m",    # steel blue
        logging.WARNING:  "\033[38;5;214m",   # amber
        logging.ERROR:    "\033[38;5;196m",   # red
        logging.CRITICAL: "\033[38;5;201m",   # magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record):
        color = self.ANSI.get(record.levelno, "")
        ts    = self.formatTime(record, "%H:%M:%S")
        name  = f"\033[38;5;245m[{record.name}]{self.RESET}"
        level = f"{color}{self.BOLD}{record.levelname:<8}{self.RESET}"
        msg   = f"{color}{record.getMessage()}{self.RESET}"
        return f"\033[38;5;238m{ts}{self.RESET}  {level}  {name}  {msg}"

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler])

log_ui      = logging.getLogger("UI")
log_engine  = logging.getLogger("ENGINE")
log_render  = logging.getLogger("RENDER")
log_transport = logging.getLogger("TRANSPORT")

# ── Tooltip data — one entry per control key ──────────────────────────────────
TOOLTIPS = {
    # Transport Mechanics
    "ips_base": (
        "CAPSTAN SPEED (IPS)",
        "Sets how fast the tape moves past the heads, measured in inches per second. "
        "The capstan is the precision rotating shaft that drives the tape. Higher IPS "
        "means wider frequency response and lower noise — professional studio machines "
        "ran at 15 or 30 IPS. Consumer cassettes ran at 1.875 IPS, causing their "
        "characteristic muffled, wobbly sound."
    ),
    "motor_health": (
        "VOLTAGE DRIFT / JITTER",
        "Replicates an ageing or failing motor power supply. Real tape machines suffer "
        "from capacitor wear and transformer hum that causes the motor voltage to wander "
        "unpredictably, making the tape speed drift in slow, irregular surges. High "
        "values produce the lurching, seasick instability of a dying machine."
    ),
    "motor_drag": (
        "TORQUE LOAD",
        "Simulates mechanical resistance on the capstan and reel motors — friction from "
        "dried lubrication, a tight brake, or a heavy full reel fighting the motor. "
        "Reduces average speed and makes the tape run slightly slow. Combined with Motor "
        "Boost, the competing forces create stick-slip instability and mechanical lurching."
    ),
    "motor_boost": (
        "MOTOR BOOST",
        "Pushes the capstan motor harder than nominal, running the tape slightly fast. "
        "Used to model an over-driven motor or a machine set up for a different standard. "
        "When both Boost and Torque Load are active simultaneously, the conflicting forces "
        "create mechanical instability — speed oscillation, sudden lurches, and tape "
        "tension fighting, exactly like a machine under competing mechanical stresses."
    ),
    "wow_dep": (
        "WOW INTENSITY",
        "Wow is very slow speed variation (0.5–4 Hz) caused by eccentricity in the "
        "capstan or the reel hubs — a slightly off-centre roller or a warped reel causes "
        "the tape to speed up and slow down once per rotation. You hear it as a slow, "
        "wavering pitch that makes sustained notes sound like they are bending. "
        "Particularly audible on cassettes with worn transport mechanisms."
    ),
    "flutter_dep": (
        "FLUTTER INTENSITY",
        "Flutter is faster speed variation (10–100 Hz) caused by mechanical resonances "
        "in the tape path — a worn pinch roller, guide post vibration, or tape tension "
        "pulses from the reel motors. You hear it as a rapid, chorus-like shimmer on "
        "sustained tones. Severe flutter creates the characteristic warbling of a "
        "degraded cassette or a machine with a worn rubber pinch roller."
    ),
    "scrape_flutter": (
        "SCRAPE FLUTTER",
        "A very rapid (2–5 kHz) modulation caused by the tape sticking and releasing "
        "microscopically as it scrapes across the stationary erase or guide heads. "
        "Unlike rotational flutter, scrape flutter is highly irregular and incoherent. "
        "It adds a subtle roughness and grain to high-frequency content, particularly "
        "noticeable on transients like cymbals and sibilant vocals."
    ),
    "tension_load": (
        "REEL TENSION DYNAMICS",
        "Models the varying mechanical tension in the tape as it moves between the "
        "supply and takeup reels. Real tape machines use tension arms and servo systems "
        "to regulate this, but imperfect tension causes subtle speed variations that "
        "track the tape position — the tape runs slightly different at the start of a "
        "reel (full, heavy supply reel) versus the end."
    ),
    "dropout_rate": (
        "OXIDE DROPOUT RATE",
        "Oxide dropouts occur when a spot on the tape physically lacks magnetic coating "
        "— a scratch, a dust particle, or shed oxide clumping. The signal briefly "
        "disappears for a few milliseconds. On damaged archive tapes you hear these as "
        "rhythmic clicks, gaps, or crackles. Higher values increase both the frequency "
        "and depth of these dropout events, from occasional glitches to severe data loss."
    ),
    # Magnetic Flux
    "drive": (
        "HEAD SATURATION",
        "Controls how hard the signal is driven into the magnetic recording layer. "
        "At low levels the tape responds linearly; as you push harder, the oxide "
        "particles reach their maximum magnetisation (saturation) and the waveform "
        "is gently compressed and harmonically distorted. This adds the characteristic "
        "warmth and second-harmonic colour of analogue tape. Very high values produce "
        "the thick, fuzzy distortion of an overdriven machine."
    ),
    "bias": (
        "AC BIAS TUNING",
        "During recording, a high-frequency bias signal (typically 50–100 kHz) is "
        "mixed with the audio to linearise the magnetic recording process. Correct bias "
        "gives flat frequency response; under-bias increases high-frequency response but "
        "adds odd-harmonic distortion; over-bias rolls off the highs but reduces "
        "distortion. This simulates a machine set up for the wrong tape formulation, "
        "or a bias adjustment that has drifted over time."
    ),
    "replay_diff": (
        "REPLAY HEAD DIFFERENTIATION",
        "The replay head is an electromagnetic transducer that reads the rate of change "
        "of the magnetic flux, not the flux itself. This differentiation adds 6 dB per "
        "octave of high-frequency emphasis to the replayed signal. Higher values increase "
        "this effect, brightening high frequencies and adding a slight edge or air to the "
        "sound — the replay EQ curves on professional machines were designed to compensate "
        "for exactly this phenomenon."
    ),
    "asperities": (
        "ASPERITIES (MODULATION NOISE)",
        "The magnetic coating is not perfectly smooth — microscopic surface irregularities "
        "called asperities modulate the signal as the tape moves past the head. This "
        "creates a low-level noise that is amplitude-correlated with the signal: louder "
        "passages have more noise, quiet passages less. It sounds like a subtle, "
        "signal-dependent graininess rather than a constant hiss."
    ),
    "barkhausen": (
        "BARKHAUSEN GRAIN NOISE",
        "Named after physicist Heinrich Barkhausen, this noise arises from the discrete, "
        "quantised nature of magnetic domain switching in the oxide particles. As the "
        "field changes, domains flip in sudden jumps rather than continuously, generating "
        "tiny random voltage pulses. The result is a fine, grainy texture on transients "
        "and fast-changing signal content — the microscopic sound of magnetism itself."
    ),
    "crosstalk": (
        "STEREO CROSSTALK",
        "On multi-track tape, adjacent tracks are physically close together. Imperfect "
        "head alignment or magnetic fringing lets some of the left channel bleed into the "
        "right and vice versa. High crosstalk narrows the stereo image and causes "
        "low-level ghosting of the opposite channel — common in worn cassette heads "
        "where the gap azimuth has drifted or the head has partially demagnetised."
    ),
    "print_through": (
        "PRINT-THROUGH (GHOST ECHO)",
        "On a wound reel, adjacent tape layers are in close magnetic contact for days or "
        "years. Strong signals slowly magnetise neighbouring layers, leaving a faint copy "
        "of the recording at a fixed time offset (typically 1.5 seconds, the reel "
        "circumference). You hear it as a ghostly pre-echo before loud sounds or a "
        "post-echo after them — famously audible at the start of many analogue recordings "
        "before the first downbeat."
    ),
    "demagnetization": (
        "DEMAGNETISATION (HF LOSS)",
        "Over time, stored tapes partially demagnetise, particularly at high frequencies. "
        "The small, rapidly-alternating domains that encode high-frequency content are "
        "less stable than the large, slow-changing domains of bass content. The result is "
        "a progressive high-frequency rolloff — old tapes sound progressively darker and "
        "more muffled. Playback heads also demagnetise gradually from use."
    ),
    "oxide_shedding": (
        "OXIDE SHEDDING TEXTURE",
        "As magnetic tape ages, the binder holding the oxide particles to the backing "
        "deteriorates. Particles literally flake off during playback, clogging the heads "
        "and causing intermittent level drops, gritty texture, and irregular dropouts. "
        "This is a physically destructive process on real tapes — baking the tape in an "
        "oven temporarily reverses it. Here it adds a degraded, crumbling character "
        "to the signal."
    ),
    # Electronics & Wear
    "hiss": (
        "NOISE FLOOR",
        "Sets the level of broadband white noise generated by the playback electronics — "
        "the head preamplifier transistors, the playback head gap, and the oxide particle "
        "noise. This is the characteristic 'ssss' of tape that sits under the music. "
        "Consumer cassettes at 1.875 IPS have much higher noise floors than 30 IPS "
        "studio machines because the slower tape speed means worse signal-to-noise ratio."
    ),
    "hiss_color": (
        "HISS COLOUR (PINK TILT)",
        "Real tape hiss is not purely white (equal energy at all frequencies) — it has "
        "more low-mid content due to the 1/f noise characteristics of the transistor "
        "preamplifiers and the magnetic properties of the oxide. This control tilts the "
        "noise spectrum from white toward pink (equal energy per octave) or even brown, "
        "replicating the warmer, fuller noise character of vintage electronics."
    ),
    "mains_hum": (
        "60 Hz MAINS HUM",
        "Electromagnetic interference from the AC power supply induces a 60 Hz tone "
        "(50 Hz in Europe) into the audio chain via the transformer, motor windings, and "
        "ground loops. You also hear the third harmonic at 180 Hz. Poorly shielded "
        "machines, damaged ground connections, or motors placed too close to the audio "
        "path all contribute to this distinctive electrical hum."
    ),
    "cutoff_base": (
        "AZIMUTH CUTOFF",
        "The replay head has a finite gap width, which causes a progressive high-frequency "
        "rolloff — frequencies whose wavelength approaches the gap width are attenuated. "
        "This sets the effective bandwidth ceiling. Reduce it to simulate a worn head "
        "with a widened gap, a misaligned head, or a slower tape speed where the "
        "recorded wavelengths are physically shorter."
    ),
    "head_bump": (
        "HEAD BUMP (LF EQ)",
        "Professional tape machines exhibit a low-frequency resonance peak (the 'head "
        "bump') caused by the physical dimensions of the replay head gap interacting with "
        "the tape. It typically sits between 50–200 Hz depending on tape speed, adding a "
        "characteristic warmth and weight to bass content. Studio engineers often relied "
        "on this bump to make kick drums and bass guitars sound fuller."
    ),
    "azimuth_drift": (
        "AZIMUTH PHASE DRIFT",
        "Azimuth is the angle of the head gap relative to the tape travel direction. "
        "If the replay head is not perfectly perpendicular, high frequencies in the "
        "left channel arrive at a slightly different time than the right — causing "
        "phase differences that comb-filter the stereo image and reduce high-frequency "
        "content. This is a common symptom of a head that needs alignment."
    ),
    "sticky_shed": (
        "STICKY SHED SYNDROME",
        "A tape disease affecting recordings from the 1970s–80s where the polyurethane "
        "binder absorbs moisture and becomes sticky. As the tape plays, it squeals, "
        "slows, and sheds oxide onto the heads. The standard restoration technique is "
        "baking the tape at low heat to drive off the moisture. This simulates the "
        "progressive degradation — high-frequency loss, speed instability, and the "
        "characteristic gummy, sluggish transport sound."
    ),
}


class Tooltip:
    """
    Shows a dark tooltip window after the pointer has rested on a widget
    for HOVER_DELAY milliseconds. A single instance can be shared across
    multiple child widgets — only one popup ever appears.
    """
    HOVER_DELAY = 900   # ms before tooltip appears
    MAX_WIDTH   = 420   # px — tooltip wraps beyond this

    def __init__(self, widget, title, body):
        self._widget  = widget   # used for after() scheduling and geometry
        self._title   = title
        self._body    = body
        self._id      = None   # pending after() id
        self._tip_win = None   # active Toplevel

        widget.bind("<Enter>",       self._on_enter, add="+")
        widget.bind("<Leave>",       self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event):
        # Use the event widget for geometry so the tip appears near the cursor
        self._last_widget = event.widget
        self._cancel()
        self._id = event.widget.after(self.HOVER_DELAY, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._id is not None:
            try:
                self._widget.after_cancel(self._id)
            except Exception:
                pass
            self._id = None

    def _show(self):
        if self._tip_win:
            return
        ref = getattr(self, '_last_widget', self._widget)
        try:
            x = ref.winfo_rootx() + 24
            y = ref.winfo_rooty() + ref.winfo_height() + 4
        except Exception:
            return

        self._tip_win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        tw.configure(bg=C["border"])

        outer = tk.Frame(tw, bg=C["border"], padx=1, pady=1)
        outer.pack()

        inner = tk.Frame(outer, bg=C["bg4"])
        inner.pack()

        tk.Label(inner, text=self._title,
                 bg=C["bg4"], fg=C["amber"],
                 font=("Consolas", 9, "bold"),
                 anchor="w", padx=12, pady=6).pack(fill="x")

        tk.Frame(inner, bg=C["amber_dim"], height=1).pack(fill="x", padx=12)

        tk.Label(inner, text=self._body,
                 bg=C["bg4"], fg=C["grey_lt"],
                 font=("Consolas", 8),
                 justify="left", anchor="w",
                 wraplength=self.MAX_WIDTH,
                 padx=12, pady=8).pack(fill="x")

        tw.update_idletasks()
        screen_w = tw.winfo_screenwidth()
        tip_w    = tw.winfo_width()
        if x + tip_w > screen_w - 16:
            x = screen_w - tip_w - 16
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self):
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None


# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "bg":          "#0A0A0B",
    "bg2":         "#111114",
    "bg3":         "#18181C",
    "bg4":         "#1E1E24",
    "border":      "#2A2A35",
    "amber":       "#FFB000",
    "amber_dim":   "#7A5500",
    "cyan":        "#00D4FF",
    "cyan_dim":    "#005566",
    "green":       "#39FF14",
    "green_dim":   "#1A5C09",
    "red":         "#FF3333",
    "red_dim":     "#5C1111",
    "orange":      "#FF6B00",
    "orange_dim":  "#5C2800",
    "purple":      "#B44FFF",
    "purple_dim":  "#3D1A66",
    "grey":        "#444455",
    "grey_lt":     "#888899",
    "white":       "#E8E8F0",
    "tab_bg":      "#141418",
    "tab_sel":     "#1E1E28",
    "slider_trough":"#050508",
}

# Tab accent colours — each tab gets its own identity
TAB_COLORS = {
    "Transport Mechanics": C["amber"],
    "Magnetic Flux":       C["cyan"],
    "Electronics & Wear":  C["purple"],
}


class ForensicTapeStudio:
    def __init__(self, root):
        log_ui.info("Initialising Forensic Tape Studio")
        self.root = root
        self.root.title("NAGRA-V  ·  ANALOG FORENSICS  ·  MODULAR ARCHITECTURE")
        self.root.geometry("1160x1080")
        self.root.configure(bg=C["bg"])

        self.engine       = TapeEngine()
        self.controls     = {}
        self.is_rewinding = False
        self.is_ffing     = False

        # Render options
        self.render_sample_rate  = tk.IntVar(value=44100)
        self.render_bit_depth    = tk.IntVar(value=24)
        self.render_sim_quality  = tk.StringVar(value="High")
        self.render_oversample   = tk.IntVar(value=1)
        self.render_dither       = tk.BooleanVar(value=True)
        self.render_dc_block     = tk.BooleanVar(value=True)
        self.render_normalize    = tk.BooleanVar(value=False)
        self.render_output_path  = tk.StringVar(value="")

        self._setup_styles()
        self.setup_gui()
        log_ui.info("Applying default preset: Master Studio Reel (30ips)")
        self.apply_builtin_preset("Master Studio Reel (30ips)")
        self.update_telemetry()
        log_ui.info("GUI ready")

    # ── ttk styles ─────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook",
                    background=C["bg"], borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab",
                    background=C["bg2"], foreground=C["grey_lt"],
                    padding=[14, 6], font=("Consolas", 9, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", C["tab_sel"])],
              foreground=[("selected", C["amber"])])
        s.configure("TCombobox",
                    fieldbackground=C["bg3"], background=C["bg3"],
                    foreground=C["white"], arrowcolor=C["amber"],
                    selectbackground=C["bg4"], selectforeground=C["amber"])
        s.map("TCombobox",
              fieldbackground=[("readonly", C["bg3"])],
              foreground=[("readonly", C["white"])])

    # ── main GUI ────────────────────────────────────────────────────────────────
    def setup_gui(self):
        log_ui.debug("Building GUI layout")

        # ── HEADER ──
        header = tk.Frame(self.root, bg=C["bg"], bd=0)
        header.pack(fill="x", side="top", padx=12, pady=(10, 4))

        # Left: rich status panel
        status_block = tk.Frame(header, bg="#000",
                                highlightbackground=C["border"], highlightthickness=1)
        status_block.pack(side="left", padx=(0, 12), pady=2)

        def _sl(text, fg=C["grey"], font=("Consolas", 8), **kw):
            """Helper: add a status line to the panel."""
            lbl = tk.Label(status_block, text=text, bg="#000", fg=fg,
                           font=font, justify="left", anchor="w",
                           padx=10, pady=1, **kw)
            lbl.pack(fill="x")
            return lbl

        # Section header
        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x")
        _hdr = tk.Label(status_block, text=" MACHINE STATUS ",
                        bg=C["bg4"], fg=C["amber"], font=("Consolas", 7, "bold"),
                        anchor="w", padx=10, pady=2)
        _hdr.pack(fill="x")
        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x")

        tk.Frame(status_block, bg="#000", height=3).pack()

        self.lbl_sys_state  = _sl("STATE    : IDLE",      fg=C["grey"])
        self.lbl_sys_speed  = _sl("SPEED    : 00.000 IPS", fg=C["grey"])
        self.lbl_sys_dev    = _sl("DEVIATION: ±0.000%",   fg=C["grey"])

        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x", padx=8, pady=2)

        self.lbl_volt_bar   = _sl("VOLT  [                    ]", fg=C["grey"],
                                  font=("Consolas", 7))
        self.lbl_volt_val   = _sl("PWR SUPPLY: STABLE  0.00V", fg=C["grey"])

        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x", padx=8, pady=2)

        self.lbl_drop_count = _sl("DROPOUTS : 0  LAST: --.-s", fg=C["grey"])
        self.lbl_head_temp  = _sl("HEAD WEAR: NOMINAL",        fg=C["grey"])

        tk.Frame(status_block, bg="#000", height=3).pack()
        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x")

        # Centre: title
        title_frame = tk.Frame(header, bg=C["bg"])
        title_frame.pack(side="left", expand=True)
        tk.Label(title_frame, text="NAGRA-V",
                 bg=C["bg"], fg=C["amber"], font=("Consolas", 22, "bold")).pack()
        tk.Label(title_frame, text="ANALOG FORENSICS  ·  MODULAR DSP",
                 bg=C["bg"], fg=C["grey"], font=("Consolas", 9)).pack()

        # Right: IPS counter + reel
        meter_block = tk.Frame(header, bg="#000", bd=0,
                               highlightbackground=C["border"], highlightthickness=1)
        meter_block.pack(side="right", padx=(12, 0), pady=2)

        self.lbl_ips = tk.Label(meter_block, text="▶ 00.000",
                                bg="#000", fg=C["amber"],
                                font=("Consolas", 30, "bold"), width=11, padx=10, pady=4)
        self.lbl_ips.pack()

        self.lbl_reel = tk.Label(meter_block,
                                 text="SUPPLY: ●●●●●●●●  TAKEUP: ○○○○○○○○",
                                 bg="#000", fg=C["grey"], font=("Consolas", 8), pady=4)
        self.lbl_reel.pack()

        # Thin separator
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=12)

        # ── TOP BAR ──
        top_bar = tk.Frame(self.root, bg=C["bg2"])
        top_bar.pack(fill="x", padx=12, pady=(6, 4))

        self._btn(top_bar, "LOAD TAPE", self.load_file,
                  bg=C["green_dim"], fg=C["green"]).pack(side="left", padx=(0, 8))

        tk.Label(top_bar, text="PRESET:", bg=C["bg2"],
                 fg=C["grey_lt"], font=("Consolas", 9)).pack(side="left")
        self.preset_var = tk.StringVar()
        presets = [
            "Master Studio Reel (30ips)", "Standard Tape (15ips)",
            "Consumer Cassette (1.875ips)", "Chrome Cassette (1.875ips)",
            "Metal Tape (1.875ips)", "VHS Linear Audio",
            "Sticky Shed Disaster", "Abandoned Attic Find", "Demagnetised Archive",
        ]
        cb = ttk.Combobox(top_bar, textvariable=self.preset_var, values=presets,
                          state="readonly", width=28)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>",
                lambda e: self.apply_builtin_preset(self.preset_var.get()))

        tk.Label(top_bar, text="OXIDE:", bg=C["bg2"],
                 fg=C["grey_lt"], font=("Consolas", 9), padx=8).pack(side="left")
        self.oxide_var = tk.StringVar(value="Fe2O3")
        oxide_cb = ttk.Combobox(top_bar, textvariable=self.oxide_var,
                                values=["Fe2O3", "CrO2", "Metal", "FeCo"],
                                state="readonly", width=8)
        oxide_cb.pack(side="left", padx=2)
        oxide_cb.bind("<<ComboboxSelected>>", lambda e: self.sync())

        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=12)

        # ── TABS ──
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=8)

        # Transport Mechanics
        t_mech = tk.Frame(self.tabs, bg=C["tab_bg"])
        self.tabs.add(t_mech, text="  Transport Mechanics  ")
        ac = TAB_COLORS["Transport Mechanics"]
        tk.Frame(t_mech, bg=ac, height=2).pack(fill="x")
        self.ctrl("ips_base",       t_mech, "CAPSTAN SPEED (IPS)",          0.5,  30.0,  15.0, ac)
        self.ctrl("motor_health",   t_mech, "VOLTAGE DRIFT (JITTER)",       0.0,  10.0,   0.5, ac)
        self.ctrl("motor_drag",     t_mech, "TORQUE LOAD",                  0.0,   0.9,   0.0, ac)
        self.ctrl("motor_boost",    t_mech, "MOTOR BOOST (SPEED UP)",       0.0,   0.9,   0.0, ac)
        self.ctrl("wow_dep",        t_mech, "WOW INTENSITY",                0.0,  30.0,   0.2, ac)
        self.ctrl("flutter_dep",    t_mech, "FLUTTER INTENSITY",            0.0,  10.0,  0.05, ac)
        self.ctrl("scrape_flutter", t_mech, "SCRAPE FLUTTER (3kHz)",        0.0,   1.0,   0.1, ac)
        self.ctrl("tension_load",   t_mech, "REEL TENSION DYNAMICS",        0.0,   0.5,  0.05, ac)
        self.ctrl("dropout_rate",   t_mech, "OXIDE DROPOUT RATE",           0.0,   1.0,   0.0, ac)

        # Magnetic Flux
        t_mag = tk.Frame(self.tabs, bg=C["tab_bg"])
        self.tabs.add(t_mag, text="  Magnetic Flux  ")
        ac = TAB_COLORS["Magnetic Flux"]
        tk.Frame(t_mag, bg=ac, height=2).pack(fill="x")
        self.ctrl("drive",           t_mag, "HEAD SATURATION",              1.0,  20.0,   1.2, ac)
        self.ctrl("bias",            t_mag, "AC BIAS TUNING",               0.5,   3.0,   1.0, ac)
        self.ctrl("replay_diff",     t_mag, "REPLAY HEAD DIFFERENTIATION",  0.0,   1.0,   0.3, ac)
        self.ctrl("asperities",      t_mag, "ASPERITIES (MOD NOISE)",       0.0,   0.5,  0.05, ac)
        self.ctrl("barkhausen",      t_mag, "BARKHAUSEN GRAIN",             0.0,   0.1,  0.01, ac)
        self.ctrl("crosstalk",       t_mag, "STEREO CROSSTALK",             0.0,   0.5,  0.02, ac)
        self.ctrl("print_through",   t_mag, "PRINT-THROUGH (GHOST)",        0.0,   0.1,   0.0, ac)
        self.ctrl("demagnetization", t_mag, "DEMAGNETISATION (HF LOSS)",    0.0,  0.99,   0.0, ac)
        self.ctrl("oxide_shedding",  t_mag, "OXIDE SHEDDING TEXTURE",       0.0,   1.0,   0.0, ac)

        # Electronics & Wear
        t_elec = tk.Frame(self.tabs, bg=C["tab_bg"])
        self.tabs.add(t_elec, text="  Electronics & Wear  ")
        ac = TAB_COLORS["Electronics & Wear"]
        tk.Frame(t_elec, bg=ac, height=2).pack(fill="x")
        self.ctrl("hiss",          t_elec, "NOISE FLOOR",                   0.0, 0.02,  0.001, ac)
        self.ctrl("hiss_color",    t_elec, "HISS COLOUR (PINK TILT)",       0.0,  1.0,   0.0, ac)
        self.ctrl("mains_hum",     t_elec, "60Hz MAINS HUM",                0.0, 0.05,  0.001, ac)
        self.ctrl("cutoff_base",   t_elec, "AZIMUTH CUTOFF (Hz)",           500, 22000, 18000, ac)
        self.ctrl("head_bump",     t_elec, "HEAD BUMP (LF EQ)",             0.0,  5.0,   0.5, ac)
        self.ctrl("azimuth_drift", t_elec, "AZIMUTH PHASE DRIFT",           0.0,  1.0,  0.05, ac)
        self.ctrl("sticky_shed",   t_elec, "STICKY SHED INTENSITY",         0.0,  1.0,   0.0, ac)

        # ── FOOTER ──
        footer = tk.Frame(self.root, bg=C["bg2"],
                          highlightbackground=C["border"], highlightthickness=1)
        footer.pack(fill="x", side="bottom")

        tk.Frame(footer, bg=C["border"], height=1).pack(fill="x")

        btn_row = tk.Frame(footer, bg=C["bg2"])
        btn_row.pack(fill="x", padx=16, pady=(12, 10))

        self.btn_play = self._btn(btn_row, "▶  PLAY",     self.start_forward,
                                  bg=C["amber_dim"], fg=C["amber"], width=10)
        self.btn_play.pack(side="left", padx=3)

        self.btn_rev_play = self._btn(btn_row, "◀  REVERSE", self.start_reverse,
                                      bg=C["orange_dim"], fg=C["orange"], width=12)
        self.btn_rev_play.pack(side="left", padx=3)

        self.btn_stop = self._btn(btn_row, "■  STOP",    self.stop_transport,
                                  bg=C["red_dim"], fg=C["red"], width=9)
        self.btn_stop.pack(side="left", padx=3)

        self.btn_rewind = self._btn(btn_row, "◀◀  REWIND", self.toggle_rewind,
                                    bg=C["cyan_dim"], fg=C["cyan"], width=11)
        self.btn_rewind.pack(side="left", padx=3)

        self.btn_ff = self._btn(btn_row, "FF  ▶▶", self.toggle_ff,
                                bg=C["green_dim"], fg=C["green"], width=11)
        self.btn_ff.pack(side="left", padx=3)

        # Render button — right side
        self.btn_export = self._btn(btn_row, "⬛  FORENSIC RENDER",
                                    self.open_render_dialog,
                                    bg=C["purple_dim"], fg=C["purple"],
                                    font=("Consolas", 12, "bold"))
        self.btn_export.pack(side="right", padx=3)

        log_ui.debug("GUI layout complete")

    # ── Widget helpers ──────────────────────────────────────────────────────────
    def _btn(self, parent, text, cmd, bg=None, fg=None, width=None, font=None):
        kw = dict(text=text, command=cmd,
                  bg=bg or C["bg3"], fg=fg or C["white"],
                  activebackground=fg or C["white"], activeforeground=C["bg"],
                  font=font or ("Consolas", 12, "bold"),
                  relief="flat", bd=0, cursor="hand2",
                  padx=10, pady=6)
        if width:
            kw["width"] = width
        return tk.Button(parent, **kw)

    def ctrl(self, key, parent, label, mn, mx, df, accent=None):
        ac = accent or C["amber"]
        row = tk.Frame(parent, bg=C["tab_bg"], pady=0)
        row.pack(fill="x", padx=0)

        # Coloured left accent strip
        tk.Frame(row, bg=ac, width=3).pack(side="left", fill="y")

        inner = tk.Frame(row, bg=C["bg3"], pady=6)
        inner.pack(fill="x", side="left", expand=True, padx=(0, 0))

        lbl = tk.Label(inner, text=label, bg=C["bg3"], fg=C["grey_lt"],
                      font=("Consolas", 9), width=32, anchor="w",
                      padx=12)
        lbl.pack(side="left")

        v = tk.DoubleVar(value=df)

        # Value readout label
        val_lbl = tk.Label(inner, text=f"{df:.3f}", bg=C["bg3"], fg=ac,
                           font=("Consolas", 9), width=8, anchor="e", padx=6)
        val_lbl.pack(side="right")

        def on_change(*_):
            val_lbl.config(text=f"{v.get():.4g}")
            self.sync()
            log_ui.debug(f"Param '{key}' → {v.get():.5g}")

        tk.Scale(inner, variable=v, from_=mn, to=mx, resolution=0.0001,
                 orient="horizontal", showvalue=False,
                 bg=C["bg3"], fg=ac,
                 highlightthickness=0, troughcolor=C["slider_trough"],
                 activebackground=ac, sliderrelief="flat",
                 sliderlength=14).pack(side="left", fill="x", expand=True, padx=6)

        v.trace_add("write", on_change)
        self.controls[key] = v

        # One Tooltip instance shared across all parts of the row.
        # Each child that gets an <Enter> event cancels the old timer and
        # starts a fresh one — only one popup ever appears.
        if key in TOOLTIPS:
            tip_title, tip_body = TOOLTIPS[key]
            tip = Tooltip(row, tip_title, tip_body)
            for w in (inner, lbl):
                w.bind("<Enter>",       tip._on_enter, add="+")
                w.bind("<Leave>",       tip._on_leave, add="+")
                w.bind("<ButtonPress>", tip._on_leave, add="+")
            for child in inner.winfo_children():
                child.bind("<Enter>",       tip._on_enter, add="+")
                child.bind("<Leave>",       tip._on_leave, add="+")
                child.bind("<ButtonPress>", tip._on_leave, add="+")

        # Thin separator line between rows
        tk.Frame(row, bg=C["border"], height=1).pack(fill="x", side="bottom")

    def sync(self):
        for k, v in self.controls.items():
            self.engine.params[k] = v.get()
        self.engine.params["oxide_type"] = self.oxide_var.get()

    # ── Render options dialog ───────────────────────────────────────────────────
    def open_render_dialog(self):
        log_ui.info("Opening render options dialog")
        if self.engine.audio_data is None:
            messagebox.showwarning("No Tape", "Load a tape file first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Render Options")
        win.configure(bg=C["bg"])
        win.geometry("720x700")
        win.resizable(True, True)
        win.minsize(640, 560)
        win.grab_set()

        # ── Fixed bottom button bar (packed first so it doesn't get squashed) ──
        tk.Frame(win, bg=C["border"], height=1).pack(fill="x", side="bottom")
        btn_row = tk.Frame(win, bg=C["bg2"], pady=10)
        btn_row.pack(fill="x", side="bottom")

        def do_render():
            path = self.render_output_path.get().strip()
            if not path:
                path = filedialog.asksaveasfilename(defaultextension=".wav",
                    filetypes=[("WAV", "*.wav"), ("All files", "*.*")])
                if not path:
                    return
                self.render_output_path.set(path)
            win.destroy()
            self._run_render(path, self.render_sample_rate.get(),
                             self.render_bit_depth.get(),
                             self.render_sim_quality.get())

        self._btn(btn_row, "CANCEL", win.destroy,
                  bg=C["bg3"], fg=C["grey_lt"], font=("Consolas", 11)).pack(side="left", padx=16)
        self._btn(btn_row, "⬛  RENDER", do_render,
                  bg=C["purple_dim"], fg=C["purple"],
                  font=("Consolas", 12, "bold")).pack(side="right", padx=16)

        # ── Scrollable content area ────────────────────────────────────────────
        canvas = tk.Canvas(win, bg=C["bg"], highlightthickness=0)
        vsb    = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        sf = tk.Frame(canvas, bg=C["bg"])
        sf_id = canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(sf_id, width=e.width))
        sf.bind("<Configure>",     lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120 or e.num == 4 and -1 or 1)), "units")
        win.bind("<MouseWheel>", _wheel)
        win.bind("<Button-4>",   _wheel)
        win.bind("<Button-5>",   _wheel)

        # ── Helper: section header ─────────────────────────────────────────────
        def section(title):
            tk.Frame(sf, bg=C["border"], height=1).pack(fill="x", padx=0)
            h = tk.Frame(sf, bg=C["bg3"])
            h.pack(fill="x")
            tk.Label(h, text=f"  {title}", bg=C["bg3"], fg=C["amber"],
                     font=("Consolas", 9, "bold"), pady=5, anchor="w").pack(fill="x")
            tk.Frame(sf, bg=C["border"], height=1).pack(fill="x")

        # ── Helper: radio group that wraps onto multiple lines ─────────────────
        def radio_group(var, options, cols=4):
            """options = [(value, label), ...]"""
            outer = tk.Frame(sf, bg=C["bg4"])
            outer.pack(fill="x", padx=16, pady=(4, 8))
            for i, (val, lbl) in enumerate(options):
                r = i // cols
                c = i % cols
                tk.Radiobutton(outer, text=lbl, variable=var, value=val,
                               bg=C["bg4"], fg=C["white"], font=("Consolas", 10),
                               selectcolor=C["amber"], activebackground=C["bg4"],
                               highlightthickness=0,
                               indicatoron=0,
                               relief="flat", bd=0,
                               padx=10, pady=6,
                               activeforeground=C["amber"],
                               ).grid(row=r, column=c, sticky="ew", padx=3, pady=2)
            for col in range(cols):
                outer.columnconfigure(col, weight=1)
            return outer

        # ── Helper: labelled row ───────────────────────────────────────────────
        def label_row(text):
            tk.Label(sf, text=text, bg=C["bg"], fg=C["grey"],
                     font=("Consolas", 8), anchor="w", padx=16, pady=3).pack(fill="x")

        # ── Helper: checkbox row ───────────────────────────────────────────────
        def check_row(var, label):
            f = tk.Frame(sf, bg=C["bg4"])
            f.pack(fill="x", padx=16, pady=2)
            tk.Checkbutton(f, text=label, variable=var,
                           bg=C["bg4"], fg=C["white"],
                           selectcolor=C["bg2"], activebackground=C["bg4"],
                           font=("Consolas", 10), highlightthickness=0,
                           padx=10, pady=6).pack(side="left")

        # ── Header ─────────────────────────────────────────────────────────────
        hdr = tk.Frame(sf, bg=C["bg2"], pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⬛  FORENSIC RENDER", bg=C["bg2"], fg=C["purple"],
                 font=("Consolas", 16, "bold"), padx=20, anchor="w").pack(fill="x")
        total_s = self.engine.total_samples / 44100
        tk.Label(hdr, text=f"  Source: {int(total_s//60):02d}:{total_s%60:05.2f}  ·  44100 Hz  ·  Stereo",
                 bg=C["bg2"], fg=C["grey"], font=("Consolas", 9), padx=20, anchor="w").pack(fill="x")

        # ── AUDIO QUALITY ──────────────────────────────────────────────────────
        section("AUDIO QUALITY")
        label_row("Sample Rate")
        radio_group(self.render_sample_rate, [
            (44100, "44.1 kHz"), (48000, "48 kHz"),
            (88200, "88.2 kHz"), (96000, "96 kHz"),
        ], cols=4)

        label_row("Bit Depth")
        radio_group(self.render_bit_depth, [
            (16, "16-bit PCM"), (24, "24-bit PCM"), (32, "32-bit Float"),
        ], cols=3)

        # ── SIMULATION QUALITY ────────────────────────────────────────────────
        section("SIMULATION QUALITY")

        sq_opts = [
            ("Draft",           "Draft  —  fastest"),
            ("Standard",        "Standard"),
            ("High",            "High  —  realtime"),
            ("Ultra",           "Ultra  —  512 blk"),
            ("Pristine (2×OS)", "Pristine  2×OS"),
            ("Extreme (4×OS)",  "Extreme   4×OS"),
            ("Archival (8×OS)", "Archival  8×OS"),
        ]
        radio_group(self.render_sim_quality, sq_opts, cols=4)

        sq_desc = {
            "Draft":           "Block: 4096 · No barkhausen · No print-through · Fastest render",
            "Standard":        "Block: 2048 · Barkhausen enabled, print-through disabled",
            "High":            "Block: 1024 · Full simulation — identical to realtime playback",
            "Ultra":           "Block:  512 · Smaller blocks, tighter transient accuracy",
            "Pristine (2×OS)": "Block: 1024 · 2× oversample nonlinear stages — cleaner harmonics",
            "Extreme (4×OS)":  "Block: 1024 · 4× oversample — removes high-order intermodulation",
            "Archival (8×OS)": "Block:  512 · 8× oversample — maximum nonlinear accuracy, slowest",
        }
        desc_lbl = tk.Label(sf, text=sq_desc.get(self.render_sim_quality.get(), ""),
                            bg=C["bg"], fg=C["grey_lt"], font=("Consolas", 8),
                            anchor="w", padx=20, pady=2, wraplength=640, justify="left")
        desc_lbl.pack(fill="x")
        def _upd_desc(*_):
            desc_lbl.config(text=sq_desc.get(self.render_sim_quality.get(), ""))
        self.render_sim_quality.trace_add("write", _upd_desc)

        # ── SIMULATION OPTIONS ────────────────────────────────────────────────
        section("SIMULATION OPTIONS")
        check_row(self.render_dither,    "Apply TPDF dither before bit-truncation  (recommended for 16/24-bit)")
        check_row(self.render_dc_block,  "DC blocking — remove low-frequency offset accumulated by DSP chain")
        check_row(self.render_normalize, "Peak normalize output to −0.1 dBFS after processing")

        # ── OUTPUT FILE ───────────────────────────────────────────────────────
        section("OUTPUT FILE")
        path_row = tk.Frame(sf, bg=C["bg4"], pady=8)
        path_row.pack(fill="x", padx=16, pady=4)
        tk.Label(path_row, text="Output Path", bg=C["bg4"], fg=C["grey_lt"],
                 font=("Consolas", 10), padx=10, anchor="w").pack(side="left")
        tk.Entry(path_row, textvariable=self.render_output_path,
                 bg=C["bg2"], fg=C["white"], insertbackground=C["white"],
                 font=("Consolas", 9), relief="flat", bd=0
                 ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        def browse():
            p = filedialog.asksaveasfilename(defaultextension=".wav",
                filetypes=[("WAV", "*.wav"), ("All files", "*.*")])
            if p:
                self.render_output_path.set(p)
        self._btn(path_row, "Browse", browse,
                  bg=C["bg2"], fg=C["grey_lt"], font=("Consolas", 9)).pack(side="left")

        tk.Frame(sf, bg=C["bg"], height=12).pack()   # bottom padding


    def _run_render(self, path, sample_rate, bit_depth, sim_quality):
        log_render.info(f"Starting render → {path}")
        log_render.info(f"  Sample rate : {sample_rate} Hz")
        log_render.info(f"  Bit depth   : {bit_depth}-bit")
        log_render.info(f"  Sim quality : {sim_quality}")

        # Quality → block size, oversampling, and feature flags
        quality_settings = {
            "Draft":           {"block": 4096, "oversample": 1, "barkhausen": False, "print_through": False},
            "Standard":        {"block": 2048, "oversample": 1, "barkhausen": True,  "print_through": False},
            "High":            {"block": 1024, "oversample": 1, "barkhausen": True,  "print_through": True},
            "Ultra":           {"block":  512, "oversample": 1, "barkhausen": True,  "print_through": True},
            "Pristine (2×OS)": {"block": 1024, "oversample": 2, "barkhausen": True,  "print_through": True},
            "Extreme (4×OS)":  {"block": 1024, "oversample": 4, "barkhausen": True,  "print_through": True},
            "Archival (8×OS)": {"block":  512, "oversample": 8, "barkhausen": True,  "print_through": True},
        }
        qs = quality_settings.get(sim_quality, quality_settings["High"])
        block_size  = qs["block"]
        oversample  = qs["oversample"]
        do_dither   = self.render_dither.get()
        do_dc_block = self.render_dc_block.get()
        do_norm     = self.render_normalize.get()
        log_render.debug(f"Block size: {block_size}  Oversample: {oversample}×")
        log_render.debug(f"Dither: {do_dither}  DC-block: {do_dc_block}  Normalize: {do_norm}")

        # Stop playback and wait for threads
        self.engine.is_playing = False
        self.is_rewinding      = False
        self.btn_export.config(text="RENDERING…", state="disabled")

        def _run():
            log_render.info("Waiting for playback threads to exit…")
            time.sleep(0.15)

            if self.engine.is_reversed:
                log_render.debug("Audio was reversed — restoring forward orientation")
                self.engine.set_reverse(False)

            # Override sim-quality params if needed
            saved_bark   = self.engine.params.get("barkhausen", 0.0)
            saved_print  = self.engine.params.get("print_through", 0.0)
            if not qs["barkhausen"]:
                self.engine.params["barkhausen"] = 0.0
                log_render.debug("Draft: barkhausen disabled")
            if not qs["print_through"]:
                self.engine.params["print_through"] = 0.0
                log_render.debug("Draft/Standard: print-through disabled")

            with self.engine.lock:
                self.engine.reset_state()

            total_samples = self.engine.total_samples
            log_render.info(f"Source: {total_samples} samples "
                            f"({total_samples/44100:.1f} s)")

            chunks      = []
            block_count = 0
            t_start     = time.time()
            last_log    = t_start

            while True:
                block = self.engine.dsp_process(block_size, oversample=oversample)
                if block is None:
                    break
                # Resample block if target SR differs from 44100
                if sample_rate != 44100:
                    from scipy.signal import resample_poly
                    block = resample_poly(block, sample_rate, 44100, axis=0)
                chunks.append(block)
                block_count += 1

                now = time.time()
                if now - last_log >= 1.0:
                    rendered_samps = block_count * block_size
                    pct = min(100, rendered_samps / max(total_samples, 1) * 100)
                    rt  = now - t_start
                    spd = rendered_samps / 44100 / max(rt, 0.001)
                    log_render.info(f"  {pct:5.1f}%  block {block_count}  "
                                    f"elapsed {rt:.1f}s  ({spd:.1f}× realtime)")
                    last_log = now

            # Restore overridden params
            self.engine.params["barkhausen"]    = saved_bark
            self.engine.params["print_through"] = saved_print

            t_elapsed = time.time() - t_start
            log_render.info(f"DSP complete: {block_count} blocks in {t_elapsed:.2f}s")

            # ── Post-processing ──────────────────────────────────────────────
            out_arr = np.vstack(chunks).astype(np.float64)

            if do_dc_block:
                log_render.debug("Applying DC blocking filter")
                from scipy.signal import butter, lfilter
                b_dc, a_dc = butter(1, 10.0 / (sample_rate / 2.0), btype="high")
                out_arr = lfilter(b_dc, a_dc, out_arr, axis=0)

            if do_norm:
                peak = np.max(np.abs(out_arr))
                if peak > 0:
                    target = 10 ** (-0.1 / 20)
                    out_arr *= target / peak
                    log_render.debug(f"Normalized: peak {peak:.4f} → {target:.4f}")

            if do_dither and bit_depth < 32:
                log_render.debug(f"Applying TPDF dither for {bit_depth}-bit")
                # TPDF: two uniform noise sources, peak = ±1 LSB
                lsb = 1.0 / (2 ** (bit_depth - 1))
                dither = (np.random.uniform(-lsb, lsb, out_arr.shape) +
                          np.random.uniform(-lsb, lsb, out_arr.shape))
                out_arr = out_arr + dither

            out_arr = np.clip(out_arr, -1.0, 1.0)

            # ── Encode ───────────────────────────────────────────────────────
            log_render.info(f"Encoding to {bit_depth}-bit WAV at {sample_rate} Hz…")

            if bit_depth == 32:
                raw = out_arr.astype(np.float32).tobytes()
                AudioSegment(raw, frame_rate=sample_rate,
                             sample_width=4, channels=2).export(path, format="wav")
            elif bit_depth == 24:
                try:
                    from scipy.io import wavfile
                    data_24 = (out_arr * 8388607).astype(np.int32)
                    wavfile.write(path, sample_rate, data_24)
                    log_render.debug("Wrote 24-bit via scipy.io.wavfile")
                except Exception as e:
                    log_render.warning(f"scipy 24-bit failed ({e}), falling back to 16-bit")
                    data_16 = (out_arr * 32767).astype(np.int16)
                    AudioSegment(data_16.tobytes(), frame_rate=sample_rate,
                                 sample_width=2, channels=2).export(path, format="wav")
            else:
                data_16 = (out_arr * 32767).astype(np.int16)
                AudioSegment(data_16.tobytes(), frame_rate=sample_rate,
                             sample_width=2, channels=2).export(path, format="wav")

            log_render.info(f"Render complete → {path}")
            self.root.after(0, lambda: [
                messagebox.showinfo("Render Complete",
                                    f"✓  Saved to:\n{path}\n\n"
                                    f"{sample_rate//1000}kHz  ·  {bit_depth}-bit  ·  {sim_quality}"),
                self.btn_export.config(text="⬛  FORENSIC RENDER", state="normal"),
            ])

        threading.Thread(target=_run, daemon=True, name="RenderThread").start()

    # ── Presets ──────────────────────────────────────────────────────────────────
    def apply_builtin_preset(self, name):
        log_ui.info(f"Applying preset: {name}")
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

    # ── Telemetry ────────────────────────────────────────────────────────────────
    def update_telemetry(self):
        # ── Reel position ──────────────────────────────────────────────────
        if self.engine.total_samples > 0:
            ph           = self.engine.play_head
            ts           = self.engine.total_samples
            fwd_progress = float(np.clip(
                (1.0 - ph / ts) if self.engine.is_reversed else (ph / ts), 0, 1))
            filled   = int(fwd_progress * 8)
            supply_s = "●" * (8 - filled) + "○" * filled
            takeup_s = "○" * (8 - filled) + "●" * filled
            elapsed_s = ph / 44100.0
            remain_s  = (ts - ph) / 44100.0
            pos_str   = f"{int(elapsed_s//60):02d}:{elapsed_s%60:05.2f}"
        else:
            supply_s  = "●●●●●●●●"
            takeup_s  = "○○○○○○○○"
            pos_str   = "00:00.00"
            remain_s  = 0.0

        transport = self.engine.transport
        inst_spd  = getattr(transport, 'last_instant_speed', 1.0)
        conflict  = getattr(transport, 'last_conflict', 0.0)
        drop_n    = getattr(transport, 'dropout_count', 0)
        drop_t    = getattr(transport, 'last_dropout_time', -999.0)
        cur_time  = self.engine.current_time
        ips_set   = self.controls["ips_base"].get()

        # ── Simulated voltage noise for VU bar ────────────────────────────
        health    = self.controls["motor_health"].get()
        drag      = self.controls["motor_drag"].get()
        boost     = self.controls["motor_boost"].get()
        shed      = self.controls["sticky_shed"].get()

        if self.engine.is_playing:
            # Voltage wanders with motor health and conflict
            if not hasattr(self, '_volt_state'):
                self._volt_state = 0.0
            noise = np.random.normal(0, health * 0.008 + conflict * 0.02)
            self._volt_state += noise - self._volt_state * 0.05
            volt_dev = float(np.clip(self._volt_state, -1.0, 1.0))
            # Nominal voltage 12V + deviation
            volt_v = 12.0 + volt_dev * (0.5 + health * 0.4)
        else:
            self._volt_state = 0.0
            volt_dev = 0.0
            volt_v = 12.0

        # ── Voltage bar (20 chars wide) ────────────────────────────────────
        BAR_W = 20
        centre = BAR_W // 2
        fill_amt = int(abs(volt_dev) * centre)
        if volt_dev >= 0:
            bar_chars = list("·" * BAR_W)
            for i in range(centre, min(BAR_W, centre + fill_amt)):
                bar_chars[i] = "█"
        else:
            bar_chars = list("·" * BAR_W)
            for i in range(max(0, centre - fill_amt), centre):
                bar_chars[i] = "█"
        bar_chars[centre] = "│"
        bar_str = "".join(bar_chars)

        # ── Voltage colour + label ─────────────────────────────────────────
        abs_dev = abs(volt_dev)
        if abs_dev < 0.15:
            volt_fg    = C["grey"]
            volt_state = "STABLE"
        elif abs_dev < 0.40:
            volt_fg    = C["amber"]
            volt_state = "DRIFTING"
        elif abs_dev < 0.70:
            volt_fg    = C["orange"]
            volt_state = "UNSTABLE"
        else:
            volt_fg    = C["red"]
            volt_state = "CRITICAL"

        # ── Speed deviation % ─────────────────────────────────────────────
        spd_dev_pct = (inst_spd - 1.0) * 100.0

        # ── Dropout last-seen countdown ───────────────────────────────────
        if drop_t > 0 and self.engine.is_playing:
            since = cur_time - drop_t
            if since < 9.9:
                drop_last_str = f"{since:4.1f}s"
            else:
                drop_last_str = " >9s"
        else:
            drop_last_str = "  --"

        # ── Head wear from sticky shed + demagnetisation ──────────────────
        demag = self.controls["demagnetization"].get()
        wear_score = shed * 0.6 + demag * 0.4
        if wear_score < 0.05:
            wear_str = "NOMINAL"
            wear_fg  = C["grey"]
        elif wear_score < 0.25:
            wear_str = "MINOR WEAR"
            wear_fg  = C["amber"]
        elif wear_score < 0.60:
            wear_str = "DEGRADED"
            wear_fg  = C["orange"]
        else:
            wear_str = "SEVERE DAMAGE"
            wear_fg  = C["red"]

        # ── State string + colours ─────────────────────────────────────────
        tick = int(cur_time * 8) % 2
        spin = ["◆", "◇"]

        if self.is_rewinding:
            state_str  = f"{spin[tick]} REWINDING"
            state_fg   = C["cyan"]
            ips_str    = f"◀◀ {ips_set * 40.0:06.2f}"
            ips_fg     = C["cyan"]
            reel_str   = f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}"
            reel_fg    = C["cyan"]
        elif self.is_ffing:
            state_str  = f"{spin[tick]} FAST FWD"
            state_fg   = C["green"]
            ips_str    = f"▶▶ {ips_set * 40.0:06.2f}"
            ips_fg     = C["green"]
            reel_str   = f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}"
            reel_fg    = C["green"]
        elif self.engine.is_playing:
            actual     = ips_set * inst_spd
            rev        = self.engine.is_reversed
            arrow      = "◀" if rev else "▶"
            if conflict > 0.1:
                state_str = f"{spin[tick]} FIGHTING"
                state_fg  = C["red"]
            elif abs(spd_dev_pct) > 8:
                state_str = f"{spin[tick]} UNSTABLE"
                state_fg  = C["red"]
            elif rev:
                state_str = f"{spin[tick]} REVERSE"
                state_fg  = C["orange"]
            else:
                state_str = f"{spin[tick]} RUNNING"
                state_fg  = C["amber"]
            ips_str    = f"{arrow} {actual:06.3f}"
            ips_fg     = C["orange"] if rev else C["amber"]
            reel_str   = f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}"
            reel_fg    = ips_fg
        else:
            state_str  = "  IDLE"
            state_fg   = C["grey"]
            ips_str    = "▶ 00.000"
            ips_fg     = C["grey"]
            reel_str   = f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}"
            reel_fg    = C["grey"]

        # ── Apply to widgets ───────────────────────────────────────────────
        self.lbl_sys_state.config(
            text=f"STATE    : {state_str:<14s}  {pos_str}",
            fg=state_fg)
        self.lbl_sys_speed.config(
            text=f"SPEED    : {ips_set * (inst_spd if self.engine.is_playing else 1.0):06.3f} IPS",
            fg=ips_fg if self.engine.is_playing else C["grey"])
        self.lbl_sys_dev.config(
            text=f"DEVIATION: {spd_dev_pct:+.2f}%  CONFLICT: {conflict:.2f}",
            fg=(C["red"] if abs(spd_dev_pct) > 8 or conflict > 0.2
                else C["amber"] if abs(spd_dev_pct) > 2
                else C["grey"]))

        self.lbl_volt_bar.config(
            text=f"VOLT  [{bar_str}]",
            fg=volt_fg)
        self.lbl_volt_val.config(
            text=f"PSU   : {volt_state:<10s} {volt_v:+.2f}V",
            fg=volt_fg)

        self.lbl_drop_count.config(
            text=f"DROPOUT  : #{drop_n:<5d} LAST:{drop_last_str}",
            fg=(C["orange"] if drop_n > 0 else C["grey"]))
        self.lbl_head_temp.config(
            text=f"HEAD WEAR: {wear_str}",
            fg=wear_fg)

        # ── IPS counter + reel (right block, unchanged) ───────────────────
        self.lbl_ips.config(text=ips_str, fg=ips_fg)
        self.lbl_reel.config(text=reel_str, fg=reel_fg)

        self.root.after(40, self.update_telemetry)

    # ── Transport controls ───────────────────────────────────────────────────────
    def _stop_all(self):
        log_transport.debug("_stop_all: halting playback, rewind and FF")
        self.engine.is_playing = False
        self.is_rewinding      = False
        self.is_ffing          = False
        self.btn_play.config(text="▶  PLAY",     fg=C["amber"],  state="normal")
        self.btn_rev_play.config(text="◀  REVERSE", fg=C["orange"], state="normal")
        self.btn_rewind.config(text="◀◀  REWIND", fg=C["cyan"])
        self.btn_ff.config(text="FF  ▶▶", fg=C["green"])

    def start_forward(self):
        log_transport.info("START FORWARD")
        if self.engine.audio_data is None:
            log_transport.warning("No audio loaded")
            return
        already_playing = self.engine.is_playing and not self.engine.is_reversed
        self._stop_all()
        time.sleep(0.06)          # let audio callback exit
        # set_reverse only resets head position if direction actually changes
        self.engine.set_reverse(False)
        if already_playing:
            log_transport.debug("Resuming forward (head position preserved)")
        self.engine.is_playing = True
        self.btn_play.config(text="▶  PLAYING", fg=C["red"])
        threading.Thread(target=self._audio_thread, daemon=True,
                         name="AudioFwd").start()

    def start_reverse(self):
        log_transport.info("START REVERSE")
        if self.engine.audio_data is None:
            log_transport.warning("No audio loaded")
            return
        already_reversing = self.engine.is_playing and self.engine.is_reversed
        self._stop_all()
        time.sleep(0.06)
        self.engine.set_reverse(True)
        if already_reversing:
            log_transport.debug("Resuming reverse (head position preserved)")
        self.engine.is_playing = True
        self.btn_rev_play.config(text="◀  REVERSING", fg=C["red"])
        threading.Thread(target=self._audio_thread, daemon=True,
                         name="AudioRev").start()

    def stop_transport(self):
        log_transport.info("STOP")
        self._stop_all()
        if self.engine.is_reversed:
            threading.Thread(target=lambda: self.engine.set_reverse(False),
                             daemon=True).start()

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            log_ui.info(f"Loading file: {path}")
            self._stop_all()
            threading.Thread(target=self.engine.load_file, args=(path,),
                             daemon=True, name="LoadFile").start()

    def _audio_thread(self):
        log_transport.debug("Audio thread started")
        def cb(out, frames, time_info, status):
            d = self.engine.dsp_process(frames)
            if d is None:
                raise sd.CallbackStop
            out[:] = d

        with sd.OutputStream(samplerate=44100, channels=2,
                              callback=cb, blocksize=1024):
            while self.engine.is_playing:
                sd.sleep(50)

        log_transport.debug("Audio thread: stream ended")
        self.engine.is_playing = False
        if self.engine.is_reversed:
            self.engine.set_reverse(False)
        self.root.after(0, lambda: [
            self.btn_play.config(text="▶  PLAY",     fg=C["amber"]),
            self.btn_rev_play.config(text="◀  REVERSE", fg=C["orange"]),
        ])

    # ── Rewind ──────────────────────────────────────────────────────────────────
    def toggle_rewind(self):
        if self.engine.audio_data is None:
            return
        if self.is_rewinding:
            log_transport.info("REWIND stop")
            self.is_rewinding = False
            self.btn_rewind.config(text="◀◀  REWIND", fg=C["cyan"])
        else:
            log_transport.info("REWIND start")
            self._stop_all()
            self.is_rewinding = True
            self.btn_rewind.config(text="■  STOP RWD", fg=C["red"])
            threading.Thread(target=self._shuttle_thread,
                             args=("rewind",), daemon=True, name="Rewind").start()

    # ── Fast Forward ─────────────────────────────────────────────────────────
    def toggle_ff(self):
        if self.engine.audio_data is None:
            return
        if self.is_ffing:
            log_transport.info("FF stop")
            self.is_ffing = False
            self.btn_ff.config(text="FF  ▶▶", fg=C["green"])
        else:
            log_transport.info("FF start")
            self._stop_all()
            self.is_ffing = True
            self.btn_ff.config(text="■  STOP FF", fg=C["red"])
            threading.Thread(target=self._shuttle_thread,
                             args=("ff",), daemon=True, name="FastFwd").start()

    # ── Shared shuttle thread (rewind + FF) ──────────────────────────────────
    def _shuttle_thread(self, direction):
        """
        Shared transport thread for both rewind and fast-forward.
        direction = "rewind" | "ff"
        Produces authentic mechanical squeal pitched to reel diameter and speed.
        """
        log_transport.debug(f"Shuttle thread started ({direction})")
        SR           = 44100
        block        = 512
        base_ips     = self.controls["ips_base"].get()
        squeal_phase = 0.0

        is_active    = lambda: self.is_rewinding if direction == "rewind" else self.is_ffing
        stop_flag    = "is_rewinding" if direction == "rewind" else "is_ffing"
        btn          = self.btn_rewind if direction == "rewind" else self.btn_ff
        btn_idle_txt = "◀◀  REWIND" if direction == "rewind" else "FF  ▶▶"
        btn_idle_fg  = C["cyan"]     if direction == "rewind" else C["green"]
        # Multiplier: rewind moves backward, FF moves forward
        direction_sign = -1 if direction == "rewind" else +1

        def cb(out, frames, time_info, status):
            nonlocal squeal_phase
            if not is_active():
                raise sd.CallbackStop

            with self.engine.lock:
                pos   = self.engine.play_head
                total = self.engine.total_samples

            # Check end conditions
            at_end   = (direction == "rewind" and pos <= 0) or                        (direction == "ff"     and pos >= total - block)
            if at_end:
                clamp = 0.0 if direction == "rewind" else float(total - 1)
                with self.engine.lock:
                    self.engine.play_head    = clamp
                    self.engine.current_time = clamp / SR
                setattr(self, stop_flag, False)
                self.root.after(0, lambda: btn.config(text=btn_idle_txt, fg=btn_idle_fg))
                raise sd.CallbackStop

            step    = (base_ips / 15.0) * SR * 40.0 * frames / SR * direction_sign
            new_pos = float(np.clip(pos + step, 0.0, float(total - 1)))
            with self.engine.lock:
                self.engine.play_head    = new_pos
                self.engine.current_time = new_pos / SR

            # Squeal pitch: depends on which reel is under tension
            # Rewind: supply reel shrinks (higher pitch as it empties)
            # FF:     takeup reel grows (lower pitch as it fills)
            progress    = float(new_pos / max(total, 1))
            if direction == "rewind":
                reel_load = 1.0 + (1.0 - progress) * 1.5   # higher pitch near end
            else:
                reel_load = 1.0 + progress * 1.5            # higher pitch near end

            squeal_hz  = float(np.clip(
                1800.0 * (base_ips / 15.0) * reel_load, 150, 16000))
            phase_inc  = 2 * np.pi * squeal_hz / SR
            phases     = squeal_phase + np.arange(frames) * phase_inc
            # Richer harmonic content than a simple sine
            squeal_sig = (np.sin(phases)         * 0.50
                        + np.sin(phases * 3)     * 0.15
                        + np.sin(phases * 5)     * 0.07
                        + np.sin(phases * 7)     * 0.03
                        + np.sin(phases * 0.5)   * 0.08     # sub-harmonic flutter
                        + np.random.normal(0, 0.008, frames))  # mechanical noise
            squeal_phase = (squeal_phase + frames * phase_inc) % (2 * np.pi)

            # Amplitude envelope: louder in the middle of the reel travel
            env_progress = progress if direction == "ff" else (1.0 - progress)
            amp = 0.20 * np.sin(np.pi * env_progress) + 0.04
            # Stereo: slight pan toward supply reel side
            pan_l = 1.00 if direction == "rewind" else 0.92
            pan_r = 0.92 if direction == "rewind" else 1.00
            out[:] = (squeal_sig * amp)[:, None] * np.array([[pan_l, pan_r]])

        with sd.OutputStream(samplerate=SR, channels=2, callback=cb,
                             blocksize=block, dtype="float32"):
            while is_active():
                sd.sleep(50)

        log_transport.debug(f"Shuttle thread ended ({direction})")
        setattr(self, stop_flag, False)
        self.root.after(0, lambda: btn.config(text=btn_idle_txt, fg=btn_idle_fg))

    # ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_ui.info("Launching application")
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = ForensicTapeStudio(root)
    root.mainloop()
    log_ui.info("Application closed")
