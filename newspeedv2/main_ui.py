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
        self._syncing     = False   # suppresses per-slider sync() during batch preset apply

        # Render options
        self.render_sample_rate  = tk.IntVar(value=44100)
        self.render_bit_depth    = tk.IntVar(value=24)
        self.render_sim_quality  = tk.StringVar(value="High")
        self.render_oversample   = tk.IntVar(value=1)
        self.render_dither         = tk.BooleanVar(value=True)
        self.render_dc_block       = tk.BooleanVar(value=True)
        self.render_normalize      = tk.BooleanVar(value=True)
        self.render_match_loudness = tk.BooleanVar(value=False)
        self.render_direction      = tk.StringVar(value="forward")
        self.render_output_path    = tk.StringVar(value="")
        self.render_threads        = tk.IntVar(value=1)
        self.render_preroll        = tk.DoubleVar(value=0.0)  # seconds of silence/noise before audio

        self._closing = False
        self._setup_styles()
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
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

        def _sep():
            tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x", padx=8, pady=2)

        # Section header
        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x")
        tk.Label(status_block, text=" MACHINE STATUS ",
                 bg=C["bg4"], fg=C["amber"], font=("Consolas", 7, "bold"),
                 anchor="w", padx=10, pady=2).pack(fill="x")
        tk.Frame(status_block, bg=C["border"], height=1).pack(fill="x")
        tk.Frame(status_block, bg="#000", height=3).pack()

        # ── Transport ────────────────────────────────────────────────────────
        self.lbl_sys_state  = _sl("STATE    : IDLE",       fg=C["grey"])
        self.lbl_sys_speed  = _sl("SPEED    : 00.000 IPS", fg=C["grey"])
        self.lbl_sys_dev    = _sl("DEVIATION: ±0.000%",    fg=C["grey"])
        self.lbl_sys_pos    = _sl("POSITION : 00:00.00  REMAIN: 00:00.00", fg=C["grey"])
        _sep()

        # ── Power supply ─────────────────────────────────────────────────────
        self.lbl_volt_bar   = _sl("VOLT  [                    ]", fg=C["grey"],
                                  font=("Consolas", 7))
        self.lbl_volt_val   = _sl("PWR SUPPLY: STABLE  0.00V",   fg=C["grey"])
        _sep()

        # ── Signal / oxide ───────────────────────────────────────────────────
        self.lbl_sig_level  = _sl("SIGNAL   : ----  NOISE: ----",  fg=C["grey"])
        self.lbl_oxide      = _sl("OXIDE    : Fe2O3  SAT: LOW",    fg=C["grey"])
        self.lbl_bias_state = _sl("BIAS     : NOMINAL",             fg=C["grey"])
        _sep()

        # ── Tape condition ───────────────────────────────────────────────────
        self.lbl_drop_count = _sl("DROPOUTS : 0  LAST: --.-s",    fg=C["grey"])
        self.lbl_shed_state = _sl("BINDER   : INTACT",             fg=C["grey"])
        self.lbl_demag_state= _sl("DEMAG    : CLEAR",              fg=C["grey"])
        self.lbl_head_temp  = _sl("HEAD WEAR: NOMINAL",             fg=C["grey"])
        _sep()

        # ── Active effects ───────────────────────────────────────────────────
        self.lbl_fx_chain   = _sl("FX       : ···",                fg=C["grey"],
                                  font=("Consolas", 7))

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
            # ── Studio / Pro ──────────────────────────────────────────────────
            "Ampex 456 (30ips)",  "Ampex 456 (15ips)",
            "Studer A820 (30ips)", "Studer A820 (15ips)",
            "Otari MTR-90 (30ips)", "Otari MTR-90 (15ips)",
            "MCI JH-24 (30ips)",
            "Scotch 226 (7.5ips)",
            "Revox B77 (7.5ips)", "Revox B77 (3.75ips)",
            # ── Consumer reel ─────────────────────────────────────────────────
            "BASF LH Super (7.5ips)",
            "Maxell UD (7.5ips)",
            "Tascam 38 (7.5ips)",
            # ── Multitrack ────────────────────────────────────────────────────
            "4-Track Portastudio (1.875ips)",
            "8-Track Cartridge",
            # ── Cassette ──────────────────────────────────────────────────────
            "Type I (Fe2O3) Normal",
            "Type II Chrome (CrO2)",
            "Type IV Metal",
            "Dolby B (Type I)",
            "Dolby C (Type II)",
            "Lo-Fi Bedroom (Type I)",
            # ── Video / Broadcast ─────────────────────────────────────────────
            "VHS Linear Audio",
            "Betamax Audio",
            "U-Matic Low Band",
            # ── Damaged / Warped ──────────────────────────────────────────────
            "Sticky Shed Syndrome",
            "Baked Tape (Post-Oven)",
            "Mouldy Attic Find",
            "Dropout Disaster",
            "Heavily Demagnetised",
            "Warped & Fighting Motors",
            "Chewed Tape",
            "Print-Through Ghost",
            "Stretched Tape",
            "Heat Warped",
            "Spliced Archive",
            "Soviet ORWO Copy",
            # ── Radio / Broadcast ─────────────────────────────────────────────
            "BBC Radiophonic (7.5ips)",
            "AM Radio Dub",
            # ── Lo-Fi / Special ───────────────────────────────────────────────
            "Ghetto Blaster",
            "Answering Machine",
            "Handheld Dictaphone",
            "Toy Piano Recording",
            # ── More Damaged ──────────────────────────────────────────────────
            "Tsunami Flood Tape",
            "Fire-Damaged Archive",
            "Played 1000 Times",
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

        # ── Render progress bar (hidden until render starts) ──────────────────
        prog_row = tk.Frame(footer, bg=C["bg2"])
        prog_row.pack(fill="x", padx=16, pady=(0, 6))
        self._render_prog_row  = prog_row

        import tkinter.ttk as ttk_prog
        style = ttk_prog.Style()
        style.theme_use("default")
        style.configure("Render.Horizontal.TProgressbar",
                        troughcolor=C["bg3"], background=C["purple"],
                        thickness=6, borderwidth=0)
        self._render_prog_bar = ttk_prog.Progressbar(
            prog_row, orient="horizontal", length=400, mode="determinate",
            style="Render.Horizontal.TProgressbar")
        self._render_prog_lbl = tk.Label(
            prog_row, text="", bg=C["bg"], fg=C["purple"],
            font=("Consolas", 8), anchor="w")
        # Packed dynamically when render starts
        prog_row.pack_forget()

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
        if self._syncing:
            return
        new_params = {k: v.get() for k, v in self.controls.items()}
        new_params["oxide_type"] = self.oxide_var.get()
        with self.engine.lock:
            self.engine.params = new_params

    # ── Render progress helpers (must be called from main thread via root.after) ──
    def _render_prog_show(self):
        """Reveal the progress bar row and reset it to 0."""
        self._render_prog_bar["value"] = 0
        self._render_prog_lbl.config(text="RENDERING…  0%")
        self._render_prog_bar.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._render_prog_lbl.pack(side="left")
        self._render_prog_row.pack(fill="x", padx=16, pady=(0, 6))

    def _render_prog_update(self, pct, eta_str):
        """Update bar value and label — call via root.after(0, ...) from render thread."""
        self._render_prog_bar["value"] = pct
        self._render_prog_lbl.config(text=f"RENDERING…  {pct:.0f}%  {eta_str}")

    def _render_prog_hide(self):
        """Collapse the progress bar row after render finishes or fails."""
        self._render_prog_row.pack_forget()
        self._render_prog_bar.pack_forget()
        self._render_prog_lbl.pack_forget()

    # ── Render options dialog ─────────────────────────────────────────────
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
                             self.render_sim_quality.get(),
                             self.render_direction.get(),
                             self.render_preroll.get())

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

        def _update_scroll(e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        sf.bind("<Configure>", _update_scroll)
        # Force scrollregion update after all widgets are drawn
        win.after(50, _update_scroll)

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
        label_row("Render Direction")
        radio_group(self.render_direction, [
            ("forward", "Forward  ▶"),
            ("reverse", "Reverse  ◀  (bake reverse into file)"),
        ], cols=2)

        check_row(self.render_dither,         "Apply TPDF dither before bit-truncation  (recommended for 16/24-bit)")
        check_row(self.render_dc_block,       "DC blocking — remove low-frequency offset accumulated by DSP chain")
        check_row(self.render_normalize,      "Peak normalize to −0.1 dBFS  (recommended — tape compression reduces level)")
        check_row(self.render_match_loudness, "RMS loudness match — match output loudness to input source level")

        # ── PRE-ROLL ─────────────────────────────────────────────────────────
        section("PRE-ROLL  (leader noise before audio)")
        label_row("Seconds of machine noise before the recording starts — hiss, hum, motor spin-up")
        preroll_opts = [(0.0, "None"), (0.5, "0.5 s"), (1.0, "1 s"),
                        (2.0, "2 s"), (3.0, "3 s"), (5.0, "5 s")]
        radio_group(self.render_preroll, preroll_opts, cols=6)
        label_row("Pre-roll renders electronics noise (hiss + hum) at full level with tape spin-up inertia.")

        # ── PERFORMANCE ───────────────────────────────────────────────────────
        section("PERFORMANCE  (experimental)")
        label_row("Render Threads  —  splits audio into parallel chunks, one per thread")
        radio_group(self.render_threads, [
            (1, "1 thread  (default)"),
            (2, "2 threads"),
            (4, "4 threads"),
            (8, "8 threads"),
        ], cols=4)
        tk.Label(sf, text="  ⚠  Parallel threads may introduce subtle level discontinuities at chunk boundaries.",
                 bg=C["bg"], fg=C["orange"], font=("Consolas", 8),
                 anchor="w", padx=20, pady=2).pack(fill="x")

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


    def _run_render(self, path, sample_rate, bit_depth, sim_quality,
                    direction="forward", preroll=0.0):
        log_render.info(f"Starting render → {path}")
        log_render.info(f"  Sample rate : {sample_rate} Hz")
        log_render.info(f"  Bit depth   : {bit_depth}-bit")
        log_render.info(f"  Sim quality : {sim_quality}")
        log_render.info(f"  Direction   : {direction}")

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
        n_threads   = self.render_threads.get()
        do_dither         = self.render_dither.get()
        do_dc_block       = self.render_dc_block.get()
        do_norm           = self.render_normalize.get()
        do_match_loudness = self.render_match_loudness.get()
        log_render.debug(f"Block size: {block_size}  Oversample: {oversample}×  Threads: {n_threads}")
        log_render.debug(f"Dither: {do_dither}  DC-block: {do_dc_block}  Normalize: {do_norm}")

        # Stop playback and wait for threads
        self.engine.is_playing = False
        self.is_rewinding      = False
        self.btn_export.config(text="RENDERING…", state="disabled")

        def _run():
            # Show progress bar
            self.root.after(0, self._render_prog_show)
            try:
                _run_body()
            except Exception as exc:
                import traceback
                log_render.error(f"Render failed: {exc}\n{traceback.format_exc()}")
                err_msg = str(exc)   # capture before Python deletes `exc` after except block
                self.root.after(0, lambda m=err_msg: [
                    messagebox.showerror("Render Failed",
                                         f"Render encountered an error:\n\n{m}"),
                    self.btn_export.config(text="⬛  FORENSIC RENDER", state="normal"),
                    self._render_prog_hide(),
                ])
            else:
                self.root.after(0, self._render_prog_hide)

        def _run_body():
            log_render.info("Waiting for playback threads to exit…")
            time.sleep(0.15)

            want_reverse = (direction == "reverse")
            log_render.debug(f"Setting render direction: {'reverse' if want_reverse else 'forward'}")
            self.engine.set_reverse(want_reverse, force_reset=True)

            saved_bark  = self.engine.params.get("barkhausen", 0.0)
            saved_print = self.engine.params.get("print_through", 0.0)

            # Override sim-quality params if needed
            if not qs["barkhausen"]:
                self.engine.params["barkhausen"] = 0.0
                log_render.debug("Draft: barkhausen disabled")
            if not qs["print_through"]:
                self.engine.params["print_through"] = 0.0
                log_render.debug("Draft/Standard: print-through disabled")

            if do_match_loudness and self.engine.audio_data is not None:
                src_rms = float(np.sqrt(np.mean(self.engine.audio_data.astype(np.float64)**2)))
                log_render.debug(f"Source RMS: {src_rms:.5f}")
            else:
                src_rms = None

            with self.engine.lock:
                self.engine.reset_state()

            total_samples = self.engine.total_samples
            log_render.info(f"Source: {total_samples} samples "
                            f"({total_samples/44100:.1f} s)")

            # ── Pre-roll: leader noise + motor spin-up before audio ───────────
            # Renders preroll seconds of electronics-only silence: hiss, hum, and
            # the capstan spinning up from rest.  No tape audio content is read —
            # the engine sees a silent buffer so only the noise floor is audible.
            preroll_chunks = []
            preroll_samps  = int(preroll * 44100)
            if preroll_samps > 0:
                log_render.info(f"Pre-roll: {preroll:.1f}s of leader noise")
                silence = np.zeros((preroll_samps + block_size * 4, 2), dtype=np.float32)
                pr_eng = TapeEngine()
                pr_eng.audio_data    = silence
                pr_eng.total_samples = len(silence)
                pr_eng.params        = dict(self.engine.params)
                pr_eng.is_reversed   = False
                # motor_engage starts at 0 from reset — ramps to 1 naturally,
                # giving the characteristic pitch-rise of a machine spinning up.
                pr_done = 0
                while pr_done < preroll_samps:
                    blk = pr_eng.dsp_process(block_size, oversample=oversample)
                    if blk is None:
                        break
                    keep = min(len(blk), preroll_samps - pr_done)
                    preroll_chunks.append(blk[:keep].astype(np.float32))
                    pr_done += keep
                log_render.info(f"Pre-roll complete: {pr_done} samples")
                # Flush motor_engage state into the main engine so the audio
                # picks up at full speed rather than spinning up a second time.
                self.engine.transport.motor_engage = pr_eng.transport.motor_engage
                self.engine.transport.current_motor_speed = (
                    pr_eng.transport.current_motor_speed)

            t_start = time.time()

            def _report_progress(block_count, total_samps, t_start, last_log_ref):
                """Log to console + push UI bar update every second."""
                now = time.time()
                if now - last_log_ref[0] < 1.0:
                    return
                rendered_samps = block_count * block_size
                pct     = min(100.0, rendered_samps / max(total_samps, 1) * 100.0)
                elapsed = max(now - t_start, 0.001)
                spd     = rendered_samps / 44100.0 / elapsed
                eta_s   = max(0.0, (total_samps - rendered_samps) / 44100.0 / max(spd, 0.001))
                eta_str = (f"ETA {int(eta_s//60):02d}:{int(eta_s%60):02d}"
                           if eta_s > 1 else "finishing…")
                log_render.info(f"  {pct:5.1f}%  blk {block_count}  "
                                f"{elapsed:.1f}s elapsed  {spd:.1f}×RT  {eta_str}")
                self.root.after(0, lambda p=pct, e=eta_str: self._render_prog_update(p, e))
                last_log_ref[0] = now

            if n_threads <= 1:
                # ── Single-threaded serial render ─────────────────────────────
                chunks      = []
                block_count = 0
                last_log    = [t_start]   # list so _report_progress can mutate it

                while True:
                    block = self.engine.dsp_process(block_size, oversample=oversample)
                    if block is None:
                        break
                    chunks.append(block.astype(np.float32))
                    block_count += 1
                    _report_progress(block_count, total_samples, t_start, last_log)
            else:
                # ── Multi-threaded parallel render ────────────────────────────
                # Strategy: divide the source into N equal slices, spin up N
                # independent TapeEngine instances (each with own state), render
                # each slice in parallel, then concatenate results in order.
                #
                # Each worker engine gets a copy of audio_data and params, but
                # starts at the correct play_head offset for its slice.
                # IIR state (replay_diff, demagnetisation, azimuth) is cold at
                # each slice boundary — this is the noted discontinuity trade-off.
                from concurrent.futures import ThreadPoolExecutor, as_completed

                log_render.info(f"Multi-thread render: {n_threads} workers")

                # Pre-compute how many blocks each worker renders
                # Each worker gets a contiguous range of source samples
                slice_samples = total_samples // n_threads

                futures = {}
                # Shared atomic counter — workers increment it each block so
                # the progress bar reflects actual sample throughput rather than
                # coarse worker-completion steps (25%/50%/75%/100%).
                import threading as _threading
                _mt_lock          = _threading.Lock()
                _mt_blocks_done   = [0]   # mutable list so workers can mutate it
                _total_blocks_est = max(1, total_samples // block_size)

                # All workers share the same oscillator seed so their wow/flutter
                # LFO phases are identical — no pitch step at slice boundaries.
                _shared_seed = int(self.engine.total_samples) ^ 0xA5A5A5A5

                def render_slice(worker_id, start_sample, end_sample):
                    """Render samples [start_sample, end_sample) on a warmed-up engine."""
                    log_render.debug(
                        f"Worker {worker_id}: start={start_sample} end={end_sample}")

                    # make_worker_engine runs warm-up blocks before start_sample
                    # to settle all IIR filter states, so there are no clicks or
                    # level steps at the joins between parallel chunks.
                    eng = self.engine.make_worker_engine(
                        start_sample, block_size, oversample,
                        shared_seed=_shared_seed,
                        n_warmup_blocks=16)

                    local_chunks = []
                    last_prog_t  = [time.time()]
                    while True:
                        if eng.play_head >= end_sample:
                            break
                        block = eng.dsp_process(block_size, oversample=oversample)
                        if block is None:
                            break
                        if eng.play_head > end_sample:
                            trim = int(eng.play_head - end_sample)
                            block = block[:-trim] if trim < len(block) else block
                        local_chunks.append(block.astype(np.float32))

                        # Update shared counter; push UI update at most once per second
                        with _mt_lock:
                            _mt_blocks_done[0] += 1
                            done = _mt_blocks_done[0]
                        now = time.time()
                        if now - last_prog_t[0] >= 1.0:
                            pct = min(99.0, done / _total_blocks_est * 100.0)
                            elapsed = max(now - t_start, 0.001)
                            spd = done * block_size / 44100.0 / elapsed
                            eta_s = max(0.0, (_total_blocks_est - done) * block_size
                                        / 44100.0 / max(spd, 0.001))
                            eta_str = (f"ETA {int(eta_s//60):02d}:{int(eta_s%60):02d}"
                                       if eta_s > 1 else "finishing…")
                            log_render.info(f"  {pct:5.1f}%  blk {done}/{_total_blocks_est}"
                                            f"  {elapsed:.1f}s  {spd:.1f}×RT  {eta_str}")
                            self.root.after(0,
                                lambda p=pct, e=eta_str: self._render_prog_update(p, e))
                            last_prog_t[0] = now

                    n_samp = sum(len(c) for c in local_chunks)
                    log_render.debug(
                        f"Worker {worker_id}: done — {len(local_chunks)} blocks, {n_samp} samples")
                    return worker_id, local_chunks

                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    for i in range(n_threads):
                        s = i * slice_samples
                        e = total_samples if i == n_threads - 1 else (i + 1) * slice_samples
                        fut = pool.submit(render_slice, i, s, e)
                        futures[fut] = i

                # Collect results — wrap fut.result() so a single worker failure
                # is logged and reported clearly rather than silently dropped.
                results = {}
                errors  = []
                for fut in as_completed(futures):
                    try:
                        wid, wchunks = fut.result()
                        results[wid] = wchunks
                        log_render.info(
                            f"  Worker {wid} done ({len(wchunks)} blocks)")
                    except Exception as exc:
                        import traceback
                        wid = futures.get(fut, "?")
                        log_render.error(
                            f"Worker {wid} raised: {exc}\n{traceback.format_exc()}")
                        errors.append((wid, str(exc)))

                if errors:
                    raise RuntimeError(
                        f"{len(errors)} render worker(s) failed: "
                        + "; ".join(str(e) for _, e in errors))

                chunks = []
                for i in range(n_threads):
                    chunks.extend(results.get(i, []))   # .get() guards missing workers
                block_count = len(chunks)
                if block_count == 0:
                    raise RuntimeError("Multi-thread render produced no audio blocks.")

            t_elapsed_dsp = time.time() - t_start
            log_render.info(f"DSP complete: {block_count} blocks in {t_elapsed_dsp:.2f}s")

            # ── Assemble at native 44100 Hz, then convert SR as ONE operation ─
            # Prepend pre-roll chunks (may be empty if preroll=0)
            all_chunks = preroll_chunks + chunks
            if not all_chunks:
                raise RuntimeError("Render produced no audio blocks.")
            out_arr = np.vstack(all_chunks).astype(np.float64)

            # ── Crossfade at multi-thread chunk boundaries ────────────────────
            # Each worker starts its IIR filters from warm-up state, and the
            # trim logic creates small gaps in read position at every boundary.
            # A short equal-power crossfade (XFADE_LEN samples) at each seam
            # eliminates clicks from any residual discontinuity.
            # 256 samples = 5.8 ms — inaudible as a blend on music.
            if n_threads > 1:
                XFADE = 256
                # Rebuild per-worker arrays so we know exactly where each seam is
                worker_arrs = []
                for i in range(n_threads):
                    wchunks = results.get(i, [])
                    if wchunks:
                        worker_arrs.append(np.vstack(wchunks).astype(np.float64))

                if len(worker_arrs) > 1:
                    # Crossfade adjacent workers
                    fade_out = np.linspace(1.0, 0.0, XFADE)[:, None]
                    fade_in  = np.linspace(0.0, 1.0, XFADE)[:, None]
                    merged = worker_arrs[0]
                    for nxt in worker_arrs[1:]:
                        # Each worker may not have enough samples for a full xfade —
                        # clamp to the shorter side.
                        xf = min(XFADE, len(merged), len(nxt))
                        if xf < 2:
                            merged = np.vstack([merged, nxt])
                            continue
                        fo = np.linspace(1.0, 0.0, xf)[:, None]
                        fi = np.linspace(0.0, 1.0, xf)[:, None]
                        # Blend: keep all of `merged` up to the seam, then overlay
                        merged[-xf:] = merged[-xf:] * fo + nxt[:xf] * fi
                        merged = np.vstack([merged, nxt[xf:]])

                    # Rebuild out_arr including pre-roll
                    if preroll_chunks:
                        pr_arr = np.vstack(preroll_chunks).astype(np.float64)
                        out_arr = np.vstack([pr_arr, merged])
                    else:
                        out_arr = merged

                    log_render.info(f"Applied {XFADE}-sample crossfades at "
                                    f"{len(worker_arrs)-1} chunk boundaries")

            total_dur = len(out_arr) / 44100
            log_render.info(f"Assembled: {len(out_arr)} samples at 44100 Hz "
                            f"({total_dur:.2f}s, pre-roll={preroll:.1f}s)")

            if sample_rate != 44100:
                log_render.info(f"Resampling {44100} Hz → {sample_rate} Hz…")
                from scipy.signal import resample_poly
                import math
                g = math.gcd(sample_rate, 44100)
                out_arr = resample_poly(out_arr, sample_rate // g, 44100 // g, axis=0)
                log_render.info(f"Resampled: {len(out_arr)} samples at {sample_rate} Hz "
                                f"({len(out_arr)/sample_rate:.2f}s)")

            # ── Post-processing ───────────────────────────────────────────────
            if do_dc_block:
                log_render.debug("Applying DC blocking filter")
                from scipy.signal import butter, lfilter
                b_dc, a_dc = butter(1, 10.0 / (sample_rate / 2.0), btype="high")
                out_arr = lfilter(b_dc, a_dc, out_arr, axis=0)

            if do_match_loudness and src_rms is not None:
                out_rms = float(np.sqrt(np.mean(out_arr**2)))
                if out_rms > 1e-6:
                    gain = min(src_rms / out_rms, 10 ** (12/20))
                    out_arr *= gain
                    log_render.debug(f"Loudness match: gain={20*np.log10(gain):.1f}dB")
            elif do_norm:
                peak = np.max(np.abs(out_arr))
                if peak > 0:
                    target = 10 ** (-0.1 / 20)
                    out_arr *= target / peak
                    log_render.debug(f"Normalized: peak {peak:.4f} → {target:.4f}")

            if do_dither and bit_depth < 32:
                log_render.debug(f"Applying TPDF dither for {bit_depth}-bit")
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
                # Proper 24-bit PCM: pack each int32 value as 3 bytes little-endian
                import struct, wave
                data_24_int = np.clip(out_arr * 8388607, -8388608, 8388607).astype(np.int32)
                # Pack as 3-byte little-endian signed integers
                raw_24 = bytearray()
                flat = data_24_int.flatten()
                for s in flat:
                    raw_24 += struct.pack("<i", int(s))[0:3]
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(3)      # 3 bytes = 24-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(bytes(raw_24))
                log_render.debug("Wrote true 24-bit PCM via wave module")
            else:
                data_16 = (out_arr * 32767).astype(np.int16)
                AudioSegment(data_16.tobytes(), frame_rate=sample_rate,
                             sample_width=2, channels=2).export(path, format="wav")

            log_render.info(f"Render complete → {path}")
            self.root.after(0, lambda: self._render_prog_update(100, "done"))

            # Always restore overridden params and orientation
            self.engine.params["barkhausen"]    = saved_bark
            self.engine.params["print_through"] = saved_print
            if want_reverse:
                log_render.debug("Restoring forward orientation after reverse render")
                self.engine.set_reverse(False, force_reset=True)

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

        # Default values matching ctrl() definitions — every preset starts here.
        # This guarantees every control is explicitly set, no stale bleed-through.
        DEFAULTS = {
            "ips_base":       15.0,  "motor_health":  0.0,   "motor_drag":    0.0,
            "motor_boost":    0.0,   "wow_dep":        0.0,   "flutter_dep":   0.0,
            "scrape_flutter": 0.0,   "tension_load":   0.0,   "dropout_rate":  0.0,
            "drive":          1.2,   "bias":           1.0,   "replay_diff":   0.3,
            "asperities":     0.0,   "barkhausen":     0.0,   "crosstalk":     0.0,
            "print_through":  0.0,   "demagnetization":0.0,   "oxide_shedding":0.0,
            "hiss": 0.000000,   "hiss_color":     0.0,   "mains_hum":     0.0,
            "cutoff_base":    22000, "head_bump":      0.0,   "azimuth_drift": 0.0,
            "sticky_shed":    0.0,
        }

        # Hiss reference: measured noise floor → hiss param
        #   −72 dBFS ≈ 0.000025  (finest studio)
        #   −68 dBFS ≈ 0.000040  (good studio 30ips)
        #   −62 dBFS ≈ 0.000079  (studio 15ips)
        #   −56 dBFS ≈ 0.000158  (prosumer 7.5ips)
        #   −52 dBFS ≈ 0.000251  (chrome cassette)
        #   −48 dBFS ≈ 0.000398  (type I cassette)
        #   −44 dBFS ≈ 0.000631  (Betamax / VHS good)
        #   −40 dBFS ≈ 0.001000  (VHS worn)
        #   −36 dBFS ≈ 0.001585  (damaged)

        p = {
            # ── Studio / Pro ──────────────────────────────────────────────────
            "Ampex 456 (30ips)": {
                "ips_base": 30.0, "motor_health": 0.02, "wow_dep": 0.008,
                "flutter_dep": 0.005, "scrape_flutter": 0.04,
                "drive": 1.05, "bias": 1.0,
                "hiss": 0.000020, "hiss_color": 0.1, "cutoff_base": 22000,
                "head_bump": 0.35, "print_through": 0.025, "replay_diff": 0.32,
                "mains_hum": 0.000008, "barkhausen": 0.005, "asperities": 0.008,
            },
            "Ampex 456 (15ips)": {
                "ips_base": 15.0, "motor_health": 0.05, "wow_dep": 0.02,
                "flutter_dep": 0.008, "scrape_flutter": 0.05,
                "drive": 1.1, "bias": 1.0,
                "hiss": 0.000039, "hiss_color": 0.15, "cutoff_base": 20000,
                "head_bump": 0.55, "print_through": 0.018, "replay_diff": 0.3,
                "mains_hum": 0.000012, "barkhausen": 0.008, "asperities": 0.012,
            },
            "Studer A820 (30ips)": {
                "ips_base": 30.0, "motor_health": 0.01, "wow_dep": 0.005,
                "flutter_dep": 0.003, "scrape_flutter": 0.03,
                "drive": 1.02, "bias": 1.0,
                "hiss": 0.000013, "hiss_color": 0.08, "cutoff_base": 22000,
                "head_bump": 0.25, "print_through": 0.015, "replay_diff": 0.35,
                "mains_hum": 0.000005, "barkhausen": 0.003, "asperities": 0.005,
            },
            "Studer A820 (15ips)": {
                "ips_base": 15.0, "motor_health": 0.03, "wow_dep": 0.012,
                "flutter_dep": 0.005, "scrape_flutter": 0.04,
                "drive": 1.06, "bias": 1.0,
                "hiss": 0.000025, "hiss_color": 0.1, "cutoff_base": 21000,
                "head_bump": 0.45, "print_through": 0.012, "replay_diff": 0.32,
                "mains_hum": 0.000008, "barkhausen": 0.005, "asperities": 0.008,
            },
            "Otari MTR-90 (30ips)": {
                "ips_base": 30.0, "motor_health": 0.015, "wow_dep": 0.006,
                "flutter_dep": 0.004, "scrape_flutter": 0.035,
                "drive": 1.04, "bias": 1.0,
                "hiss": 0.000016, "hiss_color": 0.09, "cutoff_base": 22000,
                "head_bump": 0.3, "print_through": 0.018, "replay_diff": 0.33,
                "mains_hum": 0.000006, "barkhausen": 0.004, "asperities": 0.006,
            },
            "Otari MTR-90 (15ips)": {
                "ips_base": 15.0, "motor_health": 0.04, "wow_dep": 0.015,
                "flutter_dep": 0.006, "scrape_flutter": 0.045,
                "drive": 1.08, "bias": 1.0,
                "hiss": 0.000031, "hiss_color": 0.12, "cutoff_base": 20500,
                "head_bump": 0.5, "print_through": 0.014, "replay_diff": 0.31,
                "mains_hum": 0.000010, "barkhausen": 0.006, "asperities": 0.009,
            },
            "MCI JH-24 (30ips)": {
                # MCI slightly warmer than Studer, transformer coloration
                "ips_base": 30.0, "motor_health": 0.025, "wow_dep": 0.01,
                "flutter_dep": 0.006, "scrape_flutter": 0.05,
                "drive": 1.08, "bias": 1.0,
                "hiss": 0.000020, "hiss_color": 0.2, "cutoff_base": 21000,
                "head_bump": 0.45, "print_through": 0.022, "replay_diff": 0.28,
                "mains_hum": 0.000015, "barkhausen": 0.007, "asperities": 0.01,
            },
            "Scotch 226 (7.5ips)": {
                "ips_base": 7.5, "motor_health": 0.2, "wow_dep": 0.08,
                "flutter_dep": 0.02, "scrape_flutter": 0.08,
                "drive": 1.3, "bias": 0.95,
                "hiss": 0.000079, "hiss_color": 0.25, "cutoff_base": 16000,
                "head_bump": 0.85, "print_through": 0.03, "replay_diff": 0.28,
                "mains_hum": 0.000020, "barkhausen": 0.015, "asperities": 0.02,
            },
            "Revox B77 (7.5ips)": {
                "ips_base": 7.5, "motor_health": 0.15, "wow_dep": 0.06,
                "flutter_dep": 0.015, "scrape_flutter": 0.07,
                "drive": 1.25, "bias": 0.97,
                "hiss": 0.000063, "hiss_color": 0.2, "cutoff_base": 17000,
                "head_bump": 0.75, "print_through": 0.025, "replay_diff": 0.29,
                "mains_hum": 0.000018, "barkhausen": 0.012, "asperities": 0.016,
            },
            "Revox B77 (3.75ips)": {
                "ips_base": 3.75, "motor_health": 0.4, "wow_dep": 0.25,
                "flutter_dep": 0.04, "scrape_flutter": 0.12,
                "drive": 1.6, "bias": 0.9,
                "hiss": 0.000158, "hiss_color": 0.35, "cutoff_base": 12000,
                "head_bump": 1.2, "print_through": 0.04, "replay_diff": 0.24,
                "mains_hum": 0.000025, "barkhausen": 0.02, "asperities": 0.028,
            },
            # ── Consumer reel ─────────────────────────────────────────────────
            "BASF LH Super (7.5ips)": {
                "ips_base": 7.5, "motor_health": 0.4, "wow_dep": 0.15,
                "flutter_dep": 0.03, "scrape_flutter": 0.09,
                "drive": 1.4, "bias": 0.9,
                "hiss": 0.000100, "hiss_color": 0.3, "cutoff_base": 14000,
                "head_bump": 1.0, "print_through": 0.035, "replay_diff": 0.25,
                "mains_hum": 0.000025, "barkhausen": 0.02, "asperities": 0.025,
                "crosstalk": 0.04,
            },
            "Maxell UD (7.5ips)": {
                "ips_base": 7.5, "motor_health": 0.35, "wow_dep": 0.12,
                "flutter_dep": 0.025, "scrape_flutter": 0.08,
                "drive": 1.35, "bias": 0.92,
                "hiss": 0.000089, "hiss_color": 0.28, "cutoff_base": 15000,
                "head_bump": 0.9, "print_through": 0.028, "replay_diff": 0.26,
                "mains_hum": 0.000020, "barkhausen": 0.018, "asperities": 0.022,
                "crosstalk": 0.03,
            },
            "Tascam 38 (7.5ips)": {
                # 8-track 1/2" consumer studio — noticeably noisier than pro 2"
                "ips_base": 7.5, "motor_health": 0.5, "wow_dep": 0.2,
                "flutter_dep": 0.04, "scrape_flutter": 0.1,
                "drive": 1.5, "bias": 0.92,
                "hiss": 0.000125, "hiss_color": 0.32, "cutoff_base": 14500,
                "head_bump": 1.1, "print_through": 0.03, "replay_diff": 0.26,
                "crosstalk": 0.07, "mains_hum": 0.000030,
                "barkhausen": 0.022, "asperities": 0.028,
            },
            # ── Multitrack ────────────────────────────────────────────────────
            "4-Track Portastudio (1.875ips)": {
                # Tascam Portastudio on Type I at slow speed — the classic lo-fi sound
                "ips_base": 1.875, "motor_health": 1.2, "wow_dep": 2.0,
                "flutter_dep": 0.18, "scrape_flutter": 0.25,
                "drive": 2.8, "bias": 0.82,
                "hiss": 0.000316, "hiss_color": 0.6, "cutoff_base": 10000,
                "head_bump": 2.0, "replay_diff": 0.18, "crosstalk": 0.22,
                "mains_hum": 0.000040, "barkhausen": 0.05, "asperities": 0.06,
                "tension_load": 0.1,
            },
            "8-Track Cartridge": {
                # 8-track at 3.75 IPS — continuous loop, head bump from cart pressure
                "ips_base": 3.75, "motor_health": 1.5, "wow_dep": 1.8,
                "flutter_dep": 0.15, "scrape_flutter": 0.28,
                "drive": 2.5, "bias": 0.85,
                "hiss": 0.000251, "hiss_color": 0.55, "cutoff_base": 9000,
                "head_bump": 2.5, "replay_diff": 0.17, "crosstalk": 0.3,
                "mains_hum": 0.000035, "tension_load": 0.15,
                "barkhausen": 0.04, "asperities": 0.055,
            },
            # ── Cassette ──────────────────────────────────────────────────────
            "Type I (Fe2O3) Normal": {
                "ips_base": 1.875, "motor_health": 1.0, "wow_dep": 1.5,
                "flutter_dep": 0.12, "scrape_flutter": 0.18,
                "drive": 2.2, "bias": 0.85,
                "hiss": 0.000199, "hiss_color": 0.5, "cutoff_base": 12500,
                "head_bump": 1.8, "replay_diff": 0.2,
                "mains_hum": 0.000030, "barkhausen": 0.04, "asperities": 0.05,
                "crosstalk": 0.18, "tension_load": 0.08,
            },
            "Type II Chrome (CrO2)": {
                "ips_base": 1.875, "motor_health": 0.7, "wow_dep": 1.0,
                "flutter_dep": 0.09, "scrape_flutter": 0.14,
                "drive": 1.8, "bias": 1.35,
                "hiss": 0.000125, "hiss_color": 0.35, "cutoff_base": 15000,
                "head_bump": 1.3, "replay_diff": 0.25,
                "mains_hum": 0.000022, "barkhausen": 0.025, "asperities": 0.03,
                "crosstalk": 0.1, "tension_load": 0.05,
            },
            "Type IV Metal": {
                "ips_base": 1.875, "motor_health": 0.5, "wow_dep": 0.7,
                "flutter_dep": 0.06, "scrape_flutter": 0.10,
                "drive": 1.5, "bias": 1.7,
                "hiss": 0.000079, "hiss_color": 0.2, "cutoff_base": 18000,
                "head_bump": 0.9, "replay_diff": 0.3,
                "mains_hum": 0.000015, "barkhausen": 0.015, "asperities": 0.018,
                "crosstalk": 0.06, "tension_load": 0.03,
            },
            "Dolby B (Type I)": {
                "ips_base": 1.875, "motor_health": 0.9, "wow_dep": 1.2,
                "flutter_dep": 0.10, "scrape_flutter": 0.15,
                "drive": 2.0, "bias": 0.88,
                "hiss": 0.000063, "hiss_color": 0.2, "cutoff_base": 14000,
                "head_bump": 1.6, "replay_diff": 0.22,
                "mains_hum": 0.000025, "barkhausen": 0.03, "asperities": 0.035,
                "crosstalk": 0.15,
            },
            "Dolby C (Type II)": {
                # Dolby C ≈ 20dB NR — noticeably quieter than Dolby B
                "ips_base": 1.875, "motor_health": 0.65, "wow_dep": 0.9,
                "flutter_dep": 0.08, "scrape_flutter": 0.12,
                "drive": 1.75, "bias": 1.35,
                "hiss": 0.000031, "hiss_color": 0.15, "cutoff_base": 15500,
                "head_bump": 1.2, "replay_diff": 0.24,
                "mains_hum": 0.000018, "barkhausen": 0.02, "asperities": 0.025,
                "crosstalk": 0.08, "tension_load": 0.04,
            },
            "Lo-Fi Bedroom (Type I)": {
                # Cheap deck, no alignment, generic off-brand tape
                "ips_base": 1.875, "motor_health": 2.0, "wow_dep": 3.0,
                "flutter_dep": 0.25, "scrape_flutter": 0.35,
                "drive": 3.5, "bias": 0.75,
                "hiss": 0.000316, "hiss_color": 0.7, "cutoff_base": 9000,
                "head_bump": 2.5, "replay_diff": 0.15,
                "mains_hum": 0.000060, "barkhausen": 0.06, "asperities": 0.08,
                "crosstalk": 0.25, "tension_load": 0.15, "azimuth_drift": 0.15,
            },
            # ── Video / Broadcast ─────────────────────────────────────────────
            "VHS Linear Audio": {
                "ips_base": 1.3125, "motor_health": 1.8, "wow_dep": 2.5,
                "flutter_dep": 0.2, "scrape_flutter": 0.3,
                "drive": 2.5, "bias": 0.8,
                "hiss": 0.000397, "hiss_color": 0.65, "cutoff_base": 8000,
                "head_bump": 2.2, "replay_diff": 0.15,
                "mains_hum": 0.000100, "crosstalk": 0.35, "asperities": 0.07,
                "tension_load": 0.12,
            },
            "Betamax Audio": {
                "ips_base": 1.873, "motor_health": 1.4, "wow_dep": 1.8,
                "flutter_dep": 0.15, "scrape_flutter": 0.22,
                "drive": 2.2, "bias": 0.82,
                "hiss": 0.000316, "hiss_color": 0.55, "cutoff_base": 9500,
                "head_bump": 2.0, "replay_diff": 0.17,
                "mains_hum": 0.000080, "crosstalk": 0.28, "asperities": 0.06,
                "tension_load": 0.10,
            },
            "U-Matic Low Band": {
                "ips_base": 3.75, "motor_health": 0.9, "wow_dep": 1.0,
                "flutter_dep": 0.1, "scrape_flutter": 0.18,
                "drive": 2.0, "bias": 0.85,
                "hiss": 0.000251, "hiss_color": 0.5, "cutoff_base": 10000,
                "head_bump": 1.8, "replay_diff": 0.2,
                "mains_hum": 0.000060, "crosstalk": 0.22, "asperities": 0.05,
                "tension_load": 0.08, "print_through": 0.01,
            },
            # ── Damaged / Warped ──────────────────────────────────────────────
            "Sticky Shed Syndrome": {
                "ips_base": 7.5, "sticky_shed": 0.9, "motor_health": 4.0,
                "wow_dep": 8.0, "flutter_dep": 0.5, "scrape_flutter": 0.7,
                "drive": 3.0, "hiss": 0.000600, "hiss_color": 0.7,
                "cutoff_base": 3500, "head_bump": 2.5,
                "dropout_rate": 0.5, "oxide_shedding": 0.7, "tension_load": 0.25,
            },
            "Baked Tape (Post-Oven)": {
                "ips_base": 7.5, "sticky_shed": 0.25, "motor_health": 1.0,
                "wow_dep": 1.5, "flutter_dep": 0.08, "scrape_flutter": 0.15,
                "drive": 2.0, "hiss": 0.000250, "hiss_color": 0.4,
                "cutoff_base": 10000, "head_bump": 1.2,
                "dropout_rate": 0.08, "oxide_shedding": 0.15,
                "tension_load": 0.08, "demagnetization": 0.15,
            },
            "Mouldy Attic Find": {
                "ips_base": 7.5, "motor_health": 2.5, "wow_dep": 4.0,
                "flutter_dep": 0.3, "scrape_flutter": 0.45,
                "drive": 3.5, "hiss": 0.000450, "hiss_color": 0.75,
                "cutoff_base": 5000, "head_bump": 2.5,
                "print_through": 0.08, "demagnetization": 0.55,
                "dropout_rate": 0.35, "oxide_shedding": 0.4,
                "asperities": 0.12, "barkhausen": 0.06,
            },
            "Dropout Disaster": {
                "ips_base": 15.0, "motor_health": 0.5, "wow_dep": 0.5,
                "drive": 1.8, "hiss": 0.000100, "hiss_color": 0.3,
                "cutoff_base": 14000, "head_bump": 0.8,
                "dropout_rate": 0.85, "oxide_shedding": 0.6,
                "asperities": 0.08, "demagnetization": 0.1,
            },
            "Heavily Demagnetised": {
                "ips_base": 15.0, "motor_health": 0.4, "wow_dep": 0.3,
                "drive": 1.5, "hiss": 0.000150, "hiss_color": 0.5,
                "cutoff_base": 6000, "head_bump": 1.2,
                "demagnetization": 0.85, "print_through": 0.04, "replay_diff": 0.15,
            },
            "Warped & Fighting Motors": {
                "ips_base": 15.0, "motor_health": 5.0, "motor_drag": 0.6,
                "motor_boost": 0.6, "wow_dep": 12.0, "flutter_dep": 0.8,
                "tension_load": 0.4, "drive": 2.0,
                "hiss": 0.000199, "cutoff_base": 12000, "head_bump": 1.0,
                "dropout_rate": 0.1, "scrape_flutter": 0.3,
            },
            "Chewed Tape": {
                "ips_base": 7.5, "motor_health": 3.0, "wow_dep": 6.0,
                "flutter_dep": 0.6, "scrape_flutter": 0.8,
                "drive": 4.0, "hiss": 0.000600, "hiss_color": 0.8,
                "cutoff_base": 4000, "head_bump": 3.0,
                "dropout_rate": 0.7, "oxide_shedding": 0.8,
                "asperities": 0.2, "demagnetization": 0.3, "tension_load": 0.35,
            },
            "Print-Through Ghost": {
                "ips_base": 15.0, "motor_health": 0.08, "wow_dep": 0.04,
                "drive": 1.1, "hiss": 0.000035, "hiss_color": 0.1,
                "cutoff_base": 19000, "head_bump": 0.4,
                "print_through": 0.09, "replay_diff": 0.3,
                "demagnetization": 0.05, "barkhausen": 0.006,
            },
            "Stretched Tape": {
                # Physically elongated tape — permanent pitch instability,
                # elastic resonance, uneven thickness modulation
                "ips_base": 15.0, "motor_health": 1.5, "wow_dep": 5.0,
                "flutter_dep": 0.35, "scrape_flutter": 0.2, "tension_load": 0.3,
                "drive": 2.0, "hiss": 0.000158, "hiss_color": 0.4,
                "cutoff_base": 11000, "head_bump": 1.5,
                "dropout_rate": 0.15, "oxide_shedding": 0.2,
                "demagnetization": 0.1,
            },
            "Heat Warped": {
                # Left in a hot car — hub deformed, layer separation,
                # extreme wow from uneven winding
                "ips_base": 7.5, "motor_health": 3.5, "wow_dep": 10.0,
                "flutter_dep": 0.45, "scrape_flutter": 0.5, "tension_load": 0.35,
                "drive": 2.5, "hiss": 0.000315, "hiss_color": 0.6,
                "cutoff_base": 7000, "head_bump": 2.0,
                "dropout_rate": 0.25, "oxide_shedding": 0.35,
                "demagnetization": 0.25, "sticky_shed": 0.3,
            },
            "Spliced Archive": {
                # Professional reel with many edit splices — tiny level bumps
                # and phase glitches at each join, some print-through
                "ips_base": 15.0, "motor_health": 0.15, "wow_dep": 0.15,
                "flutter_dep": 0.012, "scrape_flutter": 0.06,
                "drive": 1.15, "hiss": 0.000050, "hiss_color": 0.15,
                "cutoff_base": 18000, "head_bump": 0.5,
                "print_through": 0.045, "replay_diff": 0.3,
                "dropout_rate": 0.05, "asperities": 0.03,
                "mains_hum": 0.000010,
            },
            "Soviet ORWO Copy": {
                # ORWO RN55 / UN54 — East German oxide, somewhat harsh HF,
                # noisier than Western equivalents at similar speeds
                "ips_base": 15.0, "motor_health": 0.6, "wow_dep": 0.4,
                "flutter_dep": 0.05, "scrape_flutter": 0.12,
                "drive": 1.6, "bias": 0.88,
                "hiss": 0.000125, "hiss_color": 0.4, "cutoff_base": 15000,
                "head_bump": 0.9, "print_through": 0.04, "replay_diff": 0.27,
                "mains_hum": 0.000045, "barkhausen": 0.03, "asperities": 0.04,
                "crosstalk": 0.05,
            },
            # ── Radio / Broadcast ─────────────────────────────────────────────
            "BBC Radiophonic (7.5ips)": {
                # BBC Radiophonic Workshop — Ampex/EMI machines, precise alignment,
                # heavy tape manipulation, some splicing artefacts
                "ips_base": 7.5, "motor_health": 0.12, "wow_dep": 0.05,
                "flutter_dep": 0.012, "scrape_flutter": 0.07,
                "drive": 1.2, "bias": 0.98,
                "hiss": 0.000079, "hiss_color": 0.2, "cutoff_base": 16000,
                "head_bump": 0.7, "print_through": 0.03, "replay_diff": 0.3,
                "mains_hum": 0.000020, "barkhausen": 0.01, "asperities": 0.015,
                "dropout_rate": 0.03,
            },
            "AM Radio Dub": {
                # Tape of a radio broadcast — AM bandwidth, 50Hz hum from receiver,
                # slight wow from the consumer deck used for recording
                "ips_base": 3.75, "motor_health": 0.6, "wow_dep": 0.4,
                "flutter_dep": 0.04, "scrape_flutter": 0.1,
                "drive": 2.2, "bias": 0.88,
                "hiss": 0.000200, "hiss_color": 0.55, "cutoff_base": 5000,
                "head_bump": 1.2, "replay_diff": 0.22,
                "mains_hum": 0.000080, "crosstalk": 0.12, "asperities": 0.04,
            },
            # ── Lo-Fi / Special ───────────────────────────────────────────────
            "Ghetto Blaster": {
                # High-bias chrome in a cheap portable — thin plastic head,
                # worn pinch roller, drift from batteries running low
                "ips_base": 1.875, "motor_health": 2.5, "wow_dep": 3.5,
                "flutter_dep": 0.3, "scrape_flutter": 0.4,
                "drive": 3.0, "bias": 1.1,
                "hiss": 0.000398, "hiss_color": 0.65, "cutoff_base": 10000,
                "head_bump": 2.2, "replay_diff": 0.17,
                "mains_hum": 0.0, "crosstalk": 0.28, "asperities": 0.07,
                "tension_load": 0.2, "azimuth_drift": 0.2,
            },
            "Answering Machine": {
                # Micro-cassette at 1.2 IPS — tiny head, extreme HF loss,
                # massive wow from the tiny motor
                "ips_base": 1.2, "motor_health": 3.0, "wow_dep": 5.0,
                "flutter_dep": 0.5, "scrape_flutter": 0.55,
                "drive": 4.0, "bias": 0.75,
                "hiss": 0.000631, "hiss_color": 0.75, "cutoff_base": 5500,
                "head_bump": 3.0, "replay_diff": 0.12,
                "mains_hum": 0.0, "crosstalk": 0.4, "asperities": 0.1,
                "azimuth_drift": 0.3, "tension_load": 0.25,
            },
            "Handheld Dictaphone": {
                # Compact cassette at slow speed, AGC compression artefacts,
                # mono head on stereo tape
                "ips_base": 0.9375, "motor_health": 2.8, "wow_dep": 4.5,
                "flutter_dep": 0.45, "scrape_flutter": 0.6,
                "drive": 3.5, "bias": 0.78,
                "hiss": 0.000794, "hiss_color": 0.8, "cutoff_base": 4500,
                "head_bump": 2.8, "replay_diff": 0.1,
                "mains_hum": 0.0, "crosstalk": 0.45, "asperities": 0.12,
                "azimuth_drift": 0.35,
            },
            "Toy Piano Recording": {
                # Cheap toy recorder from 1970s — elastic band drive,
                # completely random speed, lo-fi magic
                "ips_base": 1.875, "motor_health": 8.0, "wow_dep": 18.0,
                "flutter_dep": 1.5, "scrape_flutter": 0.9,
                "drive": 5.0, "bias": 0.65,
                "hiss": 0.001000, "hiss_color": 0.85, "cutoff_base": 3000,
                "head_bump": 4.0, "replay_diff": 0.1,
                "mains_hum": 0.0, "crosstalk": 0.5, "asperities": 0.18,
                "dropout_rate": 0.15, "tension_load": 0.4,
            },
            # ── More Damaged ──────────────────────────────────────────────────
            "Tsunami Flood Tape": {
                # Water-damaged, dried without cleaning — oxide loosened,
                # erratic dropout storms, residue on heads
                "ips_base": 7.5, "motor_health": 3.5, "wow_dep": 5.0,
                "flutter_dep": 0.4, "scrape_flutter": 0.65,
                "drive": 3.5, "hiss": 0.000700, "hiss_color": 0.7,
                "cutoff_base": 4500, "head_bump": 3.0,
                "dropout_rate": 0.65, "oxide_shedding": 0.75,
                "demagnetization": 0.4, "asperities": 0.18,
                "sticky_shed": 0.6, "tension_load": 0.3,
            },
            "Fire-Damaged Archive": {
                # Heat warped, partial oxide sublimation, magnetic domains
                # partially randomised — thin wispy signal, constant dropout
                "ips_base": 7.5, "motor_health": 4.0, "wow_dep": 7.0,
                "flutter_dep": 0.55, "scrape_flutter": 0.75,
                "drive": 5.0, "hiss": 0.000630, "hiss_color": 0.8,
                "cutoff_base": 3000, "head_bump": 2.5,
                "dropout_rate": 0.75, "oxide_shedding": 0.9,
                "demagnetization": 0.7, "asperities": 0.25,
                "sticky_shed": 0.8, "tension_load": 0.45,
            },
            "Played 1000 Times": {
                # Heavily worn — oxide thinned from head friction, print-through
                # from years of storage, loss of high coercivity particles
                "ips_base": 7.5, "motor_health": 0.8, "wow_dep": 0.8,
                "flutter_dep": 0.1, "scrape_flutter": 0.2,
                "drive": 2.5, "hiss": 0.000316, "hiss_color": 0.45,
                "cutoff_base": 8000, "head_bump": 1.5,
                "dropout_rate": 0.12, "oxide_shedding": 0.45,
                "demagnetization": 0.5, "print_through": 0.06,
                "asperities": 0.09, "barkhausen": 0.04,
            },
        }

        oxide_map = {
            "Type II Chrome (CrO2)":  "CrO2",
            "Type IV Metal":          "Metal",
            "Dolby B (Type I)":       "Fe2O3",
            "Dolby C (Type II)":      "CrO2",
        }
        self.oxide_var.set(oxide_map.get(name, "Fe2O3"))

        if name in p:
            # Suppress per-slider sync() calls — each controls[k].set() would otherwise
            # call sync() and push a half-applied param set into the live engine, causing
            # up to 25 intermediate flickers and a race with the audio callback thread.
            self._syncing = True
            try:
                merged = dict(DEFAULTS)
                merged.update(p[name])
                for k, v in merged.items():
                    if k in self.controls:
                        self.controls[k].set(v)
            finally:
                self._syncing = False

            # Build the complete new params dict once, including oxide, then swap it
            # atomically under the engine lock so the audio thread never sees a
            # half-written state.
            new_params = {k: v.get() for k, v in self.controls.items()}
            new_params["oxide_type"] = self.oxide_var.get()

            with self.engine.lock:
                self.engine.params = new_params
                # Trigger a short fade-in ramp on the engine output.
                # This smooths over any IIR-state or parameter-jump transients
                # on the first blocks after the preset change without needing
                # to zero any cross-block state (zeroing _last_proc etc. actually
                # makes the pop worse by forcing a cold-start spike).
                self.engine._fade_in = 512

    # ── Telemetry
    # ── Telemetry ────────────────────────────────────────────────────────────────
    def _on_close(self):
        """Clean shutdown — set flag, stop audio, exit mainloop, destroy."""
        log_ui.info("Window close requested — shutting down")
        self._closing = True
        self.engine.is_playing = False
        self.is_rewinding      = False
        self.is_ffing          = False
        # quit() exits mainloop() without touching widgets, so no
        # in-flight after() callbacks can fire against half-destroyed widgets.
        self.root.quit()
        self.root.destroy()

    def update_telemetry(self):
        if self._closing:
            return

        # ── Reel position ──────────────────────────────────────────────────
        if self.engine.total_samples > 0:
            ph           = self.engine.play_head
            ts           = self.engine.total_samples
            fwd_progress = float(np.clip(
                (1.0 - ph / ts) if self.engine.is_reversed else (ph / ts), 0, 1))
            filled    = int(fwd_progress * 8)
            supply_s  = "●" * (8 - filled) + "○" * filled
            takeup_s  = "○" * (8 - filled) + "●" * filled
            elapsed_s = ph / 44100.0
            remain_s  = (ts - ph) / 44100.0
            pos_str   = f"{int(elapsed_s//60):02d}:{elapsed_s%60:05.2f}"
            rem_str   = f"{int(remain_s//60):02d}:{remain_s%60:05.2f}"
        else:
            supply_s = "●●●●●●●●"
            takeup_s = "○○○○○○○○"
            pos_str  = "00:00.00"
            rem_str  = "00:00.00"
            remain_s = 0.0

        transport = self.engine.transport
        inst_spd  = getattr(transport, 'last_instant_speed', 1.0)
        conflict  = getattr(transport, 'last_conflict', 0.0)
        drop_n    = getattr(transport, 'dropout_count', 0)
        drop_t    = getattr(transport, 'last_dropout_time', -999.0)
        cur_time  = self.engine.current_time
        ips_set   = self.controls["ips_base"].get()

        # ── Read all control values ────────────────────────────────────────
        health    = self.controls["motor_health"].get()
        drag      = self.controls["motor_drag"].get()
        boost     = self.controls["motor_boost"].get()
        shed      = self.controls["sticky_shed"].get()
        demag     = self.controls["demagnetization"].get()
        drive     = self.controls["drive"].get()
        bias_v    = self.controls["bias"].get()
        hiss_v    = self.controls["hiss"].get()
        dropout_r = self.controls["dropout_rate"].get()
        wow       = self.controls["wow_dep"].get()
        flutter   = self.controls["flutter_dep"].get()
        scrape    = self.controls["scrape_flutter"].get()
        print_v   = self.controls["print_through"].get()
        bark      = self.controls["barkhausen"].get()
        asper     = self.controls["asperities"].get()
        crosstk   = self.controls["crosstalk"].get()
        oxide_n   = self.oxide_var.get()

        # ── Simulated voltage noise for PSU bar ───────────────────────────
        if self.engine.is_playing:
            if not hasattr(self, '_volt_state'):
                self._volt_state = 0.0
            noise = np.random.normal(0, health * 0.008 + conflict * 0.02)
            self._volt_state += noise - self._volt_state * 0.05
            volt_dev = float(np.clip(self._volt_state, -1.0, 1.0))
            volt_v   = 12.0 + volt_dev * (0.5 + health * 0.4)
        else:
            self._volt_state = 0.0
            volt_dev = 0.0
            volt_v   = 12.0

        # ── PSU bar ────────────────────────────────────────────────────────
        BAR_W  = 20
        centre = BAR_W // 2
        fill_amt = int(abs(volt_dev) * centre)
        bar_chars = list("·" * BAR_W)
        if volt_dev >= 0:
            for i in range(centre, min(BAR_W, centre + fill_amt)): bar_chars[i] = "█"
        else:
            for i in range(max(0, centre - fill_amt), centre):     bar_chars[i] = "█"
        bar_chars[centre] = "│"
        bar_str = "".join(bar_chars)

        abs_dev = abs(volt_dev)
        if abs_dev < 0.15:   volt_fg, volt_state = C["grey"],   "STABLE"
        elif abs_dev < 0.40: volt_fg, volt_state = C["amber"],  "DRIFTING"
        elif abs_dev < 0.70: volt_fg, volt_state = C["orange"], "UNSTABLE"
        else:                volt_fg, volt_state = C["red"],    "CRITICAL"

        # ── Speed deviation ────────────────────────────────────────────────
        spd_dev_pct = (inst_spd - 1.0) * 100.0

        # ── Dropout countdown ──────────────────────────────────────────────
        if drop_t > 0 and self.engine.is_playing:
            since = cur_time - drop_t
            drop_last_str = f"{since:4.1f}s" if since < 9.9 else " >9s"
        else:
            drop_last_str = "  --"

        # ── Oxide / saturation state ───────────────────────────────────────
        # Estimate saturation from drive parameter relative to oxide ceiling
        oxide_ceilings = {"Fe2O3": 1.0, "CrO2": 1.1, "Metal": 1.2, "FeCo": 1.25}
        ceiling = oxide_ceilings.get(oxide_n, 1.0)
        sat_ratio = drive / (ceiling * 4.0)   # normalised 0-1 across slider range
        if sat_ratio < 0.15:    sat_str, sat_fg = "CLEAN",    C["grey"]
        elif sat_ratio < 0.30:  sat_str, sat_fg = "WARM",     C["amber"]
        elif sat_ratio < 0.55:  sat_str, sat_fg = "DRIVEN",   C["orange"]
        elif sat_ratio < 0.75:  sat_str, sat_fg = "SATURATED",C["orange"]
        else:                   sat_str, sat_fg = "OVERDRIVEN",C["red"]

        # ── Bias state ────────────────────────────────────────────────────
        if abs(bias_v - 1.0) < 0.08:   bias_str, bias_fg = "NOMINAL",    C["grey"]
        elif bias_v < 1.0:              bias_str, bias_fg = "UNDERBIAS",  C["orange"]
        elif bias_v < 1.5:              bias_str, bias_fg = "OVERBIAS",   C["amber"]
        else:                           bias_str, bias_fg = "HIGH BIAS",  C["red"]

        # ── Noise floor estimate ───────────────────────────────────────────
        speed_factor = ips_set / 15.0
        eff_noise = hiss_v / max(speed_factor ** 0.5, 0.01)
        if eff_noise < 0.0001:  noise_str = "< -80dB"
        else:                   noise_str = f"{20*np.log10(eff_noise+1e-10):.0f}dB"

        # ── Simulated signal level meter (peek at play_head region) ───────
        if self.engine.is_playing and self.engine.audio_data is not None:
            ph_i = int(np.clip(self.engine.play_head, 0,
                               self.engine.total_samples - 256))
            chunk = self.engine.audio_data[ph_i:ph_i+256]
            sig_peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
            sig_rms  = float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0
            if sig_peak < 0.001:   sig_str, sig_fg = "SILENCE",  C["grey"]
            elif sig_peak < 0.3:   sig_str, sig_fg = f"{20*np.log10(sig_rms+1e-10):.0f}dBFS", C["grey"]
            elif sig_peak < 0.7:   sig_str, sig_fg = f"{20*np.log10(sig_rms+1e-10):.0f}dBFS", C["amber"]
            elif sig_peak < 0.95:  sig_str, sig_fg = f"{20*np.log10(sig_rms+1e-10):.0f}dBFS", C["orange"]
            else:                  sig_str, sig_fg = f"HOT {20*np.log10(sig_rms+1e-10):.0f}dBFS", C["red"]
        else:
            sig_str, sig_fg = "----", C["grey"]

        # ── Binder / sticky shed ──────────────────────────────────────────
        if shed < 0.05:    shed_str, shed_fg = "INTACT",        C["grey"]
        elif shed < 0.3:   shed_str, shed_fg = "SOFTENING",     C["amber"]
        elif shed < 0.65:  shed_str, shed_fg = "SHEDDING",      C["orange"]
        else:              shed_str, shed_fg = "CRITICAL SHED", C["red"]

        # ── Demagnetisation ───────────────────────────────────────────────
        if demag < 0.05:   demag_str, demag_fg = "CLEAR",       C["grey"]
        elif demag < 0.25: demag_str, demag_fg = "MINOR LOSS",  C["amber"]
        elif demag < 0.60: demag_str, demag_fg = "DEGRADED",    C["orange"]
        else:              demag_str, demag_fg = "SEVERE",       C["red"]

        # ── Head wear (combined) ──────────────────────────────────────────
        wear_score = shed * 0.5 + demag * 0.3 + dropout_r * 0.2
        if wear_score < 0.05:  wear_str, wear_fg = "NOMINAL",      C["grey"]
        elif wear_score < 0.2: wear_str, wear_fg = "MINOR WEAR",   C["amber"]
        elif wear_score < 0.5: wear_str, wear_fg = "DEGRADED",     C["orange"]
        else:                  wear_str, wear_fg = "SEVERE DAMAGE",C["red"]

        # ── Active effects summary ────────────────────────────────────────
        fx = []
        if wow > 0.05:     fx.append(f"WOW:{wow:.2f}")
        if flutter > 0.01: fx.append(f"FLT:{flutter:.2f}")
        if scrape > 0.01:  fx.append(f"SCR:{scrape:.2f}")
        if bark > 0.001:   fx.append(f"BRK:{bark:.3f}")
        if asper > 0.01:   fx.append(f"ASP:{asper:.2f}")
        if crosstk > 0.01: fx.append(f"XTK:{crosstk:.2f}")
        if print_v > 0.01: fx.append(f"PRT:{print_v:.2f}")
        if shed > 0.05:    fx.append("SHED")
        if demag > 0.05:   fx.append("DEMAG")
        if conflict > 0.1: fx.append(f"CONFLICT:{conflict:.2f}")
        fx_str = "  ".join(fx) if fx else "BYPASS"
        fx_fg  = C["red"] if conflict > 0.2 or shed > 0.5 or demag > 0.5 else                  C["orange"] if fx else C["grey"]

        # ── State string ──────────────────────────────────────────────────
        tick  = int(cur_time * 8) % 2
        spin  = ["◆", "◇"]

        if self.is_rewinding:
            state_str = f"{spin[tick]} REWINDING"; state_fg = C["cyan"]
            ips_str   = f"◀◀ {ips_set * 40.0:06.2f}"; ips_fg = C["cyan"]
        elif self.is_ffing:
            state_str = f"{spin[tick]} FAST FWD";  state_fg = C["green"]
            ips_str   = f"▶▶ {ips_set * 40.0:06.2f}"; ips_fg = C["green"]
        elif self.engine.is_playing:
            actual = ips_set * inst_spd
            rev    = self.engine.is_reversed
            arrow  = "◀" if rev else "▶"
            if conflict > 0.1:
                state_str = f"{spin[tick]} FIGHTING"; state_fg = C["red"]
            elif abs(spd_dev_pct) > 8:
                state_str = f"{spin[tick]} UNSTABLE"; state_fg = C["red"]
            elif rev:
                state_str = f"{spin[tick]} REVERSE";  state_fg = C["orange"]
            else:
                state_str = f"{spin[tick]} RUNNING";  state_fg = C["amber"]
            ips_str = f"{arrow} {actual:06.3f}"; ips_fg = C["orange"] if rev else C["amber"]
        else:
            state_str = "  IDLE"; state_fg = C["grey"]
            ips_str   = "▶ 00.000"; ips_fg = C["grey"]

        reel_str = f"SUPPLY: {supply_s}  TAKEUP: {takeup_s}"
        reel_fg  = ips_fg if self.engine.is_playing else C["grey"]

        # ── Apply to widgets ───────────────────────────────────────────────
        self.lbl_sys_state.config(
            text=f"STATE    : {state_str}",  fg=state_fg)
        self.lbl_sys_speed.config(
            text=f"SPEED    : {ips_set * (inst_spd if self.engine.is_playing else 1.0):06.3f} IPS",
            fg=ips_fg if self.engine.is_playing else C["grey"])
        self.lbl_sys_dev.config(
            text=f"DEVIATION: {spd_dev_pct:+.2f}%  CONFLICT:{conflict:.2f}",
            fg=(C["red"] if abs(spd_dev_pct) > 8 or conflict > 0.2
                else C["amber"] if abs(spd_dev_pct) > 2 else C["grey"]))
        self.lbl_sys_pos.config(
            text=f"POSITION : {pos_str}  REMAIN: {rem_str}",
            fg=C["amber"] if self.engine.is_playing else C["grey"])

        self.lbl_volt_bar.config(text=f"VOLT  [{bar_str}]", fg=volt_fg)
        self.lbl_volt_val.config(
            text=f"PSU   : {volt_state:<10s} {volt_v:+.2f}V", fg=volt_fg)

        self.lbl_sig_level.config(
            text=f"SIGNAL   : {sig_str:<8s}  NOISE:{noise_str}", fg=sig_fg)
        self.lbl_oxide.config(
            text=f"OXIDE    : {oxide_n:<6s}  SAT:{sat_str}", fg=sat_fg)
        self.lbl_bias_state.config(
            text=f"BIAS     : {bias_str}  ({bias_v:.2f})", fg=bias_fg)

        self.lbl_drop_count.config(
            text=f"DROPOUTS : #{drop_n:<5d} LAST:{drop_last_str}",
            fg=C["orange"] if drop_n > 0 else C["grey"])
        self.lbl_shed_state.config(text=f"BINDER   : {shed_str}", fg=shed_fg)
        self.lbl_demag_state.config(text=f"DEMAG    : {demag_str}", fg=demag_fg)
        self.lbl_head_temp.config(text=f"HEAD WEAR: {wear_str}", fg=wear_fg)

        self.lbl_fx_chain.config(text=f"FX  {fx_str[:52]}", fg=fx_fg)

        # ── Reel counter (right panel) ─────────────────────────────────────
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
