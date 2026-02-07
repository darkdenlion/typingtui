from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.containers import Container
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, Static
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "")
    hint = "python3 -m pip install -U rich textual"
    print(f"Missing dependency '{missing}'. Install with: {hint}")
    raise SystemExit(1) from exc


# ---------------------------
# Persistence helpers
# ---------------------------

def _default_data_dir() -> Path:
    """
    Local-only score storage:
    - macOS: ~/Library/Application Support/typing-tui
    - Linux: $XDG_DATA_HOME/typing-tui or ~/.local/share/typing-tui
    """
    home = Path.home()
    if sys_platform() == "darwin":
        return home / "Library" / "Application Support" / "typing-tui"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "typing-tui"
    return home / ".local" / "share" / "typing-tui"


def sys_platform() -> str:
    # avoid importing platform for simplicity; os.uname exists on Unix
    try:
        return os.uname().sysname.lower()
    except Exception:
        # fallback (windows etc.)
        return os.name.lower()


SCORES_PATH = _default_data_dir() / "scores.json"
CONFIG_PATH = Path(__file__).resolve().parent / "typing_tui.config.json"


@dataclass
class ScoreEntry:
    ts: str
    seconds: int
    wpm: float
    cwpm: float
    accuracy: float
    words_done: int
    words_correct: int
    words_wrong: int
    mode: str = "words"
    difficulty: str = "normal"


_SCORE_FIELDS = {f.name for f in __import__("dataclasses").fields(ScoreEntry)}


def load_scores() -> List[ScoreEntry]:
    try:
        data = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        out: List[ScoreEntry] = []
        for row in data:
            if "words_correct" not in row or "words_wrong" not in row:
                words_correct = int(row.get("words_done", 0)) - int(row.get("words_wrong", 0))
                row["words_correct"] = max(0, words_correct)
                row["words_wrong"] = int(row.get("words_wrong", 0))
            # Filter to known fields so extra/future keys don't crash
            filtered = {k: v for k, v in row.items() if k in _SCORE_FIELDS}
            out.append(ScoreEntry(**filtered))
        return out
    except FileNotFoundError:
        return []
    except Exception:
        # if file is corrupted, don't crash; start fresh
        return []


def save_scores(scores: List[ScoreEntry]) -> None:
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.write_text(
        json.dumps([asdict(s) for s in scores], indent=2),
        encoding="utf-8",
    )


# ---------------------------
# Word source (offline)
# ---------------------------

FALLBACK_WORDS = [
    "a", "about", "above", "after", "again", "air", "all", "almost", "also", "always",
    "am", "among", "an", "and", "another", "any", "are", "around", "as", "ask",
    "at", "away", "back", "be", "because", "been", "before", "being", "below", "best",
    "between", "big", "both", "but", "by", "call", "came", "can", "car", "case",
    "change", "child", "city", "close", "come", "company", "could", "country", "course", "day",
    "did", "different", "do", "does", "down", "each", "early", "end", "enough", "even",
    "every", "example", "eye", "face", "fact", "family", "far", "feel", "few", "find",
    "first", "for", "found", "from", "full", "get", "give", "go", "good", "great",
    "group", "grow", "had", "hand", "hard", "has", "have", "he", "head", "health",
    "hear", "help", "her", "here", "high", "him", "his", "home", "house", "how",
    "however", "I", "if", "in", "into", "is", "it", "its", "just", "keep",
    "kind", "know", "large", "last", "late", "learn", "left", "life", "like", "line",
    "little", "live", "long", "look", "love", "made", "make", "man", "many", "may",
    "me", "mean", "men", "might", "more", "most", "move", "much", "must", "my",
    "near", "need", "never", "new", "next", "night", "no", "not", "now", "number",
    "of", "off", "often", "old", "on", "once", "one", "only", "or", "other",
    "our", "out", "over", "own", "part", "people", "place", "point", "problem", "program",
    "public", "put", "question", "right", "room", "run", "said", "same", "saw", "say",
    "school", "see", "seem", "set", "she", "should", "show", "since", "small", "so",
    "some", "something", "sound", "still", "study", "such", "system", "take", "tell", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "thing", "think",
    "this", "those", "time", "to", "today", "together", "too", "town", "try", "two",
    "under", "up", "use", "very", "want", "was", "water", "way", "we", "week",
    "well", "went", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "word", "work", "world", "would", "write", "year", "you", "your",
]

QUOTES = [
    "Stay hungry, stay foolish.",
    "Simplicity is the ultimate sophistication.",
    "Make it work, make it right, make it fast.",
    "Programs must be written for people to read, and only incidentally for machines to execute.",
    "The most disastrous thing that you can ever learn is your first programming language.",
    "Small steps every day add up to big results.",
]

CODE_TOKENS = [
    "def", "class", "return", "yield", "async", "await", "lambda", "import", "from",
    "True", "False", "None", "self", "args", "kwargs", "try", "except", "raise",
    "if", "elif", "else", "for", "while", "break", "continue", "with", "as",
    "print()", "len()", "range()", "dict", "list", "set", "tuple", "str", "int",
    "==", "!=", ">=", "<=", "->", "=>", "+=", "-=", "*=", "/=", "::", "//",
    "()", "[]", "{}", "0", "1", "2", "10", "42", "3.14", "0xFF",
    "snake_case", "camelCase", "kebab-case", "api/v1", "node.js", "user_id",
]

KALI_TOOLS = [
    "airbase-ng", "aircrack-ng", "airdecap-ng", "aireplay-ng", "airodump-ng",
    "airmon-ng", "airolib-ng", "airserv-ng", "airtun-ng", "airgraph-ng",
    "amass", "androguard", "apkid", "apktool", "arp-scan",
    "arping", "assetfinder", "autopsy", "baksmali", "beef-xss",
    "bettercap", "binwalk", "bulk-extractor", "bully", "burpsuite",
    "cewl", "chkrootkit", "clamav", "crackmapexec", "crunch",
    "cupp", "dex2jar", "dirb", "dirbuster", "dirsearch",
    "dnsenum", "dnsrecon", "dnsx", "droopescan", "dsniff",
    "enum4linux", "enum4linux-ng", "ettercap", "exiftool", "eyewitness",
    "fcrackzip", "feroxbuster", "fierce", "ffuf", "fping",
    "foremost", "frida", "gau", "gauplus", "gdb",
    "gf", "ghidra", "gobuster", "gowitness", "gospider",
    "gvm", "hakrawler", "hashcat", "hashid", "hcxdumptool",
    "hcxpcaptool", "hcxtools", "hping3", "httrack", "httprint",
    "httpx", "httprobe", "hydra", "ike-scan", "impacket",
    "iw", "jadx", "john", "joomscan", "katana",
    "kismet", "lbd", "ltrace", "maltego", "maskprocessor",
    "masscan", "mdk3", "mdk4", "medusa", "metagoofil",
    "metasploit", "mimikatz", "mitmproxy", "msfconsole", "msfvenom",
    "ncat", "nbtscan", "netcat", "netdiscover", "nikto",
    "nmap", "ncrack", "nuclei", "objection", "openssl",
    "osintgram", "owasp-zap", "paramspider", "patator", "pixiewps",
    "princeprocessor", "radare2", "reaver", "recon-ng", "responder",
    "rfkill", "rpcclient", "scalpel", "searchsploit", "setoolkit",
    "shuffledns", "skipfish", "smbclient", "smbmap", "snort",
    "socat", "spiderfoot", "sqlmap", "sslyze", "statsprocessor",
    "strace", "subfinder", "subjack", "sublist3r", "suricata",
    "tcpdump", "termshark", "tshark", "theharvester", "traceroute",
    "unicornscan", "volatility", "volatility3", "wafw00f", "wash",
    "waybackurls", "whatweb", "wfuzz", "wifite", "wireshark",
    "wpscan", "xplico", "yara", "zmap", "altdns",
    "aquatone", "arjun", "chaos", "dalfox", "dnsgen",
    "dnsprobe", "dnsvalidator", "findomain", "kxss", "massdns",
    "puredns", "qsreplace", "subzy", "unfurl", "urlfinder",
    "urldedupe", "waybackrobots", "xsstrike"
]

EXTRA_TOKENS = [
    "v2", "v3", "alpha-1", "beta-2", "x86", "x64", "3.14", "99", "config.json",
    "path/to", "hello!", "wow?", "done.", "ready,", "commit;", "push:",
]

DIFFICULTIES = ["easy", "normal", "hard"]
MODES = ["words", "quotes", "code tokens", "hacker"]

THEMES: Dict[str, Dict[str, str]] = {
    "slate": {
        "screen_bg": "transparent",
        "card_bg": "#111827",
        "stats_bg": "#0f172a",
        "prompt_bg": "#0b1220",
        "input_bg": "#0b0f14",
        "border": "#1f2937",
        "title": "#e5e7eb",
        "muted": "#64748b",
        "hint": "#93c5fd",
        "ok": "#a7f3d0",
        "bad": "#fca5a5",
        "active_ok": "#86efac",
        "active_bad": "#fb7185",
        "active_fg": "#e5e7eb",
        "upcoming": "#cbd5e1",
        "flash": "#3f1d2a",
        "bar_fg": "#60a5fa",
        "bar_bg": "#1e293b",
    },
    "ember": {
        "screen_bg": "transparent",
        "card_bg": "#1f140f",
        "stats_bg": "#21140e",
        "prompt_bg": "#1a1210",
        "input_bg": "#130c0a",
        "border": "#3b1d14",
        "title": "#fef3c7",
        "muted": "#d6a08a",
        "hint": "#fbbf24",
        "ok": "#fcd34d",
        "bad": "#f87171",
        "active_ok": "#fde68a",
        "active_bad": "#fb7185",
        "active_fg": "#fde68a",
        "upcoming": "#f3e8e1",
        "flash": "#3d0f12",
        "bar_fg": "#f97316",
        "bar_bg": "#3b1d14",
    },
    "mint": {
        "screen_bg": "transparent",
        "card_bg": "#0b1f24",
        "stats_bg": "#0b1c22",
        "prompt_bg": "#0a1b1f",
        "input_bg": "#07161a",
        "border": "#12323a",
        "title": "#d1fae5",
        "muted": "#7dd3c7",
        "hint": "#5eead4",
        "ok": "#a7f3d0",
        "bad": "#fb7185",
        "active_ok": "#5eead4",
        "active_bad": "#fb7185",
        "active_fg": "#d1fae5",
        "upcoming": "#c7f9f1",
        "flash": "#0f2f2a",
        "bar_fg": "#34d399",
        "bar_bg": "#12323a",
    },
}


def load_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_word_pool() -> List[str]:
    """
    Use the built-in common list for consistent everyday words.
    """
    return FALLBACK_WORDS[:]


def filter_word_pool(pool: List[str], difficulty: str) -> List[str]:
    if difficulty == "easy":
        min_len, max_len = 3, 6
    elif difficulty == "hard":
        min_len, max_len = 2, 10
    else:
        min_len, max_len = 2, 8
    filtered = [w for w in pool if w.isalpha() and min_len <= len(w) <= max_len]
    if difficulty == "hard":
        filtered = filtered + EXTRA_TOKENS[:]
    return filtered if filtered else pool[:]


def build_quote_words(min_count: int) -> List[str]:
    words: List[str] = []
    while len(words) < min_count:
        quote = random.choice(QUOTES)
        words.extend(quote.split())
    return words[:min_count]


def filter_code_tokens(difficulty: str) -> List[str]:
    if difficulty == "easy":
        return [t for t in CODE_TOKENS if t.isalnum()]
    if difficulty == "hard":
        return CODE_TOKENS[:]
    return [t for t in CODE_TOKENS if len(t) <= 8]

def filter_hacker_tools(difficulty: str) -> List[str]:
    if difficulty == "easy":
        min_len, max_len = 3, 8
    elif difficulty == "hard":
        min_len, max_len = 2, 18
    else:
        min_len, max_len = 3, 12
    filtered = [t for t in KALI_TOOLS if min_len <= len(t) <= max_len]
    return filtered if filtered else KALI_TOOLS[:]


# ---------------------------
# Typing math
# ---------------------------

def char_match_count(typed: str, target: str) -> int:
    n = min(len(typed), len(target))
    good = 0
    for i in range(n):
        if typed[i] == target[i]:
            good += 1
    return good


def compute_wpm(chars: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0:
        return 0.0
    minutes = elapsed_sec / 60.0
    return (chars / 5.0) / minutes


# ---------------------------
# UI widgets
# ---------------------------

class ScoreBar(Static):
    """Top bar: high scores + last run summary."""
    pass


class StatsBar(Static):
    """Live stats line."""
    pass


class PromptView(Static):
    """Prompt rendering area."""
    pass


class HelpBar(Static):
    """Help / controls."""
    pass


# ---------------------------
# App
# ---------------------------

class TypingTUI(App):
    CSS = """
    Screen {
        background: transparent;
    }

    #root {
        height: 100%;
        padding: 1 2;
    }

    ScoreBar {
        background: #111827;
        border: round #1f2937;
        padding: 0 2;
        height: 4;
    }

    StatsBar {
        background: #0f172a;
        border: round #1f2937;
        padding: 0 2;
        height: 6;
    }

    HelpBar {
        background: #0f172a;
        border: round #1f2937;
        padding: 0 2;
        height: 3;
    }

    PromptView {
        background: #0b1220;
        border: round #1f2937;
        padding: 1 2;
        height: 1fr;
    }

    Input {
        border: round #1f2937;
        background: #0b0f14;
        padding: 1 2;
        height: 3;
    }
    """

    TITLE = "Typing TUI"
    SUB_TITLE = "60s practice — local only"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+r", "restart", "Restart"),
        ("ctrl+n", "cycle_mode", "Mode"),
        ("ctrl+d", "cycle_difficulty", "Difficulty"),
        ("ctrl+t", "cycle_theme", "Theme"),
    ]

    duration_sec: int = 60

    # reactive state
    remaining: float = reactive(60.0)
    started_at: Optional[float] = reactive(None)
    finished: bool = reactive(False)
    flash_error: bool = reactive(False)

    # content state
    word_pool: List[str]
    words: List[str]
    status: List[Optional[bool]]  # per word: None=not done, True/False=done correctness
    idx: int
    current_fragment: str
    window_start: int
    window_size: int
    window_step: int
    window_buffer: int

    # metrics
    typed_chars: int
    correct_chars: int
    word_mistakes: List[int]
    word_durations: List[float]
    word_start_ts: Optional[float]

    def __init__(self) -> None:
        super().__init__()
        config = load_config()
        self.palettes = THEMES.copy()
        extra_themes = config.get("themes")
        if isinstance(extra_themes, dict):
            for name, colors in extra_themes.items():
                if isinstance(colors, dict):
                    self.palettes[name] = {**self.palettes.get("slate", {}), **colors}
        self.theme_name = str(config.get("theme", "slate"))
        if self.theme_name not in self.palettes:
            self.theme_name = "slate"
        self.palette = self.palettes[self.theme_name]
        self.mode = str(config.get("mode", "words"))
        if self.mode == "code":
            self.mode = "code tokens"
        if self.mode not in MODES:
            self.mode = "words"
        self.difficulty = str(config.get("difficulty", "normal"))
        if self.difficulty not in DIFFICULTIES:
            self.difficulty = "normal"
        try:
            self.duration_sec = int(config.get("duration_sec", self.duration_sec))
        except Exception:
            self.duration_sec = 60

    def compose(self) -> ComposeResult:
        with Container(id="root"):
            self.stats_bar = StatsBar()
            self.help_bar = HelpBar()
            self.prompt_view = PromptView()
            self.input = Input(placeholder="Type here… (space to advance)")
            self.score_bar = ScoreBar()
            yield self.stats_bar
            yield self.help_bar
            yield self.prompt_view
            yield self.input
            yield self.score_bar

    def on_mount(self) -> None:
        self.word_pool = load_word_pool()
        self.apply_theme()
        self._apply_layout(self.size.height)
        self.reset_run()
        self.set_interval(0.1, self._tick)
        self.input.focus()

    def on_resize(self, event) -> None:
        height = getattr(getattr(event, "size", None), "height", None)
        if isinstance(height, int):
            self._apply_layout(height)

    def reset_run(self) -> None:
        self.finished = False
        self.started_at = None
        self.remaining = float(self.duration_sec)

        self.words = self.build_words()
        self.status = [None] * len(self.words)
        self.idx = 0
        self.current_fragment = ""
        self.window_start = 0
        self.window_size = 90
        self.window_step = 40
        self.window_buffer = 20

        self.typed_chars = 0
        self.correct_chars = 0
        self.word_mistakes = [0] * len(self.words)
        self.word_durations = [0.0] * len(self.words)
        self.word_start_ts = None

        self.input.disabled = False
        self.input.value = ""
        self._render_all()

    def _apply_layout(self, height: int) -> None:
        # Reserve space for root vertical padding (top + bottom).
        available = max(0, height - 2)
        stats_h = 5
        help_h = 3
        input_h = 3
        score_h = 4
        prompt_min = 5

        def total_needed() -> int:
            return stats_h + help_h + input_h + score_h + prompt_min

        if total_needed() > available:
            score_h = 3
        if total_needed() > available:
            score_h = 0
        if total_needed() > available:
            help_h = 0

        self.stats_bar.styles.height = stats_h
        self.input.styles.height = input_h
        self.help_bar.styles.height = help_h if help_h else 3
        self.score_bar.styles.height = score_h if score_h else 3
        self.help_bar.styles.display = "block" if help_h else "none"
        self.score_bar.styles.display = "block" if score_h else "none"

    def apply_theme(self) -> None:
        palette = self.palette
        self.screen.styles.background = palette["screen_bg"]
        self.score_bar.styles.background = palette["card_bg"]
        self.stats_bar.styles.background = palette["stats_bg"]
        self.help_bar.styles.background = palette["stats_bg"]
        self.prompt_view.styles.background = palette["prompt_bg"]
        self.input.styles.background = palette["input_bg"]
        border_def = (("round", palette["border"]),)
        self.score_bar.styles.border = border_def
        self.stats_bar.styles.border = border_def
        self.help_bar.styles.border = border_def
        self.prompt_view.styles.border = border_def
        self.input.styles.border = border_def

    def build_words(self) -> List[str]:
        if self.mode == "quotes":
            return build_quote_words(250)
        if self.mode == "code tokens":
            pool = filter_code_tokens(self.difficulty)
            return random.choices(pool, k=250)
        if self.mode == "hacker":
            pool = filter_hacker_tools(self.difficulty)
            return random.choices(pool, k=250)
        pool = filter_word_pool(self.word_pool, self.difficulty)
        return random.choices(pool, k=250)

    def _tick(self) -> None:
        if self.finished:
            return
        if self.started_at is None:
            return
        elapsed = time.time() - self.started_at
        self.remaining = max(0.0, self.duration_sec - elapsed)
        self._render_stats()
        if self.remaining <= 0.0:
            self._finish_run()

    def _finish_run(self) -> None:
        self.finished = True
        # lock input
        self.input.disabled = True
        self.input.placeholder = "Time's up. Press Ctrl+R to restart."

        elapsed = float(self.duration_sec)
        wpm = compute_wpm(self.typed_chars, elapsed)
        cwpm = compute_wpm(self.correct_chars, elapsed)
        acc = (self.correct_chars / self.typed_chars) if self.typed_chars > 0 else 0.0

        words_done = sum(1 for s in self.status if s is not None)
        words_correct = sum(1 for s in self.status if s is True)
        words_wrong = sum(1 for s in self.status if s is False)
        entry = ScoreEntry(
            ts=datetime.now().isoformat(timespec="seconds"),
            seconds=self.duration_sec,
            wpm=round(wpm, 1),
            cwpm=round(cwpm, 1),
            accuracy=round(acc * 100.0, 1),
            words_done=words_done,
            words_correct=words_correct,
            words_wrong=words_wrong,
            mode=self.mode,
            difficulty=self.difficulty,
        )

        scores = load_scores()
        scores.append(entry)
        # sort by correct WPM, then accuracy
        scores.sort(key=lambda s: (s.cwpm, s.accuracy), reverse=True)
        scores = scores[:50]
        save_scores(scores)

        breakdown = self._build_breakdown()
        self._render_scorebar(last=entry, scores=scores, breakdown=breakdown)
        self._render_stats(final=True)
        self._render_help()

    def _ensure_started(self) -> None:
        if self.started_at is None:
            self.started_at = time.time()
            self.input.disabled = False

    def _trigger_flash(self) -> None:
        if self.flash_error:
            return
        self.flash_error = True
        self.prompt_view.styles.background = self.palette["flash"]
        self.set_timer(0.12, self._clear_flash)

    def _clear_flash(self) -> None:
        self.flash_error = False
        self.prompt_view.styles.background = self.palette["prompt_bg"]

    def action_restart(self) -> None:
        self.input.disabled = False
        self.input.placeholder = "Type here… (space to advance)"
        self.reset_run()
        self.input.focus()

    def action_cycle_mode(self) -> None:
        self.mode = self._cycle_value(self.mode, MODES)
        self.action_restart()

    def action_cycle_difficulty(self) -> None:
        self.difficulty = self._cycle_value(self.difficulty, DIFFICULTIES)
        self.action_restart()

    def action_cycle_theme(self) -> None:
        self.theme_name = self._cycle_value(self.theme_name, list(self.palettes.keys()))
        self.palette = self.palettes[self.theme_name]
        self.apply_theme()
        self._render_all()

    def _cycle_value(self, current: str, options: List[str]) -> str:
        if current not in options:
            return options[0]
        idx = options.index(current)
        return options[(idx + 1) % len(options)]

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.finished:
            return
        self._ensure_started()

        value = event.value
        now = time.time()
        if not value:
            self.word_start_ts = None
        # Process complete words separated by spaces (allows fluid typing)
        completed, fragment = split_completed_words(value)
        if self.word_start_ts is None and (completed or fragment):
            self.word_start_ts = now
        for w in completed:
            if self.idx >= len(self.words):
                break
            target = self.words[self.idx]
            correct = (w == target)
            self.status[self.idx] = correct

            self.typed_chars += len(w)
            matched = char_match_count(w, target)
            self.correct_chars += matched
            self.word_mistakes[self.idx] = max(len(w), len(target)) - matched
            if self.word_start_ts is not None:
                self.word_durations[self.idx] = max(0.0, now - self.word_start_ts)
                self.word_start_ts = now
            if not correct:
                self._trigger_flash()

            self.idx += 1
            if self.idx >= self.window_start + self.window_size - self.window_buffer:
                self.window_start = min(
                    self.window_start + self.window_step,
                    max(0, len(self.words) - self.window_size),
                )

        self.current_fragment = fragment
        if self.idx < len(self.words) and fragment:
            target = self.words[self.idx]
            if not target.startswith(fragment):
                self._trigger_flash()
        # Keep only the fragment in the input box
        # (prevents the input from growing unbounded)
        if value != fragment:
            self.input.value = fragment

        self._render_prompt()
        self._render_stats()

    def _scores_for_current(self, scores: List[ScoreEntry]) -> List[ScoreEntry]:
        """Filter scores to the current mode and difficulty."""
        return [s for s in scores if s.mode == self.mode and s.difficulty == self.difficulty]

    def _render_all(self) -> None:
        scores = load_scores()
        self._render_scorebar(scores=scores)
        self._render_stats()
        self._render_help()
        self._render_prompt()

    def _build_breakdown(self) -> Dict[str, List[Tuple[str, float]]]:
        mistakes: List[Tuple[str, float]] = []
        slowest: List[Tuple[str, float]] = []
        for word, miss, dur in zip(self.words, self.word_mistakes, self.word_durations):
            if miss > 0:
                mistakes.append((word, float(miss)))
            if dur > 0:
                slowest.append((word, dur))
        mistakes.sort(key=lambda x: x[1], reverse=True)
        slowest.sort(key=lambda x: x[1], reverse=True)
        return {
            "mistakes": mistakes[:5],
            "slowest": slowest[:5],
        }

    def _latest_score(self, scores: List[ScoreEntry]) -> Optional[ScoreEntry]:
        if not scores:
            return None
        try:
            return max(scores, key=lambda s: datetime.fromisoformat(s.ts))
        except Exception:
            return scores[-1]

    def _render_scorebar(
        self,
        last: Optional[ScoreEntry] = None,
        scores: Optional[List[ScoreEntry]] = None,
        breakdown: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ) -> None:
        if scores is None:
            scores = load_scores()

        scoped = self._scores_for_current(scores)
        top = scoped[:3]
        best = scoped[0] if scoped else None
        last_run = last or self._latest_score(scoped)
        theme = self.palette
        text = Text()
        text.append("Top scores", style=f"bold {theme['title']}")
        text.append("  |  ", style=theme["muted"])
        text.append(
            f"{self.mode} • {self.difficulty} • {self.duration_sec}s",
            style=theme["muted"],
        )
        text.append("\n", style="")

        if last_run is not None:
            text.append("Last run  ", style=theme["muted"])
            text.append(f"{last_run.cwpm} cwpm", style=f"bold {theme['title']}")
            text.append("  ", style=theme["muted"])
            text.append(f"{last_run.accuracy}% acc", style=theme["upcoming"])
            text.append("  ", style=theme["muted"])
            text.append(f"{last_run.words_correct} ok/{last_run.words_wrong} wrong", style=theme["muted"])
            text.append("\n", style="")

        if best is not None:
            text.append("Best score ", style=theme["muted"])
            text.append(f"{best.cwpm} cwpm", style=f"bold {theme['title']}")
            text.append("  ", style=theme["muted"])
            text.append(f"{best.accuracy}% acc", style=theme["upcoming"])
            text.append("  ", style=theme["muted"])
            text.append(f"{best.words_correct} ok/{best.words_wrong} wrong", style=theme["muted"])
            text.append("\n", style="")

        if top:
            for i, s in enumerate(top, start=1):
                text.append(f"{i}. ", style=theme["bar_fg"])
                text.append(f"{s.cwpm:>5} cwpm", style=f"bold {theme['title']}")
                text.append("  ", style=theme["muted"])
                text.append(f"{s.accuracy:>5.1f}% ", style=theme["upcoming"])
                text.append("  ", style=theme["muted"])
                text.append(f"{s.words_correct} ok/{s.words_wrong} wrong", style=theme["muted"])
                text.append("  ", style=theme["muted"])
                text.append(f"{s.ts}\n", style=theme["muted"])
        else:
            text.append("No saved runs yet.\n", style=theme["muted"])

        if breakdown:
            mistakes = breakdown.get("mistakes", [])
            slowest = breakdown.get("slowest", [])
            if mistakes:
                text.append("Mistakes: ", style=theme["muted"])
                for word, miss in mistakes:
                    text.append(f"{word}", style=theme["bad"])
                    text.append(f"({int(miss)}) ", style=theme["muted"])
                text.append("\n", style="")
            if slowest:
                text.append("Slowest: ", style=theme["muted"])
                for word, dur in slowest:
                    text.append(f"{word}", style=theme["active_fg"])
                    text.append(f"({dur:.2f}s) ", style=theme["muted"])

        self.score_bar.update(text)

    def _render_stats(self, final: bool = False) -> None:
        elapsed = 0.0 if self.started_at is None else min(self.duration_sec, time.time() - self.started_at)
        remaining = max(0.0, self.duration_sec - elapsed)
        wpm = compute_wpm(self.typed_chars, max(0.001, elapsed)) if self.started_at else 0.0
        cwpm = compute_wpm(self.correct_chars, max(0.001, elapsed)) if self.started_at else 0.0
        acc = (self.correct_chars / self.typed_chars) if self.typed_chars > 0 else 0.0
        words_done = sum(1 for s in self.status if s is not None)
        words_correct = sum(1 for s in self.status if s is True)
        words_wrong = sum(1 for s in self.status if s is False)

        theme = self.palette
        minutes = int(remaining) // 60
        seconds = int(remaining) % 60
        total_min = int(self.duration_sec) // 60
        total_sec = int(self.duration_sec) % 60
        progress = 0.0 if self.duration_sec == 0 else (elapsed / float(self.duration_sec))
        bar_len = 34
        filled = int(bar_len * min(1.0, max(0.0, progress)))
        bar_full = "=" * filled
        bar_empty = "." * (bar_len - filled)
        percent = 0 if self.duration_sec == 0 else int((elapsed / float(self.duration_sec)) * 100)

        text = Text()
        text.append("Time ", style=theme["muted"])
        text.append(f"{minutes:02d}:{seconds:02d}", style=f"bold {theme['title']}")
        text.append(" / ", style=theme["muted"])
        text.append(f"{total_min:02d}:{total_sec:02d}", style=theme["muted"])
        text.append("  ", style=theme["muted"])
        text.append(f"{percent:>3}%", style=theme["bar_fg"])
        text.append("\n", style="")
        text.append("[", style=theme["muted"])
        if bar_full:
            text.append(bar_full, style=theme["bar_fg"])
        if bar_empty:
            text.append(bar_empty, style=theme["bar_bg"])
        text.append("]", style=theme["muted"])
        text.append("\n", style="")
        text.append("WPM ", style=theme["muted"])
        text.append(f"{wpm:>5.1f}", style=f"bold {theme['title']}")
        text.append("   ", style=theme["muted"])
        text.append("CWPM ", style=theme["muted"])
        text.append(f"{cwpm:>5.1f}", style=f"bold {theme['title']}")
        text.append("   ", style=theme["muted"])
        text.append("Acc ", style=theme["muted"])
        text.append(f"{acc*100:>5.1f}%", style=f"bold {theme['title']}")
        text.append("   ", style=theme["muted"])
        text.append("Correct ", style=theme["muted"])
        text.append(f"{words_correct}", style=f"bold {theme['title']}")
        text.append("   ", style=theme["muted"])
        text.append("Wrong ", style=theme["muted"])
        text.append(f"{words_wrong}", style=f"bold {theme['title']}")
        self.stats_bar.update(text)

    def _render_help(self) -> None:
        theme = self.palette
        text = Text()
        if self.started_at is None and not self.finished:
            text.append("Start typing to begin. ", style=theme["hint"])
        elif self.finished:
            text.append("Time's up. ", style=theme["hint"])
        text.append("Ctrl+R restart", style=theme["hint"])
        text.append("  ", style=theme["muted"])
        text.append("Ctrl+N mode", style=theme["hint"])
        text.append("  ", style=theme["muted"])
        text.append("Ctrl+D difficulty", style=theme["hint"])
        text.append("  ", style=theme["muted"])
        text.append("Ctrl+T theme", style=theme["hint"])
        text.append("  ", style=theme["muted"])
        text.append("Ctrl+Q quit", style=theme["hint"])
        self.help_bar.update(text)

    def _render_prompt(self) -> None:
        # Build a highlighted prompt with current word focus and partial typing feedback.
        text = Text()
        theme = self.palette

        # Render a chunked window to avoid constant scrolling.
        start = self.window_start
        end = min(len(self.words), self.window_start + self.window_size)
        window = self.words[start:end]
        window_status = self.status[start:end]

        for j, word in enumerate(window):
            absolute = start + j
            st = window_status[j]

            if absolute < self.idx:
                # done
                if st is True:
                    text.append(word, style=f"bold {theme['ok']}")
                else:
                    text.append(word, style=f"bold {theme['bad']}")
            elif absolute == self.idx:
                # current: show typed fragment
                frag = self.current_fragment
                good = 0
                for k in range(min(len(frag), len(word))):
                    if frag[k] == word[k]:
                        good += 1
                    else:
                        break
                # correct prefix
                if good:
                    text.append(word[:good], style=f"bold {theme['active_ok']} underline")
                # incorrect part of fragment
                if len(frag) > good:
                    bad_part = frag[good:]
                    # clamp to word length for rendering
                    bad_render = bad_part[: max(0, len(word) - good)]
                    if bad_render:
                        text.append(bad_render, style=f"bold {theme['active_bad']} underline")
                # remaining of target word
                remaining = word[max(len(frag), good):]
                if remaining:
                    text.append(remaining, style=f"bold {theme['active_fg']} underline")
            else:
                # upcoming
                text.append(word, style=theme["upcoming"])

            text.append(" ", style="")

        self.prompt_view.update(text)


def split_completed_words(value: str) -> Tuple[List[str], str]:
    """
    Return (completed_words, fragment_without_spaces).
    Treat one or more spaces as delimiter.
    """
    if not value:
        return [], ""
    if " " not in value:
        return [], value

    parts = value.split(" ")
    completed = [p for p in parts[:-1] if p != ""]
    fragment = parts[-1]
    return completed, fragment


if __name__ == "__main__":
    TypingTUI().run()
