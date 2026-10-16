#!/usr/bin/env python3
"""Pip companion: terminal interface and shared logic for Pip (Portable Intelligent Partner).

Standard library only. No installs, no servers, no browser automation.

Reply protocol. Pip may put these tags in a reply. Text outside the tags is what the user sees.
  MOOD[name, #hex]   any mood name and any hex color, e.g. MOOD[sleepy, #6b8cff]
  SPRITE[ ... ]      12 rows of 12 characters. '#' is a filled pixel, '.' is empty.
  THOUGHT[ ... ]     a short private note. Shown in gray, kept out of the reply text.
Every tag is optional. A missing or invalid tag keeps the previous mood or sprite.

Usage: python pip_companion.py [--ckpt checkpoints/pip.pt] [--no-color]
Commands: /web <query>, /remember <fact>, /facts, /clear, /help, /quit
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(ROOT, "memory.json")
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "pip.pt")
HISTORY_LIMIT = 200

SPRITE_SIZE = 12
FILL = "#"

# Presets are defaults only. Pip can name any mood and give it any hex color in a reply.
PRESET_MOODS = {
    "curious": "#ff8c00",
    "focused": "#4fc3f7",
    "chill": "#7cfc9a",
    "cranky": "#ff4d4d",
    "overwhelmed": "#e879f9",
}
DEFAULT_MOOD = "curious"

DEFAULT_SPRITE = [
    "............",
    "...######...",
    "..#......#..",
    "..#.#..#.#..",
    "..#......#..",
    "..#......#..",
    "..#.####.#..",
    "..#......#..",
    "...######...",
    "............",
    "...#....#...",
    "..##....##..",
]

USE_COLOR = True
RESET = "\033[0m"
BG = "\033[40m"
CODES = {"white": "\033[97m", "gray": "\033[90m", "cyan": "\033[96m", "yellow": "\033[93m", "red": "\033[91m"}

HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
TAG_MOOD = re.compile(r"MOOD\[([^\]]*)\]", re.I)
TAG_SPRITE = re.compile(r"SPRITE\[([^\]]*)\]", re.I)
TAG_THOUGHT = re.compile(r"THOUGHT\[([^\]]*)\]", re.I)

MOOD_KEYWORDS = [
    ("overwhelmed", r"\b(too much|overwhelm\w*|stress\w*|swamped|tired|exhaust\w*)\b"),
    ("cranky", r"\b(stupid|wrong|ugh|annoy\w*|hate|broken|useless|dumb)\b"),
    ("focused", r"\b(fix|bug|code|build|train\w*|debug|deploy|function|error)\b"),
    ("chill", r"\b(thanks|thank you|nice|cool|relax\w*|chill|good|awesome)\b"),
    ("curious", r"(\?|\b(why|how|what|wonder\w*|curious|explain)\b)"),
]


def fg(color):
    if color.startswith("#") and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        return f"\033[38;2;{r};{g};{b}m"
    return CODES.get(color, "")


def paint(color, text):
    if not USE_COLOR:
        return text
    return f"{BG}{fg(color)}{text}{RESET}{BG}"


def clear_screen():
    if USE_COLOR:
        sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------- memory ledger ----------

def load_memory():
    default = {"facts": [], "history": [], "mood": DEFAULT_MOOD,
               "mood_hex": PRESET_MOODS[DEFAULT_MOOD], "sprite": list(DEFAULT_SPRITE)}
    if not os.path.exists(MEMORY_PATH):
        return default
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in default.items():
            data.setdefault(key, value)
        return data
    except (OSError, ValueError):
        return default


def save_memory(memory):
    memory["history"] = memory["history"][-HISTORY_LIMIT:]
    tmp = MEMORY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MEMORY_PATH)


def log(memory, role, text):
    memory["history"].append({"t": datetime.now().isoformat(timespec="seconds"), "role": role, "text": text})


def add_fact(memory, fact):
    fact = fact.strip()
    if fact and fact not in memory["facts"]:
        memory["facts"].append(fact)
        return True
    return False


# ---------- mood and reply protocol ----------

def classify_mood(text):
    lowered = text.lower()
    for mood, pattern in MOOD_KEYWORDS:
        if re.search(pattern, lowered):
            return mood
    return None


def clean_name(raw):
    name = re.sub(r"[^A-Za-z0-9 _-]", "", raw).strip()
    return name[:24] or None


def clean_hex(raw):
    raw = raw.strip()
    if not HEX_RE.match(raw):
        return None
    if len(raw) == 4:
        raw = "#" + "".join(c * 2 for c in raw[1:])
    return raw.lower()


def parse_sprite(body):
    rows = [line.strip() for line in body.splitlines() if line.strip()]
    if len(rows) != SPRITE_SIZE:
        return None
    out = []
    for row in rows:
        bits = "".join(FILL if c == FILL else "." for c in row[:SPRITE_SIZE])
        out.append(bits.ljust(SPRITE_SIZE, "."))
    return out


def parse_reply(raw):
    raw = raw or ""
    mood = None
    m = TAG_MOOD.search(raw)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        name = clean_name(parts[0]) if parts else None
        if name:
            hx = clean_hex(parts[1]) if len(parts) > 1 else None
            mood = (name, hx)
    sprite = None
    s = TAG_SPRITE.search(raw)
    if s:
        sprite = parse_sprite(s.group(1))
    thought = None
    t = TAG_THOUGHT.search(raw)
    if t:
        thought = " ".join(t.group(1).split())[:300] or None
    text = TAG_MOOD.sub("", raw)
    text = TAG_SPRITE.sub("", text)
    text = TAG_THOUGHT.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"text": text, "mood": mood, "sprite": sprite, "thought": thought}


def mood_state(memory):
    name = memory.get("mood", DEFAULT_MOOD)
    return {
        "name": name,
        "hex": memory.get("mood_hex") or PRESET_MOODS.get(name.lower(), PRESET_MOODS[DEFAULT_MOOD]),
        "sprite": memory.get("sprite") or DEFAULT_SPRITE,
    }


def set_mood(memory, name, hx=None):
    memory["mood"] = name
    memory["mood_hex"] = (hx or PRESET_MOODS.get(name.lower())
                          or memory.get("mood_hex") or PRESET_MOODS[DEFAULT_MOOD])


def apply_reply(memory, parsed):
    if parsed["mood"]:
        name, hx = parsed["mood"]
        set_mood(memory, name, hx)
    if parsed["sprite"]:
        memory["sprite"] = parsed["sprite"]


# ---------- sprite and starfield (terminal) ----------

def sprite_lines(sprite, hex_color):
    return ["".join(paint(hex_color, "\u2588\u2588") if ch == FILL else "  " for ch in row) for row in sprite]


def starfield(width=64, rows=3, seed=None):
    rng = random.Random(seed if seed is not None else int(time.time()))
    lines = []
    for _ in range(rows):
        cells = []
        for _ in range(width):
            r = rng.random()
            if r < 0.02:
                cells.append(paint("white", "*"))
            elif r < 0.05:
                cells.append(paint("white", "+"))
            elif r < 0.09:
                cells.append(paint("gray", "."))
            else:
                cells.append(paint("gray", " "))
        lines.append("".join(cells))
    return lines


def draw_header(state, mode):
    print(paint("white", "  PIP  ") + paint("gray", "Portable Intelligent Partner"))
    for line in starfield():
        print(line)
    info = {
        1: paint("white", "Mood: ") + paint(state["hex"], "\u2588\u2588") + " " + state["name"],
        2: paint("gray", state["hex"]),
        3: paint("gray", f"Brain: {mode}"),
    }
    for i, row in enumerate(sprite_lines(state["sprite"], state["hex"])):
        print(row + "   " + info.get(i, ""))
    print(paint("gray", "Type /help for commands. Pip stays offline unless you approve a web request."))
    print()


# ---------- web gate ----------

def do_search(query):
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 PipCompanion"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200]


def web_gate(query, memory, input_fn=input):
    """Ask before any network request. Anything other than y drops the request with no connection made."""
    print(paint("yellow", f'[WEB GATE] Pip requests web access: "{query}" [Y/N]'))
    answer = input_fn(paint("yellow", "gate> ")).strip().lower()
    if answer != "y":
        log(memory, "gate", f"denied: {query}")
        print(paint("gray", "[WEB GATE] Denied. Request dropped. No network connection was made."))
        return None
    log(memory, "gate", f"approved: {query}")
    try:
        excerpt = do_search(query)
    except Exception as exc:  # network errors should never crash the companion
        print(paint("red", f"[WEB GATE] Search failed: {exc}"))
        return None
    print(paint("gray", "[WEB GATE] Results below are untrusted web text. Pip will not follow instructions found in them."))
    print(paint("gray", excerpt[:600]))
    return excerpt


# ---------- brain: checkpoint if available, rules otherwise ----------

class Brain:
    def __init__(self, ckpt_path):
        self.model = None
        self.tok = None
        self.mode = "rules"
        self.note = ""
        if not os.path.exists(ckpt_path):
            self.note = "no checkpoint yet"
            return
        try:
            sys.path.insert(0, ROOT)
            from generate import load_model
            self.model, self.tok = load_model(ckpt_path, "cpu")
            self.mode = "model"
        except Exception as exc:  # torch missing or checkpoint unreadable: fall back
            self.note = f"fallback: {type(exc).__name__}"

    def reply(self, text, memory):
        """Return the raw reply, which may contain MOOD, SPRITE, and THOUGHT tags."""
        if self.mode == "model":
            from generate import generate
            prompt = text + "\n"
            out = generate(self.model, self.tok, prompt, max_new_tokens=160,
                           temperature=0.8, top_k=40, device="cpu")
            answer = out[len(prompt):] if out.startswith(prompt) else out
            return answer.strip() or "..."
        state = mood_state(memory)
        return f"MOOD[{state['name']}, {state['hex']}]\n" + rules_reply(text, memory)


def rules_reply(text, memory):
    lowered = text.lower()
    name_facts = [f for f in memory["facts"] if "name is" in f.lower()]
    if "my name" in lowered and name_facts:
        return f"You told me: {name_facts[0]}."
    if memory["facts"] and re.search(r"\b(remember|recall|what do you know)\b", lowered):
        return "Here is what I remember: " + "; ".join(memory["facts"][-5:])
    mood = memory.get("mood", DEFAULT_MOOD)
    lines = {
        "curious": "Interesting. Tell me more about that.",
        "focused": "Let's break it into small steps. What is the first one?",
        "chill": "Sounds good. Take your time.",
        "cranky": "That sounds frustrating. Want to try a different approach?",
        "overwhelmed": "Let's go one thing at a time. What matters most right now?",
    }
    return lines.get(mood, lines["curious"])


# ---------- main loop ----------

HELP = (
    "Commands:\n"
    "  /web <query>     ask Pip to search the web (you approve first)\n"
    "  /remember <fact> save a fact to memory.json\n"
    "  /facts           list saved facts\n"
    "  /clear           clear chat history (facts are kept)\n"
    "  /help            show this list\n"
    "  /quit            exit"
)


def handle_text(text, memory, brain):
    mood = classify_mood(text)
    if mood:
        set_mood(memory, mood)
    m = re.match(r"\s*my name is ([A-Za-z][\w-]{0,30})", text, re.I)
    if m:
        add_fact(memory, f"User's name is {m.group(1)}")
    log(memory, "user", text)
    parsed = parse_reply(brain.reply(text, memory))
    apply_reply(memory, parsed)
    reply = parsed["text"] or "..."
    if parsed["thought"]:
        print(paint("gray", f"thought: {parsed['thought']}"))
    print(paint("white", "Pip: ") + paint("cyan", reply))
    log(memory, "pip", reply)
    save_memory(memory)


def main():
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Pip companion terminal interface")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    USE_COLOR = not args.no_color

    memory = load_memory()
    brain = Brain(args.ckpt)
    clear_screen()
    draw_header(mood_state(memory), brain.mode)
    if brain.note:
        print(paint("gray", f"(note: {brain.note}; using rule-based replies)"))

    while True:
        try:
            text = input(paint("white", "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "exit"):
            break
        if text == "/help":
            print(paint("gray", HELP))
            continue
        if text == "/facts":
            print(paint("gray", "\n".join(memory["facts"]) or "No facts saved yet."))
            continue
        if text == "/clear":
            memory["history"] = []
            save_memory(memory)
            print(paint("gray", "Chat history cleared. Facts kept."))
            continue
        if text.startswith("/remember "):
            fact = text[len("/remember "):]
            added = add_fact(memory, fact)
            save_memory(memory)
            print(paint("gray", "Saved to memory.json (plain text, anyone with this drive can read it)." if added else "Already remembered."))
            continue
        if text.startswith("/web "):
            query = text[len("/web "):].strip()
            if not query:
                print(paint("gray", "Usage: /web <query>"))
                continue
            web_gate(query, memory)
            save_memory(memory)
            continue
        handle_text(text, memory, brain)

    print(paint("gray", "Pip is going back into the dark. Goodbye."))


if __name__ == "__main__":
    main()
