#!/usr/bin/env python3
"""Pip companion: a space-themed terminal interface for Pip (Portable Intelligent Partner).

Standard library only. No installs, no servers, no browser automation.

Features:
- Black background with a starfield header.
- Mood sprite that shifts with the tone of the conversation (curious, focused, chill, cranky, overwhelmed).
- Air-gapped web gate: Pip never touches the network unless you type y at the [WEB GATE] prompt.
- Local memory ledger in memory.json (facts you ask Pip to remember, and recent chat history).
- Uses the trained checkpoint (checkpoints/pip.pt) when torch and the checkpoint are present.
  Otherwise it falls back to rule-based replies, so it works before training.

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

USE_COLOR = True
RESET = "\033[0m"
BG = "\033[40m"
CODES = {
    "white": "\033[97m", "gray": "\033[90m", "cyan": "\033[96m", "blue": "\033[94m",
    "green": "\033[92m", "red": "\033[91m", "magenta": "\033[95m", "yellow": "\033[93m",
}

MOODS = {
    # mood: (color, status label)
    "curious": ("cyan", "curious"),
    "focused": ("blue", "focused"),
    "chill": ("green", "chill"),
    "cranky": ("red", "cranky"),
    "overwhelmed": ("magenta", "overwhelmed"),
}

# Each sprite is 8 columns wide and 4 rows tall.
EYES = {
    "curious": ("o", "o"), "focused": ("-", "-"), "chill": ("^", "^"),
    "cranky": (">", "<"), "overwhelmed": ("@", "@"),
}
MOUTH = {
    "curious": "  ?   ", "focused": "  _   ", "chill": "  u   ",
    "cranky": "  n   ", "overwhelmed": " ~~~~ ",
}

MOOD_KEYWORDS = [
    ("overwhelmed", r"\b(too much|overwhelm\w*|stress\w*|swamped|tired|exhaust\w*)\b"),
    ("cranky", r"\b(stupid|wrong|ugh|annoy\w*|hate|broken|useless|dumb)\b"),
    ("focused", r"\b(fix|bug|code|build|train\w*|debug|deploy|function|error)\b"),
    ("chill", r"\b(thanks|thank you|nice|cool|relax\w*|chill|good|awesome)\b"),
    ("curious", r"(\?|\b(why|how|what|wonder\w*|curious|explain)\b)"),
]


def paint(color, text):
    if not USE_COLOR:
        return text
    return f"{BG}{CODES.get(color, '')}{text}{RESET}{BG}"


def clear_screen():
    if USE_COLOR:
        sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------- memory ledger ----------

def load_memory():
    default = {"facts": [], "history": [], "mood": "curious"}
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


# ---------- mood ----------

def classify_mood(text):
    lowered = text.lower()
    for mood, pattern in MOOD_KEYWORDS:
        if re.search(pattern, lowered):
            return mood
    return None


def sprite_lines(mood):
    left, right = EYES[mood]
    color = MOODS[mood][0]
    rows = [
        " .----. ",
        f"| {left}  {right} |",
        f"|{MOUTH[mood]}|",
        " '----' ",
    ]
    return [paint(color, row) for row in rows]


def mood_square(mood):
    return paint(MOODS[mood][0], "\u2588\u2588") + " " + MOODS[mood][1]


# ---------- starfield ----------

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


def draw_header(mood, mode):
    print(paint("white", "  PIP  ") + paint("gray", "Portable Intelligent Partner"))
    for line in starfield():
        print(line)
    sprite = sprite_lines(mood)
    print(sprite[0] + "   " + paint("white", "Mood: ") + mood_square(mood))
    print(sprite[1] + "   " + paint("gray", f"Brain: {mode}"))
    print(sprite[2])
    print(sprite[3])
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
        if self.mode == "model":
            from generate import generate
            prompt = text + "\n"
            out = generate(self.model, self.tok, prompt, max_new_tokens=160,
                           temperature=0.8, top_k=40, device="cpu")
            answer = out[len(prompt):] if out.startswith(prompt) else out
            return answer.strip() or "..."
        return rules_reply(text, memory)


def rules_reply(text, memory):
    lowered = text.lower()
    name_facts = [f for f in memory["facts"] if "name is" in f.lower()]
    if "my name" in lowered and name_facts:
        return f"You told me: {name_facts[0]}."
    if memory["facts"] and re.search(r"\b(remember|recall|what do you know)\b", lowered):
        return "Here is what I remember: " + "; ".join(memory["facts"][-5:])
    mood = memory.get("mood", "curious")
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
        memory["mood"] = mood
    m = re.match(r"\s*my name is ([A-Za-z][\w-]{0,30})", text, re.I)
    if m:
        add_fact(memory, f"User's name is {m.group(1)}")
    log(memory, "user", text)
    context = ""
    reply = brain.reply(text, memory)
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
    draw_header(memory.get("mood", "curious"), brain.mode)
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
