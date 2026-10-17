#!/usr/bin/env python3
"""Pip companion: terminal interface and shared logic for Pip (Portable Intelligent Partner).

Standard library only. No installs, no servers, no browser automation.

Reply protocol. Pip may put these tags in a reply. Text outside the tags is what the user sees.
  MOOD[name, #hex]        any mood name and any hex color, e.g. MOOD[sleepy, #6b8cff]
  SPRITE[ ... ]           12 rows of 12 characters. '#' is filled, '.' is empty.
  THOUGHT[ ... ]          a short private note, shown in gray, kept out of the reply text
  REMEMBER[fact]          save a fact. REMEMBER[key: value] also sets profile[key].
  READ_FILE[name]         read a file from pip_files/ (no approval needed)
  WRITE_FILE[name, text]  write a file in pip_files/ (you approve first)
  KNOWLEDGE[query]        search the .txt and .md files in pip_files/ (no approval, no network)
  SEARCH[query]           web search (you approve first)
  OBSERVATION[...]        written by the runtime only. Pip text containing it is removed.
Every tag is optional. A missing or invalid tag keeps the previous value.

Turn loop: when Pip emits a tool tag, the runtime cuts the reply there, runs the tool, appends
OBSERVATION[result] to the prompt, and lets Pip continue. At most 2 tool rounds per turn.

Context: before each turn, the most relevant saved facts are placed at the head of the prompt as
FACTS: [...], within a fixed character budget. Relevant snippets from pip_files/*.txt and *.md are
added after them as DOCS: [...], also within a budget. Nothing leaves the machine for either.

Vault: an optional master passphrase (vault.py) is asked for before memory loads. /lock locks the
session and asks again. /password sets, changes, or removes it. The lock does not encrypt files on disk.

Usage: python pip_companion.py [--ckpt PATH] [--no-color]
With no --ckpt, Pip uses checkpoints/smollm2 if that folder exists, otherwise it boots
HuggingFaceTB/SmolLM2-135M-Instruct from the Hugging Face hub. Plain `python pip_companion.py` is enough.
Commands: /web <query>, /docs <query>, /remember <fact>, /facts, /files, /lock, /password, /clear,
          /help, /quit
"""
import argparse
import getpass
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

import vault

ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(ROOT, "memory.json")
FILES_DIR = os.path.join(ROOT, "pip_files")
LOCAL_SMOL_DIR = os.path.join(ROOT, "checkpoints", "smollm2")
DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
HISTORY_LIMIT = 200

SPRITE_SIZE = 12
FILL = "#"

MAX_CONTEXT_CHARS = 360   # budget for the FACTS line (about 90 tokens with the byte BPE)
MAX_CONTEXT_FACTS = 8
MAX_DOC_CONTEXT_CHARS = 360  # budget for the DOCS line
MAX_DOC_HITS = 3
MAX_DOC_FILES = 50
MAX_DOC_READ = 200000        # bytes of each doc read at most
MAX_DOC_CHUNK = 400          # characters per snippet
MAX_TOOL_ROUNDS = 2
MAX_NEW_TOKENS = 320  # room for MOOD, THOUGHT, a 12-row SPRITE, and the reply text
MAX_FILE_CHARS = 4000
MAX_OBS_CHARS = 700
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,58}\.[A-Za-z0-9]{1,8}$")
WORD_RE = re.compile(r"[a-z]{3,}")
DOC_EXTS = (".txt", ".md")
STOPWORDS = {"the", "and", "for", "what", "how", "you", "are", "that", "this", "with", "from",
             "can", "does", "have", "was", "were", "will", "about", "tell", "me", "you", "his", "her"}

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
TAG_REMEMBER = re.compile(r"REMEMBER\[([^\]]*)\]", re.I)
TAG_SEARCH = re.compile(r"SEARCH\[([^\]]*)\]", re.I)
TAG_READ = re.compile(r"READ_FILE\[([^\]]*)\]", re.I)
TAG_WRITE = re.compile(r"WRITE_FILE\[([^,\]]*),([^\]]*)\]", re.I)
TAG_KNOWLEDGE = re.compile(r"KNOWLEDGE\[([^\]]*)\]", re.I)
TAG_OBS = re.compile(r"OBSERVATION\[[^\]]*\]", re.I)
TOOL_TAGS = [("search", TAG_SEARCH), ("read", TAG_READ), ("write", TAG_WRITE), ("knowledge", TAG_KNOWLEDGE)]

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
    default = {"facts": [], "profile": {}, "notes": [], "history": [], "mood": DEFAULT_MOOD,
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


def remember(memory, value):
    """Save one fact. 'key: value' also sets memory['profile'][key]."""
    value = value.replace("[", "").replace("]", "").strip()[:300]
    if not value:
        return False
    key, sep, val = value.partition(":")
    if sep and key.strip() and val.strip():
        memory["profile"][key.strip()[:40]] = val.strip()[:200]
    return add_fact(memory, value)


def context_facts(memory, query):
    """Pick facts for the FACTS line: word overlap with the query, plus the 3 newest, within budget."""
    words = set(WORD_RE.findall(query.lower()))
    facts = memory["facts"]
    ranked = []
    for i, f in enumerate(facts):
        overlap = len(words & set(WORD_RE.findall(f.lower())))
        recent = i >= len(facts) - 3
        if overlap or recent:
            ranked.append((overlap, i, f))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    picked, used = [], 0
    for _, i, f in ranked:
        if len(picked) >= MAX_CONTEXT_FACTS:
            break
        cost = len(f) + 2
        if used + cost > MAX_CONTEXT_CHARS:
            continue
        picked.append((i, f))
        used += cost
    picked.sort()
    return [f for _, f in picked]


# ---------- local docs (pip_files/*.txt and *.md) ----------

def doc_chunks():
    """Split every .txt and .md file in pip_files/ into paragraphs. Returns [(file name, text)]."""
    chunks = []
    if not os.path.isdir(FILES_DIR):
        return chunks
    names = sorted(n for n in os.listdir(FILES_DIR) if n.lower().endswith(DOC_EXTS))[:MAX_DOC_FILES]
    for name in names:
        path = os.path.join(FILES_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                body = f.read(MAX_DOC_READ)
        except OSError:
            continue
        for para in re.split(r"\n\s*\n", body):
            para = " ".join(para.split())
            if para:
                chunks.append((name, para[:MAX_DOC_CHUNK]))
    return chunks


def search_docs(query, limit=MAX_DOC_HITS):
    """Keyword search over the local docs. Returns up to limit (file name, paragraph) pairs."""
    qwords = set(WORD_RE.findall((query or "").lower())) - STOPWORDS
    if not qwords:
        return []
    need = min(2, len(qwords))
    scored = []
    for name, chunk in doc_chunks():
        overlap = len(qwords & set(WORD_RE.findall(chunk.lower())))
        if overlap >= need:
            scored.append((overlap, name, chunk))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(name, chunk) for _, name, chunk in scored[:limit]]


def clean_doc(text):
    return text.replace("[", "(").replace("]", ")")


def doc_context(query):
    """The DOCS line text for a query, within budget. Empty string when nothing matches."""
    parts, used = [], 0
    for name, chunk in search_docs(query):
        piece = name + ": " + clean_doc(chunk)
        if used + len(piece) + 2 > MAX_DOC_CONTEXT_CHARS:
            break
        parts.append(piece)
        used += len(piece) + 2
    return "; ".join(parts)


def build_prompt(memory, text):
    facts = context_facts(memory, text)
    head = ("FACTS: [" + "; ".join(facts) + "]\n") if facts else ""
    docs = doc_context(text)
    dhead = ("DOCS: [" + docs + "]\n") if docs else ""
    return head + dhead + "User: " + text + "\nPip:"


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


TAG_SPRITE_OPEN = re.compile(r"SPRITE\s*\[.*\Z", re.I | re.S)  # unclosed SPRITE[ block, cut to end
SPRITE_ROW_RE = re.compile(r"^[\s.#]{12,}$")                    # bare sprite rows, with or without the tag


def strip_sprite_leftovers(text):
    """Remove an unclosed SPRITE[ block and any stray sprite-only rows from the reply text."""
    text = TAG_SPRITE_OPEN.sub("", text)
    kept = []
    for line in text.split("\n"):
        if (SPRITE_ROW_RE.match(line) and "#" in line) or line.strip() in ("]", "["):
            continue
        kept.append(line)
    return "\n".join(kept)


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
    remember_values = [r.strip() for r in TAG_REMEMBER.findall(raw) if r.strip()]
    text = raw
    for pattern in (TAG_MOOD, TAG_SPRITE, TAG_THOUGHT, TAG_REMEMBER, TAG_SEARCH, TAG_READ, TAG_WRITE,
                    TAG_KNOWLEDGE, TAG_OBS):
        text = pattern.sub("", text)
    text = strip_sprite_leftovers(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"text": text, "mood": mood, "sprite": sprite, "thought": thought, "remember": remember_values}


def first_tool_call(raw):
    """Return (kind, match) for the earliest tool tag in raw, or None."""
    best = None
    for kind, pattern in TOOL_TAGS:
        m = pattern.search(raw)
        if m and (best is None or m.start() < best[1].start()):
            best = (kind, m)
    return best


def clean_turn(raw):
    """Stop the model before it writes the user's next line."""
    return re.split(r"\nUser:", raw or "", maxsplit=1)[0]


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


# ---------- file tools (sandboxed to pip_files/) ----------

def safe_file_name(name):
    name = (name or "").strip()
    return name if FILE_NAME_RE.match(name) else None


def read_pip_file(name):
    path = os.path.join(FILES_DIR, name)
    if not os.path.isfile(path):
        return name + " does not exist"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return name + ": " + f.read(MAX_FILE_CHARS)


def write_pip_file(name, content):
    os.makedirs(FILES_DIR, exist_ok=True)
    path = os.path.join(FILES_DIR, name)
    tmp = path + ".tmp"
    body = content[:MAX_FILE_CHARS]
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
    os.replace(tmp, path)
    return "wrote " + name + " (" + str(len(body)) + " chars)"


def clean_observation(text):
    text = " ".join(str(text).split()).replace("[", "(").replace("]", ")")
    return text[:MAX_OBS_CHARS]


def run_tool(kind, match, memory, approve, notes):
    """Run one tool call and return the observation text. approve(message) -> bool asks the user."""
    if kind == "search":
        query = match.group(1).strip()[:300]
        if not query:
            return "empty search query"
        if not approve('Pip wants to search the web: "' + query + '"'):
            log(memory, "gate", "denied: " + query)
            notes.append("search not approved: no network connection was made")
            return "search not approved by the user; no network connection made"
        log(memory, "gate", "approved: " + query)
        try:
            excerpt = do_search(query)[:600]
        except Exception as exc:  # network errors should never crash the companion
            notes.append("search failed: " + type(exc).__name__)
            return "search failed"
        notes.append("searched: " + query)
        return excerpt
    if kind == "read":
        name = safe_file_name(match.group(1))
        if not name:
            return "invalid file name"
        notes.append("read file: " + name)
        return read_pip_file(name)
    if kind == "write":
        name = safe_file_name(match.group(1))
        if not name:
            return "invalid file name"
        content = match.group(2).strip()
        if not approve("Pip wants to write " + name + " (" + str(len(content)) + " chars) in pip_files/"):
            notes.append("write not approved: " + name)
            return "write not approved by the user"
        result = write_pip_file(name, content)
        notes.append(result)
        return result
    if kind == "knowledge":
        query = match.group(1).strip()[:200]
        if not query:
            return "empty knowledge query"
        notes.append("searched local docs: " + query)
        hits = search_docs(query)
        if not hits:
            return "no matching local documents"
        return "; ".join(name + ": " + clean_doc(chunk) for name, chunk in hits)
    return "unknown tool"


# ---------- one turn, including tool rounds ----------

def apply_parsed(memory, parsed, notes):
    apply_reply(memory, parsed)
    for value in parsed["remember"]:
        if remember(memory, value):
            notes.append("remembered: " + value[:120])


def fallback_reply(text, memory):
    """Used when the model produced no reply text (for example it only emitted a sprite)."""
    if memory.get("mood", DEFAULT_MOOD) == "overwhelmed":
        return "That got tangled for me. Could you ask again in a shorter way?"
    return "I lost my words on that one. Could you ask again, or rephrase it?"


NAME_SET_RE = re.compile(r"^\s*(?:my name is|my name's|call me)\s+([A-Za-z][\w-]{0,30})", re.I)
NAME_QUERY_RE = re.compile(r"^\s*(?:what(?:[\u2019']s| is) my name|who am i|do you know my name)\s*[?.!]*\s*$", re.I)


def known_name(memory):
    """The user's name from the profile or saved facts, or None."""
    name = (memory.get("profile") or {}).get("name")
    if name:
        return name
    for fact in memory.get("facts", []):
        m = re.match(r"User's name is ([A-Za-z][\w-]{0,30})$", fact)
        if m:
            return m.group(1)
    return None


def answer_name_query(text, memory):
    """Answer a question about the user's name straight from memory. None if text is not such a question."""
    if not NAME_QUERY_RE.match(text):
        return None
    name = known_name(memory)
    if name:
        return "You told me your name is " + name + "."
    return "I do not know your name yet. What should I call you?"


def run_turn(text, memory, brain, approve):
    """Handle one user message. Returns {text, thought, notes}. Caller saves memory."""
    mood = classify_mood(text)
    if mood:
        set_mood(memory, mood)
    m = NAME_SET_RE.match(text)
    if m:
        name = m.group(1)
        add_fact(memory, f"User's name is {name}")
        memory.setdefault("profile", {})["name"] = name[:40]
    log(memory, "user", text)

    direct = answer_name_query(text, memory)
    if direct is not None:
        log(memory, "pip", direct)
        return {"text": direct, "thought": "They asked for their name. Answer from memory.", "notes": []}

    prompt = build_prompt(memory, text)
    notes, thought, final_text = [], None, ""
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        raw = clean_turn(brain.reply(prompt, text, memory))
        call = first_tool_call(raw) if brain.mode == "model" else None
        if call is not None and round_no < MAX_TOOL_ROUNDS:
            kind, match = call
            head = raw[:match.end()]
            parsed = parse_reply(head)
            apply_parsed(memory, parsed, notes)
            thought = parsed["thought"] or thought
            obs = clean_observation(run_tool(kind, match, memory, approve, notes))
            prompt = prompt + head + "\nOBSERVATION[" + obs + "]\nPip:"
            continue
        cut = raw if call is None else raw[:call[1].end()]
        parsed = parse_reply(cut)
        apply_parsed(memory, parsed, notes)
        thought = parsed["thought"] or thought
        final_text = parsed["text"]
        break
    final_text = final_text or fallback_reply(text, memory)
    log(memory, "pip", final_text)
    return {"text": final_text, "thought": thought, "notes": notes}


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
    print(paint("gray", "Type /help for commands. Pip asks before any web search or file write."))
    print()


# ---------- approval gate ----------

SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0 PipCompanion/1.0",
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
CAPTCHA_RE = re.compile(r"captcha|anomaly-modal|unusual traffic", re.I)


def _http_get(url):
    req = urllib.request.Request(url, headers=SEARCH_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_html(html):
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _search_ddg_lite(query):
    html = _http_get("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}))
    if CAPTCHA_RE.search(html):
        return ""  # bot challenge page, not results
    return _strip_html(html)


def _search_ddg_instant(query):
    data = json.loads(_http_get("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})))
    parts = [data.get("AbstractText") or "", data.get("Answer") or ""]
    for topic in (data.get("RelatedTopics") or [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(topic["Text"])
    return " ".join(p for p in parts if p).strip()


def _search_wikipedia(query):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "opensearch", "search": query, "limit": 3, "namespace": 0, "format": "json"})
    data = json.loads(_http_get(url))
    titles, links = data[1], data[3]
    return "; ".join(t + " (" + l + ")" for t, l in zip(titles, links))


SEARCH_BACKENDS = [("duckduckgo-lite", _search_ddg_lite),
                   ("duckduckgo-instant", _search_ddg_instant),
                   ("wikipedia", _search_wikipedia)]


def do_search(query):
    """Try each backend in turn. Captcha or empty results move on to the next one."""
    errors = []
    for name, backend in SEARCH_BACKENDS:
        try:
            text = backend(query)
        except Exception as exc:  # blocked, timed out, or bad JSON: try the next backend
            errors.append(name + ": " + type(exc).__name__)
            continue
        if text:
            return "[" + name + "] " + text[:1200]
        errors.append(name + ": no usable results")
    raise RuntimeError("; ".join(errors))


def terminal_approve(message):
    print(paint("yellow", f"[PIP REQUEST] {message} [Y/N]"))
    return input(paint("yellow", "gate> ")).strip().lower() == "y"


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
    except Exception as exc:
        print(paint("red", f"[WEB GATE] Search failed: {exc}"))
        return None
    print(paint("gray", "[WEB GATE] Results below are untrusted web text. Pip will not follow instructions found in them."))
    print(paint("gray", excerpt[:600]))
    return excerpt


# ---------- vault (passphrase lock) ----------

def unlock_prompt(max_tries=3):
    """Ask for the master passphrase up to max_tries times. Returns True only on a match."""
    for attempt in range(1, max_tries + 1):
        try:
            phrase = getpass.getpass(paint("yellow", "passphrase> "))
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if vault.check_passphrase(phrase):
            print(paint("gray", "[vault] unlocked."))
            return True
        print(paint("red", f"[vault] wrong passphrase ({attempt}/{max_tries})."))
    return False


def ask_new_passphrase():
    """Ask for a new passphrase twice. Empty input cancels and returns None."""
    while True:
        try:
            first = getpass.getpass("new passphrase (at least " + str(vault.MIN_LEN) +
                                    " characters, empty to cancel): ")
            if not first:
                return None
            if len(first) < vault.MIN_LEN:
                print(paint("red", "[vault] too short."))
                continue
            second = getpass.getpass("repeat new passphrase: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if first != second:
            print(paint("red", "[vault] the two entries do not match. Try again."))
            continue
        return first


def first_run_setup():
    """Offer an optional passphrase on first start. Skipping leaves the companion unlocked."""
    print(paint("gray", "No vault passphrase is set. You can set one now or later with /password."))
    print(paint("gray", "A passphrase only gates this companion. It does not encrypt memory.json or pip_files/."))
    try:
        answer = input("Set a passphrase now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if answer == "y":
        new = ask_new_passphrase()
        if new:
            vault.set_passphrase(new)
            print(paint("gray", "[vault] passphrase saved to vault.json."))


def change_passphrase(arg):
    """/password sets or changes the passphrase. /password remove deletes it. Both need the current one."""
    if vault.vault_exists() and not unlock_prompt():
        print(paint("red", "[vault] passphrase not changed."))
        return
    if arg == "remove":
        if not vault.vault_exists():
            print(paint("gray", "No passphrase is set."))
            return
        vault.remove_passphrase()
        print(paint("gray", "[vault] passphrase removed. The companion no longer asks for one."))
        return
    new = ask_new_passphrase()
    if new is None:
        print(paint("gray", "[vault] cancelled."))
        return
    vault.set_passphrase(new)
    print(paint("gray", "[vault] passphrase saved."))


def lock_session():
    """Hide the session and require the passphrase again. Returns False if the user cannot unlock."""
    if not vault.vault_exists():
        print(paint("gray", "No passphrase is set, so there is nothing to lock. Use /password to set one."))
        return True
    clear_screen()
    print(paint("yellow", "[vault] session locked."))
    return unlock_prompt()


# ---------- brain: checkpoint if available, rules otherwise ----------

def resolve_model_source(ckpt_arg):
    """Explicit --ckpt wins. Otherwise a local SmolLM2 folder, otherwise the Hugging Face model id."""
    if ckpt_arg:
        return ckpt_arg
    if os.path.isdir(LOCAL_SMOL_DIR) and os.path.exists(os.path.join(LOCAL_SMOL_DIR, "config.json")):
        return LOCAL_SMOL_DIR
    return DEFAULT_HF_MODEL


class Brain:
    def __init__(self, ckpt_path):
        self.model = None
        self.tok = None
        self.mode = "rules"
        self.note = ""
        is_hub_id = (not os.path.exists(ckpt_path) and "/" in ckpt_path and not ckpt_path.startswith((".", "/"))
                   and not ckpt_path.endswith((".pt", ".bin", ".safetensors")))
        if not os.path.exists(ckpt_path) and not is_hub_id:
            self.note = "no checkpoint yet"
            return
        try:
            sys.path.insert(0, ROOT)
            if is_hub_id:
                # Hugging Face model id: load with transformers (downloads once, then cached).
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(ckpt_path)
                self.model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype="auto")
                self.model.eval()
            else:
                from generate import load_model
                self.model, self.tok = load_model(ckpt_path, "cpu")
            self.mode = "model"
        except Exception as exc:  # torch/transformers missing, offline, or checkpoint unreadable: fall back
            self.note = f"fallback: {type(exc).__name__}"

    def reply(self, prompt, text, memory):
        """Return only the continuation after prompt. It may contain MOOD, SPRITE, THOUGHT, and tool tags."""
        if self.mode == "model":
            from generate import generate
            out = generate(self.model, self.tok, prompt, max_new_tokens=MAX_NEW_TOKENS,
                           temperature=0.8, top_k=40, device="cpu")
            return out[len(prompt):] if out.startswith(prompt) else out
        state = mood_state(memory)
        pre = ""
        m = re.match(r"\s*remember (?:that )?(.+)", text, re.I)
        if m:
            pre = "REMEMBER[" + m.group(1)[:200] + "]\n"
        return f"MOOD[{state['name']}, {state['hex']}]\n" + pre + rules_reply(text, memory)


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
    "  /docs <query>    search the .txt and .md files in pip_files/ (offline)\n"
    "  /remember <fact> save a fact to memory.json\n"
    "  /facts           list saved facts and profile\n"
    "  /files           list files in pip_files/\n"
    "  /lock            lock the session and ask for the passphrase again\n"
    "  /password        set or change the passphrase (/password remove deletes it)\n"
    "  /clear           clear chat history (facts are kept)\n"
    "  /help            show this list\n"
    "  /quit            exit"
)


def handle_text(text, memory, brain):
    result = run_turn(text, memory, brain, terminal_approve)
    for note in result["notes"]:
        print(paint("gray", f"[tool] {note}"))
    if result["thought"]:
        print(paint("gray", f"thought: {result['thought']}"))
    print(paint("white", "Pip: ") + paint("cyan", result["text"]))
    save_memory(memory)


def main():
    global USE_COLOR
    ap = argparse.ArgumentParser(description="Pip companion terminal interface")
    ap.add_argument("--ckpt", default=None, help="local checkpoint or folder; default is SmolLM2")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    USE_COLOR = not args.no_color

    # The vault gate runs before any memory or model is loaded.
    if vault.vault_exists():
        if not unlock_prompt():
            print(paint("red", "[vault] too many wrong attempts. Pip stays locked."))
            return
    else:
        first_run_setup()

    memory = load_memory()
    brain = Brain(resolve_model_source(args.ckpt))
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
        if text == "/lock":
            if not lock_session():
                print(paint("red", "[vault] too many wrong attempts. Pip stays locked."))
                return
            clear_screen()
            draw_header(mood_state(memory), brain.mode)
            continue
        if text == "/password" or text.startswith("/password "):
            arg = text[len("/password"):].strip().lower()
            change_passphrase(arg)
            continue
        if text == "/facts":
            print(paint("gray", "\n".join(memory["facts"]) or "No facts saved yet."))
            if memory["profile"]:
                print(paint("gray", "profile: " + json.dumps(memory["profile"], ensure_ascii=False)))
            continue
        if text == "/files":
            names = sorted(os.listdir(FILES_DIR)) if os.path.isdir(FILES_DIR) else []
            print(paint("gray", "\n".join(names) or "No files in pip_files/ yet."))
            continue
        if text.startswith("/docs "):
            hits = search_docs(text[len("/docs "):].strip())
            if not hits:
                print(paint("gray", "No matching lines in pip_files/*.txt or *.md."))
            for name, chunk in hits:
                print(paint("gray", name + ": " + chunk))
            continue
        if text == "/clear":
            memory["history"] = []
            save_memory(memory)
            print(paint("gray", "Chat history cleared. Facts kept."))
            continue
        if text.startswith("/remember "):
            fact = text[len("/remember "):]
            added = remember(memory, fact)
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
