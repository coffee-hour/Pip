#!/usr/bin/env python3
"""Pip web server. Standard library only. Serves web/index.html and a small JSON API.

Usage: python server.py [--port 8000] [--ckpt checkpoints/pip.pt]

Binds to 127.0.0.1 only, so nothing on your network can reach it. Open the page in your
browser (in Codespaces: Ports tab, port 8000, Open in Browser).

API:
  GET  /api/state                 mood, brain mode, facts, recent history
  POST /api/chat     {text}       reply from the trained model (or rules if no checkpoint)
  POST /api/remember {fact}       save a fact to memory.json
  POST /api/clear    {}           clear chat history (facts are kept)
  POST /api/web      {query, approved}
                                  the server only searches when approved is exactly true,
                                  which the page sends only after you click Y at the web gate
"""
import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pip_companion as pc

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
MAX_BODY = 64 * 1024
MAX_TEXT = 2000

LOCK = threading.Lock()
MEM = pc.load_memory()
BRAIN = None


def state_payload():
    return {
        "mood": MEM.get("mood", "curious"),
        "brain": BRAIN.mode,
        "facts": MEM["facts"],
        "history": MEM["history"][-50:],
    }


def save():
    pc.save_memory(MEM)


def api_chat(body):
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "text is required"}
    text = text.strip()[:MAX_TEXT]
    mood = pc.classify_mood(text)
    if mood:
        MEM["mood"] = mood
    m = re.match(r"\s*my name is ([A-Za-z][\w-]{0,30})", text, re.I)
    if m:
        pc.add_fact(MEM, f"User's name is {m.group(1)}")
    pc.log(MEM, "user", text)
    try:
        reply = BRAIN.reply(text, MEM)
    except Exception as exc:
        save()
        return 500, {"error": f"could not generate a reply: {type(exc).__name__}"}
    pc.log(MEM, "pip", reply)
    save()
    return 200, {"reply": reply, "mood": MEM["mood"], "brain": BRAIN.mode}


def api_remember(body):
    fact = body.get("fact")
    if not isinstance(fact, str) or not fact.strip():
        return 400, {"error": "fact is required"}
    added = pc.add_fact(MEM, fact[:300])
    save()
    msg = "Saved to memory.json (plain text, anyone with this drive can read it)." if added else "Already remembered."
    return 200, {"message": msg, "facts": MEM["facts"]}


def api_clear(body):
    MEM["history"] = []
    save()
    return 200, {"ok": True}


def api_web(body):
    query = body.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 300:
        return 400, {"error": "query is required (max 300 characters)"}
    query = query.strip()
    if body.get("approved") is not True:
        pc.log(MEM, "gate", f"denied: {query}")
        save()
        return 403, {"status": "denied", "message": "Request dropped. No network connection was made."}
    pc.log(MEM, "gate", f"approved: {query}")
    try:
        excerpt = pc.do_search(query)[:600]
    except Exception as exc:
        save()
        return 502, {"error": f"search failed: {type(exc).__name__}"}
    save()
    return 200, {"status": "ok", "excerpt": excerpt}


POST_ROUTES = {
    "/api/chat": api_chat,
    "/api/remember": api_remember,
    "/api/clear": api_clear,
    "/api/web": api_web,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "PipWeb/1.0"

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; the page shows what matters

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            with LOCK:
                payload = state_payload()
            self._json(200, payload)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        fn = POST_ROUTES.get(path)
        if fn is None:
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return self._json(413, {"error": "body too large or invalid"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "invalid JSON"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "JSON object expected"})
        with LOCK:
            code, payload = fn(body)
        self._json(code, payload)


def main():
    global BRAIN
    ap = argparse.ArgumentParser(description="Pip web server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default=pc.DEFAULT_CKPT)
    args = ap.parse_args()

    BRAIN = pc.Brain(args.ckpt)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Pip is running at http://127.0.0.1:{args.port}  (brain: {BRAIN.mode})", flush=True)
    if BRAIN.note:
        print(f"note: {BRAIN.note}; using rule-based replies", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nPip is going back into the dark.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
