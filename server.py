#!/usr/bin/env python3
"""Pip web server. Standard library only. Serves web/index.html and a small JSON API.

Usage: python server.py [--port 8000] [--ckpt checkpoints/pip.pt]

Binds to 127.0.0.1 only, so nothing on your network can reach it. Open the page in your
browser (in Codespaces: Ports tab, port 8000, Open in Browser).

Passphrase: if a vault passphrase is set (vault.json), the server asks for it in the terminal
before loading memory. The browser then shows a sign-in page, and every /api route except
/api/login needs a session cookie from a correct sign-in. Sessions live in memory only, so a
restart asks again. With no passphrase set, the page and API are open, as before.

API:
  POST /api/login {passphrase}    sign in. Sets an HttpOnly session cookie. 401 on a wrong passphrase.
  POST /api/logout {}             end the session
  GET  /api/state                 mood {name, hex, sprite}, brain mode, facts, profile, recent history
  POST /api/chat     {text}       reply {reply, thought, notes, mood, brain}. Tags (MOOD, SPRITE,
                                  THOUGHT, REMEMBER, tool tags) are handled and never shown as text.
                                  Web search and file writes are NOT approved over HTTP: the page has
                                  no approval prompt for them, so the request is declined.
  POST /api/remember {fact}       save a fact to memory.json
  POST /api/clear    {}           clear chat history (facts are kept)
  POST /api/web      {query, approved}
                                  the server only searches when approved is exactly true,
                                  which the page sends only after you click Y at the web gate
"""
import argparse
import json
import os
import secrets
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pip_companion as pc
import vault

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
MAX_BODY = 64 * 1024
MAX_TEXT = 2000
SESSION_COOKIE = "pip_session"
PUBLIC_POST = ("/api/login", "/api/logout")

LOCK = threading.Lock()
SESSIONS = set()
MEM = None
BRAIN = None

LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pip sign in</title>
<style>
body{background:#000;color:#ddd;font-family:monospace;display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
form{display:flex;flex-direction:column;gap:12px;width:260px}
input,button{font:inherit;padding:8px;background:#111;color:#eee;border:1px solid #444}
button{cursor:pointer}
#msg{color:#f88;min-height:1.2em}
</style></head>
<body>
<form id="f">
<div>PIP is locked</div>
<input id="p" type="password" autocomplete="current-password" placeholder="passphrase" required>
<button type="submit">Unlock</button>
<div id="msg"></div>
</form>
<script>
document.getElementById("f").addEventListener("submit", async function (e) {
  e.preventDefault();
  var msg = document.getElementById("msg");
  msg.textContent = "";
  var r = await fetch("/api/login", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({passphrase: document.getElementById("p").value})
  });
  if (r.ok) { location.reload(); } else { msg.textContent = "Wrong passphrase."; }
});
</script>
</body></html>
"""


def session_token(headers):
    jar = SimpleCookie()
    try:
        jar.load(headers.get("Cookie", ""))
    except Exception:
        return None
    morsel = jar.get(SESSION_COOKIE)
    return morsel.value if morsel is not None else None


def authed(headers):
    """True when no passphrase is set, or when the request carries a valid session cookie."""
    if not vault.vault_exists():
        return True
    token = session_token(headers)
    if not token:
        return False
    with LOCK:
        return any(secrets.compare_digest(token, s) for s in SESSIONS)


def state_payload():
    return {
        "mood": pc.mood_state(MEM),
        "brain": BRAIN.mode,
        "facts": MEM["facts"],
        "profile": MEM["profile"],
        "history": MEM["history"][-50:],
    }


def save():
    pc.save_memory(MEM)


def decline_over_http(message):
    """The web page has no approval prompt for tool actions, so they are declined here."""
    return False


def api_chat(body):
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "text is required"}
    text = text.strip()[:MAX_TEXT]
    try:
        result = pc.run_turn(text, MEM, BRAIN, decline_over_http)
    except Exception as exc:
        save()
        return 500, {"error": f"could not generate a reply: {type(exc).__name__}"}
    reply = result["text"][:MAX_TEXT] or "..."
    save()
    return 200, {
        "reply": reply,
        "thought": result["thought"],
        "notes": result["notes"],
        "mood": pc.mood_state(MEM),
        "brain": BRAIN.mode,
    }


def api_remember(body):
    fact = body.get("fact")
    if not isinstance(fact, str) or not fact.strip():
        return 400, {"error": "fact is required"}
    added = pc.remember(MEM, fact[:300])
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

    def _send(self, code, data, ctype, extra_headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj, extra_headers=()):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", extra_headers)

    def _read_body(self):
        """Return the parsed JSON object, or None after sending an error response."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self._json(413, {"error": "body too large or invalid"})
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json(400, {"error": "invalid JSON"})
            return None
        if not isinstance(body, dict):
            self._json(400, {"error": "JSON object expected"})
            return None
        return body

    def _login(self):
        body = self._read_body()
        if body is None:
            return
        if not vault.vault_exists():
            return self._json(200, {"ok": True, "message": "no passphrase is set"})
        phrase = body.get("passphrase")
        if not isinstance(phrase, str) or not vault.check_passphrase(phrase):
            time.sleep(1.0)  # slow down guessing
            return self._json(401, {"error": "wrong passphrase"})
        token = secrets.token_urlsafe(32)
        with LOCK:
            SESSIONS.add(token)
        cookie = f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"
        self._json(200, {"ok": True}, [("Set-Cookie", cookie)])

    def _logout(self):
        token = session_token(self.headers)
        if token:
            with LOCK:
                SESSIONS.discard(token)
        cookie = f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
        self._json(200, {"ok": True}, [("Set-Cookie", cookie)])

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not authed(self.headers):
                return self._send(200, LOGIN_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/state":
            if not authed(self.headers):
                return self._json(401, {"error": "locked: sign in first"})
            with LOCK:
                payload = state_payload()
            self._json(200, payload)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/login":
            return self._login()
        if path == "/api/logout":
            return self._logout()
        fn = POST_ROUTES.get(path)
        if fn is None:
            return self._json(404, {"error": "not found"})
        if not authed(self.headers):
            return self._json(401, {"error": "locked: sign in first"})
        body = self._read_body()
        if body is None:
            return
        with LOCK:
            code, payload = fn(body)
        self._json(code, payload)


def main():
    global BRAIN, MEM
    ap = argparse.ArgumentParser(description="Pip web server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default=None, help="checkpoint folder or Hugging Face id; default is SmolLM2")
    args = ap.parse_args()

    if vault.vault_exists():
        print("A vault passphrase is set. Enter it to start the server.", flush=True)
        if not pc.unlock_prompt():
            print("[vault] too many wrong attempts. Pip stays locked.", flush=True)
            return
    MEM = pc.load_memory()
    BRAIN = pc.Brain(pc.resolve_model_source(args.ckpt))
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Pip is running at http://127.0.0.1:{args.port}  (brain: {BRAIN.mode})", flush=True)
    if vault.vault_exists():
        print("The browser will ask for the same passphrase.", flush=True)
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
