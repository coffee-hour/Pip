# Browser chat server for the LoRA model. Black background (#000000), white text (#ffffff), no amber.
# Usage: python serve_lora.py   then open http://localhost:8000
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from chat import DEFAULT_MODEL, load, reply

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pip</title>
<style>
html,body{margin:0;background:#000000;color:#ffffff;font-family:ui-monospace,Menlo,Consolas,monospace;}
main{max-width:760px;margin:0 auto;padding:24px;}
h1{font-size:20px;font-weight:normal;border-bottom:1px solid #ffffff;padding-bottom:8px;}
#log{min-height:360px;white-space:pre-wrap;line-height:1.5;}
.you{color:#ffffff;opacity:0.7;}
form{display:flex;gap:8px;margin-top:16px;}
input{flex:1;background:#000000;color:#ffffff;border:1px solid #ffffff;padding:10px;font:inherit;}
button{background:#000000;color:#ffffff;border:1px solid #ffffff;padding:10px 16px;font:inherit;cursor:pointer;}
button:hover{background:#ffffff;color:#000000;}
</style></head>
<body><main>
<h1>Pip</h1>
<div id="log"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Type a message" required><button>Send</button></form>
</main>
<script>
const history=[];const log=document.getElementById('log');
function add(cls,text){const d=document.createElement('div');d.className=cls;d.textContent=text;log.appendChild(d);}
document.getElementById('f').addEventListener('submit',async(e)=>{
  e.preventDefault();const q=document.getElementById('q');const msg=q.value.trim();if(!msg)return;
  q.value='';add('you','you> '+msg);history.push({role:'user',content:msg});
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});
  const j=await r.json();const a=j.reply||('Error: '+(j.error||'unknown'));
  history.push({role:'assistant',content:a});add('pip','pip> '+a);
});
</script></body></html>"""

LOCK = threading.Lock()
STATE = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/chat":
            return self._send(404, json.dumps({"error": "not found"}), "application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages", [])
            with LOCK:
                text = reply(STATE["tok"], STATE["model"], messages, STATE["max_new_tokens"], STATE["temperature"])
            self._send(200, json.dumps({"reply": text}), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}), "application/json")

    def log_message(self, fmt, *args):
        return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default="lora_out/adapter")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    print("Loading model (this takes a minute)...")
    tok, model = load(args.model, args.adapter)
    STATE.update(tok=tok, model=model, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(f"Pip is running at http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
