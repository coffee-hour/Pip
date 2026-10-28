"""Orchestrator: connects the satellite subnets to the main PipLM. model.py is not modified.

Flow per turn: router -> thalamus -> endocrine -> hippocampus recall -> specialist (if routed)
-> generate with hormone-shaped sampling -> critic -> hippocampus store -> optional dream pass.

Usage (repo root, after ./train.sh has written a checkpoint):
    from generate import load_model
    from subnets.orchestrator import Orchestrator
    model, tok = load_model("checkpoints/pip.pt")
    pip = Orchestrator(model, tok, memory_path="checkpoints/episodes.json")
    print(pip.respond("hello there"))
"""
import torch

from generate import generate
from subnets import critic, router, specialists, thalamus
from subnets.endocrine import Endocrine
from subnets.hippocampus import Hippocampus


class Orchestrator:
    def __init__(self, model, tok, memory_path=None, device=None, max_new_tokens=200):
        self.model = model
        self.tok = tok
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.endo = Endocrine()
        self.hip = Hippocampus(memory_path)
        self.max_new_tokens = max_new_tokens

    def _specialist_note(self, route: str, text: str) -> str:
        if route == "math":
            try:
                return f"[math] {text.strip()} = {specialists.math_eval(text)}"
            except (ValueError, SyntaxError, ZeroDivisionError):
                return ""
        if route == "code":
            r = specialists.python_check(text)
            return "" if r["ok"] else f"[code] syntax issue: {r['error']}"
        return ""

    def respond(self, user_text: str) -> str:
        route = router.classify(user_text)
        stim = thalamus.perceive(user_text)
        levels = self.endo.tick(stim.valence, stim.arousal, stim.threat)
        temp, top_k = self.endo.sampling()

        memories = self.hip.recall(user_text, k=2)
        note = self._specialist_note(route, user_text)

        parts = []
        if memories:
            parts.append("Memory: " + " | ".join(memories))
        if note:
            parts.append(note)
        parts.append(f"User: {user_text}")
        prompt = "\n".join(parts) + "\nPip:"

        reply = generate(self.model, self.tok, prompt, max_new_tokens=self.max_new_tokens,
                         temperature=temp, top_k=top_k, device=self.device)
        reply = reply[len(prompt):].strip()

        verdict = critic.evaluate(reply)
        if not verdict.ok:
            # one retry with a cooler temperature, which the critic often prefers
            reply2 = generate(self.model, self.tok, prompt, max_new_tokens=self.max_new_tokens,
                              temperature=max(0.4, temp - 0.2), top_k=top_k, device=self.device)
            reply2 = reply2[len(prompt):].strip()
            if critic.evaluate(reply2).score > verdict.score:
                reply = reply2

        self.hip.store(f"User: {user_text} | Pip: {reply}", salience=stim.arousal)
        return reply
