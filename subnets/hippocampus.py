"""Hippocampus: episodic memory with hashed bag-of-words vectors and cosine retrieval.

Pure torch, no embedding model. Each episode stores text, a salience score, and a unit vector.
Storage is a JSON file so it can be inspected, copied, or deleted by hand.
zlib.crc32 is used for hashing (not Python hash(), which changes between processes).
"""
import json
import re
import time
import zlib
from pathlib import Path

import torch

DIM = 512
_WORD = re.compile(r"[a-z0-9']+")


def embed(text: str) -> torch.Tensor:
    v = torch.zeros(DIM)
    for w in _WORD.findall(text.lower()):
        v[zlib.crc32(w.encode("utf-8")) % DIM] += 1.0
    n = v.norm()
    return v / n if n > 0 else v


class Hippocampus:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.episodes = []  # dicts: text, salience, t, vec (list)
        if self.path and self.path.exists():
            self.episodes = json.loads(self.path.read_text(encoding="utf-8"))

    def store(self, text: str, salience: float = 0.5) -> None:
        self.episodes.append({"text": text, "salience": float(salience), "t": time.time(),
                              "vec": embed(text).tolist()})
        self.save()

    def recall(self, query: str, k: int = 3) -> list:
        if not self.episodes:
            return []
        q = embed(query)
        M = torch.tensor([e["vec"] for e in self.episodes])
        scores = M @ q
        top = torch.topk(scores, min(k, len(self.episodes))).indices.tolist()
        return [self.episodes[i]["text"] for i in top if scores[i] > 0]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.episodes), encoding="utf-8")
