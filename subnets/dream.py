"""Dream: offline consolidation for hippocampus episodes.

Run between sessions, not during chat. It drops near-duplicate episodes (cosine above a
threshold), keeps the most salient survivors up to a budget, and writes them back. Nothing
is sent anywhere; it only reads and rewrites the local episode file.

    python -m subnets.dream checkpoints/episodes.json --keep 500
"""
import argparse

import torch

from subnets.hippocampus import Hippocampus


def consolidate(hip: Hippocampus, keep: int = 500, dup_threshold: float = 0.95) -> dict:
    before = len(hip.episodes)
    ranked = sorted(hip.episodes, key=lambda e: e["salience"], reverse=True)
    kept, kept_vecs = [], []
    for ep in ranked:
        v = torch.tensor(ep["vec"])
        if kept_vecs and float(torch.stack(kept_vecs) @ v).__class__ and (torch.stack(kept_vecs) @ v).max() >= dup_threshold:
            continue
        kept.append(ep)
        kept_vecs.append(v)
        if len(kept) >= keep:
            break
    kept.sort(key=lambda e: e["t"])
    hip.episodes = kept
    hip.save()
    return {"before": before, "after": len(kept), "dropped": before - len(kept)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--keep", type=int, default=500)
    p.add_argument("--dup", type=float, default=0.95)
    a = p.parse_args()
    print(consolidate(Hippocampus(a.path), a.keep, a.dup))
