"""Sample text from a trained Pip checkpoint with temperature and top-k.

Usage: python generate.py --ckpt checkpoints/pip.pt --prompt "Once" --max_new_tokens 300
"""
import argparse

import torch
import torch.nn.functional as F

from model import PipConfig, PipLM
from tokenizer import CharTokenizer


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = PipConfig(**ckpt["config"])
    model = PipLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer(ckpt["chars"])
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt="", max_new_tokens=200, temperature=0.8, top_k=40, device="cpu"):
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tok.decode(idx[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pip.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.ckpt, device)
    print(generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device))
