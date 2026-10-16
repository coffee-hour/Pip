"""Sample text from a trained Pip checkpoint with temperature and top-k.

Usage: python generate.py --ckpt checkpoints/pip.pt --prompt "Once" --max_new_tokens 300
       add --no_cache to use the original full-recompute path (the fallback)

KV cache (default): the prompt is run once, then each new token is one forward step that reuses
the stored keys and values. When the context fills the block size, the window is re-run over the
last block_size tokens, which is exactly what the full path does, so the output matches it.

Loads both float checkpoints (from train.py) and int8 checkpoints (from quantize.py).
"""
import argparse

import torch
import torch.nn.functional as F

from model import PipConfig, PipLM
from tokenizer import BPETokenizer


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("format") == "int8-dynamic":
        model = ckpt["quantized_model"]
    else:
        cfg = PipConfig(**ckpt["config"])
        model = PipLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.from_state(ckpt["tokenizer"])
    return model, tok


def _sample(logits, temperature, top_k):
    logits = logits[:, -1, :] / max(temperature, 1e-8)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def _generate_full(model, tok, ids, max_new_tokens, temperature, top_k, device):
    """Original path: re-runs the whole (cropped) context for every new token."""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        next_id = _sample(logits, temperature, top_k)
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return idx[0].tolist()


@torch.no_grad()
def _generate_cached(model, tok, ids, max_new_tokens, temperature, top_k, device):
    """KV cache path. Invariant: past holds keys/values for out[:-1] after each sample."""
    bs = model.cfg.block_size
    out = list(ids)
    past = None
    L = 0  # number of positions held in past
    for _ in range(max_new_tokens):
        if past is None or L >= bs:
            window = out[-bs:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits, _, past = model(x, use_cache=True)
            L = len(window)
        else:
            x = torch.tensor([[out[-1]]], dtype=torch.long, device=device)
            logits, _, past = model(x, past=past, use_cache=True, pos_offset=L)
            L += 1
        out.append(_sample(logits, temperature, top_k))
    return out


def generate(model, tok, prompt="", max_new_tokens=200, temperature=0.8, top_k=40, device="cpu",
             use_cache=True):
    ids = tok.encode(prompt) or [0]
    if use_cache:
        out = _generate_cached(model, tok, ids, max_new_tokens, temperature, top_k, device)
    else:
        out = _generate_full(model, tok, ids, max_new_tokens, temperature, top_k, device)
    return tok.decode(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/pip.pt")
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--no_cache", action="store_true", help="use the full-recompute fallback path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.ckpt, device)
    print(generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device,
                   use_cache=not args.no_cache))
