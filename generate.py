"""Sample text from a trained Pip checkpoint with temperature and top-k.

Usage: python generate.py --ckpt checkpoints/pip.pt --prompt "Once" --max_new_tokens 300
       add --no_cache to use the original full-recompute path (the fallback)
       python generate.py --ckpt checkpoints/smollm2 --prompt "Hi Pip"   (fine-tuned SmolLM2 folder)

KV cache (default): the prompt is run once, then each new token is one forward step that reuses
the stored keys and values. When the context fills the block size, the window is re-run over the
last block_size tokens, which is exactly what the full path does, so the output matches it.

Stop: generation ends early when the new text contains a stop string (default "\nUser:"), so the
model does not keep writing turns after its own reply. The stop string is cut from the output.

Loads both float checkpoints (from train.py) and int8 checkpoints (from quantize.py).

SmolLM2 backend: if --ckpt is a directory containing config.json (a Hugging Face model folder
written by train_smol.py), it is loaded with transformers on CPU instead. The scratch PipLM path
is untouched. The HF path returns the same shape as the Pip path: prompt followed by the new text.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from model import PipConfig, PipLM
from tokenizer import BPETokenizer

DEFAULT_STOP = ("\nUser:",)


def is_hf_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def load_model(ckpt_path, device="cpu"):
    if is_hf_dir(ckpt_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ckpt_path)
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float32).to(device)
        model.eval()
        return model, tok
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


def _hit_stop(tok, new_ids, stops):
    """True when the generated text so far contains any stop string."""
    if not stops or not new_ids:
        return False
    text = tok.decode(new_ids)
    return any(s in text for s in stops)


@torch.no_grad()
def _generate_full(model, tok, ids, max_new_tokens, temperature, top_k, device, stops):
    """Original path: re-runs the whole (cropped) context for every new token."""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    n0 = len(ids)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        next_id = _sample(logits, temperature, top_k)
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if _hit_stop(tok, idx[0, n0:].tolist(), stops):
            break
    return idx[0].tolist()


@torch.no_grad()
def _generate_cached(model, tok, ids, max_new_tokens, temperature, top_k, device, stops):
    """KV cache path. Invariant: past holds keys/values for out[:-1] after each sample."""
    bs = model.cfg.block_size
    out = list(ids)
    n0 = len(ids)
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
        if _hit_stop(tok, out[n0:], stops):
            break
    return out


def _cut_at_stop(text, stops):
    """Remove everything from the first stop string onward."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


@torch.no_grad()
def _generate_hf(model, tok, prompt, max_new_tokens, temperature, top_k, device, stops):
    """SmolLM2 / transformers path. Returns prompt + new text, cut at the first stop string."""
    enc = tok(prompt, return_tensors="pt").to(device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=max(temperature, 1e-8),
                  top_k=top_k if top_k and top_k > 0 else None, pad_token_id=pad)
    if stops:
        kwargs.update(stop_strings=list(stops), tokenizer=tok)
    out = model.generate(**enc, **kwargs)
    new_text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return prompt + _cut_at_stop(new_text, stops)


def generate(model, tok, prompt="", max_new_tokens=200, temperature=0.8, top_k=40, device="cpu",
             use_cache=True, stop=DEFAULT_STOP):
    stops = tuple(stop or ())
    if not isinstance(model, PipLM):
        return _generate_hf(model, tok, prompt, max_new_tokens, temperature, top_k, device, stops)
    ids = tok.encode(prompt) or [0]
    if use_cache:
        out = _generate_cached(model, tok, ids, max_new_tokens, temperature, top_k, device, stops)
    else:
        out = _generate_full(model, tok, ids, max_new_tokens, temperature, top_k, device, stops)
    text = tok.decode(out)
    if stops:
        # keep the prompt intact; only the generated part is cut at the stop string
        head = tok.decode(ids)
        if text.startswith(head):
            text = head + _cut_at_stop(text[len(head):], stops)
        else:
            text = _cut_at_stop(text, stops)
    return text


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
