"""Train Pip on a plain-text file: BPE tokenizer, AdamW, warmup + cosine LR, cross-entropy loss.

Usage: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000
A fresh run trains the BPE tokenizer on the data. Resuming reuses the tokenizer stored in the
checkpoint, so the token IDs stay consistent.

Progress: a live line prints every --log_every steps (step, loss, lr, steps/sec, ETA).
A full validation pass prints every --eval_every steps. All output is flushed immediately,
so it shows up live in the terminal and in `tail -f train.log` even without `python -u`.
"""
import argparse
import math
import os
import sys
import time

import torch

from model import PipConfig, PipLM
from tokenizer import BPETokenizer

# Line-buffer stdout so every print appears right away (also helps when piped to a log file).
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def log(msg):
    print(msg, flush=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, args):
    """Linear warmup, then cosine decay down to 10% of the peak learning rate."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    min_lr = 0.1 * args.lr
    return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def save_checkpoint(path, model, optimizer, step, tokenizer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.cfg.__dict__,
        "step": step,
        "tokenizer": tokenizer.state(),
    }, path)
    tokenizer.save(os.path.join(os.path.dirname(path) or ".", "tokenizer.json"))


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = PipConfig(**ckpt["config"])
    model = PipLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="input.txt")
    p.add_argument("--out", default="checkpoints/pip.pt")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10, help="print a live progress line every N steps")
    p.add_argument("--eval_every", type=int, default=250, help="full train/val evaluation every N steps")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--vocab_size", type=int, default=2048)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=352)
    p.add_argument("--block_size", type=int, default=256)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(1337)
    log(f"device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    ckpt = None
    start_step = 0
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        cfg = model.cfg
        tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])
        start_step = ckpt["step"]
        log(f"resumed from {args.resume} at step {start_step}")
    else:
        log("training BPE tokenizer (this can take a few minutes on a large file)...")
        t_tok = time.time()
        tokenizer = BPETokenizer.train(text, args.vocab_size)
        log(f"tokenizer ready in {fmt_time(time.time() - t_tok)}")
        cfg = PipConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        model = PipLM(cfg).to(device)

    log("encoding data...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    log(f"vocab: {tokenizer.vocab_size} | tokens: {len(data)} | chars: {len(text)}")
    log(f"parameters: {model.num_parameters() / 1e6:.2f}M")
    log(f"training steps {start_step} to {args.steps} | batch {args.batch_size} | block {cfg.block_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    t0 = time.time()
    run_loss = None
    last_log_time, last_log_step = t0, start_step
    for step in range(start_step, args.steps):
        lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % args.eval_every == 0 or step == args.steps - 1:
            losses = estimate_loss(model, splits, cfg, args.batch_size, device)
            log(f"[eval] step {step}/{args.steps}: train {losses['train']:.4f} val {losses['val']:.4f} "
                f"lr {lr:.2e} elapsed {fmt_time(time.time() - t0)}")

        x, y = get_batch(splits["train"], cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lv = loss.item()
        run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            now = time.time()
            span = max(now - last_log_time, 1e-9)
            sps = (done - last_log_step) / span
            remaining = (args.steps - done) / sps if sps > 0 else 0
            pct = 100.0 * done / args.steps
            log(f"step {done:>6}/{args.steps} ({pct:5.1f}%) | loss {lv:.4f} | smooth {run_loss:.4f} | "
                f"lr {lr:.2e} | {sps:6.2f} steps/s | elapsed {fmt_time(now - t0)} | eta {fmt_time(remaining)}")
            last_log_time, last_log_step = now, done

        if done % args.save_every == 0:
            save_checkpoint(args.out, model, optimizer, done, tokenizer)
            log(f"checkpoint saved at step {done} -> {args.out}")

    save_checkpoint(args.out, model, optimizer, args.steps, tokenizer)
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
