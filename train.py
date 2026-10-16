"""Train Pip on a plain-text file: BPE tokenizer, AdamW, warmup + cosine LR, cross-entropy loss.

Usage: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000

AUTO-RESUME: if the --out checkpoint already exists, training continues from it automatically
(model, optimizer, step counter, and tokenizer are all restored). Use --fresh to ignore an
existing checkpoint and start over. --resume PATH resumes from a specific file.

SAFE STOPS: on SIGTERM (for example from `kill`) or Ctrl-C, training finishes the current step,
saves a checkpoint, and exits. Saves are written to a temp file first, so a checkpoint is never
left half-written.

Progress: the first 10 steps print one line each. After that a line prints every --log_every
steps. A validation pass runs every --eval_every steps. All output is flushed immediately.
"""
import argparse
import math
import os
import signal
import sys
import time

import torch

from model import PipConfig, PipLM
from tokenizer import BPETokenizer

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

STOP = {"requested": False}


def log(msg):
    print(msg, flush=True)


def request_stop(signum, frame):
    STOP["requested"] = True
    log(f"\nstop signal {signum} received: finishing this step, then saving...")


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
def estimate_loss(model, splits, cfg, batch_size, device, iters):
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
    """Write to a temp file, then rename, so an interrupted save never corrupts the old checkpoint."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.cfg.__dict__,
        "step": step,
        "tokenizer": tokenizer.state(),
    }, tmp)
    os.replace(tmp, path)
    tok_path = os.path.join(folder, "tokenizer.json")
    tokenizer.save(tok_path + ".tmp")
    os.replace(tok_path + ".tmp", tok_path)


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
    p.add_argument("--resume", default=None, help="resume from this checkpoint path")
    p.add_argument("--fresh", action="store_true", help="ignore an existing --out checkpoint and start over")
    p.add_argument("--steps", type=int, default=5000, help="total step target (not additional steps)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=5)
    p.add_argument("--eval_at_start", action="store_true")
    p.add_argument("--save_every", type=int, default=200, help="save a checkpoint every N steps")
    p.add_argument("--vocab_size", type=int, default=2048)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=352)
    p.add_argument("--block_size", type=int, default=256)
    args = p.parse_args()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    device = get_device()
    torch.manual_seed(1337)
    log(f"device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    resume_path = args.resume
    if resume_path is None and not args.fresh and os.path.exists(args.out):
        resume_path = args.out
        log(f"found existing checkpoint {args.out}, resuming automatically (use --fresh to start over)")

    ckpt = None
    start_step = 0
    if resume_path:
        model, ckpt = load_checkpoint(resume_path, device)
        cfg = model.cfg
        tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])
        start_step = ckpt["step"]
        log(f"resumed from {resume_path} at step {start_step}")
        if start_step >= args.steps:
            log(f"checkpoint is already at step {start_step} which is >= --steps {args.steps}. "
                f"Raise --steps to train further.")
    else:
        log("training BPE tokenizer (this can take a few minutes on a large file, no progress shown)...")
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
    done = start_step

    if args.eval_at_start:
        log(f"running validation before step {start_step} ({args.eval_iters} batches per split)...")
        losses = estimate_loss(model, splits, cfg, args.batch_size, device, args.eval_iters)
        log(f"[eval] step {start_step}: train {losses['train']:.4f} val {losses['val']:.4f}")

    log("starting training loop, first progress line in a few seconds...")
    try:
        for step in range(start_step, args.steps):
            if STOP["requested"]:
                break
            lr = lr_at(step, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = get_batch(splits["train"], cfg.block_size, args.batch_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
            done = step + 1

            show_now = done <= 10 or done % args.log_every == 0 or done == args.steps
            if show_now:
                now = time.time()
                span = max(now - last_log_time, 1e-9)
                sps = (done - last_log_step) / span
                remaining = (args.steps - done) / sps if sps > 0 else 0
                pct = 100.0 * done / args.steps
                log(f"step {done:>6}/{args.steps} ({pct:5.1f}%) | loss {lv:.4f} | smooth {run_loss:.4f} | "
                    f"lr {lr:.2e} | {sps:6.2f} steps/s | elapsed {fmt_time(now - t0)} | eta {fmt_time(remaining)}")
                last_log_time, last_log_step = now, done

            if done % args.eval_every == 0 or done == args.steps:
                log(f"running validation at step {done} ({args.eval_iters} batches per split, may take a moment)...")
                losses = estimate_loss(model, splits, cfg, args.batch_size, device, args.eval_iters)
                log(f"[eval] step {done}/{args.steps}: train {losses['train']:.4f} val {losses['val']:.4f} "
                    f"elapsed {fmt_time(time.time() - t0)}")
                last_log_time, last_log_step = time.time(), done

            if done % args.save_every == 0:
                save_checkpoint(args.out, model, optimizer, done, tokenizer)
                log(f"checkpoint saved at step {done} -> {args.out}")
    except KeyboardInterrupt:
        STOP["requested"] = True

    # Always save the last completed step, whether we finished, were stopped, or were interrupted.
    save_checkpoint(args.out, model, optimizer, done, tokenizer)
    if STOP["requested"]:
        log(f"stopped at step {done}. checkpoint saved to {args.out}. Re-run the same command to continue.")
    else:
        log(f"finished at step {done}. saved {args.out}")


if __name__ == "__main__":
    main()
