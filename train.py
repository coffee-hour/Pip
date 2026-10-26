"""Train Pip on a plain-text file: BPE tokenizer, AdamW, warmup + cosine LR, cross-entropy loss.

Default model is Pip-Medium (about 26.5M parameters): 8 layers, 8 heads, width 512, context 512.

Usage: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000

AUTO-RESUME: if the --out checkpoint already exists, training continues from it automatically
(model, optimizer, step counter, and tokenizer are all restored). Use --fresh to ignore an
existing checkpoint and start over. --resume PATH resumes from a specific file. The architecture
of a resumed run always comes from the checkpoint itself, so a checkpoint of a different size
(for example the old 9.7M Pip) must be moved aside before training Pip-Medium. train_curriculum.py
does this automatically.

SAFE STOPS: on SIGTERM (for example from `kill`) or Ctrl-C, training finishes the current step,
saves a checkpoint, and exits. Saves are written to a temp file first, so a checkpoint is never
left half-written.

LR SCHEDULE: linear warmup for --warmup steps (default 500), then cosine decay to 10% of --lr.
GRADIENT CLIPPING: global norm clipped to --clip (default 1.0) every optimizer step.

GRADIENT ACCUMULATION: --accum_steps N runs N micro-batches of --batch_size per optimizer step, so
the effective batch is batch_size * accum_steps. Each micro-batch loss is divided by N before backward,
so gradients match one larger batch. --steps counts optimizer steps, not micro-batches.

MIXED PRECISION: --bf16 runs the forward pass under torch.autocast in bfloat16 on CPU or CUDA.
Weights and optimizer state stay float32, so checkpoints are unchanged. If bfloat16 is not usable on
the device, training falls back to float32.

SPEED FLAGS (all opt-in, all off by default):
  --threads N   set torch CPU threads to N (0 = leave torch's default, which is the physical core count)
  --bf16        bfloat16 autocast (CPU or CUDA). Falls back to float32 if bfloat16 is not usable.
  --compile     wrap the model in torch.compile for training steps. Falls back to eager if unsupported.
                Checkpoints always store the plain model, never the compiled wrapper.

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
    # Same random draws as before (so resumed runs see the same batches), but the slicing is one
    # gather per tensor instead of a Python loop of slices and stacks.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    offs = torch.arange(block_size)
    rows = ix.unsqueeze(1) + offs
    x = data[rows]
    y = data[rows + 1]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def bf16_usable(device_type="cpu"):
    """True only if a bfloat16 autocast region runs on this device type."""
    try:
        if device_type == "cuda" and not torch.cuda.is_bf16_supported():
            return False
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            probe = torch.ones(2, 2, device=device_type) @ torch.ones(2, 2, device=device_type)
        return bool(probe.dtype is not None)
    except Exception:
        return False


def set_threads(n):
    if n and n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


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
    p.add_argument("--steps", type=int, default=5000, help="total optimizer step target (not additional steps)")
    p.add_argument("--batch_size", type=int, default=32, help="micro-batch size per forward pass")
    p.add_argument("--accum_steps", type=int, default=1, help="micro-batches per optimizer step (gradient accumulation)")
    p.add_argument("--lr", type=float, default=6e-4, help="peak learning rate (Pip-Medium)")
    p.add_argument("--warmup", type=int, default=500, help="linear warmup optimizer steps")
    p.add_argument("--clip", type=float, default=1.0, help="gradient norm clip")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=5)
    p.add_argument("--eval_at_start", action="store_true")
    p.add_argument("--save_every", type=int, default=200, help="save a checkpoint every N optimizer steps")
    p.add_argument("--vocab_size", type=int, default=2048)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = torch default)")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast on CPU or CUDA (falls back to float32)")
    p.add_argument("--compile", action="store_true", help="torch.compile the training forward (falls back to eager)")
    args = p.parse_args()

    if args.accum_steps < 1:
        p.error("--accum_steps must be at least 1")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    device = get_device()
    torch.manual_seed(1337)
    if device == "cpu":
        log(f"cpu threads: {set_threads(args.threads)}")
    log(f"device: {device}")

    use_bf16 = False
    amp_device = device if device in ("cpu", "cuda") else "cpu"
    if args.bf16:
        if device in ("cpu", "cuda") and bf16_usable(amp_device):
            use_bf16 = True
            log(f"bf16 autocast: on ({amp_device}; weights and optimizer stay float32)")
        else:
            log("bf16 autocast: requested but not usable here, running float32")

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

    # train_model is used for training forwards only. The plain model is always what gets saved
    # and what eval uses, so checkpoints and resume never see a compiled wrapper.
    train_model = model
    if args.compile:
        try:
            train_model = torch.compile(model)
            log("torch.compile: on (first step will be slow while it compiles)")
        except Exception as exc:
            log(f"torch.compile: unavailable ({type(exc).__name__}), running eager")

    log("encoding data...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    log(f"vocab: {tokenizer.vocab_size} | tokens: {len(data)} | chars: {len(text)}")
    log(f"model: {cfg.n_layer} layers, {cfg.n_head} heads, width {cfg.n_embd}, context {cfg.block_size}")
    log(f"parameters: {model.num_parameters() / 1e6:.2f}M")
    log(f"training steps {start_step} to {args.steps} | micro-batch {args.batch_size} x accum {args.accum_steps} "
        f"= effective batch {args.batch_size * args.accum_steps} | block {cfg.block_size}")

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

            optimizer.zero_grad(set_to_none=True)
            micro_sum = 0.0
            for _ in range(args.accum_steps):
                x, y = get_batch(splits["train"], cfg.block_size, args.batch_size, device)
                with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_bf16):
                    _, loss = train_model(x, y)
                (loss / args.accum_steps).backward()
                micro_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            lv = micro_sum / args.accum_steps
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
