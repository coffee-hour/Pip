#!/usr/bin/env python3
"""Pip pipeline: check, merge the corpus, train (auto-resume), validate, then chat or run one-shot inference.

Run from the Pip folder:
  python3 run_pip_pipeline.py                   merge data, train, validate, print test commands
  python3 run_pip_pipeline.py --prepare-only    merge and check the data only
  python3 run_pip_pipeline.py --chat            chat with the saved checkpoint in this terminal
  python3 run_pip_pipeline.py --prompt "Hello"   one-shot generation from the saved checkpoint

Training resumes automatically from checkpoints/pip.pt, so re-running the same command continues.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

CORPUS = os.path.join("data", "pip_curated_corpus.txt")
BASE_TEXT = "input.txt"
MERGED = os.path.join("checkpoints", "train_merged.txt")
CKPT = os.path.join("checkpoints", "pip.pt")
TOK_JSON = os.path.join("checkpoints", "tokenizer.json")
CURRENT_ARCH = {"n_layer": 22, "n_head": 16, "n_embd": 2048, "block_size": 512}  # Pip Ultra (~1.11B)


def fail(msg):
    sys.exit(f"\nERROR: {msg}\n")


def check_environment():
    if sys.version_info < (3, 9):
        fail("Python 3.9 or newer is required.")
    try:
        import torch
    except ImportError:
        fail("PyTorch is not installed. Run: python3 -m pip install -r requirements.txt\n"
             "See RUNBOOK.md for the CPU and NVIDIA GPU install commands.")
    return torch


def merge_corpus():
    """Concatenate data/pip_curated_corpus.txt and input.txt into one training file."""
    for path in (CORPUS, BASE_TEXT):
        if not os.path.isfile(path):
            fail(f"Missing training file: {path}. Is the repo fully cloned?")
        if os.path.getsize(path) == 0:
            fail(f"Training file is empty: {path}")
    os.makedirs(os.path.dirname(MERGED), exist_ok=True)
    total = 0
    with open(MERGED, "w", encoding="utf-8") as out:
        for path in (CORPUS, BASE_TEXT):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                fail(f"{path} is not valid UTF-8 text.")
            out.write(text + "\n\n")
            total += len(text)
            print(f"  merged {path}: {len(text):,} characters")
    print(f"  wrote {MERGED}: {total:,} characters")


def check_arch_compat(torch):
    """Move aside a checkpoint of a different model size so train.py does not resume the wrong architecture."""
    if not os.path.isfile(CKPT):
        return
    cfg = torch.load(CKPT, map_location="cpu", weights_only=False).get("config", {})
    mismatch = {k: (cfg[k], v) for k, v in CURRENT_ARCH.items() if k in cfg and cfg[k] != v}
    if mismatch:
        old = CKPT.replace(".pt", "_old_arch.pt")
        print(f"  existing checkpoint has a different size {mismatch}; moving it to {old}")
        shutil.move(CKPT, old)
    else:
        print(f"  existing checkpoint found at {CKPT}; training will resume from it")


def train(torch, steps, fresh, bf16_cpu, no_bf16):
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        batch, accum = 4, 8  # Pip Ultra: micro-batch 4 x accum 8 = effective batch 32
        print("  device: cuda (bfloat16 autocast on; falls back to float32 if unsupported)")
    else:
        batch, accum = 8, 4
        print("  device: cpu (float32 by default; add --bf16 to try bfloat16 autocast on CPU)")
        print("  warning: Pip Ultra on CPU is not practical; use a CUDA GPU with about 24 GB VRAM")
    cmd = [sys.executable, "train.py", "--data", MERGED, "--out", CKPT,
           "--steps", str(steps), "--batch_size", str(batch), "--accum_steps", str(accum),
           "--save_every", "200", "--eval_every", "250"]
    if (on_gpu and not no_bf16) or bf16_cpu:
        cmd.append("--bf16")
    if fresh:
        cmd.append("--fresh")
    print("  $ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        fail(f"train.py exited with code {rc}. Re-run the same command to resume from the last checkpoint.")


def validate(torch):
    """Confirm the checkpoint and tokenizer were written and can be loaded and used."""
    from tokenizer import BPETokenizer
    if not os.path.isfile(CKPT):
        fail(f"No checkpoint was written to {CKPT}.")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    for key in ("model", "config", "step", "tokenizer"):
        if key not in ckpt:
            fail(f"Checkpoint {CKPT} is missing '{key}'. Delete it and re-run with --fresh.")
    tok = BPETokenizer.from_state(ckpt["tokenizer"])
    sample = "Pip is a sovereign companion."
    if tok.decode(tok.encode(sample)) != sample:
        fail("Tokenizer round-trip check failed.")
    if not os.path.isfile(TOK_JSON):
        print(f"  warning: {TOK_JSON} not found")
    print(f"  checkpoint OK: step {ckpt['step']}, vocab {tok.vocab_size}, file {CKPT}")
    return ckpt["step"]


def load_model(torch):
    import generate as gen
    if not os.path.isfile(CKPT):
        fail(f"No checkpoint at {CKPT}. Run: python3 run_pip_pipeline.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = gen.load_model(CKPT, device)
    return gen, model, tok, device


def main():
    ap = argparse.ArgumentParser(description="Merge, train, and run Pip.")
    ap.add_argument("--steps", type=int, default=5000, help="total optimizer steps (resumes from checkpoint)")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on CPU too")
    ap.add_argument("--no-bf16", action="store_true", help="turn off bfloat16 autocast on GPU")
    ap.add_argument("--prepare-only", action="store_true", help="merge the data and stop")
    ap.add_argument("--chat", action="store_true", help="interactive chat with the saved checkpoint")
    ap.add_argument("--prompt", default=None, help="one-shot generation from the saved checkpoint")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    torch = check_environment()

    if args.prompt is not None:
        gen, model, tok, device = load_model(torch)
        print(gen.generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device))
        return

    if args.chat:
        gen, model, tok, device = load_model(torch)
        print("Pip chat. Type a message and press Enter. /quit exits.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "/quit":
                break
            out = gen.generate(model, tok, line + "\n", args.max_new_tokens, args.temperature, args.top_k, device)
            reply = out[len(line) + 1:] if out.startswith(line + "\n") else out
            print("pip> " + reply.strip())
        return

    print("[1/4] merge training data")
    merge_corpus()
    if args.prepare_only:
        print("Data is ready. Training was skipped (--prepare-only).")
        return

    print("[2/4] check existing checkpoint")
    if not args.fresh:
        check_arch_compat(torch)

    print("[3/4] train (tokenizer is built on first run; progress prints below)")
    train(torch, args.steps, args.fresh, args.bf16, args.no_bf16)

    print("[4/4] validate checkpoint and tokenizer")
    step = validate(torch)
    print(f"\nDone. Trained to step {step}.")
    print("Chat now:            python3 run_pip_pipeline.py --chat")
    print('One-line test:       python3 run_pip_pipeline.py --prompt "Hello Pip"')


if __name__ == "__main__":
    main()
