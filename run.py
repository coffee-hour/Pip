#!/usr/bin/env python3
"""One entry point for Pip. Picks the right backend for this machine.

Usage:
  python run.py            start the browser app (server.py) on http://127.0.0.1:8000
  python run.py --terminal start the terminal companion (pip_companion.py)
  python run.py --big      force the 7B LoRA browser chat (serve_lora.py), needs an NVIDIA GPU
  python run.py --port 8001 (any other flags are passed through to the chosen script)

Choice:
  NVIDIA GPU found and lora_out/adapter exists  -> 7B LoRA chat (serve_lora.py)
  NVIDIA GPU found, no adapter yet              -> SmolLM2 path, with a note about training
  No GPU (Codespaces, Chromebook)               -> SmolLM2 path (small download, runs on CPU)
No adapter is required for the SmolLM2 path. It runs on base weights.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(ROOT, "lora_out", "adapter")


def has_cuda():
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main():
    args = sys.argv[1:]
    mode = "app"
    force_big = False
    rest = []
    for a in args:
        if a == "--terminal":
            mode = "terminal"
        elif a == "--big":
            force_big = True
        else:
            rest.append(a)

    gpu = has_cuda()
    adapter = os.path.isdir(ADAPTER_DIR)

    if force_big or (gpu and adapter):
        if not gpu and force_big:
            print("Warning: --big needs an NVIDIA GPU. Trying anyway; it will fail without one.", flush=True)
        if not adapter:
            print("No LoRA adapter at lora_out/adapter yet. Run: python train_lora.py", flush=True)
            sys.exit(1)
        print("Backend: 7B LoRA (GPU). Starting serve_lora.py", flush=True)
        script = "serve_lora.py"
        if mode == "terminal":
            script = "chat.py"
    else:
        print("Backend: SmolLM2-135M (small, CPU friendly). No adapter needed.", flush=True)
        if gpu and not adapter:
            print("NVIDIA GPU found, but no LoRA adapter yet. Train one with: python train_lora.py", flush=True)
        elif not gpu:
            print("No NVIDIA GPU found. Using the CPU path. First run downloads about 270 MB once.", flush=True)
        script = "pip_companion.py" if mode == "terminal" else "server.py"

    cmd = [sys.executable, os.path.join(ROOT, script)] + rest
    try:
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
