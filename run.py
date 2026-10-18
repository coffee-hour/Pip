#!/usr/bin/env python3
"""One entry point for Pip. Picks the backend for this machine and runs fully offline when bundled.

Usage:
  python run.py            start the browser app (server.py) on http://127.0.0.1:8000
  python run.py --terminal start the terminal companion (pip_companion.py)
  python run.py --big      force the 7B LoRA browser chat (serve_lora.py), needs an NVIDIA GPU
  python run.py --port 8001 (other flags are passed through to the chosen script)

Offline: if checkpoints/smollm2 holds the bundled model (made once by bundle_offline.py), run.py
sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE so nothing contacts Hugging Face. Pip then loads only
from the local folder. The browser page is served from 127.0.0.1 and makes no external requests.

Choice:
  NVIDIA GPU found and lora_out/adapter exists  -> 7B LoRA chat (serve_lora.py)
  otherwise                                     -> SmolLM2 path (bundled if present)
No adapter is required for the SmolLM2 path.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(ROOT, "lora_out", "adapter")
BUNDLE_DIR = os.path.join(ROOT, "checkpoints", "smollm2")


def has_cuda():
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def bundle_ready():
    return os.path.isfile(os.path.join(BUNDLE_DIR, "config.json")) and os.path.isfile(
        os.path.join(BUNDLE_DIR, "model.safetensors"))


def enable_offline():
    """Stop Hugging Face libraries from making any network call."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    if bundle_ready():
        enable_offline()
        print("Offline bundle found at checkpoints/smollm2. Running with no network access.", flush=True)
    else:
        print("No offline bundle at checkpoints/smollm2. Pip will need internet once to fetch SmolLM2.", flush=True)
        print("To bundle it for offline use, run once with internet: python bundle_offline.py", flush=True)

    if force_big or (gpu and adapter):
        if not gpu and force_big:
            print("Warning: --big needs an NVIDIA GPU. Trying anyway; it will fail without one.", flush=True)
        if not adapter:
            print("No LoRA adapter at lora_out/adapter yet. Run: python train_lora.py", flush=True)
            sys.exit(1)
        print("Backend: 7B LoRA (GPU). Starting serve_lora.py", flush=True)
        script = "chat.py" if mode == "terminal" else "serve_lora.py"
    else:
        print("Backend: SmolLM2-135M (small, CPU friendly). No adapter needed.", flush=True)
        if gpu and not adapter:
            print("NVIDIA GPU found, but no LoRA adapter yet. Train one with: python train_lora.py", flush=True)
        elif not gpu:
            print("No NVIDIA GPU found. Using the CPU path.", flush=True)
        script = "pip_companion.py" if mode == "terminal" else "server.py"

    cmd = [sys.executable, os.path.join(ROOT, script)] + rest
    try:
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
