#!/usr/bin/env python3
"""Bundle the SmolLM2-135M-Instruct brain into this Pip folder so it runs with no internet.

Run this ONCE on a machine that has internet (it downloads about 270 MB):
    pip install huggingface_hub
    python bundle_offline.py

It writes the model to checkpoints/smollm2 (the folder Pip reads first). Then copy the whole
Pip folder, including checkpoints/, to the USB stick. On the offline machine run: python run.py

The model files are not committed to git (they are large). The USB stick is the bundle.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEST = os.path.join(ROOT, "checkpoints", "smollm2")
PATTERNS = ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"]
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json"]


def verify(dest):
    missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(dest, f))]
    return missing


def main():
    ap = argparse.ArgumentParser(description="Bundle SmolLM2 for offline Pip")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--check", action="store_true", help="only verify an existing bundle, no download")
    args = ap.parse_args()

    if not args.check:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("Missing package. Run: pip install huggingface_hub   then run this again.")
        print(f"Downloading {args.repo} into {DEST} ...", flush=True)
        try:
            snapshot_download(repo_id=args.repo, local_dir=DEST, allow_patterns=PATTERNS)
        except Exception as exc:
            sys.exit(f"Download failed: {type(exc).__name__}: {exc}\nCheck your internet connection and try again.")

    missing = verify(DEST)
    if missing:
        sys.exit("Bundle is incomplete. Missing: " + ", ".join(missing) + ". Run this script again with internet.")
    size_mb = sum(os.path.getsize(os.path.join(DEST, f)) for f in os.listdir(DEST)
                  if os.path.isfile(os.path.join(DEST, f))) / (1024 * 1024)
    print(f"OK. Offline bundle at {DEST} ({size_mb:.0f} MB).", flush=True)
    print("Next: copy the whole Pip folder to the USB stick. On the offline machine run: python run.py", flush=True)


if __name__ == "__main__":
    main()
