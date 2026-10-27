# Pip RUNBOOK

## Setup

    git clone https://github.com/coffee-hour/Pip.git && cd Pip
    python3 -m venv .venv && source .venv/bin/activate
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install -r requirements.txt

NVIDIA GPU: install the CUDA build of torch from https://pytorch.org instead. Training then uses bfloat16 automatically.

## Prepare data and train (one command)

    python3 run_pip_pipeline.py

This merges data/pip_curated_corpus.txt with input.txt, trains with train.py (auto-resume, gradient accumulation, checkpoints every 200 steps to checkpoints/pip.pt), and validates the checkpoint and tokenizer. Stop with Ctrl-C and re-run the same command to continue.

Options: --steps N, --fresh, --bf16 (CPU), --prepare-only.

## Test locally

    python3 run_pip_pipeline.py --prompt "Hello Pip"

Interactive chat: python3 run_pip_pipeline.py --chat

## Notes

Do not commit checkpoints/ or checkpoints/train_merged.txt. They are generated locally.
