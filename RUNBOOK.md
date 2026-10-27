# Pip RUNBOOK

## Setup

    git clone https://github.com/coffee-hour/Pip.git && cd Pip
    python3 -m venv .venv && source .venv/bin/activate
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install -r requirements.txt

NVIDIA GPU: install the CUDA build of torch from https://pytorch.org instead. Training then uses bfloat16 automatically.

## The two commands

    ./train.sh     to train
    ./chat.sh      to talk to Pip

Both scripts fall back to python if python3 is not found. If you see "Permission denied", run once: chmod +x train.sh chat.sh

## Train

    ./train.sh

This merges data/pip_curated_corpus.txt with input.txt, trains with train.py (auto-resume, gradient accumulation, checkpoints every 200 steps to checkpoints/pip.pt), and validates the checkpoint and tokenizer. Stop with Ctrl-C and run ./train.sh again to continue.

Extra options are passed through, for example: ./train.sh --steps N, ./train.sh --fresh, ./train.sh --bf16 (CPU), ./train.sh --prepare-only.

## Talk to Pip

    ./chat.sh

Type a message and press Enter. Type /quit to exit.

One-line test: python3 run_pip_pipeline.py --prompt "Hello Pip"

## Notes

Do not commit checkpoints/ or checkpoints/train_merged.txt. They are generated locally.
