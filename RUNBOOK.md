# Pip RUNBOOK

## Setup

    git clone https://github.com/coffee-hour/Pip.git && cd Pip
    python3 -m venv .venv && source .venv/bin/activate
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install -r requirements.txt

NVIDIA GPU: install the CUDA build of torch from https://pytorch.org instead. Training then uses bfloat16 automatically.

## Model size and hardware (Pip-Beast)

The default model is Pip-Beast: about 1.11B parameters, 22 layers, 16 heads, width 2048, context 512, vocab 2048.

- GPU required. Plan for an NVIDIA GPU with about 24 GB of VRAM. Training keeps fp32 weights, gradients, and AdamW states, about 18 GB before activations.
- CPU training is not practical at this size.
- On GPU, run_pip_pipeline.py uses micro-batch 4 with 8 accumulation steps (effective batch 32). If you run out of memory, edit the batch line in train() in run_pip_pipeline.py to a smaller micro-batch and raise accumulation to keep the effective batch the same. train.sh does not accept --batch_size.
- Disk: each checkpoint is about 13 GB (weights plus optimizer state). Saves write a temp file first, so keep about 30 GB free.
- Moving from the old 26.5M Pip-Medium: run_pip_pipeline.py moves a checkpoint of a different size to checkpoints/pip_old_arch.pt and starts fresh. Delete that file once you no longer need it.

## The two commands

    ./train.sh     to train
    ./chat.sh      to talk to Pip

Both scripts fall back to python if python3 is not found. If you see "Permission denied", run once: chmod +x train.sh chat.sh

## Train

    ./train.sh

This merges data/pip_curated_corpus.txt with input.txt, trains with train.py (auto-resume, gradient accumulation, checkpoints every 200 steps to checkpoints/pip.pt), and validates the checkpoint and tokenizer. Stop with Ctrl-C and run ./train.sh again to continue.

Extra options are passed through, for example: ./train.sh --steps N, ./train.sh --fresh, ./train.sh --bf16 (CPU), ./train.sh --prepare-only.

The tokenizer is trained in pure Python on the first run and can take several minutes on the 4.5 MB input file.

## Talk to Pip

    ./chat.sh

Type a message and press Enter. Type /quit to exit.

One-line test: python3 run_pip_pipeline.py --prompt "Hello Pip"

## Notes

Do not commit checkpoints/ or checkpoints/train_merged.txt. They are generated locally.
