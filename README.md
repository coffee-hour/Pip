# Pip

Pip is a custom PyTorch transformer that you train on your own machine. It is sovereign: the model, tokenizer, and training data all live in this repo, and nothing is sent anywhere.

No bloat. The core pipeline is a few Python files plus two shell scripts.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m pip install -r requirements.txt

NVIDIA GPU: install the CUDA build of torch from https://pytorch.org instead.

## Commands

    ./train.sh
    ./chat.sh

./train.sh trains Pip and resumes automatically if a checkpoint exists. Stop with Ctrl-C and run it again to continue.

./chat.sh opens a chat with the trained model. Type /quit to exit.

If you see "Permission denied", run once: chmod +x train.sh chat.sh

## Web UI (Pip Ultra)

index.html is a single zero-build page: architecture and parameter breakdown, an interactive satellite subnets map, and a mock Sovereign Console. It needs no install and no build step. Fonts load from Google Fonts; everything else is local.

Open it directly:

    open index.html        # macOS
    xdg-open index.html    # Linux

Or serve it locally (recommended, avoids browser file restrictions):

    python3 -m http.server 8000

Then visit http://localhost:8000. The console tab is a preview only: its replies are canned and no model runs in the browser.

## Files

- index.html: Pip Ultra web UI (zero build, open directly or serve)
- train.sh, chat.sh: entry points
- run_pip_pipeline.py: merges the data, trains, validates, and runs chat
- train.py: training loop
- model.py, pip_model.py: transformer architecture
- tokenizer.py: byte-pair encoding tokenizer
- generate.py: text generation
- subnets/: satellite subnets (router, thalamus, endocrine, hippocampus, specialists, critic, dream, orchestrator)
- data/pip_curated_corpus.txt, input.txt: training text
- RUNBOOK.md: detailed run notes

## Notes

- Checkpoints in checkpoints/ are generated locally and must not be committed.
- To train on different text, replace input.txt with any UTF-8 text file.
