# Pip

Pip is a sovereign companion that runs on your own machine. Black and white, no amber. The web page is served only on 127.0.0.1, and nothing leaves the machine unless you approve a web search.

## Step 1 (once, with internet): bundle the model for offline use

    pip install huggingface_hub
    python bundle_offline.py

This downloads SmolLM2-135M-Instruct (about 270 MB) into checkpoints/smollm2. That folder is the offline bundle. Check it any time with: python bundle_offline.py --check

## Step 2: start Pip

    python run.py

Then open http://127.0.0.1:8000 in your browser. In Codespaces: Ports tab, port 8000, Open in Browser.
Want the terminal instead? Run: python run.py --terminal

With a bundle present, run.py turns on Hugging Face offline mode, so Pip makes no network calls to load the model.

## Offline USB stick on Linux (zero internet)

1. On a machine with internet: do Step 1, then install the Python packages once (pip install -r requirements.txt).
2. Copy the whole Pip folder to the USB stick, including checkpoints/smollm2. Copy the Python packages too, or use a venv on the stick.
3. On the offline Linux machine, open a terminal in the Pip folder and run: python3 run.py
4. Open http://127.0.0.1:8000 in any browser.

If checkpoints/smollm2 is missing, run.py tells you so and Pip needs internet once to fetch the model. Run bundle_offline.py to fix that.

## What run.py picks

- No NVIDIA GPU (Codespaces, Chromebook, USB laptop): the SmolLM2-135M-Instruct brain, CPU only.
- NVIDIA GPU and a trained adapter in lora_out/adapter: the 7B LoRA brain (serve_lora.py).
- NVIDIA GPU, no adapter yet: the SmolLM2 brain, with a note telling you how to train.

No adapter is required. Pip talks out of the box on base weights.

## Big GPU path (7B LoRA)

1. pip install -r requirements-lora.txt
2. python train_lora.py
3. python run.py --big   (or python run.py, which picks it automatically when the adapter exists)

See LORA_README.md for details. Training needs about 12 GB of GPU memory.

## Legacy scratch model (not the default)

The original 9.7M-parameter model (model.py, tokenizer.py, train.py, generate.py, setup.sh) still works. Example: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000

## Notes

- input.txt is the training text. Replace it with any UTF-8 text file to train on different data.
- Checkpoints, lora_out/, and memory.json stay on your machine. Do not commit them. The model bundle is large, so copy it to the USB stick by hand rather than through git.
