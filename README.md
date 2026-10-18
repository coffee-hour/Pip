# Pip

Pip is a sovereign companion that runs on your own machine. Black and white, no amber.

## Start Pip (one command)

    pip install -r requirements.txt
    python run.py

Then open http://127.0.0.1:8000 in your browser. In Codespaces: Ports tab, port 8000, Open in Browser.

Want the terminal instead? Run: python run.py --terminal

## What run.py picks

- No NVIDIA GPU (Codespaces, Chromebook): the small SmolLM2-135M-Instruct brain. It is about 270 MB, downloads once, and runs on CPU.
- NVIDIA GPU and a trained adapter in lora_out/adapter: the 7B LoRA brain (serve_lora.py).
- NVIDIA GPU, no adapter yet: the SmolLM2 brain, with a note telling you how to train.

No adapter is required. Pip talks out of the box on base weights.

## Big GPU path (7B LoRA)

1. pip install -r requirements-lora.txt
2. python train_lora.py
3. python run.py --big   (or python run.py, which picks it automatically when the adapter exists)

See LORA_README.md for details. Training needs about 12 GB of GPU memory.

## Offline on a USB drive

Copy the whole Pip folder, plus the Hugging Face cache folder (~/.cache/huggingface) so the SmolLM2 weights come with it. Then on the offline machine run python run.py. Nothing is downloaded when the cache is present.

## Legacy scratch model (not the default)

The original 9.7M-parameter model (model.py, tokenizer.py, train.py, generate.py, setup.sh) still works. Example: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000

## Notes

- input.txt is the training text. Replace it with any UTF-8 text file to train on different data.
- Checkpoints, lora_out/, and memory.json stay on your machine. Do not commit them.
