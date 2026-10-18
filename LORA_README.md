# Pip: LoRA fine-tuning of a big open-weights model

Default base model: mistralai/Mistral-7B-Instruct-v0.3 (7B class, no gated license).
Optional base model: meta-llama/Llama-3.1-8B-Instruct (8B class, gated; see Llama section).
Training method: QLoRA (4-bit base weights, LoRA adapters trained in bf16/fp16).

Theme: black background (#000000), white text (#ffffff), no amber.

## Requirements

- An NVIDIA GPU with about 12 GB or more VRAM (Colab T4 works, a GPU Codespace works).
- Python 3.10 to 3.12.
- Chromebook or Codespaces without a GPU: use Google Colab with a T4 GPU, or run the commands below there.

## Commands (run these in order)

1. Install: pip install -r requirements-lora.txt
2. Train: python train_lora.py
3. Chat in the terminal: python chat.py
4. Chat in the browser: python serve_lora.py   then open http://localhost:8000

Training defaults: input.txt, 512-token blocks, up to 2000 blocks, 1 epoch, LoRA rank 16, learning rate 2e-4. Output goes to lora_out/adapter.

Quick test run: python train_lora.py --max_blocks 100

## Using Llama-3.1-8B-Instruct instead

1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Run: huggingface-cli login (paste a read token)
3. Train: python train_lora.py --model meta-llama/Llama-3.1-8B-Instruct
4. Chat: python chat.py --model meta-llama/Llama-3.1-8B-Instruct
5. Serve: python serve_lora.py --model meta-llama/Llama-3.1-8B-Instruct

## CPU only

Training a 7B model on CPU is not practical. python train_lora.py --allow_cpu exists for testing only and needs about 32 GB RAM. Chat on CPU also works but is slow.

## Notes

- The adapter is small (tens of MB). The base model downloads from Hugging Face on first run.
- Delete lora_out/ to retrain from scratch.
- The original from-scratch 9.7M model (train.py, generate.py, setup.sh) is kept as legacy and is no longer the default.
