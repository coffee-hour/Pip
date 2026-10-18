# Pip

Pip is now a LoRA fine-tuning project on a big open-weights model. The default base model is Mistral-7B-Instruct-v0.3, trained with QLoRA (4-bit).

Full instructions: see LORA_README.md.

## Quick start (big model)

1. pip install -r requirements-lora.txt
2. python train_lora.py
3. python chat.py   (terminal) or python serve_lora.py   (browser at http://localhost:8000)

Theme: black background (#000000), white text (#ffffff), no amber.

## Legacy small model (not the default)

The original from-scratch model (about 9.7M parameters) is still in the repo: model.py, tokenizer.py, train.py, generate.py, agent.py, setup.sh, requirements.txt. Its commands are unchanged, for example: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000 and python generate.py --ckpt checkpoints/pip.pt --prompt "Hello".

## Notes

- input.txt is the training text. Replace it with any UTF-8 text file to train on different data.
- Checkpoints and lora_out/ are git-ignored or local outputs; do not commit them.
