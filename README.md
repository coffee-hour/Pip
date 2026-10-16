# Pip

Pip is a small decoder-only transformer language model built from scratch in PyTorch. It is about 10M parameters with the default config (6 layers, 6 heads, 384 embedding size, 256 context).

## Files

- model.py: causal multi-head self-attention, positional embeddings, layer norms, MLP feed-forward, and language model head (weight-tied).
- tokenizer.py: character-level tokenizer (BPE can be added later).
- train.py: training loop with AdamW, cross-entropy loss, periodic evaluation, checkpoint saving and resuming.
- generate.py: sampling with temperature and top_k.
- agent.py: PipAgent wrapper for drop-in use.

## Quick start on a Chromebook (GitHub Codespaces)

1. Open this repo on GitHub and click Code, then Codespaces, then Create codespace on main.
2. In the terminal run: pip install -r requirements.txt
3. Add a plain-text training file named input.txt to the project root (any large UTF-8 text works).
4. Train: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000
5. Generate: python generate.py --ckpt checkpoints/pip.pt --prompt "Hello"

Codespaces with a GPU is optional. Training runs on CPU, just more slowly. Use a smaller model for quick tests: python train.py --n_layer 4 --n_embd 256 --n_head 4 --steps 1000

## Google Colab alternative

1. Open colab.research.google.com, create a notebook, and set Runtime to T4 GPU.
2. Run: !git clone https://github.com/coffee-hour/Pip.git (use a token if the repo stays private) then %cd Pip
3. Run: !pip install -r requirements.txt
4. Upload input.txt, then run the train and generate commands above with ! in front.
5. Copy checkpoints/pip.pt to Google Drive if you want to keep it.

## Notes

- Checkpoints are saved to checkpoints/ and are git-ignored.
- train.py picks cuda, then mps, then cpu automatically.
- Use --resume checkpoints/pip.pt to continue training.
