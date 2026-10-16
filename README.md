# Pip

Pip is a small decoder-only transformer language model built from scratch in PyTorch, about 9.7M parameters.

## Architecture

- Tokenizer: byte-level BPE, vocabulary 2048 (256 byte tokens plus 1792 learned merges). Trained on input.txt, pure Python, no extra dependencies. Any character can be encoded.
- Layers: 6 transformer blocks, 8 attention heads, width 352, feed-forward width 1408 (GELU).
- Context: 256 tokens. Learned positional embeddings.
- Pre-LayerNorm blocks, causal self-attention, no biases, output head tied to the token embedding.
- Training: AdamW (betas 0.9, 0.95, weight decay 0.1), peak learning rate 1e-3 with 200 warmup steps and cosine decay to 1e-4, batch size 32, gradient clipping at 1.0.

Parameter budget: the embedding table is 2048 x 352 (about 0.72M), so most parameters sit in the transformer blocks (about 8.9M).

## Files

- model.py: PipConfig and PipLM (attention, MLP, LayerNorm, weight-tied head).
- tokenizer.py: BPETokenizer (train, encode, decode, save, load).
- train.py: trains the tokenizer on the first run, then trains the model with AdamW, warmup and cosine LR, periodic evaluation, checkpoints, and resume.
- generate.py: sampling with temperature and top_k.
- agent.py: PipAgent wrapper for drop-in use.
- setup.sh: installs numpy and CPU-only torch on Python 3.11 or 3.12, then starts training.
- .devcontainer/devcontainer.json: Codespaces config using Python 3.11 (installs only, does not train).

## Quick start in GitHub Codespaces

1. Open the repo on GitHub, click Code, then Codespaces, then Create codespace on main.
2. Wait for the container to finish setup (installs dependencies only).
3. Train: bash setup.sh (installs if needed, then trains for 5000 steps), or run the command directly: python train.py --data input.txt --out checkpoints/pip.pt --steps 5000
4. Generate: python generate.py --ckpt checkpoints/pip.pt --prompt "Hello" --max_new_tokens 200

A quick test run: python train.py --steps 1000 --eval_every 250

Do not install Python 3.14 for this project. PyTorch wheels are not reliably available for it yet. Use 3.11 or 3.12.

## Google Colab alternative

1. Open colab.research.google.com, create a notebook, and set Runtime to T4 GPU.
2. Run: !git clone https://github.com/coffee-hour/Pip.git (use a token if the repo stays private) then %cd Pip
3. Run: !pip install -r requirements.txt
4. Run the train and generate commands above with ! in front.
5. Copy checkpoints/pip.pt to Google Drive if you want to keep it.

## Notes

- Checkpoints from the earlier character-level version are not compatible. Delete checkpoints/ before training again.
- A checkpoint stores its own tokenizer, so generate.py and resume always use the matching vocabulary. tokenizer.json is also written next to each checkpoint.
- input.txt is already in the repo. Replace it with any UTF-8 text file to train on different data.
- train.py picks cuda, then mps, then cpu automatically.
- Use --resume checkpoints/pip.pt to continue training.
- Checkpoints are saved to checkpoints/ and are git-ignored.
