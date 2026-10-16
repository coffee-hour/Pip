"""Train Pip on a plain-text file with AdamW and cross-entropy loss.

Usage: python train.py --data input.txt --out checkpoints --steps 5000
"""
import argparse
import os

import torch

from model import PipConfig, PipLM
from tokenizer import CharTokenizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, batch_size, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, step, tokenizer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.cfg.__dict__,
        "step": step,
        "chars": tokenizer.chars,
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = PipConfig(**ckpt["config"])
    model = PipLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="input.txt")
    p.add_argument("--out", default="checkpoints/pip.pt")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(1337)
    print(f"device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(list(set(text)))
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    start_step = 0
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        cfg = model.cfg
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        cfg = PipConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        model = PipLM(cfg).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])

    for step in range(start_step, args.steps):
        if step % args.eval_every == 0 or step == args.steps - 1:
            losses = estimate_loss(model, splits, cfg, args.batch_size, device)
            print(f"step {step}: train {losses['train']:.4f} val {losses['val']:.4f}")
        x, y = get_batch(splits["train"], cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % args.save_every == 0:
            save_checkpoint(args.out, model, optimizer, step + 1, tokenizer)
            tokenizer.save(os.path.join(os.path.dirname(args.out) or ".", "tokenizer.json"))

    save_checkpoint(args.out, model, optimizer, args.steps, tokenizer)
    tokenizer.save(os.path.join(os.path.dirname(args.out) or ".", "tokenizer.json"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
