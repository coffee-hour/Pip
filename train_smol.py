"""Fine-tune HuggingFaceTB/SmolLM2-135M-Instruct on a plain-text file for Pip (CPU friendly).

Usage: python train_smol.py --data input.txt --out checkpoints/smollm2 --steps 1000
       python train_smol.py --data input.txt --out checkpoints/smollm2 --steps 1000 --threads 4

The output folder is a standard Hugging Face model folder (config.json, weights, tokenizer files).
pip_companion.py loads it through generate.load_model, so pass --ckpt checkpoints/smollm2 when
running the companion. The scratch checkpoints/pip.pt path still works unchanged.

LR schedule: linear warmup, then cosine decay to 10% of --lr. Gradient clipping at 1.0.
Saves every --save_every steps and at the end. Saves go to the folder directly.
"""
import argparse
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    min_lr = 0.1 * args.lr
    return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def save(model, tok, out):
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--data", default="input.txt")
    p.add_argument("--out", default="checkpoints/smollm2")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(1337)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    model.train()

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    if len(data) <= args.block_size + 1:
        raise SystemExit(f"data too short: {len(data)} tokens for block size {args.block_size}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | tokens: {len(data)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    offs = torch.arange(args.block_size)
    t0 = time.time()
    done = 0
    try:
        for step in range(args.steps):
            lr = lr_at(step, args)
            for g in opt.param_groups:
                g["lr"] = lr
            ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
            x = data[ix.unsqueeze(1) + offs]
            # HF shifts labels internally, so labels=x gives next-token loss.
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            done = step + 1
            if done % args.log_every == 0 or done == 1:
                print(f"step {done}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | "
                      f"elapsed {time.time() - t0:.0f}s", flush=True)
            if done % args.save_every == 0:
                save(model, tok, args.out)
    except KeyboardInterrupt:
        print("interrupted, saving last completed step", flush=True)
    save(model, tok, args.out)
    print(f"finished at step {done}", flush=True)


if __name__ == "__main__":
    main()
