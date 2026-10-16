"""Export a trained Pip checkpoint as an INT8 dynamic-quantized model for low-RAM CPU inference.

Usage: python quantize.py --ckpt checkpoints/pip.pt --out checkpoints/pip_int8.pt

What gets quantized: every nn.Linear (attention, MLP, and the output head) is converted to int8
weights with dynamic activation quantization, via torch.ao.quantization.quantize_dynamic. The token
and position embeddings stay float32 (quantize_dynamic does not touch nn.Embedding).

Size: the weight matrices shrink about 4x. The whole file is smaller by roughly 3x, because the
embeddings stay float32. The source training checkpoint also stores optimizer state, so it is
larger than the model alone. Compare against the model size, not the training file.

The output keeps the same tokenizer and config, so generate.py and pip_companion.py load it with
the same --ckpt flag. Training and resume do not accept this file: it has no optimizer state, and
train.py expects the float checkpoint format. The source checkpoint is never modified.

Quality: int8 weights change the outputs slightly. Compare sample text before relying on it.
Standard library only apart from torch.
"""
import argparse
import os

import torch
from torch.ao.quantization import quantize_dynamic

from model import PipConfig, PipLM

ROOT = os.path.dirname(os.path.abspath(__file__))


def quantize_checkpoint(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("refusing to overwrite the source checkpoint; pick a different --out")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = PipConfig(**ckpt["config"])
    model = PipLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out = {
        "format": "int8-dynamic",
        "quantized_model": qmodel,
        "config": ckpt["config"],
        "step": ckpt["step"],
        "tokenizer": ckpt["tokenizer"],
    }
    folder = os.path.dirname(os.path.abspath(dst))
    os.makedirs(folder, exist_ok=True)
    tmp = dst + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, dst)
    return os.path.getsize(src), os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description="Pip INT8 dynamic quantization export")
    ap.add_argument("--ckpt", default=os.path.join("checkpoints", "pip.pt"))
    ap.add_argument("--out", default=os.path.join("checkpoints", "pip_int8.pt"))
    args = ap.parse_args()
    src = os.path.join(ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    dst = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    if not os.path.exists(src):
        raise SystemExit("checkpoint not found: " + src)
    src_size, dst_size = quantize_checkpoint(src, dst)
    print("wrote " + dst + " (" + str(dst_size // 1024) + " KB); source file is " +
          str(src_size // 1024) + " KB (includes optimizer state)", flush=True)


if __name__ == "__main__":
    main()
