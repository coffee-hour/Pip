# LoRA / QLoRA fine-tuning of a big open-weights instruct model (default: Mistral-7B-Instruct-v0.3) on input.txt.
# Usage: python train_lora.py            (needs an NVIDIA GPU, 4-bit QLoRA)
#        python train_lora.py --model meta-llama/Llama-3.1-8B-Instruct
import argparse
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune a big open-weights model.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id (base, instruct-tuned preferred).")
    p.add_argument("--data", default="input.txt", help="UTF-8 plain text file to train on.")
    p.add_argument("--out", default="lora_out", help="Folder for the adapter and checkpoints.")
    p.add_argument("--block", type=int, default=512, help="Tokens per training example.")
    p.add_argument("--max_blocks", type=int, default=2000, help="Cap on training examples (keeps runs short).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=1, help="Per-device batch size.")
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--allow_cpu", action="store_true", help="Permit a CPU run (extremely slow, needs about 32 GB RAM).")
    return p.parse_args()


def build_dataset(tok, path, block, max_blocks):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = min(len(ids) // block, max_blocks)
    if n == 0:
        sys.exit(f"Not enough text in {path} for even one block of {block} tokens.")
    rows = []
    for i in range(n):
        chunk = ids[i * block:(i + 1) * block]
        rows.append({"input_ids": chunk, "attention_mask": [1] * block, "labels": list(chunk)})
    print(f"Training examples: {n} blocks of {block} tokens ({n * block} tokens total)")
    return Dataset.from_list(rows)


def main():
    args = parse_args()
    has_gpu = torch.cuda.is_available()
    if not has_gpu and not args.allow_cpu:
        sys.exit(
            "No NVIDIA GPU found. QLoRA 4-bit needs a CUDA GPU (use Colab with a T4, or a GPU Codespace).\n"
            "To force a very slow CPU run anyway, add --allow_cpu."
        )

    print(f"Base model: {args.model}")
    print(f"Device: {'CUDA GPU (4-bit QLoRA)' if has_gpu else 'CPU (float32, LoRA only, very slow)'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dataset = build_dataset(tok, args.data, args.block, args.max_blocks)

    if has_gpu:
        bf16 = torch.cuda.is_bf16_supported()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        bf16 = False
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16,
        fp16=has_gpu and not bf16,
        gradient_checkpointing=has_gpu,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=dataset, data_collator=default_data_collator)
    trainer.train()

    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"Done. LoRA adapter saved to {adapter_dir}")
    print("Chat with it: python chat.py")


if __name__ == "__main__":
    main()
