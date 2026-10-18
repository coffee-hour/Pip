# Terminal chat with the fine-tuned LoRA adapter on top of the big base model.
# Usage: python chat.py   (or: python chat.py --model meta-llama/Llama-3.1-8B-Instruct --adapter lora_out/adapter)
import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load(model_id, adapter_dir):
    has_gpu = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if has_gpu:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    else:
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tok, model


def reply(tok, model, history, max_new_tokens, temperature):
    prompt_ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = prompt_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][prompt_ids.shape[-1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default="lora_out/adapter")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    try:
        tok, model = load(args.model, args.adapter)
    except OSError as e:
        sys.exit(f"Could not load model or adapter: {e}\nRun python train_lora.py first, or check --adapter.")

    print("Pip LoRA chat. Type your message. Commands: /reset to clear, /quit to exit.")
    history = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = []
            print("(history cleared)")
            continue
        history.append({"role": "user", "content": user})
        answer = reply(tok, model, history, args.max_new_tokens, args.temperature)
        history.append({"role": "assistant", "content": answer})
        print(f"pip> {answer}")


if __name__ == "__main__":
    main()
