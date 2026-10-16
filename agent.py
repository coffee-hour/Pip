"""Drop-in wrapper so Pip can be called like any other PortableAI model component.

Adapt the class name or method signature to match the PortableAI interface if needed.
"""
from generate import generate, load_model


class PipAgent:
    def __init__(self, ckpt_path="checkpoints/pip.pt", device="cpu"):
        self.model, self.tokenizer = load_model(ckpt_path, device)
        self.device = device

    def respond(self, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
        return generate(self.model, self.tokenizer, prompt, max_new_tokens,
                        temperature, top_k, self.device)
