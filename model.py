"""Pip: a decoder-only transformer language model written from scratch in PyTorch.

Default config is Pip-Medium, about 26.5M parameters: vocab 2048, 8 layers, 8 heads (head_dim 64),
width 512, context 512. The feed-forward width is 4 x 512 = 2048 (GELU). The embedding table
(2048 x 512, tied with the output head) is about 1.05M; the transformer blocks hold most of the rest.

KV cache: forward() takes an optional past (one (k, v) pair per layer) and use_cache flag. With
use_cache=False (the default) the call and its return value are the same as before, so training
and old checkpoints are unaffected. With use_cache=True it returns (logits, loss, presents) so the
generator can feed one new token at a time instead of recomputing the whole context.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PipConfig:
    vocab_size: int = 2048     # byte-level BPE vocabulary (256 bytes + merges)
    block_size: int = 512      # context length (Pip-Medium)
    n_layer: int = 8           # Pip-Medium depth
    n_head: int = 8            # head_dim = n_embd / n_head = 64
    n_embd: int = 512          # width (Pip-Medium); must be divisible by n_head
    dropout: float = 0.1
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: PipConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, past=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None
        S = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if past is None:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        else:
            # New query row i sits at absolute position (S - T) + i and may see keys 0..that position.
            rows = torch.arange(T, device=x.device).view(T, 1) + (S - T)
            cols = torch.arange(S, device=x.device).view(1, S)
            allowed = cols <= rows
            att = att.masked_fill(~allowed.view(1, 1, T, S), float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), present


class MLP(nn.Module):
    def __init__(self, cfg: PipConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: PipConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, past=None, use_cache=False):
        a, present = self.attn(self.ln1(x), past, use_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class PipLM(nn.Module):
    def __init__(self, cfg: PipConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past=None, use_cache=False, pos_offset=0):
        B, T = idx.shape
        assert pos_offset + T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        presents = []
        for i, block in enumerate(self.blocks):
            layer_past = past[i] if past is not None else None
            x, present = block(x, layer_past, use_cache)
            presents.append(present)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, loss, presents
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
