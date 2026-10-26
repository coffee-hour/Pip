"""Pip transformer architecture (decoder-only, pre-LayerNorm, weight-tied, GELU MLP).

The implementation lives in model.py. This module is the stable import path for the architecture,
so training and generation code can depend on pip_model without caring where the code is kept.

Default config is Pip-Medium (about 26.5M parameters): vocab 2048, 8 layers, 8 heads, width 512,
context 512.
"""
from model import Block, CausalSelfAttention, MLP, PipConfig, PipLM

__all__ = ["PipConfig", "PipLM", "Block", "CausalSelfAttention", "MLP"]
