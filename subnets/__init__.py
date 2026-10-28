"""Satellite sub-networks for Pip. Each module is standalone; orchestrator.py wires them to PipLM."""
from subnets import critic, dream, endocrine, hippocampus, router, specialists, thalamus

__all__ = ["critic", "dream", "endocrine", "hippocampus", "router", "specialists", "thalamus"]
