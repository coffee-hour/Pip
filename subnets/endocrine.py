"""Hypothalamus: recurrent state for dopamine, serotonin, cortisol and oxytocin.

Each level is in [0, 1] and relaxes toward its baseline with an exponential half-life, so
old stimuli fade the way they would in a body. A Stimulus (from thalamus.py) nudges the
levels each turn. The levels then set sampling temperature and top-k for generation.
"""
import math
from dataclasses import dataclass, field

BASELINE = {"dopamine": 0.5, "serotonin": 0.6, "cortisol": 0.2, "oxytocin": 0.4}
HALF_LIFE_TURNS = {"dopamine": 2.0, "serotonin": 12.0, "cortisol": 5.0, "oxytocin": 8.0}


@dataclass
class Endocrine:
    levels: dict = field(default_factory=lambda: dict(BASELINE))

    def decay(self, turns: float = 1.0) -> None:
        for k, base in BASELINE.items():
            keep = 0.5 ** (turns / HALF_LIFE_TURNS[k])
            self.levels[k] = base + (self.levels[k] - base) * keep

    def stimulate(self, valence: float, arousal: float, threat: float) -> None:
        L = self.levels
        L["dopamine"] = _clip(L["dopamine"] + 0.2 * max(0.0, valence) * arousal)
        L["serotonin"] = _clip(L["serotonin"] + 0.1 * valence - 0.05 * threat)
        L["cortisol"] = _clip(L["cortisol"] + 0.3 * threat + 0.1 * max(0.0, -valence))
        L["oxytocin"] = _clip(L["oxytocin"] + 0.1 * max(0.0, valence) * (1.0 - threat))

    def tick(self, valence: float, arousal: float, threat: float) -> dict:
        """One conversational turn: decay first, then apply the new stimulus."""
        self.decay(1.0)
        self.stimulate(valence, arousal, threat)
        return dict(self.levels)

    def sampling(self, base_temp: float = 0.8, base_top_k: int = 40) -> tuple:
        """Map hormone state to (temperature, top_k). High cortisol narrows sampling; dopamine widens it."""
        L = self.levels
        temp = base_temp * (1.0 + 0.25 * L["dopamine"] - 0.2 * L["cortisol"]) 
        temp = min(1.2, max(0.4, temp))
        top_k = max(10, int(round(base_top_k * (1.0 + 0.5 * L["serotonin"] - 0.5 * L["cortisol"]))))
        return round(temp, 3), top_k


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


__all__ = ["Endocrine", "BASELINE", "HALF_LIFE_TURNS", "math"]
