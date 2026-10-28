"""Thalamus: turn raw text into a small stimulus vector for the endocrine system.

Returns valence in [-1, 1] (negative to positive), arousal in [0, 1] (calm to intense),
and threat in [0, 1] (conflict or danger words). Lexicon based, no model call.
"""
import re
from dataclasses import dataclass

_POS = {"thanks", "great", "love", "happy", "nice", "good", "glad", "awesome", "proud", "friend"}
_NEG = {"sad", "bad", "hate", "angry", "tired", "awful", "stupid", "lonely", "hurt", "broken"}
_THREAT = {"deadline", "urgent", "fail", "failed", "danger", "threat", "panic", "emergency", "late"}
_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class Stimulus:
    valence: float
    arousal: float
    threat: float


def perceive(text: str) -> Stimulus:
    words = _WORD.findall(text.lower())
    n = max(1, len(words))
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    threat_hits = sum(w in _THREAT for w in words)
    valence = max(-1.0, min(1.0, (pos - neg) / max(1.0, pos + neg)))
    excl = min(1.0, text.count("!") * 0.25)
    caps = sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text))
    arousal = min(1.0, 0.3 * excl + 0.5 * caps + 0.2 * min(1.0, abs(valence)) + 0.1 * (pos + neg + threat_hits) / n * 5)
    threat = min(1.0, 0.5 * threat_hits + 0.2 * neg)
    return Stimulus(valence=valence, arousal=arousal, threat=threat)
