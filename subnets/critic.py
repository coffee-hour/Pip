"""Critic: prefrontal filter. Scores a reply for voice and authenticity before it is shown.

Checks are rule based: repetition, empty or runaway output, leaked role markers, and
generic assistant boilerplate. Returns a score in [0, 1] and the list of issues found.
"""
import re
from dataclasses import dataclass, field

_BOILERPLATE = re.compile(r"as an ai( language model)?|i am just a|i cannot feel|i'm just an ai", re.I)
_ROLE_LEAK = re.compile(r"^\s*(User|Pip)\s*:", re.M)


@dataclass
class Verdict:
    score: float
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.score >= 0.6


def evaluate(reply: str, max_chars: int = 1200) -> Verdict:
    issues = []
    text = reply.strip()
    if not text:
        return Verdict(0.0, ["empty"])
    score = 1.0
    if len(text) > max_chars:
        issues.append("too long")
        score -= 0.2
    words = text.lower().split()
    if len(words) >= 8:
        ratio = len(set(words)) / len(words)
        if ratio < 0.4:
            issues.append("repetitive")
            score -= 0.4
    if _ROLE_LEAK.search(text):
        issues.append("role marker leaked")
        score -= 0.3
    if _BOILERPLATE.search(text):
        issues.append("generic assistant boilerplate")
        score -= 0.3
    return Verdict(max(0.0, round(score, 3)), issues)
