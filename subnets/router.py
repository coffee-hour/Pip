"""Brainstem: classify an input and pick the satellite route. Keyword rules only, no model call."""
import re

ROUTES = ("math", "code", "memory", "chat")

_MATH = re.compile(r"(\d\s*[-+*/^%]\s*\d)|\b(solve|integral|derivative|sqrt|sum of|calculate)\b", re.I)
_CODE = re.compile(r"```|\b(def|class|import|function|bug|traceback|python|javascript|compile|stack trace)\b", re.I)
_MEM = re.compile(r"\b(remember|last time|earlier|you said|we talked|recall)\b", re.I)


def classify(text: str) -> str:
    """Return one of ROUTES. Order matters: code beats math, memory beats chat."""
    if _CODE.search(text):
        return "code"
    if _MATH.search(text):
        return "math"
    if _MEM.search(text):
        return "memory"
    return "chat"
