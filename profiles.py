"""Pip profiles: Default (neutral) and Xavier (personal, password protected).

Standard library only. The Xavier password is read from the environment variable
PIP_XAVIER_PASSWORD. When that variable is unset or empty, the built-in default is used, so the
profile works out of the box on a thumb drive with zero setup. Set PIP_XAVIER_PASSWORD to override it.

Context files live in profiles/<name>/. Every .md file in the folder is loaded in name order and
joined into one context block, within MAX_PROFILE_CHARS.
"""
import hmac
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(ROOT, "profiles")
PASSWORD_ENV = "PIP_XAVIER_PASSWORD"
DEFAULT_XAVIER_PASSWORD = "FinnPipXavier"
DEFAULT_PROFILE = "default"
XAVIER_PROFILE = "xavier"
PROFILE_NAMES = (DEFAULT_PROFILE, XAVIER_PROFILE)
MAX_PROFILE_CHARS = 2000  # keeps the always-on context inside the small model's window


def xavier_password():
    """The Xavier password: the environment variable if set, otherwise the built-in default."""
    return os.environ.get(PASSWORD_ENV, DEFAULT_XAVIER_PASSWORD) or DEFAULT_XAVIER_PASSWORD


def check_password(candidate):
    """Constant-time check. True only when the candidate matches the Xavier password."""
    if not isinstance(candidate, str):
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), xavier_password().encode("utf-8"))


def resolve_profile(requested, authenticated):
    """Xavier is granted only to an authenticated caller. Anything else gets Default."""
    if requested == XAVIER_PROFILE and authenticated:
        return XAVIER_PROFILE
    return DEFAULT_PROFILE


def load_profile_context(name):
    """Join the .md files in profiles/<name>/ into one context string, within budget."""
    folder = os.path.join(PROFILES_DIR, name)
    if not os.path.isdir(folder):
        return ""
    parts, used = [], 0
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".md"):
            continue
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            body = " ".join(f.read().split())
        room = MAX_PROFILE_CHARS - used
        if room <= 0:
            break
        body = body[:room]
        parts.append(body)
        used += len(body) + 1
    return " ".join(parts)
