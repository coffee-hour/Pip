"""Passphrase lock for Pip. Standard library only (hashlib, hmac, base64, secrets, json, os).

What this does: stores a salted PBKDF2-SHA256 hash of the master passphrase in vault.json next to
pip_companion.py. The companion asks for the passphrase before it loads memory or lets you use
the session.

What this does NOT do: it does not encrypt memory.json or pip_files/. Anyone who opens the drive
can still read those files directly. Real at-rest encryption needs a vetted library, and this
project is standard library only, so do not rely on this lock for sensitive data on a lost drive.

vault.json format: {"v": 1, "alg": "pbkdf2-sha256", "iterations": N, "salt": b64, "hash": b64}
"""
import base64
import hashlib
import hmac
import json
import os
import secrets

ROOT = os.path.dirname(os.path.abspath(__file__))
VAULT_PATH = os.path.join(ROOT, "vault.json")
ITERATIONS = 200000
MIN_LEN = 8


def vault_exists():
    return os.path.exists(VAULT_PATH)


def _derive(passphrase, salt, iterations):
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations)


def _load():
    """Return the stored vault dict, or None if the file is missing or unreadable."""
    try:
        with open(VAULT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("alg") != "pbkdf2-sha256":
            return None
        int(data["iterations"])
        base64.b64decode(data["salt"], validate=True)
        base64.b64decode(data["hash"], validate=True)
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def set_passphrase(passphrase):
    """Store a new passphrase. Raises ValueError if it is too short."""
    if not isinstance(passphrase, str) or len(passphrase) < MIN_LEN:
        raise ValueError("passphrase must be at least " + str(MIN_LEN) + " characters")
    salt = secrets.token_bytes(16)
    digest = _derive(passphrase, salt, ITERATIONS)
    data = {
        "v": 1,
        "alg": "pbkdf2-sha256",
        "iterations": ITERATIONS,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(digest).decode("ascii"),
    }
    tmp = VAULT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, VAULT_PATH)


def check_passphrase(passphrase):
    """True if no vault is set, or if the passphrase matches. False on mismatch or a bad vault file."""
    if not vault_exists():
        return True
    data = _load()
    if data is None or not isinstance(passphrase, str):
        return False
    salt = base64.b64decode(data["salt"])
    expected = base64.b64decode(data["hash"])
    got = _derive(passphrase, salt, int(data["iterations"]))
    return hmac.compare_digest(got, expected)


def remove_passphrase():
    """Delete the vault file. Caller must have checked the current passphrase first."""
    if vault_exists():
        os.remove(VAULT_PATH)
