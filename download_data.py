"""Download the TinyStories validation split (about 19 MB of plain text) into input.txt.

Source: https://huggingface.co/datasets/roneneldan/TinyStories (CDLA-Sharing-1.0 license).
Usage: python download_data.py
Overwrites input.txt. Standard library only.
After running, train with --fresh so the old checkpoint is not reused.
"""
import os
import urllib.request

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")


def main():
    print(f"downloading {URL}")
    tmp = OUT + ".part"
    req = urllib.request.Request(URL, headers={"User-Agent": "Pip-download-data"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        expected = int(resp.headers.get("Content-Length", 0) or 0)
        size = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
            print(f"\r{size / 1e6:.1f} MB", end="", flush=True)
    print()
    if expected and size != expected:
        os.remove(tmp)
        raise SystemExit(f"incomplete download: {size} of {expected} bytes")
    with open(tmp, "r", encoding="utf-8") as f:
        f.read()  # confirm it decodes as UTF-8
    os.replace(tmp, OUT)
    print(f"saved {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
