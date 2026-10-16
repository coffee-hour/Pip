#!/usr/bin/env bash
# Set up Pip on Python 3.11 or 3.12 with CPU-only PyTorch (no CUDA downloads, no source builds).
set -euo pipefail

PY=""
for candidate in python3.11 python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    case "$ver" in
      3.11|3.12) PY="$candidate"; break ;;
    esac
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: Python 3.11 or 3.12 is required. Found: $(python3 --version 2>&1 || echo none)."
  echo "Install one with pyenv (pyenv install 3.11) or use the devcontainer."
  exit 1
fi

echo "Using $PY ($($PY --version))"
"$PY" -m pip install --upgrade pip

# CPU-only wheel index: small download, no CUDA, no compilation.
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu

"$PY" -m pip install -r requirements.txt

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
echo "Pip setup complete. Next: python train.py --data input.txt --steps 1000"
