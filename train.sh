#!/usr/bin/env bash
# Pip one-command launcher. Runs run_pip_pipeline.py, which merges the corpus, trains,
# validates, and auto-resumes from checkpoints/pip.pt.
#   - Installs requirements.txt if torch is missing.
#   - Downloads TinyShakespeare to input.txt only if input.txt is missing (never overwrites it).
#   - Device: CUDA, then MPS, then CPU (train.py picks the device itself).
#   - Mixed precision: bf16 autocast on CUDA (when the GPU supports it), fp32 on MPS and CPU.
# Usage: bash train.sh [pipeline flags, e.g. --steps 8000 or --fresh]
set -euo pipefail
cd "$(dirname "$0")"

DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found." >&2; exit 1
fi

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "Installing dependencies from requirements.txt..."
  "$PY" -m pip install -q -r requirements.txt
fi

if [ ! -s input.txt ]; then
  echo "input.txt is missing. Downloading TinyShakespeare..."
  curl -fsSL -o input.txt "$DATA_URL"
fi

DEVICE=$("$PY" -c "import torch; m = getattr(torch.backends, 'mps', None); print('cuda' if torch.cuda.is_available() else ('mps' if m and m.is_available() else 'cpu'))")
echo "device: $DEVICE"

exec "$PY" run_pip_pipeline.py "$@"
