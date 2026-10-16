#!/usr/bin/env bash
# Pip setup: installs numpy + CPU-only torch on Python 3.11/3.12, then (unless --install-only) starts training.
#   bash setup.sh                 install dependencies, then train
#   bash setup.sh --install-only  install dependencies and exit (used by the devcontainer)
set -euo pipefail

INSTALL_ONLY=0
if [ "${1:-}" = "--install-only" ]; then INSTALL_ONLY=1; fi

stage() { echo; echo "==> [$1] $2"; }

stage 1/5 "Finding Python 3.11 or 3.12"
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

# Use --user only if the site-packages directory is not writable.
USER_FLAG=""
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
if [ ! -w "$SITE" ]; then USER_FLAG="--user"; echo "Site-packages not writable, installing with --user"; fi

PIP="$PY -m pip install --disable-pip-version-check --progress-bar on --retries 5 --timeout 120 $USER_FLAG"

stage 2/5 "Checking whether dependencies are already installed"
if "$PY" -c 'import numpy, torch' >/dev/null 2>&1; then
  echo "numpy and torch already import. Skipping download."
else
  stage 3/5 "Installing numpy and CPU-only torch (about 200 MB, this can take a few minutes)"
  # Torch CPU wheels come from the PyTorch index; numpy and other deps come from PyPI.
  $PIP numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
fi

stage 4/5 "Installing remaining requirements from requirements.txt"
$PIP -r requirements.txt

"$PY" -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo "Dependencies ready."

if [ "$INSTALL_ONLY" = "1" ]; then
  echo "Install-only mode: not starting training. Run: python train.py --data input.txt --steps 1000"
  exit 0
fi

stage 5/5 "Starting training (5000 steps). Progress prints every 500 steps."
exec "$PY" train.py --data input.txt --out checkpoints/pip.pt --steps 5000
