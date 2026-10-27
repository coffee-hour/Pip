#!/usr/bin/env bash
# Train Pip: runs run_pip_pipeline.py from this folder. Extra args are passed through.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else echo "ERROR: python3 (or python) not found." >&2; exit 1; fi
exec "$PY" run_pip_pipeline.py "$@"
