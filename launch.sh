#!/usr/bin/env bash
# Pip portable launcher for Linux. Run via Pip.desktop (double-click) or: bash launch.sh
# Offline-first: nothing is downloaded unless the dependencies are missing AND no ./wheels folder exists.
#
#   1. Pick Python 3.11, 3.12, or 3.13 (drive .venv first, then system python3.x).
#   2. Check numpy, torch, transformers import. If not, make .venv here and install from ./wheels.
#   3. Start run.py on PIP_PORT (default 8000), wait until it answers, open the default browser.
#
# Env: PIP_PORT=8000   PIP_NO_BROWSER=1 to skip opening a browser.
# Log: pip_launch.log next to this script (also printed to the terminal).

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

PORT="${PIP_PORT:-8000}"
URL="http://localhost:${PORT}"
VENV="$ROOT/.venv"
WHEELS="$ROOT/wheels"
LOG="$ROOT/pip_launch.log"

say() { echo "[pip] $*"; }

HAS_DISPLAY=0
if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then HAS_DISPLAY=1; fi

# Show a popup when there is a display. Never fails the script.
notify() {
  say "$1"
  if [ "$HAS_DISPLAY" = "1" ]; then
    if command -v zenity >/dev/null 2>&1; then
      zenity --info --title="Pip" --text="$1" >/dev/null 2>&1 || true
    elif command -v notify-send >/dev/null 2>&1; then
      notify-send "Pip" "$1" >/dev/null 2>&1 || true
    fi
  fi
}

fail() {
  notify "Pip could not start: $1. Details are in $LOG"
  exit 1
}

# A file manager runs this with no terminal. Reopen in a terminal so the output is visible.
if [ ! -t 1 ] && [ "${PIP_IN_TERMINAL:-0}" != "1" ] && [ "$HAS_DISPLAY" = "1" ]; then
  for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal lxterminal xterm; do
    if command -v "$t" >/dev/null 2>&1; then
      if [ "$t" = "gnome-terminal" ]; then
        exec "$t" -- env PIP_IN_TERMINAL=1 bash "$ROOT/launch.sh"
      else
        exec "$t" -e "env PIP_IN_TERMINAL=1 bash '$ROOT/launch.sh'"
      fi
    fi
  done
fi

# Send everything to the terminal and to the log.
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

# Find a usable Python 3.11-3.13. Prints the path, or returns 1.
find_python() {
  if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; return 0; fi
  for c in python3.12 python3.11 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
      case "$v" in 3.11|3.12|3.13) command -v "$c"; return 0 ;; esac
    fi
  done
  return 1
}

# True when the deps Pip needs all import with the given python.
deps_ok() { "$1" -c 'import numpy, torch, transformers' >/dev/null 2>&1; }

# True when something already answers on the port. Uses Python, so curl is not needed.
server_up() {
  "$1" -c "import urllib.request,sys
urllib.request.urlopen('$URL/', timeout=2)" >/dev/null 2>&1
}

say "Pip launcher in $ROOT"

PY="$(find_python)" || fail "no Python 3.11, 3.12, or 3.13 was found on this computer"
say "Python: $PY ($("$PY" --version 2>&1))"

if ! deps_ok "$PY"; then
  say "Dependencies are missing for this Python."
  if [ ! -d "$VENV" ]; then
    say "Creating a portable environment at $VENV"
    "$PY" -m venv "$VENV" || fail "could not create .venv (install the python3-venv package)"
  fi
  PY="$VENV/bin/python"
  if deps_ok "$PY"; then
    say "Portable environment is ready."
  elif [ -d "$WHEELS" ]; then
    say "Installing from ./wheels with no network access"
    "$PY" -m pip install --no-index --find-links "$WHEELS" -r requirements.txt \
      || fail "installing from ./wheels failed; see the log"
  else
    say "No ./wheels folder. Trying the network once to install dependencies."
    "$PY" -m pip install -r requirements.txt \
      || fail "dependencies are missing and the install needs internet"
  fi
fi

deps_ok "$PY" || fail "numpy, torch, or transformers still do not import with $PY"
say "Dependencies OK."

SERVER_PID=""
if server_up "$PY"; then
  say "Pip is already running at $URL"
else
  say "Starting Pip at $URL"
  "$PY" "$ROOT/run.py" --port "$PORT" &
  SERVER_PID=$!
  started=0
  for i in $(seq 1 90); do
    if server_up "$PY"; then started=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      fail "Pip stopped during startup; see $LOG"
    fi
    sleep 1
  done
  [ "$started" = "1" ] || fail "Pip did not answer on $URL within 90 seconds"
fi

if [ "${PIP_NO_BROWSER:-0}" != "1" ]; then
  opened=0
  for opener in xdg-open sensible-browser x-www-browser; do
    if command -v "$opener" >/dev/null 2>&1; then
      "$opener" "$URL" >/dev/null 2>&1 && opened=1 && break
    fi
  done
  if [ "$opened" = "0" ] && command -v gio >/dev/null 2>&1; then
    gio open "$URL" >/dev/null 2>&1 && opened=1
  fi
  [ "$opened" = "1" ] || say "Could not open a browser. Open $URL yourself."
fi

say "Pip is running at $URL. Close this window to stop it."

if [ -n "$SERVER_PID" ]; then
  wait "$SERVER_PID"
fi
