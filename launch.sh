#!/usr/bin/env bash
# Pip portable launcher for Linux. Double-click Pip.desktop, or run: bash launch.sh
# Works from a USB stick. Offline-first: it never downloads anything unless you choose to.
#
# Order of work:
#   1. Pick a Python 3.11 or 3.12 (drive .venv first, then system python3.11/3.12/3.13).
#   2. Make sure numpy, torch, and transformers import. If not, install from ./wheels offline.
#      Only if that fails does it try the network, and only with your consent.
#   3. Start run.py in the background, wait until port 8000 answers, then open the browser.
#
# Set PIP_PORT to use another port (default 8000). Set PIP_NO_BROWSER=1 to skip opening a browser.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

PORT="${PIP_PORT:-8000}"
URL="http://localhost:${PORT}"
VENV="$ROOT/.venv"
WHEELS="$ROOT/wheels"
LOG="$ROOT/pip_launch.log"

# ---------- helpers ----------

say() { echo "[pip] $*"; }

# Show a dialog when there is a display, otherwise just print. Never fails the script.
notify() {
  local msg="$1"
  say "$msg"
  if command -v zenity >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    zenity --info --title="Pip" --text="$msg" >/dev/null 2>&1 || true
  elif command -v notify-send >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    notify-send "Pip" "$msg" >/dev/null 2>&1 || true
  fi
}

fail() {
  notify "Pip could not start: $1. Details are in $LOG"
  exit 1
}

# Pick a terminal to show output when launched from a file manager (no terminal attached).
open_terminal_if_needed() {
  if [ -t 1 ] || [ "${PIP_IN_TERMINAL:-0}" = "1" ]; then return 0; fi
  for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal lxterminal xterm; do
    if command -v "$t" >/dev/null 2>&1; then
      case "$t" in
        gnome-terminal) exec gnome-terminal -- env PIP_IN_TERMINAL=1 bash "$0" "$@" ;;
        *) exec "$t" -e env PIP_IN_TERMINAL=1 bash "$0" "$@" ;;
      esac
    fi
  done
  # No terminal emulator found: keep going with output in the log only.
}

# Find a Python 3.11/3.12/3.13 binary, or print nothing.
find_python() {
  if [ -x "$VENV/bin/python" ]; then
    echo "$VENV/bin/python"; return 0
  fi
  for c in python3.12 python3.11 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
      case "$v" in 3.11|3.12|3.13) command -v "$c"; return 0 ;; esac
    fi
  done
  return 1
}

# True when the deps Pip needs all import with the given python.
deps_ok() {
  "$1" -c 'import numpy, torch, transformers' >/dev/null 2>&1
}

# ---------- start ----------

# Reset the log for this run.
: > "$LOG"
exec >>"$LOG" 2>&1
open_terminal_if_needed "$@"

say "Pip launcher starting in $ROOT"

PY="$(find_python)" || fail "no Python 3.11, 3.12, or 3.13 was found on this computer"
say "Using Python: $PY ($("$PY" --version 2>&1))"

# Make a portable .venv on the drive if none exists and dependencies are missing.
if ! deps_ok "$PY"; then
  say "Dependencies are missing for this Python. Looking for an offline wheel folder..."
  if [ ! -d "$VENV" ]; then
    say "Creating a portable environment at $VENV"
    "$PY" -m venv "$VENV" || fail "could not create .venv (is python3-venv installed?)"
  fi
  PY="$VENV/bin/python"
  if deps_ok "$PY"; then
    say "Portable environment is ready."
  elif [ -d "$WHEELS" ]; then
    say "Installing from ./wheels with no network access"
    "$PY" -m pip install --no-index --find-links "$WHEELS" -r requirements.txt \
      || fail "installing from ./wheels failed; see the log"
  else
    say "No ./wheels folder and dependencies are missing. Trying the network once."
    "$PY" -m pip install -r requirements.txt \
      || fail "dependencies are missing and the install needs internet"
  fi
fi

if ! deps_ok "$PY"; then
  fail "numpy, torch, or transformers still do not import with $PY"
fi
say "Dependencies OK."

# Already running on this port? Just open the browser.
if curl -fsS -o /dev/null "$URL/api/state" 2>/dev/null || curl -fsS -o /dev/null "$URL/" 2>/dev/null; then
  say "Pip is already running at $URL"
else
  say "Starting Pip at $URL"
  "$PY" "$ROOT/run.py" --port "$PORT" &
  SERVER_PID=$!
  # Wait up to 60 seconds for the page to answer.
  for i in $(seq 1 60); do
    if curl -fsS -o /dev/null "$URL/" 2>/dev/null; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      fail "Pip stopped during startup; see $LOG"
    fi
    sleep 1
  done
fi

# Open the default browser without blocking.
if [ "${PIP_NO_BROWSER:-0}" != "1" ]; then
  for opener in xdg-open sensible-browser x-www-browser gio; do
    if command -v "$opener" >/dev/null 2>&1; then
      if [ "$opener" = "gio" ]; then
        gio open "$URL" >/dev/null 2>&1 && break
      else
        "$opener" "$URL" >/dev/null 2>&1 && break
      fi
    fi
  done
fi
say "Pip is running at $URL. Close this window to stop it."

# Keep the launcher attached so closing the window stops Pip.
if [ -n "${SERVER_PID:-}" ]; then
  wait "$SERVER_PID"
fi
