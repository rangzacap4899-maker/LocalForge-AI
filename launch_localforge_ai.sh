#!/usr/bin/env bash
set -u

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
API_URL="${LOCALFORGE_API_URL:-${GEMMA_API_URL:-http://127.0.0.1:8080}}"

show_error() {
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="LocalForge AI" --width=460 --text="$1"
  else
    printf '%s\n' "$1" >&2
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  show_error "ไม่พบ Python environment ที่ $PYTHON_BIN\n\nเปิด Terminal แล้วรัน:\ncd $APP_DIR && python3 -m venv .venv && .venv/bin/pip install -r python/chatbot_requirements.txt"
  exit 1
fi

export LOCALFORGE_API_URL="$API_URL"
export LOCALFORGE_WORKSPACE="${LOCALFORGE_WORKSPACE:-${GEMMA_WORKSPACE:-/home/addrang}}"
cd "$APP_DIR"
"$PYTHON_BIN" "$APP_DIR/python/chatbot_app.py"
