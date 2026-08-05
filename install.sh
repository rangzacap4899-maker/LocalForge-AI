#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
REQUIREMENTS="$APP_DIR/python/chatbot_requirements.txt"

echo "==> LocalForge AI installer"
echo "    App dir : $APP_DIR"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --skip-deps    Skip installing system packages (only set up venv + desktop entry)
  --help         Show this help
EOF
}

SKIP_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --skip-deps) SKIP_DEPS=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

# --- 1. System dependencies -----------------------------------------------
if [[ "$SKIP_DEPS" -eq 0 ]]; then
  if command -v dnf >/dev/null 2>&1; then
    PKG_MGR="sudo dnf install -y"
    PKGS=(
      python3 python3-pip python3-virtualenv python3-tkinter
      zenity aria2 libnotify xdg-utils wl-clipboard
      pipewire-utils speech-dispatcher espeak-ng
      noto-sans-thai-fonts noto-sans-cjk-sc-fonts noto-sans-cjk-jp-fonts
    )
  elif command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="sudo apt-get install -y"
    PKGS=(
      python3 python3-pip python3-venv python3-tk
      zenity aria2 libnotify-bin xdg-utils wl-clipboard
      pipewire-bin speech-dispatcher espeak-ng
      fonts-noto-thai fonts-noto-cjk
    )
  elif command -v pacman >/dev/null 2>&1; then
    PKG_MGR="sudo pacman -S --needed --noconfirm"
    PKGS=(
      python python-pip python-virtualenv python-tkinter
      zenity aria2 libnotify xdg-utils wl-clipboard
      pipewire speech-dispatcher espeak-ng
      noto-fonts noto-fonts-cjk
    )
  else
    echo "!! Cannot detect package manager (dnf/apt/pacman)."
    echo "   Install the required packages manually, or re-run with --skip-deps"
    exit 1
  fi
  echo "==> Installing system packages..."
  echo "    $PKG_MGR ${PKGS[*]}"
  $PKG_MGR "${PKGS[@]}"
fi

# --- 2. Python virtual environment -----------------------------------------
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "==> Creating virtual environment..."
  python3 -m venv "$APP_DIR/.venv"
fi
echo "==> Installing Python requirements..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS"

# --- 3. Desktop entry --------------------------------------------------------
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/localforge-ai.desktop"
echo "==> Installing desktop entry -> $DESKTOP_FILE"
mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=LocalForge AI
Comment=ผู้ช่วย AI ภายในเครื่องสำหรับสนทนา สร้างไฟล์ และค้นเว็บ
Exec=$APP_DIR/launch_localforge_ai.sh
Icon=system-help
Terminal=false
Categories=Utility;Development;
StartupNotify=true
StartupWMClass=LocalForge AI
EOF
chmod +x "$APP_DIR/launch_localforge_ai.sh"
chmod +x "$APP_DIR/install.sh"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
fi

# --- 4. Sanity check ----------------------------------------------------------
echo "==> Verifying installation..."
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$APP_DIR/python'); import customtkinter, localforge_i18n" 2>/dev/null \
  && echo "    Python imports OK" || echo "    !! Python imports failed"
if "$PYTHON_BIN" -m unittest discover -s "$APP_DIR/python" -p 'test_*.py' >/dev/null 2>&1; then
  echo "    Unit tests OK"
else
  echo "    !! Unit tests failed (run manually to see details)"
fi

cat <<EOF

==> Done! Start LocalForge AI with:
    $APP_DIR/launch_localforge_ai.sh

    Or launch it from your application menu as "LocalForge AI".
    The app will use the llama.cpp server already running on http://127.0.0.1:8080
    (override with LOCALFORGE_API_URL).
EOF
