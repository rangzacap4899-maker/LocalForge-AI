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
  --skip-deps        Skip installing system packages (only set up venv + desktop entry)
  --llama MODE       Install llama-server from prebuilt binaries: auto | vulkan | cpu | skip
                     (default: auto -- pick vulkan when a Vulkan driver is present, else cpu)
  --llama-tag TAG    Pin a specific llama.cpp release tag (default: latest release)
  --help             Show this help
EOF
}

SKIP_DEPS=0
LLAMA_MODE=auto
LLAMA_TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-deps) SKIP_DEPS=1; shift ;;
    --llama) LLAMA_MODE="${2:-auto}"; shift 2 ;;
    --llama=*) LLAMA_MODE="${1#*=}"; shift ;;
    --llama-tag) LLAMA_TAG="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done
case "$LLAMA_MODE" in
  auto|vulkan|cpu|skip) ;;
  *) echo "!! Invalid --llama mode: $LLAMA_MODE (expected auto|vulkan|cpu|skip)"; exit 1 ;;
esac

FALLBACK_LLAMA_TAG="b10312"
LLAMA_BIN_DIR="$APP_DIR/runtime/llama.cpp/build-vulkan/bin"

# --- llama.cpp prebuilt helpers ---------------------------------------------
detect_vulkan() {
  # True when a Vulkan driver (ICD) is installed or vulkaninfo reports a GPU.
  local icd
  for icd in /usr/share/vulkan/icd.d/*.json /usr/lib/vulkan/icd.d/*.json /usr/lib64/vulkan/icd.d/*.json; do
    [ -e "$icd" ] && return 0
  done
  command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>/dev/null | grep -qi "gpu" && return 0
  return 1
}

fetch_llama_tag() {
  # Query the latest llama.cpp release tag, falling back to a known-good tag.
  local tag=""
  if [ -x "$PYTHON_BIN" ]; then
    tag=$("$PYTHON_BIN" -c "import urllib.request,json;print(json.load(urllib.request.urlopen('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'))['tag_name'])" 2>/dev/null || true)
  fi
  if [ -z "$tag" ]; then
    tag=$(curl -fsSL -H "User-Agent: LocalForge-AI" https://api.github.com/repos/ggml-org/llama.cpp/releases/latest 2>/dev/null \
      | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')
  fi
  if [ -z "$tag" ]; then
    echo "    !! Could not query latest llama.cpp release; using known tag $FALLBACK_LLAMA_TAG"
    tag="$FALLBACK_LLAMA_TAG"
  fi
  echo "$tag"
}

install_llama_server() {
  local backend="$1"
  local tag="$2"
  local arch asset url tmp tarball
  case "$(uname -m)" in
    x86_64) arch="x64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) echo "!! Unsupported architecture: $(uname -m) (prebuilt llama.cpp builds exist for x64/arm64)"; exit 1 ;;
  esac
  if [[ "$backend" == "vulkan" ]]; then
    asset="llama-$tag-bin-ubuntu-vulkan-$arch.tar.gz"
  else
    asset="llama-$tag-bin-ubuntu-$arch.tar.gz"
  fi
  url="https://github.com/ggml-org/llama.cpp/releases/download/$tag/$asset"
  echo "==> Downloading llama-server ($tag, $backend) from llama.cpp releases..."
  echo "    $url"
  tmp=$(mktemp -d)
  tarball="$tmp/$asset"
  if ! curl -fsSL -o "$tarball" "$url"; then
    echo "!! Download failed. Retry with: ./install.sh --llama $backend --llama-tag $tag"
    rm -rf "$tmp"
    exit 1
  fi
  echo "==> Extracting to $LLAMA_BIN_DIR ..."
  mkdir -p "$LLAMA_BIN_DIR"
  if ! tar -xzf "$tarball" --strip-components=1 -C "$LLAMA_BIN_DIR"; then
    echo "!! Failed to extract $asset"
    rm -rf "$tmp"
    exit 1
  fi
  rm -rf "$tmp"
  chmod +x "$LLAMA_BIN_DIR/llama-server" 2>/dev/null || true
  echo "==> Verifying $LLAMA_BIN_DIR/llama-server ..."
  if LD_LIBRARY_PATH="$LLAMA_BIN_DIR" "$LLAMA_BIN_DIR/llama-server" --version >/dev/null 2>&1; then
    echo "    llama-server $tag ready"
    return 0
  fi
  echo "!! llama-server binary did not run (missing libraries?)."
  echo "   Try the CPU build: ./install.sh --llama cpu, or build from source."
  exit 1
}

# --- 1. System dependencies -----------------------------------------------
if [[ "$SKIP_DEPS" -eq 0 ]]; then
  if command -v dnf >/dev/null 2>&1; then
    PKG_MGR="sudo dnf install -y"
    PKGS=(
      python3 python3-pip python3-virtualenv python3-tkinter
      zenity aria2 libnotify xdg-utils wl-clipboard
      pipewire-utils speech-dispatcher espeak-ng
      google-noto-sans-thai-fonts google-noto-sans-cjk-fonts
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

# --- 3. llama-server ---------------------------------------------------------
if [[ "$LLAMA_MODE" == "skip" ]]; then
  echo "==> Skipping llama-server install (--llama skip)"
elif [ -x "$LLAMA_BIN_DIR/llama-server" ]; then
  echo "==> llama-server already present: $LLAMA_BIN_DIR/llama-server"
  echo "    (remove it, or pass --llama vulkan/cpu to reinstall)"
else
  if [[ "$LLAMA_MODE" == "auto" ]]; then
    if detect_vulkan; then
      LLAMA_MODE="vulkan"
    else
      echo "    No Vulkan driver detected; falling back to CPU build."
      LLAMA_MODE="cpu"
    fi
  fi
  if [ -z "$LLAMA_TAG" ]; then
    echo "==> Resolving latest llama.cpp release tag..."
    LLAMA_TAG=$(fetch_llama_tag)
    echo "    Using tag $LLAMA_TAG"
  fi
  install_llama_server "$LLAMA_MODE" "$LLAMA_TAG"
fi

# --- 4. Desktop entry --------------------------------------------------------
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

# --- 5. Sanity check ----------------------------------------------------------
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
    The app will open llama-server from:
    $LLAMA_BIN_DIR
    (override the binary path with LLAMA_SERVER_BIN).
EOF
