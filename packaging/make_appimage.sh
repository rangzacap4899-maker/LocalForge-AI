#!/usr/bin/env bash
# Build a self-contained LocalForge AI AppImage (Python + Tk + app + llama-server).
# Output: packaging/build-appimage/dist/LocalForge-AI-<date>-x86_64.AppImage
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-appimage"
DIST_DIR="$BUILD_DIR/dist"
APPDIR="$BUILD_DIR/AppDir"
TOOLS_DIR="$BUILD_DIR/tools"
ARCH="x86_64"
BACKEND="vulkan"
LLAMA_TAG=""
PBS_TAG=""

usage() {
  cat <<'EOF'
Usage: ./packaging/make_appimage.sh [options]

Options:
  --cpu            Bundle the CPU llama-server build instead of Vulkan
  --llama-tag TAG  Pin a specific llama.cpp release tag (default: latest)
  --python-tag TAG Pin the python-build-standalone release tag (default: latest)
  --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) BACKEND="cpu"; shift ;;
    --llama-tag) LLAMA_TAG="${2:-}"; shift 2 ;;
    --python-tag) PBS_TAG="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$DIST_DIR" "$TOOLS_DIR"

curl_dl() { # url outfile
  if ! curl -fsSL -o "$2" "$1"; then
    echo "!! Download failed: $1"
    exit 1
  fi
}

github_latest_tag() { # owner/repo
  curl -fsSL -H "User-Agent: LocalForge-AI" "https://api.github.com/repos/$1/releases/latest" \
    | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/'
}

echo "==> 1/5 Resolving versions..."
if [ -z "$PBS_TAG" ]; then
  PBS_TAG=$(github_latest_tag astral-sh/python-build-standalone || echo "20260807")
  echo "    python-build-standalone tag: $PBS_TAG"
fi
if [ -z "$LLAMA_TAG" ]; then
  LLAMA_TAG=$(github_latest_tag ggml-org/llama.cpp || echo "b10312")
  echo "    llama.cpp tag: $LLAMA_TAG"
fi
APP_VERSION="$(date +%Y-%m-%d)"
APP_NAME="LocalForge-AI-$APP_VERSION-$ARCH.AppImage"

echo "==> 2/5 Fetching build tools..."
APPIMAGETOOL="$TOOLS_DIR/appimagetool-x86_64.AppImage"
if [ ! -x "$APPIMAGETOOL" ]; then
  echo "    Downloading appimagetool ..."
  curl_dl "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" "$APPIMAGETOOL"
  chmod +x "$APPIMAGETOOL"
fi

echo "==> 3/5 Bundling Python + Tk (python-build-standalone)..."
PBS_ASSET="cpython-3.12.13+$PBS_TAG-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PBS_TARBALL="$BUILD_DIR/$PBS_ASSET"
if [ ! -f "$PBS_TARBALL" ]; then
  curl_dl "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/$PBS_ASSET" "$PBS_TARBALL"
fi
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr"
tar -xzf "$PBS_TARBALL" --strip-components=1 -C "$APPDIR/usr"
PYTHON_BIN="$APPDIR/usr/bin/python3"
PYVER="$(ls "$APPDIR/usr/lib" | grep -E '^python3' | head -1)"
if [ -z "$PYVER" ]; then
  echo "!! Could not find python site-packages dir in the standalone build"
  exit 1
fi
SITE_PACKAGES="$APPDIR/usr/lib/$PYVER/site-packages"
mkdir -p "$SITE_PACKAGES"
"$PYTHON_BIN" -m pip install --quiet --target "$SITE_PACKAGES" 'customtkinter>=5.2,<6'

echo "==> 4/5 Copying app + llama-server..."
mkdir -p "$APPDIR/opt/localforge"
cp -r "$REPO_ROOT/python" "$APPDIR/opt/localforge/python"
rm -rf "$APPDIR/opt/localforge/python/__pycache__"
LLAMA_BIN="$APPDIR/opt/localforge/runtime/llama.cpp/build-vulkan/bin"
mkdir -p "$LLAMA_BIN"
LLAMA_ASSET="llama-$LLAMA_TAG-bin-ubuntu-$BACKEND-x64.tar.gz"
curl_dl "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$LLAMA_ASSET" "$BUILD_DIR/$LLAMA_ASSET"
tar -xzf "$BUILD_DIR/$LLAMA_ASSET" --strip-components=1 -C "$LLAMA_BIN"
chmod +x "$LLAMA_BIN/llama-server"
echo "    llama-server $LLAMA_TAG ($BACKEND) bundled"

echo "==> 5/5 Assembling AppImage..."
cp "$SCRIPT_DIR/localforge-ai.desktop" "$APPDIR/localforge-ai.desktop"
cp "$SCRIPT_DIR/icons/localforge-ai.svg" "$APPDIR/localforge-ai.svg"
cp "$APPDIR/localforge-ai.svg" "$APPDIR/.DirIcon"

cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
set -u
HERE="$(dirname "$(readlink -f "$0")")"

export LOCALFORGE_API_URL="${LOCALFORGE_API_URL:-${GEMMA_API_URL:-http://127.0.0.1:8080}}"
export LOCALFORGE_WORKSPACE="${LOCALFORGE_WORKSPACE:-${GEMMA_WORKSPACE:-$HOME}}"
export LOCALFORGE_MODEL_ROOT="${LOCALFORGE_MODEL_ROOT:-${GEMMA_MODEL_ROOT:-$HOME/.local/share/localforge-ai/models}}"
export LLAMA_SERVER_BIN="$HERE/opt/localforge/runtime/llama.cpp/build-vulkan/bin/llama-server"

# Point Tcl/Tk at the Tcl/Tk support files bundled in the AppImage.
for dir in "$HERE"/usr/lib/tcl* "$HERE"/usr/lib/tk*; do
  if [ -d "$dir" ]; then
    case "$(basename "$dir")" in
      tcl*) export TCL_LIBRARY="$dir" ;;
      tk*) export TK_LIBRARY="$dir" ;;
    esac
  fi
done

mkdir -p "$LOCALFORGE_MODEL_ROOT"
# Note: this python-build-standalone build strips the script dir from sys.path
# even with plain invocation; point PYTHONPATH at the bundled app explicitly.
export PYTHONPATH="$HERE/opt/localforge/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/usr/bin/python3" -P "$HERE/opt/localforge/python/chatbot_app.py"
EOF
chmod +x "$APPDIR/AppRun"

APPIMAGE_TMP="$BUILD_DIR/appimagetool-extract"
rm -rf "$APPIMAGE_TMP"
mkdir -p "$APPIMAGE_TMP"
(cd "$APPIMAGE_TMP" && "$APPIMAGETOOL" --appimage-extract >/dev/null)
"$APPIMAGE_TMP/squashfs-root/AppRun" "$APPDIR" "$DIST_DIR/$APP_NAME"
echo "==> Built $DIST_DIR/$APP_NAME"
