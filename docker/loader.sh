#!/usr/bin/env bash
# Launch standalone SimNIBS container with GUI support (macOS + XQuartz).
#
# Usage:
#   ./run_simnibs.sh /path/to/data

DATA_DIR="${1:-}"
IMAGE="${2:-idossha/simnibs_clean:4.6}"

if [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 <data-directory> [image-name]"
    echo "  e.g. $0 ~/subjects simnibs/simnibs:4.6"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: directory not found: $DATA_DIR"
    exit 1
fi

# Resolve to absolute path
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

# --- XQuartz checks ---

if [ ! -d /Applications/Utilities/XQuartz.app ]; then
    echo "ERROR: XQuartz is not installed."
    echo "  brew install --cask xquartz"
    exit 1
fi

# Check that "Allow connections from network clients" is enabled
nolisten=$(defaults read org.xquartz.X11 nolisten_tcp 2>/dev/null || echo "1")
if [ "$nolisten" != "0" ]; then
    echo "ERROR: XQuartz network clients are disabled."
    echo "  Open XQuartz → Preferences → Security → check 'Allow connections from network clients'"
    echo "  Then restart XQuartz."
    exit 1
fi

# Start XQuartz if not running
if ! pgrep -q Xquartz; then
    echo "Starting XQuartz..."
    open -a XQuartz
    sleep 2
fi

# Allow local connections (target the local XQuartz display, not Docker's)
DISPLAY=:0 xhost +localhost 2>&1 || echo "WARN: xhost failed — GUI may not work"

# --- Launch container ---

echo "Mounting: $DATA_DIR → /data"
echo "Image:    $IMAGE"
echo ""

exec docker run --rm -it \
    --platform linux/amd64 \
    -e DISPLAY=host.docker.internal:0 \
    -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e QT_AUTO_SCREEN_SCALE_FACTOR=0 \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$DATA_DIR":/data \
    "$IMAGE"
