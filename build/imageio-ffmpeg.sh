#!/usr/bin/env bash
# Install imageio-ffmpeg for VideoHelperSuite custom node video output.
#
# Idempotent: skips if already importable.

set -euo pipefail

if [ -z "${VENV:-}" ]; then
    VENV="$HOME/ComfyUI/.venv"
    echo "[ff] VENV env var not set — using default: $VENV"
fi

if "$VENV/bin/python" -c "import imageio_ffmpeg" 2>/dev/null; then
    echo "[ff] imageio_ffmpeg already installed — skipping"
    exit 0
fi

echo "[ff] installing imageio-ffmpeg..."
"$VENV/bin/pip" install imageio-ffmpeg 2>&1 | tail -3
"$VENV/bin/python" -c "import imageio_ffmpeg; print(f'[ff] version {imageio_ffmpeg.__version__}')" 2>&1 | grep -v "FutureWarning" | grep -v "pynvml"
echo "[ff] done"
