#!/usr/bin/env bash
# Resolve opencv 3-way conflict (opencv-python + opencv-python-headless + opencv-contrib-python).
#
# Custom node installers on ComfyUI commonly add opencv-python or opencv-python-headless
# alongside opencv-contrib-python. They overwrite each other's cv2/ directory and you end
# up with a broken half-install — symptom: "Cannot import name 'guidedFilter' from 'cv2.ximgproc'"
# (LayerStyle node fails).
#
# Fix: uninstall all opencv variants, remove leftover cv2/ dirs, install ONLY
# opencv-contrib-python (which is a superset of opencv-python).
#
# Idempotent: skips if guidedFilter import already works.

set -euo pipefail

if [ -z "${VENV:-}" ]; then
    VENV="$HOME/ComfyUI/.venv"
    echo "[opencv] VENV env var not set — using default: $VENV"
fi

# Idempotent check
if "$VENV/bin/python" -c "from cv2.ximgproc import guidedFilter" 2>/dev/null; then
    echo "[opencv] cv2.ximgproc.guidedFilter already importable — skipping"
    exit 0
fi

echo "[opencv] guidedFilter import broken — fixing"
echo "[opencv] uninstalling all opencv variants..."
"$VENV/bin/pip" uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>&1 | grep -E "Uninstall|Successfully" || true

echo "[opencv] removing leftover cv2 dirs..."
SITE="$VENV/lib/python3.12/site-packages"
rm -rf "$SITE/cv2" "$SITE/cv2.dist-info" "$SITE/opencv_python.libs" "$SITE/opencv_python_headless.libs"

echo "[opencv] installing opencv-contrib-python..."
"$VENV/bin/pip" install opencv-contrib-python 2>&1 | tail -5

echo "[opencv] verifying..."
"$VENV/bin/python" -c "
import cv2
from cv2.ximgproc import guidedFilter
print(f'[opencv] cv2 version: {cv2.__version__}')
print('[opencv] guidedFilter: OK')
" 2>&1 | grep -v "FutureWarning" | grep -v "pynvml" | tail -5

echo "[opencv] done"
