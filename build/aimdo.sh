#!/usr/bin/env bash
# Build comfy-aimdo's aimdo.so for aarch64 (DGX Spark / GB10).
#
# The PyPI wheel ships a Linux x86_64 .so. On aarch64 it doesn't load and
# ComfyUI logs "No working comfy-aimdo install detected. DynamicVRAM support
# disabled." — costing you the better VRAM management.
#
# Why DynamicVRAM matters extra on Spark: as of this writing, when the box
# runs out of unified memory the whole system hangs — no graceful OOM kill,
# requires holding the power button for a hard power cycle to recover (may get
# fixed at the platform level down the road). The whole memory-management
# approach in this repo (DynamicVRAM staging via aimdo, mmap-based loading,
# --reserve-vram 8, Patch 1's accurate free-mem reporting) is motivated by
# avoiding OOM at all costs while still keeping performance reasonable.
#
# v0.2.12 is used because the installed Python wrapper (also v0.2.12) expects
# bool init(int) — the master branch refactored to bool init(const int*, size_t)
# AND added a funchook+distorm dependency where distorm doesn't support aarch64.
#
# Idempotent: skips if aimdo.so already exists as an aarch64 ELF.

set -euo pipefail

if [ -z "${COMFY:-}" ]; then
    COMFY="$HOME/ComfyUI"
    echo "[aimdo] COMFY env var not set — using default: $COMFY"
fi
if [ -z "${VENV:-}" ]; then
    VENV="$COMFY/.venv"
    echo "[aimdo] VENV env var not set — using default: $VENV"
fi
SITE="$VENV/lib/python3.12/site-packages"
TARGET="$SITE/comfy_aimdo/aimdo.so"
BUILD_DIR="${BUILD_DIR:-/tmp/aimdo_build}"

if [ ! -d "$SITE/comfy_aimdo" ]; then
    echo "[aimdo] comfy_aimdo Python package not in venv. Install it first:"
    echo "[aimdo]   $VENV/bin/pip install comfy-aimdo"
    exit 1
fi

# Check if existing aimdo.so is already aarch64 (idempotent skip)
if [ -f "$TARGET" ]; then
    if file "$TARGET" | grep -q "ARM aarch64"; then
        echo "[aimdo] $TARGET already aarch64 ELF — skipping rebuild"
        # Sanity verify it loads + has the expected symbols
        "$VENV/bin/python" -c "
import torch; torch.cuda.init(); _=torch.zeros(1, device='cuda')
import comfy_aimdo.control as c, ctypes
c.init()
c.lib.init.argtypes=[ctypes.c_int]; c.lib.init.restype=ctypes.c_bool
assert c.lib.init(0) is True, 'lib.init(0) returned False — broken install'
print('[aimdo] verified working')
" 2>&1 | grep -v "FutureWarning" | grep -v "pynvml" | tail -5
        exit 0
    else
        echo "[aimdo] existing aimdo.so is NOT aarch64 — backing up and rebuilding"
        mv "$TARGET" "$TARGET.x86_backup"
    fi
fi

echo "[aimdo] cloning Comfy-Org/comfy-aimdo and checking out v0.2.12..."
rm -rf "$BUILD_DIR"
git clone --quiet https://github.com/Comfy-Org/comfy-aimdo "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout --quiet v0.2.12

echo "[aimdo] compiling aimdo.so for aarch64..."
mkdir -p comfy_aimdo
gcc -shared -o comfy_aimdo/aimdo.so -fPIC -O2 \
    -I/usr/local/cuda/include \
    -I src \
    -L/usr/local/cuda/targets/sbsa-linux/lib/stubs/ \
    src/*.c src-posix/*.c -lcuda

if [ ! -f comfy_aimdo/aimdo.so ]; then
    echo "[aimdo] BUILD FAILED — aimdo.so not produced" >&2
    exit 1
fi

echo "[aimdo] installing into $TARGET..."
cp comfy_aimdo/aimdo.so "$TARGET"

echo "[aimdo] verifying..."
"$VENV/bin/python" -c "
import torch; torch.cuda.init(); _=torch.zeros(1, device='cuda')
import comfy_aimdo.control as c, ctypes
c.init()
c.lib.init.argtypes=[ctypes.c_int]; c.lib.init.restype=ctypes.c_bool
assert c.lib.init(0) is True, 'lib.init(0) returned False'
print('[aimdo] verified — DynamicVRAM ready')
" 2>&1 | grep -v "FutureWarning" | grep -v "pynvml" | tail -5

echo "[aimdo] done"
