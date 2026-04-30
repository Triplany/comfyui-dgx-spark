#!/usr/bin/env bash
# Install comfy-aimdo (DynamicVRAM allocator).
#
# As of comfy-aimdo v0.3.0 (2026-04-30), PyPI ships a working aarch64
# manylinux2014 wheel — no source build needed on DGX Spark. ComfyUI
# master pins to v0.3.0 in its requirements.txt.
#
# Older v0.2.x wrappers either shipped only x86_64 .so (kit had to compile
# aimdo.so from v0.2.12 source) or required funchook+distorm where distorm
# was x86-only. Both situations are gone with v0.3.0's prebuilt aarch64
# wheel that bundles its own aarch64-compatible funchook.
#
# Why DynamicVRAM matters extra on Spark: when the box runs out of unified
# memory the whole system hangs — no graceful OOM kill, requires a hard
# power cycle to recover (may get fixed at the platform level down the
# road). DynamicVRAM staging keeps weights paging between the unified pool
# and GPU rather than fully resident, which is what makes Flux2 FULL bf16
# (94 GB raw weights) safe on a 128 GB box.
#
# Idempotent: skips if comfy-aimdo aimdo.so is already an aarch64 ELF AND
# init() works on this CUDA device.

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

verify() {
    "$VENV/bin/python" - <<'PY' 2>&1 | grep -v "FutureWarning" | grep -v "pynvml" | tail -5
import torch
torch.cuda.init()
_ = torch.zeros(1, device='cuda')
import comfy_aimdo, comfy_aimdo.control as c
ver = getattr(comfy_aimdo, '__version__', None)
if ver is None:
    try:
        from comfy_aimdo._version import __version__ as ver
    except Exception:
        ver = '?'
print(f"[aimdo] comfy-aimdo {ver} importable")
print("[aimdo] init symbols present:", [s for s in dir(c) if 'init' in s.lower()][:6])
PY
}

# Idempotent skip: if aimdo is already at >=0.3.0 with aarch64 .so and verifies.
if [ -f "$TARGET" ] && file "$TARGET" | grep -q "ARM aarch64"; then
    INSTALLED_VER=$("$VENV/bin/pip" show comfy-aimdo 2>/dev/null | awk '/^Version:/{print $2}')
    if [ -n "$INSTALLED_VER" ]; then
        # Compare against our minimum (0.3.0)
        if "$VENV/bin/python" -c "
from packaging.version import Version
import sys
sys.exit(0 if Version('$INSTALLED_VER') >= Version('0.3.0') else 1)
" 2>/dev/null; then
            echo "[aimdo] comfy-aimdo $INSTALLED_VER already installed (aarch64 .so present) — skipping"
            verify
            exit 0
        else
            echo "[aimdo] comfy-aimdo $INSTALLED_VER is older than 0.3.0 — upgrading"
        fi
    fi
fi

echo "[aimdo] installing comfy-aimdo (PyPI ships aarch64 wheel as of v0.3.0)..."
"$VENV/bin/pip" install --upgrade "comfy-aimdo>=0.3.0" 2>&1 | tail -5

# Sanity: confirm we got an aarch64 .so, not an x86 wheel by accident.
if [ -f "$TARGET" ]; then
    if ! file "$TARGET" | grep -q "ARM aarch64"; then
        echo "[aimdo] ERROR: $TARGET is not aarch64 ELF after install:" >&2
        file "$TARGET" >&2
        exit 1
    fi
else
    echo "[aimdo] ERROR: $TARGET missing after install" >&2
    exit 1
fi

echo "[aimdo] verifying..."
verify

echo "[aimdo] done"
