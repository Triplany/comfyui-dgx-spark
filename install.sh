#!/usr/bin/env bash
# comfyui-spark — orchestrator
#
# One-shot install of all DGX Spark optimizations and patches.
# Each step is idempotent — safe to re-run after any change or git pull.
#
# What this does, in order:
#   1. build/imageio-ffmpeg.sh — pip install (VideoHelperSuite dep)
#   2. build/opencv.sh         — fix opencv 3-way conflict
#   3. build/onnxruntime.sh    — install community sm_121 wheel
#   4. build/sage.sh           — verify SageAttention has sm_121 native, rebuild if not
#   5. build/aimdo.sh          — build comfy-aimdo aimdo.so for aarch64
#   6. dgx_spark_patches.sh    — apply 3 ComfyUI source patches (idempotent)
#   7. install run_dgx_spark.sh — copy launcher template into ComfyUI dir if missing
#
# What this does NOT do:
#   - Install ComfyUI itself (assumes you have it at $COMFY)
#   - Install PyTorch (assumes 2.11+ cu130 already in venv)
#   - Touch any model files
#   - Start or restart ComfyUI
#
# Required env vars (with defaults):
#   COMFY  — path to ComfyUI install dir (default: $HOME/ComfyUI)
#   VENV   — path to ComfyUI venv (default: $COMFY/.venv)
#
# Usage:
#   bash install.sh
#   COMFY=/path/to/ComfyUI bash install.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${COMFY:-}" ]; then
    COMFY="$HOME/ComfyUI"
    echo "[install] COMFY env var not set — using default: $COMFY"
    echo "[install]   (override with COMFY=/path/to/ComfyUI bash install.sh)"
fi
if [ -z "${VENV:-}" ]; then
    VENV="$COMFY/.venv"
    echo "[install] VENV env var not set — using default: $VENV"
fi
export COMFY VENV

echo "================================================================"
echo " comfyui-spark installer"
echo "================================================================"
echo "  ComfyUI: $COMFY"
echo "  Venv:    $VENV"
echo "  Repo:    $REPO_DIR"
echo ""

# ---- Sanity checks ----
if [ ! -d "$COMFY" ]; then
    echo "[install] ERROR: ComfyUI not found at $COMFY" >&2
    exit 1
fi
if [ ! -x "$VENV/bin/python" ]; then
    echo "[install] ERROR: venv python not found at $VENV/bin/python" >&2
    exit 1
fi

PY_VER=$("$VENV/bin/python" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
echo "[install] Python: $PY_VER"

TORCH_INFO=$("$VENV/bin/python" -c "
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')
print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '(0,0)')
" 2>&1 | grep -v FutureWarning | grep -v pynvml)
echo "[install] PyTorch info:"
echo "$TORCH_INFO" | sed 's/^/[install]   /'

if ! echo "$TORCH_INFO" | head -1 | grep -qE "^2\.(11|12|13)"; then
    echo "[install] WARN: PyTorch < 2.11 detected. Spark needs cu130 wheels." >&2
    echo "[install]   See README for install command." >&2
fi
if ! echo "$TORCH_INFO" | sed -n '4p' | grep -q "12, 1"; then
    echo "[install] WARN: GPU compute capability is not (12, 1) — this script is for DGX Spark / GB10." >&2
fi

echo ""
echo "================================================================"
echo " Step 1/7 — imageio-ffmpeg"
echo "================================================================"
bash "$REPO_DIR/build/imageio-ffmpeg.sh"

echo ""
echo "================================================================"
echo " Step 2/7 — opencv 3-way conflict cleanup"
echo "================================================================"
bash "$REPO_DIR/build/opencv.sh"

echo ""
echo "================================================================"
echo " Step 3/7 — ONNX Runtime sm_121 community wheel"
echo "================================================================"
bash "$REPO_DIR/build/onnxruntime.sh"

echo ""
echo "================================================================"
echo " Step 4/7 — SageAttention sm_121 native kernels"
echo "================================================================"
bash "$REPO_DIR/build/sage.sh"

echo ""
echo "================================================================"
echo " Step 5/7 — comfy-aimdo aarch64 build (DynamicVRAM)"
echo "================================================================"
bash "$REPO_DIR/build/aimdo.sh"

echo ""
echo "================================================================"
echo " Step 6/7 — apply ComfyUI source patches"
echo "================================================================"
# Patch script wants to run from ComfyUI dir
PATCHFILE="$COMFY/dgx_spark_patches.sh"
if [ ! -f "$PATCHFILE" ] || ! diff -q "$REPO_DIR/dgx_spark_patches.sh" "$PATCHFILE" >/dev/null 2>&1; then
    echo "[install] copying patch script into $COMFY..."
    cp "$REPO_DIR/dgx_spark_patches.sh" "$PATCHFILE"
    chmod +x "$PATCHFILE"
fi
( cd "$COMFY" && bash dgx_spark_patches.sh )

echo ""
echo "================================================================"
echo " Step 7/7 — launcher template"
echo "================================================================"
LAUNCHER="$COMFY/run_dgx_spark.sh"
if [ -f "$LAUNCHER" ]; then
    if diff -q "$REPO_DIR/run_dgx_spark.sh" "$LAUNCHER" >/dev/null 2>&1; then
        echo "[install] $LAUNCHER already up to date"
    else
        echo "[install] $LAUNCHER differs from template — NOT overwriting (you may have customized it)"
        echo "[install]   diff: diff $REPO_DIR/run_dgx_spark.sh $LAUNCHER"
    fi
else
    cp "$REPO_DIR/run_dgx_spark.sh" "$LAUNCHER"
    chmod +x "$LAUNCHER"
    echo "[install] installed launcher at $LAUNCHER"
fi

echo ""
echo "================================================================"
echo " Install complete"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Run  bash $REPO_DIR/verify.sh  to confirm everything is healthy"
echo "  2. Start ComfyUI:  bash $LAUNCHER"
echo "  3. Look for these in startup log:"
echo "       Using sage attention"
echo "       aimdo: comfy-aimdo inited for GPU: NVIDIA GB10"
echo "       DynamicVRAM support detected and enabled"
echo ""
echo "After every \`git pull\` of ComfyUI core, re-run:"
echo "  bash $COMFY/dgx_spark_patches.sh"
echo "(Idempotent — safe to run repeatedly. Will warn if upstream changed the lines we patch.)"
