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
#   5. build/aimdo.sh          — install comfy-aimdo ≥ v0.3.0 (PyPI aarch64 wheel)
#   6. dgx_spark_patches.sh    — apply ComfyUI source patches (idempotent, version-aware)
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

# Detect ComfyUI commit + which patches will apply / skip on this tree.
# Mirrors the ancestor checks inside dgx_spark_patches.sh so the user sees the
# decision up front before any work begins.
if [ -d "$COMFY/.git" ]; then
    COMFY_HEAD=$(git -C "$COMFY" rev-parse HEAD 2>/dev/null || echo "(unknown)")
    echo "[install] ComfyUI HEAD: ${COMFY_HEAD:0:12}"

    # Recommended target: master @ b6332446 (2026-04-30) or newer.
    # That's the commit this kit's detection logic is validated against.
    RECOMMENDED_COMMIT="b6332446"
    MODERN=0
    if git -C "$COMFY" merge-base --is-ancestor "$RECOMMENDED_COMMIT" HEAD 2>/dev/null; then
        echo "[install]   ✓ at or past recommended commit $RECOMMENDED_COMMIT (2026-04-30)"
        MODERN=1
    elif git -C "$COMFY" merge-base --is-ancestor 9d8a8179 HEAD 2>/dev/null; then
        echo "[install]   PARTIAL: past async-offload PR (#10953) but pre-recommended commit ($RECOMMENDED_COMMIT)."
        echo "[install]              Some legacy patches may still apply. \`git pull\` recommended."
    else
        echo "[install]   LEGACY: pre-async-offload (pre-#10953, 2025-11-27)."
        echo "[install]              Kit will apply legacy memory-bookkeeping patches."
        echo "[install]              Strongly recommend updating ComfyUI: \`git pull\` then re-run this installer."
    fi

    # Patch 1: PR #10953 (commit 9d8a8179, 2025-11-27) — async offload by default
    if git -C "$COMFY" merge-base --is-ancestor 9d8a8179 HEAD 2>/dev/null; then
        echo "[install]   Patch 1 (mem_get_info → psutil): SKIP — upstream PR #10953 addresses this. Override with FORCE_LEGACY_PATCH1=1."
    else
        echo "[install]   Patch 1 (mem_get_info → psutil): apply (pre-PR #10953)"
    fi

    # Patch 3: PR #13486 (commit ad94d472, 2026-04-21) — audio VAE refactor removed target
    if [ ! -f "$COMFY/comfy/ldm/lightricks/vae/audio_vae.py" ] || \
       ! grep -q "def ensure_model_loaded" "$COMFY/comfy/ldm/lightricks/vae/audio_vae.py" 2>/dev/null; then
        echo "[install]   Patch 3 (audio VAE eviction): SKIP — upstream PR #13486 removed the target"
    else
        echo "[install]   Patch 3 (audio VAE eviction): apply"
    fi

    echo "[install]   Patch 2 (LTX NaN audio clamp): apply — defensive guard for upstream Lightricks LTX bug; keeps stacking until that's fixed"

    # comfy-aimdo v0.3.0 integration came in PR #13604 (commit e514119e, 2026-04-29), post-v0.20.1.
    # Older trees pin to comfy-aimdo 0.2.x; the install.sh aimdo step will install >=0.3.0 anyway,
    # but the import path may differ on legacy ComfyUI. Note for users on older trees.
    if [ "$MODERN" = "0" ] && grep -q "comfy-aimdo" "$COMFY/requirements.txt" 2>/dev/null; then
        AIMDO_PIN=$(awk -F'==' '/^comfy-aimdo/{print $2}' "$COMFY/requirements.txt" | head -1)
        if [ -n "$AIMDO_PIN" ] && [ "$AIMDO_PIN" != "0.3.0" ] && [ "${AIMDO_PIN%%.*}" = "0" ] && [ "$(echo "$AIMDO_PIN" | cut -d. -f2)" -lt 3 ] 2>/dev/null; then
            echo "[install]   NOTE: ComfyUI requirements.txt pins comfy-aimdo==$AIMDO_PIN. Kit will upgrade to >=0.3.0."
        fi
    fi
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
echo " Step 5/7 — comfy-aimdo install (PyPI aarch64 wheel, DynamicVRAM)"
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
echo "       Using async weight offloading with 2 streams       ← PR #10953 default"
echo "       aimdo: comfy-aimdo inited for GPU: NVIDIA GB10"
echo "       DynamicVRAM support detected and enabled"
echo "       comfy-aimdo version: 0.3.0                         ← (or newer)"
echo ""
echo "After every \`git pull\` of ComfyUI core, re-run:"
echo "  bash $COMFY/dgx_spark_patches.sh    # only Patch 2 actually applies on modern ComfyUI"
echo "(Idempotent — safe to run repeatedly. Will warn if upstream changed the lines we patch.)"
