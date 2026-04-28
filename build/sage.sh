#!/usr/bin/env bash
# Verify (and rebuild if needed) SageAttention 2.2 with sm_121 native kernels.
#
# The PyPI wheel typically ships sm_80/sm_89 only. On Blackwell GB10 (sm_121)
# the kernels run in compatibility mode rather than as native sm_121 kernels.
#
# We stay on SageAttention 2.2. Sage 3 has reports of mosaic visual artifacts
# on DGX Spark in upstream issue thu-ml/SageAttention#321; we haven't tested
# Sage 3 ourselves but skipped it to avoid the risk.
#
# Build flag uses TORCH_CUDA_ARCH_LIST="12.0" because setup.py's SUPPORTED_ARCHS
# explicitly lists "12.0" but not "12.1" — sm_120 is binary-compatible with sm_121.

set -euo pipefail

if [ -z "${VENV:-}" ]; then
    VENV="$HOME/ComfyUI/.venv"
    echo "[sage] VENV env var not set — using default: $VENV"
fi
SITE="$VENV/lib/python3.12/site-packages"
SAGE_DIR="$SITE/sageattention"
BUILD_DIR="${BUILD_DIR:-/tmp/sage22_build}"

if [ ! -d "$SAGE_DIR" ]; then
    echo "[sage] SageAttention not installed in venv. Run:"
    echo "[sage]   $VENV/bin/pip install sageattention"
    echo "[sage] then re-run this script."
    exit 1
fi

# Check if existing kernels already contain sm_121 (idempotent skip)
echo "[sage] checking existing kernel architectures via cuobjdump..."
HAS_SM121=true
for f in "$SAGE_DIR"/_qattn_sm80*.so "$SAGE_DIR"/_qattn_sm89*.so; do
    [ -f "$f" ] || continue
    archs=$(/usr/local/cuda/bin/cuobjdump --list-text "$f" 2>/dev/null | grep -oE "sm_[0-9]+" | sort -u)
    echo "[sage]   $(basename "$f"): $archs"
    if ! echo "$archs" | grep -q "sm_121"; then
        HAS_SM121=false
    fi
done

if $HAS_SM121; then
    echo "[sage] existing install already has sm_121 native kernels — skipping rebuild"
    exit 0
fi

echo "[sage] rebuilding from source with TORCH_CUDA_ARCH_LIST=12.0..."
rm -rf "$BUILD_DIR"
git clone --quiet https://github.com/thu-ml/SageAttention "$BUILD_DIR"
cd "$BUILD_DIR"
git checkout --quiet v2.2.0

source "$VENV/bin/activate"
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install . --no-build-isolation 2>&1 | tail -5

echo "[sage] verifying new kernels..."
for f in "$SAGE_DIR"/_qattn_sm80*.so "$SAGE_DIR"/_qattn_sm89*.so; do
    [ -f "$f" ] || continue
    archs=$(/usr/local/cuda/bin/cuobjdump --list-text "$f" 2>/dev/null | grep -oE "sm_[0-9]+" | sort -u)
    echo "[sage]   $(basename "$f"): $archs"
done

echo "[sage] done"
