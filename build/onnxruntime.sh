#!/usr/bin/env bash
# Install ONNX Runtime GPU built for sm_121 / aarch64 / CUDA 13.
#
# The PyPI onnxruntime-gpu wheel does NOT include sm_121 kernels and falls
# back to CPU on DGX Spark. DWPose, controlnet preprocessors, and any other
# ONNX models then run on CPU — slowly.
#
# We use the community-built wheel from Jay0515 on Hugging Face — the only
# sm_121/aarch64/cu13 build I'm aware of at the time of writing.
#
# Idempotent: skips if CUDAExecutionProvider already in available providers.

set -euo pipefail

if [ -z "${VENV:-}" ]; then
    VENV="$HOME/ComfyUI/.venv"
    echo "[ort] VENV env var not set — using default: $VENV"
fi
WHEEL_URL="https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl"

# Idempotent skip check
PROVIDERS=$("$VENV/bin/python" -c "
import onnxruntime
print(','.join(onnxruntime.get_available_providers()))
" 2>/dev/null | grep -v "FutureWarning" | grep -v "pynvml" || echo "")

if echo "$PROVIDERS" | grep -q CUDAExecutionProvider; then
    echo "[ort] CUDAExecutionProvider already available — skipping"
    echo "[ort] providers: $PROVIDERS"
    exit 0
fi

echo "[ort] current providers: $PROVIDERS (no CUDA)"
echo "[ort] uninstalling existing onnxruntime + onnxruntime-gpu..."
"$VENV/bin/pip" uninstall -y onnxruntime onnxruntime-gpu 2>&1 | grep -E "Uninstall|Successfully" || true

echo "[ort] installing community sm_121 wheel..."
"$VENV/bin/pip" install "$WHEEL_URL" 2>&1 | tail -5

echo "[ort] verifying..."
"$VENV/bin/python" -c "
import onnxruntime
providers = onnxruntime.get_available_providers()
print(f'[ort] version: {onnxruntime.__version__}')
print(f'[ort] providers: {providers}')
assert 'CUDAExecutionProvider' in providers, 'CUDA provider missing after install!'
print('[ort] verified')
" 2>&1 | grep -v "FutureWarning" | grep -v "pynvml" | tail -5

echo "[ort] done"
