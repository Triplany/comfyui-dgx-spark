#!/usr/bin/env bash
# ComfyUI launcher tuned for NVIDIA DGX Spark (GB10 / sm_121 / aarch64 / 128 GB unified).
# Tested with ComfyUI v0.18.2 through v0.20.1, PyTorch 2.11.0+cu130, SageAttention 2.2,
# comfy-aimdo 0.2.12, and the 3 patches in dgx_spark_patches.sh.
set -euo pipefail

# Resolve ROOT from script location so the launcher works regardless of whose
# $HOME it lives under. install.sh places this script at $COMFY/run_dgx_spark.sh.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

# ---- Environment (Spark unified-memory + sm_121 quirks) ----
# 1 = let unified-memory fabric manage allocations (vs PyTorch's caching allocator hoarding).
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"
# Disable torch.compile / dynamo. Triton's sm_121 support has been spotty in
# community reports for Spark; disabling preemptively avoids potential crashes.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# Saturate all 20 ARM cores during VAE decode / preprocessing.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"

FLAGS=(
  --listen 0.0.0.0
  --port 8188

  # DO NOT add --gpu-only on Spark. Without it ComfyUI's LRU model cache evicts
  # correctly between runs. With it on a unified-memory system, eviction breaks
  # and the pool fills up — easy OOM in stacked workflows.

  --reserve-vram 8           # Headroom for activations on the unified pool.
  --disable-pinned-memory    # Pinned memory is meaningless on unified architecture
                             # (no PCIe transfer to accelerate, just locks pages).

  # NO --fp16-* / --bf16-* flags. ComfyUI's auto-detect path reads each model's
  # metadata and picks a dtype per model from its own logic in
  # comfy/model_management.py. In our testing, forcing --fp16-vae globally
  # produced all-black LTX 2.3 video (LTX's UNet metadata sets manual cast
  # bf16; the bf16 → fp16 cast on latents at the VAE input is the plausible
  # cause). Removing the global force eliminated the failure. README explains.

  --dont-upcast-attention    # Keep attention in whatever dtype the model uses.

  # NO --disable-mmap. Memory-mapped model loading uses the kernel page cache
  # directly so the model file isn't fully duplicated into the process's
  # working set during load.
)

# Auto-enable SageAttention if installed
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("sageattention") else 1)
PY
then
  FLAGS+=(--use-sage-attention)
fi

cd "$ROOT"
exec "$PYTHON" main.py "${FLAGS[@]}" "$@"
