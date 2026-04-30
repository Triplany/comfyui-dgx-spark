#!/usr/bin/env bash
# ComfyUI launcher tuned for NVIDIA DGX Spark (GB10 / sm_121 / aarch64 / 128 GB unified).
# Tested against ComfyUI master @ commit b6332446 (2026-04-30, post-v0.20.1) with
# PyTorch 2.11.0+cu130, comfy-aimdo 0.3.0 (PyPI aarch64 wheel), and Patch 2 from
# dgx_spark_patches.sh applied.
#
# DESIGN: minimal. ComfyUI master post-PR #10953 (async offload default) and
# PR #13603 (dynamicVRAM + --cache-ram) handles most of the memory work itself.
# Earlier versions of this launcher carried more flags and env tweaks; bench
# data showed most are now redundant or noise-level. See "What was dropped"
# at the bottom for the audit trail and how to re-add if needed.
set -euo pipefail

# Resolve ROOT from script location so the launcher works regardless of $HOME.
# install.sh places this script at $COMFY/run_dgx_spark.sh.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

# ---- Environment (only what the bench data justifies on current ComfyUI) ----
# PYTORCH_NO_CUDA_MEMORY_CACHING=1: tell PyTorch's caching allocator not to
# hoard VRAM. On unified memory, the caching allocator can hold pages that the
# kernel could otherwise hand out for OS / other processes. On a discrete-GPU
# box this is harmless; on Spark it bounds the runaway behavior we hit during
# stacked workflows.
export PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-1}"

FLAGS=(
  --listen 0.0.0.0           # Bind for remote access (set to 127.0.0.1 if local-only).
  --port 8188

  --reserve-vram 8           # Headroom for activations on the unified pool.
                             # ComfyUI's auto-reserve is a smaller default; bumping
                             # this avoids late-iteration allocation failures during
                             # heavy workflows.

  --disable-pinned-memory    # Pinned memory targets PCIe transfer to a discrete GPU.
                             # On unified memory there is no transfer to accelerate,
                             # just locked pages. PR #13221 fixed pinned-mem accounting
                             # but the "unified memory makes pinning pointless"
                             # property is unchanged.

  # NO --gpu-only on Spark. Without it ComfyUI's LRU model cache evicts cleanly
  # between runs. With it + unified memory, the pool fills until OOM (we hit
  # 108+ GB before bisecting this).

  # NO --disable-dynamic-vram. Flux2 FULL bf16 is 94 GB raw weight footprint;
  # without DynamicVRAM staging the working set is all-resident and any
  # activation churn pushes past the 110 GB safety ceiling.

  # NO --fp16-* / --bf16-* flags. ComfyUI auto-picks dtype per model from each
  # model's quantization metadata. Forcing --fp16-vae globally silently broke
  # LTX 2.3 (all-black video) in our testing; auto-detect does the right thing.
)

cd "$ROOT"
exec "$PYTHON" main.py "${FLAGS[@]}" "$@"

# ---- What was dropped (and how to re-add) ----
#
# --use-sage-attention
#   Bench (2026-04-30) on Flux1-dev and Flux2 FULL bf16 showed sage delta within
#   1-2% of pytorch attention per-step — both directions, noise-level. Cold init
#   adds ~3s. ComfyUI master defaults to pytorch attention which is fine on
#   Spark with PyTorch 2.11+cu130. Re-add with: append --use-sage-attention to
#   FLAGS.
#
# --dont-upcast-attention
#   Defensive flag; ComfyUI's per-model dtype handling is correct on current
#   master without it. Re-add if you see attention upcasting hurt perf or
#   change outputs.
#
# TORCH_COMPILE_DISABLE / TORCHDYNAMO_DISABLE env vars
#   ComfyUI core doesn't use torch.compile. These were defensive against
#   custom_nodes that might. If a custom_node crashes due to Triton/sm_121 lack
#   of support, re-export both as 1 before launch.
#
# OMP_NUM_THREADS=20
#   The OS / OpenMP runtime picks reasonable thread counts on its own. Re-add
#   if you observe VAE decode or preprocessing CPU-bound at <20 cores.
