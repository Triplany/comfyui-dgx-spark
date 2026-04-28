# Changelog

All notable changes to this repo. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Initial drop

Bundles the DGX Spark optimizations developed during a hands-on session porting ComfyUI to a GB10 / sm_121 / aarch64 box. The pieces here came from a mix of community advice and patches I needed to write to get my workloads running reliably.

#### Added

- **`build/aimdo.sh`** — builds comfy-aimdo's `aimdo.so` from source for aarch64. Uses v0.2.12 to match the installed Python wrapper API and avoid the funchook+distorm dep on master (distorm is x86-only). Enables ComfyUI's DynamicVRAM mode on Spark.
- **`build/sage.sh`** — verifies SageAttention 2.2's installed `_qattn_sm*.so` files contain sm_121 native kernels via `cuobjdump`. Rebuilds from source with `TORCH_CUDA_ARCH_LIST="12.0"` if not (sm_120 is binary-compatible with sm_121). Stays on SageAttention 2.2 — Sage 3 has reports of mosaic artifacts on Spark per upstream issue thu-ml/SageAttention#321; we haven't tested Sage 3 ourselves and skipped it to avoid the risk.
- **`build/onnxruntime.sh`** — installs the community-built `onnxruntime-gpu-aarch64-cuda13-sm121` wheel from Hugging Face. The PyPI `onnxruntime-gpu` falls back to CPU on Spark; DWPose, controlnet preprocessors, and any other ONNX models then run on CPU instead of GPU.
- **`build/opencv.sh`** — resolves the opencv 3-way conflict (`opencv-python` + `opencv-python-headless` + `opencv-contrib-python` overwriting each other's `cv2/` dir) that breaks LayerStyle nodes' `guidedFilter` import.
- **`build/imageio-ffmpeg.sh`** — installs the dep VideoHelperSuite needs.
- **`dgx_spark_patches.sh`** — three idempotent ComfyUI source patches:
  - **Patch 1**: `comfy/model_management.py` — replace `torch.cuda.mem_get_info()` with `psutil.virtual_memory().available` for accurate free-memory reporting on unified memory. The cuda call ignores reclaimable OS page cache and under-reports free memory by tens of GB on Spark; ComfyUI's load/unload heuristics use that figure to decide whether to evict models, so the bogus low number caused panic-eviction of models that didn't need to go (verified observation: unpatched ComfyUI was unloading models when there was plenty of free memory). The patched `psutil` call reflects what the kernel can actually hand out, and the eviction churn stops. Unified-memory-specific — don't apply on discrete-GPU systems.
  - **Patch 2**: `comfy_api/latest/_input_impl/video_types.py` — `np.nan_to_num` clamp before AAC audio encoding. Workaround for LTX 2.3's audio VAE producing NaN/Inf values that cause `avcodec_send_frame() returned 22` crashes (upstream issue Lightricks/ComfyUI-LTXVideo#430).
  - **Patch 3**: `comfy/ldm/lightricks/vae/audio_vae.py` — skip the `free_memory()` call in audio VAE's `ensure_model_loaded`. On Spark with DynamicVRAM enabled, that eviction triggers an evict-then-restage cycle on the staged VideoVAE; the audio VAE is small enough (~700 MB) that on a 128 GB unified pool there's nothing to evict for. **Verified to improve LTX 2.3 generation speed** by removing the eviction-restage cycle. Note: this patch was originally hypothesized to also fix LTX 2.3 black frames; bisecting later showed the actual black-frames cause was a separate `--fp16-vae` precision-mismatch issue (see launcher rationale in README). Patch 3 is kept independently because of the verified speed improvement.
- **`run_dgx_spark.sh`** — launcher template. mmap enabled, `--disable-pinned-memory`, `OMP_NUM_THREADS=20`, `TORCHDYNAMO_DISABLE=1`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, `--reserve-vram 8`, auto-enable `--use-sage-attention`. **Deliberately omits `--fp16-*` / `--bf16-*` precision flags** — most Spark guides (SparkyUI etc.) recommend forcing fp16 globally, but in our testing that produced all-black LTX 2.3 video (LTX's UNet metadata sets `manual cast: torch.bfloat16`, so the bf16 → fp16 cast at the VAE input is the plausible failure point). Without these flags, ComfyUI auto-detects a dtype per model from its `model_management.py` logic. README's "Why no precision flags" section explains.
- **`workflows/fix-flux-lora-clip.py`** — converts Flux workflows using `LoraLoaderModelOnly` nodes (which silently drop kohya `lora_te1_*` text-encoder LoRA keys) to `LoraLoader` with proper CLIP threading. Idempotent. Backs up the workflow JSON before editing.
- **`install.sh`** — orchestrates all of the above in dependency order. Each step is idempotent.
- **`verify.sh`** — post-install health check. Confirms PyTorch + CUDA + sm_121, SageAttention native kernels, ONNX CUDA provider, aimdo init, opencv guidedFilter, imageio_ffmpeg, all 3 patches applied, launcher present.
- **`README.md`** — full setup guide including hardware/OS context, what to do, what NOT to do, why each piece exists.

#### Sources / credits

- **SparkyUI** ([ecarmen16/SparkyUI](https://github.com/ecarmen16/SparkyUI/)) — first reference for the SageAttention sm_121 build pattern
- **Jay0515 on Hugging Face** ([huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121](https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121)) — the community ONNX Runtime wheel
- **Comfy-Org/comfy-aimdo** ([github.com/Comfy-Org/comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo)) — the DynamicVRAM allocator we build for aarch64
- **thu-ml/SageAttention** ([github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)) — the attention kernels
- **Lightricks/ComfyUI-LTXVideo issue #430** — confirmed the AAC NaN bug as upstream
- **NVIDIA Developer Forums** ([forums.developer.nvidia.com/t/sage-attention-with-comfyui/350423](https://forums.developer.nvidia.com/t/sage-attention-with-comfyui/350423)) — Sage / DGX Spark discussions
