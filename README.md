# comfyui-dgx-spark

ComfyUI that just works on the NVIDIA DGX Spark (GB10, sm_121, aarch64).

## Install

```bash
git clone https://github.com/Triplany/comfyui-dgx-spark
cd comfyui-dgx-spark

# Default expects ComfyUI at $HOME/ComfyUI with venv at .venv inside it.
# Override with COMFY=... if your install is elsewhere.
bash install.sh

# Confirm everything is healthy:
bash verify.sh

# Start ComfyUI:
bash $HOME/ComfyUI/run_dgx_spark.sh
```

That's it. After the launcher starts, expect these in the log:

```
Using sage attention
aimdo: comfy-aimdo inited for GPU: NVIDIA GB10 (VRAM: 124546 MB)
DynamicVRAM support detected and enabled
Starting server
To see the GUI go to: http://0.0.0.0:8188
```

## Why this exists

The Spark is a great box for image and video diffusion — 128 GB unified memory, sm_121 tensor cores — but stock ComfyUI on it had real problems for me out of the box:

- **Wan 2.2 at full fp16 would hang periodically** — sampler stuck mid-run, GPU idle, generation never finishing.
- **Flux1 at full fp16 hung the same way but more often.** I had to drop to fp8 quant just to get it to run reliably on a fresh install.
- **LTX 2.3** silently produced all-black video, and the audio side crashed.
- SageAttention's PyPI wheel ships sm_80/sm_89 kernels only — no sm_121, so it runs in compatibility mode on Spark rather than as native kernels.
- The DynamicVRAM memory allocator doesn't have a binary that works on this hardware out of the box.

A few other things I wanted to solve along the way, observed across various setups I tried:

- Memory usage creeping up gen after gen until the box OOMed.
- Switching workflows or models not cleanly unloading the previous one — stale weights piling up across model swaps.
- Flux2-dev at full quant just wouldn't run for me.

Each setup I tried would fix something and introduce something else — that's how this kit ended up the way it did, by working through the trade-offs piece by piece.

**Why memory matters so much on Spark:** as of this writing, when the box runs out of unified memory the whole system hangs — no graceful OOM kill, you have to hold the power button to hard-cycle it to recover. May get fixed at the platform level down the road, but right now everything in this kit related to memory (DynamicVRAM staging via comfy-aimdo, mmap-based loading, `--reserve-vram 8`, Patch 1's accurate free-mem reporting) is built around avoiding OOM at all costs while keeping performance reasonable.

The high memory usage was the most frustrating part. Because I run **Flux1, Flux2, Qwen 2512, Wan 2.2, and LTX 2.3** and switch between them constantly, those issues compounded fast. **None of them happen on this stack — Flux2-dev full quant runs fine now, memory stays steady gen-to-gen, and the load/unload behavior is what you'd expect: re-running the same workflow keeps the model loaded (fast warm path), switching to a different workflow or model cleanly unloads the old one (no stale weights, no creep).**

I started from advice on the NVIDIA developer forums and a few community projects that came before this one (credits at the bottom). After experimenting with settings and three small ComfyUI source patches, all of those issues went away.

My day-to-day on this stack is **Flux1, Flux2, Qwen 2512, Wan 2.2, and LTX 2.3** — all running reliably at the precision each model was designed for, no hangs, no black frames.

This repo is what I ended up with. One installer, one verifier, three idempotent patches. **Big thanks to the community.**

## What's in here

| File | What it does |
|---|---|
| `install.sh` | One-shot installer. Idempotent, safe to re-run. |
| `verify.sh` | Health check after install. |
| `run_dgx_spark.sh` | Launcher with the right flags for Spark. |
| `dgx_spark_patches.sh` | Three small ComfyUI source patches (memory bookkeeping + LTX audio fixes). |
| `build/*.sh` | sm_121 SageAttention rebuild, aarch64 comfy-aimdo build, sm_121 ONNX wheel install, opencv conflict cleanup, imageio-ffmpeg. Each idempotent. |
| `workflows/fix-flux-lora-clip.py` | Fixes Flux LoRA workflows that silently drop kohya-trained text-encoder weights. |

---

## What you need first

| Component | Required | Notes |
|-----------|----------|-------|
| Hardware | **NVIDIA DGX Spark (GB10)** | sm_121 / aarch64 / 128 GB unified memory |
| OS | Ubuntu 24.04 LTS | What I tested on |
| CUDA | 13.0+ | Spark ships with this |
| GCC | 13+ | For aimdo + Sage builds |
| Python | 3.12 | What ComfyUI's venv uses |
| **PyTorch** | **2.11.0+cu130 aarch64** | **What I tested with.** Older PyTorch versions (cu128 etc.) untested — may or may not work on Spark. cu130 is the safe path. |
| ComfyUI | any recent version | I tested v0.18.2 through v0.20.1 |

PyTorch install (if you don't have it yet):

```bash
source $HOME/ComfyUI/.venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Tested on

Verified working with these exact versions. Newer likely works too; documented for reproducibility.

| Component | Version |
|-----------|---------|
| Hardware | NVIDIA DGX Spark (GB10), 128 GB unified memory |
| Driver | 580.95+ |
| CUDA Toolkit | 13.0 |
| OS | Ubuntu 24.04 LTS, kernel `Linux 6.17.0-1014-nvidia` |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 (aarch64) |
| ComfyUI | v0.18.2 → v0.20.1 |
| comfy-aimdo | 0.2.12 (Python wrapper); `aimdo.so` built from v0.2.12 source for aarch64 |
| ComfyUI-LTXVideo | 2026-04 |
| comfyui_controlnet_aux | 1.1.5 |
| SageAttention | 2.2.0 (rebuilt for sm_121 native kernels) |
| onnxruntime-gpu | 1.25.0 (Jay0515 sm_121/aarch64/cu13 wheel) |

Workloads exercised end-to-end on this stack: **Flux1 + LoRA**, **Flux2**, **Qwen 2512**, **Wan 2.2 14B T2V** (training and inference), **LTX 2.3 T2V/I2V** (with audio), **DWPose preprocessing**, and stacked **Flux + ControlNet + LoRA + upscale + DWPose** workflows.

---

## What `install.sh` does

Each step is idempotent.

| # | Script | What it does |
|---|--------|--------------|
| 1 | `build/imageio-ffmpeg.sh` | `pip install imageio-ffmpeg` (VideoHelperSuite dep) |
| 2 | `build/opencv.sh` | Cleans the opencv 3-way conflict (`opencv-python` + `-headless` + `-contrib-python` overwriting each other's `cv2/`) |
| 3 | `build/onnxruntime.sh` | Replaces PyPI `onnxruntime-gpu` with [Jay0515's community sm_121/aarch64/cu13 wheel](https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121). Without this, DWPose / controlnet preprocessors fall back to CPU. |
| 4 | `build/sage.sh` | Verifies SageAttention 2.2's `_qattn_sm*.so` files have sm_121 native kernels. Rebuilds from source if not. **Stays on Sage 2.2 — Sage 3 has reports of mosaic visual artifacts on Spark in upstream issue #321; I haven't tested 3 myself.** |
| 5 | `build/aimdo.sh` | Builds [comfy-aimdo's](https://github.com/Comfy-Org/comfy-aimdo) `aimdo.so` from source for aarch64 (PyPI ships only x86_64). Pinned to v0.2.12 to match the Python wrapper API. |
| 6 | `dgx_spark_patches.sh` | Applies three ComfyUI source patches — see below. |
| 7 | Installs `run_dgx_spark.sh` into your ComfyUI dir (only if not already present — won't overwrite customizations). |

---

## The three patches

In `dgx_spark_patches.sh`. Each is idempotent and re-checked on every run; the script warns explicitly if upstream changed the surrounding code in a way it can't safely re-apply.

### Patch 1 — psutil free-memory reporting

**File:** `comfy/model_management.py`. ComfyUI's stock free-memory query under-reports on Spark (it doesn't account for reclaimable OS page cache). ComfyUI's load/unload heuristics use that figure to decide when to evict models — when it's artificially low, ComfyUI panics and evicts models that didn't need to go, then reloads them on the next gen. Verified observation: unpatched ComfyUI was unloading models when there was plenty of free memory. The patched query reflects what the kernel can actually hand out and the unnecessary churn stops. **Unified-memory specific** — don't apply this on a discrete-GPU box.

### Patch 2 — NaN/Inf clamp before AAC encode

**File:** `comfy_api/latest/_input_impl/video_types.py`. LTX 2.3's audio VAE sometimes outputs NaN/Inf values that the AAC encoder rejects with `avcodec_send_frame() returned 22`. Upstream issue [Lightricks/ComfyUI-LTXVideo#430](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/430) is open without a fix. We clamp NaN/Inf to 0 before the encode — audio becomes silent in those regions instead of the entire save crashing.

### Patch 3 — Skip free_memory in audio VAE load

**File:** `comfy/ldm/lightricks/vae/audio_vae.py`. ComfyUI's recent "Make ltx audio vae more native" PR added an unconditional `free_memory()` call before loading the audio VAE. On Spark with DynamicVRAM that triggers eviction of staged models (notably the VideoVAE) for a 700 MB allocation that fits trivially in the 128 GB pool. Removing the call eliminates the eviction churn — verified to improve LTX 2.3 generation speed.

---

## Launcher flags (`run_dgx_spark.sh`)

| Setting | Value | Why |
|---|---|---|
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | 1 | Let the unified-memory fabric manage allocations rather than PyTorch's caching allocator hoarding |
| `TORCH_COMPILE_DISABLE` / `TORCHDYNAMO_DISABLE` | 1 | Disabled preemptively — Triton's sm_121 support has been spotty in community reports for Spark |
| `OMP_NUM_THREADS` | 20 | Use all 20 ARM cores during VAE decode / preprocessing |
| `--reserve-vram` | 8 | Headroom for activations |
| `--disable-pinned-memory` | yes | Pinned memory is meaningless on unified architecture |
| (no `--fp16-*` / `--bf16-*` flags) | — | See "no global precision flags" below |
| `--dont-upcast-attention` | yes | Keeps attention in whatever dtype the model uses |
| `--use-sage-attention` | auto | Added if `sageattention` is importable |

### No global precision flags

Many DGX Spark guides recommend forcing `--fp16-unet --fp16-vae --fp16-text-enc` globally. **In my testing, forcing `--fp16-vae` produced all-black LTX 2.3 video** — no error, the gen looked like it succeeded. Removing the global force eliminated the failure. Without these flags, ComfyUI auto-detects a dtype per model from its metadata, which is what you want when your install runs more than one model family.

**Don't add to this launcher**, even if some guide tells you to:

- `--gpu-only` — fights unified memory, breaks LRU eviction, easy OOM
- `--disable-mmap` — forces a full read-and-copy of the model file at load time instead of letting the kernel page-map it lazily
- `--fp16-unet/vae/text-enc` (globally) — broke LTX 2.3 in my testing
- `--bf16-unet/vae/text-enc` (globally) — symmetric risk for any model that wants fp16
- `--force-fp16` — same problem family

---

## Maintenance

**After every `git pull` of ComfyUI core**, do both:

```bash
source $HOME/ComfyUI/.venv/bin/activate
cd $HOME/ComfyUI
pip install --upgrade -r requirements.txt    # picks up new comfyui-frontend-package etc.
bash dgx_spark_patches.sh                     # re-applies the three source patches
```

The pip step matters — `git pull` updates ComfyUI's Python code but the frontend (and other pip-managed deps) stay at whatever version was last installed. If you skip it you'll see a "Frontend version X is outdated" banner at startup.

The patch script is idempotent — no-ops if patches still applied, re-applies if a pull wiped them, warns explicitly if upstream changed the surrounding code.

**After ComfyUI Manager "Update All"** — same. If a custom node update reinstalls `opencv-python`, re-run `bash build/opencv.sh`.

**After upgrading PyTorch** — re-run `bash build/sage.sh` to verify (and rebuild if necessary) sm_121 native kernels.

---

## What NOT to do

- **Don't use SageAttention 3.** I haven't tried it. Upstream issue [thu-ml/SageAttention#321](https://github.com/thu-ml/SageAttention/issues/321) reports mosaic visual artifacts on Spark, so I stayed on 2.2 to avoid the risk.
- **Don't install flash-attention without testing it first.** I haven't tested it on Spark and don't include it in this stack. SageAttention 2.2 (sm_121 native after `build/sage.sh`) plus PyTorch SDPA covers attention here; if you want to try flash-attention, verify on your end before relying on it.
- **Don't follow "use NVFP4" advice from LLM guides for image diffusion.** That advice is aimed at fitting 70B-param LLMs into the unified pool. Image diffusion model files ship with their own dtype baked in; I didn't test NVFP4 quantization for image diffusion on this stack and wouldn't recommend it without separate validation.
- **Don't use `comfy-aimdo`'s master branch.** Master depends on funchook+distorm and distorm is x86-only — won't build on aarch64. `build/aimdo.sh` pins to v0.2.12 source, which builds clean.

---

## Cosmetic warnings I see in startup logs

Things that show up in the log and don't appear to break anything (this stack still passes `verify.sh` and runs all the workloads listed above with these warnings present):

- `pynvml package is deprecated` — appears at startup; deprecation notice from a transitively-loaded library.
- `GPU device discovery failed: ... /sys/class/drm/card0/device/vendor` — appears from onnxruntime's GPU probe at startup. The GPU is still detected by everything that matters: `verify.sh` confirms `CUDAExecutionProvider` is available, and DWPose / ONNX-backed nodes run on GPU.
- `xFormers not available` — we don't install xformers. ComfyUI falls back to its built-in SDPA path plus SageAttention, both of which work.
- `Nvidia APEX normalization not installed, using PyTorch LayerNorm` — APEX isn't installed. PyTorch's LayerNorm is the fallback path, and gens succeed without APEX.
- `[DEPRECATION WARNING]` from various custom nodes — from third-party custom node code using older ComfyUI APIs. Comes from the node, not from this repo.

If you see a warning that ISN'T in this list and you're worried, run `bash verify.sh` — if it passes cleanly and your gens succeed, it's almost certainly cosmetic too.

---

## Acknowledgments

This repo wouldn't exist without:

- **[ecarmen16/SparkyUI](https://github.com/ecarmen16/SparkyUI/)** — first reference for SageAttention sm_121 build pattern on DGX Spark
- **[Jay0515 on Hugging Face](https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121)** — the community ONNX Runtime wheel I'm using (the only sm_121/aarch64/cu13 build I'm aware of)
- **[Comfy-Org/comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo)** — the DynamicVRAM allocator
- **[thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)** — the attention kernels themselves
- **[natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup)** — broader DGX Spark ML setup notes
- **[NVIDIA Developer Forums DGX Spark community](https://forums.developer.nvidia.com/c/data-center-cards/dgx-spark/)** — the slow accumulation of working configurations

Upstream issues coordinated with:

- [Lightricks/ComfyUI-LTXVideo#430](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/430) — LTX 2.3 audio VAE NaN output (Patch 2 workaround)
- [thu-ml/SageAttention#321](https://github.com/thu-ml/SageAttention/issues/321) — Sage 3 mosaic artifacts on Spark

---

## License

MIT — see [LICENSE](LICENSE).

Maintained by [Triplany](https://github.com/Triplany).
