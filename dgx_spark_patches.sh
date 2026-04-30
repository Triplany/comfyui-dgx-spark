#!/usr/bin/env bash
# DGX Spark ComfyUI patches — idempotent. Re-run after `git pull`.
#
# Patch 1: unified-memory free-mem reporting  ── MOSTLY OBSOLETE AS OF 2026-04
#   comfy/model_management.py:~1520 — swap cuda.mem_get_info for psutil.available.
#   cuda.mem_get_info() ignores reclaimable OS page cache, so on Spark it
#   under-reports free memory by tens of GB. ComfyUI's load/unload heuristics
#   read that figure to decide when to evict models — when it's artificially
#   low, ComfyUI panic-evicts models that didn't need to go, then reloads them.
#
#   Upstream fix: the cluster of PRs around async-offload addresses the
#   underlying load/unload behavior this patch was working around:
#     9d8a8179  PR #10953  2025-11-27  Enable async offloading by default on Nvidia
#     519c9411  PR #11069  2025-12-03  Reduce massive LoRA reservations (esp Flux2)
#     e136b6db  PR #11171  2025-12-08  Dequantization offload accounting (Flux2 OOMs)
#     8d723d2c  PR #13221  2026-03-29  Fix/tweak pinned memory accounting
#
#   Verified empirically (2026-04-30): on master commit 38ecad8f8a (post-fix),
#   Flux1 → Flux2 model switch on Spark unified memory evicts cleanly to ~8 GB
#   before the second model loads. No double-spike. The behavior this patch
#   chased no longer reproduces.
#
#   This script now defaults to SKIP Patch 1 when ComfyUI HEAD is at or past
#   commit 9d8a8179. Set FORCE_LEGACY_PATCH1=1 to apply anyway (e.g. you still
#   observe panic-eviction during long single-model runs and want to test).
#
# Patch 2: LTX 2.3 audio VAE NaN/Inf → AAC encode crash
#   comfy_api/latest/_input_impl/video_types.py:~452 — clamp NaN/Inf in the audio
#   waveform before AAC encoding. LTX 2.3's audio VAE sometimes outputs NaN/Inf
#   values which the AAC encoder rejects with EINVAL.
#   See: https://github.com/Lightricks/ComfyUI-LTXVideo/issues/430
#
# Patch 3: LTX 2.3 audio VAE eviction churn under DynamicVRAM  ── OBSOLETE AS OF 2026-04
#   comfy/ldm/lightricks/vae/audio_vae.py — originally skipped the free_memory()
#   call in ensure_model_loaded(). On Spark with DynamicVRAM enabled, that
#   eviction triggered an evict-then-restage cycle on the staged VideoVAE.
#
#   Upstream fix: PR #13486 / commit ad94d472 (2026-04-21) "Make the ltx audio
#   vae more native" rewrote the audio VAE handling and removed
#   ensure_model_loaded() entirely. Patch 3 has nothing to attach to on a
#   current ComfyUI tree.
#
#   The Patch 3 block below already detects this gracefully — when
#   ensure_model_loaded() is gone, it prints a skip message and exits 0.
#   No action needed; documented here for transparency.
#
# Run: ./dgx_spark_patches.sh

set -euo pipefail
cd "$(dirname "$0")"

# ---- Patch 1: psutil free-mem (ancestor-gated) ----
FILE="comfy/model_management.py"
PATCH1_FIX_COMMIT="9d8a8179"   # PR #10953 — Enable async offloading by default on Nvidia
PATCH1_NEEDED=1                # default: apply (legacy ComfyUI tree, no fix yet)

if git rev-parse --verify HEAD >/dev/null 2>&1; then
    if git merge-base --is-ancestor "$PATCH1_FIX_COMMIT" HEAD 2>/dev/null; then
        if [ "${FORCE_LEGACY_PATCH1:-0}" = "1" ]; then
            echo "[dgx-patches] Patch 1: FORCE_LEGACY_PATCH1=1 — applying despite upstream fix"
        else
            echo "[dgx-patches] Patch 1: SKIP — upstream PR #10953 (commit $PATCH1_FIX_COMMIT, 2025-11-27) addresses this"
            echo "[dgx-patches]            Set FORCE_LEGACY_PATCH1=1 to apply anyway."
            PATCH1_NEEDED=0
        fi
    fi
fi

if [ "$PATCH1_NEEDED" = "1" ]; then
    OLD='            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)'
    NEW='            # DGX Spark unified-memory patch: cuda.mem_get_info under-reports free
            # memory on GB10 because it ignores reclaimable OS page cache. Use
            # psutil.available which reflects what the kernel can actually hand out.
            # Revert this if this install is ever moved to a discrete-GPU box.
            import psutil as _psutil; mem_free_cuda = _psutil.virtual_memory().available'

    if grep -q "DGX Spark unified-memory patch" "$FILE"; then
        echo "[dgx-patches] model_management.py: already patched (legacy)"
    elif grep -qF "$OLD" "$FILE"; then
        python3 - <<PY
from pathlib import Path
p = Path("$FILE")
t = p.read_text()
old = """$OLD"""
new = """$NEW"""
assert t.count(old) == 1, f"expected exactly one match, found {t.count(old)}"
p.write_text(t.replace(old, new, 1))
PY
        echo "[dgx-patches] model_management.py: patched (legacy)"
    else
        echo "[dgx-patches] WARN: model_management.py: neither original nor patched line found — upstream may have changed the code. Inspect manually." >&2
        exit 1
    fi
fi

# ---- Patch 2: NaN/Inf clamp in audio AAC encode (video_types.py) ----
FILE2="comfy_api/latest/_input_impl/video_types.py"
OLD2='                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='\''fltp'\'', layout=layout)'
NEW2='                # DGX Spark NaN audio patch: LTX 2.3 audio VAE sometimes outputs NaN/Inf
                # values that AAC encoder rejects with EINVAL. Clamp to 0 before encoding.
                # See https://github.com/Lightricks/ComfyUI-LTXVideo/issues/430
                import numpy as _np_dgx
                _wv_dgx = waveform.float().cpu().contiguous().numpy()
                _wv_dgx = _np_dgx.nan_to_num(_wv_dgx, nan=0.0, posinf=0.0, neginf=0.0)
                frame = av.AudioFrame.from_ndarray(_wv_dgx, format='\''fltp'\'', layout=layout)'

if grep -q "DGX Spark NaN audio patch" "$FILE2"; then
    echo "[dgx-patches] video_types.py: already patched"
elif grep -qF "$OLD2" "$FILE2"; then
    python3 - <<PY
from pathlib import Path
p = Path("$FILE2")
t = p.read_text()
old = """$OLD2"""
new = """$NEW2"""
assert t.count(old) == 1, f"expected exactly one match, found {t.count(old)}"
p.write_text(t.replace(old, new, 1))
PY
    echo "[dgx-patches] video_types.py: patched"
else
    echo "[dgx-patches] WARN: video_types.py: neither original nor patched line found — upstream may have changed the code. Inspect manually." >&2
    exit 1
fi

# ---- Patch 3: skip free_memory in audio VAE ensure_model_loaded ----
# Multi-line patch — done in pure Python to avoid grep multi-line quirks
# (grep -F treats embedded newlines as separator, not part of pattern).
FILE3="comfy/ldm/lightricks/vae/audio_vae.py"
python3 - <<'PY' "$FILE3"
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("[dgx-patches] audio_vae.py: not present — skipping (no LTX audio VAE in this ComfyUI version)")
    sys.exit(0)

t = p.read_text()

if "DGX Spark patch: skip free_memory call" in t:
    print("[dgx-patches] audio_vae.py: already patched")
    sys.exit(0)

old = """    def ensure_model_loaded(self) -> None:
        comfy.model_management.free_memory(
            self.patcher.model_size(),
            self.patcher.load_device,
        )
        comfy.model_management.load_model_gpu(self.patcher)"""

new = """    def ensure_model_loaded(self) -> None:
        # DGX Spark patch: skip free_memory call. On Spark unified memory its
        # eviction of staged DynamicVRAM models (VideoVAE) corrupts their state
        # when re-staged, causing black video frames in LTX 2.3 workflows.
        # Audio VAE is ~700 MB; on a 128 GB unified pool no eviction is needed.
        # comfy.model_management.free_memory(self.patcher.model_size(), self.patcher.load_device)
        comfy.model_management.load_model_gpu(self.patcher)"""

n = t.count(old)
if n == 1:
    p.write_text(t.replace(old, new, 1))
    print("[dgx-patches] audio_vae.py: patched")
    sys.exit(0)

if "def ensure_model_loaded" not in t:
    # Upstream refactored AudioVAE (e.g. v0.20.0+) — the ModelDeviceManager class
    # with ensure_model_loaded() is gone. The eviction issue this patch worked
    # around no longer exists. Skip cleanly.
    print("[dgx-patches] audio_vae.py: ensure_model_loaded() removed upstream — patch no longer needed, skipping")
    sys.exit(0)

print("[dgx-patches] WARN: audio_vae.py: ensure_model_loaded() exists but doesn't match expected form — upstream may have changed the code in a way we don't recognize. Inspect manually.", file=sys.stderr)
sys.exit(1)
PY

echo "[dgx-patches] done"
