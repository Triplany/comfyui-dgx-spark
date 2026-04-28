#!/usr/bin/env bash
# comfyui-spark — post-install verification
#
# Health check across all pieces. Doesn't modify anything.

set -uo pipefail  # not -e; we want to see ALL failures

if [ -z "${COMFY:-}" ]; then
    COMFY="$HOME/ComfyUI"
    echo "[verify] COMFY env var not set — using default: $COMFY"
fi
if [ -z "${VENV:-}" ]; then
    VENV="$COMFY/.venv"
    echo "[verify] VENV env var not set — using default: $VENV"
fi

PASS=0
FAIL=0
ok()    { echo "  ✅ $*"; PASS=$((PASS+1)); }
fail()  { echo "  ❌ $*"; FAIL=$((FAIL+1)); }
warn()  { echo "  ⚠️  $*"; }

echo "comfyui-spark verify"
echo ""

# 1. PyTorch + CUDA
echo "PyTorch / CUDA / GPU"
"$VENV/bin/python" - <<'PY' 2>&1 | grep -v FutureWarning | grep -v pynvml
import torch, sys
v = torch.__version__
cuda = torch.version.cuda
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(f"  PyTorch:  {v}")
print(f"  CUDA:     {cuda}")
print(f"  GPU:      {name}  capability={cap}")
sys.exit(0 if cap == (12, 1) else 2)
PY
[ $? -eq 0 ] && ok "GPU is GB10 (sm_121)" || fail "GPU not sm_121 — script is DGX Spark specific"
echo ""

# 2. SageAttention + native kernels
echo "SageAttention"
SAGE_DIR=$("$VENV/bin/python" -c "import sageattention, os; print(os.path.dirname(sageattention.__file__))" 2>/dev/null | tail -1)
if [ -d "$SAGE_DIR" ]; then
    ok "package present at $SAGE_DIR"
    HAS_SM121=true
    for f in "$SAGE_DIR"/_qattn_sm80*.so "$SAGE_DIR"/_qattn_sm89*.so; do
        [ -f "$f" ] || continue
        # Capture full cuobjdump output first (NOT piped to grep -q under pipefail —
        # grep -q early-exits on first match, causing SIGPIPE on cuobjdump,
        # which under pipefail makes the whole pipe report failure).
        out=$(/usr/local/cuda/bin/cuobjdump --list-text "$f" 2>/dev/null || true)
        if echo "$out" | grep -q "sm_121"; then
            ok "$(basename "$f") has sm_121 kernels"
        else
            fail "$(basename "$f") MISSING sm_121 — run build/sage.sh"
            HAS_SM121=false
        fi
    done
else
    fail "sageattention not installed"
fi
echo ""

# 3. ONNX Runtime
echo "ONNX Runtime"
PROVIDERS=$("$VENV/bin/python" -c "import onnxruntime; print(','.join(onnxruntime.get_available_providers()))" 2>&1 | grep -v FutureWarning | grep -v pynvml | tail -1)
if echo "$PROVIDERS" | grep -q CUDAExecutionProvider; then
    ok "CUDAExecutionProvider available"
else
    fail "no CUDAExecutionProvider — run build/onnxruntime.sh (got: $PROVIDERS)"
fi
echo ""

# 4. comfy-aimdo
echo "comfy-aimdo (DynamicVRAM)"
AIMDO_SO="$VENV/lib/python3.12/site-packages/comfy_aimdo/aimdo.so"
if [ -f "$AIMDO_SO" ]; then
    if file "$AIMDO_SO" | grep -q "ARM aarch64"; then
        ok "aimdo.so is aarch64 ELF"
        # Test init
        AIMDO_INIT=$("$VENV/bin/python" -c "
import torch; torch.cuda.init(); _=torch.zeros(1, device='cuda')
import comfy_aimdo.control as c, ctypes
c.init()
c.lib.init.argtypes=[ctypes.c_int]; c.lib.init.restype=ctypes.c_bool
print('OK' if c.lib.init(0) else 'FAIL')
" 2>&1 | grep -v FutureWarning | grep -v pynvml | grep -v aimdo: | tail -1)
        if [ "$AIMDO_INIT" = "OK" ]; then
            ok "aimdo init(0) returns True"
        else
            fail "aimdo init(0) failed: $AIMDO_INIT"
        fi
    else
        fail "aimdo.so is NOT aarch64 — run build/aimdo.sh"
    fi
else
    fail "aimdo.so missing — run build/aimdo.sh"
fi
echo ""

# 5. opencv
echo "opencv"
if "$VENV/bin/python" -c "from cv2.ximgproc import guidedFilter" 2>/dev/null; then
    ok "cv2.ximgproc.guidedFilter importable (LayerStyle nodes will work)"
else
    fail "guidedFilter import broken — run build/opencv.sh"
fi
echo ""

# 6. imageio-ffmpeg
echo "imageio-ffmpeg"
if "$VENV/bin/python" -c "import imageio_ffmpeg" 2>/dev/null; then
    ok "imageio_ffmpeg importable"
else
    fail "imageio_ffmpeg missing — run build/imageio-ffmpeg.sh"
fi
echo ""

# 7. ComfyUI patches
echo "ComfyUI source patches"
if grep -q "DGX Spark unified-memory patch" "$COMFY/comfy/model_management.py" 2>/dev/null; then
    ok "Patch 1 (psutil free-mem) applied"
else
    fail "Patch 1 NOT applied — run dgx_spark_patches.sh"
fi
if grep -q "DGX Spark NaN audio patch" "$COMFY/comfy_api/latest/_input_impl/video_types.py" 2>/dev/null; then
    ok "Patch 2 (NaN/Inf audio clamp) applied"
else
    fail "Patch 2 NOT applied — run dgx_spark_patches.sh"
fi
# Patch 3 is version-aware: only needed when ensure_model_loaded() exists in
# the audio VAE (older ComfyUI). On v0.20.0+ the upstream refactor removed
# that function, so the patch isn't applicable AND isn't needed.
AV_FILE="$COMFY/comfy/ldm/lightricks/vae/audio_vae.py"
if [ ! -f "$AV_FILE" ]; then
    ok "Patch 3 N/A — audio_vae.py not present (no LTX audio VAE in this ComfyUI version)"
elif grep -q "DGX Spark patch: skip free_memory call" "$AV_FILE"; then
    ok "Patch 3 (skip free_memory in audio VAE) applied"
elif ! grep -q "def ensure_model_loaded" "$AV_FILE"; then
    ok "Patch 3 N/A — ensure_model_loaded() removed upstream, patch not needed for this ComfyUI version"
else
    fail "Patch 3 NOT applied — run dgx_spark_patches.sh"
fi
echo ""

# 8. Launcher
echo "Launcher"
if [ ! -x "$COMFY/run_dgx_spark.sh" ]; then
    fail "launcher missing — run install.sh"
else
    ok "$COMFY/run_dgx_spark.sh present and executable"
    # The launcher should NOT have --fp16-unet/vae/text-enc — in our testing,
    # forcing fp16 globally produced all-black LTX 2.3 video.
    if grep -qE "^\s*--fp16-(unet|vae|text-enc)\b" "$COMFY/run_dgx_spark.sh"; then
        warn "launcher has --fp16-* precision flag — this produced all-black LTX 2.3 video in our testing. README's 'Why no precision flags' section explains; drop it."
    else
        ok "launcher has no global --fp16-* / --bf16-* precision flags (auto-dtype path, correct)"
    fi
    for flag in "--reserve-vram 8" "--disable-pinned-memory" "--dont-upcast-attention"; do
        if grep -qF -- "$flag" "$COMFY/run_dgx_spark.sh"; then
            ok "launcher has $flag"
        else
            warn "launcher missing $flag — check it matches the template"
        fi
    done
fi
echo ""

# Summary
echo "----------------------------------------"
echo "  passed: $PASS    failed: $FAIL"
echo "----------------------------------------"
[ $FAIL -eq 0 ] && exit 0 || exit 1
