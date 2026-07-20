#!/usr/bin/env bash
# NVIDIA Polygraphy — the pip-installable `trtexec` equivalent: builds a TensorRT
# engine from an ONNX model and profiles inference (latency -> throughput).
#   With the full TensorRT SDK you'd instead run:
#     trtexec --onnx=resnet50_fp16_dyn.onnx --shapes=x:128x3x224x224 --fp16
#
# Env: BENCH_VENV (venv to activate), BENCH_ROOT (asset root), BS (batch size).
# Arg $1 overrides the ONNX path. Builds the ONNX via ../tensorrt if absent.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # inference-bench/standards
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="${BENCH_VENV:-/root/mlperf/venv}"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"
else echo "[info] no venv at $VENV — using the current python environment"; fi

if [ -z "${BENCH_ROOT:-}" ]; then
  if [ -d /root/mlperf ]; then BENCH_ROOT=/root/mlperf; else BENCH_ROOT="$HOME/inference-bench-data"; fi
fi
mkdir -p "$BENCH_ROOT/vision"

# ENFORCE the validated pins (finding #6): an already-installed but DIFFERENT version must not pass a
# mere `pip show` / import check — compare the installed version and reinstall on mismatch.
require_version () {  # dist  expected  pip_args...
  local dist="$1" want="$2"; shift 2
  local have; have=$(python -c "from importlib.metadata import version; print(version('$dist'))" 2>/dev/null || true)
  [ "$have" = "$want" ] || { echo "[deps] $dist ${have:-missing} != $want -> pip install $*";
    pip install -q "$@" || { echo "!! failed installing $*"; exit 1; }; }
}
require_version polygraphy 0.50.3     "polygraphy==0.50.3" colored
require_version tensorrt   11.1.0.106 "tensorrt==11.1.0.106"
require_version onnx       1.22.0     "onnx==1.22.0"

# put the pip CUDA runtime libs on the loader path (polygraphy's runner needs libcudart)
NVBASE=$(python -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVBASE" ]; then
  export LD_LIBRARY_PATH="$(ls -d "$NVBASE"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
  ln -sf "$NVBASE/cuda_runtime/lib/libcudart.so.12" "$NVBASE/cuda_runtime/lib/libcudart.so" 2>/dev/null || true
fi

ONNX=${1:-$BENCH_ROOT/vision/resnet50_fp16_dyn.onnx}   # fp16 dynamic-batch ResNet-50
BS=${BS:-128}
if [ ! -s "$ONNX" ]; then
  python -c "import torch, torchvision" 2>/dev/null || {
    echo "!! torch+torchvision required to build the ONNX (setup.md §2), or pass a prebuilt ONNX as arg 1"; exit 1; }
  python "$REPO_ROOT/tensorrt/export_resnet50_onnx.py" "$ONNX"
fi

echo "polygraphy: $ONNX  batch=$BS"
POLYLOG=$(mktemp)
if ! polygraphy run "$ONNX" --trt --input-shapes x:[$BS,3,224,224] --warm-up 25 --iterations 200 2>&1 | tee "$POLYLOG"; then
  echo "!! polygraphy run FAILED — no valid throughput produced"; rm -f "$POLYLOG"; exit 1
fi
# Compute throughput from the ACTUAL measured average latency (finding #8) — never a hard-coded example.
AVG_MS=$(grep -oiE "Average inference time: [0-9.]+ ms" "$POLYLOG" | grep -oE "[0-9.]+" | tail -1)
rm -f "$POLYLOG"
if [ -n "$AVG_MS" ]; then
  awk -v bs="$BS" -v ms="$AVG_MS" 'BEGIN{printf "measured throughput = batch %d / %.3f ms = %.0f img/s\n", bs, ms, bs/(ms/1000.0)}'
else
  echo "!! could not parse the average inference time from polygraphy output — no valid throughput produced"
  exit 1
fi
