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

# Pinned to the validated stack in ../requirements.txt (finding #8: don't install "latest").
pip show polygraphy >/dev/null 2>&1 || pip install -q "polygraphy==0.50.3" colored
python -c "import tensorrt, onnx" 2>/dev/null || pip install -q "tensorrt==11.1.0.106" "onnx==1.22.0"   # need BOTH

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
if ! polygraphy run "$ONNX" --trt --input-shapes x:[$BS,3,224,224] --warm-up 25 --iterations 200; then
  echo "!! polygraphy run FAILED — no valid throughput produced"; exit 1
fi
echo "throughput = BS / (Average inference time).  e.g. 128 / 0.03228s = ~3965 img/s"
