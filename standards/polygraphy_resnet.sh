#!/usr/bin/env bash
# NVIDIA Polygraphy — the pip-installable `trtexec` equivalent: builds a TensorRT
# engine from an ONNX model and profiles inference (latency -> throughput).
#   With the full TensorRT SDK you'd instead run:
#     trtexec --onnx=resnet50_fp16_dyn.onnx --shapes=x:128x3x224x224 --fp16
set -uo pipefail
source /root/mlperf/venv/bin/activate
pip show polygraphy >/dev/null 2>&1 || pip install -q polygraphy colored

# put the pip CUDA runtime libs on the loader path (polygraphy's runner needs libcudart)
NVBASE=$(python -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d "$NVBASE"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
ln -sf "$NVBASE/cuda_runtime/lib/libcudart.so.12" "$NVBASE/cuda_runtime/lib/libcudart.so" 2>/dev/null

ONNX=${1:-/root/mlperf/vision/resnet50_fp16_dyn.onnx}   # fp16 dynamic-batch ResNet-50
BS=${BS:-128}
echo "polygraphy: $ONNX  batch=$BS"
polygraphy run "$ONNX" --trt --input-shapes x:[$BS,3,224,224] --warm-up 25 --iterations 200
echo "throughput = BS / (Average inference time).  e.g. 128 / 0.03228s = ~3965 img/s"
