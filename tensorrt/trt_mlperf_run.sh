#!/usr/bin/env bash
# MLPerf ResNet-50 with an optimized TensorRT fp16 SUT (LoadGen -> VALID results).
# Runs SingleStream (latency), Offline (throughput), and an accuracy pass.
#
# Portable: reads its backend from this repo, honours env overrides, and
# bootstraps the harness / ONNX / accuracy set if missing.
#   BENCH_VENV   python venv to activate      (default /root/mlperf/venv; skipped if absent)
#   BENCH_ROOT   asset/data root              (default /root/mlperf if it exists, else ~/inference-bench-data)
#   INFERENCE_REPO  mlcommons/inference clone (default $BENCH_ROOT/inference; cloned if missing)
#   DATA         ImageNet val subset dir      (default $BENCH_ROOT/vision/inet_val; built if missing)
#   ONNX         fp16 dynamic ONNX path       (default $BENCH_ROOT/vision/resnet50_fp16_dyn.onnx; built if missing)
#   MAXBS        max batch size               (default 128)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # inference-bench/tensorrt

# --- environment ------------------------------------------------------------
VENV="${BENCH_VENV:-/root/mlperf/venv}"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"
else echo "[info] no venv at $VENV — using the current python environment"; fi

if [ -z "${BENCH_ROOT:-}" ]; then
  if [ -d /root/mlperf ]; then BENCH_ROOT=/root/mlperf; else BENCH_ROOT="$HOME/inference-bench-data"; fi
fi
mkdir -p "$BENCH_ROOT/vision"

INFERENCE_REPO="${INFERENCE_REPO:-$BENCH_ROOT/inference}"
H="$INFERENCE_REPO/vision/classification_and_detection"
DATA="${DATA:-$BENCH_ROOT/vision/inet_val}"
VMAP="$DATA/val_map.txt"
ONNX="${ONNX:-$BENCH_ROOT/vision/resnet50_fp16_dyn.onnx}"
MAXBS=${MAXBS:-128}
export TRT_MAX_BATCHSIZE=$MAXBS
CONF="$BENCH_ROOT/vision/trt.conf"

echo "BENCH_ROOT=$BENCH_ROOT  INFERENCE_REPO=$INFERENCE_REPO  MAXBS=$MAXBS"

# --- dependencies -----------------------------------------------------------
python -c "import mlperf_loadgen" 2>/dev/null || pip install -q mlcommons-loadgen
python -c "import tensorrt, onnx"  2>/dev/null || pip install -q tensorrt onnx
python -c "import cv2"             2>/dev/null || pip install -q opencv-python-headless
python -c "import pycocotools"     2>/dev/null || pip install -q pycocotools   # main.py imports coco unconditionally

# --- harness ----------------------------------------------------------------
if [ ! -d "$H" ]; then
  echo "===== cloning mlcommons/inference into $INFERENCE_REPO ====="
  git clone --depth 1 https://github.com/mlcommons/inference.git "$INFERENCE_REPO"
fi

cat > "$CONF" <<'CONF'
resnet50.SingleStream.min_duration = 10000
resnet50.SingleStream.min_query_count = 4000
resnet50.Offline.target_qps = 12000
resnet50.Offline.min_duration = 10000
resnet50.Offline.min_query_count = 1
CONF

echo "===== install TensorRT backend into the harness ====="
cp "$SCRIPT_DIR/backend_tensorrt.py" "$H/python/backend_tensorrt.py"
python - "$H/python/main.py" <<'PY'
import sys
f = sys.argv[1]; s = open(f).read()
if "backend_tensorrt" not in s:
    a = '    elif backend == "pytorch-native":'
    ins = '    elif backend == "tensorrt":\n        from backend_tensorrt import BackendTensorRT\n\n        backend = BackendTensorRT()\n'
    s = s.replace(a, ins + a, 1); open(f, "w").write(s); print("main.py patched: +tensorrt backend")
else:
    print("main.py already has tensorrt backend")
PY

echo
echo "===== export fp16 dynamic-batch ONNX ====="
[ -s "$ONNX" ] || python "$SCRIPT_DIR/export_resnet50_onnx.py" "$ONNX"
ls -lh "$ONNX"

echo
echo "===== ensure accuracy/data subset ====="
if [ ! -s "$VMAP" ]; then
  echo "no val_map at $VMAP — building representative ImageNet subset (needs HF mirror)"
  pip show huggingface_hub >/dev/null 2>&1 || pip install -q "huggingface_hub>=0.24,<1.0" pyarrow
  BENCH_ROOT="$BENCH_ROOT" python "$SCRIPT_DIR/build_imagenet_subset.py" "$DATA" || {
    echo "!! subset build failed. Point DATA=... at a val set with val_map.txt (e.g. Imagenette) and re-run."; exit 1; }
fi
echo "DATA=$DATA  ($(wc -l < "$VMAP") images)"

cd "$H/python"
run () {  # scenario, extra-args, outdir
  python main.py --profile resnet50-pytorch --backend tensorrt --model "$ONNX" \
    --dataset-path "$DATA" --user_conf "$CONF" --max-batchsize $MAXBS \
    --scenario "$1" $2 --output "$3" >/tmp/trt_$1.log 2>&1 || true
}

echo
echo "############ SingleStream (p50/p90/p99 latency) ############"
run SingleStream "" "$BENCH_ROOT/vision/trt_ss"
sed -n '1,22p' "$BENCH_ROOT/vision/trt_ss/mlperf_log_summary.txt" 2>/dev/null || { echo "(no summary)"; tail -15 /tmp/trt_SingleStream.log; }

echo
echo "############ Offline (throughput) ############"
run Offline "" "$BENCH_ROOT/vision/trt_off"
sed -n '1,14p' "$BENCH_ROOT/vision/trt_off/mlperf_log_summary.txt" 2>/dev/null || echo "(no summary)"

echo
echo "############ Accuracy (top-1, fp16 TRT) ############"
run Offline "--accuracy" "$BENCH_ROOT/vision/trt_acc"
python ../tools/accuracy-imagenet.py --mlperf-accuracy-file "$BENCH_ROOT/vision/trt_acc/mlperf_log_accuracy.json" \
  --imagenet-val-file "$VMAP" --dtype float32 2>&1 | tail -1
echo "DONE trt mlperf"
