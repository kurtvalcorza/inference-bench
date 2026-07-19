#!/usr/bin/env bash
# ResNet-50 with an optimized TensorRT fp16 SUT under MLCommons LoadGen.
# Runs SingleStream (latency), Offline (throughput), and an accuracy pass, and FAILS LOUDLY
# (non-zero exit) if any scenario is not LoadGen-VALID or accuracy falls below ACC_MIN.
# NOTE: MLPerf-inspired only — short config + subset data, NOT a conformant MLPerf result.
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

# --- harness (pinned for reproducibility) -----------------------------------
INFERENCE_REF="${INFERENCE_REF:-da738a5}"   # commit the notebooks/results were produced against
if [ ! -d "$H" ]; then
  echo "===== cloning mlcommons/inference @ $INFERENCE_REF into $INFERENCE_REPO ====="
  git clone --filter=blob:none --no-checkout https://github.com/mlcommons/inference.git "$INFERENCE_REPO"
  git -C "$INFERENCE_REPO" checkout -q "$INFERENCE_REF" || { echo "!! could not checkout $INFERENCE_REF"; exit 1; }
fi
echo "harness commit: $(git -C "$INFERENCE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) (pin: $INFERENCE_REF)"

cat > "$CONF" <<'CONF'
resnet50.SingleStream.min_duration = 10000
resnet50.SingleStream.min_query_count = 4000
resnet50.Offline.target_qps = 12000
resnet50.Offline.min_duration = 10000
resnet50.Offline.min_query_count = 1
CONF

echo "===== install TensorRT backend into the harness ====="
cp "$SCRIPT_DIR/backend_tensorrt.py" "$H/python/backend_tensorrt.py"
if ! python - "$H/python/main.py" <<'PY'
import sys
f = sys.argv[1]; s = open(f).read()
if "backend_tensorrt" in s:
    print("main.py already has tensorrt backend"); sys.exit(0)
a = '    elif backend == "pytorch-native":'
if a not in s:                      # upstream is a fresh --depth 1 clone => a moving target
    sys.exit(f"ERROR: anchor {a!r} not found in main.py — upstream layout changed; "
             f"update the patcher in trt_mlperf_run.sh (do not run with an unpatched harness).")
ins = '    elif backend == "tensorrt":\n        from backend_tensorrt import BackendTensorRT\n\n        backend = BackendTensorRT()\n'
s = s.replace(a, ins + a, 1)
assert "backend_tensorrt" in s, "replace() did not insert the tensorrt branch"   # never silently 'succeed'
open(f, "w").write(s); print("main.py patched: +tensorrt backend")
PY
then echo "!! main.py patch failed — aborting"; exit 1; fi

echo
echo "===== export fp16 dynamic-batch ONNX ====="
if [ ! -s "$ONNX" ]; then
  python -c "import torch, torchvision" 2>/dev/null || {
    echo "!! torch+torchvision required to export $ONNX (setup.md §2: install torch cu128), or pass a prebuilt ONNX via ONNX=..."; exit 1; }
  python "$SCRIPT_DIR/export_resnet50_onnx.py" "$ONNX"
fi
ls -lh "$ONNX"

echo
echo "===== ensure accuracy/data subset ====="
if [ ! -s "$VMAP" ]; then
  echo "no val_map at $VMAP — building representative ImageNet subset (needs HF mirror)"
  # need BOTH huggingface_hub (<1.0, or it breaks transformers 4.48) AND pyarrow — check both, not just presence
  python - <<'PY' || pip install -q "huggingface_hub>=0.24,<1.0" pyarrow
import sys
try:
    import pyarrow  # noqa: F401
    import huggingface_hub as h
    sys.exit(0 if int(h.__version__.split(".")[0]) < 1 else 1)
except Exception:
    sys.exit(1)
PY
  BENCH_ROOT="$BENCH_ROOT" python "$SCRIPT_DIR/build_imagenet_subset.py" "$DATA" || {
    echo "!! subset build failed. Point DATA=... at a val set with val_map.txt (e.g. Imagenette) and re-run."; exit 1; }
fi
echo "DATA=$DATA  ($(wc -l < "$VMAP") images)"

cd "$H/python"
# Fresh, unique output dir per invocation so a failed run can NEVER reprint a prior run's summary.
STAMP=$(date +%Y%m%d-%H%M%S)
RUNROOT="$BENCH_ROOT/vision/runs/$STAMP"
mkdir -p "$RUNROOT"
ACC_MIN=${ACC_MIN:-70}          # top-1 floor (subset ResNet-50 v1: ~75% mirror / ~84% imagenette)
FAILED=0
echo "run dir: $RUNROOT"

run_main () {  # scenario, extra-args, outdir  -> returns main.py's exit code
  rm -rf "$3"; mkdir -p "$3"
  python main.py --profile resnet50-pytorch --backend tensorrt --model "$ONNX" \
    --dataset-path "$DATA" --user_conf "$CONF" --max-batchsize "$MAXBS" \
    --scenario "$1" $2 --output "$3" >"$3/run.log" 2>&1
}

check_valid () {  # scenario, outdir, rc  -> sets FAILED on any problem
  local scen="$1" out="$2" rc="$3" sum="$2/mlperf_log_summary.txt"
  if [ "$rc" -ne 0 ];              then echo "!! $scen: main.py exited $rc";  tail -15 "$out/run.log"; FAILED=1; return; fi
  if [ ! -s "$sum" ];              then echo "!! $scen: no summary written"; tail -15 "$out/run.log"; FAILED=1; return; fi
  if ! grep -q "Result is : VALID" "$sum"; then
    echo "!! $scen: LoadGen result is NOT VALID"; grep -i "result is" "$sum" || true; FAILED=1; return; fi
  echo "-- $scen: VALID --"; sed -n '1,22p' "$sum"
}

echo
echo "############ SingleStream (p50/p90/p99 latency) ############"
run_main SingleStream "" "$RUNROOT/SingleStream"; check_valid SingleStream "$RUNROOT/SingleStream" $?

echo
echo "############ Offline (throughput) ############"
run_main Offline "" "$RUNROOT/Offline"; check_valid Offline "$RUNROOT/Offline" $?

echo
echo "############ Accuracy (top-1, fp16 TRT) ############"
# main.py's post-test percentile stats crash AFTER LoadGen writes the accuracy log (known numpy
# issue) => gate on the freshly-written artifact, not the exit code.
ACC="$RUNROOT/Accuracy"
run_main Offline "--accuracy" "$ACC" || true
AJSON="$ACC/mlperf_log_accuracy.json"
if [ ! -s "$AJSON" ]; then
  echo "!! accuracy: no mlperf_log_accuracy.json produced"; tail -20 "$ACC/run.log"; FAILED=1
else
  ACC_OUT=$(python ../tools/accuracy-imagenet.py --mlperf-accuracy-file "$AJSON" \
    --imagenet-val-file "$VMAP" --dtype float32 2>&1 | tail -1)
  echo "$ACC_OUT"
  TOP1=$(echo "$ACC_OUT" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  if   [ -z "$TOP1" ];                             then echo "!! accuracy: could not parse top-1 from scorer"; FAILED=1
  elif awk "BEGIN{exit !($TOP1 < $ACC_MIN)}";      then echo "!! accuracy ${TOP1}% < floor ${ACC_MIN}% — suspect run"; FAILED=1
  else echo "accuracy OK: ${TOP1}% (>= ${ACC_MIN}%)"; fi
fi

echo
if [ "$FAILED" -ne 0 ]; then echo "TRT MLPERF: FAILED — see errors above; run dir $RUNROOT"; exit 1; fi
echo "TRT MLPERF: all checks passed (LoadGen-VALID under short config); run dir $RUNROOT"
