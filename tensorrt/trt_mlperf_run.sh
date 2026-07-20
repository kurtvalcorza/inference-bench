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

# --- dependencies (ENFORCE the validated pins in ../requirements.txt, not just "importable") -------
# An already-installed but DIFFERENT version must not silently pass (finding #6): compare the actual
# installed distribution version to the pin and reinstall on mismatch.
require_version () {  # dist_name  expected_version  pip_spec
  local dist="$1" want="$2" spec="$3" have
  have=$(python -c "from importlib.metadata import version; print(version('$dist'))" 2>/dev/null || true)
  if [ "$have" != "$want" ]; then
    echo "[deps] $dist ${have:-missing} != $want -> pip install $spec"
    pip install -q "$spec" || { echo "!! failed installing $spec"; exit 1; }
  fi
}
require_version mlcommons-loadgen      6.0.16      "mlcommons-loadgen==6.0.16"
require_version tensorrt               11.1.0.106  "tensorrt==11.1.0.106"
require_version onnx                   1.22.0      "onnx==1.22.0"
require_version opencv-python-headless 5.0.0.93    "opencv-python-headless==5.0.0.93"   # cv2
require_version pycocotools            2.0.11      "pycocotools==2.0.11"   # main.py imports coco unconditionally

# --- harness (pin ENFORCED every run, not just on fresh clone) --------------
INFERENCE_REF="${INFERENCE_REF:-da738a5}"   # commit the notebooks/results were produced against
if [ ! -d "$H" ]; then
  echo "===== cloning mlcommons/inference into $INFERENCE_REPO ====="
  git clone --filter=blob:none --no-checkout https://github.com/mlcommons/inference.git "$INFERENCE_REPO"
fi
# Resolve the pin (fetch it if a cached clone lacks it); fail on an invalid ref.
want=$(git -C "$INFERENCE_REPO" rev-parse -q --verify "$INFERENCE_REF^{commit}" 2>/dev/null || true)
if [ -z "$want" ]; then
  git -C "$INFERENCE_REPO" fetch --filter=blob:none -q origin "$INFERENCE_REF" 2>/dev/null || true
  want=$(git -C "$INFERENCE_REPO" rev-parse -q --verify "$INFERENCE_REF^{commit}" 2>/dev/null || true)
fi
[ -z "$want" ] && { echo "!! INFERENCE_REF=$INFERENCE_REF is not a valid commit in the harness clone — refusing to run"; exit 1; }
# Force the working tree to EXACTLY the pin every run. This discards any drift or a prior run's
# main.py patch (re-applied below), so a cached clone at another revision can't silently be used.
git -C "$INFERENCE_REPO" reset --hard -q "$want" || { echo "!! could not pin harness to $INFERENCE_REF"; exit 1; }
# Also drop UNTRACKED drift (a prior run's backend copy, stray files that could alter imports) so the
# harness tree is EXACTLY the pin, not just its tracked files (finding #5). backend_tensorrt.py is
# re-copied below, after this clean.
git -C "$INFERENCE_REPO" clean -fdq
have=$(git -C "$INFERENCE_REPO" rev-parse HEAD)
[ "$have" = "$want" ] || { echo "!! harness pin mismatch: HEAD $have != $want"; exit 1; }
echo "harness pinned: $(git -C "$INFERENCE_REPO" rev-parse --short HEAD) (INFERENCE_REF=$INFERENCE_REF)"

cat > "$CONF" <<'CONF'
resnet50.SingleStream.min_duration = 10000
# min_query_count must exceed QPS x (min_duration/1000) or the run ends before 10s => INVALID
# ("Min duration satisfied: NO"). The 5070 Ti does ~580 QPS at batch-1 (1.4ms), so 4000 finished
# in ~7s. 12000 keeps it >10s across laptop-throttle (~390) up to ~1000 QPS datacenter single-stream.
resnet50.SingleStream.min_query_count = 12000
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
  python - <<'PY' || pip install -q "huggingface-hub==0.36.2" "pyarrow==25.0.0"
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
# Validate the dataset whether just built OR pre-existing/cached — a non-empty val_map is NOT enough
# (a partial/poisoned set from an older build could linger). validate_dataset.py enforces an
# INDEPENDENT profile (sample-count floor, distinct-class floor, class-balance cap, unique paths,
# in-range int labels, every image present, non-empty & contained under DATA) so a truncated /
# duplicate / cherry-picked map can't self-certify a favorable accuracy subset.
# Defaults match the runner's DEFAULT dataset (the representative 5k-image / 1000-class mirror), so a
# zero-arg run is fail-closed. Pointing DATA= at the smaller Imagenette set (3925 imgs / 10 classes)
# requires opting into a reduced profile: MIN_SAMPLES=3000 MIN_CLASSES=10.
MIN_SAMPLES="${MIN_SAMPLES:-5000}" MIN_CLASSES="${MIN_CLASSES:-1000}" MAX_CLASS_FRACTION="${MAX_CLASS_FRACTION:-0.5}" \
  python "$SCRIPT_DIR/validate_dataset.py" "$DATA" "$VMAP" \
  || { echo "!! dataset validation failed at $DATA — rebuild, or set MIN_SAMPLES/MIN_CLASSES for a smaller DATA set"; exit 1; }
# EXPECTED is only trusted AFTER the profile passes (so the accuracy 'total == EXPECTED' cross-check
# can't be satisfied by a 1-row map that also sets EXPECTED=1).
EXPECTED=$(grep -c . "$VMAP")
echo "DATA=$DATA  ($EXPECTED images)"

cd "$H/python"
# Fresh, unique output dir per invocation so a failed run can NEVER reprint a prior run's summary,
# and two invocations started within the same second can't overwrite each other's logs (mktemp -d
# guarantees uniqueness — a second-resolution timestamp alone does not).
mkdir -p "$BENCH_ROOT/vision/runs"
RUNROOT=$(mktemp -d "$BENCH_ROOT/vision/runs/$(date +%Y%m%d-%H%M%S).XXXXXX")
# Publish the EXACT run dir so scripts/run_bundle.sh attaches this invocation's logs by identity, not
# by a racy global before/after listing (finding #4). Harmless when run outside the wrapper.
[ -n "${BUNDLE_RUNROOT_FILE:-}" ] && printf '%s\n' "$RUNROOT" > "$BUNDLE_RUNROOT_FILE"
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
# issue). We tolerate ONLY that: gate on the freshly-written artifact AND require the scorer to have
# processed EVERY sample (total == EXPECTED) — so a crash that produced only a favorable subset can't
# sneak past the 70% floor.
ACC="$RUNROOT/Accuracy"
run_main Offline "--accuracy" "$ACC"; ACC_RC=$?
# tolerate the known post-run numpy percentile crash; anything else is suspicious
if [ "$ACC_RC" -ne 0 ] && ! grep -qiE "percentile|numpy|IndexError|zero-size" "$ACC/run.log"; then
  echo "!! accuracy: main.py exited $ACC_RC for an UNRECOGNIZED reason (not the known post-run stats crash)"; tail -20 "$ACC/run.log"; FAILED=1
fi
AJSON="$ACC/mlperf_log_accuracy.json"
if [ ! -s "$AJSON" ]; then
  echo "!! accuracy: no mlperf_log_accuracy.json produced"; tail -20 "$ACC/run.log"; FAILED=1
else
  ACC_OUT=$(python ../tools/accuracy-imagenet.py --mlperf-accuracy-file "$AJSON" \
    --imagenet-val-file "$VMAP" --dtype float32 2>&1 | tail -1)
  echo "$ACC_OUT"
  TOP1=$(echo "$ACC_OUT" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  TOTAL=$(echo "$ACC_OUT" | grep -oE "total=[0-9]+" | grep -oE "[0-9]+" | head -1)
  if   [ -z "$TOP1" ];                              then echo "!! accuracy: could not parse top-1 from scorer"; FAILED=1
  elif [ -z "$TOTAL" ] || [ "$TOTAL" -ne "$EXPECTED" ]; then echo "!! accuracy scored ${TOTAL:-0}/$EXPECTED samples — partial/failed run, rejecting"; FAILED=1
  elif awk "BEGIN{exit !($TOP1 < $ACC_MIN)}";       then echo "!! accuracy ${TOP1}% < floor ${ACC_MIN}% — suspect run"; FAILED=1
  else echo "accuracy OK: ${TOP1}% over all $TOTAL samples (>= ${ACC_MIN}%)"; fi
fi

echo
if [ "$FAILED" -ne 0 ]; then echo "TRT MLPERF: FAILED — see errors above; run dir $RUNROOT"; exit 1; fi
echo "TRT MLPERF: all checks passed (LoadGen-VALID under short config); run dir $RUNROOT"
