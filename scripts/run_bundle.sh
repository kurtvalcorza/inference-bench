#!/usr/bin/env bash
# Wrap any benchmark command in a self-contained run bundle for reproducibility.
#
#   bash scripts/run_bundle.sh <label> -- <command...>
#
# Example:
#   bash scripts/run_bundle.sh trt-5070ti -- bash tensorrt/trt_mlperf_run.sh
#   bash scripts/run_bundle.sh micro-a100 -- python microbench/gpu_bench.py
#
# Produces results/bundles/<UTC>-<label>.<rand>/ with everything needed to trust/reproduce a number:
#   command.txt   exact command            meta.txt      repo commit+dirty, host, OS, python
#   env-vars.txt  the benchmark env knobs  repo.diff     tracked working-tree diff (if repo dirty)
#   repo.untracked untracked files + sha256 (if dirty)  env.txt  `pip freeze` AFTER run
#   nvidia-smi.txt GPU + driver             checksums.txt sha256 of the EFFECTIVE assets used (honors
#                                                         DATA/ONNX overrides) + a root hash over them
#   tensorrt_run/ the exact run dir the runner reported (by marker, not a global diff; none if absent)
#   run.log       full stdout+stderr        exit_status  the command's real exit code
#   manifest.json machine-readable summary
#
# It records the env KNOBS the runners read plus a full `pip freeze` — enough to re-run, but it is a
# best-effort record, not a guarantee it captures every result-affecting variable on your host.
#
# NOTE: a bundle is a point-in-time RECORD, not cryptographically immutable. It is self-contained
# (re-run the recorded command in the recorded env), but it is not tamper-proof. Bundles are
# gitignored (generated on the host, can be large); publish one deliberately with `git add -f` or as
# a release artifact if you want a stable link.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LABEL="${1:-run}"; shift || true
[ "${1:-}" = "--" ] && shift || true
if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/run_bundle.sh <label> -- <command...>"; exit 2
fi

# Activate the same venv the runner would, so `pip freeze` reflects the real env.
VENV="${BENCH_VENV:-/root/mlperf/venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# Resolve BENCH_ROOT exactly like the runners do, so checksums point at the assets they actually use.
if [ -z "${BENCH_ROOT:-}" ]; then
  if [ -d /root/mlperf ]; then BENCH_ROOT=/root/mlperf; else BENCH_ROOT="$HOME/inference-bench-data"; fi
fi

LABEL=$(printf '%s' "$LABEL" | tr -c 'A-Za-z0-9._-' '_')   # sanitize: no '/', no traversal
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# RESULTS_ROOT lets tests/callers redirect output away from real bundles; mktemp guarantees a
# UNIQUE dir even for the same label within one second (no silent overwrite).
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/bundles}"
mkdir -p "$RESULTS_ROOT"
B=$(mktemp -d "$RESULTS_ROOT/${STAMP}-${LABEL}.XXXXXX")

DIRTY=$([ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ] && echo yes || echo no)
{
  echo "label: $LABEL"
  echo "utc: $STAMP"
  echo "host: $(hostname)"
  echo "bench_root: $BENCH_ROOT"
  echo "repo_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "repo_dirty: $DIRTY"
  echo "os: $(uname -a 2>/dev/null)"
  echo "python: $(python --version 2>&1)"
} > "$B/meta.txt"

# Preserve the exact working-tree state when dirty, so a non-clean run is still reproducible.
# `git diff HEAD` covers TRACKED changes only — also list UNTRACKED files so `repo_dirty: yes` is
# never left unexplained (finding #5).
if [ "$DIRTY" = yes ]; then
  git -C "$REPO_ROOT" diff HEAD > "$B/repo.diff" 2>/dev/null
  # List untracked files WITH their content sha256 (not just names) so a dirty run is verifiable —
  # names alone can't reproduce what those files contained (finding #5).
  git -C "$REPO_ROOT" ls-files --others --exclude-standard -z 2>/dev/null \
    | while IFS= read -r -d '' f; do
        if [ -f "$REPO_ROOT/$f" ]; then
          printf '%s  %s\n' "$(sha256sum "$REPO_ROOT/$f" | cut -d' ' -f1)" "$f"
        fi
      done > "$B/repo.untracked" 2>/dev/null
fi

# Record the benchmark env knobs that change results. Keep this in sync with the env vars the runners
# actually read (finding #5): batch size, harness/model paths, build parallelism, opt-in unsafe load,
# dataset floors/revision, and asset hashes all change the result or its provenance.
{ for v in BENCH_ROOT BENCH_VENV INFERENCE_REF INFERENCE_REPO LLAMA_REF DATA ONNX RESNET50_PTH \
           MAXBS BS BATCHES MATMUL_SIZES REPEATS JOBS CUDA_ARCH CUDA_HOME ACC_MIN MIN_SAMPLES \
           MIN_CLASSES MODE N_UTT ALLOW_UNSAFE_PICKLE GGUF_SHA256 IMAGENETTE_SHA256 \
           DATASET_REVISION CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$v" "${!v-}"
  done; } > "$B/env-vars.txt"

printf '%q ' "$@" > "$B/command.txt"; echo >> "$B/command.txt"

# Attribute TRT logs by IDENTITY, not a racy global before/after listing (finding #4): tell the runner
# where to record its exact run dir. Only trt_mlperf_run.sh writes this; other commands leave it empty.
TRT_MARKER="$B/.trt_runroot"
export BUNDLE_RUNROOT_FILE="$TRT_MARKER"

echo "=== run bundle $B ==="
"$@" 2>&1 | tee "$B/run.log"
RC=${PIPESTATUS[0]}            # the command's REAL exit code, not tee's
echo "$RC" > "$B/exit_status"

# Capture env + checksums AFTER the run so freshly installed deps / downloaded / rebuilt assets show up.
pip freeze > "$B/env.txt" 2>/dev/null || echo "(pip freeze unavailable)" > "$B/env.txt"
nvidia-smi > "$B/nvidia-smi.txt" 2>&1 || echo "(no nvidia-smi)" > "$B/nvidia-smi.txt"
: > "$B/checksums.txt"
# Hash the assets the wrapped command ACTUALLY used — honor DATA/ONNX overrides, don't hardcode the
# defaults (finding #3). Resolve them the same way the runners do.
EFF_DATA="${DATA:-$BENCH_ROOT/vision/inet_val}"
EFF_ONNX="${ONNX:-$BENCH_ROOT/vision/resnet50_fp16_dyn.onnx}"
for f in "$EFF_ONNX" "$EFF_DATA/val_map.txt" "$EFF_DATA/dataset_manifest.txt" \
         "$BENCH_ROOT/llm/tinyllama.gguf"; do
  [ -f "$f" ] && sha256sum "$f" >> "$B/checksums.txt"
done
# Also hash any positional FILE argument (e.g. polygraphy's ONNX model passed as $1), so a run using a
# non-default model isn't recorded with the default model's hash.
for a in "$@"; do
  [ -f "$a" ] && [ "$a" != "$EFF_ONNX" ] && sha256sum "$a" >> "$B/checksums.txt"
done
# Attest to image CONTENT of the EFFECTIVE data dir, not just val_map.txt: one deterministic root hash
# over every val image's sha256, so swapped/altered images are detectable.
# ONLY write the line when images actually exist — a naive find|xargs sha256sum|sha256sum over a dir
# with no matches hashes EMPTY input and yields a non-empty ROOT (e3b0c442…), a bogus line attesting
# nothing (and `xargs -r` doesn't help: the OUTER sha256sum still hashes empty). Match case-insensitive
# .jpg/.jpeg so non-default DATA= sets are covered, not just the mirror's uppercase .JPEG.
if [ -d "$EFF_DATA" ]; then
  _first_img=$(find "$EFF_DATA" -type f \( -iname '*.jpeg' -o -iname '*.jpg' \) -print 2>/dev/null | head -1)
  if [ -n "$_first_img" ]; then
    ROOT=$(find "$EFF_DATA" -type f \( -iname '*.jpeg' -o -iname '*.jpg' \) -print0 2>/dev/null | sort -z \
           | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
    echo "$ROOT  (root sha256 of all *.jp*g under $EFF_DATA)" >> "$B/checksums.txt"
  fi
fi

# Attach the TRT per-scenario LoadGen logs by IDENTITY (finding #4): copy exactly the run dir the
# runner reported via the marker — no global before/after diff, so overlapping wrappers can't steal
# each other's dirs. A copy failure FAILS the bundle rather than silently producing a PASS with no logs.
if [ -s "$TRT_MARKER" ]; then
  RR=$(head -1 "$TRT_MARKER")
  if [ -d "$RR" ]; then
    mkdir -p "$B/tensorrt_run"
    cp -r "$RR" "$B/tensorrt_run/" || { echo "!! failed copying TRT logs from $RR — bundle incomplete"; exit 1; }
  else
    echo "(runner reported run dir '$RR' but it does not exist)" > "$B/tensorrt_run.note"
  fi
  rm -f "$TRT_MARKER"
else
  echo "(no TRT run dir created by this command)" > "$B/tensorrt_run.note"
fi

if ! python - "$B" "$LABEL" "$STAMP" "$RC" <<'PY'
import json, os, sys
B, label, stamp, rc = sys.argv[1:5]
def read(p):
    try:
        # explicit utf-8 (+ replace): files may hold non-ASCII (e.g. repo.diff), and Python's
        # default encoding is cp1252 on Windows — read() there would crash the manifest step.
        with open(os.path.join(B, p), encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None
manifest = {
    "label": label, "utc": stamp, "exit_status": int(rc), "passed": int(rc) == 0,
    "command": read("command.txt"), "meta": read("meta.txt"),
    "env_vars": read("env-vars.txt"), "checksums": read("checksums.txt"),
    "repo_dirty_diff": bool(read("repo.diff")),
    "repo_untracked": bool(read("repo.untracked")),
    "files": sorted(os.listdir(B)),
}
with open(os.path.join(B, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
PY
then
  echo "!! manifest generation FAILED — bundle incomplete"; exit 1
fi

# Validate the bundle before claiming success: manifest must exist and be valid JSON.
if ! python -c "import json,sys; json.load(open(sys.argv[1]))" "$B/manifest.json" 2>/dev/null; then
  echo "!! bundle validation FAILED (bad/missing manifest.json)"; exit 1
fi

echo
echo "bundle: $B  (exit $RC, $([ "$RC" -eq 0 ] && echo PASS || echo FAIL))"
exit "$RC"
