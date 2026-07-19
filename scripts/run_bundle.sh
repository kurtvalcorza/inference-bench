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
#   repo.untracked untracked files (if dirty) env.txt     `pip freeze` AFTER run
#   nvidia-smi.txt GPU + driver             checksums.txt sha256 of assets + a root hash over every
#                                                         inet_val image (captures rebuilt/downloaded)
#   tensorrt_run/ per-scenario LoadGen logs THIS command created (none attached if it created none)
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
  git -C "$REPO_ROOT" ls-files --others --exclude-standard > "$B/repo.untracked" 2>/dev/null
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

# Snapshot the set of existing TRT run dirs BEFORE the command, so afterwards we can attach ONLY the
# dirs this invocation created — never a pre-existing run's logs (finding #2).
TRT_RUNS_DIR="$BENCH_ROOT/vision/runs"
PRE_TRT=$(ls -d "$TRT_RUNS_DIR"/*/ 2>/dev/null | sort)

echo "=== run bundle $B ==="
"$@" 2>&1 | tee "$B/run.log"
RC=${PIPESTATUS[0]}            # the command's REAL exit code, not tee's
echo "$RC" > "$B/exit_status"

# Capture env + checksums AFTER the run so freshly installed deps / downloaded / rebuilt assets show up.
pip freeze > "$B/env.txt" 2>/dev/null || echo "(pip freeze unavailable)" > "$B/env.txt"
nvidia-smi > "$B/nvidia-smi.txt" 2>&1 || echo "(no nvidia-smi)" > "$B/nvidia-smi.txt"
: > "$B/checksums.txt"
for f in "$BENCH_ROOT/vision/resnet50_fp16_dyn.onnx" "$BENCH_ROOT/vision/inet_val/val_map.txt" \
         "$BENCH_ROOT/vision/inet_val/dataset_manifest.txt" "$BENCH_ROOT/llm/tinyllama.gguf"; do
  [ -f "$f" ] && sha256sum "$f" >> "$B/checksums.txt"
done
# Attest to image CONTENT, not just val_map.txt (finding #4): one deterministic root hash over every
# val image's sha256, so swapped/altered images are detectable even when val_map is unchanged.
VAL_DIR="$BENCH_ROOT/vision/inet_val"
if [ -d "$VAL_DIR" ]; then
  ROOT=$(find "$VAL_DIR" -type f -name '*.JPEG' -print0 2>/dev/null | sort -z \
         | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
  [ -n "$ROOT" ] && echo "$ROOT  (root sha256 of all inet_val/*.JPEG)" >> "$B/checksums.txt"
fi

# Snapshot the detailed TRT per-scenario LoadGen logs — but ONLY dirs this run created (finding #2).
# Diff the post-run dir set against the pre-run snapshot; if the wrapped command created none, attach
# nothing rather than a stale, unrelated run's logs.
POST_TRT=$(ls -d "$TRT_RUNS_DIR"/*/ 2>/dev/null | sort)
NEW_TRT=$(comm -13 <(printf '%s\n' "$PRE_TRT") <(printf '%s\n' "$POST_TRT") | sed '/^$/d')
if [ -n "$NEW_TRT" ]; then
  mkdir -p "$B/tensorrt_run"
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    cp -r "$d" "$B/tensorrt_run/" 2>/dev/null || true
  done <<< "$NEW_TRT"
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
