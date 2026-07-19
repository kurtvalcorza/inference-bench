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
#   env-vars.txt  the benchmark env knobs  repo.diff     working-tree diff (only if repo was dirty)
#   env.txt       `pip freeze` AFTER run    nvidia-smi.txt GPU + driver
#   checksums.txt sha256 of assets AFTER run (captures freshly downloaded/rebuilt ones)
#   tensorrt_run/ copied detailed per-scenario LoadGen logs (if a TRT run produced them)
#   run.log       full stdout+stderr        exit_status  the command's real exit code
#   manifest.json machine-readable summary
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

# Preserve the exact working-tree diff when dirty, so a non-clean run is still reproducible.
[ "$DIRTY" = yes ] && git -C "$REPO_ROOT" diff HEAD > "$B/repo.diff" 2>/dev/null

# Record the benchmark env knobs that change results.
{ for v in BENCH_ROOT BENCH_VENV INFERENCE_REF LLAMA_REF DATA ONNX MAXBS BATCHES MATMUL_SIZES \
           REPEATS CUDA_ARCH ACC_MIN MODE GGUF_SHA256 CUDA_VISIBLE_DEVICES; do
    printf '%s=%s\n' "$v" "${!v-}"
  done; } > "$B/env-vars.txt"

printf '%q ' "$@" > "$B/command.txt"; echo >> "$B/command.txt"

echo "=== run bundle $B ==="
"$@" 2>&1 | tee "$B/run.log"
RC=${PIPESTATUS[0]}            # the command's REAL exit code, not tee's
echo "$RC" > "$B/exit_status"

# Capture env + checksums AFTER the run so freshly installed deps / downloaded / rebuilt assets show up.
pip freeze > "$B/env.txt" 2>/dev/null || echo "(pip freeze unavailable)" > "$B/env.txt"
nvidia-smi > "$B/nvidia-smi.txt" 2>&1 || echo "(no nvidia-smi)" > "$B/nvidia-smi.txt"
: > "$B/checksums.txt"
for f in "$BENCH_ROOT/vision/resnet50_fp16_dyn.onnx" "$BENCH_ROOT/vision/inet_val/val_map.txt" \
         "$BENCH_ROOT/llm/tinyllama.gguf"; do
  [ -f "$f" ] && sha256sum "$f" >> "$B/checksums.txt"
done

# Snapshot the detailed TRT per-scenario LoadGen logs (they live under BENCH_ROOT, not stdout).
NEWEST_TRT=$(ls -dt "$BENCH_ROOT"/vision/runs/*/ 2>/dev/null | head -1)
if [ -n "$NEWEST_TRT" ]; then
  mkdir -p "$B/tensorrt_run"; cp -r "$NEWEST_TRT". "$B/tensorrt_run/" 2>/dev/null || true
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
