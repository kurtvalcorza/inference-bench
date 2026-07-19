#!/usr/bin/env bash
# Wrap any benchmark command in an IMMUTABLE run bundle for reproducibility.
#
#   bash scripts/run_bundle.sh <label> -- <command...>
#
# Example:
#   bash scripts/run_bundle.sh trt-5070ti -- bash tensorrt/trt_mlperf_run.sh
#   bash scripts/run_bundle.sh micro-a100 -- python microbench/gpu_bench.py
#
# Produces results/bundles/<UTC>-<label>/ containing everything needed to trust and
# reproduce the number:
#   command.txt      the exact command
#   meta.txt         repo commit + dirty flag, host, OS, python
#   env.txt          full `pip freeze`
#   nvidia-smi.txt   GPU + driver
#   checksums.txt    sha256 of the ONNX / val_map / gguf assets that were present
#   run.log          full stdout+stderr
#   exit_status      the command's real exit code
#   manifest.json    machine-readable summary of the above
#
# The bundle — not a docs table — is the citable artifact. Bundles are gitignored
# (generated on the benchmark host); copy the directory out to share it.
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

LABEL=$(printf '%s' "$LABEL" | tr -c 'A-Za-z0-9._-' '_')   # sanitize: no '/', no traversal
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# RESULTS_ROOT lets tests/callers redirect output away from real bundles; mktemp guarantees a
# UNIQUE dir even for the same label within one second (no silent overwrite).
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/bundles}"
mkdir -p "$RESULTS_ROOT"
B=$(mktemp -d "$RESULTS_ROOT/${STAMP}-${LABEL}.XXXXXX")

{
  echo "label: $LABEL"
  echo "utc: $STAMP"
  echo "host: $(hostname)"
  echo "repo_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "repo_dirty: $([ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ] && echo yes || echo no)"
  echo "os: $(uname -a 2>/dev/null)"
  echo "python: $(python --version 2>&1)"
} > "$B/meta.txt"

printf '%q ' "$@" > "$B/command.txt"; echo >> "$B/command.txt"
pip freeze > "$B/env.txt" 2>/dev/null || echo "(pip freeze unavailable)" > "$B/env.txt"
nvidia-smi > "$B/nvidia-smi.txt" 2>&1 || echo "(no nvidia-smi)" > "$B/nvidia-smi.txt"

: > "$B/checksums.txt"
BR="${BENCH_ROOT:-/root/mlperf}"
for f in "$BR/vision/resnet50_fp16_dyn.onnx" "$BR/vision/inet_val/val_map.txt" "$BR/llm/tinyllama.gguf"; do
  [ -f "$f" ] && sha256sum "$f" >> "$B/checksums.txt"
done

echo "=== run bundle $B ==="
"$@" 2>&1 | tee "$B/run.log"
RC=${PIPESTATUS[0]}            # the command's REAL exit code, not tee's
echo "$RC" > "$B/exit_status"

python - "$B" "$LABEL" "$STAMP" "$RC" <<'PY'
import json, os, sys
B, label, stamp, rc = sys.argv[1:5]
def read(p):
    try:
        return open(os.path.join(B, p)).read().strip()
    except OSError:
        return None
manifest = {
    "label": label,
    "utc": stamp,
    "exit_status": int(rc),
    "passed": int(rc) == 0,
    "command": read("command.txt"),
    "meta": read("meta.txt"),
    "checksums": read("checksums.txt"),
    "files": sorted(os.listdir(B)),
}
json.dump(manifest, open(os.path.join(B, "manifest.json"), "w"), indent=2)
PY

echo
echo "bundle: $B  (exit $RC, $([ "$RC" -eq 0 ] && echo PASS || echo FAIL))"
exit "$RC"
