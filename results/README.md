# results/

**Run bundles** produced by [`../scripts/run_bundle.sh`](../scripts/run_bundle.sh) land in
`results/bundles/<UTC>-<label>.<rand>/`. Each bundle is a self-contained record for a number — it
captures the exact command, the benchmark env knobs, repo commit (+ working-tree diff if dirty),
`pip freeze` and asset SHA-256s taken **after** the run, GPU/driver, the detailed TRT per-scenario
logs, complete stdout/stderr, and the real exit status.

A bundle is a point-in-time **record**, not cryptographically immutable — it's reproducible (re-run
the recorded command in the recorded env), not tamper-proof. Bundles are **gitignored** (generated on
the host, can be large), so a number described as "bundle-backed" in the docs was verified against a
bundle **on the author's machine** — to make it independently checkable, publish that bundle
deliberately with `git add -f` or as a release artifact.

```bash
# wrap any runner in a bundle:
bash scripts/run_bundle.sh trt-5070ti -- bash tensorrt/trt_mlperf_run.sh
bash scripts/run_bundle.sh micro-a100 -- python microbench/gpu_bench.py
```

The prose tables in [../docs/results.md](../docs/results.md) are point-in-time summaries, **not**
substantiated artifacts — see that file's *Provenance & caveats*. Prefer regenerating a bundle over
citing the table.
