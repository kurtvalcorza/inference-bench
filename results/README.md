# results/

Immutable **run bundles** produced by [`../scripts/run_bundle.sh`](../scripts/run_bundle.sh) land in
`results/bundles/<UTC>-<label>/`. Each bundle is the citable artifact for a number — it captures the
exact command, repo commit, full `pip freeze`, GPU/driver, asset SHA-256s, complete logs, and the
real exit status.

Bundles are **gitignored** (they're generated on the benchmark host and can be large). To share one,
copy its directory out, or commit a specific bundle deliberately with `git add -f`.

```bash
# wrap any runner in a bundle:
bash scripts/run_bundle.sh trt-5070ti -- bash tensorrt/trt_mlperf_run.sh
bash scripts/run_bundle.sh micro-a100 -- python microbench/gpu_bench.py
```

The prose tables in [../docs/results.md](../docs/results.md) are point-in-time summaries, **not**
substantiated artifacts — see that file's *Provenance & caveats*. Prefer regenerating a bundle over
citing the table.
