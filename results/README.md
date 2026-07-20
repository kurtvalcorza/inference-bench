# results/

**Run bundles** produced by [`../scripts/run_bundle.sh`](../scripts/run_bundle.sh) land in
`results/bundles/<UTC>-<label>.<rand>/`. Each bundle is a self-contained record for a number — it
captures the exact command, the benchmark env knobs, repo commit (+ working-tree diff if dirty),
`pip freeze` and asset SHA-256s taken **after** the run, GPU/driver, the detailed TRT per-scenario
logs, complete stdout/stderr, and the real exit status.

A bundle is a point-in-time **record**, not cryptographically immutable — it's reproducible (re-run
the recorded command in the recorded env), not tamper-proof. Bundles are **gitignored by default**
(generated on the host, can be large). **Three are committed** (force-added past the `.gitignore`) as
the citable evidence for the headline 5070 Ti numbers — the LoadGen+TensorRT, polygraphy, and
llama-bench (@ b10068) runs, each `repo_dirty: no` — so a reader can check their raw logs directly.
Any *other* "bundle-backed" number was verified against a bundle on the author's machine; publish it
the same way (`git add -f` or a release artifact) to make it independently checkable.

```bash
# wrap any runner in a bundle:
bash scripts/run_bundle.sh trt-5070ti -- bash tensorrt/trt_mlperf_run.sh
bash scripts/run_bundle.sh micro-a100 -- python microbench/gpu_bench.py
```

The prose tables in [../docs/results.md](../docs/results.md) are point-in-time summaries; the three
optimized 5070 Ti rows are now backed by the committed bundles above, but the rest are unsubstantiated
— see that file's *Provenance & caveats*. Prefer regenerating (or citing a committed) bundle over the
table.
