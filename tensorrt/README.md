# tensorrt/ — LoadGen + TensorRT backend (MLPerf-inspired)

The MLCommons LoadGen harness driving an **optimized TensorRT fp16 SUT** — SingleStream (latency),
Offline (throughput), and accuracy.

> **Not conformant MLPerf.** Runs use a short config (10 s `min_duration`, Offline
> `min_query_count=1`) on a subset dataset, not MLPerf's ~600 s / full-validation-set rules. A
> LoadGen "VALID" line here means the run met *that short config* — not MLPerf conformance. Don't
> report these under the MLPerf label. See [../docs/architecture.md](../docs/architecture.md#what-is-and-isnt-mlperf).

| File | Purpose |
|---|---|
| `backend_tensorrt.py` | MLPerf `Backend`: dynamic-batch (1..MAXBS) fp16 **strongly-typed** TRT engine; thread-locked; warmed up |
| `export_resnet50_onnx.py` | torchvision ResNet-50 → fp16 dynamic-batch ONNX |
| `trt_mlperf_run.sh` | installs the backend into the harness, patches `main.py`, exports ONNX, runs the scenarios |

## Run
```bash
bash trt_mlperf_run.sh                 # default max-batchsize 128
MAXBS=32 bash trt_mlperf_run.sh        # override
```

## Results (LoadGen-VALID under this suite's short config — not MLPerf-conformant)

| Scenario | RTX 5070 Ti | Colab T4 |
|---|---|---|
| SingleStream p90 | ~5.5 ms (VALID, laptop-noisy: 4.6–7.7) | 2.80 ms |
| Offline | 3,210 img/s (VALID) | 1,200 img/s |
| Accuracy | 75.34% / 84.5% | 84.6% |

Numbers from the committed bundle `results/bundles/20260719T122957Z-trt-5070ti-hardened.BnXBbN/`
(clean `repo_commit`); SingleStream p90 is thermally noisy on this laptop, Offline is the stable
signal — see [../docs/results.md](../docs/results.md#loadgen--tensorrt-resnet-50-fp16--loadgen-valid-under-this-suites-short-config-not-mlperf-conformant).

This is a **reference-grade SUT** (host-bound: per-query numpy copies + lock) — Offline throughput is
below the raw GPU ceiling (`microbench` 4,774). A submission would use pinned memory + async I/O.
See [../docs/architecture.md](../docs/architecture.md#the-tensorrt-backend-tensorrtbackend_tensorrtpy).
