# tensorrt/ — MLPerf ResNet-50 + TensorRT backend

The **"official MLPerf way"** of hardware benchmarking: the MLCommons LoadGen harness driving an
**optimized TensorRT fp16 SUT**. Produces **VALID** SingleStream (p99 latency) + Offline
(throughput) + accuracy.

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

## Results (all VALID)

| Scenario | RTX 5070 Ti | Colab T4 |
|---|---|---|
| SingleStream p90 | ~4.2 ms | 2.80 ms |
| Offline | ~3,100 img/s | 1,200 img/s |
| Accuracy | 75.4% / 84.5% | 84.6% |

This is a **reference-grade SUT** (host-bound: per-query numpy copies + lock) — Offline throughput is
below the raw GPU ceiling (`microbench` 4,774). A submission would use pinned memory + async I/O.
See [../docs/architecture.md](../docs/architecture.md#the-tensorrt-backend-tensorrtbackend_tensorrtpy).
