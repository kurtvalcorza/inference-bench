# microbench/ — portable GPU/CPU hardware benchmark

**Custom, not MLPerf.** Direct timing loops that characterize raw hardware. Auto-detect the device;
print a table + JSON. Runs on any CUDA GPU / any CPU.

| Script | Measures |
|---|---|
| `gpu_bench.py` | matmul TFLOPS (fp32/tf32/fp16/bf16), memory bandwidth, ResNet-50 fp16 throughput (eager / `torch.compile` / TensorRT) |
| `cpu_bench.py` | matmul GFLOPS (fp32/bf16), bandwidth, ResNet-50 throughput (fp32 / bf16 / compile) |

## Run
```bash
python gpu_bench.py
python cpu_bench.py

# big GPUs (A100/H200) — sweep larger so they aren't understated:
MATMUL_SIZES=16384,24576,32768 BATCHES=256,512,1024,2048 python gpu_bench.py
```

Each tier degrades gracefully (missing `tensorrt` → that row shows `skipped`). Paste the JSON to
compare machines.

## Results

| Metric | RTX 5070 Ti | Colab T4 | CPU (Ultra 9 275HX) |
|---|---|---|---|
| FP16 TFLOPS | 42.7 | 22.7 | — |
| Bandwidth (GB/s) | 498 | 232 | 51 |
| ResNet-50 fp16 TensorRT | 4,774 img/s | 1,945 | — |
| ResNet-50 (CPU, compiled) | — | — | 27 img/s |

Full tables: [../docs/results.md](../docs/results.md).
