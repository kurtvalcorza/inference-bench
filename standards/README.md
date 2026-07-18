# standards/ — additional benchmark standards

Beyond MLPerf (`../reference`, `../tensorrt`) and the custom microbench (`../microbench`), these are
other widely-used benchmarks/tools, added for broader coverage.

| Tool | What / framework | Status on 5070 Ti |
|---|---|---|
| **Polygraphy** (`polygraphy_resnet.sh`) | NVIDIA TensorRT profiler — pip-installable `trtexec` equivalent | ✅ ran |
| **llama.cpp `llama-bench`** (`llama_bench.sh`) | LLM token throughput (prefill + decode), CPU + CUDA | ✅ ran |
| **AI-Benchmark** (`ai_benchmark.md`) | ETH "AI Score" over ~19 models (TensorFlow) | ⚠️ TF ≠ Blackwell → docّ (run on T4/CPU) |
| **MLPerf Client** (`mlperf_client.md`) | MLCommons consumer-PC LLM benchmark | ⚠️ Windows app → documented |

## Results (RTX 5070 Ti)

| Benchmark | Metric | Value |
|---|---|---|
| Polygraphy — ResNet-50 fp16, batch 128 | latency / throughput | 32.28 ms → **~3,965 img/s** |
| llama-bench — TinyLlama-1.1B Q4, **CUDA** | prefill / decode | **19,082 / 463 t/s** |
| llama-bench — TinyLlama-1.1B Q4, **CPU** (24t) | prefill / decode | 410 / 27.2 t/s |

GPU vs CPU on the LLM: ~46× prefill, ~17× decode. (The GPU build compiled for sm_120 / Blackwell.)

Context: Polygraphy's 3,965 img/s sits between our host-bound MLPerf-TRT SUT (~3,100) and the raw
microbench ceiling (4,774) — it keeps tensors on-GPU (less host overhead than the MLPerf harness) but
isn't the LoadGen harness.

## Run

```bash
# NVIDIA TensorRT profiler (trtexec-style)
bash standards/polygraphy_resnet.sh                 # BS=256 bash ... to change batch

# LLM token throughput
MODE=cuda bash standards/llama_bench.sh             # GPU (needs CUDA toolkit)
MODE=cpu  bash standards/llama_bench.sh             # CPU only, any machine
```

`ai_benchmark.md` and `mlperf_client.md` explain how/where to run those two (they don't fit the
Blackwell-WSL headless flow — see the caveats there).

## The full benchmark landscape

For the broader survey of ML/GPU benchmarking standards (DeepBench, Triton perf_analyzer, Geekbench
ML, Procyon, STREAM/HPL, etc.) and which fit this hardware, see the notes in the project history —
the four here are the highest-value additions for cross-hardware inference comparison.
