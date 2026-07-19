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
| Polygraphy — ResNet-50 fp16, batch 128 | latency / throughput | 38.09 ms → **~3,361 img/s**§ |
| llama-bench — TinyLlama-1.1B Q4, **CUDA** @ b10068 | prefill / decode | **17,693 / 313 t/s**† |
| llama-bench — TinyLlama-1.1B Q4, **CPU** (24t) | prefill / decode | 410 / 27.2 t/s |

† **Committed** bundle `results/bundles/20260719T131317Z-llama-5070ti-b10068.TLuwNJ/` (`repo_dirty:
no`, model SHA-256-verified, pinned `LLAMA_REF=b10068`, arch `120`). Measured in a thermally throttled
session; prefill is also inherently noisy (±~2,000 t/s). Cooler earlier runs: ~19–21k / ~434–463 t/s.
§ **Committed** bundle `results/bundles/20260719T131444Z-polygraphy-5070ti.HgydkD/` (`repo_dirty: no`,
avg 38.09 ms over 200 iters). Cooler earlier runs: ~32 ms → ~3,965–4,125 img/s.
GPU vs CPU on the LLM: ~43× prefill, ~12× decode.

**Verified model hash** (TinyLlama-1.1B-Chat-v1.0 Q4_K_M, TheBloke GGUF), enforce it with:
```bash
GGUF_SHA256=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0 bash standards/llama_bench.sh
```

**Colab T4 llama-bench GPU: not obtained** — free Colab T4 VMs have only 2 vCPUs, so llama.cpp's CUDA
build doesn't finish within the session lifetime (even with `sm_75` + `-DGGML_CUDA_FORCE_CUBLAS=ON`).
Use Colab Pro or a real T4 box.

Context: Polygraphy's ~3,361 img/s (committed bundle) sits between our host-bound MLPerf-TRT SUT (~3,210) and the raw
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
