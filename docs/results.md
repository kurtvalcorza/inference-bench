# Results

All numbers measured with torch **2.11.0+cu128**, TensorRT **11.1**. Laptop 5070 Ti figures vary
±10% run-to-run (thermal throttling); datacenter/Colab figures are stable.

## Hardware

| GPU | Arch | VRAM | SMs | Notes |
|---|---|---|---|---|
| RTX 5070 Ti Laptop | Blackwell sm_120 | 12.8 GB | 46 | thermally limited (laptop) |
| Colab T4 | Turing sm_75 | 16 GB | 40 | no TF32/BF16 tensor cores |
| CPU: Intel Core Ultra 9 275HX | — | — | 24 threads | |

---

## MLPerf Inference — reference implementations

| Domain | Model / dataset | Metric | RTX 5070 Ti | Colab T4 |
|---|---|---|---|---|
| NLP | BERT-Large / SQuAD v1.1 | f1 (Offline) | **90.40** | **90.40** |
| | | throughput | 25.7 samples/s | 10.9 samples/s |
| Vision | ResNet-50 / ImageNet | top-1 (Imagenette) | 84.5% | 84.6% |
| | | top-1 (repr. 1000-class) | 75.4% | — |
| | | throughput (Offline) | 552 samples/s | 304 samples/s |
| Speech | whisper-large-v3 / LibriSpeech | WER (dev-clean) | 3.6–5.0% | 2.16% |
| | | RTF | ~0.16 | ~0.31 |

CPU (5070 Ti host, reference): BERT f1 88.74 @ 1.37 samples/s.
Accuracy is essentially hardware-independent (f1 90.40 identical across GPUs); throughput is the
hardware signal.

---

## MLPerf ResNet-50 + TensorRT (fp16, LoadGen) — all VALID

| Scenario | Metric | RTX 5070 Ti | Colab T4 |
|---|---|---|---|
| SingleStream | p50 latency | 2.23 ms | 2.63 ms |
| | **p90 latency** | ~4.2 ms† | **2.80 ms** |
| | p99 latency | 8.12 ms | 4.70 ms |
| Offline | **throughput** | **~3,100–3,325 img/s** | 1,200 img/s |
| Accuracy | top-1 | 75.4% (repr) / 84.5% (Imagenette) | 84.6% (Imagenette) |

† Laptop SingleStream latency is noisy (4.17 ms one run, 10.3 ms another — thermal).

**Findings.** The T4 has *lower, cleaner* single-stream latency (stable clock beats a throttling
laptop at batch-1, where the workload is latency/host-bound). The 5070 Ti has ~2.7× the Offline
throughput (sustained compute wins). `max_batchsize` 32→128 did **not** help (host-bound SUT).

---

## Microbenchmarks (custom, not MLPerf)

### GPU

| Metric | RTX 5070 Ti | Colab T4 |
|---|---|---|
| FP32 TFLOPS | 11.8 | 3.9 |
| TF32 TFLOPS | 20.4 | 3.9¹ |
| FP16 TFLOPS | 42.7 | 22.7 |
| BF16 TFLOPS | 51.4 | 2.1² |
| Memory bandwidth | 498 GB/s | 232 GB/s |
| ResNet-50 fp16 — eager | 1,873 img/s | 1,069 |
| ResNet-50 fp16 — torch.compile | 2,982 img/s | 1,519 |
| ResNet-50 fp16 — **TensorRT** | **4,774 img/s** | **1,945** |

¹ Turing has no TF32 tensor cores (TF32 == FP32). ² Turing has no BF16 tensor cores (slow fallback).

### CPU — Intel Core Ultra 9 275HX (24 threads)

| Metric | Value |
|---|---|
| FP32 GFLOPS | 638 |
| BF16 GFLOPS | 1,309 |
| Memory bandwidth | 51 GB/s |
| ResNet-50 fp32 | 20.8 img/s |
| ResNet-50 torch.compile | 27.2 img/s |

The GPU (TensorRT) is **~175×** the CPU on ResNet-50 — why inference runs on GPUs.

---

## Other standards (`standards/`)

| Benchmark | Metric | RTX 5070 Ti | Colab T4 |
|---|---|---|---|
| Polygraphy (trtexec equiv) — ResNet-50 fp16 bs128 | throughput | ~3,965 img/s | — |
| llama.cpp llama-bench — TinyLlama-1.1B Q4, **GPU** | prefill / decode | 19,082 / 463 t/s | N/A† |
| llama.cpp llama-bench — TinyLlama-1.1B Q4, **CPU** (24t) | prefill / decode | 410 / 27.2 t/s | — |
| AI-Benchmark (ETH) | AI Score | run on T4/CPU (TF ≠ Blackwell) | |
| MLPerf Client | tokens/s, TTFT | native-Windows app (see doc) | |

GPU vs CPU on the LLM (5070 Ti): ~46× prefill, ~17× decode.

† **T4 llama-bench GPU: not obtained on free Colab.** Free Colab T4 VMs have only **2 vCPUs**;
llama.cpp's CUDA build (many flash-attention / kernel template instances) doesn't finish within the
session lifetime, even with `-DGGML_CUDA_FORCE_CUBLAS=ON` and pinned `sm_75`. Get it on Colab Pro
(more vCPUs) or a real T4 box, or use a prebuilt CUDA binary.

## Reference vs optimized vs raw (ResNet-50, 5070 Ti)

| Path | img/s | What it measures |
|---|---|---|
| MLPerf reference (PyTorch) | 552 | unoptimized reference harness |
| MLPerf + TensorRT (this suite) | ~3,100 | LoadGen + optimized backend, VALID |
| Raw microbench (TensorRT) | 4,774 | GPU ceiling, no harness/host overhead |

The gap between the middle and bottom rows is the reference-grade SUT's host overhead
(per-query numpy copies + lock), not the GPU — see [architecture.md](architecture.md).

---

## Pending

- **A100 (sm_80) / H200 (sm_90)** — run `microbench/gpu_bench.py` (with a large sweep) and/or
  `tensorrt/trt_mlperf_run.sh` on the work boxes (or a Brev cloud instance) and paste the JSON.
- **Work-machine CPUs** — `microbench/cpu_bench.py`.
