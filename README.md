# inference-bench

A self-contained suite for running **ML inference benchmarks** and **GPU/CPU hardware
benchmarks** across heterogeneous hardware — a laptop **RTX 5070 Ti** (Blackwell, sm_120), a
**Colab T4** (Turing, sm_75), datacenter GPUs (**A100/H200**), and CPUs — with reproducible
scripts, notebooks, and results.

It contains two distinct things, kept clearly separate:

| | What | Framework |
|---|---|---|
| **`reference/`** | MLPerf Inference reference implementations (BERT, ResNet-50, Whisper) | **Official MLCommons** LoadGen harness |
| **`tensorrt/`** | MLPerf ResNet-50 with an optimized **TensorRT** backend | **Official MLCommons** LoadGen + custom SUT |
| **`microbench/`** | Raw GPU/CPU microbenchmarks (TFLOPS, bandwidth, throughput) | **Custom** (not MLPerf) |
| **`standards/`** | Other standards: Polygraphy/trtexec, llama.cpp, AI-Benchmark, MLPerf Client | Third-party |

> **Honesty note.** Only `reference/` and `tensorrt/` use the real MLCommons framework.
> `microbench/` is homegrown — accurate for comparing hardware, but not MLPerf. See
> [docs/architecture.md](docs/architecture.md#what-is-and-isnt-mlperf).

## Quick start

Everything runs in the isolated `mlperf` WSL distro (setup in [docs/setup.md](docs/setup.md)):

```bash
wsl -d mlperf && source /root/mlperf/venv/bin/activate

# 1. Portable hardware microbench (fastest, any GPU)
python microbench/gpu_bench.py            # GPU: TFLOPS, bandwidth, ResNet-50 throughput
python microbench/cpu_bench.py            # CPU-only version

# 2. MLPerf + TensorRT (the "official MLPerf way" of HW benchmarking)
bash tensorrt/trt_mlperf_run.sh           # ResNet-50: VALID SingleStream + Offline + accuracy

# 3. MLPerf reference implementations — via Jupyter notebooks
jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root
#   then open reference/local/*.ipynb, pick the "mlperf (venv)" kernel, Run All
```

Colab (headless via the `google-colab-cli`) and per-benchmark instructions are in
[docs/user-guide.md](docs/user-guide.md).

## Repository layout

```
inference-bench/
├── README.md                     ← you are here
├── docs/
│   ├── setup.md                  ← WSL distro, deps, Colab CLI, HF, TensorRT
│   ├── user-guide.md             ← how to run each benchmark, step by step
│   ├── architecture.md           ← design: harness, backends, what is/isn't MLPerf
│   ├── results.md                ← all measured results (5070 Ti, T4, CPU)
│   └── gotchas.md                ← every hard-won fix (Blackwell SDPA, TRT, threading…)
├── reference/                    ← MLPerf reference implementations
│   ├── local/                    ← Jupyter notebooks (run in the WSL distro)
│   │   ├── mlperf_bert_squad_local.ipynb
│   │   ├── mlperf_resnet50_local.ipynb
│   │   └── mlperf_whisper_local.ipynb
│   └── colab/                    ← Colab notebooks (+ executed *_output copies)
├── tensorrt/                     ← MLPerf ResNet-50 + TensorRT backend
│   ├── backend_tensorrt.py       ← the SUT backend (dynamic-batch fp16 engine)
│   ├── export_resnet50_onnx.py   ← torchvision → fp16 dynamic ONNX
│   └── trt_mlperf_run.sh         ← installs backend, patches main.py, runs scenarios
├── microbench/
│   ├── gpu_bench.py              ← GPU TFLOPS / bandwidth / ResNet throughput
│   └── cpu_bench.py              ← CPU version
└── standards/                    ← other benchmark standards
    ├── polygraphy_resnet.sh      ← NVIDIA TensorRT profiler (trtexec equivalent)
    ├── llama_bench.sh            ← llama.cpp LLM token throughput (CPU + CUDA)
    ├── ai_benchmark.md           ← AI-Benchmark (ETH) — TF, run on T4/CPU
    └── mlperf_client.md          ← MLPerf Client — consumer-PC LLM (Windows app)
```

## Results at a glance

| Benchmark | RTX 5070 Ti | Colab T4 |
|---|---|---|
| **MLPerf ResNet-50 (TensorRT)** — Offline | ~3,100 img/s | 1,200 img/s |
| **MLPerf ResNet-50 (TensorRT)** — SingleStream p90 | ~4.2 ms | 2.80 ms |
| MLPerf BERT/SQuAD (reference) — f1 | 90.40 | 90.40 |
| MLPerf ResNet-50 (reference) — top-1 | 75.4% / 84.5% | 84.6% |
| MLPerf Whisper (reference) — WER | ~3.5–5% | 2.16% |
| microbench ResNet-50 fp16 (TensorRT) | 4,774 img/s | 1,945 img/s |
| microbench FP16 / BF16 TFLOPS | 42.7 / 51.4 | 22.7 / 2.1 |
| Polygraphy (trtexec) ResNet-50 fp16 bs128 | ~3,965 img/s | — |
| llama-bench TinyLlama-1.1B Q4 (GPU) | 19,082 / 463 t/s | — |
| llama-bench TinyLlama-1.1B Q4 (CPU) | 410 / 27 t/s | — |

Full tables (incl. CPU, latency percentiles, per-batch curves) in [docs/results.md](docs/results.md).

## Hardware covered

- **RTX 5070 Ti Laptop** (sm_120, 12.8 GB) — local, in the `mlperf` WSL distro
- **Colab T4** (sm_75, 16 GB) — headless via `google-colab-cli`
- **CPU** — Intel Core Ultra 9 275HX (24 threads), and any work machine
- **A100 / H200** — pending (run `microbench/` or `tensorrt/` directly and paste results)

## License / provenance

Wraps [`mlcommons/inference`](https://github.com/mlcommons/inference) (Apache-2.0). Models/datasets
are downloaded from their original sources (Zenodo, OpenSLR, Hugging Face, fast.ai) at run time.
The suite's own scripts and docs are provided as-is.
