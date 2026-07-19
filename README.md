# inference-bench

A self-contained suite for running **ML inference benchmarks** and **GPU/CPU hardware
benchmarks** across heterogeneous hardware — a laptop **RTX 5070 Ti** (Blackwell, sm_120), a
**Colab T4** (Turing, sm_75), datacenter GPUs (**A100/H200**), and CPUs — with reproducible
scripts, notebooks, and results.

> ## ⚠️ These are NOT official MLPerf results
>
> This is an **unofficial, MLPerf-*inspired* smoke-test suite** for quick cross-hardware
> comparison — **not** a conformant [MLPerf Inference](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
> submission. Specifically:
> - Runs use **short, non-conformant configs** (10–60 s, reduced query counts) instead of MLPerf's
>   ~600 s minimums, and **subset datasets** (Imagenette / a 5,000-image ImageNet mirror / 1,000
>   SQuAD examples) instead of the full validation sets.
> - **Whisper does not use LoadGen at all** — it's a custom sequential loop (same model + WER metric).
> - A LoadGen "**VALID**" line here means the run met *its own short config's* timing — it is **not**
>   a conformant MLPerf VALID result.
>
> **Do not report these numbers as MLPerf results or use them for hardware-procurement decisions
> under the MLPerf label.** Treat them as "does my GPU roughly do what I expect" checks. Full detail:
> [docs/architecture.md](docs/architecture.md#what-is-and-isnt-mlperf).

It contains four kinds of thing, kept clearly separate:

| | What | Framework |
|---|---|---|
| **`reference/`** | MLPerf-*inspired* reference runs (BERT, ResNet-50, Whisper) | MLCommons LoadGen for **BERT + ResNet** (short configs); **Whisper = custom loop, no LoadGen** |
| **`tensorrt/`** | ResNet-50 with an optimized **TensorRT** backend | MLCommons LoadGen + custom SUT (short, non-conformant config) |
| **`microbench/`** | Raw GPU/CPU microbenchmarks (TFLOPS, bandwidth, throughput) | **Custom** (not MLPerf) |
| **`standards/`** | Other standards: Polygraphy/trtexec, llama.cpp, AI-Benchmark, MLPerf Client | Third-party |

## Quick start

Everything runs in the isolated `mlperf` WSL distro (setup in [docs/setup.md](docs/setup.md)):

```bash
wsl -d mlperf && source /root/mlperf/venv/bin/activate

# 1. Portable hardware microbench (fastest, any GPU)
python microbench/gpu_bench.py            # GPU: TFLOPS, bandwidth, ResNet-50 throughput
python microbench/cpu_bench.py            # CPU-only version

# 2. LoadGen + TensorRT (MLPerf-inspired, non-conformant smoke test)
bash tensorrt/trt_mlperf_run.sh           # ResNet-50: SingleStream + Offline + accuracy

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
├── HANDOFF.md                    ← hand-off checklist to run on a work A100/H200 (native Linux)
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

Point-in-time smoke-test numbers (see the disclaimer above — **not** official MLPerf); laptop
figures vary ±10% run-to-run. Provenance and caveats: [docs/results.md](docs/results.md).

| Benchmark | RTX 5070 Ti | Colab T4 |
|---|---|---|
| LoadGen+TensorRT ResNet-50 — Offline | 3,652 img/s (VALID) | 1,200 img/s |
| LoadGen+TensorRT ResNet-50 — SingleStream p90 | 2.39 ms (VALID) | 2.80 ms |
| BERT/SQuAD (LoadGen, 1k subset) — f1 | 90.40 | 90.40 |
| ResNet-50 (LoadGen, subset) — top-1 | 75.4% / 84.5% | 84.6% |
| Whisper (custom loop, no LoadGen) — WER | ~3.5–5% | 2.16% |
| microbench ResNet-50 fp16 (TensorRT) | 4,774 img/s | 1,945 img/s |
| microbench FP16 / BF16 TFLOPS | 42.7 / 51.4 | 22.7 / 2.1 |
| Polygraphy (trtexec) ResNet-50 fp16 bs128 | ~3,965 img/s | — |
| llama-bench TinyLlama-1.1B Q4 (GPU) | 20,842 / 434 t/s | — |
| llama-bench TinyLlama-1.1B Q4 (CPU) | 410 / 27 t/s | — |

Full tables (incl. CPU, latency percentiles, per-batch curves) in [docs/results.md](docs/results.md).

## Hardware covered

- **RTX 5070 Ti Laptop** (sm_120, 12.8 GB) — local, in the `mlperf` WSL distro
- **Colab T4** (sm_75, 16 GB) — headless via `google-colab-cli`
- **CPU** — Intel Core Ultra 9 275HX (24 threads), and any work machine
- **A100 / H200** — pending; the scripts are portable to a native-Linux box — hand off with
  [HANDOFF.md](HANDOFF.md) and paste the results

## License / provenance

This suite is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

It wraps [`mlcommons/inference`](https://github.com/mlcommons/inference) (Apache-2.0), cloned at run
time rather than redistributed. Models/datasets are downloaded from their original sources (Zenodo,
OpenSLR, Hugging Face, fast.ai) at run time under their own licenses.
