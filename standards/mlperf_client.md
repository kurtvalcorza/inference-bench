# MLPerf Client

The MLCommons benchmark for **LLM inference on consumer PCs** (not the datacenter MLPerf
Inference). Measures things like time-to-first-token and tokens/sec for a chat LLM (e.g. Llama-2-7B)
on a client GPU/NPU/CPU — designed for exactly the class of hardware the RTX 5070 Ti is.

## What it is

A standalone application (GUI + CLI) that runs an LLM through a pluggable **execution provider**:
- **ONNX Runtime GenAI** with DirectML (any DX12 GPU), CUDA, or the Intel/AMD/Qualcomm backends
- vendor paths (NVIDIA TensorRT-LLM, etc.)

It is primarily a **Windows** app; download the release from
<https://mlcommons.org/benchmarks/client/> (or the `mlcommons/mlperf_client` GitHub), pick an
execution provider that matches your GPU, and run the benchmark from the app or its CLI.

## ⚠️ Why it isn't run headless in this suite

- It's a **Windows consumer application**, not a Linux/WSL Python tool — it doesn't fit the
  `mlperf` WSL distro's headless flow the other benchmarks use.
- Blackwell (RTX 50-series) support depends on the execution provider's build; the **CUDA / DirectML
  path** is the one to use on the 5070 Ti (run it natively on Windows, not in WSL).

## How to run it (native Windows, the right way for the 5070 Ti)

1. Download the MLPerf Client release for Windows.
2. Choose the execution provider: **ONNX Runtime GenAI + CUDA** (or DirectML) for the 5070 Ti.
3. Run the default LLM workload → it reports TTFT + tokens/sec.

## Status in this suite

Documented. It's the *most official* consumer-LLM benchmark for the 5070 Ti, but it runs as a native
Windows app rather than in this suite's WSL/Colab pipeline. For a scriptable LLM number here, use
[`llama_bench.sh`](llama_bench.sh) instead (same tokens/sec metric, CPU + CUDA, fully headless).
