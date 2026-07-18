# AI-Benchmark (ETH Zürich)

A Python package that runs ~19 CV/NLP models and produces a single comparable **AI Score**
(inference + training sub-scores). Good for one portable number across machines.

## Run

```bash
pip install ai_benchmark tensorflow          # use a SEPARATE venv — see caveat
python -c "from ai_benchmark import AIBenchmark; AIBenchmark().run()"
```

## ⚠️ Hardware caveat — do NOT run this on the RTX 5070 Ti (Blackwell)

AI-Benchmark is **TensorFlow-based**, and TensorFlow has **no sm_120 (Blackwell) GPU support** — it
won't use the 5070 Ti's GPU (it would fall back to CPU, or fail). It also drags in TensorFlow's own
CUDA/cuDNN stack, which **conflicts with the torch cu128 + TensorRT** environment this suite relies
on — so install it in a **separate venv**, never the `mlperf` venv.

Where it *does* work well:
- **Colab T4 / older GPUs** (sm_75/80/86) — TF-GPU is supported there. Run it in a fresh Colab
  runtime for a clean AI Score.
- **CPU** — works anywhere but is slow (~10+ min) and not a GPU comparison.

## Status in this suite

Documented, not run on the 5070 Ti (Blackwell/TF incompatibility). Best used on the T4 or a
CUDA-11/older-GPU box. If you want, run it on the Colab T4 via the `google-colab-cli` for a
comparable AI Score alongside the other numbers.
