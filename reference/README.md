# reference/ — MLPerf Inference reference implementations

The **official MLCommons** reference harness (LoadGen + reference PyTorch backends) for three
domains. Unoptimized by design — they define the benchmarks and produce real accuracy numbers, not
hardware records (for optimized hardware numbers see [`../tensorrt/`](../tensorrt)).

| Domain | Notebook (local) | Metric |
|---|---|---|
| NLP | `local/mlperf_bert_squad_local.ipynb` | f1 ≈ 90.4 |
| Vision | `local/mlperf_resnet50_local.ipynb` | top-1 84.5% / 75.4% |
| Speech | `local/mlperf_whisper_local.ipynb` | WER ≈ 3.5–5% |

- **`local/`** — Jupyter notebooks that run in the `mlperf` WSL distro (local paths, `%pip` cells,
  venv-activated bash, Blackwell math-SDP guard, all harness fixes baked in).
- **`colab/`** — the same benchmarks as Colab notebooks, runnable headless via `google-colab-cli`.
  `*_output.ipynb` are executed copies with T4 outputs baked in.

Running instructions: [../docs/user-guide.md](../docs/user-guide.md#3-mlperf-reference-implementations-bert--resnet-50--whisper).
Fixes explained: [../docs/gotchas.md](../docs/gotchas.md).
