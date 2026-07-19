# reference/ — MLPerf-inspired reference runs

> **Not conformant MLPerf** (short configs, subset datasets; see
> [top-level README](../README.md) and [architecture.md](../docs/architecture.md#what-is-and-isnt-mlperf)).

Reference-style PyTorch runs for three domains. **BERT and ResNet-50** drive the real MLCommons
LoadGen harness (but with short, non-conformant configs on subset data). **Whisper does *not* use
LoadGen** — it's a custom sequential loop over ~30–100 files with the same model and WER metric
(the MLCommons Whisper reference SUT uses vLLM, which we avoided on Blackwell). Unoptimized by design;
for optimized hardware numbers see [`../tensorrt/`](../tensorrt).

| Domain | Notebook (local) | Harness | Metric |
|---|---|---|---|
| NLP | `local/mlperf_bert_squad_local.ipynb` | LoadGen (1k-example subset) | f1 ≈ 90.4 |
| Vision | `local/mlperf_resnet50_local.ipynb` | LoadGen (subset) | top-1 84.5% / 75.4% |
| Speech | `local/mlperf_whisper_local.ipynb` | **custom loop, no LoadGen** | WER ≈ 3.5–5% |

- **`local/`** — Jupyter notebooks that run in the `mlperf` WSL distro (local paths, `%pip` cells,
  venv-activated bash, Blackwell math-SDP guard, all harness fixes baked in).
- **`colab/`** — the same benchmarks as Colab notebooks, runnable headless via `google-colab-cli`.
  `*_output.ipynb` are executed copies with T4 outputs baked in.

Running instructions: [../docs/user-guide.md](../docs/user-guide.md#3-mlperf-reference-implementations-bert--resnet-50--whisper).
Fixes explained: [../docs/gotchas.md](../docs/gotchas.md).

> **⚠️ Supply-chain caveat — these notebooks are NOT hardened.** Unlike the pinned/verified scripts in
> [`../tensorrt/`](../tensorrt) and [`../standards/`](../standards), the reference notebooks still
> clone upstream at a floating revision, `pip install` unpinned packages, download models/archives
> without checksum verification, and load the ResNet checkpoint with `torch.load(weights_only=False)`
> (which can execute arbitrary code from a tampered file). They also commonly run **as root** in the
> WSL distro. Only run them against sources you trust. For a reproducible, checksum-verified path use
> `tensorrt/trt_mlperf_run.sh` (pinned harness `INFERENCE_REF`, `weights_only=True` exporter) instead.
> Hardening the notebooks themselves is tracked as future work.
