# Setup

Everything here targets an **isolated Ubuntu 24.04 WSL2 distro** named `mlperf`, with an NVIDIA GPU
passed through. This keeps the toolchain separate from the rest of the machine.

## 1. Create the `mlperf` WSL distro

`wsl --install <name>` times out via WinINET on some machines, so import a rootfs directly:

```powershell
# download the Ubuntu 24.04 WSL rootfs
curl.exe -L -o "$env:USERPROFILE\WSL\mlperf\rootfs.tar.gz" `
  "https://cloud-images.ubuntu.com/wsl/releases/noble/current/ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz"

# import it
wsl --import mlperf "$env:USERPROFILE\WSL\mlperf" "$env:USERPROFILE\WSL\mlperf\rootfs.tar.gz" --version 2
```

GPU passthrough works out of the box — verify:

```bash
wsl -d mlperf
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ls -l /dev/dxg            # present => WSL GPU access
```

### Native Linux instead (datacenter A100/H200 / work box) — skip WSL

On a real Linux box (the likely home of an A100/H200) there is **no WSL step** — Section 1 above is
Windows-only. Just verify the driver and go straight to Section 2:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
```

The scripts don't assume WSL or `root` — they honour these env vars (defaults preserve the laptop's
`/root/mlperf` layout, so nothing changes there):

| Var | Purpose | Default |
|---|---|---|
| `BENCH_VENV` | venv to activate | `/root/mlperf/venv` (skipped if absent → uses current env) |
| `BENCH_ROOT` | asset/data/build root | `/root/mlperf` if it exists, else `$HOME/inference-bench-data` |
| `INFERENCE_REPO` | mlcommons/inference clone | `$BENCH_ROOT/inference` (auto-cloned) |
| `CUDA_ARCH` | llama.cpp CUDA arch | `native` (auto: A100=80, H200=90) |

So a delegate typically runs, from the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
export BENCH_ROOT="$HOME/inference-bench-data"     # BENCH_VENV unset => uses this active venv
python microbench/gpu_bench.py                      # portable, no other setup
bash   tensorrt/trt_mlperf_run.sh                   # self-bootstraps harness + ONNX + data subset
```

See [HANDOFF.md](../HANDOFF.md) for the full hand-off checklist.

## 2. Base toolchain + Python venv

```bash
apt-get update && apt-get install -y python3 python3-venv python3-pip build-essential cmake git wget ffmpeg
python3 -m venv /root/mlperf/venv
source /root/mlperf/venv/bin/activate
```

### PyTorch (CUDA 12.8 — required for Blackwell / RTX 50-series, works everywhere else)

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# -> 2.11.0+cu128 True (12, 0)     # sm_120 = Blackwell
```

## 3. Clone the MLPerf inference repo + build LoadGen

```bash
git clone --depth 1 https://github.com/mlcommons/inference.git /root/mlperf/inference
pip install mlcommons-loadgen          # PyPI wheel — no C++ build needed
# (or build from source: cd /root/mlperf/inference/loadgen && pip install .)
python -c "import mlperf_loadgen; print('loadgen ok')"
```

## 4. Per-benchmark dependencies

| Benchmark | Extra deps |
|---|---|
| BERT / SQuAD | `transformers==4.48.3` |
| ResNet-50 / ImageNet | `torchvision opencv-python-headless pycocotools` |
| Whisper / LibriSpeech | `openai-whisper jiwer soundfile` + apt `ffmpeg` |
| TensorRT backend | `tensorrt onnx` (TRT 11.x; supports sm_75/80/90/120) |
| microbench (full) | `torchvision`, and `tensorrt onnx` for the TRT tier |

Datasets/models download at run time (Zenodo, OpenSLR, fast.ai Imagenette, Hugging Face).

## 5. (Optional) Colab CLI — for headless Colab GPU runs

The `google-colab-cli` runs Colab GPUs from the terminal. Install it in its own distro/venv:

```bash
uv tool install google-colab-cli        # or: pip install google-colab-cli
colab new -s bench --gpu T4             # first run triggers a one-time OAuth sign-in
```

Then `colab exec -s bench -f notebook.ipynb`, `colab install`, `colab upload`, `colab stop`.
See [user-guide.md](user-guide.md#running-on-colab-headless) for the full flow (and the
timeout-resilient "launch detached + poll a results file" pattern).

## 6. (Optional) Jupyter, for the reference notebooks

```bash
source /root/mlperf/venv/bin/activate
pip install jupyterlab ipykernel
python -m ipykernel install --user --name mlperf --display-name "mlperf (venv)"
jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root   # --allow-root: distro runs as root
```

Open the printed `http://127.0.0.1:8888/...` URL in your Windows browser (WSL2 forwards localhost),
open a notebook, select the **mlperf (venv)** kernel, Run All.

## 7. (Optional) Hugging Face — representative ImageNet

The full ImageNet-1k val set is access-gated, but an ungated mirror
(`Tsomaros/Imagenet-1k_validation`, standard 1000-class labels) works with no token:

```bash
pip install "huggingface_hub>=0.24,<1.0" pyarrow   # NB: hub 1.x breaks transformers 4.48; pin <1.0
```

See [user-guide.md](user-guide.md#representative-imagenet) for building the balanced subset.
