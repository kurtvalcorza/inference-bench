# User Guide

How to run each benchmark. All commands assume the `mlperf` WSL distro with the venv active
(`wsl -d mlperf && source /root/mlperf/venv/bin/activate`) unless noted. Setup: [setup.md](setup.md).

---

## 0. Recording a citable result (recommended)

Any number worth keeping should be produced inside a **run bundle** — a self-contained (not
cryptographically immutable) record of the command, repo commit + working-tree diff, the env knobs
the runners read, `pip freeze`, GPU/driver, asset SHA-256s (including a root hash over every val
image), full logs, and the real exit status:

```bash
bash scripts/run_bundle.sh trt-5070ti -- bash tensorrt/trt_mlperf_run.sh
bash scripts/run_bundle.sh micro-a100 -- python microbench/gpu_bench.py
```

Bundles land in `results/bundles/<UTC>-<label>/` (gitignored). That directory — not the
[results.md](results.md) tables — is what you cite or hand back. See [results/README.md](../results/README.md).

## 1. Microbenchmarks (fastest — start here)

Custom, portable, not MLPerf. Auto-detect the device; print a table + JSON.

```bash
python microbench/gpu_bench.py        # GPU: TFLOPS, bandwidth, ResNet-50 eager/compile/TensorRT
python microbench/cpu_bench.py        # CPU: GFLOPS, bandwidth, ResNet-50 throughput
```

Tuning for big GPUs (A100/H200) — sweep larger sizes/batches so they aren't understated:

```bash
MATMUL_SIZES=16384,24576,32768 BATCHES=256,512,1024,2048 python microbench/gpu_bench.py
```

Each tier degrades gracefully (a missing `tensorrt` just shows `skipped`). Paste the JSON to
compare machines. Runs on any CUDA GPU / any CPU.

---

## 2. LoadGen + TensorRT ResNet-50 (MLPerf-inspired, non-conformant)

LoadGen harness driving an optimized TensorRT fp16 SUT → SingleStream/Offline/accuracy. A LoadGen
"VALID" line means the run met *this suite's short config* (10 s), **not** conformant MLPerf — see
[architecture.md](architecture.md#what-is-and-isnt-mlperf).

```bash
bash tensorrt/trt_mlperf_run.sh                 # default max-batchsize 128
MAXBS=32 bash tensorrt/trt_mlperf_run.sh        # override batch size
```

What it does: copies `backend_tensorrt.py` into the harness, patches `main.py`'s `get_backend()`
to add a `tensorrt` branch, exports a dynamic-batch fp16 ONNX, then runs **SingleStream**
(latency), **Offline** (throughput), and an **accuracy** pass. Uses the representative ImageNet
subset at `/root/mlperf/vision/inet_val` (build it per §5 first, or point `DATA` at Imagenette).

Prereqs: `pip install tensorrt onnx`, the resnet50 `.pth`, and a val set with `val_map.txt`.

---

## 3. MLPerf reference implementations (BERT / ResNet-50 / Whisper)

The unoptimized reference harness. Two ways to run: **local notebooks** or **Colab**.

### Local (Jupyter in the WSL distro)

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name mlperf --display-name "mlperf (venv)"
jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root
```

Open the printed `127.0.0.1:8888` URL in your Windows browser, open one of:

- `reference/local/mlperf_bert_squad_local.ipynb` → f1 ≈ 90.4
- `reference/local/mlperf_resnet50_local.ipynb` → top-1 84.5% (Imagenette) or 75.4% (representative)
- `reference/local/mlperf_whisper_local.ipynb` → WER ≈ 3.5–5%

Pick the **mlperf (venv)** kernel → **Run All**. Assets cache under `/root/mlperf`, so re-runs skip
downloads. Each notebook contains the Blackwell math-SDP guard and all harness fixes.

### Running on Colab (headless)

Via `google-colab-cli` (one-time OAuth done once):

```bash
wsl -d colab
colab new  -s bench --gpu T4
colab exec -s bench -f reference/colab/mlperf_resnet50_colab.ipynb
colab log  -s bench -o exec_log.md
colab stop -s bench
```

**Timeout-resilient pattern** (for long installs/runs — the CLI has a per-cell reply timeout):
upload a self-contained script and launch it *detached* on the VM, then poll a results file:

```bash
colab upload -s bench setup.sh /content/setup.sh
echo 'import subprocess; subprocess.Popen("bash /content/setup.sh > /content/run.log 2>&1 &", shell=True)' | colab exec -s bench
# ...later:
echo 'print(open("/content/results.txt").read())' | colab exec -s bench
```

`*_output.ipynb` files in `reference/colab/` are executed copies with the T4 outputs baked in.

---

## 4. CPU benchmark on work machines

`microbench/cpu_bench.py` needs only `torch torchvision` (CPU build — no CUDA):

```bash
pip install torch torchvision
python microbench/cpu_bench.py            # prints CPU model, cores, GFLOPS, bandwidth, ResNet img/s
```

Runs on Windows / Linux / Mac. Paste each machine's JSON to compare.

---

## 5. Representative ImageNet (optional, for a real ~76% top-1)

The full ImageNet val is gated; use the ungated HF mirror to build a balanced 5/class subset (all
1000 classes) — no token needed. **Use the validated builder** (it writes to a temp dir, checks the
5×1000 balance and that every image is present, then atomically swaps into place — so a failed
download never leaves a poisoned `val_map.txt`). Don't hand-roll a direct-write loop:

```bash
pip install "huggingface_hub>=0.24,<1.0" pyarrow      # hub 1.x breaks transformers 4.48
BENCH_ROOT=/root/mlperf python tensorrt/build_imagenet_subset.py /root/mlperf/vision/inet_val
```

`trt_mlperf_run.sh` calls this automatically when `DATA` has no `val_map.txt`, and validates any
pre-existing dataset before use.

Then point any ResNet-50 run's `--dataset-path` at `/root/mlperf/vision/inet_val`.

---

## 6. Other benchmark standards (`standards/`)

```bash
# NVIDIA TensorRT profiler (pip trtexec equivalent) — ResNet-50 latency/throughput
bash standards/polygraphy_resnet.sh                 # BS=256 bash ... to change batch

# LLM token throughput (prefill + decode), TinyLlama-1.1B
MODE=cuda bash standards/llama_bench.sh             # GPU (needs the CUDA toolkit — see setup.md)
MODE=cpu  bash standards/llama_bench.sh             # CPU only, any machine
```

- **AI-Benchmark** (`standards/ai_benchmark.md`) — TensorFlow "AI Score"; run on a **T4 or CPU**, not
  the 5070 Ti (TF has no Blackwell support, and it conflicts with the torch/tensorrt venv).
- **MLPerf Client** (`standards/mlperf_client.md`) — the official consumer-PC LLM benchmark; a native
  **Windows** app (ONNX Runtime GenAI + CUDA/DirectML). For a scriptable LLM number here, use
  `llama_bench.sh` instead.

To build llama.cpp with CUDA you need the CUDA toolkit (nvcc):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 cuda-nvrtc-dev-12-8
```

## Which benchmark should I run?

| Goal | Run |
|---|---|
| Quick cross-hardware compare | `microbench/gpu_bench.py` |
| Compare CPUs (work machines) | `microbench/cpu_bench.py` |
| A LoadGen+TensorRT hardware number (MLPerf-inspired, not conformant) | `tensorrt/trt_mlperf_run.sh` |
| MLPerf accuracy + reference behavior | `reference/local/*.ipynb` |
| Run on a cloud GPU without a local one | `reference/colab/*.ipynb` via colab CLI |
