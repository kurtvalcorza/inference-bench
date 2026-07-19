# Delegation checklist — run these benchmarks on a work GPU (A100 / H200)

A stand-alone guide for someone running this suite on a **native-Linux datacenter box**, with no
prior context on the repo. You need: an NVIDIA GPU, a recent driver, git, python3, and internet.
Everything else is downloaded or built by the scripts.

The scripts were originally wired to the author's laptop WSL distro; they are now path-portable and
honour the env vars below. **Defaults preserve the laptop**, so on a work box set `BENCH_ROOT`.

---

## 0. One-time setup (~5 min + downloads)

```bash
git clone <this-repo-url> inference-bench && cd inference-bench

# check the GPU + note its compute capability (A100=8.0, H200=9.0)
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

# python env with CUDA 12.8 torch (works on A100/H200/Blackwell)
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# tell the scripts where to stage assets (anything writable; NOT /root unless you are root)
export BENCH_ROOT="$HOME/inference-bench-data"
# BENCH_VENV is left unset => scripts use THIS active venv.
```

> If `nvidia-smi` fails, stop — the driver/container isn't exposing the GPU; sort that with the box's
> admin first. Nothing here works without it.

---

## 1. Microbenchmarks — always run these (fast, fully portable)

```bash
python microbench/gpu_bench.py       # TFLOPS (fp32/tf32/fp16/bf16), bandwidth, ResNet-50 throughput
python microbench/cpu_bench.py       # CPU: GFLOPS, bandwidth, ResNet-50 img/s
```

Big GPUs need a larger sweep to reach peak (otherwise they're understated):

```bash
MATMUL_SIZES=16384,24576,32768 BATCHES=256,512,1024,2048 python microbench/gpu_bench.py
```

**Deliverable:** copy the JSON each script prints at the end (one per machine). That alone gives a
solid A100/H200-vs-5070Ti/T4 comparison.

---

## 2. MLPerf ResNet-50 + TensorRT — the "official MLPerf" number (VALID LoadGen result)

```bash
pip install tensorrt onnx                    # (the script also does this if missing)
bash tensorrt/trt_mlperf_run.sh              # ~10-15 min; MAXBS=256 bash ... to raise batch size
```

This self-bootstraps: clones `mlcommons/inference`, patches in the TensorRT backend, exports the
fp16 ONNX (auto-downloads ResNet-50 weights), builds the representative ImageNet subset from an
ungated HF mirror, then runs **SingleStream** (latency), **Offline** (throughput), and **accuracy**.

**Deliverable:** the three `mlperf_log_summary.txt` blocks it prints (SingleStream p50/p90/p99,
Offline throughput, and the top-1 accuracy line). Note the run's `MAXBS`.

> If the HF subset download is blocked on the work network, point `DATA` at any ImageNet val dir that
> has a `val_map.txt` (`<file.JPEG> <class_idx>` per line) and re-run:
> `DATA=/path/to/val bash tensorrt/trt_mlperf_run.sh`

---

## 3. Standards — optional but valuable on a datacenter GPU

```bash
# TensorRT profiler (trtexec equivalent) — pure GPU ceiling for ResNet-50
bash standards/polygraphy_resnet.sh                  # BS=256 bash ... to change batch

# LLM token throughput — TinyLlama-1.1B (prefill + decode)
bash standards/llama_bench.sh                        # CUDA auto-detects arch (A100=80, H200=90)
MODE=cpu bash standards/llama_bench.sh               # CPU-only comparison
```

`llama_bench.sh` (CUDA) needs the **CUDA toolkit / nvcc** on the box. If `nvcc` is missing it prints
`nvcc=MISSING` and the build fails — install the toolkit (see [docs/setup.md](docs/setup.md) §6) or
skip to `MODE=cpu`. `CUDA_ARCH=native` auto-targets the GPU; override with e.g. `CUDA_ARCH=90`.

**Deliverable:** polygraphy's "Average inference time" (→ throughput) and llama-bench's pp512/tg128
tokens/s table.

---

## 4. Send back

Paste into a reply or a file, per machine:

1. `nvidia-smi` name + compute_cap.
2. Both microbench JSON blobs (§1).
3. The three MLPerf-TRT summary blocks + `MAXBS` (§2).
4. (If run) polygraphy + llama-bench numbers (§3).

The author drops these into the `A100` / `H200` columns of [docs/results.md](docs/results.md) (which
already have placeholders) — no further interpretation needed.

## Troubleshooting quick-ref

| Symptom | Fix |
|---|---|
| `no CUDA GPU` / `nvidia-smi` fails | driver/container not exposing the GPU — box admin |
| `no venv at /root/mlperf/venv` (info line) | expected off-laptop — it's using your active venv, carry on |
| HF ImageNet download blocked | `DATA=/path/to/val_with_val_map bash tensorrt/trt_mlperf_run.sh` |
| `nvcc=MISSING` in llama_bench | install CUDA toolkit (setup.md §6) or use `MODE=cpu` |
| ResNet numbers look low on A100/H200 | use the larger `MATMUL_SIZES`/`BATCHES` sweep (§1) |

Full detail for every step: [docs/setup.md](docs/setup.md) and [docs/user-guide.md](docs/user-guide.md).
