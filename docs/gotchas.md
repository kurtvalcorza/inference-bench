# Gotchas & Lessons

Every non-obvious fix discovered while building this suite. Grouped by area.

## Blackwell (RTX 50-series, sm_120)

- **Fused SDPA attention crashes — and can BSOD the host.** transformers' default fused
  scaled-dot-product-attention kernel is invalid on sm_120 → `CUDA error: illegal instruction`
  mid-inference; a repeat hit the same path and bugchecked Windows (a GPU fault in WSL can crash
  the host). **Fix:** force the math SDP backend before any model runs:
  ```python
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_mem_efficient_sdp(False)
  torch.backends.cuda.enable_math_sdp(True)
  ```
  Needed for **BERT** and **Whisper** (attention models). Not needed for ResNet (conv-only). Plain
  cuBLAS matmul works on sm_120 even without this — only fused attention is broken.
- **Requires torch cu128** (`--index-url https://download.pytorch.org/whl/cu128`); older CUDA has no
  sm_120 kernels.

## TensorRT (11.x)

- **`BuilderFlag.FP16` no longer exists.** TRT ≥10 uses **strongly-typed networks** — precision is
  inferred from the ONNX dtypes. Export an **fp16** ONNX and build with
  `create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))`, no precision flag.
- **Execution contexts are NOT thread-safe.** MLPerf's Offline QueueRunner calls the SUT from ~24
  threads → segfault. Serialize with a `threading.Lock` around the predict body.
- **First-query latency outlier.** Lazy TRT/CUDA init makes the first query huge (skews p99, even a
  negative-min artifact). Add a warmup inference in `load()`.
- **Modern runtime API:** `context.set_tensor_address(name, tensor.data_ptr())` +
  `context.execute_async_v3(stream)` with `torch.cuda.current_stream().cuda_stream`; use torch
  tensors as I/O buffers (no pycuda).
- `pip install tensorrt onnx` (TRT 11.1) does **not** disturb torch 2.11+cu128, and supports
  sm_75/80/90/120.

## MLPerf vision harness (`vision/classification_and_detection`)

- **`main.py` imports `coco` unconditionally** → needs `pycocotools` even for ResNet.
- **`--user_conf` must be an absolute path** (default looks in cwd = `python/`, not the repo root).
- **Accuracy mode crashes in post-test percentile stats** (numpy 2.x on an empty latency array),
  *after* LoadGen writes the accuracy log. Tolerate it (`|| true`) and score externally:
  `python ../tools/accuracy-imagenet.py --mlperf-accuracy-file … --imagenet-val-file val_map.txt --dtype float32`.
- **`backend_pytorch_native` returns a bare GPU tensor** → the post-process (`np.argmax(results[0])`)
  throws, gets caught, and writes an **empty** accuracy log. Patch it to
  `return [output.cpu().numpy()]` (a list whose `[0]` is the numpy batch, like the onnx backend).
- **Class-index offset:** use the `imagenet_pytorch` dataset (`PostProcessArgMax(offset=0)`) with a
  torchvision (1000-class, 0-indexed) model + torchvision-index labels — self-consistent. The
  TF/onnx path uses a 1001-class model with `offset=-1`; don't mix them.
- **Offline perf over-issues** (query = `target_qps × min_duration`); a naive CPU run took ~3 h and
  wrote a 550 MB trace. Bound it via user.conf (`target_qps`, `min_duration`, `min_query_count`).
- **VALID tuning:** a run must exceed `min_duration` (10 s). SingleStream → raise `min_query_count`
  (≈4000 at ~390 QPS); Offline → raise `target_qps` so the coalesced query runs > 10 s.

## MLPerf BERT harness

- **`tokenization` is missing.** It lives in the multi-GB NVIDIA `DeepLearningExamples` submodule;
  google's `tokenization.py` imports TensorFlow, NVIDIA's imports `file_utils`. Drop in a
  **minimal self-contained** `tokenization.py` (`convert_to_unicode`, `printable_text`,
  `whitespace_tokenize`, `BasicTokenizer` — `unicodedata` only).
- **`run.py --accuracy` auto-scores with wrong default paths** (harmless non-zero exit); the
  accuracy log is written regardless → score explicitly with `accuracy-squad.py`.

## Whisper

- **The master reference SUT uses vLLM** (heavy, Blackwell-risky). We run whisper-large-v3 via
  **`openai-whisper`** instead — same model, dataset, and WER metric, simpler and reliable.
- openai-whisper also uses fused SDPA → **apply the Blackwell math-SDP guard**.
- WER varies run-to-run (~3.5–5% on a 100-utt subset) — temperature-fallback decoding has no fixed
  seed; it stabilizes near ~2–3% on the full dev set.

## Datasets / Hugging Face

- **ImageNet-1k val is access-gated.** The ungated mirror `Tsomaros/Imagenet-1k_validation` has the
  standard 1000-class labels and needs no token. It's **class-sorted (50/class)**, so keep every
  10th row for a balanced 5/class × 1000-class subset. Extract raw jpg bytes from the parquet
  `image` struct (no re-decode).
- **`pip install -U huggingface_hub` bumps it to 1.x, which breaks transformers 4.48 / tokenizers**
  (they want `<1.0`). Pin back: `pip install "huggingface_hub>=0.24,<1.0"`.
- torchvision resnet50 `.pth` (Zenodo) is a **legacy tar-format** state_dict → load with
  `torch.load(..., weights_only=False)`.

## Environment / tooling

- **`wsl.exe` mangles `$vars` in `bash -c`** (they expand to empty). Put commands in **script files**
  staged on `/mnt/c`, not inline `$VAR`.
- **Git Bash rewrites `/mnt/c/...` paths** when invoking `wsl` (`C:/Program Files/Git/mnt/c/...`).
  Invoke `wsl` from **PowerShell** for WSL paths.
- **Jupyter as root** needs `--allow-root` or it exits. Open the printed `127.0.0.1:8888` URL in the
  Windows browser (WSL2 forwards localhost).
- **`colab` CLI isn't on PATH in a non-login shell** → `export PATH="/home/kurt/.local/bin:$PATH"`.
- **`colab exec` has a per-cell reply TimeoutError** on long installs/runs (e.g. `tensorrt` ~1 GB) —
  but the work usually completes on the VM anyway. For long jobs: **launch detached**
  (`subprocess.Popen("bash setup.sh > log 2>&1 &")`) and **poll a results file**; use
  `colab restart-kernel` to clear a stuck kernel.
- **Verify the real exit code, not a pipeline's:** `python … | grep | tail` reports `tail`'s exit
  (always 0). Capture `${PIPESTATUS[0]}` when gating on success.
