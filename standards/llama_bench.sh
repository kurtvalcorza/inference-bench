#!/usr/bin/env bash
# llama.cpp `llama-bench` — LLM token throughput: prefill (pp512) + decode (tg128).
# MODE=cuda (default) or MODE=cpu.  Model: TinyLlama-1.1B Q4_K_M (ungated).
#
# Env: BENCH_ROOT (build/model dir), CUDA_ARCH (default "native" -> auto-detects the
#      GPU's compute capability at build time: A100=80, H200=90, Blackwell=120),
#      CUDA_HOME (default /usr/local/cuda).
set -uo pipefail

MODE=${MODE:-cuda}

if [ -z "${BENCH_ROOT:-}" ]; then
  if [ -d /root/mlperf ]; then BENCH_ROOT=/root/mlperf; else BENCH_ROOT="$HOME/inference-bench-data"; fi
fi
LLM="$BENCH_ROOT/llm"
mkdir -p "$LLM"; cd "$LLM"
LLAMA_REF="${LLAMA_REF:-}"           # set to a commit/tag to pin; empty = latest (resolved sha printed below)
if [ ! -d llama.cpp ]; then
  if [ -n "$LLAMA_REF" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git
    git -C llama.cpp checkout -q "$LLAMA_REF" || { echo "!! could not checkout llama.cpp @ $LLAMA_REF"; exit 1; }
  else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
  fi
fi
echo "llama.cpp commit: $(git -C llama.cpp rev-parse --short HEAD) (pin: ${LLAMA_REF:-none — set LLAMA_REF to pin})"
cd llama.cpp

if [ "$MODE" = cuda ]; then
  # needs the CUDA toolkit (nvcc). NVIDIA apt pkgs: cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 cuda-nvrtc-dev-12-8
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  [ -d "$CUDA_HOME/bin" ] && export PATH="$CUDA_HOME/bin:$PATH" CUDACXX="$CUDA_HOME/bin/nvcc"
  # Detect the GPU's compute capability (A100->80, H200->90, Blackwell->120) so the build works on
  # any CMake version. Only fall back to "native" (needs CMake>=3.24, absent on Ubuntu 22.04's 3.22)
  # if detection fails. Override explicitly with CUDA_ARCH=80,90,...
  if [ -n "${CUDA_ARCH:-}" ]; then ARCH="$CUDA_ARCH"
  else
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    ARCH="${CC:-native}"
  fi
  BUILD=build-cuda; EXTRA="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH"; NGL="-ngl 99"
  echo "CUDA build: arch=$ARCH  nvcc=$(command -v nvcc || echo MISSING)"
else
  BUILD=build-cpu; EXTRA="-DGGML_NATIVE=ON"; NGL=""
fi

# server OFF (avoids the memory-heavy cpp-httplib) and -j4 (nvcc/httplib OOM at -j24)
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release $EXTRA \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j"${JOBS:-4}" --target llama-bench

cd "$LLM"
[ -s tinyllama.gguf ] || wget -q -O tinyllama.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# Verify the model hash. Set GGUF_SHA256 to enforce; otherwise the actual hash is recorded so you can pin it.
# Known-good (TheBloke TinyLlama-1.1B-Chat-v1.0 Q4_K_M): 9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0
ACTUAL_SHA=$(sha256sum tinyllama.gguf | cut -d' ' -f1)
if [ -n "${GGUF_SHA256:-}" ]; then
  [ "$ACTUAL_SHA" = "$GGUF_SHA256" ] || { echo "!! tinyllama.gguf sha256 mismatch: got $ACTUAL_SHA, expected $GGUF_SHA256"; exit 1; }
  echo "gguf sha256 OK ($ACTUAL_SHA)"
else
  echo "[warn] tinyllama.gguf sha256=$ACTUAL_SHA is NOT pinned — set GGUF_SHA256=$ACTUAL_SHA to enforce it"
fi
"llama.cpp/$BUILD/bin/llama-bench" -m tinyllama.gguf -p 512 -n 128 $NGL
