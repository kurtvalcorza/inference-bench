#!/usr/bin/env bash
# llama.cpp `llama-bench` — LLM token throughput: prefill (pp512) + decode (tg128).
# MODE=cuda (default) or MODE=cpu.  Model: TinyLlama-1.1B Q4_K_M (ungated).
#
# Env: BENCH_ROOT (build/model dir); CUDA_ARCH (default = nvidia-smi compute_cap: A100=80,
#      H200=90, Blackwell=120); CUDA_HOME (default /usr/local/cuda); LLAMA_REF (pin llama.cpp
#      commit/tag, enforced even on a cached clone); GGUF_SHA256 (model hash — enforced by default
#      against the known-good TinyLlama Q4_K_M; set to another hash to use a different model).
set -uo pipefail

MODE=${MODE:-cuda}

if [ -z "${BENCH_ROOT:-}" ]; then
  if [ -d /root/mlperf ]; then BENCH_ROOT=/root/mlperf; else BENCH_ROOT="$HOME/inference-bench-data"; fi
fi
LLM="$BENCH_ROOT/llm"
mkdir -p "$LLM"; cd "$LLM"
# Fail-closed by default (finding #8): pin to a known tag rather than following upstream tip. The
# historical numbers were from an unpinned build; this pin makes future runs reproducible. Override
# LLAMA_REF to build another commit/tag deliberately.
LLAMA_REF="${LLAMA_REF:-b10068}"
if [ ! -d llama.cpp ]; then
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git
fi
# Resolve the pin (fetch by tag/commit name if a cached clone lacks it); fail on an invalid ref.
w=$(git -C llama.cpp rev-parse -q --verify "$LLAMA_REF^{commit}" 2>/dev/null || true)
if [ -z "$w" ]; then
  git -C llama.cpp fetch --filter=blob:none -q origin "$LLAMA_REF" 2>/dev/null \
    || git -C llama.cpp fetch --filter=blob:none -q --tags origin 2>/dev/null || true
  w=$(git -C llama.cpp rev-parse -q --verify "$LLAMA_REF^{commit}" 2>/dev/null || true)
fi
[ -z "$w" ] && { echo "!! LLAMA_REF=$LLAMA_REF not found in llama.cpp clone"; exit 1; }
# reset --hard (not checkout) so a cached clone with dirty TRACKED changes can't silently be built.
git -C llama.cpp reset --hard -q "$w" || { echo "!! could not pin llama.cpp to $LLAMA_REF"; exit 1; }
# Also drop UNTRACKED drift that could alter the CMake configure / compile (finding #5), but keep our
# own build output dirs so a cached build isn't thrown away every run.
git -C llama.cpp clean -fdq -e build-cuda -e build-cpu
echo "llama.cpp pinned: $(git -C llama.cpp rev-parse --short HEAD) (LLAMA_REF=$LLAMA_REF)"
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

# server OFF (avoids the memory-heavy cpp-httplib) and -j4 (nvcc/httplib OOM at -j24).
# GATE both steps: a failed (re)build must NOT fall through to running a stale binary.
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release $EXTRA \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  || { echo "!! cmake configure failed"; exit 1; }
cmake --build "$BUILD" -j"${JOBS:-4}" --target llama-bench \
  || { echo "!! llama-bench build FAILED — refusing to run a possibly-stale binary"; exit 1; }

cd "$LLM"
BIN="$LLM/llama.cpp/$BUILD/bin/llama-bench"
[ -x "$BIN" ] || { echo "!! llama-bench binary missing at $BIN"; exit 1; }
[ -s tinyllama.gguf ] || wget -q -O tinyllama.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# Enforce the model hash BY DEFAULT (known-good TheBloke TinyLlama-1.1B-Chat-v1.0 Q4_K_M).
# To use a different model, set GGUF_SHA256 to its hash.
GGUF_SHA256="${GGUF_SHA256:-9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0}"
ACTUAL_SHA=$(sha256sum tinyllama.gguf | cut -d' ' -f1)
[ "$ACTUAL_SHA" = "$GGUF_SHA256" ] || {
  echo "!! tinyllama.gguf sha256 mismatch: got $ACTUAL_SHA, expected $GGUF_SHA256"
  echo "   (set GGUF_SHA256=$ACTUAL_SHA if you intend to use this file)"; exit 1; }
echo "gguf sha256 OK ($ACTUAL_SHA)"
"$BIN" -m tinyllama.gguf -p 512 -n 128 $NGL
