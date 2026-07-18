#!/usr/bin/env bash
# llama.cpp `llama-bench` — LLM token throughput: prefill (pp512) + decode (tg128).
# MODE=cuda (default) or MODE=cpu.  Model: TinyLlama-1.1B Q4_K_M (ungated).
set -uo pipefail
MODE=${MODE:-cuda}
LLM=/root/mlperf/llm
mkdir -p "$LLM"; cd "$LLM"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

if [ "$MODE" = cuda ]; then
  # needs the CUDA toolkit (nvcc); install via NVIDIA apt repo:
  #   cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 cuda-nvrtc-dev-12-8
  export PATH=/usr/local/cuda-12.8/bin:$PATH CUDACXX=/usr/local/cuda-12.8/bin/nvcc
  BUILD=build-cuda; EXTRA="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"; NGL="-ngl 99"
else
  BUILD=build-cpu; EXTRA="-DGGML_NATIVE=ON"; NGL=""
fi

# server OFF (avoids the memory-heavy cpp-httplib) and -j4 (nvcc/httplib OOM at -j24)
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release $EXTRA \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j4 --target llama-bench

cd "$LLM"
[ -s tinyllama.gguf ] || wget -q -O tinyllama.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
"llama.cpp/$BUILD/bin/llama-bench" -m tinyllama.gguf -p 512 -n 128 $NGL
