#!/usr/bin/env bash
# build_replay.sh <llama.cpp-dir>: compile tools/replay.cpp against a CPU build of llama.cpp in <dir>/build.
set -euo pipefail
L=${1:?llama.cpp dir}
cd "$(dirname "$0")"
g++ -std=c++17 -O3 -march=native replay.cpp -o replay -I"$L/include" -I"$L/ggml/include" \
  -L"$L/build/bin" -lllama -lggml -lggml-cpu -lggml-base -Wl,-rpath,"$L/build/bin"
