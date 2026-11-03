#!/usr/bin/env bash
# T4 half of the SM-count / cuBLAS test (runs/2026-11-03-nsm). Run on the GCE T4 VM from ~/gemma-graveyard.
# 1. builds a b11699 copy with tools/graveyard-probe.patch at /opt1/llama.cpp-nsm (sm_75, -j2)
# 2. records the T4's SM count from the driver
# 3. runs Q8_0 twice at default settings with per-token scores:
#      t4-patched   no variables set; must equal runs/2026-11-02-probe/t4/default (patch is inert)
#      t4-vec       GGML_CUDA_SPLIT_TO_VEC=1: the one cuBLAS matrix (per_layer_model_proj) goes through the
#                   one-column vector kernel; compared with the 1650's nsm40-vec run
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${SRC:-/opt1/llama.cpp}
L=${L:-/opt1/llama.cpp-nsm}
G=${G:-$HOME/graveyard-gguf/e2b-Q8_0.gguf}
D=runs/2026-11-03-nsm
export TMPDIR=${TMPDIR:-/opt1/tmp}
if [ ! -x "$L/build/bin/llama-server" ]; then
  [ -d "$L" ] || git -C "$SRC" worktree add --detach "$L" 7182ef71e6d5dd0c8ea76486fac134fcd25025d4
  git -C "$L" apply "$PWD/tools/graveyard-probe.patch"
  cmake -S "$L" -B "$L/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON > "$L/build-config.log" 2>&1
  cmake --build "$L/build" --target llama-server -j2 > "$L/build.log" 2>&1
fi
mkdir -p "$D"
{ git -C "$L" describe --tags; git -C "$L" rev-parse HEAD; git -C "$L" diff --stat; python3 tools/sm_count.py; nvidia-smi; lscpu | grep 'Model name'; free -m; } > "$D/t4-host.txt"
echo "$(date -u +%FT%TZ) Q8_0 $(sha256sum "$G" | cut -d' ' -f1)" > "$D/t4-sha.txt"
{
  echo "$(date -u +%FT%TZ) start t4-patched"
  ./graveyard.py run Q8_0 "$G" data/heldout.jsonl "$D/t4-patched" --llama "$L" --top-logprobs 2 | tail -1
  echo "$(date -u +%FT%TZ) start t4-vec"
  GGML_CUDA_LOG_CUBLAS=1 GGML_CUDA_SPLIT_TO_VEC=1 ./graveyard.py run Q8_0 "$G" data/heldout.jsonl "$D/t4-vec" --llama "$L" --top-logprobs 2 | tail -1
  echo "$(date -u +%FT%TZ) ALL DONE"
} 2>&1 | tee "$D/t4-vec.log"
