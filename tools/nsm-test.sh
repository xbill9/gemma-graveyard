#!/usr/bin/env bash
# SM-count test on the GTX 1650 Ti (16 SMs): does llama.cpp's default prompt route differ from the T4's
# only because of the SM count it reads? Uses a b11699 copy whose one change is GGML_CUDA_NSM_OVERRIDE
# (replaces prop.multiProcessorCount at ggml-cuda.cu:353). Two Q8_0 runs, default settings:
#   own   override unset; must equal runs/2026-11-02-probe/1650/default word for word (patch is inert)
#   nsm40 override 40, the T4's SM count; compared with runs/2026-11-02-probe/t4/default
set -euo pipefail
cd "$(dirname "$0")/.."
D=${1:-runs/2026-11-03-nsm}
L=${2:-$HOME/graveyard-llama-nsm}
G=${3:-$HOME/graveyard-gguf/e2b-Q8_0.gguf}
mkdir -p "$D"
{ git -C "$L" describe --tags; git -C "$L" rev-parse HEAD; git -C "$L" diff; nvidia-smi; lscpu | grep 'Model name'; free -m; } > "$D/host.txt"
echo "$(date -u +%FT%TZ) Q8_0 $(sha256sum "$G" | cut -d' ' -f1)" > "$D/sha.txt"
{
  echo "$(date -u +%FT%TZ) start own"
  env -u GGML_CUDA_NSM_OVERRIDE ./graveyard.py run Q8_0 "$G" data/heldout.jsonl "$D/own" --llama "$L" --top-logprobs 2 | tail -1
  echo "$(date -u +%FT%TZ) start nsm40"
  GGML_CUDA_NSM_OVERRIDE=40 ./graveyard.py run Q8_0 "$G" data/heldout.jsonl "$D/nsm40" --llama "$L" --top-logprobs 2 | tail -1
  echo "$(date -u +%FT%TZ) ALL DONE"
} 2>&1 | tee "$D/nsm-test.log"
