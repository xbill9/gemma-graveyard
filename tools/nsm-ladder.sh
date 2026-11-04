#!/usr/bin/env bash
# The rest of the ladder under the two settings that made the 1650 Ti and the T4 match at Q8_0 and Q4_0
# (runs/2026-11-03-nsm/REPORT.md): SM count 40 and per_layer_model_proj on the vector route. Rungs that fit on
# the 1650's GPU only; BF16 and F16 put 22 of 36 layers on the CPU there.
# usage: tools/nsm-ladder.sh TAG GGUF_DIR LLAMA [rung ...]
#   1650: tools/nsm-ladder.sh 1650 ~/graveyard-gguf ~/graveyard-llama-nsm          (sets GGML_CUDA_NSM_OVERRIDE=40)
#   T4:   tools/nsm-ladder.sh t4 /opt1/graveyard-gguf /opt1/llama.cpp-nsm          (real SM count, 40)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1 GG=$2 L=$3
shift 3
RUNGS=${*:-Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K}
D=runs/2026-11-03-nsm/ladder
mkdir -p "$D"
export GGML_CUDA_LOG_CUBLAS=1 GGML_CUDA_SPLIT_TO_VEC=1
[ "$TAG" = 1650 ] && export GGML_CUDA_NSM_OVERRIDE=40
for r in $RUNGS; do
  out="$D/$TAG-$r"
  [ -s "$out/$r.jsonl" ] && [ "$(wc -l < "$out/$r.jsonl")" -eq 250 ] && continue
  rm -rf "$out"
  g="$GG/e2b-$r.gguf"
  echo "$(date -u +%FT%TZ) $TAG $r $(sha256sum "$g" | cut -d' ' -f1)" >> "$D/$TAG-sha.txt"
  echo "$(date -u +%FT%TZ) start $r"
  ./graveyard.py run "$r" "$g" data/heldout.jsonl "$out" --llama "$L" --top-logprobs 2 2>&1 | tail -1
done 2>&1 | tee -a "$D/$TAG.log"
echo "$(date -u +%FT%TZ) ALL DONE" | tee -a "$D/$TAG.log"
