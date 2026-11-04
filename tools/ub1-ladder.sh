#!/usr/bin/env bash
# Stock b11699, -ub 1 (every prompt token through the one-token kernels, the route a generated token takes), on the
# rungs that fit on the 1650's GPU. Q8_0 under -ub 1 is runs/2026-11-02-probe/{1650,t4}/ub1: 250 of 250 match there.
# usage: tools/ub1-ladder.sh TAG GGUF_DIR LLAMA [rung ...]
#   1650: tools/ub1-ladder.sh 1650 ~/graveyard-gguf ~/graveyard-llama
#   T4:   tools/ub1-ladder.sh t4 ~/graveyard-gguf /opt1/llama.cpp
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1 GG=$2 L=$3
shift 3
RUNGS=${*:-Q4_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K}
D=runs/2026-11-04-ub1
mkdir -p "$D"
{ git -C "$L" describe --tags; git -C "$L" rev-parse HEAD; git -C "$L" diff --stat; nvidia-smi; } > "$D/$TAG-host.txt"
for r in $RUNGS; do
  out="$D/$TAG-$r"
  [ -s "$out/$r.jsonl" ] && [ "$(wc -l < "$out/$r.jsonl")" -eq 250 ] && continue
  rm -rf "$out"
  g="$GG/e2b-$r.gguf"
  echo "$(date -u +%FT%TZ) $TAG $r $(sha256sum "$g" | cut -d' ' -f1)" >> "$D/$TAG-sha.txt"
  echo "$(date -u +%FT%TZ) start $r"
  ./graveyard.py run "$r" "$g" data/heldout.jsonl "$out" --llama "$L" --top-logprobs 2 --server-arg=-ub --server-arg=1 2>&1 | tail -1
done 2>&1 | tee -a "$D/$TAG.log"
echo "$(date -u +%FT%TZ) ALL DONE" | tee -a "$D/$TAG.log"
