#!/usr/bin/env bash
# Check 5: every rung on the 250 held-out questions, then bury. A rung with a finished meta.json is skipped,
# so the script can be re-run after an interruption.
set -uo pipefail
cd /home/xbill_glitnir_com/gemma-graveyard
D=runs/2026-10-27-ladder
RUNGS=(BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K)
for r in "${RUNGS[@]}"; do
  if [ -s "$D/$r.meta.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/$r.meta.json')).get('n')==250 else 1)"; then
    echo "skip $r"; continue
  fi
  echo "$(date -u +%FT%TZ) start $r"
  ./graveyard.py run "$r" "/opt1/graveyard-gguf/e2b-$r.gguf" data/heldout.jsonl "$D" --llama /opt1/llama.cpp | tail -1
done
echo "$(date -u +%FT%TZ) bury"
./graveyard.py bury "$D" "${RUNGS[@]}" > "$D/bury.txt" 2>&1
echo "$(date -u +%FT%TZ) done exit $?"
