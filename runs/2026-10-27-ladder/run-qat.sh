#!/usr/bin/env bash
# Check 6: Google's QAT Q4_0 GGUF on the same 250 questions, after the ladder has finished. Then bury it beside
# BF16 and the plain Q4_0.
set -uo pipefail
cd /home/xbill_glitnir_com/gemma-graveyard
D=runs/2026-10-27-ladder
until grep -q ' done exit' "$D/run-ladder.log"; do sleep 30; done
if ! { [ -s "$D/QAT_Q4_0.meta.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/QAT_Q4_0.meta.json')).get('n')==250 else 1)"; }; then
  echo "$(date -u +%FT%TZ) start QAT_Q4_0"
  ./graveyard.py run QAT_Q4_0 /opt1/models/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf data/heldout.jsonl "$D" --llama /opt1/llama.cpp | tail -1
fi
echo "$(date -u +%FT%TZ) bury"
./graveyard.py bury "$D" BF16 QAT_Q4_0 Q4_0 > "$D/bury-qat.txt" 2>&1
echo "$(date -u +%FT%TZ) done exit $?"
