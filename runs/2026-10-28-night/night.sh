#!/usr/bin/env bash
# The overnight queue: closes the gaps in runs/2026-10-27-ladder/REPORT.md "Not measured", most
# important first. Every stage skips a rung whose meta.json already has the full n, so after a stop,
# start it again and it carries on with the rung that was cut off.
#
#   A repeat    Q4_0, Q3_K_M, Q8_0, F16 again, thinking off, 250 questions: determinism below BF16
#   B imatrix   importance matrix from 2,000 GSM8K train problems; Q4_K_M, Q4_0, Q3_K_M, Q2_K with it
#   C think     thinking on, 4,096 max tokens; BF16, a 50-question BF16 repeat, then the ladder and QAT
#   D rest      the other 1,069 GSM8K test questions, thinking off, as many rungs as the night allows
set -uo pipefail
cd /home/xbill_glitnir_com/gemma-graveyard
L=/opt1/llama.cpp
G=/opt1/graveyard-gguf
QAT=/opt1/models/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf
LADDER=runs/2026-10-27-ladder
log() { echo "$(date -u +%FT%TZ) $*"; }

done_n() {  # done_n <dir> <rung> <n>: true when the rung finished all n questions
  [ -s "$1/$2.meta.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$1/$2.meta.json')).get('n')==$3 else 1)"
}

gguf() {  # gguf <rung>: the file for a rung name
  case $1 in
    QAT_Q4_0) echo "$QAT" ;;
    *_IMAT) echo "$G/e2b-${1%_IMAT}-imat.gguf" ;;
    *) echo "$G/e2b-$1.gguf" ;;
  esac
}

rung() {  # rung <dir> <rung> <data> <n> [run flags...]
  local d=$1 r=$2 data=$3 n=$4; shift 4
  if done_n "$d" "$r" "$n"; then log "skip $d $r"; return; fi
  log "start $d $r"
  ./graveyard.py run "$r" "$(gguf "$r")" "$data" "$d" --llama "$L" "$@" | tail -1
}

bury_done() {  # bury_done <dir> <n> <rung>...: bury and autopsy the rungs that are complete, first is the reference
  local d=$1 n=$2; shift 2
  local have=()
  for r in "$@"; do done_n "$d" "$r" "$n" && have+=("$r"); done
  [ "${#have[@]}" -ge 2 ] || return 0
  ./graveyard.py bury "$d" "${have[@]}" --json graveyard.json > "$d/bury.txt" 2>&1
  ./graveyard.py autopsy "$d" "${have[@]}" > "$d/autopsy.txt" 2>&1
}

copy_rows() {  # copy_rows <from> <to> <rung>...: reuse finished rows as the reference or a comparison
  local f=$1 t=$2; shift 2
  for r in "$@"; do cp -n "$f/$r.jsonl" "$f/$r.meta.json" "$t/"; done
}

# A: repeat four rungs of the ladder
A=runs/2026-10-28-repeat; mkdir -p $A
for r in Q4_0 Q3_K_M Q8_0 F16; do
  rung $A $r data/heldout.jsonl 250
  ./graveyard.py repeat $LADDER $A Q4_0 Q3_K_M Q8_0 F16 --common > $A/repeat.txt 2>&1
done
log "A done"

# B: importance matrix
B=runs/2026-10-28-imatrix; mkdir -p $B
IM=$G/imatrix-gsm8k-train.gguf
if [ ! -s "$IM" ]; then
  log "start imatrix"
  $L/build/bin/llama-imatrix -m $G/e2b-BF16.gguf -f /opt1/tmp/gsm8k/calib.txt -o $IM -ngl 99 -c 512 \
    --chunks 300 > $B/imatrix.log 2>&1 || log "imatrix exit $?"
  sha256sum /opt1/tmp/gsm8k/calib.txt $IM > $B/imatrix.sha256
fi
for r in Q4_K_M Q4_0 Q3_K_M Q2_K; do
  f=$G/e2b-$r-imat.gguf
  if [ ! -s "$f" ]; then
    log "quantize $r with imatrix"
    $L/build/bin/llama-quantize --imatrix $IM $G/e2b-BF16.gguf "$f.part" $r > $B/quantize-$r.log 2>&1 && mv "$f.part" "$f"
  fi
done
copy_rows $LADDER $B BF16 Q4_K_M Q4_0 Q3_K_M Q2_K
for r in Q4_K_M_IMAT Q4_0_IMAT Q3_K_M_IMAT Q2_K_IMAT; do
  rung $B $r data/heldout.jsonl 250
  bury_done $B 250 BF16 Q4_K_M Q4_K_M_IMAT Q4_0 Q4_0_IMAT Q3_K_M Q3_K_M_IMAT Q2_K Q2_K_IMAT
done
log "B done"

# C: thinking on
C=runs/2026-10-28-think; C2=runs/2026-10-28-think-repeat; mkdir -p $C $C2
TH=(--think --max-tokens 4096 --ctx 8192)
rung $C BF16 data/heldout.jsonl 250 "${TH[@]}"
rung $C2 BF16 data/heldout.jsonl 50 "${TH[@]}" --limit 50
./graveyard.py repeat $C $C2 BF16 --common > $C2/repeat.txt 2>&1
CR=(BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K QAT_Q4_0)
for r in Q4_0 Q3_K_M Q5_K_M Q4_K_M Q8_0 Q6_K F16 QAT_Q4_0 Q2_K; do
  rung $C $r data/heldout.jsonl 250 "${TH[@]}"
  bury_done $C 250 "${CR[@]}"
done
log "C done"

# D: the other 1,069 questions
D=runs/2026-10-28-rest; mkdir -p $D
DR=(BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K QAT_Q4_0)
for r in BF16 Q4_0 Q3_K_M Q4_K_M Q5_K_M QAT_Q4_0 Q8_0 Q6_K F16 Q2_K; do
  rung $D $r data/heldout-rest.jsonl 1069
  bury_done $D 1069 "${DR[@]}"
done
log "D done"
log "done exit 0"
