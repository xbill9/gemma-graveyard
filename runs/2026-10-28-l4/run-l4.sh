#!/usr/bin/env bash
# The T4 ladder again on one NVIDIA L4 (AWS g6.xlarge, us-west-2): same llama.cpp tag (b11699, built for
# compute capability 8.9), same converter and quantizer, same 250 questions and sampling. Checks the nine
# GGUFs against the T4's sha256, runs the determinism check, the ladder and QAT Q4_0, then buries both.
# A rung with a finished meta.json is skipped, so the script can be re-run after an interruption.
set -uo pipefail
cd "$HOME/gemma-graveyard"
W=/opt/dlami/nvme
L=$W/llama.cpp
G=$W/graveyard-gguf
QAT=$W/models/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf
D=runs/2026-10-28-l4
T4=runs/2026-10-27-ladder
RUNGS=(BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K)
log() { echo "$(date -u +%FT%TZ) $*"; }

done_n() {  # done_n <dir> <rung> <n>: true when the rung finished all n questions
  [ -s "$1/$2.meta.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$1/$2.meta.json')).get('n')==$3 else 1)"
}

rung() {  # rung <dir> <rung> <gguf> [run flags...]
  local d=$1 r=$2 m=$3; shift 3
  log "start $d $r"
  ./graveyard.py run "$r" "$m" data/heldout.jsonl "$d" --llama "$L" "$@" | tail -1
}

mkdir -p $D
{ git -C $L describe --tags; nvidia-smi; lscpu | grep 'Model name'; free -m; } > $D/host.txt

# Check 1: the ladder, and whether each file is byte-identical to the T4's
log "ladder"
./ladder.sh $W/models/gemma-4-E2B-it $G $L > $D/ladder.log 2>&1 || { log "ladder failed"; exit 1; }
cp $G/ladder.tsv $D/ladder.tsv
diff <(cut -f1,3,4 $T4/ladder.tsv) <(cut -f1,3,4 $D/ladder.tsv) > $D/ladder-vs-t4.diff && log "ladder matches T4 sha256" || log "ladder differs from T4"
printf 'QAT_Q4_0\t%s\t%s\t%s\n' "$QAT" "$(stat -c %s $QAT)" "$(sha256sum $QAT | cut -d' ' -f1)" > $D/qat.tsv

# Check 4: determinism on this machine
for i in 1 2; do
  done_n $D/determinism-$i BF16 50 || rung $D/determinism-$i BF16 $G/e2b-BF16.gguf --limit 50
done
./graveyard.py repeat $D/determinism-1 $D/determinism-2 BF16 > $D/determinism.txt 2>&1

# Check 5 and 6: the ladder and QAT Q4_0
for r in "${RUNGS[@]}"; do done_n $D $r 250 || rung $D $r $G/e2b-$r.gguf; done
done_n $D QAT_Q4_0 250 || rung $D QAT_Q4_0 $QAT
log "bury"
./graveyard.py bury $D "${RUNGS[@]}" --json graveyard-ladder.json > $D/bury.txt 2>&1
./graveyard.py bury $D BF16 QAT_Q4_0 Q4_0 --json graveyard-qat.json > $D/bury-qat.txt 2>&1
./graveyard.py autopsy $D "${RUNGS[@]}" > $D/autopsy.txt 2>&1
./graveyard.py autopsy $D BF16 QAT_Q4_0 Q4_0 > $D/autopsy-qat.txt 2>&1
log "done"
