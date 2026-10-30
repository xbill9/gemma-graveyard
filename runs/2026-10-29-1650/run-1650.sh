#!/usr/bin/env bash
# The T4 ladder again on a GTX 1650 laptop (yogagpu, 4096 MiB, compute capability 7.5): same llama.cpp tag
# (b11699, built for sm 75 in ~/graveyard-llama), same converter and quantizer, same 250 questions and
# sampling. Checks the nine GGUFs against the T4's sha256, runs the determinism check, the ladder and QAT
# Q4_0, then buries both. BF16 and F16 run with --ngl auto, so llama.cpp splits them between GPU and CPU;
# <rung>.placement.txt records where each rung's weights went. A rung with a finished meta.json is skipped, so the script can be re-run after
# an interruption.
set -uo pipefail
cd "$HOME/gemma-graveyard"
L=$HOME/graveyard-llama
G=$HOME/graveyard-gguf
HF=$HOME/models/gemma-4-E2B-it  # symlink to HF snapshot 3e22461f; the converter names the model after this directory
QAT=$HOME/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
D=runs/2026-10-29-1650
T4=runs/2026-10-27-ladder
RUNGS=(BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K)
log() { echo "$(date -u +%FT%TZ) $*"; }

done_n() {  # done_n <dir> <rung> <n>: true when the rung finished all n questions
  [ -s "$1/$2.meta.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$1/$2.meta.json')).get('n')==$3 else 1)"
}

ngl() {  # ngl <rung>: the 16-bit rungs (about 4.4 GB of GPU weights) do not fit in 4 GB, so -fit places them
  case $1 in BF16|F16) echo auto ;; *) echo 99 ;; esac
}

placement() {  # placement <dir> <rung> <gguf>: where the weights go, from one verbose load with run's flags
  local f=$1/$2.placement.txt n
  [ -s "$f" ] && return
  mkdir -p "$1"
  n=$(ngl "$2"); [ "$n" = auto ] && n=() || n=(-ngl "$n")
  "$L/build/bin/llama-server" -m "$3" "${n[@]}" -c 4096 -np 1 --port 8091 --seed 0 -lv 4 > "$f.log" 2>&1 & local p=$!
  for _ in $(seq 300); do curl -sf localhost:8091/health > /dev/null && break; sleep 1; done
  kill $p; wait $p 2> /dev/null
  grep -E 'common_params_fit_impl|offload|buffer size' "$f.log" > "$f"; rm -f "$f.log"
}

rung() {  # rung <dir> <rung> <gguf> [run flags...]
  local d=$1 r=$2 m=$3; shift 3
  placement "$d" "$r" "$m"
  log "start $d $r"
  ./graveyard.py run "$r" "$m" data/heldout.jsonl "$d" --llama "$L" --ngl "$(ngl "$r")" "$@" | tail -1
}

mkdir -p $D
{ git -C $L describe --tags; nvidia-smi; lscpu | grep 'Model name'; free -m; } > $D/host.txt

# Check 1: the ladder, and whether each file is byte-identical to the T4's
log "ladder"
./ladder.sh $HF $G $L > $D/ladder.log 2>&1 || { log "ladder failed"; exit 1; }
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
