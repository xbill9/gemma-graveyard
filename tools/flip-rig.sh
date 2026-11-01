#!/usr/bin/env bash
# flip-rig.sh <card>: one AWS GPU for the flip study. Builds llama.cpp b11699 for this card, rebuilds BF16, Q8_0 and Q4_0
# and checks them against the T4's sha256, then runs each rung under the modes below, keeping every answer's text.
#   default  the ladder's own settings
#   repeat   default again: is this rung deterministic on this card?
#   ub1      -ub 1: the prompt goes through the GPU one token at a time, on the kernels decode uses
#   f32      GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32: cuBLAS works in 32-bit floats (L4 only: its prompts take the cuBLAS path)
# A rung with a finished meta.json is skipped, so the script can be re-run after an interruption.
set -uo pipefail
CARD=${1:?card}
W=/opt/dlami/nvme
L=$W/llama.cpp
G=$W/gguf
H=$W/ckpt/gemma-4-E2B-it
D=runs/2026-11-01-flips/$CARD
export PIP_CACHE_DIR=$W/pipcache TMPDIR=$W/tmp HF_HOME=$W/hf PYTHONUSERBASE=$W/pyuser PATH=$W/pyuser/bin:$PATH
mkdir -p "$TMPDIR" "$D"
# 16 GB of RAM does not hold the conversion of a 10 GB checkpoint
swapon --show | grep -q . || { sudo fallocate -l 32G $W/swapfile && sudo chmod 600 $W/swapfile && sudo mkswap $W/swapfile >/dev/null && sudo swapon $W/swapfile; }
log() { echo "$(date -u +%FT%TZ) $*"; }
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d .)
log "card $CARD compute cap $CC"
{ nvidia-smi; lscpu | grep 'Model name'; free -m; } > "$D/host.txt"

if [ ! -x "$L/build/bin/llama-server" ]; then
  git clone -q https://github.com/ggml-org/llama.cpp "$L" && git -C "$L" checkout -q b11699
  (cd "$L" && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CC" -DCMAKE_BUILD_TYPE=Release > "$W/cmake.log" 2>&1 \
    && cmake --build build -j"$(nproc)" --target llama-server llama-quantize >> "$W/cmake.log" 2>&1) || { log "build failed"; exit 1; }
fi
git -C "$L" describe --tags >> "$D/host.txt"
log "build done"

if [ ! -s "$G/e2b-Q4_0.gguf" ]; then
  python3 -m pip install -q --user --break-system-packages -r "$L/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub 2>&1 | tail -1
  python3 -m pip install -q --user --break-system-packages transformers==5.19.0 2>&1 | tail -1  # the pinned requirement cannot read this tokenizer config
  hf download google/gemma-4-E2B-it --revision 3e22461f --local-dir "$H" 2>&1 | tail -1
  mkdir -p "$G"
  python3 "$L/convert_hf_to_gguf.py" "$H" --outtype bf16 --outfile "$G/e2b-BF16.gguf" > "$W/convert.log" 2>&1 || { log "convert failed"; exit 1; }
  for r in Q8_0 Q4_0; do "$L/build/bin/llama-quantize" "$G/e2b-BF16.gguf" "$G/e2b-$r.gguf" "$r" > "$W/q-$r.log" 2>&1; done
fi
for r in BF16 Q8_0 Q4_0; do
  a=$(sha256sum "$G/e2b-$r.gguf" | cut -d' ' -f1)
  b=$(awk -v r="$r" '$1==r{print $4}' runs/2026-10-27-ladder/ladder.tsv)
  [ "$a" = "$b" ] || { log "sha mismatch $r $a $b"; exit 1; }
  log "sha ok $r"
done | tee -a "$D/sha.txt"
grep -q mismatch "$D/sha.txt" && exit 1

done_n() { [ -s "$1/$2.meta.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$1/$2.meta.json')).get('n')==250 else 1)"; }
one() {  # one <mode> <rung> [flags...]
  local m=$1 r=$2; shift 2
  done_n "$D/$m" "$r" && return
  log "start $m $r"
  ./graveyard.py run "$r" "$G/e2b-$r.gguf" data/heldout.jsonl "$D/$m" --llama "$L" "$@" | tail -1
}
one default Q8_0
one ub1 Q8_0 --server-arg -ub --server-arg 1
one default BF16
one ub1 BF16 --server-arg -ub --server-arg 1
one repeat Q8_0
one default Q4_0
one ub1 Q4_0 --server-arg -ub --server-arg 1
one repeat Q4_0
if [ "$CARD" = l4 ]; then
  export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  one f32 Q8_0
  one f32 Q4_0
  unset GGML_CUDA_CUBLAS_COMPUTE_TYPE
fi
log "ALL DONE"
