#!/usr/bin/env bash
# probe-rig.sh <card> <workdir> [<llama.cpp dir>]: the GPU scores behind a flip. Q8_0, 250 questions, each run also
# records every token's top two candidates and their logprobs (graveyard.py --top-logprobs 2), so fork_probe.py can
# show, at the token where two cards' answers part, what each card scored the two candidates.
#   default    the ladder's own settings
#   ub1        -ub 1: prompt through the decode kernels, one token at a time
#   cc75-ub1   (l4 only) llama.cpp built for the T4's target, sm_75, with -ub 1: the setup that matched T4 -ub 1 word for word
# On a fresh AWS box (no llama.cpp dir given) it builds b11699 for this card and rebuilds Q8_0 from the checkpoint,
# checked against the T4 ladder's sha256. A run with a finished meta.json is skipped.
set -uo pipefail
CARD=${1:?card}
W=${2:?workdir}
L=${3:-$W/llama.cpp}
G=$W/gguf
H=$W/ckpt/gemma-4-E2B-it
D=runs/2026-11-02-probe/$CARD
export PIP_CACHE_DIR=$W/pipcache TMPDIR=$W/tmp HF_HOME=$W/hf PYTHONUSERBASE=$W/pyuser PATH=$W/pyuser/bin:$PATH
mkdir -p "$TMPDIR" "$D"
log() { echo "$(date -u +%FT%TZ) $*"; }
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d .)
log "card $CARD compute cap $CC"
{ nvidia-smi; lscpu | grep 'Model name'; free -m; } > "$D/host.txt"

build() {  # build <dir> <arch>
  [ -x "$1/build/bin/llama-server" ] && return
  [ -d "$1" ] || { git clone -q https://github.com/ggml-org/llama.cpp "$1" && git -C "$1" checkout -q b11699; }
  (cd "$1" && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$2" -DCMAKE_BUILD_TYPE=Release > "$W/cmake-$2.log" 2>&1 \
    && cmake --build build -j"$(nproc)" --target llama-server llama-quantize >> "$W/cmake-$2.log" 2>&1) || { log "build $2 failed"; exit 1; }
}
build "$L" "$CC"
git -C "$L" describe --tags >> "$D/host.txt"
git -C "$L" rev-parse HEAD >> "$D/host.txt"
[ "$CARD" = l4 ] && build "$W/llama-cc75" 75
log "build done"

if [ ! -s "$G/e2b-Q8_0.gguf" ]; then
  swapon --show | grep -q . || { sudo fallocate -l 32G "$W/swapfile" && sudo chmod 600 "$W/swapfile" && sudo mkswap "$W/swapfile" >/dev/null && sudo swapon "$W/swapfile"; }
  python3 -m pip install -q --user --break-system-packages -r "$L/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub 2>&1 | tail -1
  python3 -m pip install -q --user --break-system-packages transformers==5.19.0 2>&1 | tail -1
  hf download google/gemma-4-E2B-it --revision 3e22461f --local-dir "$H" 2>&1 | tail -1
  mkdir -p "$G"
  python3 "$L/convert_hf_to_gguf.py" "$H" --outtype bf16 --outfile "$G/e2b-BF16.gguf" > "$W/convert.log" 2>&1 || { log "convert failed"; exit 1; }
  "$L/build/bin/llama-quantize" "$G/e2b-BF16.gguf" "$G/e2b-Q8_0.gguf" Q8_0 > "$W/q-Q8_0.log" 2>&1
fi
a=$(sha256sum "$G/e2b-Q8_0.gguf" | cut -d' ' -f1)
b=$(awk '$1=="Q8_0"{print $4}' runs/2026-10-27-ladder/ladder.tsv)
[ "$a" = "$b" ] || { log "sha mismatch Q8_0 $a $b" | tee -a "$D/sha.txt"; exit 1; }
log "sha ok Q8_0 $a" | tee -a "$D/sha.txt"

one() {  # one <mode> <llama dir> [flags...]
  local m=$1 l=$2; shift 2
  [ -s "$D/$m/Q8_0.meta.json" ] && return
  log "start $m"
  ./graveyard.py run Q8_0 "$G/e2b-Q8_0.gguf" data/heldout.jsonl "$D/$m" --llama "$l" --top-logprobs 2 "$@" | tail -1
}
one default "$L"
one ub1 "$L" --server-arg=-ub --server-arg=1
[ "$CARD" = l4 ] && one cc75-ub1 "$W/llama-cc75" --server-arg=-ub --server-arg=1
log "ALL DONE"
