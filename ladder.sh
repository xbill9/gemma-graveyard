#!/usr/bin/env bash
# ladder.sh -- build the quantization ladder for Gemma 4 E2B as GGUF files.
#
#   ladder.sh <hf-model-dir> <out-dir> [llama.cpp-dir]
#
# BF16 is the reference (the checkpoint's own dtype, lossless). F16 is converted from the same
# safetensors, so its only difference is the narrower exponent range. Every quantized rung is made
# from the BF16 file with plain llama-quantize: no importance matrix, so each rung is the default
# a user gets. Writes <out-dir>/ladder.tsv: rung, file, bytes, sha256.
set -euo pipefail

HF=${1:?hf model dir}
OUT=${2:?out dir}
LLAMA=${3:-$HOME/llama.cpp}
RUNGS=(Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K)

mkdir -p "$OUT"
for t in bf16 f16; do
  f="$OUT/e2b-${t^^}.gguf"
  [ -s "$f" ] || python3 "$LLAMA/convert_hf_to_gguf.py" "$HF" --outtype "$t" --outfile "$f"
done
for r in "${RUNGS[@]}"; do
  f="$OUT/e2b-$r.gguf"
  [ -s "$f" ] || "$LLAMA/build/bin/llama-quantize" "$OUT/e2b-BF16.gguf" "$f" "$r"
done

{
  printf 'rung\tfile\tbytes\tsha256\n'
  for r in BF16 F16 "${RUNGS[@]}"; do
    f="$OUT/e2b-$r.gguf"
    printf '%s\t%s\t%s\t%s\n' "$r" "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
} > "$OUT/ladder.tsv"
cat "$OUT/ladder.tsv"
