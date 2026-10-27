# SPEC — gemma-graveyard

Hacktoberfest Week 4 (Ghost in the Machine, due 2026-10-31). Quantize Gemma 4 E2B down a ladder, ask
every rung the same GSM8K questions, and bury each answer at the rung where it first goes wrong. Each
tombstone gives the quality cost of a rung, and each rung's file size, GPU memory and decode speed give
what that loss buys.

## Decisions taken (2026-10-27)

- Model: `google/gemma-4-E2B-it` (the base export, bf16). Text model only.
- Ladder, top to bottom: **BF16** (reference: the checkpoint's own dtype, lossless), **F16**, Q8_0, Q6_K,
  Q5_K_M, Q4_K_M, Q4_0, Q3_K_M, Q2_K. Plain `llama-quantize` from the BF16 GGUF, no importance matrix.
- Extra point, off the ladder: Google's QAT Q4_0 GGUF (`google/gemma-4-E2B-it-qat-q4_0-gguf`). It has
  different weights from the base (trained for Q4_0), so it is compared against the base BF16 reference
  and sits beside the plain Q4_0, never in the ladder order.
- Questions: the 250 GSM8K test problems in `data/heldout.jsonl` (seed 0, from the Week 3 repo). The
  other 1,069 test problems in `data/heldout-rest.jsonl` are there if time allows.
- Sampling: greedy, one request at a time, no prompt cache, thinking off, 1,024 max tokens. Thinking on
  is a second pass only if time allows.
- **Every rung runs on one machine and one llama.cpp build.** Speed and memory are compared only within
  that machine. Record `git -C ~/llama.cpp describe --tags` and `nvidia-smi` in the run directory.
- Prize category: Best Use of Gemma.

## Machine

**Chosen 2026-10-27: the T4 VM.** Host details, disk layout and the BF16-on-Turing caveat are in
`CLAUDE.md` → "Host: the T4 VM".

| Machine | Open question, answered by check 1 |
|---|---|
| 3050 desktop (`gpu`, 8192 MiB) | Does the BF16 GGUF fit fully on the GPU at `-c 4096`? |
| T4 VM (16 GB) | Does llama.cpp's CUDA backend run BF16 at a usable speed on Turing (no native bf16)? |
| 4 GB laptop | BF16 and F16 will not fit on the GPU; usable only for the low rungs. |

Sizes are unmeasured until check 1 records them.

## Checks, in order (each is a go/no-go)

1. `ladder.sh <hf-dir> ~/graveyard-gguf` — build all nine files and `ladder.tsv` (bytes, sha256).
   Record BF16 size and whether it fits the GPU.
2. `f16_range.py <model.safetensors>` on the base checkpoint — how many weights overflow or underflow in
   the F16 cast. (On the QAT-unquantized checkpoint, 2026-10-27 on `yoga9i`: 0 overflow, 5,517,338
   underflow, 5,507,733 of them in `model.vision_tower.patch_embedder.position_embedding_table`, which
   the text GGUF does not carry. `runs/2026-10-27-f16-range-qat/out.txt`.)
3. Smoke test: `graveyard.py run BF16 ... --limit 10`. Rows have `right`, `decode_tps`, no errors.
4. Determinism: run BF16 `--limit 50` twice; every `extracted` must match. If not, a tombstone could be
   noise, and that has to be solved before the ladder runs.
5. Full ladder, then `graveyard.py bury runs/<dir> BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K`.
6. QAT Q4_0: `graveyard.py run QAT_Q4_0 ...`, then `bury runs/<dir> BF16 QAT_Q4_0 Q4_0` as its own table.

## Run commands

On the T4 (paths on `/opt1`; see `CLAUDE.md`):

```
hf download google/gemma-4-E2B-it --local-dir /opt1/models/gemma-4-E2B-it
./ladder.sh /opt1/models/gemma-4-E2B-it /opt1/graveyard-gguf /opt1/llama.cpp
ln -sfn /opt1/graveyard-gguf ~/graveyard-gguf
# and add --llama /opt1/llama.cpp to every graveyard.py run below
D=runs/$(date +%F)-ladder
for r in BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K; do
  ./graveyard.py run $r ~/graveyard-gguf/e2b-$r.gguf data/heldout.jsonl $D
done
./graveyard.py bury $D BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K
```

Stop any `llama-server` holding the GPU first (`pgrep -a llama-server`; note its command line).

## Words used in the output

- **dead**: the reference got it right, this rung gets it wrong.
- **tombstone**: one per question that dies, at the first rung where it does. `right_again_at` lists
  lower rungs that get it right again.
- **ghost**: the reference got it wrong, this rung gets it right. Ghosts are the size of the noise in
  "dead"; the article reports both.
