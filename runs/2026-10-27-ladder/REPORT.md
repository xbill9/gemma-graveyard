# Ladder run, 2026-10-27 — Gemma 4 E2B, BF16 to Q2_K, 250 GSM8K questions

Every number below is from a file in this directory, computed by `graveyard.py bury` or `graveyard.py autopsy`.

## Setup

- Machine: GCE `n1-standard-2`, `us-west2-b`, Tesla T4 15360 MiB, driver 615.71.10 (`host.txt`).
- llama.cpp `b11699`, CUDA, `CMAKE_CUDA_ARCHITECTURES=75`.
- Model: `google/gemma-4-E2B-it`, text model only. BF16 GGUF converted from the checkpoint, every other rung
  from the BF16 GGUF with plain `llama-quantize`, no importance matrix (`ladder.tsv`: bytes, sha256).
- QAT point: `google/gemma-4-E2B-it-qat-q4_0-gguf`, `gemma-4-E2B_q4_0-it.gguf` (`qat.tsv`).
- Questions: the 250 GSM8K test problems in `data/heldout.jsonl`.
- Sampling: temperature 0, one request at a time, no prompt cache, thinking off, 1,024 max tokens,
  `llama-server -ngl 99 -c 4096 -np 1 --seed 0`. Graded in code on `content` only.
- Determinism (check 4): BF16 on 50 questions twice gave 0 mismatched answers and 0 mismatched text
  (`../2026-10-27-determinism-{1,2}`). A changed answer at a lower rung comes from the weights.

## The ladder (`bury.txt`)

Reference BF16: 229 of 250 right. **dead**: BF16 right, this rung wrong. **ghost**: BF16 wrong, this rung
right. **changed**: extracted answer differs from BF16's.

| Rung | Bytes | % of BF16 | GPU MiB | Right | Dead | Ghosts | Changed | Cut off | Median tok/s |
|---|---|---|---|---|---|---|---|---|---|
| BF16 | 9,311,305,152 | 100.0 | 4,671 | 229 | 0 | 0 | 0 | 4 | 40.0 |
| F16 | 9,311,305,152 | 100.0 | 4,671 | 229 | 1 | 1 | 2 | 4 | 39.6 |
| Q8_0 | 4,967,497,152 | 53.3 | 2,631 | 227 | 4 | 2 | 11 | 9 | 64.3 |
| Q6_K | 3,845,346,752 | 41.3 | 2,103 | 229 | 4 | 4 | 13 | 4 | 70.8 |
| Q5_K_M | 3,630,288,320 | 39.0 | 1,897 | 231 | 3 | 5 | 12 | 3 | 78.7 |
| Q4_K_M | 3,427,880,384 | 36.8 | 1,705 | 224 | 9 | 4 | 20 | 6 | 84.1 |
| Q4_0 | 3,360,161,216 | 36.1 | 1,639 | 222 | 15 | 8 | 27 | 10 | 93.6 |
| Q3_K_M | 3,201,351,104 | 34.4 | 1,489 | 191 | 42 | 4 | 57 | 49 | 74.2 |
| Q2_K | 2,989,088,192 | 32.1 | 1,287 | 3 | 226 | 0 | 247 | 246 | 74.0 |

"GPU MiB" is `nvidia-smi` memory used after the model loaded (GPU empty before). It is about half the file
size at every rung because the two embedding tables stay memory-mapped on the CPU (4,480 + 768 MiB at BF16,
from the smoke test's `-lv 4` log); all 36 layers are on the GPU.

**Tombstones: 226.** Where each question first dies: F16 1, Q8_0 4, Q6_K 1, Q5_K_M 2, Q4_K_M 6, Q4_0 9,
Q3_K_M 26, Q2_K 177. **Ghosts: 11** questions, each right at one or more rungs below BF16.

Per-question tombstones and ghosts: `graveyard-ladder.json`.

## How the answers fail (`autopsy.txt`)

**looped**: one line of 8 or more characters appears 5 or more times in the output. **empty**: `content` is
empty, so there is nothing to grade.

| Rung | Cut off | Looped | Empty | Dead | Dead, cut off | Dead, looped | First deaths | …right again lower |
|---|---|---|---|---|---|---|---|---|
| F16 | 4 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| Q8_0 | 9 | 0 | 0 | 4 | 2 | 0 | 4 | 4 |
| Q6_K | 4 | 0 | 0 | 4 | 1 | 0 | 1 | 1 |
| Q5_K_M | 3 | 0 | 0 | 3 | 1 | 0 | 2 | 1 |
| Q4_K_M | 6 | 0 | 0 | 9 | 3 | 0 | 6 | 4 |
| Q4_0 | 10 | 0 | 0 | 15 | 8 | 0 | 9 | 5 |
| Q3_K_M | 49 | 0 | 0 | 42 | 34 | 0 | 26 | 0 |
| Q2_K | 246 | 228 | 63 | 226 | 222 | 207 | 177 | 0 |

Three kinds of death:

1. **F16 to Q4_0: a different path through the same problem.** 23 first deaths; 16 of them are right again
   at some lower rung, 7 stay dead. Answers here end normally and give a wrong number. The one F16 death,
   `test-687` (answer 13), reads one condition the other way and answers 17; Q8_0, Q6_K, Q5_K_M and Q4_K_M
   all get it right again. Down to Q5_K_M, ghosts (1 to 5 per rung) match deaths (1 to 4) in size, so
   the score does not move; from Q4_K_M down, deaths outnumber ghosts (9 to 4, 15 to 8).
2. **Q3_K_M: runs out of room.** 34 of its 42 deaths were cut off at 1,024 tokens. The cut-off answers
   end mid-working; none loop. `test-621`, for one, is still on "Final check on the interpretation of
   the problem" when it hits the limit. BF16 used a median 466.5 tokens on the 44 questions Q3_K_M got wrong while cut off.
3. **Q2_K: collapse.** 246 of 250 cut off, 228 repeat one line 5 or more times, and 75 open a thinking
   block with thinking off (no other rung, and not QAT_Q4_0, writes any `reasoning_content`). In 63 of
   those the whole answer went into the thinking block, leaving `content` empty. Graded on the reasoning
   instead, 2 of those 63 hold the right number (diagnostic only; `content` is what counts).

## QAT Q4_0 beside the plain Q4_0 (`bury-qat.txt`, `autopsy-qat.txt`)

| File | Bytes | GPU MiB | Right | Dead | Ghosts | Changed | Cut off | Median tok/s |
|---|---|---|---|---|---|---|---|---|
| BF16 | 9,311,305,152 | 4,671 | 229 | 0 | 0 | 0 | 4 | 40.0 |
| QAT_Q4_0 | 3,349,516,256 | 1,629 | 221 | 15 | 7 | 27 | 6 | 87.8 |
| Q4_0 | 3,360,161,216 | 1,639 | 222 | 15 | 8 | 27 | 10 | 93.6 |

Same score within one question. Of the 15 deaths each, 6 are the same question; 9 are lost only by QAT_Q4_0
and 9 only by the plain Q4_0. QAT_Q4_0 has fewer deaths cut off at 1,024 tokens (2 against 8). The QAT
weights differ from the base checkpoint (trained for Q4_0), so this compares two files at one size against
one BF16 reference. Per-question detail: `graveyard-qat.json`.

## Speed

Decode speed, median of 250 answers, this T4 only. BF16 (40.0 tok/s) and F16 (39.6) run at the same speed:
Turing has no bfloat16 datapath, and llama.cpp offloads all 36 BF16 layers anyway. Q4_0 is the fastest rung
(93.6). Q3_K_M (74.2) and Q2_K (74.0) decode more slowly than Q4_0 and Q4_K_M on this card.

## Not measured

- Thinking on (`--think`). Every run here has thinking off.
- The other 1,069 GSM8K test problems (`data/heldout-rest.jsonl`).
- Importance-matrix quantization; any machine other than this T4.
- Determinism at a rung other than BF16.

## Files

`ladder.tsv`, `qat.tsv` (bytes, sha256) · `<rung>.jsonl` (one row per question) · `<rung>.meta.json` ·
`<rung>.server.log` · `bury.txt`, `graveyard-ladder.json` · `bury-qat.txt`, `graveyard-qat.json` ·
`autopsy.txt`, `autopsy-qat.txt` · `run-ladder.sh`, `run-qat.sh` and their logs · `host.txt`.

Rebuild the tables:

```
D=runs/2026-10-27-ladder
./graveyard.py bury $D BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K --json graveyard-ladder.json
./graveyard.py bury $D BF16 QAT_Q4_0 Q4_0 --json graveyard-qat.json
./graveyard.py autopsy $D BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K
./graveyard.py autopsy $D BF16 QAT_Q4_0 Q4_0
```
