# Ladder run, 2026-10-29 — the T4 ladder again on an RTX 3050 desktop

Every number below is from a file in this directory: `graveyard.py bury` and `autopsy` for this GPU,
`determinism.txt` (`graveyard.py repeat`), and `four-gpus.txt` (`python3 docs/gpu4_derived.py`) for the
comparison with the T4 (`../2026-10-27-ladder`), the L4 (`../2026-10-28-l4`) and the GTX 1650
(`../2026-10-29-1650`).

## Setup

- Machine: `gpu`, NVIDIA GeForce RTX 3050 OEM, 8192 MiB, compute capability 8.6, driver 615.78.08,
  Intel Core i7-5820K, 48,089 MiB RAM (`host.txt`).
- llama.cpp `b11699`, CUDA, `CMAKE_CUDA_ARCHITECTURES=86`, built in `/android/graveyard/llama.cpp`.
- The same nine GGUFs as the T4, L4 and 1650: `ladder.sh` on `google/gemma-4-E2B-it` revision `3e22461f`
  gives the T4's size and sha256 for every rung (`ladder-vs-t4.diff` is empty). The QAT file matches the
  L4's (`qat.tsv`).
- Questions, sampling and grading as the T4 run: 250 GSM8K questions, temperature 0, thinking off,
  1,024 max tokens, `-ngl 99 -c 4096 -np 1 --seed 0` on every rung (`<rung>.meta.json`).
- Every rung fits on the GPU. BF16 and F16 take 4,651 MiB once loaded.
- Determinism (check 4): BF16 on 50 questions twice gave 0 mismatched answers and 0 mismatched text, 46 right
  both times (`determinism.txt`).
- Run by `run-3050.sh` (the L4 script with this machine's paths), log in `run-3050.log`.

## The ladder (`bury.txt`)

Reference BF16: 232 of 250 right. **Lost**: BF16 right, this rung wrong. **Gained**: BF16 wrong, this rung
right.

| Rung | % of BF16 bytes | GPU MiB | Right | Lost | Gained | Cut off | Median tok/s |
|---|---|---|---|---|---|---|---|
| BF16 | 100.0 | 4,651 | 232 | — | — | 4 | 40.3 |
| F16 | 100.0 | 4,651 | 232 | 1 | 1 | 4 | 40.3 |
| Q8_0 | 53.3 | 2,611 | 226 | 7 | 1 | 9 | 67.8 |
| Q6_K | 41.3 | 2,083 | 227 | 8 | 3 | 6 | 75.3 |
| Q5_K_M | 39.0 | 1,877 | 226 | 8 | 2 | 2 | 82.2 |
| Q4_K_M | 36.8 | 1,685 | 221 | 14 | 3 | 7 | 91.5 |
| Q4_0 | 36.1 | 1,619 | 218 | 16 | 2 | 9 | 99.7 |
| Q3_K_M | 34.4 | 1,469 | 191 | 44 | 3 | 45 | 64.2 |
| Q2_K | 32.1 | 1,267 | 3 | 229 | 0 | 249 | 95.0 |

**Tombstones: 230.** Where each question first dies: F16 1, Q8_0 6, Q6_K 4, Q5_K_M 5, Q4_K_M 7, Q4_0 8,
Q3_K_M 28, Q2_K 171. **Ghosts: 11.**

BF16 and F16 run at the same speed on this GPU. Q3_K_M is the only rung slower than the one above it; its
answers average 622.7 tokens against Q4_0's 432.0.

## How answers fail (`autopsy.txt`)

- Q8_0 to Q4_0: between 1 and 4 of each rung's lost answers are cut off at 1,024 tokens; none loop and none
  open a thinking block.
- Q3_K_M: 33 of its 44 lost answers are cut off and 2 loop. None of its 28 first deaths is right again lower
  down.
- Q2_K: 249 answers cut off and 208 loop. 78 open a thinking block unprompted, and 67 have empty `content`.

## QAT Q4_0 (`bury-qat.txt`)

| Rung | Right | Lost | Gained | Cut off | Median tok/s |
|---|---|---|---|---|---|
| QAT_Q4_0 | 221 | 15 | 4 | 6 | 99.8 |
| Q4_0 | 218 | 16 | 2 | 9 | 99.7 |

5 questions are lost under both; 10 only under QAT, 11 only under plain Q4_0.

## Four GPUs, question by question (`four-gpus.txt`)

| Rung | Right: T4 / L4 / 1650 / 3050 | Lost: T4 / L4 / 1650 / 3050 | Lost on all four |
|---|---|---|---|
| BF16 | 229 / 229 / 229 / 232 | — | — |
| F16 | 229 / 229 / 228 / 232 | 1 / 1 / 2 / 1 | 0 |
| Q8_0 | 227 / 228 / 229 / 226 | 4 / 3 / 2 / 7 | 0 |
| Q6_K | 229 / 227 / 226 / 227 | 4 / 6 / 7 / 8 | 2 |
| Q5_K_M | 231 / 228 / 226 / 226 | 3 / 6 / 6 / 8 | 1 |
| Q4_K_M | 224 / 221 / 222 / 221 | 9 / 14 / 12 / 14 | 5 |
| Q4_0 | 222 / 220 / 216 / 218 | 15 / 16 / 19 / 16 | 5 |
| Q3_K_M | 191 / 186 / 183 / 191 | 42 / 45 / 49 / 44 | 20 |
| Q2_K | 3 / 6 / 5 / 3 | 226 / 225 / 225 / 229 | 216 |
| QAT_Q4_0 | 221 / 219 / 222 / 221 | 15 / 15 / 11 / 15 | 6 |

Questions graded differently, pairs involving the 3050:

| Rung | T4–3050 | L4–3050 | 1650–3050 |
|---|---|---|---|
| BF16 | 3 | 3 | 3 |
| F16 | 3 | 5 | 4 |
| Q8_0 | 9 | 6 | 9 |
| Q6_K | 4 | 10 | 7 |
| Q5_K_M | 9 | 10 | 10 |
| Q4_K_M | 15 | 14 | 13 |
| Q4_0 | 18 | 12 | 14 |
| Q3_K_M | 34 | 31 | 32 |
| Q2_K | 6 | 9 | 6 |
| QAT_Q4_0 | 8 | 12 | 11 |

- BF16 is right on 227 questions on all four GPUs and wrong on 18 on all four. The 3050 gets 1 question right
  that the other three all miss, and misses none that the other three all get right; its other 4 right
  answers are questions the other GPUs split on.
- Tombstones: 226 on each of the T4, L4 and 1650, 230 on the 3050. 218 questions are tombstoned on all four,
  and 147 of them first die at the same rung on all four. One question is tombstoned only on the 3050
  (`test-358`).
- Against the L4 at BF16, 31 of 250 answers differ in text; 4 have a different extracted answer and 3 a
  different grade (`graveyard.py repeat ../2026-10-28-l4 . BF16`).
- No question is lost at F16 or Q8_0 on all four GPUs. The counts at those rungs agree within a few
  questions; which questions they are depends on the GPU.

## Not measured

- Determinism below BF16 on this GPU (only BF16 was run twice).
- Layer placement: `graveyard.py` gained `<rung>.placement.txt` after this run started, so this directory has
  none. Every rung ran with `-ngl 99`.
- Speed against the other GPUs: it is compared only within one machine.
