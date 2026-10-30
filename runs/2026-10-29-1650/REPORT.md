# Ladder run, 2026-10-29 — the T4 ladder again on a GTX 1650 laptop

Every number below is from a file in this directory: `graveyard.py bury` and `autopsy` for this GPU, and
`three-gpus.txt` (`python3 docs/gpu3_derived.py`) for the comparison with the T4 (`../2026-10-27-ladder`)
and the L4 (`../2026-10-28-l4`).

## Setup

- Machine: `yogagpu`, NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 4096 MiB, compute capability 7.5,
  Intel Core i7-10750H, 15,659 MiB RAM (`host.txt`).
- llama.cpp `b11699`, CUDA, `CMAKE_CUDA_ARCHITECTURES=75`, built in `~/graveyard-llama`.
- The same nine GGUFs as the T4 and L4: `ladder.sh` on `google/gemma-4-E2B-it` revision `3e22461f` gives the
  T4's sha256 for every rung (`ladder-vs-t4.diff` is empty). The QAT file matches too (`qat.tsv`).
  The converter writes the model card into the GGUF header, so the checkpoint directory needs `README.md`
  and the name `gemma-4-E2B-it` to give the same bytes.
- Questions, sampling and grading as the T4 run: 250 GSM8K questions, temperature 0, thinking off,
  1,024 max tokens, `-c 4096 -np 1 --seed 0`.
- **BF16 and F16 do not fit on this GPU.** Their 36 layers need a 4,385 MiB GPU buffer (T4 `-lv 4` log) and 3,671 MiB is
  free. They run with `--ngl auto` (no `-ngl` flag), and llama.cpp's `-fit` puts 14 of 36 layers on the GPU,
  2,408 MiB, with the KV cache on the CPU. Every other rung runs with `-ngl 99`, all 36 layers on the GPU
  (`<rung>.placement.txt`).
- Determinism (check 4): BF16 on 50 questions twice gave 0 mismatched answers and 0 mismatched text
  (`determinism.txt`).

## The ladder (`bury.txt`)

Reference BF16: 229 of 250 right. **Lost**: BF16 right, this rung wrong. **Gained**: BF16 wrong, this rung
right.

| Rung | % of BF16 bytes | GPU MiB | Right | Lost | Gained | Cut off | Median tok/s |
|---|---|---|---|---|---|---|---|
| BF16 | 100.0 | 2,617 | 229 | — | — | 4 | 9.2 |
| F16 | 100.0 | 2,617 | 228 | 2 | 1 | 5 | 9.6 |
| Q8_0 | 53.3 | 2,583 | 229 | 2 | 2 | 4 | 45.9 |
| Q6_K | 41.3 | 2,055 | 226 | 7 | 4 | 6 | 57.2 |
| Q5_K_M | 39.0 | 1,849 | 226 | 6 | 3 | 2 | 61.9 |
| Q4_K_M | 36.8 | 1,657 | 222 | 12 | 5 | 8 | 66.5 |
| Q4_0 | 36.1 | 1,591 | 216 | 19 | 6 | 7 | 70.8 |
| Q3_K_M | 34.4 | 1,441 | 183 | 49 | 3 | 49 | 55.9 |
| Q2_K | 32.1 | 1,239 | 5 | 225 | 1 | 247 | 62.2 |

BF16 and F16 speed is with 22 of 36 layers on the CPU: comparable only with each other. **Tombstones: 226.**
Where each question first dies: F16 2, Q8_0 2, Q6_K 5, Q5_K_M 4, Q4_K_M 9, Q4_0 12, Q3_K_M 29, Q2_K 163.
**Ghosts: 14.**

## QAT Q4_0 (`bury-qat.txt`)

| Rung | Right | Lost | Gained | Cut off | Median tok/s |
|---|---|---|---|---|---|
| QAT_Q4_0 | 222 | 11 | 4 | 6 | 70.9 |
| Q4_0 | 216 | 19 | 6 | 7 | 70.8 |

## Three GPUs, question by question (`three-gpus.txt`)

| Rung | Right: T4 / L4 / 1650 | Graded differently: T4–L4 / T4–1650 / L4–1650 | Lost on all three |
|---|---|---|---|
| BF16 | 229 / 229 / 229 | 2 / 2 / 4 | — |
| F16 | 229 / 229 / 228 | 4 / 3 / 5 | 0 |
| Q8_0 | 227 / 228 / 229 | 3 / 4 / 5 | 0 |
| Q6_K | 229 / 227 / 226 | 6 / 7 / 7 | 2 |
| Q5_K_M | 231 / 228 / 226 | 7 / 9 / 8 | 2 |
| Q4_K_M | 224 / 221 / 222 | 15 / 10 / 9 | 6 |
| Q4_0 | 222 / 220 / 216 | 16 / 22 / 22 | 5 |
| Q3_K_M | 191 / 186 / 183 | 33 / 26 / 31 | 25 |
| Q2_K | 3 / 6 / 5 | 9 / 6 / 7 | 218 |
| QAT_Q4_0 | 221 / 219 / 222 | 12 / 13 / 11 | 6 |

- BF16 is right on 227 questions on all three GPUs and wrong on 19 on all three.
- Each GPU has 226 tombstones; 220 questions are tombstoned on all three, and 160 of them first die at the
  same rung on all three. One question is tombstoned only on the 1650 (`test-147`).
- Splitting BF16 and F16 between GPU and CPU moves no more grades than changing the GPU: the 1650 is 2 and 3
  questions from the T4 at BF16 and F16, the L4 is 2 and 4.
- On the 1650, QAT Q4_0 scores 6 above the plain Q4_0 (222 against 216). On the T4 and the L4 it scores 1
  below.

## Not measured

- Determinism below BF16 on this GPU (only BF16 was run twice).
- BF16 or F16 speed with every layer on this GPU: the card has too little memory.
