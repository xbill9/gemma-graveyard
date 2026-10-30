# gemma-graveyard

Gemma 4 E2B quantized from BF16 down to Q2_K, every rung asked the same 250 GSM8K questions, and each
answer buried at the rung where it first goes wrong. Hacktoberfest 2026 Week 4: Ghost in the Machine.

Status: every check in `SPEC.md` has passed; results in
[`runs/2026-10-27-ladder/REPORT.md`](runs/2026-10-27-ladder/REPORT.md).

Follow-on (not part of the challenge entry): the same ladder on an NVIDIA L4 (AWS `g6.xlarge`) with
byte-identical GGUFs, in [`runs/2026-10-28-l4/`](runs/2026-10-28-l4/) and
[`docs/devto-gemma-graveyard-l4.md`](docs/devto-gemma-graveyard-l4.md). `docs/gpu_derived.py` compares the two
GPUs question by question. A third run, on a 4 GB GTX 1650 laptop with the same files and llama.cpp build
(BF16 and F16 split between GPU and CPU), is in [`runs/2026-10-29-1650/`](runs/2026-10-29-1650/REPORT.md);
`docs/gpu3_derived.py` compares all three GPUs.

On one Tesla T4, against the 229 of 250 questions BF16 gets right: Q8_0 to Q5_K_M stay within 2 of it,
Q4_K_M and Q4_0 lose 5 and 7, Q3_K_M loses 38 (most of its deaths cut off at 1,024 tokens), and Q2_K gets 3
right. Google's QAT Q4_0 scores 221 against the plain Q4_0's 222, and the two lose different questions.
Every rung run twice gives the same text on every question. With thinking on (seven rungs, BF16 to Q3_K_M,
4,096 tokens), Q3_K_M climbs from 191 to 208 right and the rungs above it lose different questions.

- `ladder.sh` — builds the BF16, F16 and seven quantized GGUFs with llama.cpp.
- `f16_range.py` — counts BF16 weights that overflow or underflow when cast to F16.
- `graveyard.py run` — one rung: llama-server, greedy answers, file size, GPU memory, decode speed.
- `graveyard.py bury` — deaths, tombstones and ghosts per rung, computed in code.
- `graveyard.py autopsy` — per rung, answers cut off, looping, thinking unprompted, or with empty content.
- `graveyard.py repeat` — two runs of one rung compared question by question.
- `docs/think_derived.py` — thinking on against thinking off, question by question.

Article: [`docs/devto-gemma-graveyard.md`](docs/devto-gemma-graveyard.md), its cover drawn from the rows by
`docs/make_cover.py`, every figure traced to a file in `docs/evidence/`.

Reused: `lazy_eval.py` (prompt, answer extraction, grading) and `data/` (GSM8K test split, seed 0) from
[xbill9/lazy-gemma](https://github.com/xbill9/lazy-gemma); the grader there came from
`gemma4-dev/jev-tpu-v5e1/gen_eval.py`.
