# gemma-graveyard

Gemma 4 E2B quantized from BF16 down to Q2_K, every rung asked the same 250 GSM8K questions, and each
answer buried at the rung where it first goes wrong. Hacktoberfest 2026 Week 4: Ghost in the Machine.

Status: harness written, no ladder run yet. See `SPEC.md`.

- `ladder.sh` — builds the BF16, F16 and seven quantized GGUFs with llama.cpp.
- `f16_range.py` — counts BF16 weights that overflow or underflow when cast to F16.
- `graveyard.py run` — one rung: llama-server, greedy answers, file size, GPU memory, decode speed.
- `graveyard.py bury` — deaths, tombstones and ghosts per rung, computed in code.

Reused: `lazy_eval.py` (prompt, answer extraction, grading) and `data/` (GSM8K test split, seed 0) from
[xbill9/lazy-gemma](https://github.com/xbill9/lazy-gemma); the grader there came from
`gemma4-dev/jev-tpu-v5e1/gen_eval.py`.
