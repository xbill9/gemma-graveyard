# CLAUDE.md

Hacktoberfest Week 4 entry: which GSM8K answers Gemma 4 E2B loses at each quantization rung, from BF16
down to Q2_K. `SPEC.md` is the working plan; its checks are go/no-go gates, run in order.
`RULES-week4.md` is the challenge text, `TEMPLATE-week4.md` the required post headings.

## Deadline

2026-10-31. Any commit after it must be noted in `README.md`. Credit reused code (`lazy_eval.py` and the
GSM8K split come from `xbill9/lazy-gemma`) in the README and the post.

## Measurement rules

- Correctness is graded by code, never by a model. Deaths, ghosts and every other count come from
  `graveyard.py bury`, never from reading rows.
- Every rung runs on the same machine, llama.cpp build, questions and sampling. Speed and memory are
  compared only within one machine.
- The BF16 reference is the checkpoint's own dtype. F16 is a rung, not a reference.
- A tombstone counts only after check 4 (determinism) has passed on that machine.
- Anything not measured here is marked as unmeasured.

## Gemma 4 + llama-server

Thinking goes to `reasoning_content`; only `content` is graded. Use `/v1/chat/completions`. Thinking is
off by default (`--think` turns it on), so `max_tokens` 1024 is enough.

## Layout

Scripts at the root, `Makefile` with `test` and `lint` (`ruff check` on an explicit file list: add each
new module), runs under `runs/<date>-<name>/` with a `REPORT.md`. Python: the machine's own `python3`,
no virtualenvs. GGUF files live outside the repo (`*.gguf` is ignored).
