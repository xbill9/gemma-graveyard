# Follow-up: does vLLM flip answers between cards the same way?

Kept out of the flips article (`docs/devto-gemma-graveyard-flips.md`), which stands on llama.cpp alone.

## Already on disk (not re-run here)

- `~/gemma4-dev/gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-suite-g6/ACCURACY.txt`: GSM8K, 1,319
  questions, greedy, thinking off, one request at a time, same q4w4a16emb4 weights, L4 against RTX 3050.
  Questions graded differently between the cards (`differ`): vLLM 32, llama.cpp 50, PyTorch int4 engine 6.
  vLLM was the same build on both cards; llama.cpp's 3050 run was on `77c4540ce`, so its 50 is not a clean pair.
- Confound: the L4 (Ada) and the 3050 (Ampere) are different generations, so these pairs do not isolate SM count.

## Source (from `~/vllm-v030` at `ced6857`; not confirmed to match the 0.33.0 that ran)

- `csrc/libtorch_stable/quantization/marlin/marlin.cu:564` reads `cudaDevAttrMultiProcessorCount`.
- `marlin_template.h:242` partitions the weight matrix into per-SM "stripes" with cross-block reductions:
  the same shape as llama.cpp's stream-k split.
- `VLLM_BATCH_INVARIANT` exists (`vllm/envs.py:619`). Whether it covers Marlin, or Turing, is unmeasured.
- vLLM adds a source llama.cpp avoids at one request: batch composition.

## Plan

1. vLLM W4A16 on the T4 (40 SMs) and the GTX 1650 Ti (16 SMs), same generation as the article's pair, over
   `data/heldout.jsonl`, top-2 logprobs per token, one request at a time. Unmeasured: whether vLLM loads E2B in
   the 1650 Ti's 4 GB. If it does not, find another same-generation pair with different SM counts.
2. Same card twice first (determinism), then card against card: text, grade, per-token score matches.
3. Repeat with `VLLM_BATCH_INVARIANT=1`, and with concurrent requests to measure batch dependence.
