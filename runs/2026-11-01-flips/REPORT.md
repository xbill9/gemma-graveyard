# Flip study: which GSM8K answers change between GPUs, and why

Gemma 4 E2B, llama.cpp b11699 (7182ef71e), 250 held-out GSM8K questions, greedy decoding, seed 0, thinking off.
Cards: T4 (GCE, 2026-10-27, grades only), L4 (2026-10-28), GTX 1650 and RTX 3050 (2026-10-29), and two AWS
instances on 2026-11-01: g6.xlarge L4 (AMD EPYC 7R13) and g4dn.xlarge T4 (Xeon 8259CL), both driver 595.112.07.
Every count below comes from a script in `tools/`; its output is in this directory.

## Result

1. **A card with a given build is deterministic.** Repeat runs matched word for word on all 250 answers:
   T4 and L4, Q8_0 and Q4_0. A new L4 instance four days later matched the recorded L4 run word for word at
   BF16, Q8_0 and Q4_0 (`modes.txt`).
2. **A flip starts at a near-tie between the top two next tokens.** Replaying recorded answers on CPU, the
   median lead of the top token is 16.6 (BF16), 17.0 (Q8_0), 15.3 (Q4_0) logits. At the first token where two
   cards' texts part, the median lead is 0.044, 0.203 and 0.401 (`replay/`, `tools/fork_gaps.py`; BF16 from 103
   answers). Near-ties are equally common at every rung: about 1.1–1.4% of tokens lead by under 0.25.
3. **Most text differences change nothing.** Text differs between cards on 165–250 of 250 answers; grades
   differ on 3–16 above Q3_K_M. Forks that flip a grade fall early in the answer and on wording, not on
   arithmetic: false `a op b = c` claims stay at 3–9 per 1000 from BF16 to Q3_K_M and reach 597 at Q2_K
   (`tools/flip_arith.py`).
4. **Flips come from a fixed pool of fragile questions.** Over 28 runs (4 cards × BF16..Q4_0): 184 always right,
   7 never, 59 contested. The 30 most contested hold 80.7% of minority results. Contested questions have longer
   solutions (BF16 median 477 vs 329 tokens, AUC 0.81) and usually two competing answers (median 2).
   Which fragile questions flip changes with every card and rung.
5. **Totals stay steady because flips go both ways.** Spread of the right-answer count across four cards
   matches what independent coin flips on the contested questions predict (SD 1.3–2.6 measured vs 1.5–2.6
   predicted, Q8_0..Q4_0). Shuffling card labels: no card is better (all p ≥ 0.27).
6. **The difference between L4 and T4 is in the software build and its code path, not the chip.**
   L4 compiled for the T4's target (`CMAKE_CUDA_ARCHITECTURES=75`) and run with `-ub 1` produced the same text
   as T4 with `-ub 1` on **all 250** Q8_0 answers. Each half alone leaves the cards apart:

   | Q8_0, answers whose text differs (of 250) | text |
   |---|---|
   | L4 vs T4, normal | 172 |
   | L4 vs T4, both `-ub 1` | 161 |
   | L4 built sm_75 vs T4, normal | 174 |
   | L4 built sm_75 vs T4, both `-ub 1` | **0** |
   | same L4 chip, sm_89 vs sm_75 build, both `-ub 1` | 161 |

   Two parts of the code differ by card:
   - **Compile target.** `mmvq.cu` picks a parameter table at compile time (`MMVQ_PARAMETERS_TURING` for
     Volta–Turing, `GENERIC` for Ada), which sets the number of warps per dot product (`calc_nwarps`). That
     changes how the sum is split. **Untested inference:** a different split adds in a different order and
     rounds differently.
   - **Prompt route.** `ggml_cuda_should_use_mmq`: on Ada with fp16 tensor cores, batches of 64+ tokens go to
     cuBLAS; on Turing the quantized path is always MMQ. Prompts here are 64–175 tokens. `-ub 1` sends the
     prompt one token at a time, which takes both cards off cuBLAS.
7. **BF16 is mostly the prompt route.** L4 vs T4 at BF16: 25 answers differ normally, 3 with `-ub 1`.
   Quantized rungs barely move with `-ub 1` (Q8_0 172→161, Q4_0 238→238), and 32-bit cuBLAS on L4 changes as
   many Q8_0 answers (159) as the route does.

## Forced branches (`branch-Q8_0.jsonl`, `branch-Q4_0.jsonl`)

Each distinct grade flip, restarted on CPU from 12 characters past the fork on each side (a flip seen in two
card pairs is tested once):

| rung | distinct flips | both sides right | reproduces the flip | reversed | both wrong |
|---|---|---|---|---|---|
| Q8_0 | 19 | 12 | 5 | 2 | 0 |
| Q4_0 | 35 | 16 | 12 | 4 | 3 |

The first fork alone reproduces the flip in 17 of 54. The rest of the answer, generated on CPU, often lands right
from either side: a flip usually needs more than the first near-tie.

## Unmeasured

- The closing runs on the L4 were lost when the instance shut down at 07:00 UTC before they were copied: L4
  sm_75 with `GGML_CUDA_FORCE_MMQ` at normal settings (would test item 6 without `-ub 1`), L4 sm_75 `-ub 1` at
  BF16 and Q4_0, and the direct GPU logit comparison between prompt routes (6 of 60 BF16 answers survive in
  `l4/nudge/`; measured on Q8_0 in `../2026-11-02-probe/`). So the 0 holds for Q8_0 under `-ub 1` only.
- The rounding-order mechanism in item 6 is read from source, not measured.
- GTX 1650 and RTX 3050 were not rerun: no repeat runs at quantized rungs, no `-ub 1`. The 1650 shares the T4's
  target and differs from AWS T4 on 165 Q8_0 answers; it has no tensor cores, so its MMQ path may differ. Untested.
- GCE T4 (2026-10-27) has grades only; AWS T4 text cannot be compared to it word for word.

## Files

`modes.txt` (`tools/flip_modes.py`), `replay/` (`tools/replay.cpp`, `tools/fork_gaps.py`), `branch-*.jsonl`
(`tools/branch_test.py`), `l4/`, `t4/` (`tools/flip-rig.sh`: answers, server logs, `host.txt`, `sha.txt` with
GGUF sha256 matching the T4 ladder), `prompts.json`. Fork classes, fragility, wrong kinds:
`tools/flip_classes.py`, `tools/flip_core.py`, `tools/flip_kinds.py`, `tools/flip_arith.py`.
