# One stock setting: `-ub 1` makes the GTX 1650 Ti and the T4 match

Gemma 4 E2B, 250 held-out GSM8K questions, greedy, seed 0, thinking off, `-ngl 99 -c 4096 -np 1`, per-token top-2
scores (`--top-logprobs 2`), plus `-ub 1`. Stock llama.cpp b11699 (7182ef71e), built for sm_75 with CUDA 13.4, no patch.
Rig: `tools/ub1-ladder.sh`. Every count is from `tools/fork_probe.py`; its output is the `*.json` here.

`-ub 1` sends the prompt through llama.cpp one token at a time, on the kernels a generated token uses. That route
avoids both causes found in `../2026-11-03-nsm`: the matrix kernel's split of work by SM count (16 on the 1650, 40 on
the T4), and the cuBLAS product of `per_layer_model_proj`.

## Result

| pair (stock b11699, `-ub 1`, 250 answers) | text differs | grade differs | tokens compared | same score on both | right, each card |
|---|---|---|---|---|---|
| Q8_0, 1650 vs GCE T4 (`../2026-11-02-probe/1650-ub1-vs-t4-ub1.json`) | **0** | 0 | 97,450 | **97,450 (100%)** | 226 |
| Q4_0, 1650 vs GCE T4 (`Q4_0-1650-vs-t4.json`) | **0** | 0 | 107,195 | **107,195 (100%)** | 226 |

The GGUF sha256 matches on both machines for each rung (`1650-sha.txt`, `t4-sha.txt`; Q8_0 in
`../2026-11-02-probe/{1650,t4}/sha.txt`). At default settings the same pairs differ on 165 (Q8_0) and 238 (Q4_0)
answers (`../2026-11-03-nsm/REPORT.md`).

1. **One stock setting is enough on these two rungs.** With `-ub 1` on both cards, every answer matches word for word
   and every per-token score is the same number. No patch and no change to the SM count.
2. **It agrees with the two-setting result.** `../2026-11-03-nsm` matched the cards by fixing the SM count and moving
   the one cuBLAS product to the one-token kernel. `-ub 1` puts every product on that kernel, so both causes go at once.
3. **Both cards still disagree with their own default runs.** On one card, `-ub 1` against default settings changes
   about two answers in three (T4 Q8_0: 166 of 250, `../2026-11-02-probe`). The setting makes the cards agree with each
   other. Accuracy stays level: 226 of 250 here, 229 at default settings on Q8_0 under the two-setting fix.

## Unmeasured

- The K-quant rungs (Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K) on the T4. Its Q6_K run stopped at answer 7 on 2026-11-04:
  an `apt-get upgrade` on the VM at 17:33 UTC filled the root disk and moved the NVIDIA user-space libraries to
  620.70.06 while the loaded kernel module stayed 615.71.10, so every GPU call fails until a reboot. The T4's Q4_0 run
  finished at 17:32, before the upgrade, on driver 615.71.10 like every earlier T4 run. The 1650's K-quant runs under
  `-ub 1` are here (`1650-<rung>/`) with no T4 run to compare.
- Prompt speed under `-ub 1`. On the 1650, the whole Q8_0 run took 44.9 minutes against 39.3 at default settings
  (summed per-answer times, `../2026-11-02-probe/1650`); prompt time alone was not separated.
- Cards with another SM count or compute capability. An L4 matched the T4 under `-ub 1` only when llama.cpp was built
  for the T4's target, sm_75 (`../2026-11-01-flips`).

## Files

`1650-<rung>/`, `1650.log`, `1650-sha.txt`, `1650-host.txt` (1650); `t4-Q4_0/`, `t4.log`, `t4-sha.txt` (T4).
`t4-host.txt` was rewritten by a restart after the driver upgrade and holds only the build and the NVML error.
