# SM-count test: the GTX 1650 Ti told it has the T4's 40 SMs

Gemma 4 E2B Q8_0 (sha256 `6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a`), 250 held-out GSM8K
questions, greedy, seed 0, thinking off, default llama-server settings (`-ngl 99 -c 4096 -np 1`), per-token top-2
scores (`--top-logprobs 2`). Card: NVIDIA GeForce GTX 1650 Ti with Max-Q Design (yogagpu), 16 SMs and compute
capability 7.5, read from the driver (`cuDeviceGetAttribute`) on 2026-11-03.

llama.cpp: b11699 (7182ef71e), built for sm_75 like `~/graveyard-llama`, with one change (`host.txt` holds the
diff): the environment variable `GGML_CUDA_NSM_OVERRIDE` replaces `prop.multiProcessorCount`, the SM count that
llama.cpp reads once at `ggml-cuda.cu:353` and that the matrix, attention, softmax and sum kernels use to size
their work. The T4's SM count, 40, is from published specs (Tom's Hardware, ComputerBase), not measured here.
Rig: `tools/nsm-test.sh`. Every count is from `tools/fork_probe.py`; its output is the `*.json` here.

## Result

| pair (Q8_0, 250 answers) | text differs | grade differs | tokens compared | same score on both |
|---|---|---|---|---|
| 1650, patched build, override unset vs 1650 2026-11-02 default | **0** | 0 | 97,116 | **97,116 (100%)** |
| 1650 told 40 SMs vs T4 default | **32** | 2 | 87,580 | 85,282 (97.4%) |
| 1650 own 16 SMs vs T4 default (2026-11-02) | 165 | 4 | 50,572 | 29,969 (59.3%) |
| 1650 told 40 SMs vs 1650 own 16 SMs | 160 | 6 | 51,714 | 30,896 (59.7%) |

1. **The patch alone changes nothing.** With the variable unset, the patched build gives the 2026-11-02 1650 run's
   text on 250 of 250 answers and the same score at every one of 97,116 tokens.
2. **The SM count accounts for most of the difference from the T4.** Telling the 1650 it has 40 SMs takes the
   answers that differ from the T4 from 165 to 32, and the scores that differ from 40.7% to 2.6% of tokens.
   The same setting moves the 1650 away from its own normal run on 160 answers.
3. **Something else accounts for the remaining 32.** 31 of them were among the 165 that differed before. They are
   still near-ties: median lead 0.14 and 0.12 at the split, largest 0.75; 31 of 32 swap the top two. Grades
   differ on 2. Right answers: 225 told 40 SMs, 229 on the 1650's own count, 227 on the T4 (`../2026-11-02-probe/t4/default`).

## Unmeasured

- What causes the remaining 32. llama.cpp reads other device properties besides the SM count (for example shared
  memory per block), and some operations may go through cuBLAS, which picks its own routines per device. Neither
  was tested.
- The override's log line does not appear in `nsm40/Q8_0.server.log`; the server log captures no ggml-cuda start-up
  lines at all. The override taking effect is shown by the output change (item 2), not by the log.
- The T4's SM count was not read on the T4.
- Q8_0 only, one card.

## Files

`own/`, `nsm40/`: answers with per-token scores, server logs, meta. `host.txt` (build, diff, `nvidia-smi`),
`sha.txt`, `nsm-test.log`. Pair outputs: `own-vs-1650-default.json`, `nsm40-vs-t4-default.json`,
`nsm40-vs-1650-default.json`.
