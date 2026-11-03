# SM count and cuBLAS: what makes the GTX 1650 Ti and the T4 differ at default settings

Gemma 4 E2B Q8_0 (sha256 `6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a` on both machines),
250 held-out GSM8K questions, greedy, seed 0, thinking off, default llama-server settings (`-ngl 99 -c 4096 -np 1`),
per-token top-2 scores (`--top-logprobs 2`).

Cards, read from the driver with `tools/sm_count.py` on 2026-11-03:

| card | SMs | compute capability | shared memory per block |
|---|---|---|---|
| NVIDIA GeForce GTX 1650 Ti with Max-Q Design (yogagpu laptop) | 16 | 7.5 | 49,152 |
| Tesla T4 (GCE `debian13`, n1-standard-2) | 40 | 7.5 | 49,152 |

llama.cpp: b11699 (7182ef71e), built for sm_75 with CUDA 13.4 on both, plus `tools/graveyard-probe.patch`
(`host.txt`, `t4-host.txt`). The patch adds three environment variables and changes nothing when none is set:

- `GGML_CUDA_NSM_OVERRIDE=n` replaces the SM count llama.cpp reads once (`ggml-cuda.cu:353`) and that its matrix,
  attention, softmax and sum kernels use to divide their work.
- `GGML_CUDA_LOG_CUBLAS=1` prints the first 200 matrix products that leave llama.cpp's own kernels for cuBLAS.
- `GGML_CUDA_SPLIT_TO_VEC=1` sends such a product through llama.cpp's one-column vector kernel instead, one prompt
  token at a time: the route a single generated token takes.

Rigs: `tools/nsm-test.sh` (1650), `tools/t4-vec.sh` (T4). Every count is from `tools/fork_probe.py`; its output is
the `*.json` here.

## Result

| pair (Q8_0, 250 answers) | text differs | grade differs | tokens compared | same score on both |
|---|---|---|---|---|
| 1650 own 16 SMs vs T4, both stock (2026-11-02) | 165 | 4 | 50,572 | 29,969 (59.3%) |
| 1650 told 40 SMs vs T4 stock | 32 | 2 | 87,580 | 85,282 (97.4%) |
| **1650 told 40 SMs, cuBLAS matrix on the vector route vs T4, same route** | **0** | 0 | 97,380 | **97,380 (100%)** |
| patch inert, 1650: variables unset vs 1650 stock (2026-11-02) | 0 | 0 | 97,116 | 97,116 (100%) |
| patch inert, T4: variables unset vs T4 stock (2026-11-02) | 0 | 0 | 98,637 | 98,637 (100%) |
| 1650 told 40 SMs: vector route vs cuBLAS | 167 | 10 | 50,648 | 30,132 (59.5%) |

1. **Two things make the cards differ at default settings, and together they are all of it.** With the 1650 told
   the T4's SM count and the one cuBLAS product moved to the vector route on both cards, every answer matches word
   for word and every one of 97,380 scores is the same number. Both runs get 229 of 250 right.
2. **The SM count accounts for 133 of the 165 differing answers.** llama.cpp divides each prompt's matrix work into
   one share per SM and adds the shares' partial sums. A different SM count means a different split and a different
   order of additions. Telling the 1650 it has 40 SMs takes the differences from 165 to 32.
3. **cuBLAS accounts for the other 32.** The "Q8_0" file holds one matrix in BF16, `per_layer_model_proj`
   (1536 × 8960); llama-quantize never quantizes it (`src/llama-quant.cpp:324`). llama.cpp has no batch kernel for
   BF16 on compute capability 7.5, so for any prompt longer than 8 tokens it hands this product to cuBLAS: the only
   cuBLAS call in a run (`GGML_CUDA_LOG_CUBLAS`, 200 of 200 logged products). cuBLAS reads the real card, which the
   SM override does not reach.
4. **The patch alone changes nothing on either card**: rows 4 and 5.
5. **Rerouting one matrix flips as many answers as changing the card.** On the 1650, moving only
   `per_layer_model_proj` off cuBLAS changes 167 of 250 answers. The splits are near-ties, as on every other pair.

## Unmeasured

- Which cuBLAS routine each card picks, and why its result differs. cuBLAS is closed source; the test removes it
  and shows the differences go with it.
- Whether a third card with compute capability 7.5 joins this match under the same two settings. Two cards tested.
- Q8_0 only. The other rungs quantize more matrices but leave `per_layer_model_proj` in BF16 too (same rule in
  llama-quant.cpp); not run.
- The server logs capture no ggml-cuda start-up lines, so the SM override's own log line never appears. The
  override taking effect is shown by the output change (row 2).

## Files

1650: `own/`, `nsm40/`, `nsm40-vec/`, `host.txt`, `sha.txt`, `nsm-test.log`, `nsm40-vec.log`.
T4: `t4-patched/`, `t4-vec/`, `t4-host.txt`, `t4-sha.txt`, `t4-vec.log`.
Pair outputs: `own-vs-1650-default.json`, `nsm40-vs-t4-default.json`, `nsm40-vs-1650-default.json`,
`nsm40-vec-vs-nsm40.json`, `t4-patched-vs-t4-default.json`, `nsm40-vec-vs-t4-vec.json`.
