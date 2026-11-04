---
title: "One Word, Two GPUs: Why Gemma 4 Answers the Same Question Differently on a Laptop and a Tesla T4"
published: false
description: "Same GGUF file, same llama.cpp build, greedy decoding, and a GTX 1650 Ti answers $20 where a Tesla T4 answers 2000. One GSM8K question followed down to the word where the two cards part, the order in which each card adds its numbers, and the one stock setting that makes them agree."
tags: gemma, llamacpp, cuda, nvidia
cover_image: https://raw.githubusercontent.com/xbill9/gemma-graveyard/main/docs/devto-cover-flips.24a2d00b.jpg
---

This article provides a step by step investigation of why Gemma 4 E2B, run from the same GGUF file with the same llama.cpp build and greedy decoding, gives different answers on a GTX 1650 Ti laptop GPU and a Tesla T4, followed through one GSM8K question from the first differing word down to the order in which each card adds its numbers.

https://github.com/xbill9/gemma-graveyard

---

#### A Week Down One Comment

On 2026-10-29 the L4 article in this series picked up one line from a reader:

> Same files, same build, and still 2 BF16 questions graded differently across the two cards is the part I keep coming back to.

The same question had been nagging here too. Byte-identical model files, one llama.cpp tag, temperature zero: two cards ought to print the same words. At BF16 they disagreed on the grade of 2 questions. At Q8_0 they disagreed on the wording of most answers.

A week of runs on three NVIDIA cards later, every one of those differences is accounted for. This article follows one of them, question **test-358**, from the answer down to the arithmetic.

---

#### The Results in Brief

- The GTX 1650 Ti answers test-358 with **$20**, which is right. The Tesla T4 answers **2000**, which is graded wrong.
- The two cards print the same 124 tokens, then part at one word: the 1650 picks " spent" at 51.4% against " saved" at 47.8%; the T4 picks " saved" at 59.1% against " spent" at 40.4%.
- Each card repeats itself exactly. Run twice on different days, the 1650 gives the same score at all 97,116 tokens compared.
- Two things make the cards differ. llama.cpp splits each big matrix product into one share per streaming multiprocessor (SM), and the 1650 has 16 where the T4 has 40. One matrix stored in BF16 goes to NVIDIA's cuBLAS library, which also picks its method by the card.
- Adding the same 8,960 float32 numbers in 16 shares gives 47.0389595; in 40 shares it gives 47.0389099.
- With one stock setting, `-ub 1`, the two cards match on all 250 answers and all 97,450 token scores at Q8_0, and on all 107,195 at Q4_0. It costs 14 to 15% of total run time.
- The model's own scores cannot say which answers will change grade. Catching all 18 that change across seven setups means flagging 164 of 250 questions.

---

#### At This Point You Should Have…

- The repository, `xbill9/gemma-graveyard`, with its 250 held-out GSM8K questions in `data/heldout.jsonl`.
- llama.cpp `b11699` (`7182ef71e`) built with CUDA for compute capability 7.5 on every card.
- The Gemma 4 E2B Q8_0 GGUF, sha256 `6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a`, on each machine.
- Two cards of the same generation: a GTX 1650 Ti with Max-Q Design (laptop, 4 GB) and a Tesla T4 (GCE `n1-standard-2`). Both are Turing, compute capability 7.5.

---

#### Step 1 — Ask Both Cards the Same Question

`graveyard.py run` starts its own `llama-server` with `-ngl 99 --seed 0`, sends each question with greedy decoding and thinking off, grades the last number in the answer, and with `--top-logprobs 2` records the two highest-scoring candidates at every generated token.

```bash
./graveyard.py run Q8_0 ~/graveyard-gguf/e2b-Q8_0.gguf data/heldout.jsonl out/1650-default \
  --llama ~/graveyard-llama --top-logprobs 2
```

The question:

```text
question: Marisa gets $5 as pocket money every day from her parents. She buys 4 lollipops worth 25 cents each every day and saves the change in her piggy bank. How much money does she put in her piggy bank if she saves for 5 days?
gold answer: 20
```

The two cards' last lines, from `docs/flip_derived.py`:

```text
GTX 1650 Ti tokens generated: 412; final line: 'Answer: 20.00'; extracted 20.0; right True
T4 tokens generated: 345; final line: 'Answer: 2000'; extracted 2000.0; right False
```

Both cards do the sum correctly. The T4 works in cents and reports 2000 cents. The grader wants dollars.

---

#### Step 2 — Find the Word Where the Answers Part

The model writes one token at a time, and each token is the highest-scoring candidate given everything written so far. So the two answers are identical up to the first token where the cards rank the candidates differently, and everything after that follows from it.

```bash
python3 docs/flip_derived.py
```

```text
tokens identical on both cards before the split: 124
text before the split (last 120 chars): 'tep 3: Calculate the amount of money Marisa spends each day.**\nMarisa receives \\$5 (which is 500 cents) every day.\nMoney'
GTX 1650 Ti at the split: ' spent' 51.4%, ' saved' 47.8%
T4 at the split: ' saved' 59.1%, ' spent' 40.4%
GTX 1650 Ti next 12 tokens: ' spent per day = Total pocket money - Cost of lollip'
T4 next 12 tokens: ' saved per day = Money received - Money spent\nMoney saved'
```

At "Money ___" the model is close to a coin flip between " spent" and " saved" on both cards. The 1650 lands on one side and the T4 on the other. From there the 1650 writes "Money spent per day", the T4 writes "Money saved per day", and 221 tokens later they reach different final lines.

🔎 **Tip: the split is almost always a near-tie.** Across all 165 Q8_0 answers where the two cards' text differs, the median lead of the top candidate at the split is 0.12 logits on the 1650 and 0.13 on the T4, and in 157 of the 165 the two cards hold the same two candidates in opposite order.

Edward Lorenz met the same shape when he repeated a weather simulation with one value rounded from .506127 to .506, and the whole pattern his program produced was transformed. A fully deterministic calculation, a difference in the sixth decimal place, and a result that changes completely.

---

#### Step 3 — Check That Each Card Repeats Itself

A difference between cards means something only if each card agrees with itself. The 1650 ran Q8_0 at default settings on 2026-11-02, and again on 2026-11-03 with a patched build whose new switches were all off:

```text
same card twice: 1650 default 2026-11-02 vs 1650 patched build, variables unset, 2026-11-03: text differs 0, grade differs 0, tokens compared 97116, same score 97116, median lead at the split None / None, top two swapped 0 of 0
```

Every answer matches word for word, and every one of 97,116 token scores is the same number. The T4 gives the same result against its own earlier run (98,637 of 98,637). Each card is deterministic. The answers depend on which card runs them.

---

#### Step 4 — Count Each Card's Processors

Both cards report compute capability 7.5. They differ in how many streaming multiprocessors (SMs) they have, read from the driver with `tools/sm_count.py`:

```text
NVIDIA GeForce GTX 1650 Ti with Max-Q Design sm=16 cc=7.5 smem_per_block=49152
Tesla T4 sm=40 cc=7.5 smem_per_block=49152
```

llama.cpp reads the SM count once at start-up (`ggml/src/ggml-cuda/ggml-cuda.cu:353`). Its matrix kernel for quantized weights splits a prompt's work into one share per SM so that every SM stays busy (`mmq.cuh`, the "stream-k" split), then adds the shares' partial sums together. Sixteen SMs means sixteen partial sums. Forty SMs means forty.

Float32 addition rounds after every step, so the grouping changes the last digits. `docs/flip_derived.py` adds the same 8,960 random float32 numbers both ways:

```text
8,960 float32 numbers, numpy default_rng(0).standard_normal, each share added in order, then the shares
16 shares of 560: 47.0389595
40 shares of 224: 47.0389099
exact sum (math.fsum in float64): 47.0389371
16-share and 40-share sums equal: False
```

Both are valid float32 results, either side of the exact sum. The model makes sums like this in every layer for every token, and at a near-tie like "Money ___" a change in the sixth digit is enough to swap the two candidates.

The test: a small patch (`tools/graveyard-probe.patch`) adds `GGML_CUDA_NSM_OVERRIDE`, which replaces the SM count llama.cpp reads. Told it has 40 SMs, the 1650 splits its work the way the T4 does:

```text
1650 default vs T4 default: text differs 165, grade differs 4, tokens compared 50572, same score 29969, ...
1650 told 40 SMs vs T4 default: text differs 32, grade differs 2, tokens compared 87580, same score 85282, ...
```

The SM count accounts for 133 of the 165 differing answers.

---

#### Step 5 — Find the Matrix That Goes to cuBLAS

The other 32 come from one matrix. The Q8_0 file stores `per_layer_model_proj` (1536 × 8960) in BF16, because llama-quantize leaves it unquantized (`src/llama-quant.cpp:324`). llama.cpp has no batch kernel for BF16 on compute capability 7.5, so for any prompt longer than 8 tokens it hands that product to cuBLAS. With the patch's `GGML_CUDA_LOG_CUBLAS=1`, 200 of 200 logged cuBLAS calls are this one matrix.

cuBLAS chooses its own method by the card. NVIDIA's documentation, section 2.1.4, states the limit:

> By design, all cuBLAS API routines from a given toolkit version, generate the same bit-wise results at every run when executed on GPUs with the same architecture and the same number of SMs.

The 1650 and the T4 share the architecture and differ in SM count, so cuBLAS is free to give them different bits. The SM override cannot reach it, because cuBLAS reads the real card.

The patch's `GGML_CUDA_SPLIT_TO_VEC=1` sends that one product through llama.cpp's single-token kernel instead. With both changes:

```text
1650 both fixes vs T4 matrix fix: text differs 0, grade differs 0, tokens compared 97380, same score 97380, ...
```

All 250 answers match, and all 97,380 token scores are the same number. The same two changes make the cards match at Q4_0 and at every K-quant rung from Q6_K to Q2_K.

---

#### Step 6 — Make Them Agree With One Stock Setting

`-ub 1` sets llama.cpp's micro-batch to one token. The prompt then goes through the same single-token kernels a generated token uses, which split no work by SM count and never call cuBLAS. It needs no patch.

```bash
tools/ub1-ladder.sh 1650 ~/graveyard-gguf ~/graveyard-llama
```

```text
1650 -ub 1 vs T4 -ub 1: text differs 0, grade differs 0, tokens compared 97450, same score 97450, ...
Q4_0: 1650 -ub 1 vs T4 -ub 1: text differs 0, grade differs 0, tokens compared 107195, same score 107195, ...
```

With `-ub 1` on both cards, both cards answer test-358 with $20 and get 226 of 250 right at Q8_0.

---

#### Step 7 — What Agreement Costs

Generation speed stays within 1 token per second of default on both cards (42.5 to 42.9 on the 1650, 57.0 to 58.0 on the T4). Prompt processing slows down, more on the T4 than on the 1650. From the server logs and the summed per-answer times:

```text
card | route | prompt tokens/s | prompt slowdown | median generation tokens/s | whole run, summed seconds | whole run vs default
GTX 1650 Ti | default | 289 | 1.0x | 42.8 | 2358 | +0%
GTX 1650 Ti | both fixes | 239 | 1.2x | 42.5 | 2403 | +2%
GTX 1650 Ti | -ub 1 | 58 | 5.0x | 42.9 | 2696 | +14%
T4 | default | 1270 | 1.0x | 57.9 | 1722 | +0%
T4 | matrix fix | 1060 | 1.2x | 57.0 | 1734 | +1%
T4 | -ub 1 | 81 | 15.7x | 58.0 | 1986 | +15%
```

GSM8K prompts are short, so a 15.7x slower prompt adds 15% to the whole run. Long prompts were not measured; they spend a larger share of each run on the prompt.

---

#### Step 8 — Ask Whether the Model Can Warn You

If the model knew when it was at a coin flip, it could flag the answers at risk. Seven Q8_0 setups give seven sets of answers: the T4, the 1650 and the L4 at default settings, the L4 with `-ub 1`, the 1650 and T4 with `-ub 1` (identical), the 1650 told 40 SMs, and the 1650 and T4 with both fixes (identical). `tools/route_spread.py` counts them:

```text
"runs": 7, "questions": 250, "text_varies": 219, "grade_varies": 18
```

219 of 250 answers are worded differently in at least one setup, and 18 change grade. test-358 is right in 3 of the 7.

Flag a question when any token in the T4's default answer had a lead below a threshold:

| flag when some token's lead is below | questions flagged | of the 18 that change grade | of the 232 that do not |
|---|---|---|---|
| 0.05 | 112 | 12 | 100 |
| 0.1 | 164 | 18 | 146 |
| 0.25 | 218 | 18 | 200 |

Catching all 18 means flagging 164 questions. Nearly every answer has a near-tie somewhere, and most of them change only the wording. Looking only at the last 12 tokens, where the final number is written, catches 1 of the 18.

---

#### Step 9 — Ask Which Setup Is Right

On the 1650 alone, each rung under `-ub 1`, under default settings and under both fixes, from `docs/evidence-flips/ladder-routes.txt`:

| rung | right, `-ub 1` | right, default | right, both fixes | grade changes, `-ub 1` vs default |
|---|---|---|---|---|
| Q8_0 | 226 | 229 | 229 | 9 |
| Q6_K | 231 🥇 | 226 | 228 | 7 |
| Q5_K_M | 228 | 226 | 230 🥇 | 10 |
| Q4_K_M | 223 | 222 | 226 🥇 | 13 |
| Q4_0 | 226 🥇 | 216 | 224 | 18 |
| Q3_K_M | 190 | 183 | 195 🥇 | 37 |
| Q2_K | 3 🥇 | 5 | 2 | 6 |

No setup wins every rung. The totals move by 3 to 12 per rung while 6 to 37 individual questions change grade. Each setup is its own sample of the model's near-ties, and the score is an average over which way they happened to fall.

---

#### Compare and Contrast

| | default settings | SM count + matrix fix | `-ub 1` |
|---|---|---|---|
| Needs a patch | no | yes | no 🥇 |
| 1650 and T4 answers match, Q8_0 | 85 of 250 | 250 of 250 🥇 | 250 of 250 🥇 |
| Whole run, T4 | baseline 🥇 | +1% | +15% |
| Prompt speed, T4 | 1,270 tokens/s 🥇 | 1,060 tokens/s | 81 tokens/s |
| Right of 250, Q8_0, T4 | 227 | 229 | 226 |

---

#### So, Which One?

For a benchmark that two people on two cards need to reproduce, run with `-ub 1` and say so: it makes these two cards agree to the last digit with stock llama.cpp, for 15% more run time on short prompts. For serving, the default settings are faster and no less accurate; the cost is that an answer from one card is a sample, and another card can word it differently or grade it differently.

Report a score with the card, the build and the settings that produced it. Two cards that agree on a total can still disagree on which questions they got right.

---

#### Summary

The goal of this article was to explain why the same Gemma 4 E2B file gives different answers on two GPUs of the same generation. The key to the solution was following one question, test-358, to the single word where the two cards part, and then removing each cause of the difference until the two cards matched at every token. The results were:

- 🟢 Each card is deterministic: the same card run twice matches on all 97,116 token scores.
- 🟢 Two causes explain every difference: llama.cpp splits matrix work by SM count (16 against 40), and cuBLAS handles one BF16 matrix by the card. Removing both gives 250 of 250 matching answers.
- 🟢 Stock `-ub 1` makes the GTX 1650 Ti and the T4 match on every token score at Q8_0 and Q4_0, for 14 to 15% more run time.
- ⚠️ At default settings the two cards word 165 of 250 Q8_0 answers differently and grade 4 differently. test-358 is right on the 1650 and wrong on the T4.
- ⚠️ Seven setups on Q8_0 change the grade of 18 questions; scores stay between 225 and 230.
- ❌ The model's own scores cannot pick out the answers at risk: catching all 18 means flagging 164 of 250.

Scope: Gemma 4 E2B, 250 held-out GSM8K questions, greedy, seed 0, thinking off, 1,024 max tokens, llama.cpp `b11699` built for compute capability 7.5 with CUDA 13.4, one run per setup. Cards: a GTX 1650 Ti with Max-Q Design (laptop), a Tesla T4 on GCE, and an L4 on AWS for the seven-setup count. The T4 rungs under `-ub 1` stop at Q4_0; the K-quant rungs under `-ub 1` are on the 1650 only. Speed is compared within each card only. Every figure here is recomputed from the committed runs by `docs/flip_derived.py` and `make verify`.

The strategy for tracing a GPU-to-GPU answer flip in Gemma 4 down to the order of its additions was validated with an incremental step by step approach.

---

#### References

- [The comment that started it](https://dev.to/robertpaschal/comment/3jd4d), on the L4 article
- [Performance vs Quality: Gemma 4 E2B Quantization on an NVIDIA L4 and a Tesla T4](https://dev.to/gde/performance-vs-quality-gemma-4-e2b-quantization-on-an-nvidia-l4-and-a-tesla-t4-4b90), the L4 article
- [Four GPUs, One Quantization Ladder: Gemma 4 E2B From a 4 GB Laptop to an NVIDIA L4](https://dev.to/gde/four-gpus-one-quantization-ladder-gemma-4-e2b-from-a-4-gb-laptop-to-an-nvidia-l4-4olp)
- [gemma-graveyard on GitHub](https://github.com/xbill9/gemma-graveyard)
- [SM count and cuBLAS report](https://github.com/xbill9/gemma-graveyard/blob/main/runs/2026-11-03-nsm/REPORT.md)
- [`-ub 1` report](https://github.com/xbill9/gemma-graveyard/blob/main/runs/2026-11-04-ub1/REPORT.md)
- [cuBLAS documentation, Results Reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
- [When the Butterfly Effect Took Flight](https://www.technologyreview.com/2011/02/22/196987/when-the-butterfly-effect-took-flight/), MIT Technology Review
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [GSM8K](https://github.com/openai/grade-school-math)
