---
title: "The Gemma Graveyard: Which Answers Gemma 4 E2B Loses at Each Quantization Rung"
published: false
description: "Gemma 4 E2B quantized from BF16 to Q2_K on one Tesla T4, every rung asked the same 250 GSM8K questions, and each answer buried at the rung where it first goes wrong. Q5_K_M and above hold the score; Q3_K_M loses 38; Q2_K gets 3 right."
tags: devchallenge, hf26challenge, gemma, llamacpp
cover_image: https://raw.githubusercontent.com/xbill9/gemma-graveyard/main/docs/devto-cover.50dc7d50.jpg
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

This article provides a step by step guide to quantizing Gemma 4 E2B down a ladder of nine GGUF files with llama.cpp, asking every rung the same 250 maths questions on one Tesla T4, and burying each answer at the rung where it first goes wrong. Every count is computed by code and every row is committed.

https://github.com/xbill9/gemma-graveyard

**A score per quantization level says how many answers a smaller file loses. A graveyard says which ones. Against the 229 of 250 questions BF16 gets right: Q8_0, Q6_K and Q5_K_M stay within 2 of it, Q4_K_M and Q4_0 lose 5 and 7, Q3_K_M loses 38, and Q2_K gets 3 right. From Q8_0 to Q5_K_M the score holds while 11 to 13 answers change, because questions die and come back at about the same rate. 16 of the 23 questions that die between F16 and Q4_0 are right again at some lower rung. Q3_K_M dies differently: 34 of its 42 deaths run out of room at 1,024 tokens. Q2_K collapses into repeated lines and, with thinking off, writes 75 answers into its thinking block. Google's quantization-aware Q4_0 scores 221 against the plain Q4_0's 222, and of the 15 questions each one loses, only 6 are the same.**

---

## What I Built

Quantization is the most common thing done to an open-weight model before it ships to a small device. The usual report is one accuracy number per file size. Two files with the same score can still get different questions wrong, and a net loss hides its parts: Q4_0 here is 7 answers down on BF16, made of 15 questions lost and 8 gained.

The gemma-graveyard asks every rung of a quantization ladder the same questions, greedy, and records each answer. A question that BF16 gets right and a lower rung gets wrong gets a tombstone at the first rung where it dies. A question BF16 gets wrong and a lower rung gets right is a ghost. Ghosts measure how much a score moves by chance between two nearly identical models, so the article reports both.

For Día de los Muertos: the dead are remembered one at a time, by name. Each tombstone here carries the question's id, the right answer, the wrong answer the rung gave, and every lower rung where it rose again.

What the project delivers:

- `ladder.sh`: BF16, F16 and seven quantized GGUFs of Gemma 4 E2B, with bytes and sha256
- `graveyard.py run`: one rung, its own llama-server, 250 questions, answers graded in code
- `graveyard.py bury`: deaths, tombstones and ghosts per rung, computed in code
- `graveyard.py autopsy`: how each rung's answers fail (cut off, looping, writing into the thinking block)
- one run of the whole ladder plus Google's QAT Q4_0 on a Tesla T4, committed with its report

---

## Demo

The cover is the graveyard. One row per rung, BF16 at the top and Q2_K at the bottom, one cell per question: marigold where the rung answers right, a grey headstone where it answers wrong. `docs/make_cover.py` draws it from the committed rows.

The ladder, from `graveyard.py bury`, on 250 GSM8K test questions:

| Rung | % of BF16 bytes | Right | Dead | Ghosts | Median tok/s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BF16 | 100.0 | 229 | 0 | 0 | 40.0 |
| F16 | 100.0 | 229 | 1 | 1 | 39.6 |
| Q8_0 | 53.3 | 227 | 4 | 2 | 64.3 |
| Q6_K | 41.3 | 229 | 4 | 4 | 70.8 |
| Q5_K_M | 39.0 | 🥇 231 | 3 | 5 | 78.7 |
| Q4_K_M | 36.8 | 224 | 9 | 4 | 84.1 |
| Q4_0 | 36.1 | 222 | 15 | 8 | 🥇 93.6 |
| Q3_K_M | 34.4 | 191 | 42 | 4 | 74.2 |
| Q2_K | 32.1 | 3 | 226 | 0 | 74.0 |

**Dead**: BF16 right, this rung wrong. **Ghosts**: BF16 wrong, this rung right.

---

## Code

https://github.com/xbill9/gemma-graveyard

| Path | What |
| :--- | :--- |
| `ladder.sh` | Converts the checkpoint to BF16 and F16 GGUF, quantizes seven rungs from BF16, writes `ladder.tsv` |
| `f16_range.py` | Counts the weights that overflow or underflow when BF16 is cast to F16 |
| `graveyard.py` | `run`, `bury` and `autopsy` |
| `lazy_eval.py` | Prompt, answer extraction and grading, reused from Week 3 |
| `run.md` | Every command for the T4, in order, and how to resume after the VM stops |
| `runs/2026-10-27-ladder/` | `REPORT.md`, the bury and autopsy tables, per-question tombstones |

---

## How I Built It

#### At This Point You Should Have…

- A GCE `n1-standard-2` in `us-west2-b` with one Tesla T4, 15,360 MiB, driver 615.71.10, 2 vCPU and 7,436 MiB of RAM
- llama.cpp built with CUDA for compute capability 7.5, at `b11699`
- `google/gemma-4-E2B-it` from Hugging Face, the BF16 checkpoint, text model only
- `google/gemma-4-E2B-it-qat-q4_0-gguf`, Google's quantization-aware Q4_0 file
- A clone: `git clone https://github.com/xbill9/gemma-graveyard`

---

#### Step 1 — Build the Ladder

`ladder.sh` converts the checkpoint twice, to BF16 and to F16, then quantizes seven rungs from the BF16 file with plain `llama-quantize`, with no importance matrix. That is the default a user gets.

```text
rung	file	bytes	sha256
BF16	/opt1/graveyard-gguf/e2b-BF16.gguf	9311305152	9fe40af9a72302151a5abc28a5057c5fe335eb80145e795ba745085fc73a2ad6
F16	/opt1/graveyard-gguf/e2b-F16.gguf	9311305152	5ab927e4b24a248de319e70467fc74d4ab87347ee9d6fb3ee60b293b12198cf8
Q8_0	/opt1/graveyard-gguf/e2b-Q8_0.gguf	4967497152	6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a
Q4_0	/opt1/graveyard-gguf/e2b-Q4_0.gguf	3360161216	2d84758474be87d0a74edfa98efbeb602484e26f2e5fa8695fd9641ad323c7b2
Q2_K	/opt1/graveyard-gguf/e2b-Q2_K.gguf	2989088192	65850f1a31efed2013ed148adc5dc937e973a8e79fd55cdba31edd53369fddb2
```

The reference is BF16 because that is the checkpoint's own dtype, so the BF16 GGUF holds the trained weights unchanged. F16 is a rung.

---

#### Step 2 — Check What F16 Costs Before Running It

F16 has more mantissa than BF16 and less exponent range, so the cast can overflow or round small weights to zero. `f16_range.py` counts both on the checkpoint:

```text
{"tensor": "model.vision_tower.patch_embedder.position_embedding_table", "n": 15728640, "overflow": 0, "underflow": 5507733, "subnormal": 834716, "absmax": 0.431640625}
{"tensor": "TOTAL", "tensors": 2011, "n": 5123178979, "overflow": 0, "underflow": 5523265, "subnormal": 7013885, "absmax": 900.0}
```

No weight overflows. Of the 5,523,265 that underflow, 5,507,733 are in the vision tower's position table, which the text GGUF does not carry.

---

#### Step 3 — Ask One Rung 250 Questions

`graveyard.py run` starts its own `llama-server` on one GGUF and asks each question once: temperature 0, one request at a time, no prompt cache, thinking off, 1,024 max tokens. Only the reply's `content` is graded, in code: the number after `Answer:`, compared with the reference.

```text
./graveyard.py run BF16 /opt1/graveyard-gguf/e2b-BF16.gguf data/heldout.jsonl runs/2026-10-27-ladder --llama /opt1/llama.cpp
{"rung": "BF16", "n": 250, "right": 229, "errors": 0, "truncated": 4, "median_decode_tps": 40.0, "mean_tokens": 391.2}
```

The whole model fits on the T4 at BF16:

```text
load_tensors: offloaded 36/36 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   768.00 MiB
load_tensors:        CUDA0 model buffer size =  4384.87 MiB
```

🔎 **Tip: Gemma 4 E2B keeps its embedding tables in CPU memory.** The per-layer embedding table holds 2,348,810,240 of the checkpoint's weights, and llama.cpp leaves it memory-mapped on the CPU with `-ngl 99`. So each rung's GPU memory is about half its file size: 4,671 MiB in use at BF16 for a 9.3 GB file, 1,639 MiB at Q4_0, 1,287 MiB at Q2_K.

---

#### Step 4 — Prove the Same Rung Gives the Same Answers

A tombstone means something only if the same model on the same question always gives the same answer. BF16 on 50 questions, twice:

```text
0 mismatched extracted, 0 mismatched text, 46/50 both times
```

Every answer and every character of text matched. A changed answer at a lower rung comes from the weights.

---

#### Step 5 — Run the Ladder and Bury It

`run-ladder.sh` runs the nine rungs in order and skips any rung whose results are complete, so it picks up again after the VM's evening stop. Then `bury`:

```text
./graveyard.py bury runs/2026-10-27-ladder BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K --json graveyard-ladder.json
{"first_deaths_by_rung": {"Q2_K": 177, "Q3_K_M": 26, "Q4_0": 9, "Q8_0": 4, "Q5_K_M": 2, "Q4_K_M": 6, "Q6_K": 1, "F16": 1}, "tombstones": 226, "ghosts": 11}
```

226 tombstones and 11 ghosts. Per-question detail is in `graveyard-ladder.json`. The one question that dies at F16:

```text
{'id': 'test-687', 'died_at': 'F16', 'answer': '13', 'said': 17.0, 'right_again_at': ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M']}
```

"One twin is 7 years older than one triplet. If their combined ages are 44, how old is one of the twins?" BF16 answers 13, from two twins and three triplets. F16 adds up two twins and one triplet, and answers 17. Four lower rungs answer 13 again.

---

#### Step 6 — Autopsy: How Each Rung Dies

`graveyard.py autopsy` counts, per rung, answers cut off at 1,024 tokens, answers that repeat one line five or more times, and answers that put text in the thinking block:

| Rung | Cut off | Looped | Thinking unprompted | Dead | Dead, cut off |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F16 | 4 | 0 | 0 | 1 | 0 |
| Q8_0 | 9 | 0 | 0 | 4 | 2 |
| Q6_K | 4 | 0 | 0 | 4 | 1 |
| Q5_K_M | 3 | 0 | 0 | 3 | 1 |
| Q4_K_M | 6 | 0 | 0 | 9 | 3 |
| Q4_0 | 10 | 0 | 0 | 15 | 8 |
| Q3_K_M | 49 | 0 | 0 | 42 | 34 |
| Q2_K | 246 | 228 | 75 | 226 | 222 |

Three kinds of death:

- **F16 to Q4_0, a different road through the same problem.** 23 first deaths; 16 of them are right again at some lower rung. The answers end normally with a wrong number. Down to Q5_K_M, ghosts (1 to 5 per rung) match deaths (1 to 4), so the score holds. From Q4_K_M down, deaths outnumber ghosts, 9 to 4 and 15 to 8.
- **Q3_K_M, out of room.** 34 of its 42 deaths were cut off at 1,024 tokens, still working, with no repeated lines. BF16 used a median 466.5 tokens on the 44 questions Q3_K_M got wrong while cut off.
- **Q2_K, collapse.** 246 of 250 answers run to the limit and 228 repeat a line five or more times. With thinking off, 75 answers open a thinking block anyway, and in 63 of them the whole answer lands there, leaving `content` empty. No other rung writes into the thinking block at all.

🔎 **Tip: give two buries in one directory two names.** `bury` writes its per-question tombstones to `graveyard.json`; `--json` picks the file, so the QAT comparison in Step 7 keeps the ladder's file intact.

---

#### Step 7 — Google's QAT Q4_0 Beside the Plain Q4_0

Google publishes a Q4_0 GGUF of Gemma 4 E2B trained for 4-bit (quantization-aware training). Its weights differ from the base checkpoint, so it sits beside the ladder, against the same BF16 reference:

```text
./graveyard.py bury runs/2026-10-27-ladder BF16 QAT_Q4_0 Q4_0 --json graveyard-qat.json
{"rung": "QAT_Q4_0", "n": 250, "right": 221, "errors": 0, "truncated": 6, "median_decode_tps": 87.8, "mean_tokens": 391.9, "dead": 15, "ghosts": 7, "answer_changed": 27, "bytes": 3349516256, "gpu_mib_loaded": 1629, "gpu_mib_before": 0}
{"rung": "Q4_0", "n": 250, "right": 222, "errors": 0, "truncated": 10, "median_decode_tps": 93.6, "mean_tokens": 430.4, "dead": 15, "ghosts": 8, "answer_changed": 27, "bytes": 3360161216, "gpu_mib_loaded": 1639, "gpu_mib_before": 0}
```

Same deaths, same number of changed answers, one right answer apart. The graves differ: 6 of the 15 dead are the same question, 9 die only under QAT and 9 only under the plain Q4_0. QAT runs out of room less often: 2 of its deaths were cut off at 1,024 tokens, against 8 for the plain Q4_0.

---

#### Compare and Contrast

| | Q5_K_M | Q4_K_M | Q4_0 | QAT Q4_0 | Q3_K_M |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Right of 250 (BF16: 229) | 🥇 231 | 🥈 224 | 222 | 221 | 191 |
| Dead | 🥇 3 | 🥈 9 | 15 | 15 | 42 |
| GPU MiB | 1,897 | 1,705 | 1,639 | 🥈 1,629 | 🥇 1,489 |
| Median tok/s on the T4 | 78.7 | 🥉 84.1 | 🥇 93.6 | 🥈 87.8 | 74.2 |
| Answers cut off | 🥇 3 | 🥈 6 | 10 | 🥈 6 | 49 |
| Verdict | 🟢 same score as BF16 | 🟢 small loss | ⚠️ 15 dead | ⚠️ 15 dead, fewer cut off | ❌ 38 fewer right |

---

#### So, Which One?

To ship Gemma 4 E2B on a small GPU with BF16's score, take Q5_K_M: 39.0% of the bytes, 231 right against 229, and 1.97 times BF16's decode speed on the T4 (arithmetic from the medians). For the fastest file that stays within 7 answers, take Q4_0 or Q4_K_M. Below Q4, the losses stop being a few flipped answers: Q3_K_M saves 150 MiB of GPU memory over Q4_0 and costs 31 right answers, and Q2_K stops answering.

---

#### Prior Work

- **Prompt, answer extraction and grading** (`lazy_eval.py`) and **the 250-question GSM8K split** (`data/`, seed 0): from my Week 3 entry, [lazy-gemma](https://github.com/xbill9/lazy-gemma), whose grader came from my earlier Gemma 4 TPU work.
- **Open source used**: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) for conversion, quantization and serving; Gemma 4 E2B and Google's QAT Q4_0 GGUF (Gemma terms); [GSM8K](https://github.com/openai/grade-school-math).

---

## Why Does Open Innovation Matter?

A graveyard needs the same model at nine sizes, run the same way.

- **The weights are on the disk, so the ladder is built here.** `ladder.sh` produces every rung from one BF16 file with the open converter and quantizer, and `ladder.tsv` records each file's sha256. A hosted API serves whichever quantization its provider chose, and it can change without notice.
- **The same answer twice.** Greedy decoding on a fixed llama.cpp build gave identical text on two runs of 50 questions, so every tombstone counts. That check needs control of the server, the seed and the build.
- **The failure is visible.** Q2_K's answers in the thinking block show up only because llama-server returns `reasoning_content` and `content` separately and every row is kept.
- **No per-token bill.** Ten models times 250 questions ran on one rented T4.

---

## The Spirit of It

Every row in the graveyard is a grade-school maths problem, and each tombstone keeps its own epitaph:

- **test-687**, twins and triplets, answer 13. Died at F16, said 17. Rose again at Q8_0, Q6_K, Q5_K_M and Q4_K_M.
- **test-2**, a house flipped for profit, answer 70,000. Died at Q4_0, said 120,000. Never rose.
- **test-564**, Judy's two weeks of reading, answer 240. Died at Q8_0, said 3. Rose again at every rung from Q6_K to Q4_0.

Most graves above Q3_K_M have someone come back to them: 16 of 23.

---

## Prize Categories

**Gemma**: Gemma 4 E2B, quantized locally with llama.cpp at nine levels and compared with Google's quantization-aware Q4_0 release, served on one Tesla T4.

---

## Summary

The goal of this project was to find which answers Gemma 4 E2B loses at each quantization level, one question at a time. The key was a deterministic run: greedy decoding on one llama.cpp build, checked to give identical text twice, so every changed answer comes from the weights. The results were:

- 🟢 Q8_0, Q6_K and Q5_K_M stay within 2 right answers of BF16's 229 of 250; Q5_K_M scores 231 at 39.0% of the bytes.
- 🟢 Every rung fits on the T4, BF16 included: all 36 layers on the GPU, 4,671 MiB in use, with the embedding tables in CPU memory.
- 🟢 Above Q3_K_M, 16 of the 23 questions that die are right again at a lower rung.
- ⚠️ Q4_K_M and Q4_0 lose 5 and 7, with 9 and 15 deaths.
- ⚠️ Google's QAT Q4_0 scores 221 against the plain Q4_0's 222; the two share 6 of their 15 deaths.
- ❌ Q3_K_M loses 38; 34 of its 42 deaths are cut off at 1,024 tokens.
- ❌ Q2_K gets 3 right: 246 answers hit the token limit, 228 loop, and 75 write into a thinking block with thinking off.

Scope: one Tesla T4 on GCE, llama.cpp `b11699`, one greedy run of 250 GSM8K test questions per rung with thinking off and a 1,024-token limit; determinism checked on BF16 only. No importance matrix, no thinking-on pass, and no other machine. Speeds compare rungs on this T4 only.

The strategy for burying Gemma 4's answers one quantization rung at a time was validated with an incremental step by step approach.

---

#### References

- [Hacktoberfest Open-Source AI Challenge, Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)
- [gemma-graveyard on GitHub](https://github.com/xbill9/gemma-graveyard)
- [Run report: runs/2026-10-27-ladder/REPORT.md](https://github.com/xbill9/gemma-graveyard/blob/main/runs/2026-10-27-ladder/REPORT.md)
- [lazy-gemma, the Week 3 entry](https://github.com/xbill9/lazy-gemma)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [GSM8K](https://github.com/openai/grade-school-math)
