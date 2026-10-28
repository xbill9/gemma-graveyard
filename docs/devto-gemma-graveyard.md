---
title: "The Gemma Graveyard: Which Answers Gemma 4 E2B Loses at Each Quantization Rung"
published: false
description: "Gemma 4 E2B quantized from BF16 to Q2_K on one Tesla T4, every rung asked the same 250 GSM8K questions, and each answer buried at the rung where it first goes wrong. Q5_K_M and above hold the score; Q3_K_M loses 38; Q2_K gets 3 right. With thinking on, Q3_K_M wins back 17."
tags: devchallenge, hf26challenge, gemma, llamacpp
cover_image: https://raw.githubusercontent.com/xbill9/gemma-graveyard/main/docs/devto-cover.50dc7d50.jpg
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

This article provides a step by step guide to quantizing Gemma 4 E2B down a ladder of nine GGUF files with llama.cpp, asking every rung the same 250 maths questions on one Tesla T4, and burying each answer at the rung where it first goes wrong. Every count is computed by code and every row is committed.

https://github.com/xbill9/gemma-graveyard

A score per quantization level says how many answers a smaller file loses; a graveyard says which ones. Against the 229 of 250 questions BF16 gets right, Q8_0, Q6_K and Q5_K_M stay within 2, Q4_K_M and Q4_0 lose 5 and 7, Q3_K_M loses 38 and Q2_K gets 3 right. Above Q3_K_M most of the dead come back: 16 of the 23 questions that die between F16 and Q4_0 are right again at a lower rung. With thinking on, Q3_K_M climbs from 191 to 208.

---

## What I Built

Quantization is the most common thing done to an open-weight model before it ships to a small device. The usual report is one accuracy number per file size. Two files with the same score can still get different questions wrong, and a net loss hides its parts: Q4_0 here is 7 answers down on BF16, made of 15 questions lost and 8 gained.

The gemma-graveyard asks every rung of a quantization ladder the same questions at temperature 0, so each answer is the model's single most likely one, and records each answer. A question that BF16 gets right and a lower rung gets wrong gets a tombstone at the first rung where it dies. A question BF16 gets wrong and a lower rung gets right is a ghost. Ghosts measure how much a score moves by chance between two nearly identical models, so the article reports both.

For Día de los Muertos: the dead are remembered one at a time, by name. Each tombstone here carries the question's id, the right answer, the wrong answer the rung gave, and every lower rung where it rose again.

What the project delivers:

- `ladder.sh`: BF16, F16 and seven quantized GGUFs of Gemma 4 E2B, with bytes and sha256
- `graveyard.py run`: one rung, its own llama-server, 250 questions, answers graded in code
- `graveyard.py bury`: deaths, tombstones and ghosts per rung, computed in code
- `graveyard.py autopsy`: how each rung's answers fail (cut off, looping, writing into the thinking block)
- `graveyard.py repeat`: two runs of one rung compared question by question
- one run of the whole ladder plus Google's QAT Q4_0 on a Tesla T4, committed with its report
- a second pass with thinking on, over seven rungs from BF16 to Q3_K_M

---

## Demo

#### The Results in Brief

Q5_K_M is the best trade between speed and accuracy on the T4. It keeps BF16's score, 231 right against 229, at 39.0% of the bytes and 1.97 times BF16's decode speed. Q4_0 is the fastest rung at 2.34 times BF16, and costs 9 right answers against Q5_K_M for 1.19 times its speed. Q4_K_M sits between the two.

Every other rung is beaten on both counts by one of those three: Q8_0 and Q6_K are slower than Q5_K_M with fewer or equal right answers, and Q3_K_M is slower than Q4_0 with 31 fewer. Below Q4 the accuracy collapses: Q3_K_M loses 38 and Q2_K gets 3 right. With thinking on, the order holds: Q5_K_M 230, Q4_K_M 226 and Q4_0 219, against BF16's 227.

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
| `graveyard.py` | `run`, `bury`, `autopsy` and `repeat` |
| `docs/think_derived.py` | Thinking on against thinking off, question by question |
| `lazy_eval.py` | Prompt, answer extraction and grading, reused from Week 3 |
| `run.md` | Every command for the T4, in order, and how to resume after the VM stops |
| `runs/2026-10-27-ladder/run-ladder.sh` | Runs the nine rungs in order, skips finished ones, then buries |
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
TMPDIR=/opt1/tmp ./ladder.sh /opt1/models/gemma-4-E2B-it /opt1/graveyard-gguf /opt1/llama.cpp
rung	file	bytes	sha256
BF16	/opt1/graveyard-gguf/e2b-BF16.gguf	9311305152	9fe40af9a72302151a5abc28a5057c5fe335eb80145e795ba745085fc73a2ad6
F16	/opt1/graveyard-gguf/e2b-F16.gguf	9311305152	5ab927e4b24a248de319e70467fc74d4ab87347ee9d6fb3ee60b293b12198cf8
Q8_0	/opt1/graveyard-gguf/e2b-Q8_0.gguf	4967497152	6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a
Q6_K	/opt1/graveyard-gguf/e2b-Q6_K.gguf	3845346752	e800e0389f48646994d8ff6d3409dbf0c13b3fc217d0647b481385da33bd3d99
Q5_K_M	/opt1/graveyard-gguf/e2b-Q5_K_M.gguf	3630288320	f6c3f4395a9d3839535d8c14ae48ab784096945cfc4485005853eede915cb4a9
Q4_K_M	/opt1/graveyard-gguf/e2b-Q4_K_M.gguf	3427880384	dd9dc465bb535989205f158d77f28f6f8348e47d8ac8870920c152b5996cbc16
Q4_0	/opt1/graveyard-gguf/e2b-Q4_0.gguf	3360161216	2d84758474be87d0a74edfa98efbeb602484e26f2e5fa8695fd9641ad323c7b2
Q3_K_M	/opt1/graveyard-gguf/e2b-Q3_K_M.gguf	3201351104	b10c1f38c93635ccee1c7f382e03f5d2f4e9516087d7d62a73ca7de0f52693ab
Q2_K	/opt1/graveyard-gguf/e2b-Q2_K.gguf	2989088192	65850f1a31efed2013ed148adc5dc937e973a8e79fd55cdba31edd53369fddb2
```

The reference is BF16 because that is the checkpoint's own dtype, so the BF16 GGUF holds the trained weights unchanged. F16 is a rung.

---

#### Step 2 — Check What F16 Costs Before Running It

F16 has more mantissa than BF16 and less exponent range, so the cast can overflow or round small weights to zero. `f16_range.py` counts both on the checkpoint, one line per tensor and a total:

```text
python3 f16_range.py /opt1/models/gemma-4-E2B-it/model.safetensors > runs/2026-10-27-f16-range/out.txt
grep -E 'position_embedding_table|TOTAL' runs/2026-10-27-f16-range/out.txt
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
for i in 1 2; do
  ./graveyard.py run BF16 /opt1/graveyard-gguf/e2b-BF16.gguf data/heldout.jsonl runs/2026-10-27-determinism-$i \
    --llama /opt1/llama.cpp --limit 50
done
```

The two runs compared in code, question by question:

```text
0 mismatched extracted, 0 mismatched text, 46/50 both times
```

Every answer and every character of text matched. The same holds below BF16. `graveyard.py repeat` compares a second full run of four rungs with the ladder's:

```text
./graveyard.py repeat runs/2026-10-27-ladder runs/2026-10-28-repeat Q4_0 Q3_K_M Q8_0 F16 --common
{"rung": "Q4_0", "n": 250, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 222, "right_b": 222}
{"rung": "Q3_K_M", "n": 250, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 191, "right_b": 191}
{"rung": "Q8_0", "n": 250, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 227, "right_b": 227}
{"rung": "F16", "n": 250, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 229, "right_b": 229}
```

A changed answer at a lower rung comes from the weights.

---

#### Step 5 — Run the Ladder and Bury It

`runs/2026-10-27-ladder/run-ladder.sh` runs the nine rungs in order and skips any rung whose results are complete, so it picks up again after the VM's evening stop. Then `bury`:

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

```text
./graveyard.py autopsy runs/2026-10-27-ladder BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K
{"rung": "Q3_K_M", "cut_off": 49, "looped": 0, "thought": 0, "empty_content": 0, "empty_right_in_reasoning": 0, "dead": 42, "dead_cut_off": 34, "dead_looped": 0, "dead_empty_content": 0, "first_deaths": 26, "first_deaths_right_again_lower": 0}
{"rung": "Q2_K", "cut_off": 246, "looped": 228, "thought": 75, "empty_content": 63, "empty_right_in_reasoning": 2, "dead": 226, "dead_cut_off": 222, "dead_looped": 207, "dead_empty_content": 60, "first_deaths": 177, "first_deaths_right_again_lower": 0}
```

Every rung:

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

- **F16 to Q4_0, a different road through the same problem.** 23 first deaths; 16 of them are right again at some lower rung. Down to Q5_K_M most of the dead answers end normally with a wrong number: 4 of 12 were cut off. At Q4_K_M and Q4_0, 11 of 24 were. From F16 to Q5_K_M, ghosts (1 to 5 per rung) match deaths (1 to 4), so the score holds. From Q4_K_M down, deaths outnumber ghosts, 9 to 4 and 15 to 8.
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

Same number of deaths, same number of changed answers, one right answer apart. The graves differ: 6 of the 15 dead are the same question, 9 die only under QAT and 9 only under the plain Q4_0. QAT runs out of room less often: 2 of its deaths were cut off at 1,024 tokens, against 8 for the plain Q4_0.

---

#### Step 8 — Bury the Ladder Again With Thinking On

Gemma 4 can think before it answers. `--think` turns it on; the thinking goes to `reasoning_content` and only `content` is graded. Thinking needs room, so this pass allows 4,096 tokens in an 8,192-token context. It covers seven rungs, BF16 to Q3_K_M; F16, Q2_K and QAT Q4_0 were not run with thinking on. Each rung is buried against BF16 with thinking on, which gets 227 right. A second BF16 run of 50 questions matched the first exactly, 47 right both times.

```text
./graveyard.py bury runs/2026-10-28-think BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M --json graveyard.json
{"rung": "BF16", "n": 250, "right": 227, "errors": 0, "truncated": 0, "median_decode_tps": 34.5, "mean_tokens": 842.2, "dead": 0, "ghosts": 0, "answer_changed": 0, "bytes": 9311305152, "gpu_mib_loaded": 4699, "gpu_mib_before": 0}
{"rung": "Q3_K_M", "n": 250, "right": 208, "errors": 0, "truncated": 11, "median_decode_tps": 68.0, "mean_tokens": 1320.1, "dead": 26, "ghosts": 7, "answer_changed": 42, "bytes": 3201351104, "gpu_mib_loaded": 1517, "gpu_mib_before": 0}
```

`docs/think_derived.py` lines the two passes up question by question:

```text
python3 docs/think_derived.py | tail -3
 "q3_dead_cut_off_with_thinking_off": 34,
 "of_them_right_with_thinking_on": 21
}
```

| Rung | Right, thinking off | Right, thinking on | Dead, off | Dead, on | Dead in both |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BF16 | 229 | 227 | — | — | — |
| Q8_0 | 227 | 🥇 230 | 4 | 2 | 0 |
| Q6_K | 229 | 🥇 230 | 4 | 6 | 0 |
| Q5_K_M | 231 | 🥇 230 | 3 | 6 | 0 |
| Q4_K_M | 224 | 226 | 9 | 9 | 1 |
| Q4_0 | 222 | 219 | 15 | 15 | 3 |
| Q3_K_M | 191 | 208 | 42 | 26 | 12 |

Three things change:

- **Q3_K_M recovers most of its loss.** It gains 17 right answers. Of the 34 questions it lost by running out of room at 1,024 tokens, 21 are right with thinking on and 4,096 tokens. Both the thinking and the larger limit changed at once, so this run cannot say which one did it.
- **Above Q3_K_M the score holds and the graves move.** Each rung stays within 3 answers of its thinking-off score, but the questions it loses are almost all new: 0 of the deaths at Q8_0, Q6_K and Q5_K_M are shared between the two passes, 1 of 9 at Q4_K_M and 3 of 15 at Q4_0. BF16 itself gets 218 questions right in both modes.
- **The cost is time.** Answers use 2.08 to 2.18 times as many tokens, decoding runs at 0.79 to 0.93 of the thinking-off speed, and each rung takes 2.22 to 2.70 times as long on the T4.

---

#### Compare and Contrast

| | Q5_K_M | Q4_K_M | Q4_0 | QAT Q4_0 | Q3_K_M |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Right of 250 (BF16: 229) | 🥇 231 | 🥈 224 | 🥉 222 | 221 | 191 |
| Dead | 🥇 3 | 🥈 9 | 🥉 15 | 🥉 15 | 42 |
| GPU MiB | 1,897 | 1,705 | 🥉 1,639 | 🥈 1,629 | 🥇 1,489 |
| Median tok/s on the T4 | 78.7 | 🥉 84.1 | 🥇 93.6 | 🥈 87.8 | 74.2 |
| Answers cut off | 🥇 3 | 🥈 6 | 10 | 🥈 6 | 49 |
| Verdict | 🟢 same score as BF16 | 🟢 small loss | ⚠️ 15 dead | ⚠️ 15 dead, fewer cut off | ❌ 38 fewer right |

---

#### So, Which One?

To ship Gemma 4 E2B on a small GPU with BF16's score, take Q5_K_M: 39.0% of the bytes, 231 right against 229, and 1.97 times BF16's decode speed on the T4 (arithmetic from the medians). For the fastest file that stays within 7 answers, take Q4_0 or Q4_K_M. Below Q4, the losses stop being a few flipped answers: Q3_K_M saves 150 MiB of GPU memory over Q4_0 and costs 31 right answers, and Q2_K stops answering. With thinking on and 4,096 tokens, Q3_K_M gets 208 right, 11 short of Q4_0, at more than twice the time per question.

---

#### Prior Work

- **Prompt, answer extraction and grading** (`lazy_eval.py`) and **the 250-question GSM8K split** (`data/`, seed 0): from my Week 3 entry, [lazy-gemma](https://github.com/xbill9/lazy-gemma), whose grader came from my earlier Gemma 4 TPU work.
- **Open source used**: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) for conversion, quantization and serving; Gemma 4 E2B and Google's QAT Q4_0 GGUF (Gemma terms); [GSM8K](https://github.com/openai/grade-school-math).

---

## Why Does Open Innovation Matter?

A graveyard needs the same model at nine sizes, run the same way.

- **The weights are on the disk, so the ladder is built here.** `ladder.sh` produces every rung from one BF16 file with the open converter and quantizer, and `ladder.tsv` records each file's sha256. A hosted API serves whichever quantization its provider chose, and it can change without notice.
- **The same answer twice.** Greedy decoding on a fixed llama.cpp build gave identical text on every repeated run, BF16 to Q3_K_M, so every tombstone counts. That check needs control of the server, the seed and the build.
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

The goal of this project was to find which answers Gemma 4 E2B loses at each quantization level, one question at a time. The key was a deterministic run: greedy decoding on one llama.cpp build, checked to give identical text twice at five rungs, so every changed answer comes from the weights. The results were:

- 🟢 Q8_0, Q6_K and Q5_K_M stay within 2 right answers of BF16's 229 of 250; Q5_K_M scores 231 at 39.0% of the bytes.
- 🟢 Every rung fits on the T4, BF16 included: all 36 layers on the GPU, 4,671 MiB in use, with the embedding tables in CPU memory.
- 🟢 Above Q3_K_M, 16 of the 23 questions that die are right again at a lower rung.
- ⚠️ Q4_K_M and Q4_0 lose 5 and 7, with 9 and 15 deaths.
- ⚠️ Google's QAT Q4_0 scores 221 against the plain Q4_0's 222; the two share 6 of their 15 deaths.
- 🟢 Q4_0, Q3_K_M, Q8_0 and F16 run a second time give the same text on all 250 questions.
- ⚠️ With thinking on, rungs above Q3_K_M keep their score but lose different questions; at Q4_0, 3 of 15 deaths are shared.
- ⚠️ Thinking on costs 2.1 times the tokens and 2.2 to 2.7 times the time per rung.
- ❌ Q3_K_M loses 38; 34 of its 42 deaths are cut off at 1,024 tokens. With thinking on and 4,096 tokens it loses 19.
- ❌ Q2_K gets 3 right: 246 answers hit the token limit, 228 loop, and 75 write into a thinking block with thinking off.

Scope: one Tesla T4 on GCE, llama.cpp `b11699`, 250 GSM8K test questions per rung, greedy. Thinking off with a 1,024-token limit on all ten models; thinking on with a 4,096-token limit on seven rungs, BF16 to Q3_K_M. Determinism checked at BF16, F16, Q8_0, Q4_0 and Q3_K_M. No importance matrix, none of the other 1,069 GSM8K test questions, and no other machine. Speeds compare rungs on this T4 only.

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
