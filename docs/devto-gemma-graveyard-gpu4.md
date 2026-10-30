---
title: "Four GPUs, One Quantization Ladder: Gemma 4 E2B From a 4 GB Laptop to an NVIDIA L4"
published: false
description: "Gemma 4 E2B from BF16 to Q2_K on a Tesla T4, an NVIDIA L4, a GTX 1650 laptop and an RTX 3050 desktop, with byte-identical GGUF files and the same llama.cpp build. The ladder keeps its shape on all four cards, 218 questions die on all four, and any two GPUs grade 2 to 4 questions differently at BF16."
tags: gemma, llamacpp, quantization, nvidia
cover_image: https://raw.githubusercontent.com/xbill9/gemma-graveyard/main/docs/devto-cover-gpu4.27a7f25f.jpg
---

This article provides a step by step guide to running Gemma 4 E2B's quantization ladder, BF16 down to Q2_K, on four NVIDIA GPUs with the same nine GGUF files and the same llama.cpp build, and comparing the four cards question by question.

https://github.com/xbill9/gemma-graveyard

The [first article](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea) asked every rung of the ladder the same 250 GSM8K questions on a Tesla T4. The [second](https://dev.to/gde/performance-vs-quality-gemma-4-e2b-quantization-on-an-nvidia-l4-and-a-tesla-t4-4b90) repeated it on an NVIDIA L4 in AWS. This one adds two consumer cards: a GTX 1650 Ti laptop GPU with 4 GB and an RTX 3050 desktop card with 8 GB. All four run llama.cpp `b11699` on files with the same sha256, and each card gives identical text when BF16 is run twice.

---

#### The Results in Brief

The ladder keeps its shape on all four GPUs. Against each card's own BF16, F16 to Q5_K_M lose 1 to 8 answers, Q4_K_M and Q4_0 lose 9 to 19, Q3_K_M loses 42 to 49, and Q2_K gets 3 to 6 right out of 250.

The counts agree; the questions move. Any two GPUs grade 2 to 4 questions differently at BF16 and 12 to 22 at Q4_0. No question is lost at F16 or Q8_0 on all four cards, while 216 are lost at Q2_K on all four. 218 questions die at some rung on every GPU, and 147 of them die first at the same rung on every GPU.

The 4 GB laptop runs every quantized rung entirely on its GPU. BF16 and F16 do not fit, so llama.cpp splits them between GPU and CPU, and they decode at 9.2 and 9.6 tokens per second against 45.9 for Q8_0 on the same card.

The cover lays the four GPUs' grades over each other. One row per rung, BF16 at the top and Q2_K at the bottom, one cell per question: marigold where all four GPUs answer right, grey where all four answer wrong, violet where they disagree. `docs/make_cover_gpu4.py` draws it from the committed rows.

---

#### What Changes and What Stays

| | Tesla T4 | NVIDIA L4 | GTX 1650 Ti | RTX 3050 |
| :--- | :--- | :--- | :--- | :--- |
| Where | GCE `n1-standard-2` | AWS `g6.xlarge` | laptop | desktop |
| GPU memory | 15,360 MiB | 23,034 MiB | 4,096 MiB | 8,192 MiB |
| Compute capability | 7.5 | 8.9 | 7.5 | 8.6 |
| Host CPU | 2 vCPU | AMD EPYC 7R13 | Intel i7-10750H | Intel i7-5820K |
| Driver | 615.71.10 | 595.112.07 | 615.71.10 | 615.78.08 |

The same on all four: llama.cpp `b11699` with `GGML_CUDA=ON`, built for each card's own compute capability; the nine GGUFs and Google's QAT Q4_0 file, byte for byte; the 250 questions in `data/heldout.jsonl`; temperature 0, one request at a time, thinking off, 1,024 max tokens, `llama-server -c 4096 -np 1 --seed 0`. Every rung runs with `-ngl 99` except BF16 and F16 on the GTX 1650 (Step 4).

---

#### At This Point You Should Have…

- An NVIDIA GPU with at least 4 GB and a CUDA toolkit
- About 58 GB of free disk for the checkpoint, the nine GGUFs and the QAT file
- A clone: `git clone https://github.com/xbill9/gemma-graveyard`
- The T4 run's committed results in `runs/2026-10-27-ladder/`

---

#### Step 1 — Build llama.cpp at the Same Tag

On the RTX 3050:

```text
git -C ~/llama.cpp worktree add --detach /android/graveyard/llama.cpp b11699
cd /android/graveyard/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12 --target llama-server llama-quantize
```

The one build flag that changes per card is `CMAKE_CUDA_ARCHITECTURES`: 75 for the T4 and the GTX 1650, 89 for the L4, 86 for the RTX 3050.

```text
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
name, memory.total [MiB], compute_cap, driver_version
NVIDIA GeForce RTX 3050 OEM, 8192 MiB, 8.6, 615.78.08
```

🔎 **Tip: keep the pinned build in its own worktree.** The desktop already had a newer llama.cpp checkout (`b11983`). `git worktree add --detach <dir> b11699` gives a second working copy at the tag from the same clone, so the measured build and the everyday build sit side by side.

---

#### Step 2 — Rebuild the Ladder and Check It Against the T4

`ladder.sh` converts the checkpoint to BF16 and F16 and quantizes seven rungs from BF16. Each run script then compares every file's size and sha256 with the T4's `ladder.tsv`:

```text
diff <(cut -f1,3,4 runs/2026-10-27-ladder/ladder.tsv) <(cut -f1,3,4 runs/2026-10-29-3050/ladder.tsv) && echo "ladder matches T4 sha256"
ladder matches T4 sha256
```

The GTX 1650 and the RTX 3050 both print the same line, and Google's QAT Q4_0 file matches too (sha256 `fa401b55…`). Across the four machines, the GGUF bytes are identical; the card, its driver, its CUDA version and the kernels built for it are what differ.

🔎 **Tip: the converter writes the checkpoint directory into the GGUF.** `convert_hf_to_gguf.py` copies the model card and takes the model's name from the directory it reads. A checkpoint taken straight from the Hugging Face cache snapshot has neither the `README.md` nor the name `gemma-4-E2B-it`, and gives different bytes. Download with `hf download google/gemma-4-E2B-it --local-dir <dir>/gemma-4-E2B-it`.

---

#### Step 3 — Prove Each Card Repeats Itself

BF16 on 50 questions, twice, on every GPU:

```text
./graveyard.py repeat runs/2026-10-29-3050/determinism-1 runs/2026-10-29-3050/determinism-2 BF16
{"rung": "BF16", "n": 50, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 46, "right_b": 46}
```

The T4, the L4 and the GTX 1650 give the same result: 0 mismatched characters and 46 of 50 right both times. Each card on its own is deterministic, so a difference between two cards comes from the GPU and its CUDA stack.

---

#### Step 4 — Fit the 16-bit Rungs on 4 GB

BF16 and F16 need a 4,384.87 MiB GPU buffer for their 36 layers (from the T4's load log), and the GTX 1650 has 4,096 MiB. With `-ngl 99` the load fails. `graveyard.py run --ngl auto` leaves `-ngl` off, and llama.cpp's fit step places what it can:

```text
common_params_fit_impl:   - CUDA0 (NVIDIA GeForce GTX 1650 Ti with Max-Q Design): 14 layers,   2546 MiB used,   1125 MiB free
load_tensors: offloaded 14/36 layers to GPU
load_tensors:        CUDA0 model buffer size =  2407.92 MiB
```

Every other rung fits whole. Q8_0 on the same card:

```text
load_tensors: offloaded 36/36 layers to GPU
load_tensors:        CUDA0 model buffer size =  2342.30 MiB
```

The split changes speed and leaves the grades in line with the other GPUs: the GTX 1650's BF16 scores 229, the same as the T4 and the L4, and grades 2 to 4 questions differently from each of them. The 8 GB RTX 3050 holds BF16 whole at 4,651 MiB.

---

#### Step 5 — Run the Ladder on Each Card and Score It

Each run script runs the nine rungs and QAT Q4_0, then scores every rung against that card's own BF16:

```text
./graveyard.py bury runs/2026-10-29-3050 BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K --json graveyard-ladder.json
```

Right answers, out of 250:

| Rung | T4 | L4 | GTX 1650 | RTX 3050 |
| :--- | :--- | :--- | :--- | :--- |
| BF16 | 229 | 229 | 229 | 232 |
| F16 | 229 | 229 | 228 | 232 |
| Q8_0 | 227 | 228 | 229 | 226 |
| Q6_K | 229 | 227 | 226 | 227 |
| Q5_K_M | 231 | 228 | 226 | 226 |
| Q4_K_M | 224 | 221 | 222 | 221 |
| Q4_0 | 222 | 220 | 216 | 218 |
| Q3_K_M | 191 | 186 | 183 | 191 |
| Q2_K | 3 | 6 | 5 | 3 |

Lost, against the same card's BF16:

| Rung | T4 | L4 | GTX 1650 | RTX 3050 |
| :--- | :--- | :--- | :--- | :--- |
| F16 | 1 | 1 | 2 | 1 |
| Q8_0 | 4 | 3 | 2 | 7 |
| Q6_K | 4 | 6 | 7 | 8 |
| Q5_K_M | 3 | 6 | 6 | 8 |
| Q4_K_M | 9 | 14 | 12 | 14 |
| Q4_0 | 15 | 16 | 19 | 16 |
| Q3_K_M | 42 | 45 | 49 | 44 |
| Q2_K | 226 | 225 | 225 | 229 |

**Lost**: right at BF16, wrong at this rung, on the same GPU.

The failures fall into the same kinds on every card. Q3_K_M runs out of room: 33 to 36 of its losses on each GPU are answers cut off at 1,024 tokens. Q2_K loops: 208 to 228 of its 250 answers repeat a line five or more times, and 246 to 249 hit the token cap.

---

#### Step 6 — Line the Four GPUs Up, Question by Question

`docs/gpu4_derived.py` rebuilds the T4's grid from its committed results and reads the other three from their rows, then compares all six pairs of GPUs at every rung:

```text
python3 docs/gpu4_derived.py > runs/2026-10-29-3050/four-gpus.txt
```

| Rung | Graded differently, any two GPUs | Lost on all four |
| :--- | :--- | :--- |
| BF16 | 2 to 4 | — |
| F16 | 3 to 5 | 0 |
| Q8_0 | 3 to 9 | 0 |
| Q6_K | 4 to 10 | 2 |
| Q5_K_M | 7 to 10 | 1 |
| Q4_K_M | 9 to 15 | 5 |
| Q4_0 | 12 to 22 | 5 |
| Q3_K_M | 26 to 34 | 20 |
| Q2_K | 6 to 9 | 216 |

**Graded differently**: right on one GPU and wrong on the other, at that rung, for the closest and the furthest of the six pairs.

- **The light rungs lose different questions on each card.** F16 and Q8_0 lose 1 to 7 answers per GPU, and none of them is lost on all four.
- **The deep losses are shared.** Q3_K_M loses 20 questions on all four GPUs and Q2_K 216.
- **Most of the graveyard is common ground.** Each GPU has 226 questions that die at some rung, the RTX 3050 230. 218 die on all four, and 147 die first at the same rung on all four. One question dies only on the RTX 3050 (`test-358`) and one only on the GTX 1650 (`test-147`).
- **BF16 itself moves a few answers.** 227 questions are right at BF16 on all four cards and 18 wrong on all four. Against the L4, the RTX 3050's BF16 changes the text of 31 answers out of 250, the extracted number of 4 and the grade of 3.

---

#### Step 7 — Google's QAT Q4_0 on Four Cards

```text
./graveyard.py bury runs/2026-10-29-3050 BF16 QAT_Q4_0 Q4_0 --json graveyard-qat.json
```

| GPU | QAT Q4_0 right | Q4_0 right | QAT Q4_0 lost | Q4_0 lost |
| :--- | :--- | :--- | :--- | :--- |
| T4 | 221 | 🥇 222 | 15 | 15 |
| L4 | 219 | 🥇 220 | 15 | 16 |
| GTX 1650 | 🥇 222 | 216 | 11 | 19 |
| RTX 3050 | 🥇 221 | 218 | 15 | 16 |

QAT Q4_0 scores 219 to 222 on the four cards and plain Q4_0 216 to 222. On the T4 and the L4 the plain file is ahead by 1; on the GTX 1650 QAT is ahead by 6 and on the RTX 3050 by 3. The two files lose mostly different questions: on the RTX 3050, 5 questions are lost under both, 10 only under QAT and 11 only under the plain Q4_0. Across the four GPUs, 6 of QAT's losses and 5 of Q4_0's are lost on every card.

---

#### Step 8 — Speed on Each Card

Median decode speed in tokens per second. Each column is one machine; read down a column.

| Rung | T4 | L4 | GTX 1650 | RTX 3050 |
| :--- | :--- | :--- | :--- | :--- |
| BF16 | 40.0 | 50.1 | 9.2 | 40.3 |
| F16 | 39.6 | 50.0 | 9.6 | 40.3 |
| Q8_0 | 64.3 | 83.0 | 45.9 | 67.8 |
| Q6_K | 70.8 | 98.1 | 57.2 | 75.3 |
| Q5_K_M | 78.7 | 107.7 | 61.9 | 82.2 |
| Q4_K_M | 84.1 | 118.7 | 66.5 | 91.5 |
| Q4_0 | 🥇 93.6 | 125.6 | 🥇 70.8 | 🥇 99.7 |
| Q3_K_M | 74.2 | 111.1 | 55.9 | 64.2 |
| Q2_K | 74.0 | 🥇 139.5 | 62.2 | 95.0 |

- **Q4_0 is the fastest rung on three of the four cards**; on the L4 it is Q2_K, which is wrong on 244 of 250 questions.
- **Q5_K_M decodes 1.21 to 1.35 times as fast as Q8_0 on each card** (arithmetic, from the table). Q8_0 is the baseline here because it is the largest rung that fits whole on all four GPUs.
- **Q3_K_M is slower than Q4_0 on every card**, and on the RTX 3050 it is slower than Q8_0.
- **BF16 and F16 run at the same speed on the T4, the L4 and the RTX 3050.** On the GTX 1650, with 22 of 36 layers on the CPU, they run at a fifth of Q8_0's speed.

The full ladder with QAT took 5 hours 16 minutes on the RTX 3050 and 12 hours 25 minutes on the GTX 1650, from the first log line to the last (`run-3050.log`, `run-1650.log`).

---

#### Compare and Contrast

| | T4 | L4 | GTX 1650 | RTX 3050 |
| :--- | :--- | :--- | :--- | :--- |
| BF16 right of 250 | 229 | 229 | 229 | 🥇 232 |
| Q5_K_M lost | 🥇 3 | 6 | 6 | 8 |
| Q4_0 right, tok/s | 222, 93.6 | 220, 125.6 | 216, 70.8 | 218, 99.7 |
| Every rung whole on the GPU | 🟢 yes | 🟢 yes | ❌ BF16, F16 split | 🟢 yes |
| Same text twice at BF16 | 🟢 yes | 🟢 yes | 🟢 yes | 🟢 yes |

---

#### So, Which One?

On any of the four cards, take Q5_K_M: 39.0% of BF16's bytes, 226 to 231 right, 3 to 8 answers lost against the card's own BF16, and 1.21 to 1.35 times Q8_0's decode speed. For the fastest file that stays within 19 answers, take Q4_0, the fastest rung on three of the cards. On a 4 GB card, every quantized rung fits whole and the 16-bit rungs run at under 10 tokens per second. To compare two quantizations question by question, run both on one GPU: a different card moves 2 to 4 grades at BF16 and up to 34 at Q3_K_M.

---

#### Summary

The goal of this article was to run Gemma 4 E2B's quantization ladder on four GPUs, from a 4 GB laptop card to a data-centre L4, and see which grades hold. The key was holding everything else fixed: byte-identical GGUFs, the same llama.cpp tag and the same sampling, with each card checked to give identical text twice. The four-GPU results were:

- 🟢 All nine GGUFs and the QAT file match the T4's sha256 on every machine, and BF16 gives identical text twice on 50 questions on every card.
- 🟢 The ladder keeps its shape on all four GPUs: F16 to Q5_K_M lose 1 to 8, Q4_K_M and Q4_0 lose 9 to 19, Q3_K_M loses 42 to 49.
- 🟢 Q5_K_M is the best trade on every card: 226 to 231 right at 1.21 to 1.35 times Q8_0's speed.
- 🟢 A 4 GB GTX 1650 runs every quantized rung whole on the GPU; BF16 and F16 run split, 14 of 36 layers on the GPU, at 9.2 and 9.6 tokens per second.
- ⚠️ Any two GPUs grade 2 to 4 questions differently at BF16 and 12 to 22 at Q4_0; no question is lost at F16 or Q8_0 on all four.
- ⚠️ 218 questions die on all four GPUs, 147 of them first at the same rung.
- ⚠️ QAT Q4_0 and plain Q4_0 trade places by card: plain ahead by 1 on the T4 and L4, QAT ahead by 6 on the GTX 1650 and by 3 on the RTX 3050.
- ❌ Q3_K_M loses 42 to 49, mostly answers cut off at 1,024 tokens; Q2_K gets 3 to 6 right, with 208 to 228 looping answers per card.

Scope: one run of each rung per GPU on four NVIDIA cards (a Tesla T4 on GCE, an L4 on AWS, a GTX 1650 Ti laptop GPU and an RTX 3050 desktop card), 250 GSM8K test questions, greedy, thinking off, 1,024 max tokens, llama.cpp `b11699` built for each card. Determinism checked at BF16 on each card. The T4 grid is rebuilt from its committed per-question results. The hosts also differ in CPU, RAM, driver and CUDA version, and on the GTX 1650 BF16 and F16 run split between GPU and CPU. Speed is compared within each card only.

The strategy for measuring Gemma 4's accuracy one quantization rung at a time on four GPUs was validated with an incremental step by step approach.

---

#### References

- [Accuracy vs Speed: Gemma 4 E2B Quantization Tradeoffs, BF16 to Q2_K](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea), the T4 article
- [Performance vs Quality: Gemma 4 E2B Quantization on an NVIDIA L4 and a Tesla T4](https://dev.to/gde/performance-vs-quality-gemma-4-e2b-quantization-on-an-nvidia-l4-and-a-tesla-t4-4b90), the L4 article
- [gemma-graveyard on GitHub](https://github.com/xbill9/gemma-graveyard)
- [GTX 1650 run: runs/2026-10-29-1650](https://github.com/xbill9/gemma-graveyard/tree/main/runs/2026-10-29-1650)
- [RTX 3050 run: runs/2026-10-29-3050](https://github.com/xbill9/gemma-graveyard/tree/main/runs/2026-10-29-3050)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [GSM8K](https://github.com/openai/grade-school-math)
