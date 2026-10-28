---
title: "Same Weights, Different Graves: Gemma 4 E2B Quantization on an NVIDIA L4 and a Tesla T4"
published: false
description: "The Gemma 4 E2B quantization graveyard again, on an AWS L4 with byte-identical GGUF files and the same llama.cpp build as the Tesla T4. The L4 decodes 1.25 to 1.89 times faster, Q5_K_M is still the best trade, and switching GPUs changes nearly as many grades as quantizing from BF16 to Q5_K_M."
tags: gemma, llamacpp, aws, quantization
cover_image: https://raw.githubusercontent.com/xbill9/gemma-graveyard/main/docs/devto-cover-l4.82c50f0e.jpg
---

This article provides a step by step guide to running the Gemma 4 E2B quantization graveyard a second time, on an NVIDIA L4 in AWS, with the same nine GGUF files and the same llama.cpp build as the Tesla T4 run, and comparing the two GPUs question by question.

https://github.com/xbill9/gemma-graveyard

The [first article](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea) quantized Gemma 4 E2B from BF16 to Q2_K on one Tesla T4, asked every rung the same 250 GSM8K questions, and buried each answer at the rung where it first goes wrong. This one moves the whole ladder to a `g6.xlarge`. The nine files have the same sha256 on both machines, both run llama.cpp `b11699`, and each machine gives identical text when a rung is run twice. The GPU changes, and with it the CUDA kernels that do the arithmetic.

---

#### The Results in Brief

The L4 decodes every rung faster than the T4, from 1.25 times at BF16 to 1.89 times at Q2_K, and Q5_K_M stays the best trade between speed and accuracy: 228 right against BF16's 229, at 2.15 times BF16's decode speed on the L4.

The graves move with the GPU. With the same file and the same build, the L4 and the T4 grade 2 questions differently at BF16, 7 at Q5_K_M, 16 at Q4_0 and 33 at Q3_K_M. At Q5_K_M that is nearly the whole quantization effect on the T4, where Q5_K_M against BF16 was 8 questions: 3 dead and 5 ghosts. Of the 226 questions buried on each GPU, 180 die at the same rung on both.

The cover is the two graveyards laid over each other. One row per rung, BF16 at the top and Q2_K at the bottom, one cell per question: marigold where both GPUs answer right, a grey headstone where both answer wrong, violet where the two GPUs disagree. `docs/make_cover_gpu.py` draws it from the committed rows.

---

#### What Changes and What Stays

| | Tesla T4 (GCE) | NVIDIA L4 (AWS) |
| :--- | :--- | :--- |
| Instance | `n1-standard-2`, `us-west2-b` | `g6.xlarge`, `us-west-2c` |
| GPU memory | 15,360 MiB | 23,034 MiB |
| Compute capability | 7.5 (Turing) | 8.9 |
| Host | 2 vCPU, 7,436 MiB RAM | 4 vCPU AMD EPYC 7R13, 15,363 MiB RAM |
| Driver, CUDA | 615.71.10, 13.4 | 595.112.07, 13.2 |

The same on both: llama.cpp `b11699` with `GGML_CUDA=ON`, built for each card's own compute capability; the nine GGUFs and Google's QAT Q4_0 file, byte for byte; the 250 questions in `data/heldout.jsonl`; temperature 0, one request at a time, no prompt cache, thinking off, 1,024 max tokens, `llama-server -ngl 99 -c 4096 -np 1 --seed 0`.

---

#### At This Point You Should Have…

- An AWS account with quota for G instances (a `g6.xlarge` uses 4 vCPU of it)
- The AWS CLI signed in, a key pair and a security group that allows SSH
- A clone: `git clone https://github.com/xbill9/gemma-graveyard`
- The T4 run's committed results in `runs/2026-10-27-ladder/`

---

#### Step 1 — Launch the L4

The AWS Deep Learning Base AMI ships with the NVIDIA driver and the CUDA toolkits, so the box needs no driver install. Its ID comes from a public SSM parameter:

```text
aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text
ami-0fcc8fa7ee8d2dc4c
```

```text
aws ec2 run-instances --image-id ami-0fcc8fa7ee8d2dc4c --instance-type g6.xlarge --key-name graveyard-l4 \
  --security-group-ids sg-0d721a15b91250482 --metadata-options HttpTokens=required \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}'
```

From `nvidia-smi` on the new box:

```text
| NVIDIA-SMI 595.112.07             Driver Version: 595.112.07     CUDA Version: 13.2     |
|   0  NVIDIA L4                      On  |   00000000:31:00.0 Off |                    0 |
| N/A   55C    P0             28W /   72W |       0MiB /  23034MiB |      0%      Default |
```

A `g6.xlarge` in us-west-2 is $0.8048 an hour on demand, from the AWS Price List API.

🔎 **Tip: work on the instance's own NVMe drive.** A `g6.xlarge` carries a local NVMe disk, which the Deep Learning AMI mounts at `/opt/dlami/nvme`. The checkpoint, the nine GGUFs, the llama.cpp build and a 16 GB swapfile all go there. It is wiped when the instance stops, so copy the results off first.

---

#### Step 2 — Build llama.cpp at the Same Tag

```text
git clone https://github.com/ggml-org/llama.cpp /opt/dlami/nvme/llama.cpp
cd /opt/dlami/nvme/llama.cpp && git checkout b11699
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-quantize llama-cli
git describe --tags
b11699
```

The one build flag that differs from the T4 is `CMAKE_CUDA_ARCHITECTURES`: 75 for the T4, 89 for the L4. llama.cpp's log on the L4 confirms it:

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 22565 MiB):
  Device 0: NVIDIA L4, compute capability 8.9, VMM: yes, VRAM: 22565 MiB
```

🔎 **Tip: llama.cpp's pinned `transformers` cannot read Gemma 4's tokenizer.** `pip install -r requirements/requirements-convert_hf_to_gguf.txt` installs `transformers` 4.57.6, and `convert_hf_to_gguf.py` then stops while loading the tokenizer:

```text
AttributeError: 'list' object has no attribute 'keys'
```

`pip install -U transformers` (5.21.0) converts the checkpoint, and Step 3 shows the files match the T4's byte for byte.

---

#### Step 3 — Rebuild the Ladder and Check It Against the T4

`ladder.sh` converts the checkpoint to BF16 and F16 and quantizes seven rungs from BF16, as on the T4. `runs/2026-10-28-l4/run-l4.sh` then compares each file's size and sha256 with the T4's `ladder.tsv`:

```text
./ladder.sh /opt/dlami/nvme/models/gemma-4-E2B-it /opt/dlami/nvme/graveyard-gguf /opt/dlami/nvme/llama.cpp
diff <(cut -f1,3,4 runs/2026-10-27-ladder/ladder.tsv) <(cut -f1,3,4 runs/2026-10-28-l4/ladder.tsv) && echo "ladder matches T4 sha256"
ladder matches T4 sha256
```

All nine files are identical, and so is Google's QAT Q4_0 file (sha256 `fa401b55…`). From here on, every answer that differs between the two machines comes from the GPU and its kernels.

---

#### Step 4 — Prove the L4 Repeats Itself

The same check as on the T4: BF16 on 50 questions, twice.

```text
./graveyard.py repeat runs/2026-10-28-l4/determinism-1 runs/2026-10-28-l4/determinism-2 BF16
{"rung": "BF16", "n": 50, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 46, "right_b": 46}
```

Every character matches. Each GPU on its own is deterministic, so a difference between the two GPUs belongs to the GPU.

---

#### Step 5 — Run the Ladder on the L4 and Bury It

`run-l4.sh` runs the nine rungs and QAT Q4_0, then buries both:

```text
./graveyard.py bury runs/2026-10-28-l4 BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K --json graveyard-ladder.json
{"first_deaths_by_rung": {"Q2_K": 169, "Q3_K_M": 27, "Q4_0": 11, "Q6_K": 3, "Q4_K_M": 9, "Q5_K_M": 3, "Q8_0": 3, "F16": 1}, "tombstones": 226, "ghosts": 15}
```

| Rung | Right, T4 | Right, L4 | Dead, L4 | Ghosts, L4 | Median tok/s, L4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BF16 | 229 | 229 | 0 | 0 | 50.1 |
| F16 | 229 | 229 | 1 | 1 | 50.0 |
| Q8_0 | 227 | 🥇 228 | 3 | 2 | 83.0 |
| Q6_K | 229 | 227 | 6 | 4 | 98.1 |
| Q5_K_M | 231 | 🥇 228 | 6 | 5 | 107.7 |
| Q4_K_M | 224 | 221 | 14 | 6 | 118.7 |
| Q4_0 | 222 | 220 | 16 | 7 | 125.6 |
| Q3_K_M | 191 | 186 | 45 | 2 | 111.1 |
| Q2_K | 3 | 6 | 225 | 2 | 🥇 139.5 |

**Dead**: BF16 right, this rung wrong. **Ghosts**: BF16 wrong, this rung right. Both are counted against the same GPU's own BF16.

The shape of the ladder holds. Q8_0 to Q5_K_M stay within 2 of BF16; Q4_K_M and Q4_0 lose 8 and 9; Q3_K_M loses 43; Q2_K collapses. The autopsy gives the same three kinds of death as on the T4: Q3_K_M runs out of room (36 of its 45 deaths cut off at 1,024 tokens), and Q2_K loops (218 of 250 answers repeat a line five or more times, 71 open a thinking block with thinking off, 63 leave `content` empty).

🔎 **Tip: llama-server can refuse an answer it cannot parse.** One Q2_K question, `test-194`, came back as HTTP 500 on the L4: `The model produced output that does not match the expected peg-gemma4 format`. `graveyard.py` records it as an error and grades it wrong, so Q2_K's 6 right are out of 250 with 1 error.

---

#### Step 6 — Line the Two GPUs Up, Question by Question

The T4's per-question rows stay on the T4. Its committed `graveyard-ladder.json` holds every tombstone with the rung where it died and every rung where it rose again, plus every ghost, so `docs/gpu_derived.py` rebuilds the T4's right/wrong grid for all 250 questions at all nine rungs from it. Twelve questions have neither a tombstone nor a ghost, so each is right at every rung or wrong at every rung; the T4 article's cover, drawn from the T4's rows with one cell per question and rung, places them. The rebuilt grid reproduces all 2,250 cells of that cover and the T4's own right, dead and ghost counts at every rung.

```text
python3 docs/gpu_derived.py > docs/evidence-l4/derived-gpu.txt
```

| Rung | Graded differently, T4 vs L4 | Dead, T4 | Dead, L4 | Dead on both |
| :--- | :--- | :--- | :--- | :--- |
| BF16 | 2 | — | — | — |
| F16 | 4 | 1 | 1 | 0 |
| Q8_0 | 3 | 4 | 3 | 2 |
| Q6_K | 6 | 4 | 6 | 3 |
| Q5_K_M | 7 | 3 | 6 | 2 |
| Q4_K_M | 15 | 9 | 14 | 7 |
| Q4_0 | 16 | 15 | 16 | 10 |
| Q3_K_M | 33 | 42 | 45 | 28 |
| Q2_K | 9 | 226 | 225 | 221 |

**Graded differently**: questions right on one GPU and wrong on the other, at that rung.

Three things show in the grid:

- **Above Q4_K_M, the GPU moves nearly as many grades as the quantization.** At Q5_K_M the two GPUs disagree on 7 questions; on the T4, Q5_K_M and BF16 disagree on 8 (3 dead, 5 ghosts). Q5_K_M's 231 on the T4 and 228 on the L4 sit either side of BF16's 229 on both.
- **Most graves are shared, at a different rung.** 222 questions are tombstoned on both GPUs, 4 only on the T4 and 4 only on the L4. 180 of them die at the same rung on both.
- **The deep deaths are the stable ones.** Q2_K's dead are 221 of 226 shared and Q3_K_M's 28 of 42. Between F16 and Q5_K_M, 7 of 12 T4 deaths are dead again on the L4 at the same rung.

The twins and triplets question shows it. The T4 article's one F16 death, `test-687` (answer 13), was right at BF16 on the T4 and answered 17 at F16. On the L4, BF16 itself answers 17, so the question is a ghost there: wrong at BF16, right again at Q8_0, Q6_K, Q5_K_M and Q4_0. `test-93` (answer 36) runs the other way: a ghost on the T4, right at BF16 on the L4 and dead from Q8_0 on.

---

#### Step 7 — Google's QAT Q4_0 on the L4

```text
./graveyard.py bury runs/2026-10-28-l4 BF16 QAT_Q4_0 Q4_0 --json graveyard-qat.json
{"rung": "QAT_Q4_0", "n": 250, "right": 219, "errors": 0, "truncated": 8, "median_decode_tps": 125.7, "mean_tokens": 394.5, "dead": 15, "ghosts": 5, "answer_changed": 27, "bytes": 3349516256, "gpu_mib_loaded": 1730, "gpu_mib_before": 0}
{"rung": "Q4_0", "n": 250, "right": 220, "errors": 0, "truncated": 12, "median_decode_tps": 125.6, "mean_tokens": 437.6, "dead": 16, "ghosts": 7, "answer_changed": 28, "bytes": 3360161216, "gpu_mib_loaded": 1740, "gpu_mib_before": 0}
```

The same picture as the T4, one right answer apart again: 219 against the plain Q4_0's 220, where the T4 had 221 against 222. On the L4 the two decode at the same speed, 125.7 and 125.6 tok/s; on the T4 the plain Q4_0 was faster, 93.6 against 87.8. 10 of QAT Q4_0's 15 deaths on the T4 are dead on the L4 too.

---

#### Step 8 — Speed and Memory on Each Card

| Rung | Median tok/s, T4 | Median tok/s, L4 | L4 ÷ T4 | × BF16, T4 | × BF16, L4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BF16 | 40.0 | 50.1 | 1.25 | 1.00 | 1.00 |
| F16 | 39.6 | 50.0 | 1.26 | 0.99 | 1.00 |
| Q8_0 | 64.3 | 83.0 | 1.29 | 1.61 | 1.66 |
| Q6_K | 70.8 | 98.1 | 1.39 | 1.77 | 1.96 |
| Q5_K_M | 78.7 | 107.7 | 1.37 | 1.97 | 2.15 |
| Q4_K_M | 84.1 | 118.7 | 1.41 | 2.10 | 2.37 |
| Q4_0 | 93.6 | 125.6 | 1.34 | 2.34 | 2.51 |
| Q3_K_M | 74.2 | 111.1 | 1.50 | 1.85 | 2.22 |
| Q2_K | 74.0 | 139.5 | 1.89 | 1.85 | 2.78 |

- **BF16 and F16 run at the same speed on both cards**, within 0.4 tok/s of each other on each GPU.
- **The smaller the file, the bigger the L4's lead**, up to 1.89 times at Q2_K. Q2_K is the fastest rung on the L4 and among the slowest on the T4; it is also wrong on 244 of 250 questions.
- **Q3_K_M is slower than Q4_K_M and Q4_0 on both cards.**
- **The L4 uses 101 MiB more GPU memory at every rung** (`nvidia-smi` after load: 4,772 against 4,671 MiB at BF16, 1,740 against 1,639 at Q4_0), the same fixed amount whatever the file size.

The L4 run took 4 hours 3 minutes from the start of the ladder build to the last bury (`run-l4.log`), about $3.26 at the on-demand price (arithmetic).

---

#### Compare and Contrast

| | Tesla T4 | NVIDIA L4 |
| :--- | :--- | :--- |
| BF16 right of 250 | 229 | 229 |
| Q5_K_M right, × BF16 speed | 🥇 231, 1.97 | 🥇 228, 2.15 |
| Q4_0 right, median tok/s | 222, 93.6 | 220, 🥇 125.6 |
| Q3_K_M right | 191 | 186 |
| Same text twice on one GPU | 🟢 yes | 🟢 yes |
| Tombstones at the same rung as the other GPU | 180 of 226 | 180 of 226 |
| Verdict | 🟢 Q5_K_M, slower card | 🟢 Q5_K_M, 1.37 times faster |

---

#### So, Which One?

On either card, take Q5_K_M: 39.0% of BF16's bytes, BF16's score within the noise on both GPUs, and 1.97 times BF16's decode speed on the T4 or 2.15 times on the L4. For the fastest file that stays within 9 answers, take Q4_0. The L4 is the faster card at every rung, by 1.25 to 1.89 times. To compare two quantizations question by question, run both on one GPU: a different card moves a few graves at every rung, and up to 33 at Q3_K_M.

---

#### Summary

The goal of this article was to rerun the Gemma 4 E2B quantization graveyard on a second GPU and see which graves stay put. The key was holding everything else fixed: byte-identical GGUFs, the same llama.cpp tag and the same sampling, with each GPU checked to give identical text twice. The L4 results were:

- 🟢 All nine GGUFs and the QAT file match the T4's sha256; BF16 on the L4 gives identical text twice.
- 🟢 BF16 scores 229 of 250 on both GPUs, and the ladder keeps its shape: Q8_0 to Q5_K_M within 2, Q4_K_M and Q4_0 within 9.
- 🟢 Q5_K_M is the best trade on both: 228 right at 2.15 times BF16's speed on the L4.
- 🟢 The L4 decodes 1.25 to 1.89 times faster than the T4, and BF16 runs at F16's speed on both cards.
- ⚠️ The two GPUs grade 2 questions differently at BF16, 7 at Q5_K_M and 16 at Q4_0; above Q4_K_M that is nearly the size of the quantization effect.
- ⚠️ 180 of 226 tombstones fall at the same rung on both GPUs; 222 questions are buried on both at some rung.
- ⚠️ QAT Q4_0 scores 219 against the plain Q4_0's 220 on the L4, one apart again, at the same speed.
- ❌ Q3_K_M loses 43, 36 of its 45 deaths cut off at 1,024 tokens; Q2_K gets 6 right, with 218 looping answers and 1 answer the server refused to parse.

Scope: one Tesla T4 on GCE (`n1-standard-2`, us-west2-b) and one NVIDIA L4 on AWS (`g6.xlarge`, us-west-2c, $0.8048 per hour on demand), one run of each rung per GPU, 250 GSM8K test questions, greedy, thinking off, 1,024 max tokens, llama.cpp `b11699` built for each card. Determinism checked on the L4 at BF16. The T4 grid is rebuilt from its committed tombstones, ghosts and cover. The two hosts also differ in CPU, RAM, driver and CUDA version.

The strategy for burying Gemma 4's answers one quantization rung at a time on two GPUs was validated with an incremental step by step approach.

---

#### References

- [Accuracy vs Speed: Gemma 4 E2B Quantization Tradeoffs, BF16 to Q2_K](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea), the T4 article
- [gemma-graveyard on GitHub](https://github.com/xbill9/gemma-graveyard)
- [L4 run: runs/2026-10-28-l4](https://github.com/xbill9/gemma-graveyard/tree/main/runs/2026-10-28-l4)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Amazon EC2 G6 instances](https://aws.amazon.com/ec2/instance-types/g6/)
- [GSM8K](https://github.com/openai/grade-school-math)
