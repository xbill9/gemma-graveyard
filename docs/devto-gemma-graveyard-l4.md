---
title: "Same Weights, Different Graves: Gemma 4 E2B Quantization on an NVIDIA L4 and a Tesla T4"
published: false
description: "The Gemma 4 E2B quantization graveyard, run again on an AWS L4 with byte-identical GGUF files and the same llama.cpp build. Each GPU gives the same text twice; the two GPUs give different text, and part of every tombstone moves with the card."
tags: gemma, llamacpp, aws, quantization
cover_image: PENDING
---

This article provides a step by step guide to running the Gemma 4 E2B quantization graveyard a second time, on an NVIDIA L4 in AWS, with the same nine GGUF files and the same llama.cpp build as the Tesla T4 run, and comparing the two GPUs question by question.

https://github.com/xbill9/gemma-graveyard

The [first article](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea) quantized Gemma 4 E2B from BF16 to Q2_K on one Tesla T4, asked every rung the same 250 GSM8K questions, and buried each answer at the rung where it first goes wrong. This one moves the whole ladder to a `g6.xlarge`. The nine files have the same sha256 on both machines, both machines run llama.cpp `b11699`, and each machine gives identical text when a rung is run twice. What changes is the GPU, and with it the CUDA kernels that do the arithmetic.

⏳ Results in brief: written when the last rung finishes.

---

#### What Changes and What Stays

| | Tesla T4 (GCE) | NVIDIA L4 (AWS) |
| :--- | :--- | :--- |
| Instance | `n1-standard-2`, `us-west2-b` | `g6.xlarge`, `us-west-2c` |
| GPU memory | 15,360 MiB | 23,034 MiB |
| Compute capability | 7.5 (Turing) | 8.9 (Ada Lovelace) |
| Native bfloat16 | no | yes |
| Host | 2 vCPU, 7,436 MiB RAM | 4 vCPU AMD EPYC 7R13, 15,363 MiB RAM |
| Driver, CUDA | 615.71.10, 13.4 | 595.112.07, 13.2 |

The same on both: llama.cpp `b11699` with `GGML_CUDA=ON`, built for each card's own compute capability; the nine GGUFs and Google's QAT Q4_0 file, byte for byte; the 250 questions in `data/heldout.jsonl`; temperature 0, one request at a time, no prompt cache, thinking off, 1,024 max tokens, `llama-server -ngl 99 -c 4096 -np 1 --seed 0`.

---

#### At This Point You Should Have…

- An AWS account with quota for G instances (a `g6.xlarge` uses 4 vCPU of it)
- The AWS CLI signed in, and a key pair for SSH
- A clone: `git clone https://github.com/xbill9/gemma-graveyard`
- The T4 run's committed results in `runs/2026-10-27-ladder/`

---

#### Step 1 — Launch the L4

The AWS Deep Learning Base AMI ships with the NVIDIA driver and CUDA toolkits, so the box needs no driver install. Its ID comes from a public SSM parameter:

```text
aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text
ami-0fcc8fa7ee8d2dc4c
aws ec2 run-instances --image-id ami-0fcc8fa7ee8d2dc4c --instance-type g6.xlarge --key-name graveyard-l4 \
  --security-group-ids sg-0d721a15b91250482 --metadata-options HttpTokens=required \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}'
```

```text
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
NVIDIA L4, 23034 MiB, 595.112.07
```

🔎 **Tip: work on the instance's own NVMe drive.** A `g6.xlarge` carries a 250 GB local NVMe disk, which the Deep Learning AMI mounts at `/opt/dlami/nvme`. The checkpoint, the nine GGUFs, the llama.cpp build and a 16 GB swapfile all go there. It is wiped when the instance stops, so copy the results off before stopping it.

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

The only build flag that differs from the T4 is `CMAKE_CUDA_ARCHITECTURES`: 75 for Turing, 89 for Ada.

🔎 **Tip: llama.cpp's pinned `transformers` cannot read Gemma 4's tokenizer.** `pip install -r requirements/requirements-convert_hf_to_gguf.txt` installs `transformers` 4.57.6, and `convert_hf_to_gguf.py` then stops with `AttributeError: 'list' object has no attribute 'keys'` while loading the tokenizer. `pip install -U transformers` (5.21.0 here) converts it, and the resulting files match the T4's byte for byte.

---

#### Step 3 — Rebuild the Ladder and Check It Against the T4

`ladder.sh` converts the checkpoint to BF16 and F16 and quantizes seven rungs from BF16, exactly as on the T4. `runs/2026-10-28-l4/run-l4.sh` then compares every file's size and sha256 with the T4's `ladder.tsv`:

```text
./ladder.sh /opt/dlami/nvme/models/gemma-4-E2B-it /opt/dlami/nvme/graveyard-gguf /opt/dlami/nvme/llama.cpp
diff <(cut -f1,3,4 runs/2026-10-27-ladder/ladder.tsv) <(cut -f1,3,4 runs/2026-10-28-l4/ladder.tsv) && echo "ladder matches T4 sha256"
ladder matches T4 sha256
```

All nine files are identical, and so is Google's QAT Q4_0 file (sha256 `fa401b55…`). From here on, any answer that differs between the two machines comes from the GPU.

---

#### Step 4 — Prove the L4 Repeats Itself

The same check as on the T4: BF16 on 50 questions, twice.

```text
./graveyard.py repeat runs/2026-10-28-l4/determinism-1 runs/2026-10-28-l4/determinism-2 BF16
{"rung": "BF16", "n": 50, "mismatched_extracted": 0, "mismatched_text": 0, "mismatched_right": 0, "right_a": 46, "right_b": 46}
```

Every character matches. Each GPU on its own is deterministic, so a difference between the two GPUs is the GPU's.

---

#### Step 5 — Run the Ladder on the L4

```text
./graveyard.py run BF16 /opt/dlami/nvme/graveyard-gguf/e2b-BF16.gguf data/heldout.jsonl runs/2026-10-28-l4 --llama /opt/dlami/nvme/llama.cpp
{"rung": "BF16", "n": 250, "right": 229, "errors": 0, "truncated": 4, "median_decode_tps": 50.1, "mean_tokens": 390.1}
```

⏳ Full ladder table and `bury` output: written when the last rung finishes.

---

#### Step 6 — Line the Two GPUs Up, Question by Question

The T4's per-question rows stay on the T4. Its committed `graveyard-ladder.json` holds every tombstone with the rung it died at and every rung where it rose again, plus every ghost, so `docs/gpu_derived.py` rebuilds the T4's right/wrong grid for all 250 questions at all nine rungs from it. Twelve questions have neither a tombstone nor a ghost, so they are right at every rung or wrong at every rung; the T4 cover image, drawn from the T4's rows with one cell per question and rung, places them. The rebuilt grid reproduces all 2,250 cells of the cover and the T4's own right, dead and ghost counts at every rung.

The five rungs finished so far:

| Rung | Right, T4 | Right, L4 | Graded differently | Dead, T4 | Dead, L4 | Dead on both |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BF16 | 229 | 229 | 2 | — | — | — |
| F16 | 229 | 229 | 4 | 1 | 1 | 0 |
| Q8_0 | 227 | 228 | 3 | 4 | 3 | 2 |
| Q6_K | 229 | 227 | 6 | 4 | 6 | 3 |
| Q5_K_M | 231 | 228 | 7 | 3 | 6 | 2 |

**Graded differently**: questions right on one GPU and wrong on the other, at that rung.

⏳ Q4_K_M to Q2_K and QAT Q4_0: written when the last rung finishes.

---

#### Step 7 — Speed on Each Card

| Rung | Median tok/s, T4 | Median tok/s, L4 | L4 ÷ T4 |
| :--- | :--- | :--- | :--- |
| BF16 | 40.0 | 50.1 | ⏳ |
| F16 | 39.6 | 50.0 | ⏳ |
| Q8_0 | 64.3 | 83.0 | ⏳ |
| Q6_K | 70.8 | 98.1 | ⏳ |
| Q5_K_M | 78.7 | 107.7 | ⏳ |

⏳ Ratios from `docs/gpu_derived.py`, and the remaining rungs.

---

#### Compare and Contrast

⏳

---

#### So, Which One?

⏳

---

#### Summary

⏳

Scope: one Tesla T4 on GCE and one NVIDIA L4 on AWS (`g6.xlarge`, us-west-2, $0.8048 per hour on demand), one run of each rung per GPU, 250 GSM8K test questions, greedy, thinking off, 1,024 max tokens. Determinism checked on both GPUs at BF16. The two hosts also differ in CPU, RAM and driver.

The strategy for burying Gemma 4's answers one quantization rung at a time on two GPUs was validated with an incremental step by step approach.

---

#### References

- [Accuracy vs Speed: Gemma 4 E2B Quantization Tradeoffs, BF16 to Q2_K](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea), the T4 article
- [gemma-graveyard on GitHub](https://github.com/xbill9/gemma-graveyard)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Amazon EC2 G6 instances](https://aws.amazon.com/ec2/instance-types/g6/)
- [GSM8K](https://github.com/openai/grade-school-math)
