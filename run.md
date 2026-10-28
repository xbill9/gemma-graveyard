# Running the graveyard on the T4 VM

How to run every check in `SPEC.md` on the GCP T4 VM (`debian13`, n1-standard-2, Tesla T4 15360 MiB),
start to finish. Run the checks in order; each one must pass before the next.

## Paths

| What | Where |
|---|---|
| Repo | `~/gemma-graveyard` |
| llama.cpp (CUDA, sm 75) | `/opt1/llama.cpp` (`b11699`) |
| HF checkpoint | `/opt1/models/gemma-4-E2B-it` |
| GGUF ladder | `/opt1/graveyard-gguf` |
| Scratch | `/opt1/tmp` |

Work on `/opt1`. The root disk has under 2 GB free, and there is no `~/llama.cpp`, so pass
`--llama /opt1/llama.cpp` to every `graveyard.py run` and `/opt1/llama.cpp` as `ladder.sh`'s third argument.

## 0. Before anything

```
pgrep -af 'llama-server|vllm'            # stop anything holding the GPU
/sbin/swapon --show                      # empty after a reboot
sudo /sbin/swapon /opt1/swapfile         # 16 GB; the GGUF conversion needs it on 7.8 GB RAM
mkdir -p /opt1/tmp
```

## 1. Build the ladder

```
TMPDIR=/opt1/tmp hf download google/gemma-4-E2B-it --local-dir /opt1/models/gemma-4-E2B-it
TMPDIR=/opt1/tmp ./ladder.sh /opt1/models/gemma-4-E2B-it /opt1/graveyard-gguf /opt1/llama.cpp
```

Writes nine GGUFs and `ladder.tsv` (rung, file, bytes, sha256). `ladder.sh` skips any file that already
exists, so if the build is interrupted, delete the half-written file before running it again.

## 2. F16 range

```
python3 f16_range.py /opt1/models/gemma-4-E2B-it/model.safetensors > runs/<date>-f16-range/out.txt
```

## 3. Smoke test

```
./graveyard.py run BF16 /opt1/graveyard-gguf/e2b-BF16.gguf data/heldout.jsonl runs/<date>-smoke \
  --llama /opt1/llama.cpp --limit 10
```

Pass: 10 rows, no errors, `decode_tps` filled in. To see where the model sits in memory, start
`llama-server` by hand with `-lv 4` and look for `offloaded` and `buffer size`.

## 4. Determinism

```
for i in 1 2; do
  ./graveyard.py run BF16 /opt1/graveyard-gguf/e2b-BF16.gguf data/heldout.jsonl runs/<date>-determinism-$i \
    --llama /opt1/llama.cpp --limit 50
done
```

Pass: every `extracted` in run 1 equals the one in run 2, compared in code.

## 5. Full ladder

```
setsid nohup runs/2026-10-27-ladder/run-ladder.sh > runs/2026-10-27-ladder/run-ladder.log 2>&1 < /dev/null &
```

`run-ladder.sh` runs BF16, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_M and Q2_K on the 250 questions in
`data/heldout.jsonl`, then writes `bury.txt`. Started this way it keeps running if Claude Code or the
terminal goes away. To resume after an interruption, start it again: a rung whose `meta.json` shows
`n` = 250 is skipped.

### After the VM stops

The VM stops itself around 6 PM. Everything on `/opt1` and in `runs/` survives. A rung writes its
`meta.json` only after its last question, and `graveyard.py run` rewrites the rung's `.jsonl` from the
start, so the rung that was cut off reruns from question 1 and every finished rung is kept. After
restarting the VM:

```
cd ~/gemma-graveyard
pgrep -af 'llama-server|vllm'                 # nothing should hold the GPU
setsid nohup runs/2026-10-27-ladder/run-ladder.sh >> runs/2026-10-27-ladder/run-ladder.log 2>&1 < /dev/null &
cat runs/2026-10-27-ladder/run-ladder.log     # "skip" for each finished rung, then "start" for the next
```

Swap is not needed for this step.

Watch it:

```
cat runs/2026-10-27-ladder/run-ladder.log                    # one line per finished rung
wc -l runs/2026-10-27-ladder/*.jsonl                         # rows so far (250 each)
pgrep -af llama-server                                       # which rung is loaded now
```

Stop it: `pkill -f run-ladder.sh; pkill -f graveyard.py; pkill llama-server`.

Bury by hand (same as the script's last step):

```
./graveyard.py bury runs/2026-10-27-ladder BF16 F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K
```

## 6. QAT Q4_0 (off the ladder)

```
TMPDIR=/opt1/tmp hf download google/gemma-4-E2B-it-qat-q4_0-gguf gemma-4-E2B_q4_0-it.gguf README.md \
  --local-dir /opt1/models/gemma-4-E2B-it-qat-q4_0-gguf
setsid nohup runs/2026-10-27-ladder/run-qat.sh > runs/2026-10-27-ladder/run-qat.log 2>&1 < /dev/null &
```

`run-qat.sh` waits for `run-ladder.log` to say `done`, runs `QAT_Q4_0` on the same 250 questions, then
writes `bury-qat.txt` (`bury BF16 QAT_Q4_0 Q4_0`). It resumes the same way as the ladder script. The GGUF is
3,349,516,256 bytes, sha256 `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`
(`runs/2026-10-27-ladder/qat.tsv`).

## Results so far (2026-10-28)

| Check | Result |
|---|---|
| 1 | Nine GGUFs, 9,311,305,152 bytes (BF16) to 2,989,088,192 (Q2_K). `runs/2026-10-27-ladder/ladder.tsv` |
| 2 | 0 overflow in 5,123,178,979 weights; 5,523,265 underflow, 5,507,733 of them in the vision tower's `position_embedding_table` (not in the text GGUF). `runs/2026-10-27-f16-range/out.txt` |
| 3 | 10/10 right, 0 errors, median 43.0 tok/s. All 36 layers on the GPU (4,384.87 MiB); the two embedding tables stay memory-mapped on the CPU (4,480 + 768 MiB). |
| 4 | 50 questions twice: 0 mismatched `extracted`, 0 mismatched text, 46/50 both times. |
| 5 | All nine rungs, 250 questions each, 0 errors. Right: BF16 229, F16 229, Q8_0 227, Q6_K 229, Q5_K_M 231, Q4_K_M 224, Q4_0 222, Q3_K_M 191, Q2_K 3. Dead vs BF16: 1, 4, 4, 3, 9, 15, 42, 226; 226 tombstones, 11 ghosts. Q2_K hits the 1,024-token cap on 246 of 250 and opens a thinking block on its own in 75 (63 with empty `content`); no other rung does. `runs/2026-10-27-ladder/bury.txt` |
| 6 | QAT Q4_0 running (started 2026-10-28 01:37 UTC). |

Run directories are gitignored (`runs/*/`); commit a report with `git add -f`.
