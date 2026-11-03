# Probe runs on the GTX 1650 (yogagpu)

The 1650 has the T4's build target (sm_75, `~/graveyard-llama`, b11699), but on 2026-10-29 it differed from the
AWS T4 on 165 of 250 Q8_0 answers. These two runs record the 1650's own top-two scores at every token, so they
can be compared with the T4 and L4 probe runs in `runs/2026-11-02-probe/`. They answer one question: with `-ub 1`,
does the 1650 give the same scores as the T4 and the sm_75 L4, or does it take a third code path? It has no
tensor cores, so its quantized matrix code may differ from the T4's. Untested.

Two runs of Q8_0, 250 questions each, about 30–35 minutes each at the 1650's 45.9 tokens/s (Q8_0, `bury.txt` of
`runs/2026-10-29-1650`).

## 1. Pull and check

```bash
cd ~/gemma-graveyard && git pull --ff-only
git log --oneline -1                                  # the commit that added this file, or later
git -C ~/graveyard-llama describe --tags              # b11699
grep ARCHITECTURES ~/graveyard-llama/build/CMakeCache.txt    # 75
sha256sum ~/graveyard-gguf/e2b-Q8_0.gguf
# 6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a
pgrep -af 'llama-server|vllm'                         # nothing; stop anything on the GPU first
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
```

Stop if the tag, the target or the sha256 differs.

## 2. Run

```bash
cd ~/gemma-graveyard
D=runs/2026-11-02-probe/1650
L=$HOME/graveyard-llama
G=$HOME/graveyard-gguf/e2b-Q8_0.gguf
mkdir -p $D
{ git -C $L describe --tags; git -C $L rev-parse HEAD; nvidia-smi; lscpu | grep 'Model name'; free -m; } > $D/host.txt
echo "$(date -u +%FT%TZ) sha ok Q8_0 $(sha256sum $G | cut -d' ' -f1)" > $D/sha.txt
{
  echo "$(date -u +%FT%TZ) start default"
  ./graveyard.py run Q8_0 $G data/heldout.jsonl $D/default --llama $L --top-logprobs 2 | tail -1
  echo "$(date -u +%FT%TZ) start ub1"
  ./graveyard.py run Q8_0 $G data/heldout.jsonl $D/ub1 --llama $L --top-logprobs 2 --server-arg=-ub --server-arg=1 | tail -1
  echo "$(date -u +%FT%TZ) ALL DONE"
} 2>&1 | tee $D/probe-1650.log
```

The settings are the ladder's (`-ngl 99 -c 4096 -np 1 --seed 0`, temperature 0, thinking off, 1,024 max tokens);
`--top-logprobs 2` only adds the scores to each answer. The machine must stay awake for the whole run: plug it in
and turn off suspend.

Expected right-answer count for `default`: 229, the 2026-10-29 Q8_0 count on this GPU (`bury.txt`). A different
count means the run differs from the earlier one; keep it and say so.

## 3. Commit and push

```bash
cd ~/gemma-graveyard
git add -f runs/2026-11-02-probe/1650
git commit -m "Probe: GTX 1650 Q8_0, default and -ub 1, per-token top-2 scores"
git push
```

## 4. Compare (back on the main machine)

```bash
P=runs/2026-11-02-probe
# the probe changes no answers: default text against the 2026-10-29 run
python3 tools/fork_probe.py $P/1650/default/Q8_0.jsonl $P/t4/default/Q8_0.jsonl \
  --plain-a runs/2026-10-29-1650/Q8_0.jsonl --out $P/1650-vs-t4-default.json | head -1
# the question: does the 1650 join the T4 / sm_75 L4 match under -ub 1?
python3 tools/fork_probe.py $P/1650/ub1/Q8_0.jsonl $P/t4/ub1/Q8_0.jsonl --out $P/1650-ub1-vs-t4-ub1.json | head -1
# same chip, two code paths
python3 tools/fork_probe.py $P/1650/default/Q8_0.jsonl $P/1650/ub1/Q8_0.jsonl --out $P/1650-default-vs-1650-ub1.json | head -1
```

`text_differs: 0` and `shared_tokens_same_logprob` equal to `shared_tokens` in the second line means the 1650 matches
the T4 exactly once the prompt route is pinned. Anything else means the 1650 runs a different code path even with
`-ub 1`; the fork list in the JSON shows where.
