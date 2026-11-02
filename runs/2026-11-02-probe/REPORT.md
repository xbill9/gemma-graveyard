# Probe: the GPU's own scores at every flip

Gemma 4 E2B Q8_0 (sha256 `6b0301104c89a43c3bec5bae4b14d744f6000bc21e00e46e887fbbfe024de38a` on both cards),
llama.cpp b11699 (7182ef71e), 250 held-out GSM8K questions, greedy, seed 0, thinking off, `cache_prompt` off.
Each run also records, for every generated token, the top two candidates and their logprobs
(`graveyard.py run --top-logprobs 2`). llama-server computes these as log-softmax over the raw logits
(`tools/server/server-common.cpp:1586` at b11699), so the gap between the top two logprobs is the gap between
their logits.

Cards: AWS g6.xlarge L4 (driver 595.112.07, built sm_89, and a second build for sm_75), 2026-11-02; GCE
n1-standard-2 Tesla T4 (driver 615.71.10, built sm_75), 2026-11-02. Every count below is from
`tools/fork_probe.py`; its full output for each pair is the `*.json` in this directory.

## Result

| pair (Q8_0, 250 answers) | text differs | grade differs | tokens compared | same score on both | splits that swap the top two |
|---|---|---|---|---|---|
| L4 sm_75 `-ub 1` vs T4 `-ub 1` | **0** | 0 | 97,450 | **97,450 (100%)** | — |
| L4 normal vs L4 `-ub 1` (same chip) | 165 | 4 | 51,758 | 58.7% | 159 of 165 |
| T4 normal vs T4 `-ub 1` (same chip) | 166 | 9 | 50,197 | 59.9% | 160 of 166 |
| L4 normal vs T4 normal | 172 | 3 | 47,942 | 58.8% | 164 of 172 |

"Tokens compared" counts every token both runs produced from the same text so far, up to the first split.
"Same score" is the chosen token's logprob being the same number on both runs.

1. **Same code, two chips: the same number at every token.** With the L4 built for the T4's target and both cards
   sending the prompt one token at a time, every one of 97,450 scores is identical, across two chips, two
   clouds and two driver versions. All 250 answers match word for word.
2. **Same chip, two code paths: the scores move.** Changing only how the prompt goes through the GPU leaves
   40–41% of scores different. The median difference is 0.00016 (L4) and 0.00014 (T4); the largest is 0.54 and 0.56.
3. **Every split is a near-tie tipped over.** At the token where two runs part, the median lead of the top token
   is 0.12–0.14 logits on each side (normally about 16, `../2026-11-01-flips/REPORT.md` item 2). The smaller of the
   two runs' leads is under 0.5 at 163 of 165 (L4), 164 of 166 (T4) and 172 of 172 (L4 vs T4) splits. In 159–164 of
   them the two runs hold the same two candidates in opposite order; the rest are three-way near-ties.
4. **Asking for scores changes nothing.** Each probe run's text equals the 2026-11-01 run of the same card and
   mode on 250 of 250 answers (L4 default, `-ub 1`, sm_75 `-ub 1`; AWS T4 runs for the T4's default and `-ub 1`).
   The GCE T4 therefore also matches the AWS T4 word for word.

Example, test-530, L4: after "Josh worked out for ", normal settings score "4" at −0.519 and "8" at −0.904;
`-ub 1` scores "8" at −0.176 and "4" at −1.823. Both answers are graded right.

## Unmeasured

- Q8_0 only. BF16 and Q4_0 were not probed.
- Which addition inside the kernels rounds differently is read from source (`../2026-11-01-flips/REPORT.md`
  item 6), not measured.
- GTX 1650 and RTX 3050 not probed.

## Files

`l4/{default,ub1,cc75-ub1}/`, `t4/{default,ub1}/`: answers with per-token scores, server logs, meta.
`host.txt`, `sha.txt`, `probe-*.log` per card. Pair outputs: `control-l4cc75ub1-vs-t4ub1.json`,
`l4-default-vs-l4-ub1.json`, `t4-default-vs-t4-ub1.json`, `l4-vs-t4-default.json`. Rig: `tools/probe-rig.sh` (L4);
the T4 ran the same two `graveyard.py run` commands against its existing build at `/opt1/llama.cpp`.
