# Can the model's own scores say which answers will flip?

Gemma 4 E2B Q8_0, 250 held-out GSM8K questions, greedy, seed 0, thinking off, llama.cpp b11699. Seven runs that give
seven different sets of answers, all with per-token top-2 scores:

| run | right of 250 |
|---|---|
| T4, default settings | 227 |
| 1650 Ti, default settings | 229 |
| L4, default settings | 228 |
| L4, `-ub 1` | 230 |
| T4 and 1650 Ti, `-ub 1` (identical) | 226 |
| 1650 Ti told 40 SMs | 225 |
| T4 and 1650 Ti, patched (identical) | 229 |

Sources: `../2026-11-02-probe/{t4,1650,l4}/{default,ub1}`, `../2026-11-03-nsm/{nsm40,nsm40-vec}`. Counts from
`tools/route_spread.py` (`q8-seven-routes.json`) and `tools/fragility.py` (`q8-t4-vs-1650.json`).

## Result

1. **Across the seven runs, 219 of 250 answers are worded differently in at least one run, and 18 change grade.**
   The other 232 questions get the same grade in all seven. Scores stay within 225 to 230.
2. **The model's scores cannot pick out the 18.** The test: flag a question when any token the T4's default run
   generated had a lead (top-1 minus top-2 log-probability) below a threshold.

   | flag when some token's lead is below | questions flagged | of the 18 that change grade | of the 232 that do not |
   |---|---|---|---|
   | 0.05 | 112 | 12 | 100 |
   | 0.1 | 164 | **18** | 146 |
   | 0.25 | 218 | 18 | 200 |

   Catching all 18 means flagging 164 of 250 questions. Nearly every answer has a near-tie somewhere; most of them
   change only the wording. Leads in the last 12 tokens, where the final number is written, catch 1 of the 18 at 0.1.
3. **A second run on the same card catches half.** The T4 at default settings against the T4 with `-ub 1`: the grade
   differs on 9 questions, all among the 18. Same for the 1650 Ti (9), and the L4 (4).
4. **Within one run nothing marks an answer as one that another card would grade differently.**

## Unmeasured

- Other rungs: the K-quant runs with per-token scores (`../2026-11-03-nsm/ladder`, `../2026-11-04-ub1`) give pairs on
  one card under two routes; not yet counted.
- Better flags than the smallest lead (a lead weighted by position, sampling several answers) were not tried.
