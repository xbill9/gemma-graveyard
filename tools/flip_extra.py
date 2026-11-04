#!/usr/bin/env python3
"""flip_extra.py: the flip-study figures that flip_core.py does not print (runs/2026-11-01-flips/REPORT.md items 4-5).
Same 28 runs (four cards x BF16..Q4_0) and grades as flip_core.py. Prints JSON; every figure is computed here.
  minority_top30_share   per contested question, minority results = min(right, wrong) over the 28 runs; the share of
                         all minority results held by the 30 questions with the most
  bf16_tokens_median     BF16 completion tokens, contested vs uncontested, per card with text, and the AUC of tokens
                         for telling contested from uncontested (Mann-Whitney, ties count half)
  sd_by_rung             spread (population SD) of the four cards' right-answer totals, observed, and the median over
                         4000 simulations where each card's grade on each question is an independent coin flip with
                         that question's share of right answers across the four cards at that rung"""
import json
import random
import statistics as st
import sys

sys.path[:0] = ["docs", "."]
from gpu_derived import IDS, LADDER, t4_grid

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}
LIGHT = LADDER[:7]


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def auc(pos, neg):
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def main():
    T = {g: {r: rows(g, r) for r in LADDER} for g in D}
    G = {"t4": t4_grid("graveyard-ladder.json", LADDER), **{g: {r: {i: x["right"] for i, x in T[g][r].items()} for r in LADDER} for g in D}}
    cards = list(G)
    n = len(cards) * len(LIGHT)
    score = {i: sum(G[g][r][i] for g in cards for r in LIGHT) for i in IDS}
    contested = [i for i in IDS if 0 < score[i] < n]
    minority = sorted((min(score[i], n - score[i]) for i in contested), reverse=True)
    out = {"runs": n, "contested": len(contested), "minority_results": sum(minority),
           "minority_top30_share": round(sum(minority[:30]) / sum(minority), 4)}
    cs = set(contested)
    tok = {}
    for g in D:
        c = [T[g]["BF16"][i]["completion_tokens"] for i in IDS if i in cs]
        u = [T[g]["BF16"][i]["completion_tokens"] for i in IDS if i not in cs]
        tok[g] = {"contested": st.median(c), "uncontested": st.median(u), "auc": auc(c, u)}
    out["bf16_tokens_median"] = tok
    rng = random.Random(0)
    sd = {}
    for r in LADDER:
        obs = st.pstdev(sum(G[g][r].values()) for g in cards)
        p = {i: sum(G[g][r][i] for g in cards) / len(cards) for i in IDS}
        sims = sorted(st.pstdev(sum(rng.random() < p[i] for i in IDS) for _ in cards) for _ in range(4000))
        sd[r] = {"observed": round(obs, 2), "independent_flips_median": round(st.median(sims), 2)}
    out["sd_by_rung"] = sd
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
