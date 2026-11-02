#!/usr/bin/env python3
"""flip_kinds.py: what the wrong answers in contested questions are, and which questions drift with the rung.
Three cards with text, BF16 to Q4_0. Wrong answers fall in the first class that fits:
no_answer_line: text has no 'Answer:' line, so the grader took the last number
cut_off: finish was length
unit: answer is gold x100 or /100
rounding: within 2% of gold
other: everything else, a different result."""
import collections as C
import json
import re
import sys

from scipy.stats import fisher_exact

sys.path[:0] = ["docs", "."]
from gpu_derived import IDS, LADDER, t4_grid

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}
LIGHT = LADDER[:7]
HI, LO = LADDER[:3], LADDER[5:7]


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def kind(x):
    t, g, a = x.get("text") or "", float(x["answer"].replace(",", "")), x.get("extracted")
    if x.get("finish") == "length":
        return "cut_off"
    if not re.search(r"Answer:", t):
        return "no_answer_line"
    if a is None:
        return "other"
    if abs(a - 100 * g) < 1e-6 or abs(a * 100 - g) < 1e-6:
        return "unit"
    if g and abs(a - g) / abs(g) <= 0.02:
        return "rounding"
    return "other"


def main():
    T = {g: {r: rows(g, r) for r in LADDER} for g in D}
    G = {"t4": t4_grid("graveyard-ladder.json", LADDER), **{g: {r: {i: x["right"] for i, x in T[g][r].items()} for r in LADDER} for g in D}}
    score = {i: sum(G[g][r][i] for g in G for r in LIGHT) for i in IDS}
    cont = [i for i in IDS if 0 < score[i] < 28]
    k, byq = C.Counter(), C.defaultdict(C.Counter)
    for i in cont:
        for g in D:
            for r in LIGHT:
                x = T[g][r][i]
                if not x["right"]:
                    c = kind(x)
                    k[c] += 1
                    byq[i][c] += 1
    out = {"wrong_runs_contested": sum(k.values()), "kinds": dict(k)}
    out["questions_mostly"] = dict(C.Counter(v.most_common(1)[0][0] for v in byq.values()))
    out["unit_questions"] = sorted(i for i in byq if byq[i]["unit"])
    out["rounding_questions"] = sorted(i for i in byq if byq[i]["rounding"])
    out["cutoff_questions"] = sorted(i for i in byq if byq[i]["cut_off"])
    out["right_in_contested_by_rung"] = {r: [sum(G[g][r][i] for i in cont) for g in G] for r in LADDER}
    drift, up = [], []
    for i in cont:
        a, b = sum(G[g][r][i] for g in G for r in HI), sum(G[g][r][i] for g in G for r in LO)
        p = fisher_exact([[a, 12 - a], [b, 8 - b]])[1]
        if p < 0.05:
            (drift if a / 12 > b / 8 else up).append((i, a, b, round(p, 4)))
    out["fall_with_rung_p05"] = drift
    out["rise_with_rung_p05"] = up
    out["no_trend"] = len(cont) - len(drift) - len(up)
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
