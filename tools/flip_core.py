#!/usr/bin/env python3
"""flip_core.py: the contested questions. The T4 has grades only, so grade work uses four cards and answer-text work three.
Writes nothing; prints JSON. Every count is computed here."""
import collections as C
import itertools
import json
import random
import statistics as st
import sys

sys.path[:0] = ["docs", "."]
from gpu_derived import IDS, LADDER, t4_grid  # noqa: E402
from lazy_eval import load  # noqa: E402

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}
LIGHT = LADDER[:7]  # BF16 to Q4_0


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def main():
    Q = {x["id"]: x for x in load("data/heldout.jsonl")}
    T = {g: {r: rows(g, r) for r in LADDER} for g in D}
    G = {"t4": t4_grid("graveyard-ladder.json", LADDER), **{g: {r: {i: x["right"] for i, x in T[g][r].items()} for r in LADDER} for g in D}}
    cards = list(G)
    n = len(cards) * len(LIGHT)
    score = {i: sum(G[g][r][i] for g in cards for r in LIGHT) for i in IDS}
    contested = [i for i in IDS if 0 < score[i] < n]
    out = {"runs": n, "always_right": sum(v == n for v in score.values()), "always_wrong": sum(v == 0 for v in score.values()), "contested": len(contested)}

    # answers per contested question, three cards with text
    per, top2, wrong_share, majority_wrong, repeat_wrong = [], [], [], 0, 0
    table = []
    for i in contested:
        ans = C.Counter(str(T[g][r][i].get("extracted")) for g in D for r in LIGHT)
        per.append(len(ans))
        top = ans.most_common()
        top2.append(sum(c for _, c in top[:2]) / 21)
        gold = str(float(Q[i]["answer"].replace(",", "")))
        wrong = [(a, c) for a, c in top if a != gold]
        if wrong and wrong[0][1] >= 2:
            repeat_wrong += 1
        majority_wrong += ans[gold] < 10.5
        table.append({"id": i, "right_of_28": score[i], "gold": gold, "answers": top[:4]})
    out["distinct_answers_median"] = st.median(per)
    out["distinct_answers_hist"] = sorted(C.Counter(per).items())
    out["top_two_answers_share_median"] = round(st.median(top2), 3)
    out["questions_top_two_cover_all"] = sum(x == 1 for x in top2)
    out["questions_top_two_cover_90pct"] = sum(x >= 0.9 for x in top2)
    out["wrong_answer_repeats"] = repeat_wrong
    out["right_answer_in_minority"] = majority_wrong

    # agreement between two runs of one contested question: which change matters more?
    def agree(f):
        a = b = 0
        for i in contested:
            for (g, r), (h, s) in itertools.combinations(itertools.product(cards, LIGHT), 2):
                if f(g, r, h, s):
                    b += 1
                    a += G[g][r][i] == G[h][s][i]
        return round(a / b, 4), b

    ok = {"same_rung_other_card": lambda g, r, h, s: r == s and g != h,
          "same_card_other_rung": lambda g, r, h, s: g == h and r != s,
          "both_differ": lambda g, r, h, s: g != h and r != s}
    out["agreement_contested"] = {k: agree(f) for k, f in ok.items()}
    adj = {k: [] for k in ("BF16_vs_Q8_0_same_card", "Q8_0_other_card")}
    for g, h in itertools.combinations(cards, 2):
        adj["Q8_0_other_card"].append(sum(G[g]["Q8_0"][i] != G[h]["Q8_0"][i] for i in IDS))
    for g in cards:
        adj["BF16_vs_Q8_0_same_card"].append(sum(G[g]["BF16"][i] != G[g]["Q8_0"][i] for i in IDS))
    out["grades_moved"] = adj

    # does the reference card's total hold if cards are reshuffled per question? (null: cards exchangeable)
    rng = random.Random(0)
    sds = {}
    for r in LADDER:
        obs = st.pstdev(sum(G[g][r].values()) for g in cards)
        sims = []
        for _ in range(4000):
            tot = C.Counter()
            for i in IDS:
                v = [G[g][r][i] for g in cards]
                rng.shuffle(v)
                for g, x in zip(cards, v):
                    tot[g] += x
            sims.append(st.pstdev(tot[g] for g in cards))
        sims.sort()
        sds[r] = {"sd_observed": round(obs, 2), "sd_if_cards_shuffled_median": round(st.median(sims), 2),
                  "share_sims_at_least_observed": round(sum(s >= obs for s in sims) / len(sims), 3)}
    out["card_shuffle"] = sds
    print(json.dumps({"summary": out, "table": table}, indent=0))


if __name__ == "__main__":
    main()
