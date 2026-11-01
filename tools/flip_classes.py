#!/usr/bin/env python3
"""flip_classes.py: for every pair of the three cards with recorded text, at every rung, find where the two answers first
differ and classify the fork. Prints JSON; every count is computed here.
A fork is 'word' (either branch starts with a letter or markup), or 'number' (either branch's first differing char is a digit,
operator or math symbol). 'False equations' are 'a op b = c' claims in the answer that do not hold when computed."""
import itertools
import json
import re
import statistics as st

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}
R = ["BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M"]
EQ = re.compile(r"(-?\d[\d,]*\.?\d*)\s*(\+|-|\\times|\*|×|/|\\div)\s*(-?\d[\d,]*\.?\d*)\s*=\s*\\?\$?(-?\d[\d,]*\.?\d*)")


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def fork(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), None if len(a) == len(b) else n)


def num(s):
    return float(s.replace(",", "").rstrip("."))


def bad_eqs(t):
    out = []
    for m in EQ.finditer(t or ""):
        try:
            a, op, b, c = num(m[1]), m[2], num(m[3]), num(m[4])
        except ValueError:
            continue
        v = a + b if op == "+" else a - b if op == "-" else a * b if op in ("\\times", "*", "×") else (a / b if b else None)
        if v is not None and abs(v - c) > max(0.0051, abs(v) * 0.006):
            out.append(m[0])
    return out


def kind(s):
    c = s[:1]
    return "number" if c.isdigit() or c in "$\\+-=*/×.,%()" else "word"


def main():
    g = {k: {r: rows(k, r) for r in R} for k in D}
    ids = list(g["l4"]["BF16"])
    out = {}
    for r in R:
        forks = flips = same_ans_diff_text = 0
        flip_kinds, all_kinds, rel, rel_flip = [], [], [], []
        for a, b in itertools.combinations(g, 2):
            for i in ids:
                x, y = g[a][r][i], g[b][r][i]
                s, t = x.get("text") or "", y.get("text") or ""
                k = fork(s, t)
                if k is None:
                    continue
                forks += 1
                pair = (kind(s[k:]), kind(t[k:]))
                ki = "word" if "word" in pair else "number"
                all_kinds.append(ki)
                rel.append(k / max(len(s), len(t), 1))
                if x["right"] != y["right"]:
                    flips += 1
                    flip_kinds.append(ki)
                    rel_flip.append(k / max(len(s), len(t), 1))
                elif x.get("extracted") == y.get("extracted"):
                    same_ans_diff_text += 1
        out[r] = {
            "forks_3pairs": forks, "grade_flips": flips, "flip_per_fork": round(flips / forks, 4) if forks else None,
            "same_answer_after_fork": same_ans_diff_text,
            "word_forks": all_kinds.count("word"), "word_share_all": round(all_kinds.count("word") / max(forks, 1), 3),
            "word_share_flips": round(flip_kinds.count("word") / max(flips, 1), 3),
            "median_rel_pos_all": round(st.median(rel), 3) if rel else None,
            "median_rel_pos_flips": round(st.median(rel_flip), 3) if rel_flip else None,
        }
        wrongs = [(k, i) for k in g for i in ids if not g[k][r][i]["right"]]
        rights = [(k, i) for k in g for i in ids if g[k][r][i]["right"]]
        out[r]["false_eq_share_wrong"] = round(sum(bool(bad_eqs(g[k][r][i].get("text"))) for k, i in wrongs) / max(len(wrongs), 1), 3)
        out[r]["false_eq_share_right"] = round(sum(bool(bad_eqs(g[k][r][i].get("text"))) for k, i in rights) / max(len(rights), 1), 3)
        out[r]["false_eq_per_1k_claims"] = round(1000 * sum(len(bad_eqs(g[k][r][i].get("text"))) for k in g for i in ids)
                                                / max(sum(len(EQ.findall(g[k][r][i].get("text") or "")) for k in g for i in ids), 1), 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
