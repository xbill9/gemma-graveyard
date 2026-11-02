#!/usr/bin/env python3
"""flip_modes.py: compare the AWS flip-study runs answer by answer (text, extracted answer, grade).

Reads runs/2026-11-01-flips/{l4,t4}/<mode>/<rung>.jsonl; prints one line per pair of runs.
"""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-11-01-flips"
PAIRS = [
    ("repeat: same card, same build", [("t4/default", "t4/repeat"), ("l4/default", "l4/repeat")], ["Q8_0", "Q4_0"]),
    ("L4 new vs L4 recorded 2026-10-28", [("l4/default", "../2026-10-28-l4")], ["BF16", "Q8_0", "Q4_0"]),
    ("L4 vs T4, normal settings", [("l4/default", "t4/default")], ["BF16", "Q8_0", "Q4_0"]),
    ("L4 vs T4, both -ub 1", [("l4/ub1", "t4/ub1")], ["BF16", "Q8_0", "Q4_0"]),
    ("same card, normal vs -ub 1", [("l4/default", "l4/ub1"), ("t4/default", "t4/ub1")], ["BF16", "Q8_0", "Q4_0"]),
    ("L4, cuBLAS 16-bit vs 32-bit", [("l4/default", "l4/f32")], ["Q8_0", "Q4_0"]),
    ("L4 built for T4 (sm_75), normal settings", [("l4/cc75-default", "t4/default"), ("l4/cc75-default", "l4/default")], ["BF16", "Q8_0"]),
    ("L4 built for T4 (sm_75), both -ub 1", [("l4/cc75-ub1", "t4/ub1"), ("l4/cc75-ub1", "l4/ub1")], ["Q8_0"]),
]


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            rows[x["id"]] = x
    return rows


def main():
    for title, pairs, rungs in PAIRS:
        print(f"## {title}")
        for a, b in pairs:
            for r in rungs:
                fa, fb = f"{R}/{a}/{r}.jsonl", f"{R}/{b}/{r}.jsonl"
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    continue
                A, B = load(fa), load(fb)
                ids = sorted(set(A) & set(B))
                text = sum(A[i]["text"] != B[i]["text"] for i in ids)
                ans = sum(A[i]["extracted"] != B[i]["extracted"] for i in ids)
                grade = sum(A[i]["right"] != B[i]["right"] for i in ids)
                ra, rb = sum(A[i]["right"] for i in ids), sum(B[i]["right"] for i in ids)
                print(f"{a:16} vs {b:18} {r:5} n {len(ids)}  text {text:3}  answer {ans:3}  grade {grade:3}  right {ra}/{rb}")


if __name__ == "__main__":
    main()
