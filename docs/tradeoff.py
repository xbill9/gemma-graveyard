#!/usr/bin/env python3
"""Speed against accuracy per rung, thinking off, from bury.txt. A rung is dominated when another rung is at
least as fast and gets at least as many right. Also tests Q5_K_M against BF16. Writes docs/evidence/tradeoff.txt:
python3 docs/tradeoff.py > docs/evidence/tradeoff.txt"""
import json
from math import comb

with open("docs/evidence/bury.txt") as f:
    rows = {x["rung"]: x for x in map(json.loads, f) if "rung" in x}
t = {r: (v["right"], v["median_decode_tps"]) for r, v in rows.items()}
for r, (right, tps) in t.items():
    dom = [s for s, (r2, t2) in t.items() if s != r and r2 >= right and t2 >= tps]
    print(json.dumps({"rung": r, "right": right, "median_decode_tps": tps,
                      "decode_x_bf16": round(tps / t["BF16"][1], 2), "dominated_by": dom}))
print(json.dumps({"q4_0_decode_x_q5_k_m": round(t["Q4_0"][1] / t["Q5_K_M"][1], 2),
                  "right_q5_k_m_minus_q4_0": t["Q5_K_M"][0] - t["Q4_0"][0]}))
# Exact two-sided McNemar test, Q5_K_M against BF16: only the questions the two disagree on count.
d, g = rows["Q5_K_M"]["dead"], rows["Q5_K_M"]["ghosts"]
p = min(1.0, 2 * sum(comb(d + g, i) for i in range(min(d, g) + 1)) / 2 ** (d + g))
print(json.dumps({"q5_k_m_vs_bf16": {"dead": d, "ghosts": g, "exact_mcnemar_p": round(p, 2)}}))
