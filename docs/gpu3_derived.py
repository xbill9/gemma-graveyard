#!/usr/bin/env python3
"""The GTX 1650 ladder against the T4 and the L4, question by question. The T4's grid is gpu_derived.py's (rebuilt
from its tombstones and checked against its bury tables and cover); the L4's and the 1650's come from their rows.
python3 docs/gpu3_derived.py > runs/2026-10-29-1650/three-gpus.txt"""
import json

from gpu_derived import (
    IDS,
    L4,
    LADDER,
    QAT,
    T4,
    counts,
    cover_cells,
    dead,
    jload,
    lines,
    t4_grid,
)

G1650 = "runs/2026-10-29-1650"
PAIRS = (("t4", "l4"), ("t4", "1650"), ("l4", "1650"))


def rows_grid(d, rungs):
    return {r: {x["id"]: x["right"] for x in lines(f"{d}/{r}.jsonl")} for r in rungs}


def main():
    out, t4_bf16 = {}, None
    for name, rungs in (("ladder", LADDER), ("qat", QAT)):
        t4 = t4_grid("graveyard-ladder.json" if rungs is LADDER else "graveyard-qat.json", rungs, t4_bf16)
        t4_bf16 = t4["BF16"]
        assert name != "ladder" or cover_cells(t4) == 0, "T4 grid disagrees with the cover"
        g = {"t4": t4, "l4": rows_grid(L4, rungs), "1650": rows_grid(G1650, rungs)}
        c = {k: counts(m, rungs) for k, m in g.items()}
        out[name] = {
            "right_t4_l4_1650": {r: [c[k][r]["right"] for k in g] for r in rungs},
            "lost_t4_l4_1650": {r: [c[k][r]["dead"] for k in g] for r in rungs[1:]},
            "gained_t4_l4_1650": {r: [c[k][r]["ghosts"] for k in g] for r in rungs[1:]},
            "graded_differently_t4l4_t41650_l41650": {r: [sum(g[a][r][i] != g[b][r][i] for i in IDS) for a, b in PAIRS] for r in rungs},
            "lost_on_all_three": {r: len(dead(g["t4"], r) & dead(g["l4"], r) & dead(g["1650"], r)) for r in rungs[1:]},
        }
        if name == "ladder":
            st = {k: {t["id"]: t["died_at"] for t in jload(f"{d}/graveyard-ladder.json")["tombstones"]} for k, d in (("t4", T4), ("l4", L4), ("1650", G1650))}
            out[name]["tombstones_t4_l4_1650"] = [len(st[k]) for k in st]
            out[name]["tombstoned_on_all_three"] = len(set(st["t4"]) & set(st["l4"]) & set(st["1650"]))
            out[name]["tombstoned_on_1650_only"] = sorted(set(st["1650"]) - set(st["t4"]) - set(st["l4"]))
            out[name]["first_lost_same_rung_on_all_three"] = sum(st["l4"].get(i) == r == st["1650"].get(i) for i, r in st["t4"].items())
            out[name]["bf16_right_on_all_three"] = sum(all(g[k]["BF16"][i] for k in g) for i in IDS)
            out[name]["bf16_wrong_on_all_three"] = sum(not any(g[k]["BF16"][i] for k in g) for i in IDS)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
