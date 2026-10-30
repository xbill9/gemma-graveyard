#!/usr/bin/env python3
"""The RTX 3050 ladder beside the T4, the L4 and the GTX 1650, question by question. Same method as gpu3_derived.py:
the T4's grid is rebuilt from its tombstones, the others come from their rows.
python3 docs/gpu4_derived.py > runs/2026-10-29-3050/four-gpus.txt"""
import json
from itertools import combinations

from gpu3_derived import G1650, rows_grid
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
    t4_grid,
)

G3050 = "runs/2026-10-29-3050"
DIRS = {"t4": T4, "l4": L4, "1650": G1650, "3050": G3050}
PAIRS = list(combinations(DIRS, 2))


def main():
    out, t4_bf16 = {"order": list(DIRS), "pairs": ["-".join(p) for p in PAIRS]}, None
    for name, rungs in (("ladder", LADDER), ("qat", QAT)):
        t4 = t4_grid("graveyard-ladder.json" if rungs is LADDER else "graveyard-qat.json", rungs, t4_bf16)
        t4_bf16 = t4["BF16"]
        assert name != "ladder" or cover_cells(t4) == 0, "T4 grid disagrees with the cover"
        g = {"t4": t4, **{k: rows_grid(DIRS[k], rungs) for k in ("l4", "1650", "3050")}}
        c = {k: counts(m, rungs) for k, m in g.items()}
        out[name] = {
            "right": {r: [c[k][r]["right"] for k in g] for r in rungs},
            "lost": {r: [c[k][r]["dead"] for k in g] for r in rungs[1:]},
            "gained": {r: [c[k][r]["ghosts"] for k in g] for r in rungs[1:]},
            "graded_differently_by_pair": {r: [sum(g[a][r][i] != g[b][r][i] for i in IDS) for a, b in PAIRS] for r in rungs},
            "lost_on_all_four": {r: len(set.intersection(*(dead(g[k], r) for k in g))) for r in rungs[1:]},
        }
        if name == "ladder":
            st = {k: {t["id"]: t["died_at"] for t in jload(f"{d}/graveyard-ladder.json")["tombstones"]} for k, d in DIRS.items()}
            others = set().union(*(st[k] for k in ("t4", "l4", "1650")))
            out[name]["tombstones"] = [len(st[k]) for k in st]
            out[name]["tombstoned_on_all_four"] = len(set.intersection(*(set(v) for v in st.values())))
            out[name]["tombstoned_on_3050_only"] = sorted(set(st["3050"]) - others)
            out[name]["first_lost_same_rung_on_all_four"] = sum(all(st[k].get(i) == r for k in st) for i, r in st["t4"].items())
            out[name]["bf16_right_on_all_four"] = sum(all(g[k]["BF16"][i] for k in g) for i in IDS)
            out[name]["bf16_wrong_on_all_four"] = sum(not any(g[k]["BF16"][i] for k in g) for i in IDS)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
