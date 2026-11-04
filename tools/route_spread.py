#!/usr/bin/env python3
"""route_spread.py REF.jsonl RUN.jsonl ... : across several runs of the same model on different cards or settings,
which questions change answer or grade, and does one run's own scores (REF, which needs --top-logprobs 2) flag them?

Per question: the number of distinct answer texts and of distinct grades across all runs; REF's smallest lead
(top-1 minus top-2 log-probability) over every token, and over the tokens of its final answer line."""

import json
import sys

THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 1.0]


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def lead(t):
    return t[1][0][1] - t[1][1][1] if len(t[1]) > 1 else float("inf")


def tail_min(tokens, n=12):
    return min((lead(t) for t in tokens[-n:]), default=float("inf"))


def main():
    ref, *others = sys.argv[1:]
    runs = [load(ref)] + [load(p) for p in others]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    rows = []
    for i in ids:
        toks = runs[0][i].get("tokens") or []
        rows.append(
            {
                "id": i,
                "texts": len({r[i].get("text") for r in runs}),
                "grades": len({bool(r[i].get("right")) for r in runs}),
                "right_in": sum(bool(r[i].get("right")) for r in runs),
                "min_lead": min((lead(t) for t in toks), default=float("inf")),
                "end_min_lead": tail_min(toks),
            }
        )
    unstable = [r for r in rows if r["grades"] > 1]
    stable = [r for r in rows if r["grades"] == 1]
    out = {
        "runs": len(runs),
        "questions": len(rows),
        "text_varies": sum(r["texts"] > 1 for r in rows),
        "grade_varies": len(unstable),
        "grade_varies_ids": [(r["id"], f"right in {r['right_in']}/{len(runs)}") for r in unstable],
        "flag_any_token": [],
        "flag_last_12_tokens": [],
    }
    for key, name in (("min_lead", "flag_any_token"), ("end_min_lead", "flag_last_12_tokens")):
        for t in THRESHOLDS:
            out[name].append(
                {
                    "t": t,
                    "flagged": sum(r[key] < t for r in rows),
                    "grade_varies_caught": f"{sum(r[key] < t for r in unstable)}/{len(unstable)}",
                    "stable_flagged": f"{sum(r[key] < t for r in stable)}/{len(stable)}",
                }
            )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
