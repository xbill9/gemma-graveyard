#!/usr/bin/env python3
"""Can one run's own scores say which answers another card or setting will change?

fragility.py A.jsonl B.jsonl [--out F.json]

Both runs need per-token top-2 scores (graveyard.py --top-logprobs 2). For every question, from run A alone:
the smallest lead (top-1 minus top-2 log-probability) over every token A generated. Then, against run B:
whether the answer text differs, and whether the grade differs.

For each threshold t, the question is flagged "fragile" when its smallest lead is below t. Reported per t:
  flagged            questions flagged
  text_caught        of the questions whose text differs, how many were flagged
  text_false_alarm   of the questions whose text is the same, how many were flagged
  grade_caught       of the questions whose grade differs, how many were flagged
Also: the lead each card had at the token where the two texts part (the fork), and A's smallest lead
on the questions that did not change.
"""

import argparse
import json

THRESHOLDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def leads(tokens):
    return [t[1][0][1] - t[1][1][1] for t in tokens if len(t[1]) > 1]


def fork(ta, tb):
    i = 0
    while i < min(len(ta), len(tb)) and ta[i][0] == tb[i][0]:
        i += 1
    if i >= min(len(ta), len(tb)):
        return None
    return i


def q(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 4) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    args = p.parse_args()
    a, b = load(args.a), load(args.b)
    ids = sorted(i for i in set(a) & set(b) if a[i].get("tokens") and b[i].get("tokens"))
    rows = []
    for i in ids:
        ta, tb = a[i]["tokens"], b[i]["tokens"]
        la = leads(ta)
        f = fork(ta, tb)
        rows.append(
            {
                "id": i,
                "min_lead": min(la) if la else None,
                "text_differs": a[i].get("text") != b[i].get("text"),
                "grade_differs": a[i].get("right") != b[i].get("right"),
                "fork_lead_a": (ta[f][1][0][1] - ta[f][1][1][1]) if f is not None and len(ta[f][1]) > 1 else None,
                "fork_lead_b": (tb[f][1][0][1] - tb[f][1][1][1]) if f is not None and len(tb[f][1]) > 1 else None,
                "fork_at": f,
                "tokens": len(ta),
            }
        )
    rows = [r for r in rows if r["min_lead"] is not None]
    td = [r for r in rows if r["text_differs"]]
    ts = [r for r in rows if not r["text_differs"]]
    gd = [r for r in rows if r["grade_differs"]]
    fl = [max(r["fork_lead_a"], r["fork_lead_b"]) for r in td if r["fork_lead_a"] is not None and r["fork_lead_b"] is not None]
    summary = {
        "a": args.a,
        "b": args.b,
        "questions": len(rows),
        "text_differs": len(td),
        "grade_differs": len(gd),
        "fork_lead_larger_of_two_median": q(fl, 0.5),
        "fork_lead_larger_of_two_max": q(fl, 1.0),
        "min_lead_unchanged_median": q([r["min_lead"] for r in ts], 0.5),
        "min_lead_changed_median": q([r["min_lead"] for r in td], 0.5),
        "by_threshold": [
            {
                "t": t,
                "flagged": sum(r["min_lead"] < t for r in rows),
                "text_caught": f"{sum(r['min_lead'] < t for r in td)}/{len(td)}",
                "text_false_alarm": f"{sum(r['min_lead'] < t for r in ts)}/{len(ts)}",
                "grade_caught": f"{sum(r['min_lead'] < t for r in gd)}/{len(gd)}",
            }
            for t in THRESHOLDS
        ],
    }
    out = json.dumps(summary, indent=1)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({**summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
