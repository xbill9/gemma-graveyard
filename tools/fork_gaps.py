#!/usr/bin/env python3
"""fork_gaps.py <rung>: join a replay of the L4's answers to the points where the 1650 and 3050 left them.
Baseline: every token of every replayed answer. Fork token: the recorded token whose span contains the first differing
character. Gap: top-1 logit minus top-2 logit on this CPU; rec_gap: top-1 logit minus the recorded token's logit.
Prints JSON; every figure is computed here."""
import json
import statistics as st
import sys

import numpy as np

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def fork(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), None if len(a) == len(b) else n)


def b(s):
    return s.encode("utf-8", "surrogateescape")


def q(v):
    v = np.array(v, float)
    return {"n": len(v), "p10": round(float(np.percentile(v, 10)), 3), "median": round(float(np.median(v)), 3),
            "p90": round(float(np.percentile(v, 90)), 3), "max": round(float(v.max()), 3)} if len(v) else {"n": 0}


def main():
    r = sys.argv[1]
    g = {k: rows(k, r) for k in D}
    gaps, recgap_all, ranks = [], [], []
    fgap, frec, fsame, other_is_top2 = [], [], 0, 0
    nf = unaligned = 0
    pos = {}
    with open(f"runs/2026-11-01-flips/replay/{r}.jsonl", errors="surrogateescape") as f:
        reps = [json.loads(x) for x in f]
    for x in reps:
        i = x["key"].split(":")[1]
        toks = x["t"][:-3]  # the text is cut mid-token for replay; its last tokens are artefacts of the cut
        gaps += [t[5] for t in toks]
        recgap_all += [t[2] for t in toks]
        ranks += [t[1] for t in toks]
        text = g["l4"][i]["text"].encode()  # tokens can split a character, so spans are in bytes
        spans, c = [], 0
        for t in toks:
            n = len(b(t[0]))
            spans.append((c, c + n))
            c += n
        if not text.startswith(b"".join(b(t[0]) for t in toks)):
            unaligned += 1
            continue
        for o in ("1650", "3050"):
            other = (g[o][i].get("text") or "").encode()
            k = fork(text, other)
            if k is None or k >= c:
                continue
            j = next(n for n, (p, e) in enumerate(spans) if p <= k < e)
            nf += 1
            t = toks[j]
            fgap.append(t[5])
            frec.append(t[2])
            alt = other[spans[j][0]:spans[j][0] + 64]
            cand = {b(t[3]), b(t[4])}
            if b(t[0]) in cand and any(alt.startswith(z) for z in cand if z != b(t[0])):
                other_is_top2 += 1
            pos[(i, o)] = j
    fg, ag = np.array(fgap), np.array(gaps)
    out = {
        "rung": r, "answers": len(reps), "tokens": len(gaps), "unaligned_answers": unaligned, "forks": nf,
        "cpu_top1_is_recorded_token": round(float(np.mean(np.array(ranks) == 1)), 4),
        "all_tokens_gap": q(gaps), "fork_tokens_gap": q(fgap), "fork_rec_gap": q(frec),
        "both_branches_are_cpu_top_two": other_is_top2,
        "share_all_gap_under": {str(c): round(float(np.mean(ag < c)), 4) for c in (0.1, 0.25, 0.5, 1, 2)},
        "share_fork_gap_under": {str(c): round(float(np.mean(fg < c)), 4) for c in (0.1, 0.25, 0.5, 1, 2)},
    }
    nonrec = [t for x in reps for t in x["t"] if t[1] != 1]
    out["recorded_not_top1_on_cpu"] = {"n": len(nonrec), "rec_gap": q([t[2] for t in nonrec])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
