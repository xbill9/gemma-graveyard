#!/usr/bin/env python3
"""fork_input.py <prompts.json> <rung> <n_control> > in.tsv: replay.cpp input for one rung, the L4's recorded answers cut
40 characters past the last point where the 1650 or 3050 text leaves them. Questions all three cards answer with identical
text are kept whole, the first n_control of them, as the baseline of tokens where no card forked."""
import json
import sys

D = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}


def rows(g, r):
    with open(f"{D[g]}/{r}.jsonl") as f:
        return {x["id"]: x for x in map(json.loads, f)}


def fork(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), None if len(a) == len(b) else n)


def main():
    with open(sys.argv[1]) as f:
        prompts = json.load(f)
    r, nc = sys.argv[2], int(sys.argv[3])
    g = {k: rows(k, r) for k in D}
    kept = 0
    for i, x in g["l4"].items():
        t = x.get("text")
        if not t:
            continue
        ks = [fork(t, g[k][i].get("text") or "") for k in ("1650", "3050")]
        if any(k is not None for k in ks):
            t = t[: max(k for k in ks if k is not None) + 40]
        elif kept < nc:
            kept += 1
        else:
            continue
        print(f"{r}:{i}\t{prompts[i]['p'].encode().hex()}\t{t.encode().hex()}")


if __name__ == "__main__":
    main()
