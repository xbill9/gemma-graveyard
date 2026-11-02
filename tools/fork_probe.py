#!/usr/bin/env python3
"""fork_probe.py: where two GPUs' answers part, what scores did each GPU give the top two tokens?

  fork_probe.py <a.jsonl> <b.jsonl> [--plain-a P] [--plain-b P] [--out FILE]

Both inputs come from `graveyard.py run --top-logprobs 2` on the same GGUF, build and questions, on
two cards. Each row holds per token [chosen, [[token, logprob], ...]]. Logprobs are log-softmax over the
raw logits, so the gap between the top two logprobs equals the gap between their logits.

Prints, computed here:
  - with --plain-a/--plain-b: on how many questions each probe run's text equals that card's plain run
    (the check that asking for logprobs did not change the answers);
  - for every question whose texts differ: the first token where the cards choose differently, each card's
    top two there and its lead, and whether the two cards hold the same two tokens in swapped order;
  - over every token both cards produced from the same text so far: how often the chosen token's
    logprob is the same number on both cards, and how far apart it is otherwise.
"""

import argparse
import json
import statistics


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def lead(top):
    return top[0][1] - top[1][1] if len(top) > 1 else None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def compare(a, b):
    forks, same_lp, diff_lp, n_shared = [], 0, [], 0
    for qid in sorted(set(a) & set(b)):
        ta, tb = a[qid].get("tokens"), b[qid].get("tokens")
        if not ta or not tb:
            continue
        i = 0
        while i < min(len(ta), len(tb)) and ta[i][0] == tb[i][0]:
            la, lb = ta[i][1][0][1], tb[i][1][0][1]  # chosen token is top 1 under greedy decoding
            n_shared += 1
            if la == lb:
                same_lp += 1
            else:
                diff_lp.append(abs(la - lb))
            i += 1
        if i == len(ta) == len(tb):
            continue
        if i >= min(len(ta), len(tb)):  # one text is a prefix of the other: a length difference, not a choice
            forks.append({"id": qid, "at": i, "kind": "length"})
            continue
        top_a, top_b = ta[i][1][:2], tb[i][1][:2]
        forks.append(
            {
                "id": qid,
                "at": i,
                "kind": "choice",
                "context": "".join(t[0] for t in ta[max(0, i - 12) : i]),
                "a_top2": top_a,
                "b_top2": top_b,
                "a_lead": lead(top_a),
                "b_lead": lead(top_b),
                "swapped": len(top_a) > 1
                and len(top_b) > 1
                and top_a[0][0] == top_b[1][0]
                and top_a[1][0] == top_b[0][0],
                "grade_a": a[qid].get("right"),
                "grade_b": b[qid].get("right"),
            }
        )
    return forks, same_lp, diff_lp, n_shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--plain-a")
    p.add_argument("--plain-b")
    p.add_argument("--out")
    x = p.parse_args()
    a, b = load(x.a), load(x.b)
    summary = {"a": x.a, "b": x.b, "questions": len(set(a) & set(b))}
    for side, probe, plain in (("a", a, x.plain_a), ("b", b, x.plain_b)):
        if plain:
            pl = load(plain)
            common = set(probe) & set(pl)
            summary[f"{side}_text_equals_plain"] = f"{sum(probe[i].get('text') == pl[i].get('text') for i in common)}/{len(common)}"
    summary["text_differs"] = sum(a[i].get("text") != b[i].get("text") for i in set(a) & set(b))
    forks, same_lp, diff_lp, n_shared = compare(a, b)
    choice = [f for f in forks if f["kind"] == "choice"]
    summary.update(
        forks=len(forks),
        forks_choice=len(choice),
        forks_swapped_top2=sum(f["swapped"] for f in choice),
        fork_lead_a_median=statistics.median(f["a_lead"] for f in choice) if choice else None,
        fork_lead_b_median=statistics.median(f["b_lead"] for f in choice) if choice else None,
        fork_lead_max=max(max(f["a_lead"], f["b_lead"]) for f in choice) if choice else None,
        grade_differs=sum(f.get("grade_a") != f.get("grade_b") for f in choice),
        shared_tokens=n_shared,
        shared_tokens_same_logprob=same_lp,
        shared_logprob_absdiff_median=statistics.median(diff_lp) if diff_lp else None,
        shared_logprob_absdiff_p99=pct(diff_lp, 0.99),
        shared_logprob_absdiff_max=max(diff_lp) if diff_lp else None,
    )
    print(json.dumps(summary))
    for f in sorted(choice, key=lambda f: (f["grade_a"] == f["grade_b"], f["id"])):
        print(json.dumps(f, ensure_ascii=False))
    if x.out:
        with open(x.out, "w") as o:
            json.dump({"summary": summary, "forks": forks}, o, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
