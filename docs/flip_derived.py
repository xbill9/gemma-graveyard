#!/usr/bin/env python3
"""Every figure in devto-gemma-graveyard-flips.md, recomputed from committed runs into docs/evidence-flips/.

  python3 docs/flip_derived.py

Writes one text file per topic. The article quotes these files; nothing is typed in by hand.
"""

import json
import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "evidence-flips")
P, N, U, F = "runs/2026-11-02-probe", "runs/2026-11-03-nsm", "runs/2026-11-04-ub1", "runs/2026-11-04-fragile"
Q8 = "Q8_0.jsonl"

ROUTES = {  # label: Q8_0 run
    "T4, default settings": f"{P}/t4/default/{Q8}",
    "GTX 1650 Ti, default settings": f"{P}/1650/default/{Q8}",
    "L4, default settings": f"{P}/l4/default/{Q8}",
    "L4, -ub 1": f"{P}/l4/ub1/{Q8}",
    "T4, -ub 1": f"{P}/t4/ub1/{Q8}",
    "GTX 1650 Ti, -ub 1": f"{P}/1650/ub1/{Q8}",
    "GTX 1650 Ti told 40 SMs": f"{N}/nsm40/{Q8}",
    "GTX 1650 Ti, both fixes": f"{N}/nsm40-vec/{Q8}",
    "T4, matrix fix": f"{N}/t4-vec/{Q8}",
}


def path(p):
    return os.path.join(ROOT, p)


def text(p):
    with open(path(p), errors="replace") as f:
        return f.read()


def load(p):
    with open(path(p)) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote docs/evidence-flips/{name}")


def pct(lp):
    return round(math.exp(lp) * 100, 1)


def test358():
    q = next(r for r in map(json.loads, text("data/heldout.jsonl").splitlines()) if r["id"] == "test-358")
    a, b = load(ROUTES["GTX 1650 Ti, default settings"])["test-358"], load(ROUTES["T4, default settings"])["test-358"]
    ta, tb = a["tokens"], b["tokens"]
    i = 0
    while ta[i][0] == tb[i][0]:
        i += 1
    lines = [
        f"question: {q['question']}",
        f"gold answer: {q['answer']}",
        f"tokens identical on both cards before the split: {i}",
        f"text before the split (last 120 chars): {''.join(t[0] for t in ta[:i])[-120:]!r}",
    ]
    for card, t in (("GTX 1650 Ti", ta[i]), ("T4", tb[i])):
        lines.append(f"{card} at the split: " + ", ".join(f"{tok!r} {pct(lp)}%" for tok, lp in t[1]))
    for card, t in (("GTX 1650 Ti", ta), ("T4", tb)):
        lines.append(f"{card} next 12 tokens: {''.join(x[0] for x in t[i:i + 12])!r}")
    for card, r in (("GTX 1650 Ti", a), ("T4", b)):
        lines.append(f"{card} tokens generated: {len(r['tokens'])}; final line: {r['text'].strip().splitlines()[-1]!r}; "
                     f"extracted {r['extracted']}; right {r['right']}")
    lines.append("")
    lines.append("test-358 on every Q8_0 route:")
    for label, p in ROUTES.items():
        r = load(p)["test-358"]
        lines.append(f"  {label}: extracted {r.get('extracted')}, right {r.get('right')}")
    write("test-358.txt", lines)


def sum_order():
    """Add the same 8,960 float32 numbers (the inner size of per_layer_model_proj) in 16 shares and in 40 shares."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8960).astype(np.float32)

    def shares(n):
        acc = np.float32(0)
        for part in np.array_split(x, n):
            s = np.float32(0)
            for v in part:
                s = np.float32(s + v)
            acc = np.float32(acc + s)
        return acc

    s16, s40 = shares(16), shares(40)
    exact = math.fsum(float(v) for v in x)
    write("sum-order.txt", [
        "8,960 float32 numbers, numpy default_rng(0).standard_normal, each share added in order, then the shares",
        f"16 shares of 560: {s16:.9g}",
        f"40 shares of 224: {s40:.9g}",
        f"exact sum (math.fsum in float64): {exact:.9g}",
        f"16-share and 40-share sums equal: {bool(s16 == s40)}",
    ])


def probe(a, b):
    out = subprocess.run([sys.executable, path("tools/fork_probe.py"), path(a), path(b)],
                         capture_output=True, text=True, check=True)
    d = json.loads(out.stdout.splitlines()[0])
    return (f"text differs {d['text_differs']}, grade differs {d['grade_differs']}, "
            f"tokens compared {d['shared_tokens']}, same score {d['shared_tokens_same_logprob']}, "
            f"median lead at the split {d['fork_lead_a_median']} / {d['fork_lead_b_median']}, "
            f"top two swapped {d['forks_swapped_top2']} of {d['forks']}")


def right(p):
    return sum(bool(r.get("right")) for r in load(p).values())


def pairs():
    rows = [
        ("1650 default vs T4 default", ROUTES["GTX 1650 Ti, default settings"], ROUTES["T4, default settings"]),
        ("1650 told 40 SMs vs T4 default", ROUTES["GTX 1650 Ti told 40 SMs"], ROUTES["T4, default settings"]),
        ("1650 both fixes vs T4 matrix fix", ROUTES["GTX 1650 Ti, both fixes"], ROUTES["T4, matrix fix"]),
        ("1650 -ub 1 vs T4 -ub 1", ROUTES["GTX 1650 Ti, -ub 1"], ROUTES["T4, -ub 1"]),
        ("Q4_0: 1650 -ub 1 vs T4 -ub 1", f"{U}/1650-Q4_0/Q4_0.jsonl", f"{U}/t4-Q4_0/Q4_0.jsonl"),
        ("same card twice: 1650 default 2026-11-02 vs 1650 patched build, variables unset, 2026-11-03",
         ROUTES["GTX 1650 Ti, default settings"], f"{N}/own/{Q8}"),
        ("T4 default vs T4 -ub 1 (same card)", ROUTES["T4, default settings"], ROUTES["T4, -ub 1"]),
    ]
    lines = [f"{label}: {probe(a, b)}" for label, a, b in rows]
    lines.append("")
    lines += [f"right of 250, {label}: {right(p)}" for label, p in ROUTES.items()]
    write("pairs.txt", lines)


def prompt_tps(p):
    s = text(p)
    ev = [(float(a), int(b)) for a, b in re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", s)]
    return round(sum(b for _, b in ev) / (sum(a for a, _ in ev) / 1000))


def decode_tps(p):
    v = sorted(r["decode_tps"] for r in load(p).values() if r.get("decode_tps"))
    return round(v[len(v) // 2], 1)


def seconds(p):
    return sum(r.get("seconds", 0) for r in load(p).values())


def speed():
    rows = [
        ("GTX 1650 Ti", "default", f"{P}/1650/default"), ("GTX 1650 Ti", "both fixes", f"{N}/nsm40-vec"),
        ("GTX 1650 Ti", "-ub 1", f"{P}/1650/ub1"),
        ("T4", "default", f"{P}/t4/default"), ("T4", "matrix fix", f"{N}/t4-vec"), ("T4", "-ub 1", f"{P}/t4/ub1"),
    ]
    base = {}
    lines = ["card | route | prompt tokens/s | prompt slowdown | median generation tokens/s | whole run, summed seconds | whole run vs default"]
    for card, route, d in rows:
        tps, sec = prompt_tps(f"{d}/Q8_0.server.log"), seconds(f"{d}/{Q8}")
        if route == "default":
            base[card] = (tps, sec)
        bt, bs = base[card]
        lines.append(f"{card} | {route} | {tps} | {bt / tps:.1f}x | {decode_tps(f'{d}/{Q8}')} | {sec:.0f} | {100 * (sec / bs - 1):+.0f}%")
    write("speed.txt", lines)


def fragile():
    lines = []
    for name in ("q8-seven-routes.json", "q8-t4-vs-1650.json"):
        d = json.loads(text(f"{F}/{name}"))
        d.pop("rows", None)
        lines.append(f"== {F}/{name}")
        lines.append(json.dumps(d, indent=1))
    write("fragile.txt", lines)


def kquant():
    lines = ["rung | right -ub 1 | right default | text/grade differ vs default | right both fixes | text/grade differ vs fixes"]
    for g in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"):
        if g == "Q8_0":
            u, d, f = ROUTES["GTX 1650 Ti, -ub 1"], ROUTES["GTX 1650 Ti, default settings"], ROUTES["GTX 1650 Ti, both fixes"]
        elif g == "Q4_0":
            u, d, f = f"{U}/1650-Q4_0/Q4_0.jsonl", f"runs/2026-10-29-1650/{g}.jsonl", f"{N}/nsm40-vec-q4/{g}.jsonl"
        else:
            u, d, f = f"{U}/1650-{g}/{g}.jsonl", f"runs/2026-10-29-1650/{g}.jsonl", f"{N}/ladder/1650-{g}/{g}.jsonl"
        A, D, X = load(u), load(d), load(f)

        def diff(B, A=A):
            ids = set(A) & set(B)
            return (f"{sum(A[i].get('text') != B[i].get('text') for i in ids)}/"
                    f"{sum(bool(A[i].get('right')) != bool(B[i].get('right')) for i in ids)}")
        lines.append(f"{g} | {right(u)} | {right(d)} | {diff(D)} | {right(f)} | {diff(X)}")
    write("ladder-routes.txt", lines)


def hosts():
    write("cards.txt", text(f"{N}/1650-sm.txt").splitlines()
          + [x for x in text(f"{N}/t4-host.txt").splitlines() if "sm=" in x])


def counts():
    """The article's "N of M" figures, each computed here."""
    d = load(ROUTES["GTX 1650 Ti, default settings"])
    t = load(ROUTES["T4, default settings"])
    ids = sorted(d)
    differ = sum(d[i]["text"] != t[i]["text"] for i in ids)
    ta, tb = d["test-358"]["tokens"], t["test-358"]["tokens"]
    split = next(i for i in range(min(len(ta), len(tb))) if ta[i][0] != tb[i][0])
    u1, u2 = load(ROUTES["GTX 1650 Ti, -ub 1"]), load(ROUTES["T4, -ub 1"])
    same_ub1 = sum(u1[i]["text"] == u2[i]["text"] for i in u1)
    spread = json.loads(text(f"{F}/q8-seven-routes.json"))
    flag = next(x for x in spread["flag_any_token"] if x["t"] == 0.1)
    log = text(f"{N}/nsm40-vec/Q8_0.server.log").splitlines()
    cub = [x for x in log if re.search(r"graveyard (split_to_vec|cublas)", x)]
    q = next(r for r in map(json.loads, text("data/heldout.jsonl").splitlines()) if r["id"] == "test-358")
    t2 = subprocess.run([sys.executable, path("tools/fork_probe.py"), path(ROUTES["T4, default settings"]),
                         path(f"{N}/t4-patched/{Q8}")], capture_output=True, text=True, check=True)
    t2 = json.loads(t2.stdout.splitlines()[0])
    write("counts.txt", [
        f"same card twice, T4: same score on {t2['shared_tokens_same_logprob']:,} of {t2['shared_tokens']:,} tokens",
        f"test-358 gold answer in dollars: ${q['answer']}",
        f"Q8_0 default, 1650 vs T4: text differs on {differ} of {len(ids)}; text matches on {len(ids) - differ} of {len(ids)}",
        f"test-358, T4 tokens after the split: {len(tb) - split} tokens",
        f"Q8_0 -ub 1, 1650 vs T4: text matches on {same_ub1} of {len(u1)}",
        f"seven setups: text varies on {spread['text_varies']} of {spread['questions']}",
        f"flag at lead 0.1: {flag['flagged']} of {spread['questions']} flagged, catching {flag['grade_varies_caught']}",
        (f"1650 both fixes, logged products sent off cuBLAS: {sum('per_layer_model_proj.weight' in x for x in cub)} of {len(cub)}"
         " are per_layer_model_proj"),
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    test358()
    sum_order()
    pairs()
    speed()
    fragile()
    kquant()
    hosts()
    counts()


if __name__ == "__main__":
    main()
