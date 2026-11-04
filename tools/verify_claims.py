#!/usr/bin/env python3
"""Recompute every headline number in the card-to-card reports from the committed run files.

verify_claims.py [--llama DIR]

Each claim is (report it appears in, what it says, expected value, computed value). Every computed value comes
from files in git under runs/ (and, with --llama, from a llama.cpp checkout at b11699 for the source lines cited).
Exit status 1 if any claim fails. Reports covered: runs/2026-11-03-nsm, runs/2026-11-04-ub1, runs/2026-11-04-fragile.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P, N, U = "runs/2026-11-02-probe", "runs/2026-11-03-nsm", "runs/2026-11-04-ub1"
LAD = f"{N}/ladder"
RUNGS = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
B11699 = "7182ef71e6d5dd0c8ea76486fac134fcd25025d4"

results = []


def claim(report, what, expected, got):
    results.append((expected == got, report, what, expected, got))


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def probe(a, b):
    out = subprocess.run([sys.executable, f"{HERE}/fork_probe.py", a, b], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout.splitlines()[0])  # first line is the summary, then one line per fork
    return (d["text_differs"], d["grade_differs"], d["shared_tokens"], d["shared_tokens_same_logprob"])


def text_pair(a, b):
    a, b = load(a), load(b)
    ids = set(a) & set(b)
    return (sum(a[i]["text"] != b[i]["text"] for i in ids), sum(bool(a[i].get("right")) != bool(b[i].get("right")) for i in ids))


def right(path):
    return sum(bool(r.get("right")) for r in load(path).values())


def cublas_log(path):
    lines = [x for x in read(path).splitlines() if re.search(r"graveyard (split_to_vec|cublas)", x)]
    return (len(lines), sum("per_layer_model_proj.weight" not in x for x in lines))


def shas(path):
    return {ln.split()[2]: ln.split()[3] for ln in read(path).splitlines() if len(ln.split()) >= 4}


def prompt_tps(path):
    s = read(path)
    p = [(float(a), int(b)) for a, b in re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", s)]
    return round(sum(b for _, b in p) / (sum(a for a, _ in p) / 1000))


def source_line(llama, rel, n):
    return read(os.path.join(llama, rel)).splitlines()[n - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", help="llama.cpp checkout at b11699, for the cited source lines")
    args = ap.parse_args()
    os.chdir(os.path.dirname(HERE))

    r = f"{N}/REPORT.md"
    claim(r, "1650 has 16 SMs", True, "sm=16 cc=7.5" in read(f"{N}/1650-sm.txt"))
    claim(r, "T4 has 40 SMs", True, "sm=40 cc=7.5" in read(f"{N}/t4-host.txt"))
    q8 = "Q8_0.jsonl"
    claim(r, "Q8_0 stock 1650 vs T4", (165, 4, 50572, 29969), probe(f"{P}/1650/default/{q8}", f"{P}/t4/default/{q8}"))
    claim(r, "Q8_0 1650 told 40 SMs vs T4 stock", (32, 2, 87580, 85282), probe(f"{N}/nsm40/{q8}", f"{P}/t4/default/{q8}"))
    claim(r, "Q8_0 both settings, 1650 vs T4", (0, 0, 97380, 97380), probe(f"{N}/nsm40-vec/{q8}", f"{N}/t4-vec/{q8}"))
    claim(r, "patch inert on 1650", (0, 0, 97116, 97116), probe(f"{N}/own/{q8}", f"{P}/1650/default/{q8}"))
    claim(r, "patch inert on T4", (0, 0, 98637, 98637), probe(f"{N}/t4-patched/{q8}", f"{P}/t4/default/{q8}"))
    claim(r, "1650 told 40 SMs: vector route vs cuBLAS", (167, 10, 50648, 30132), probe(f"{N}/nsm40-vec/{q8}", f"{N}/nsm40/{q8}"))
    claim(r, "Q8_0 both settings: right on each card", (229, 229), (right(f"{N}/nsm40-vec/{q8}"), right(f"{N}/t4-vec/{q8}")))
    claim(r, "Q4_0 stock 1650 vs AWS T4, text and grade", (238, 22),
          text_pair("runs/2026-10-29-1650/Q4_0.jsonl", "runs/2026-11-01-flips/t4/default/Q4_0.jsonl"))
    claim(r, "Q4_0 both settings, 1650 vs T4", (0, 0, 108281, 108281), probe(f"{N}/nsm40-vec-q4/Q4_0.jsonl", f"{N}/t4-vec-q4/Q4_0.jsonl"))
    claim(r, "Q4_0 both settings: right on each card", (224, 224),
          (right(f"{N}/nsm40-vec-q4/Q4_0.jsonl"), right(f"{N}/t4-vec-q4/Q4_0.jsonl")))
    for run in ("nsm40-vec", "t4-vec"):
        claim(r, f"{run}: logged cuBLAS products, others than per_layer_model_proj", (200, 0), cublas_log(f"{N}/{run}/Q8_0.server.log"))
    for run in ("nsm40-vec-q4", "t4-vec-q4"):
        claim(r, f"{run}: logged cuBLAS products, others", (200, 0), cublas_log(f"{N}/{run}/Q4_0.server.log"))

    stock = {"Q6_K": (178, 7), "Q5_K_M": (188, 9), "Q4_K_M": (203, 10), "Q3_K_M": (231, 26), "Q2_K": (192, 6)}
    fixed = {"Q6_K": 96890, "Q5_K_M": 94574, "Q4_K_M": 99439, "Q3_K_M": 153883, "Q2_K": 249899}
    rights = {"Q6_K": 228, "Q5_K_M": 230, "Q4_K_M": 226, "Q3_K_M": 195, "Q2_K": 2}
    s1650, st4 = shas(f"{LAD}/1650-sha.txt"), shas(f"{LAD}/t4-sha.txt")
    for g in RUNGS:
        j = f"{g}.jsonl"
        claim(r, f"{g} stock 1650 vs T4, text and grade", stock[g], text_pair(f"runs/2026-10-29-1650/{j}", f"{LAD}/t4-stock-2026-10-27/{j}"))
        claim(r, f"{g} both settings, 1650 vs T4", (0, 0, fixed[g], fixed[g]), probe(f"{LAD}/1650-{g}/{j}", f"{LAD}/t4-{g}/{j}"))
        claim(r, f"{g} both settings: right on each card", (rights[g], rights[g]), (right(f"{LAD}/1650-{g}/{j}"), right(f"{LAD}/t4-{g}/{j}")))
        claim(r, f"{g} same GGUF sha256 on both cards", True, s1650[g] == st4[g])
        for card in ("1650", "t4"):
            claim(r, f"{g} {card}: logged cuBLAS products, others", (200, 0), cublas_log(f"{LAD}/{card}-{g}/{g}.server.log"))

    r = f"{U}/REPORT.md"
    claim(r, "Q8_0 stock -ub 1, 1650 vs T4", (0, 0, 97450, 97450), probe(f"{P}/1650/ub1/{q8}", f"{P}/t4/ub1/{q8}"))
    claim(r, "Q4_0 stock -ub 1, 1650 vs T4", (0, 0, 107195, 107195), probe(f"{U}/1650-Q4_0/Q4_0.jsonl", f"{U}/t4-Q4_0/Q4_0.jsonl"))
    claim(r, "Q4_0 stock -ub 1: right on each card", (226, 226), (right(f"{U}/1650-Q4_0/Q4_0.jsonl"), right(f"{U}/t4-Q4_0/Q4_0.jsonl")))
    claim(r, "Q4_0 -ub 1: same GGUF sha256 on both cards", True,
          shas(f"{U}/1650-sha.txt")["Q4_0"] == shas(f"{U}/t4-sha.txt")["Q4_0"])
    claim(r, "T4 default vs T4 -ub 1, text differs", 166, probe(f"{P}/t4/default/{q8}", f"{P}/t4/ub1/{q8}")[0])
    speeds = {
        "1650 default": (f"{P}/1650/default", 289), "1650 -ub 1": (f"{P}/1650/ub1", 58), "1650 patch route": (f"{N}/nsm40-vec", 239),
        "T4 default": (f"{P}/t4/default", 1270), "T4 -ub 1": (f"{P}/t4/ub1", 81), "T4 patch route": (f"{N}/t4-vec", 1060),
    }
    for k, (d, v) in speeds.items():
        claim(r, f"Q8_0 prompt tokens/s, {k}", v, prompt_tps(f"{d}/Q8_0.server.log"))

    r = "runs/2026-11-04-fragile/REPORT.md"
    out = subprocess.run(
        [sys.executable, f"{HERE}/route_spread.py", f"{P}/t4/default/{q8}", f"{P}/1650/default/{q8}", f"{P}/l4/default/{q8}",
         f"{P}/l4/ub1/{q8}", f"{P}/t4/ub1/{q8}", f"{N}/nsm40/{q8}", f"{N}/nsm40-vec/{q8}"],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    claim(r, "seven routes: answers worded differently somewhere", 219, d["text_varies"])
    claim(r, "seven routes: answers graded differently somewhere", 18, d["grade_varies"])
    t01 = next(x for x in d["flag_any_token"] if x["t"] == 0.1)
    claim(r, "near-tie flag at 0.1: flagged, caught", (164, "18/18"), (t01["flagged"], t01["grade_varies_caught"]))

    if args.llama:
        r = f"{N}/REPORT.md"
        head = subprocess.run(["git", "-C", args.llama, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        claim(r, "llama.cpp checkout is b11699", B11699, head)
        cites = [
            ("ggml/src/ggml-cuda/ggml-cuda.cu", 353, "prop.multiProcessorCount"),
            ("src/llama-quant.cpp", 324, 'name.find("per_layer_model_proj")'),
            ("ggml/src/ggml-cuda/mmq.cuh", 951, '"stream-k" work partitioning'),
            ("ggml/src/ggml-cuda/mmq.cuh", 1399, "devices[id].nsm"),
            ("ggml/src/ggml-cuda/mmq.cuh", 1444, ": nsm, 1, 1)"),
        ]
        for rel, n, text in cites:
            claim(r, f"{rel}:{n} contains {text}", True, text in source_line(args.llama, rel, n))

    bad = 0
    for ok, rep, what, exp, got in results:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {rep}: {what}" + ("" if ok else f"  expected {exp}, computed {got}"))
    print(f"{len(results) - bad} of {len(results)} claims recomputed and matched")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
