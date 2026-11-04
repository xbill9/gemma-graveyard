#!/usr/bin/env python3
"""Recompute every headline number in the card-to-card reports from the committed run files.

verify_claims.py [--llama DIR]

Each claim is (report it appears in, what it says, expected value, computed value). Every computed value comes
from files in git under runs/ (and, with --llama, from a llama.cpp checkout at b11699 for the source lines cited).
Exit status 1 if any claim fails. Reports covered: runs/2026-11-01-flips, runs/2026-11-02-probe, runs/2026-11-03-nsm, runs/2026-11-04-ub1,
runs/2026-11-04-fragile.
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
    return (sum(a[i].get("text") != b[i].get("text") for i in ids), sum(bool(a[i].get("right")) != bool(b[i].get("right")) for i in ids))


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


def tool_json(name, *args):
    out = subprocess.run([sys.executable, f"{HERE}/{name}", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def probe_full(a, b):
    out = subprocess.run([sys.executable, f"{HERE}/fork_probe.py", a, b], capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:]]


def modes():
    """flip_modes.py lines -> {(left, right, rung): (text, answer, grade)}"""
    out = subprocess.run([sys.executable, f"{HERE}/flip_modes.py"], capture_output=True, text=True, check=True).stdout
    m = {}
    for ln in out.splitlines():
        x = re.match(r"(\S+)\s+vs\s+(\S+)\s+(\S+)\s+n\s+\d+\s+text\s+(\d+)\s+answer\s+(\d+)\s+grade\s+(\d+)", ln)
        if x:
            m[x.group(1), x.group(2), x.group(3)] = tuple(int(x.group(k)) for k in (4, 5, 6))
    return m


def text_cards_pairs(rungs):
    d = {"l4": "runs/2026-10-28-l4", "1650": "runs/2026-10-29-1650", "3050": "runs/2026-10-29-3050"}
    t, g = [], []
    for r in rungs:
        for x, y in (("l4", "1650"), ("l4", "3050"), ("1650", "3050")):
            a, b = load(f"{d[x]}/{r}.jsonl"), load(f"{d[y]}/{r}.jsonl")
            t.append(sum(a[i]["text"] != b[i]["text"] for i in a))
            g.append(sum(bool(a[i]["right"]) != bool(b[i]["right"]) for i in a))
    return (min(t), max(t)), (min(g), max(g))


def earlier_reports():
    r = "runs/2026-11-01-flips/REPORT.md"
    m = modes()
    for k in [("t4/default", "t4/repeat", "Q8_0"), ("t4/default", "t4/repeat", "Q4_0"), ("l4/default", "l4/repeat", "Q8_0"),
              ("l4/default", "l4/repeat", "Q4_0"), ("l4/default", "../2026-10-28-l4", "BF16"),
              ("l4/default", "../2026-10-28-l4", "Q8_0"), ("l4/default", "../2026-10-28-l4", "Q4_0")]:
        claim(r, f"item 1, repeat matches word for word: {' vs '.join(k)}", 0, m[k][0])
    g = {x: tool_json("fork_gaps.py", x) for x in ("BF16", "Q8_0", "Q4_0")}
    claim(r, "item 2, median lead, all tokens (BF16, Q8_0, Q4_0)", (16.6, 17.0, 15.3), tuple(round(g[x]["all_tokens_gap"]["median"], 1) for x in g))
    claim(r, "item 2, median lead at the fork (BF16, Q8_0, Q4_0)", (0.044, 0.203, 0.401), tuple(g[x]["fork_tokens_gap"]["median"] for x in g))
    claim(r, "item 2, BF16 replay answers", 103, g["BF16"]["answers"])
    claim(r, "item 2, share of tokens leading by under 0.25, percent", (1.1, 1.4),
          (round(min(g[x]["share_all_gap_under"]["0.25"] for x in g) * 100, 1), round(max(g[x]["share_all_gap_under"]["0.25"] for x in g) * 100, 1)))
    claim(r, "item 3, text cards BF16 and F16: text differs, grade differs (ranges)", ((31, 65), (3, 5)), text_cards_pairs(["BF16", "F16"]))
    claim(r, "item 3, text cards Q8_0..Q4_0: text differs, grade differs (ranges)", ((162, 242), (5, 22)),
          text_cards_pairs(["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"]))
    ar = subprocess.run([sys.executable, f"{HERE}/flip_arith.py"], capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    per = {ln.split()[0]: float(ln.split()[3]) for ln in ar if ln.split()}
    claim(r, "item 3, false equations per 1000, BF16..Q3_K_M range", (3.0, 8.6),
          (round(min(v for k, v in per.items() if k != "Q2_K"), 1), round(max(v for k, v in per.items() if k != "Q2_K"), 1)))
    claim(r, "item 3, false equations per 1000 at Q2_K", 597, round(per["Q2_K"]))
    c = tool_json("flip_core.py")["summary"]
    claim(r, "item 4, 28 runs: always right, never, contested", (28, 184, 7, 59), (c["runs"], c["always_right"], c["always_wrong"], c["contested"]))
    claim(r, "item 4, distinct answers per contested question, median", 2, c["distinct_answers_median"])
    e = tool_json("flip_extra.py")
    claim(r, "item 4, 30 most contested hold this share of minority results, percent", 80.7, round(e["minority_top30_share"] * 100, 1))
    t = e["bf16_tokens_median"]
    claim(r, "item 4, BF16 median tokens contested vs not (L4, 1650, 3050)", ((476, 328), (476, 328), (457, 328)),
          tuple((t[k]["contested"], t[k]["uncontested"]) for k in ("l4", "1650", "3050")))
    claim(r, "item 4, AUC of tokens, range", (0.79, 0.8), (round(min(v["auc"] for v in t.values()), 2), round(max(v["auc"] for v in t.values()), 2)))
    sd = e["sd_by_rung"]
    q = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"]
    claim(r, "item 5, SD of four totals Q8_0..Q4_0, observed range", (1.09, 2.24), (min(sd[x]["observed"] for x in q), max(sd[x]["observed"] for x in q)))
    claim(r, "item 5, SD if independent coin flips, Q8_0..Q4_0 range", (1.12, 1.92),
          (min(sd[x]["independent_flips_median"] for x in q), max(sd[x]["independent_flips_median"] for x in q)))
    claim(r, "item 5, card shuffle: smallest p over rungs >= 0.27", True, min(v["share_sims_at_least_observed"] for v in c["card_shuffle"].values()) >= 0.27)
    for k, v in {("l4/default", "t4/default", "Q8_0"): 172, ("l4/ub1", "t4/ub1", "Q8_0"): 161, ("l4/cc75-default", "t4/default", "Q8_0"): 174,
                 ("l4/cc75-ub1", "t4/ub1", "Q8_0"): 0, ("l4/cc75-ub1", "l4/ub1", "Q8_0"): 161, ("l4/default", "t4/default", "BF16"): 25,
                 ("l4/ub1", "t4/ub1", "BF16"): 3, ("l4/default", "t4/default", "Q4_0"): 238, ("l4/ub1", "t4/ub1", "Q4_0"): 238,
                 ("l4/default", "l4/f32", "Q8_0"): 159}.items():
        claim(r, f"items 6-7, text differs: {' vs '.join(k)}", v, m[k][0])
    for rung, exp in (("Q8_0", (19, 12, 5, 2, 0)), ("Q4_0", (35, 16, 12, 4, 3))):
        b = [json.loads(x) for x in read(f"runs/2026-11-01-flips/branch-{rung}.jsonl").splitlines()]
        got = (len(b), sum(x["right"]["right"] and x["wrong"]["right"] for x in b), sum(x["right"]["right"] and not x["wrong"]["right"] for x in b),
               sum(not x["right"]["right"] and x["wrong"]["right"] for x in b), sum(not x["right"]["right"] and not x["wrong"]["right"] for x in b))
        claim(r, f"forced branches {rung}: flips, both right, reproduces, reversed, both wrong", exp, got)

    r = f"{P}/REPORT.md"
    rows = {("l4/cc75-ub1", "t4/ub1"): (0, 0, 97450, 97450, None), ("l4/default", "l4/ub1"): (165, 4, 51758, 58.7, 159),
            ("t4/default", "t4/ub1"): (166, 9, 50197, 59.9, 160), ("l4/default", "t4/default"): (172, 3, 47942, 58.8, 164)}
    for (x, y), exp in rows.items():
        d, forks = probe_full(f"{P}/{x}/Q8_0.jsonl", f"{P}/{y}/Q8_0.jsonl")
        same = d["shared_tokens_same_logprob"] if not d["text_differs"] else round(100 * d["shared_tokens_same_logprob"] / d["shared_tokens"], 1)
        got = (d["text_differs"], d["grade_differs"], d["shared_tokens"], same, d["forks_swapped_top2"] or None)
        claim(r, f"table: {x} vs {y} (text, grade, tokens, same score, swaps)", exp, got)
        if d["text_differs"]:
            claim(r, f"item 3, {x} vs {y}: median fork lead within 0.11-0.15 on both sides", True,
                  all(0.11 <= d[k] <= 0.15 for k in ("fork_lead_a_median", "fork_lead_b_median")))
            small = sum(min(f["a_lead"], f["b_lead"]) < 0.5 for f in forks if f.get("a_lead") is not None and f.get("b_lead") is not None)
            claim(r, f"item 3, {x} vs {y}: forks where the smaller lead is under 0.5", {165: 163, 166: 164, 172: 172}[d["text_differs"]], small)
        if (x, y) in (("l4/default", "l4/ub1"), ("t4/default", "t4/ub1")):
            claim(r, f"item 2, {x} vs {y}: median, largest score difference", {"l4/default": (0.00016, 0.54), "t4/default": (0.00014, 0.56)}[x],
                  (round(d["shared_logprob_absdiff_median"], 5), round(d["shared_logprob_absdiff_max"], 2)))
    claim(r, "1650, probed 2026-11-03: -ub 1 vs T4 -ub 1", (0, 0, 97450, 97450), probe(f"{P}/1650/ub1/Q8_0.jsonl", f"{P}/t4/ub1/Q8_0.jsonl"))
    claim(r, "1650, probed 2026-11-03: default vs T4 default, text differs", 165, probe(f"{P}/1650/default/Q8_0.jsonl", f"{P}/t4/default/Q8_0.jsonl")[0])


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
    ub1_kq = {  # rung: right -ub 1, right default, (text, grade) vs default, right fix, (text, grade) vs fix
        "Q6_K": (231, 226, (168, 7), 228, (165, 7)), "Q5_K_M": (228, 226, (189, 10), 230, (194, 6)),
        "Q4_K_M": (223, 222, (206, 13), 226, (204, 15)), "Q3_K_M": (190, 183, (229, 37), 195, (229, 39)),
        "Q2_K": (3, 5, (192, 6), 2, (194, 3)),
    }
    for g, (ru, rd, vd, rf, vf) in ub1_kq.items():
        u, d, f = f"{U}/1650-{g}/{g}.jsonl", f"runs/2026-10-29-1650/{g}.jsonl", f"{N}/ladder/1650-{g}/{g}.jsonl"
        claim(r, f"{g} 1650: right under -ub 1, default, fix", (ru, rd, rf), (right(u), right(d), right(f)))
        claim(r, f"{g} 1650 -ub 1 vs default: text, grade differ", vd, text_pair(u, d))
        claim(r, f"{g} 1650 -ub 1 vs fix: text, grade differ", vf, text_pair(u, f))
    claim(r, "Q2_K test-194: 500 under fix, length limit under -ub 1",
          ("HTTPError: HTTP Error 500: Internal Server Error", "length"),
          (load(f"{N}/ladder/1650-Q2_K/Q2_K.jsonl")["test-194"].get("error"), load(f"{U}/1650-Q2_K/Q2_K.jsonl")["test-194"].get("finish")))
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

    earlier_reports()

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
