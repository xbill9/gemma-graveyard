#!/usr/bin/env python3
"""Thinking on against thinking off, question by question, on the rungs both runs finished.
Writes docs/evidence/derived-think.txt: python3 docs/think_derived.py > docs/evidence/derived-think.txt"""
import json

OFF, ON = "runs/2026-10-27-ladder", "runs/2026-10-28-think"
RUNGS = "BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M".split()


def rows(d, r):
    return {x["id"]: x for x in map(json.loads, open(f"{d}/{r}.jsonl"))}


def table(d):
    return {x["rung"]: x for x in map(json.loads, open(f"{d}/bury.txt")) if "rung" in x}


def dead(m, r):
    return {i for i in m["BF16"] if m["BF16"][i]["right"] and not m[r][i]["right"]}


off = {r: rows(OFF, r) for r in RUNGS}
on = {r: rows(ON, r) for r in RUNGS}
bo, bt = table(OFF), table(ON)
q3_cut = {i for i in dead(off, "Q3_K_M") if off["Q3_K_M"][i]["finish"] == "length"}
out = {
    "right_on_minus_off": {r: bt[r]["right"] - bo[r]["right"] for r in RUNGS},
    "mean_tokens_on_over_off": {r: round(bt[r]["mean_tokens"] / bo[r]["mean_tokens"], 2) for r in RUNGS},
    "decode_tps_on_over_off": {r: round(bt[r]["median_decode_tps"] / bo[r]["median_decode_tps"], 2) for r in RUNGS},
    "seconds_on_over_off": {
        r: round(sum(x["seconds"] for x in on[r].values()) / sum(x["seconds"] for x in off[r].values()), 2) for r in RUNGS
    },
    "bf16_right_both_modes": sum(off["BF16"][i]["right"] and on["BF16"][i]["right"] for i in off["BF16"]),
    "dead_off_on_shared": {r: [len(dead(off, r)), len(dead(on, r)), len(dead(off, r) & dead(on, r))] for r in RUNGS[1:]},
    "q3_dead_cut_off_with_thinking_off": len(q3_cut),
    "of_them_right_with_thinking_on": sum(on["Q3_K_M"][i]["right"] for i in q3_cut),
}
print(json.dumps(out, indent=1))
