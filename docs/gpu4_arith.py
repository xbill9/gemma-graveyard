#!/usr/bin/env python3
"""Every derived figure in the four-GPU article, from the bury, autopsy and log files in docs/evidence-gpu4.
python3 docs/gpu4_arith.py > docs/evidence-gpu4/arithmetic.txt"""
import json
from datetime import datetime

E = "docs/evidence-gpu4"
GPUS = ["t4", "l4", "1650", "3050"]
LADDER = ["BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]


def jl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def by_rung(p):
    return {x["rung"]: x for x in jl(p) if "rung" in x}


def main():
    b = {g: {**by_rung(f"{E}/{g}-bury.txt"), **by_rung(f"{E}/{g}-bury-qat.txt")} for g in GPUS}
    a = {g: by_rung(f"{E}/{g}-autopsy.txt") for g in GPUS}
    print("rung gpu right lost gained cut_off tok/s x_bf16 x_q8_0")
    for r in LADDER + ["QAT_Q4_0"]:
        for g in GPUS:
            x = b[g][r]
            t = x["median_decode_tps"]
            print(r, g, x["right"], x["dead"], x["ghosts"], x["truncated"], t,
                  f"{t / b[g]['BF16']['median_decode_tps']:.2f}", f"{t / b[g]['Q8_0']['median_decode_tps']:.2f}")
    print()
    for name, rungs in (("F16..Q5_K_M", LADDER[1:5]), ("Q4_K_M..Q4_0", LADDER[5:7]), ("Q3_K_M", ["Q3_K_M"]),
                        ("F16..Q4_0", LADDER[1:7])):
        lost = [b[g][r]["dead"] for g in GPUS for r in rungs]
        print(f"lost range {name}: {min(lost)} to {max(lost)}")
    for r in ("BF16", "Q5_K_M", "Q4_0", "Q2_K", "QAT_Q4_0"):
        right = [b[g][r]["right"] for g in GPUS]
        print(f"right range {r}: {min(right)} to {max(right)}")
    for g in GPUS:
        print(f"{g}: QAT_Q4_0 minus Q4_0 right = {b[g]['QAT_Q4_0']['right'] - b[g]['Q4_0']['right']}")
        print(f"{g}: Q3_K_M lost cut off {a[g]['Q3_K_M']['dead_cut_off']} of {a[g]['Q3_K_M']['dead']}; "
              f"Q2_K looped {a[g]['Q2_K']['looped']}, cut off {a[g]['Q2_K']['cut_off']} of 250")
        fast = max(LADDER, key=lambda r: b[g][r]["median_decode_tps"])
        print(f"{g}: fastest rung {fast} {b[g][fast]['median_decode_tps']}; GPU MiB BF16 {b[g]['BF16']['gpu_mib_loaded']} "
              f"Q4_0 {b[g]['Q4_0']['gpu_mib_loaded']}")
    print(f"l4 Q2_K wrong: {250 - b['l4']['Q2_K']['right']} of 250")
    for r in ("Q8_0", "Q5_K_M", "Q4_0"):
        print(f"{r} bytes as % of BF16: {100 * b['3050'][r]['bytes'] / b['3050']['BF16']['bytes']:.1f}")
    with open(f"{E}/four-gpus.txt") as f:
        f4 = json.load(f)
    for r, v in f4["ladder"]["graded_differently_by_pair"].items():
        print(f"graded differently {r}, any two GPUs: {min(v)} to {max(v)}")
    for g in ("1650", "3050"):
        with open(f"{E}/{g}-run-{g}.log") as f:
            ts = [ln.split()[0] for ln in f if ln[:2] == "20"]
        t0, t1 = (datetime.fromisoformat(t.replace("Z", "+00:00")) for t in (ts[0], ts[-1]))
        m = int((t1 - t0).total_seconds() // 60)
        print(f"{g} run: {ts[0]} to {ts[-1]}, {m // 60} h {m % 60} min")


if __name__ == "__main__":
    main()
