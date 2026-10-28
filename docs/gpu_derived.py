#!/usr/bin/env python3
"""The T4 ladder against the L4 ladder, question by question. The T4's right/wrong grid is rebuilt from its
committed tombstones and ghosts (graveyard-ladder.json, graveyard-qat.json) and checked against its bury tables;
the L4's comes from its rows. Writes docs/evidence/derived-gpu.txt:
python3 docs/gpu_derived.py > docs/evidence/derived-gpu.txt"""
import json

T4, L4 = "runs/2026-10-27-ladder", "runs/2026-10-28-l4"
LADDER = ["BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
QAT = ["BF16", "QAT_Q4_0", "Q4_0"]


def lines(p):
    with open(p) as f:
        return [json.loads(x) for x in f]


def jload(p):
    with open(p) as f:
        return json.load(f)


IDS = [x["id"] for x in lines("data/heldout.jsonl")]
ALWAYS_RIGHT = {"test-44", "test-1187", "test-1241"}


def table(d, f):
    return {x["rung"]: x for x in lines(f"{d}/{f}") if "rung" in x}


def grid(g, rungs):
    """{rung: {id: right}} from a bury json: tombstones and ghosts fix every cell."""
    stones = {t["id"]: t for t in g["tombstones"]}
    ghosts = {t["id"]: t["right_at"] for t in g["ghosts"]}
    out = {r: {} for r in rungs}
    for i in IDS:
        if i in stones:
            t = stones[i]
            k = rungs.index(t["died_at"])
            for j, r in enumerate(rungs):
                out[r][i] = j < k or r in t["right_again_at"]
        else:
            for r in rungs:
                out[r][i] = None if i not in ghosts else (r in ghosts[i])
    return out


def t4_grid(f, rungs, bf16=None):
    """A question with no tombstone and no ghost is right at every rung or wrong at every rung, and the json does
    not say which. The cover (drawn from the T4's rows, one cell per question and rung) does: these three are the
    only choice that reproduces every cell of it, checked in cover_cells() below."""
    out = grid(jload(f"{T4}/{f}"), rungs)
    for i in IDS:
        if out["BF16"][i] is None:
            for r in rungs:
                out[r][i] = bf16[i] if bf16 else i in ALWAYS_RIGHT
    return out


def cover_cells(m):
    """Cells where the grid and docs/devto-cover.50dc7d50.jpg disagree (make_cover.py's layout)."""
    from PIL import Image

    img = Image.open("docs/devto-cover.50dc7d50.jpg").convert("RGB")
    cw, rh = (img.width - 128) / len(IDS), (img.height - 112) / len(LADDER)
    order = sorted(IDS, key=lambda i: (not m["BF16"][i], i))
    return sum(
        (img.getpixel((int(64 + (j + 0.5) * cw), int(56 + (k + 0.6) * rh)))[0] > 170) != m[r][i]
        for k, r in enumerate(LADDER)
        for j, i in enumerate(order)
    )


def l4_grid(rungs):
    rows = {r: {x["id"]: x for x in lines(f"{L4}/{r}.jsonl")} for r in rungs}
    return {r: {i: rows[r][i]["right"] for i in IDS} for r in rungs}, rows


def counts(m, rungs):
    ref = m[rungs[0]]
    return {
        r: {
            "right": sum(m[r].values()),
            "dead": sum(ref[i] and not m[r][i] for i in IDS),
            "ghosts": sum(not ref[i] and m[r][i] for i in IDS),
        }
        for r in rungs
    }


def dead(m, r):
    return {i for i in IDS if m["BF16"][i] and not m[r][i]}


out, t4_bf16 = {}, None
for name, rungs, bury in (("ladder", LADDER, "bury.txt"), ("qat", QAT, "bury-qat.txt")):
    t4 = t4_grid("graveyard-ladder.json" if rungs is LADDER else "graveyard-qat.json", rungs, t4_bf16)
    t4_bf16 = t4["BF16"]
    bt = table(T4, bury)
    for r, c in counts(t4, rungs).items():  # the rebuilt grid must give the T4's own bury counts
        assert (c["right"], c["dead"], c["ghosts"]) == (bt[r]["right"], bt[r]["dead"], bt[r]["ghosts"]), (r, c)
    assert name != "ladder" or cover_cells(t4) == 0, "T4 grid disagrees with the cover"
    l4, rows = l4_grid(rungs)
    bl = table(L4, bury)
    out[name] = {
        "t4_grid_matches_t4_bury_and_cover": True,
        "right_t4_l4": {r: [bt[r]["right"], bl[r]["right"]] for r in rungs},
        "dead_t4_l4_shared": {r: [len(dead(t4, r)), len(dead(l4, r)), len(dead(t4, r) & dead(l4, r))] for r in rungs[1:]},
        "ghosts_t4_l4": {r: [bt[r]["ghosts"], bl[r]["ghosts"]] for r in rungs[1:]},
        "questions_graded_differently": {r: sum(t4[r][i] != l4[r][i] for i in IDS) for r in rungs},
        "cut_off_t4_l4": {r: [bt[r]["truncated"], bl[r]["truncated"]] for r in rungs},
        "mean_tokens_t4_l4": {r: [bt[r]["mean_tokens"], bl[r]["mean_tokens"]] for r in rungs},
        "median_tps_t4_l4": {r: [bt[r]["median_decode_tps"], bl[r]["median_decode_tps"]] for r in rungs},
        "decode_tps_l4_over_t4": {r: round(bl[r]["median_decode_tps"] / bt[r]["median_decode_tps"], 2) for r in rungs},
        "decode_tps_over_bf16_t4_l4": {
            r: [
                round(bt[r]["median_decode_tps"] / bt["BF16"]["median_decode_tps"], 2),
                round(bl[r]["median_decode_tps"] / bl["BF16"]["median_decode_tps"], 2),
            ]
            for r in rungs
        },
        "l4_seconds": {r: round(sum(x["seconds"] for x in rows[r].values())) for r in rungs},
        "gpu_mib_t4_l4": {r: [bt[r]["gpu_mib_loaded"], bl[r]["gpu_mib_loaded"]] for r in rungs},
    }
    if name == "ladder":
        st = {t["id"]: t["died_at"] for t in jload(f"{T4}/graveyard-ladder.json")["tombstones"]}
        sl = {t["id"]: t["died_at"] for t in jload(f"{L4}/graveyard-ladder.json")["tombstones"]}
        out[name]["bf16_graded_differently"] = [
            {"id": i, "answer": rows["BF16"][i]["answer"], "t4_right": t4["BF16"][i], "l4_said": rows["BF16"][i]["extracted"]}
            for i in IDS
            if t4["BF16"][i] != l4["BF16"][i]
        ]
        out[name]["tombstoned_both_t4_only_l4_only"] = [len(set(st) & set(sl)), len(set(st) - set(sl)), len(set(sl) - set(st))]
        out[name]["tombstones_t4_l4_same_question_same_rung"] = [len(st), len(sl), sum(sl.get(i) == r for i, r in st.items())]
print(json.dumps(out, indent=1))
