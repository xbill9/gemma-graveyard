#!/usr/bin/env python3
"""make_cover.py -- the article cover, drawn from the ladder's own rows.

  make_cover.py <run-dir> <out-dir>

One row per rung, BF16 at the top, one cell per question in BF16's order: marigold where the rung
answers right, stone grey where it answers wrong. No text. Writes devto-cover.<sha8>.jpg at 1376x578
(dev.to's 2.381:1 display ratio), drawn at 2x and downsampled.
"""

import hashlib
import io
import json
import os
import sys

from PIL import Image, ImageDraw

RUNGS = ["BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
W, H, S = 1376, 578, 2
BG, ALIVE, DEAD = (24, 20, 34), (244, 160, 36), (92, 92, 104)


def main(run_dir, out_dir):
    rows = {}
    for r in RUNGS:
        with open(os.path.join(run_dir, f"{r}.jsonl")) as f:
            rows[r] = {x["id"]: x["right"] for x in map(json.loads, f)}
    ids = sorted(rows["BF16"], key=lambda i: (not rows["BF16"][i], i))  # BF16's right answers first
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    mx, my = 64 * S, 56 * S
    cw = (W * S - 2 * mx) / len(ids)
    rh = (H * S - 2 * my) / len(RUNGS)
    for k, r in enumerate(RUNGS):
        y0 = my + k * rh
        for j, i in enumerate(ids):
            x0 = mx + j * cw
            box = [x0 + 0.5 * S, y0 + 0.18 * rh, x0 + cw - 0.5 * S, y0 + 0.82 * rh]
            if rows[r][i]:
                d.rectangle(box, fill=ALIVE)
            else:  # a headstone: rounded top
                d.rounded_rectangle([box[0], box[1] + 0.25 * rh, box[2], box[3]], radius=int(cw / 2), fill=DEAD)
    img = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    name = f"devto-cover.{hashlib.sha256(buf.getvalue()).hexdigest()[:8]}.jpg"
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(buf.getvalue())
    print(name)


if __name__ == "__main__":
    main(*sys.argv[1:3])
