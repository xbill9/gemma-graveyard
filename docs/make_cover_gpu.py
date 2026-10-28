#!/usr/bin/env python3
"""make_cover_gpu.py -- the L4 follow-on cover: the graveyard grid with the two GPUs laid over each other.

  make_cover_gpu.py <out-dir>

Same layout as make_cover.py (one row per rung, one cell per question in BF16's order), with the T4 grid from
gpu_derived.py and the L4 grid from its rows: marigold where both GPUs answer right, a stone headstone where
both answer wrong, violet where the two GPUs disagree. No text. Writes devto-cover-l4.<sha8>.jpg at 1376x578.
"""

import hashlib
import io
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_derived import LADDER, l4_grid, t4_grid

W, H, S = 1376, 578, 2
BG, ALIVE, DEAD, SPLIT = (24, 20, 34), (244, 160, 36), (92, 92, 104), (168, 110, 240)


def main(out_dir):
    t4 = t4_grid("graveyard-ladder.json", LADDER)
    l4, _ = l4_grid(LADDER)
    ids = sorted(t4["BF16"], key=lambda i: (not (t4["BF16"][i] or l4["BF16"][i]), i))
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    mx, my = 64 * S, 56 * S
    cw = (W * S - 2 * mx) / len(ids)
    rh = (H * S - 2 * my) / len(LADDER)
    for k, r in enumerate(LADDER):
        y0 = my + k * rh
        for j, i in enumerate(ids):
            x0 = mx + j * cw
            box = [x0 + 0.5 * S, y0 + 0.18 * rh, x0 + cw - 0.5 * S, y0 + 0.82 * rh]
            if t4[r][i] != l4[r][i]:
                d.rectangle(box, fill=SPLIT)
            elif t4[r][i]:
                d.rectangle(box, fill=ALIVE)
            else:  # a headstone: rounded top
                d.rounded_rectangle([box[0], box[1] + 0.25 * rh, box[2], box[3]], radius=int(cw / 2), fill=DEAD)
    img = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    name = f"devto-cover-l4.{hashlib.sha256(buf.getvalue()).hexdigest()[:8]}.jpg"
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(buf.getvalue())
    print(name)


if __name__ == "__main__":
    main(sys.argv[1])
