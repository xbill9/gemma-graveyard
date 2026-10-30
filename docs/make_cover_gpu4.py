#!/usr/bin/env python3
"""make_cover_gpu4.py -- the four-GPU cover: the graveyard grid with the T4, L4, GTX 1650 and RTX 3050 laid over each other.

  make_cover_gpu4.py <out-dir>

Same layout as make_cover_gpu.py (one row per rung, one cell per question), with the T4 grid from gpu_derived.py and
the other three from their rows: marigold where all four GPUs answer right, a stone headstone where all four answer
wrong, violet where the GPUs disagree. No text. Writes devto-cover-gpu4.<sha8>.jpg at 1376x578.
"""

import hashlib
import io
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu3_derived import G1650, rows_grid
from gpu4_derived import G3050
from gpu_derived import L4, LADDER, t4_grid

W, H, S = 1376, 578, 2
BG, ALIVE, DEAD, SPLIT = (24, 20, 34), (244, 160, 36), (92, 92, 104), (168, 110, 240)


def main(out_dir):
    g = [t4_grid("graveyard-ladder.json", LADDER, None)] + [rows_grid(d, LADDER) for d in (L4, G1650, G3050)]
    ids = sorted(g[0]["BF16"], key=lambda i: (not any(x["BF16"][i] for x in g), i))
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
            n = sum(x[r][i] for x in g)
            if 0 < n < len(g):
                d.rectangle(box, fill=SPLIT)
            elif n:
                d.rectangle(box, fill=ALIVE)
            else:  # a headstone: rounded top
                d.rounded_rectangle([box[0], box[1] + 0.25 * rh, box[2], box[3]], radius=int(cw / 2), fill=DEAD)
    img = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    name = f"devto-cover-gpu4.{hashlib.sha256(buf.getvalue()).hexdigest()[:8]}.jpg"
    with open(os.path.join(out_dir, name), "wb") as f:
        f.write(buf.getvalue())
    print(name)


if __name__ == "__main__":
    main(sys.argv[1])
