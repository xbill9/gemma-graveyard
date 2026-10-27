#!/usr/bin/env python3
"""f16_range.py -- how many BF16 weights in a safetensors file do not survive a cast to F16.

  f16_range.py <model.safetensors> [--top N]

BF16 keeps FP32's exponent range; F16 tops out at 65504 and flushes magnitudes below 2**-24 to zero.
Reads the file directly (8-byte header length, JSON header, raw tensors) so it needs numpy only.
Counts, per tensor and in total: values above the F16 maximum (become inf), nonzero values below the
smallest F16 subnormal (become 0), and values in the subnormal range (keep fewer significant bits).
"""

import argparse
import json
import struct
import sys

import numpy as np

F16_MAX = 65504.0
F16_MIN_SUB = 2.0**-24
F16_MIN_NORMAL = 2.0**-14


def tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name, meta in header.items():
        if name == "__metadata__" or meta["dtype"] != "BF16":
            continue
        a, b = meta["data_offsets"]
        yield name, mm[base + a : base + b].view(np.uint16)


def chunks(raw, step=1 << 26):  # 64M values (256 MiB as FP32) at a time; one tensor is 2.3G values
    for i in range(0, raw.size, step):
        yield (raw[i : i + step].astype(np.uint32) << 16).view(np.float32)


def count(raw):
    c = {"n": int(raw.size), "overflow": 0, "underflow": 0, "subnormal": 0, "absmax": 0.0}
    for x in chunks(raw):
        m = np.abs(x)
        c["overflow"] += int((m > F16_MAX).sum())
        c["underflow"] += int(((m > 0) & (m < F16_MIN_SUB)).sum())
        c["subnormal"] += int(((m >= F16_MIN_SUB) & (m < F16_MIN_NORMAL)).sum())
        c["absmax"] = max(c["absmax"], float(m.max()))
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()
    total = {"n": 0, "overflow": 0, "underflow": 0, "subnormal": 0, "absmax": 0.0}
    rows = []
    for name, x in tensors(a.path):
        c = count(x)
        rows.append((name, c))
        for k in ("n", "overflow", "underflow", "subnormal"):
            total[k] += c[k]
        total["absmax"] = max(total["absmax"], c["absmax"])
    hit = sorted(rows, key=lambda r: -(r[1]["overflow"] * 10**12 + r[1]["underflow"] + r[1]["subnormal"]))
    for name, c in hit[: a.top]:
        print(json.dumps({"tensor": name, **c}))
    print(json.dumps({"tensor": "TOTAL", "tensors": len(rows), **total}))


if __name__ == "__main__":
    sys.exit(main())
