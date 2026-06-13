#!/usr/bin/env python3
"""Parse Coordinates section from scFLOW .r file."""
import sys
from pathlib import Path
import struct
import numpy as np

R = Path(r"D:\training\cradle\CradleCFD_2023.2_ST_Example\Operation_e\ex1\ex1_e.r")
data = R.read_bytes()


def read_i32(pos: int) -> int:
    return struct.unpack_from(">i", data, pos)[0]


def find(name: str) -> int:
    pad = name.ljust(32).encode()
    i = data.find(pad)
    return i - 4 if i >= 4 and read_i32(i - 4) == 32 else -1


def parse_coords_section():
    off = data.find(b"Coordinates")
    start = off - 4 if read_i32(off - 4) == 32 else off
  # scan from section start for f64 arrays
    end = find("Velocity")
    chunk = data[start:end]
    arrays = []
    pos = 0
    while pos < len(chunk) - 12:
        if read_i32(start + pos) == 12:
            bc = read_i32(start + pos + 4)
            p = start + pos + 8
            if bc > 0 and p + bc <= end and read_i32(p + bc) == bc:
                if bc % 8 == 0:
                    arr = np.frombuffer(data, dtype=">f8", count=bc // 8, offset=p)
                    arrays.append(arr)
                pos += 8 + bc + 4
                continue
        pos += 4
    return arrays


arrays = parse_coords_section()
print("f64 arrays", [len(a) for a in arrays])
for i, a in enumerate(arrays[:6]):
    print(i, a[:3], "...", a[-1])

# also scan whole file for 21145 f64 triplets
n = 21145
for axis_name, size in [("x", 58), ("y", 21), ("z", 17), ("verts", 21145)]:
    target = size * 8
    for pos in range(0, len(data) - target, 8):
        arr = np.frombuffer(data, dtype=">f8", count=3, offset=pos)
        if arr[0] == 0.0 and 0 < arr[1] < 0.1:
            pass  # too many
