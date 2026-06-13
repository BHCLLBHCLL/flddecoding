#!/usr/bin/env python3
"""Deep parse ex1_e.r for mesh arrays."""
import sys
from pathlib import Path
import struct
import numpy as np

R = Path(r"D:\training\cradle\CradleCFD_2023.2_ST_Example\Operation_e\ex1\ex1_e.r")
data = R.read_bytes()


def read_i32(pos: int) -> int:
    return struct.unpack_from(">i", data, pos)[0]


def scan_i32_arrays(min_count: int, max_val: int):
    hits = []
    step = 4
    for pos in range(0, len(data) - min_count * 4, step):
        vals = struct.unpack_from(f">{min_count}i", data, pos)
        if all(1 <= v <= max_val for v in vals):
            if vals[0] == 1 and vals[1] == 2:  # likely cell0 pattern
                hits.append((pos, vals[:8]))
    return hits


# scan for first cell pattern [1,2,32,31,...]
hits = []
for pos in range(0, len(data) - 32, 4):
    v0, v1 = read_i32(pos), read_i32(pos + 4)
    if v0 == 1 and v1 == 2:
        vals = struct.unpack_from(">8i", data, pos)
        if all(1 <= v <= 25000 for v in vals):
            hits.append((pos, vals))

print("hits with 1,2 start", len(hits))
for h in hits[:10]:
    print(h)

# find large i32 block of length 18240*8
target = 18240 * 8 * 4
print("conn bytes", target)
for pos in range(0, len(data) - target, 4):
    # quick check first row
    vals = struct.unpack_from(">8i", data, pos)
    if vals == (1, 2, 32, 31, 651, 652, 682, 681):
        print("FOUND conn at", pos)
        conn = np.frombuffer(data, dtype=">i4", count=18240 * 8, offset=pos)
        print("conn shape", conn.shape, "max", conn.max())
        break
