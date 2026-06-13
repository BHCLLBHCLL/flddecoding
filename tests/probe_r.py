#!/usr/bin/env python3
"""Probe scFLOW .r mesh file layout."""
from pathlib import Path
import struct
import numpy as np

R_PATH = Path(r"D:\training\cradle\CradleCFD_2023.2_ST_Example\Operation_e\ex1\ex1_e.r")
data = R_PATH.read_bytes()


def read_i32(pos: int) -> int:
    return struct.unpack_from(">i", data, pos)[0]


def find_section(name: str) -> int:
    pad = name.ljust(32).encode()
    i = data.find(pad)
    if i >= 4 and read_i32(i - 4) == 32:
        return i - 4
    return -1


def iter_blocks(sec_start: int, sec_end: int):
    pos = sec_start + 40
    while pos + 8 <= sec_end:
        if read_i32(pos) != 12:
            pos += 4
            continue
        bc = read_i32(pos + 4)
        if bc <= 0 or pos + 8 + bc + 4 > sec_end:
            pos += 4
            continue
        if read_i32(pos + 8 + bc) != bc:
            pos += 4
            continue
        yield pos + 8, bc
        pos = pos + 8 + bc + 4


sections = []
for name in [
    "Precision", "RecordLength", "CycleTime_2", "Compressible",
    "CoordinatesSystem", "Elements", "Coordinates", "Velocity",
    "Pressure", "Temperature", "Material", "Parts", "Region",
    "LS_Nodes", "LS_Elements",
]:
    off = find_section(name)
    if off >= 0:
        sections.append((off, name))
sections.sort()
print("Sections:")
for off, name in sections:
    print(f"  0x{off:06X} {name}")

for sec_name in ("Coordinates", "Elements"):
    s = find_section(sec_name)
    if s < 0:
        continue
    e = sections[sections.index((s, sec_name)) + 1][0] if (s, sec_name) in sections else len(data)
    # find next section end
    e = len(data)
    for off, name in sections:
        if off > s:
            e = off
            break
    blocks = list(iter_blocks(s, e))
    print(f"\n{sec_name}: {len(blocks)} blocks, sec size {e-s}")
    for i, (p, bc) in enumerate(blocks[:15]):
        print(f"  block {i}: bc={bc}")
        if bc % 8 == 0 and bc >= 8:
            arr = np.frombuffer(data, dtype=">f8", count=min(bc // 8, 5), offset=p)
            print(f"    f64 head: {arr}")
        if bc % 4 == 0 and bc < 1000:
            arr = np.frombuffer(data, dtype=">i4", count=min(bc // 4, 20), offset=p)
            print(f"    i32 head: {arr}")
