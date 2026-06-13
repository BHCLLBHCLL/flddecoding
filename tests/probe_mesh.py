#!/usr/bin/env python3
"""Probe structured mesh indexing vs FLD."""
import numpy as np
from pathlib import Path
from fld_model import parse_fld


def parse_cxyz(text: str) -> list[np.ndarray]:
    parts = text.split("CXYZ")[1].split("PARTS")[0]
    blocks: list[np.ndarray] = []
    current: list[float] = []
    for line in parts.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "0":
            if current:
                blocks.append(np.array(current))
                current = []
            continue
        try:
            current.extend([float(x) for x in line.split()])
        except ValueError:
            pass
    if current:
        blocks.append(np.array(current))
    return blocks


def node_id(i: int, j: int, k: int, ni: int, nj: int) -> int:
    """1-based node index for structured grid (i,j,k) 0-based node indices."""
    return i + 1 + (ni + 1) * (j + (nj + 1) * k)


def cell_nodes_std(i: int, j: int, k: int, ni: int, nj: int) -> list[int]:
    """Try standard hex corner ordering."""
    n000 = node_id(i, j, k, ni, nj)
    n100 = node_id(i + 1, j, k, ni, nj)
    n010 = node_id(i, j + 1, k, ni, nj)
    n110 = node_id(i + 1, j + 1, k, ni, nj)
    n001 = node_id(i, j, k + 1, ni, nj)
    n101 = node_id(i + 1, j, k + 1, ni, nj)
    n011 = node_id(i, j + 1, k + 1, ni, nj)
    n111 = node_id(i + 1, j + 1, k + 1, ni, nj)
    return [n000, n100, n110, n010, n001, n101, n111, n011]


text = Path("tests/ex1_e.s").read_text(encoding="utf-8-sig")
cxyz = parse_cxyz(text)
ni, nj, nk = 57, 20, 16
m = parse_fld("tests/ex1_e_100.fld")
cell0 = m["cell_conn"][0]
print("fld cell0", cell0)
print("std cell0 i=0", cell_nodes_std(0, 0, 0, ni, nj))
# try permutations - fld cell0 [1, 2, 32, 31, 651, 652, 682, 681]
# node 32 might be (1,1,0) etc
for perm_name, order in [
    ("std", [0, 1, 2, 3, 4, 5, 6, 7]),
]:
    cn = cell_nodes_std(0, 0, 0, ni, nj)
    print(perm_name, [cn[x] for x in order])

# brute: find i,j,k for node ids in cell0
ids = set(cell0)
# build all node coords
x, y, z = cxyz[0], cxyz[1], cxyz[2]
coords = []
for k in range(len(z)):
    for j in range(len(y)):
        for i in range(len(x)):
            coords.append((x[i], y[j], z[k]))
print("tensor nodes", len(coords))
