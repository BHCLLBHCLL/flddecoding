#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import defaultdict
from fld_model import parse_fld
from s_model import parse_sdat_file, build_structured_coords

model = parse_sdat_file("tests/ex1_e.s")
x, y, z = build_structured_coords(model)
fld = parse_fld("tests/ex1_e_100.fld")
mat = fld["material"]
v = fld["vertices"]
ni, nj, nk = 57, 20, 16
nx, ny, nz = len(x), len(y), len(z)
box = model.parts[-1].box
i1, i2, j1, j2, k1, k2 = box
xmin, xmax = x[i1 - 1], x[i2]
ymin, ymax = y[j1 - 1], y[j2]
zmin, zmax = z[k1 - 1], z[k2]


def corner_inside(ii, jj, kk):
    return xmin <= x[ii] <= xmax and ymin <= y[jj] <= ymax and zmin <= z[kk] <= zmax


def cell_mat(i, j, k):
    corners = [
        (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
        (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
    ]
    return 2 if all(corner_inside(*c) for c in corners) else 1


cmat = np.zeros((ni, nj, nk), dtype=np.int64)
for k in range(nk):
    for j in range(nj):
        for i in range(ni):
            cmat[i, j, k] = cell_mat(i, j, k)


def needs_dup(i, j, k):
    mats = set()
    for di in (-1, 0):
        for dj in (-1, 0):
            for dk in (-1, 0):
                ci, cj, ck = i + di, j + dj, k + dk
                if 0 <= ci < ni and 0 <= cj < nj and 0 <= ck < nk:
                    mats.add(cmat[ci, cj, ck])
    return len(mats) > 1


built = {(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx) if needs_dup(i, j, k)}

# fld dups
fc = np.array([[x[i], y[j], z[k]] for k in range(nz) for j in range(ny) for i in range(nx)])
fld_at = defaultdict(list)
for nid in range(1, len(v) + 1):
    ti = int(np.argmin(np.sum((fc - v[nid - 1]) ** 2, axis=1)))
    k, j, i = np.unravel_index(ti, (nz, ny, nx))
    fld_at[(i, j, k)].append(nid)
fld_dups = {p for p, n in fld_at.items() if len(n) == 2}

print("built dup", len(built), "fld dup", len(fld_dups))
print("intersection", len(built & fld_dups))
print("built only", len(built - fld_dups))
print("fld only", len(fld_dups - built))
