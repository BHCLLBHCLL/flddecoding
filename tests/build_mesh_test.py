#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fld_model import parse_fld
from s_model import parse_sdat_file, build_structured_coords

model = parse_sdat_file("tests/ex1_e.s")
x, y, z = build_structured_coords(model)
fld = parse_fld("tests/ex1_e_100.fld")
mat_fld = fld["material"]
conn_fld = fld["cell_conn"]
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


node_ids: dict[tuple[int, int, int], list[int]] = {}
next_id = 1
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            node_ids[(i, j, k)] = [next_id]
            next_id += 1
            if needs_dup(i, j, k):
                node_ids[(i, j, k)].append(next_id)
                next_id += 1

CORNER_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]
FLD_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]


def pick_node(i, j, k, cell_m):
    ids = node_ids[(i, j, k)]
    if len(ids) == 1:
        return ids[0]
    return ids[1] if cell_m == 2 else ids[0]


def cell_conn(ci, cj, ck):
    m = cmat[ci, cj, ck]
    nodes = []
    for oi, oj, ok in CORNER_OFFSETS:
        nodes.append(pick_node(ci + oi, cj + oj, ck + ok, m))
    return nodes


# compare all cells
bad = 0
for k in range(nk):
    for j in range(nj):
        for i in range(ni):
            idx = k * ni * nj + j * ni + i
            built = cell_conn(i, j, k)
            if not np.array_equal(built, conn_fld[idx]):
                bad += 1
                if bad <= 3:
                    print("bad cell", i, j, k, "built", built, "fld", conn_fld[idx])
from mesh_builder import _flatten_cells_vendor

print("bad cells", bad, "mat match", np.all(_flatten_cells_vendor(cmat, ni, nj, nk) == mat_fld))
print("total nodes", next_id - 1)
