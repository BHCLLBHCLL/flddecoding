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
mat = fld["material"]
ni, nj, nk = 57, 20, 16
box = model.parts[-1].box
i1, i2, j1, j2, k1, k2 = box
xmin, xmax = x[i1 - 1], x[i2 - 1]
ymin, ymax = y[j1 - 1], y[j2 - 1]
zmin, zmax = z[k1 - 1], z[k2 - 1]
print("phys box", xmin, xmax, ymin, ymax, zmin, zmax)

pred = np.ones(ni * nj * nk, dtype=np.int64)
for k in range(nk):
    for j in range(nj):
        for i in range(ni):
            cx = 0.5 * (x[i] + x[i + 1])
            cy = 0.5 * (y[j] + y[j + 1])
            cz = 0.5 * (z[k] + z[k + 1])
            inside = xmin <= cx <= xmax and ymin <= cy <= ymax and zmin <= cz <= zmax
            idx = k * ni * nj + j * ni + i
            pred[idx] = 2 if inside else 1

print("pred iron", np.sum(pred == 2), "fld iron", np.sum(mat == 2))
print("match", np.all(pred == mat))

# try xmax = x[i2] etc
for label, xa, ya, za in [
    ("node i2", x[i2 - 1], y[j2 - 1], z[k2 - 1]),
    ("node i2+1", x[i2], y[j2], z[k2]),
    ("mid", 0.5 * (x[i1 - 1] + x[i2]), 0.5 * (y[j1 - 1] + y[j2]), 0.5 * (z[k1 - 1] + z[k2])),
]:
    pred2 = np.ones(ni * nj * nk, dtype=np.int64)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                cx = 0.5 * (x[i] + x[i + 1])
                cy = 0.5 * (y[j] + y[j + 1])
                cz = 0.5 * (z[k] + z[k + 1])
                inside = x[i1 - 1] <= cx <= xa and y[j1 - 1] <= cy <= ya and z[k1 - 1] <= cz <= za
                idx = k * ni * nj + j * ni + i
                pred2[idx] = 2 if inside else 1
print("match i2+1", np.all(pred2 == mat))
