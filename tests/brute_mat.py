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

best = (0, "")
for xi0 in [i1 - 1, i1]:
    for xi1 in [i2 - 1, i2, i2 + 1]:
        if xi1 >= len(x):
            continue
        for yj0 in [j1 - 1, j1]:
            for yj1 in [j2 - 1, j2]:
                for zk0 in [k1 - 1, k1]:
                    for zk1 in [k2 - 1, k2, k2 + 1]:
                        if zk1 >= len(z):
                            continue
                        xmin, xmax = x[xi0], x[xi1]
                        ymin, ymax = y[yj0], y[yj1]
                        zmin, zmax = z[zk0], z[zk1]

                        def corner_inside(ii, jj, kk):
                            return (
                                xmin <= x[ii] <= xmax
                                and ymin <= y[jj] <= ymax
                                and zmin <= z[kk] <= zmax
                            )

                        pred = []
                        for k in range(nk):
                            for j in range(nj):
                                for i in range(ni):
                                    corners = [
                                        (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
                                        (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
                                    ]
                                    pred.append(2 if all(corner_inside(*c) for c in corners) else 1)
                        pred = np.array(pred)
                        m = int(np.sum(pred == mat))
                        if m > best[0]:
                            best = (
                                m,
                                f"x[{xi0},{xi1}] y[{yj0},{yj1}] z[{zk0},{zk1}] iron={np.sum(pred == 2)}",
                            )
print("best", best)
