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
v = fld["vertices"]
nx, ny, nz = len(x), len(y), len(z)
fc = np.array([[x[i], y[j], z[k]] for k in range(nz) for j in range(ny) for i in range(nx)])
fld_at = defaultdict(list)
for nid in range(1, len(v) + 1):
    ti = int(np.argmin(np.sum((fc - v[nid - 1]) ** 2, axis=1)))
    k, j, i = np.unravel_index(ti, (nz, ny, nx))
    fld_at[(i, j, k)].append(nid)

# primary = first id at each position
primary = {p: ids[0] for p, ids in fld_at.items()}

# fit nid for i<30
samples = [(primary[(i, j, 0)], i, j, 0) for j in range(ny) for i in range(30)]
for a, i, j, k in samples[:5]:
    print(a, i, j, 1 + i + 30 * j + 650 * k)

# check i>=30 block
for i in [30, 31, 57]:
    print(i, primary.get((i, 0, 0)), "delta from i=29", primary.get((i, 0, 0)) - primary[(29, 0, 0)])

# dup adds 1 to subsequent primaries?
# count primaries before dup region
print("primary count", len(primary))
