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
dups = set(p for p, n in fld_at.items() if len(n) == 2)
box = model.parts[-1].box
i1, i2, j1, j2, k1, k2 = box


def needs_dup(i, j, k):
    in_box = i1 <= i + 1 <= i2 and j1 <= j + 1 <= j2 and k1 <= k + 1 <= k2
    on_box_surf = in_box and (
        i + 1 == i1 or i + 1 == i2 or j + 1 == j1 or j + 1 == j2 or k + 1 == k1 or k + 1 == k2
    )
    outside_max = (
        (i + 1 == i2 + 1 and j1 <= j + 1 <= j2 and k1 <= k + 1 <= k2)
        or (j + 1 == j2 + 1 and i1 <= i + 1 <= i2 and k1 <= k + 1 <= k2)
        or (k + 1 == k2 + 1 and i1 <= i + 1 <= i2 and j1 <= j + 1 <= j2)
    )
    return on_box_surf or outside_max


built = {(i, j, k) for k in range(nz) for j in range(ny) for i in range(nx) if needs_dup(i, j, k)}
print("built", len(built), "fld", len(dups))
print("missing", len(dups - built), "extra", len(built - dups))
if dups - built:
    print("sample missing", list(dups - built)[:5])
if built - dups:
    print("sample extra", list(built - dups)[:5])
