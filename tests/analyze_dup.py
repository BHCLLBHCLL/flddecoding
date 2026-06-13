#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter, defaultdict
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
dups = {p: n for p, n in fld_at.items() if len(n) == 2}
box = model.parts[-1].box
i1, i2, j1, j2, k1, k2 = box

def in_box(i, j, k):
    return i1 <= i + 1 <= i2 and j1 <= j + 1 <= j2 and k1 <= k + 1 <= k2

def on_surf(i, j, k):
    return in_box(i, j, k) and (
        i + 1 == i1 or i + 1 == i2 or j + 1 == j1 or j + 1 == j2 or k + 1 == k1 or k + 1 == k2
    )

other = [p for p in dups if not on_surf(*p)]
print("other dup count", len(other))
tags = []
for i, j, k in other:
    tag = []
    if i + 1 == i1 - 1:
        tag.append("i_low-1")
    if i + 1 == i2 + 1:
        tag.append("i_high+1")
    if j + 1 == j1 - 1:
        tag.append("j_low-1")
    if j + 1 == j2 + 1:
        tag.append("j_high+1")
    if k + 1 == k1 - 1:
        tag.append("k_low-1")
    if k + 1 == k2 + 1:
        tag.append("k_high+1")
    if in_box(i, j, k):
        tag.append("in_box_int")
    tags.append(",".join(tag) or "other")
print(Counter(tags).most_common(15))
