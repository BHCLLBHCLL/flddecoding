#!/usr/bin/env python3
"""
Build scFLOW structured hex mesh from SDAT (.s) CXYZ spacing and PARTS iron box.

Uses multi-material interface node duplication (hanging nodes) matching vendor layout.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from s_model import SdatModel, build_structured_coords

# MPI partition split along I (matches vendor ex1_e layout).
_I_SPLIT = 30

_CORNER_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]


@dataclass
class BuiltMesh:
    vertices: np.ndarray
    cell_conn: np.ndarray
    material: np.ndarray
    node_map: dict[tuple[int, int, int], list[int]]
    ni: int
    nj: int
    nk: int
    faces: list[tuple[int, int, int, int]]
    bc_plan: list[tuple[str, int, int]]
    volume_names: list[str]


def _iron_box(model: SdatModel) -> Optional[tuple[int, int, int, int, int, int]]:
    """Return iron PARTS box as (i1, i2, j1, j2, k1, k2) 1-based node indices."""
    for part in model.parts:
        if part.box and part.material_id != 1:
            return part.box
    for part in reversed(model.parts):
        if part.box:
            return part.box
    return None


def _cell_material(
    i: int, j: int, k: int,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    box: tuple[int, int, int, int, int, int],
) -> int:
    """Iron (2) when all eight cell corners lie inside the PARTS node box."""
    i1, i2, j1, j2, k1, k2 = box
    xmin, xmax = x[i1 - 1], x[i2]
    ymin, ymax = y[j1 - 1], y[j2]
    zmin, zmax = z[k1 - 1], z[k2]
    corners = [
        (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
        (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
    ]
    inside = all(
        xmin <= x[ci] <= xmax and ymin <= y[cj] <= ymax and zmin <= z[ck] <= zmax
        for ci, cj, ck in corners
    )
    return 2 if inside else 1


def _needs_dup(i: int, j: int, k: int, cmat: np.ndarray) -> bool:
    mats: set[int] = set()
    for di in (-1, 0):
        for dj in (-1, 0):
            for dk in (-1, 0):
                ci, cj, ck = i + di, j + dj, k + dk
                if 0 <= ci < cmat.shape[0] and 0 <= cj < cmat.shape[1] and 0 <= ck < cmat.shape[2]:
                    mats.add(int(cmat[ci, cj, ck]))
    return len(mats) > 1


def _assign_node_ids(
    nx: int, ny: int, nz: int, cmat: np.ndarray,
) -> dict[tuple[int, int, int], list[int]]:
    node_ids: dict[tuple[int, int, int], list[int]] = {}
    next_id = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(_I_SPLIT):
                node_ids[(i, j, k)] = [next_id]
                next_id += 1
                if _needs_dup(i, j, k, cmat):
                    node_ids[(i, j, k)].append(next_id)
                    next_id += 1
    for k in range(nz):
        for j in range(ny):
            for i in range(_I_SPLIT, nx):
                node_ids[(i, j, k)] = [next_id]
                next_id += 1
                if _needs_dup(i, j, k, cmat):
                    node_ids[(i, j, k)].append(next_id)
                    next_id += 1
    return node_ids


def _pick_node(
    i: int, j: int, k: int,
    cell_m: int,
    node_ids: dict[tuple[int, int, int], list[int]],
) -> int:
    ids = node_ids[(i, j, k)]
    if len(ids) == 2 and cell_m == 2:
        return ids[1]
    return ids[0]


def _build_vertices(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    node_ids: dict[tuple[int, int, int], list[int]],
) -> np.ndarray:
    n = max(max(ids) for ids in node_ids.values())
    verts = np.zeros((n, 3), dtype=np.float64)
    for (i, j, k), ids in node_ids.items():
        coord = np.array([x[i], y[j], z[k]], dtype=np.float64)
        for nid in ids:
            verts[nid - 1] = coord
    return verts


def _build_cell_conn(
    ni: int, nj: int, nk: int,
    cmat: np.ndarray,
    node_ids: dict[tuple[int, int, int], list[int]],
) -> np.ndarray:
    rows: list[list[int]] = []
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                m = int(cmat[i, j, k])
                row = [
                    _pick_node(i + oi, j + oj, k + ok, m, node_ids)
                    for oi, oj, ok in _CORNER_OFFSETS
                ]
                rows.append(row)
    return np.array(rows, dtype=np.int64)


def _face_quad(
    ci: int, cj: int, ck: int, axis: str, side: int,
    cmat: np.ndarray, node_ids: dict[tuple[int, int, int], list[int]],
) -> tuple[int, int, int, int]:
    """Quad on cell (ci,cj,ck) boundary; side 0=low, 1=high along axis."""
    m = int(cmat[ci, cj, ck])
    if axis == "x":
        if side == 0:
            pts = [(ci, cj, ck), (ci, cj + 1, ck), (ci, cj + 1, ck + 1), (ci, cj, ck + 1)]
        else:
            pts = [(ci + 1, cj, ck), (ci + 1, cj, ck + 1), (ci + 1, cj + 1, ck + 1), (ci + 1, cj + 1, ck)]
    elif axis == "y":
        if side == 0:
            pts = [(ci, cj, ck), (ci, cj, ck + 1), (ci + 1, cj, ck + 1), (ci + 1, cj, ck)]
        else:
            pts = [(ci, cj + 1, ck), (ci + 1, cj + 1, ck), (ci + 1, cj + 1, ck + 1), (ci, cj + 1, ck + 1)]
    else:
        if side == 0:
            pts = [(ci, cj, ck), (ci + 1, cj, ck), (ci + 1, cj + 1, ck), (ci, cj + 1, ck)]
        else:
            pts = [(ci, cj, ck + 1), (ci, cj + 1, ck + 1), (ci + 1, cj + 1, ck + 1), (ci + 1, cj, ck + 1)]
    return tuple(_pick_node(i, j, k, m, node_ids) for i, j, k in pts)


def _build_boundary_faces(
    ni: int, nj: int, nk: int,
    cmat: np.ndarray,
    node_ids: dict[tuple[int, int, int], list[int]],
    region_names: list[str],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int]]]:
    faces: list[tuple[int, int, int, int]] = []
    bc_plan: list[tuple[str, int, int]] = []

    def add_group(name: str, quads: list[tuple[int, int, int, int]]) -> None:
        if not quads:
            return
        bc_plan.append((name, len(faces), len(quads)))
        faces.extend(quads)

    xmin = [_face_quad(0, j, k, "x", 0, cmat, node_ids) for k in range(nk) for j in range(nj)]
    xmax = [_face_quad(ni - 1, j, k, "x", 1, cmat, node_ids) for k in range(nk) for j in range(nj)]
    ymin = [_face_quad(i, 0, k, "y", 0, cmat, node_ids) for k in range(nk) for i in range(ni)]
    ymax = [_face_quad(i, nj - 1, k, "y", 1, cmat, node_ids) for k in range(nk) for i in range(ni)]
    zmin = [_face_quad(i, j, 0, "z", 0, cmat, node_ids) for j in range(nj) for i in range(ni)]
    zmax = [_face_quad(i, j, nk - 1, "z", 1, cmat, node_ids) for j in range(nj) for i in range(ni)]

    name_map = {
        "Xmin": xmin, "Xmax": xmax, "Ymin": ymin, "Ymax": ymax, "Zmin": zmin, "Zmax": zmax,
    }
    for default, quads in name_map.items():
        name = default
        for r in region_names:
            if r.upper().startswith(default[0].upper()) or default.lower() in r.lower():
                name = r.split("!")[0].strip()
                break
        add_group(name, quads)

    # Fluid–solid interface (mat1/mat2).
    iface: list[tuple[int, int, int, int]] = []
    for k in range(nk):
        for j in range(nj):
            for i in range(ni - 1):
                if cmat[i, j, k] != cmat[i + 1, j, k]:
                    iface.append(_face_quad(i, j, k, "x", 1, cmat, node_ids))
            for i in range(ni):
                if j < nj - 1 and cmat[i, j, k] != cmat[i, j + 1, k]:
                    iface.append(_face_quad(i, j, k, "y", 1, cmat, node_ids))
            for i in range(ni):
                if k < nk - 1 and cmat[i, j, k] != cmat[i, j, k + 1]:
                    iface.append(_face_quad(i, j, k, "z", 1, cmat, node_ids))
    add_group("INTERFACE", iface)

    return faces, bc_plan


def build_mesh_from_sdat(
    model: SdatModel,
    volume_names: Optional[list[str]] = None,
) -> BuiltMesh:
    """Construct hex mesh from parsed SDAT model."""
    x, y, z = build_structured_coords(model)
    ni, nj, nk = model.ni, model.nj, model.nk
    if ni <= 0 or nj <= 0 or nk <= 0:
        raise ValueError("SDAT model missing mesh dimensions (ni, nj, nk)")
    box = _iron_box(model)
    if box is None:
        raise ValueError("No PARTS box found in .s for solid region")

    cmat = np.zeros((ni, nj, nk), dtype=np.int64)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                cmat[i, j, k] = _cell_material(i, j, k, x, y, z, box)

    node_ids = _assign_node_ids(len(x), len(y), len(z), cmat)
    vertices = _build_vertices(x, y, z, node_ids)
    cell_conn = _build_cell_conn(ni, nj, nk, cmat, node_ids)
    material = cmat.reshape(-1)
    faces, bc_plan = _build_boundary_faces(ni, nj, nk, cmat, node_ids, model.region_names)

    if volume_names is None:
        volume_names = ["PARTS1", "PARTS2", "Domain(cuboid)", "Iron"]

    return BuiltMesh(
        vertices=vertices,
        cell_conn=cell_conn,
        material=material,
        node_map=node_ids,
        ni=ni,
        nj=nj,
        nk=nk,
        faces=faces,
        bc_plan=bc_plan,
        volume_names=volume_names,
    )


def mesh_to_fld_dict(built: BuiltMesh, fields: dict[str, np.ndarray]) -> dict:
    """Convert BuiltMesh to parse_fld-compatible dict for fld2cgns."""
    return {
        "vertices": built.vertices,
        "n_vertices": int(built.vertices.shape[0]),
        "cell_conn": built.cell_conn,
        "material": built.material,
        "n_cells": int(built.cell_conn.shape[0]),
        "faces": built.faces,
        "bc_plan": built.bc_plan,
        "volume_names": built.volume_names,
        "fields": fields,
    }
