#!/usr/bin/env python3
"""
Build scFLOW structured hex mesh from SDAT (.s) CXYZ spacing and PARTS boxes.

Supports multiple solid parts with multiple box regions per part and material IDs 1–7.
Uses multi-material interface node duplication (one vertex per material at each node).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from surface_builder import build_vendor_surfaces, vendor_bc_plan_from_categories
from s_model import SdatModel, build_structured_coords

# MPI partition split along I: first block size is ni - ni//2 (ex2 → 65, ex4 → 49).
def _mpi_i_split(ni: int) -> int:
    return ni - ni // 2


def _uses_mpi_split_cell_order(ni: int, nk: int) -> bool:
    """True for vendor two-block I ordering (ex2/ex3); ex4 uses full-width k-j-i rows."""
    i1 = _mpi_i_split(ni)
    return nk <= 43 or ni != 2 * i1


def _iter_cells_vendor(ni: int, nj: int, nk: int):
    """Yield (i, j, k) in vendor LS_Elements order."""
    if _uses_mpi_split_cell_order(ni, nk):
        i1 = _mpi_i_split(ni)
        for k in range(nk):
            for j in range(nj):
                for i in range(i1):
                    yield i, j, k
        for k in range(nk):
            for j in range(nj):
                for i in range(i1, ni):
                    yield i, j, k
    else:
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    yield i, j, k


def _flatten_cells_vendor(arr: np.ndarray, ni: int, nj: int, nk: int) -> np.ndarray:
    """Flatten (ni, nj, nk) per-cell arrays in vendor LS_Elements order."""
    return np.array(
        [arr[i, j, k] for i, j, k in _iter_cells_vendor(ni, nj, nk)],
        dtype=arr.dtype,
    )


_CORNER_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]


@dataclass
class BuiltMesh:
    vertices: np.ndarray
    cell_conn: np.ndarray
    material: np.ndarray
    cell_part: np.ndarray
    part_names: list[str]
    node_map: dict[tuple[int, int, int], list[int]]
    ni: int
    nj: int
    nk: int
    faces: list[tuple[int, int, int, int]]
    bc_plan: list[tuple[str, int, int]]
    volume_names: list[str]
    surface_cats: dict


def _cell_corners(i: int, j: int, k: int) -> list[tuple[int, int, int]]:
    return [
        (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
        (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
    ]


def _cell_in_box(
    i: int, j: int, k: int,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    box: tuple[int, int, int, int, int, int],
) -> bool:
    """True when all eight cell corners lie inside the PARTS node box."""
    i1, i2, j1, j2, k1, k2 = box
    xmin, xmax = x[i1 - 1], x[i2]
    ymin, ymax = y[j1 - 1], y[j2]
    zmin, zmax = z[k1 - 1], z[k2]
    for ci, cj, ck in _cell_corners(i, j, k):
        if not (xmin <= x[ci] <= xmax and ymin <= y[cj] <= ymax and zmin <= z[ck] <= zmax):
            return False
    return True


def _build_cell_parts_and_materials(
    model: SdatModel,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign per-cell part id (PARTS order) and material id from box regions."""
    ni, nj, nk = model.ni, model.nj, model.nk
    cpart = np.ones((ni, nj, nk), dtype=np.int64)
    cmat = np.ones((ni, nj, nk), dtype=np.int64)
    for part in model.parts:
        if not part.boxes:
            continue
        for box in part.boxes:
            i1, i2, j1, j2, k1, k2 = box
            i_lo = max(0, i1 - 1)
            i_hi = min(ni, i2)
            j_lo = max(0, j1 - 1)
            j_hi = min(nj, j2)
            k_lo = max(0, k1 - 1)
            k_hi = min(nk, k2)
            for k in range(k_lo, k_hi):
                for j in range(j_lo, j_hi):
                    for i in range(i_lo, i_hi):
                        if _cell_in_box(i, j, k, x, y, z, box):
                            cpart[i, j, k] = part.part_id
                            cmat[i, j, k] = part.material_id
    return cpart, cmat


def _materials_at_node(
    i: int, j: int, k: int,
    cmat: np.ndarray,
) -> list[int]:
    """Sorted material ids of all cells sharing this structured node."""
    mats: set[int] = set()
    ni, nj, nk = cmat.shape
    for di in (-1, 0):
        for dj in (-1, 0):
            for dk in (-1, 0):
                ci, cj, ck = i + di, j + dj, k + dk
                if 0 <= ci < ni and 0 <= cj < nj and 0 <= ck < nk:
                    mats.add(int(cmat[ci, cj, ck]))
    return sorted(mats)


def _assign_node_ids(
    nx: int, ny: int, nz: int,
    cmat: np.ndarray,
) -> tuple[dict[tuple[int, int, int], list[int]], dict[tuple[int, int, int], dict[int, int]]]:
    """Assign node ids with one vertex per material at multi-material interfaces."""
    node_ids: dict[tuple[int, int, int], list[int]] = {}
    mat_index: dict[tuple[int, int, int], dict[int, int]] = {}
    i_split = _mpi_i_split(nx)
    next_id = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(i_split):
                mats = _materials_at_node(i, j, k, cmat)
                ids = [next_id + j for j in range(len(mats))]
                next_id += len(mats)
                node_ids[(i, j, k)] = ids
                mat_index[(i, j, k)] = {m: idx for idx, m in enumerate(mats)}
            for i in range(i_split, nx):
                mats = _materials_at_node(i, j, k, cmat)
                ids = [next_id + j for j in range(len(mats))]
                next_id += len(mats)
                node_ids[(i, j, k)] = ids
                mat_index[(i, j, k)] = {m: idx for idx, m in enumerate(mats)}
    return node_ids, mat_index


def _pick_node(
    i: int, j: int, k: int,
    cell_m: int,
    node_ids: dict[tuple[int, int, int], list[int]],
    mat_index: dict[tuple[int, int, int], dict[int, int]],
) -> int:
    idx = mat_index[(i, j, k)][cell_m]
    return node_ids[(i, j, k)][idx]


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
    mat_index: dict[tuple[int, int, int], dict[int, int]],
) -> np.ndarray:
    rows: list[list[int]] = []
    for i, j, k in _iter_cells_vendor(ni, nj, nk):
        m = int(cmat[i, j, k])
        row = [
            _pick_node(i + oi, j + oj, k + ok, m, node_ids, mat_index)
            for oi, oj, ok in _CORNER_OFFSETS
        ]
        rows.append(row)
    return np.array(rows, dtype=np.int64)


def _face_quad(
    ci: int, cj: int, ck: int, axis: str, side: int,
    cmat: np.ndarray,
    node_ids: dict[tuple[int, int, int], list[int]],
    mat_index: dict[tuple[int, int, int], dict[int, int]],
) -> tuple[int, int, int, int]:
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
    return tuple(_pick_node(i, j, k, m, node_ids, mat_index) for i, j, k in pts)


def _build_boundary_faces(
    ni: int, nj: int, nk: int,
    cmat: np.ndarray,
    cpart: np.ndarray,
    node_ids: dict[tuple[int, int, int], list[int]],
    mat_index: dict[tuple[int, int, int], dict[int, int]],
    region_names: list[str],
    material_flat: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int]], dict]:
    ymax_name = "Ymax"
    for r in region_names:
        if r.upper().startswith("Y") and "max" in r.lower():
            ymax_name = r.split("!")[0].strip()
            break

    _, cats = build_vendor_surfaces(
        ni, nj, nk, cmat, cpart,
        lambda ci, cj, ck, axis, side: _face_quad(ci, cj, ck, axis, side, cmat, node_ids, mat_index),
    )
    faces, bc_plan = vendor_bc_plan_from_categories(cats, material_flat, ymax_name=ymax_name)
    return faces, bc_plan, cats


def build_mesh_from_sdat(
    model: SdatModel,
    volume_names: Optional[list[str]] = None,
) -> BuiltMesh:
    """Construct hex mesh from parsed SDAT model."""
    x, y, z = build_structured_coords(model)
    ni, nj, nk = model.ni, model.nj, model.nk
    if ni <= 0 or nj <= 0 or nk <= 0:
        raise ValueError("SDAT model missing mesh dimensions (ni, nj, nk)")

    solid_boxes = sum(len(p.boxes) for p in model.parts)
    if solid_boxes == 0:
        raise ValueError("No PARTS box regions found in .s for solid parts")

    part_names = [p.name for p in model.parts]
    if volume_names is None:
        from xemt_model import volume_names_from_parts
        volume_names = volume_names_from_parts(part_names)

    cpart, cmat = _build_cell_parts_and_materials(model, x, y, z)
    node_ids, mat_index = _assign_node_ids(len(x), len(y), len(z), cmat)
    vertices = _build_vertices(x, y, z, node_ids)
    cell_conn = _build_cell_conn(ni, nj, nk, cmat, node_ids, mat_index)
    material = _flatten_cells_vendor(cmat, ni, nj, nk)
    cell_part = _flatten_cells_vendor(cpart, ni, nj, nk)
    faces, bc_plan, surface_cats = _build_boundary_faces(
        ni, nj, nk, cmat, cpart, node_ids, mat_index, model.region_names, material,
    )

    return BuiltMesh(
        vertices=vertices,
        cell_conn=cell_conn,
        material=material,
        cell_part=cell_part,
        part_names=part_names,
        node_map=node_ids,
        ni=ni,
        nj=nj,
        nk=nk,
        faces=faces,
        bc_plan=bc_plan,
        volume_names=volume_names,
        surface_cats=surface_cats,
    )


def mesh_to_fld_dict(built: BuiltMesh, fields: dict[str, np.ndarray]) -> dict:
    """Convert BuiltMesh to parse_fld-compatible dict for fld2cgns."""
    return {
        "vertices": built.vertices,
        "n_vertices": int(built.vertices.shape[0]),
        "cell_conn": built.cell_conn,
        "material": built.material,
        "cell_part": built.cell_part,
        "part_names": built.part_names,
        "n_cells": int(built.cell_conn.shape[0]),
        "faces": built.faces,
        "bc_plan": built.bc_plan,
        "volume_names": built.volume_names,
        "surface_cats": built.surface_cats,
        "fields": fields,
    }
