#!/usr/bin/env python3
"""
FLD (CRDL-FLD) binary mesh + solution parser for Software Cradle scFLOW field files.

FLD shares the CRDL-FLD container and big-endian record layout with GPH mesh
files, but stores hex cell connectivity, per-vertex solution fields, and
surface BC metadata instead of polyhedral LS_Links topology.
"""

import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np

_LARGE_FLD_BYTES = 512 * 1024 * 1024


def read_i32_be(data: bytes, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "big")


def find_section(data: bytes, name: str) -> int:
    """Return offset of the I4=32 marker before *name*, or -1."""
    name_padded = name.ljust(32).encode("ascii")
    idx = data.find(name_padded)
    if idx < 4:
        return -1
    if read_i32_be(data, idx - 4) == 32:
        return idx - 4
    return -1


def section_end(data: bytes, sec_start: int) -> int:
    """End offset of the section (start of next known section or EOF)."""
    candidates = [
        "FileRevision", "Application", "ApplicationVersion", "ReleaseDate",
        "GridType", "Dimension", "Bias", "Date", "Comments", "Cycle",
        "Unused", "Encoding", "HeaderDataEnd", "OverlapStart_0",
        "LS_CoordinateSystem", "Pressure", "Temperature", "CN01", "VECT",
        "HVEC", "LS_STREAMcoc", "LS_STREAMmultiblock", "LS_Nodes",
        "LS_MatOfElements", "LS_Elements", "LS_VolumeGeometryArray",
        "LS_SurfaceGeometryArray", "LS_SFile", "OverlapEnd",
    ]
    best = len(data)
    for name in candidates:
        off = find_section(data, name)
        if off > sec_start and off < best:
            best = off
    return best


def iter_data_blocks(data: bytes, sec_start: int, sec_end: int):
    """Yield ``(payload_start, byte_count)`` for each data block in a section."""
    pos = sec_start + 40
    n = len(data)
    while pos + 8 <= sec_end and pos + 8 <= n:
        if read_i32_be(data, pos) != 12:
            pos += 4
            continue
        v = read_i32_be(data, pos + 4)
        if v in (4, 8) and pos + 16 <= sec_end:
            dim0 = read_i32_be(data, pos + 8)
            dim1 = read_i32_be(data, pos + 12)
            if 0 < dim0 < 10_000_000 and 0 < dim1 < 10_000_000:
                pos += 16
                continue
        bc = v
        if bc <= 0 or pos + 8 + bc + 4 > sec_end:
            pos += 4
            continue
        payload_end = pos + 8 + bc
        if read_i32_be(data, payload_end) != bc:
            pos += 4
            continue
        yield pos + 8, bc
        pos = payload_end + 4


@contextmanager
def open_fld_buffer(filepath: str):
    """Yield a bytes-like buffer; mmap files larger than 512 MiB."""
    size = Path(filepath).stat().st_size
    if size <= _LARGE_FLD_BYTES:
        with open(filepath, "rb") as f:
            yield f.read()
        return
    import mmap
    f = open(filepath, "rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield mm
        finally:
            mm.close()
    finally:
        f.close()


def _parse_ls_nodes(data: bytes) -> tuple[Optional[np.ndarray], int]:
    sec_start = find_section(data, "LS_Nodes")
    if sec_start < 0:
        return None, 0
    sec_end = section_end(data, sec_start)
    f64_blocks = [(p, bc) for p, bc in iter_data_blocks(data, sec_start, sec_end)
                  if bc >= 8 and bc % 8 == 0]
    if len(f64_blocks) < 3:
        return None, 0
    sizes = [bc for _, bc in f64_blocks]
    target = max(set(sizes), key=sizes.count)
    trio = [(p, bc) for p, bc in f64_blocks if bc == target][:3]
    if len(trio) < 3:
        return None, 0
    n_vertices = trio[0][1] // 8
    axes = [
        np.frombuffer(data, dtype=">f8", count=n_vertices, offset=p).astype(np.float64).copy()
        for p, _ in trio
    ]
    return np.column_stack(axes), n_vertices


def _parse_hex_cells(data: bytes) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return ``(cell_conn (n_cells, 8), material_id (n_cells,))``."""
    sec_mat = find_section(data, "LS_MatOfElements")
    sec_elem = find_section(data, "LS_Elements")
    if sec_mat < 0 or sec_elem < 0:
        return None, None
    mat_blocks = list(iter_data_blocks(data, sec_mat, section_end(data, sec_mat)))
    elem_blocks = list(iter_data_blocks(data, sec_elem, section_end(data, sec_elem)))
    if not mat_blocks or not elem_blocks:
        return None, None
    mat = np.frombuffer(
        data, dtype=">i4", count=mat_blocks[0][1] // 4, offset=mat_blocks[0][0],
    ).astype(np.int64).copy()
    n_cells = mat.size
    conn_p, conn_bc = max(elem_blocks, key=lambda b: b[1])
    if conn_bc != n_cells * 32:
        return None, None
    conn = np.frombuffer(
        data, dtype=">i4", count=conn_bc // 4, offset=conn_p,
    ).astype(np.int64).copy()
    if conn.size % 8 != 0:
        return None, None
    return conn.reshape(-1, 8), mat


def _f64_field_blocks(data: bytes, section_name: str) -> list[np.ndarray]:
    """Return all float64 payload arrays in a named field section."""
    sec_start = find_section(data, section_name)
    if sec_start < 0:
        return []
    sec_end = section_end(data, sec_start)
    out: list[np.ndarray] = []
    for p, bc in iter_data_blocks(data, sec_start, sec_end):
        if bc >= 8 and bc % 8 == 0:
            out.append(
                np.frombuffer(data, dtype=">f8", count=bc // 8, offset=p)
                .astype(np.float64).copy()
            )
    return out


def _parse_volume_names(data: bytes) -> list[str]:
    sec_start = find_section(data, "LS_VolumeGeometryArray")
    if sec_start < 0:
        return []
    sec_end = section_end(data, sec_start)
    for p, bc in iter_data_blocks(data, sec_start, sec_end):
        raw = data[p : p + bc]
        if all(b == 0 or 32 <= b < 127 for b in raw):
            text = raw.decode("ascii", errors="replace").strip("\x00").rstrip()
            if text:
                names = [s.strip() for s in text.split() if s.strip()]
                if names:
                    return names
    return []


def _filter_by_mat(
    quads: list[tuple[int, int, int, int]],
    arr3_slice: np.ndarray,
    mat: np.ndarray,
    material: int,
) -> list[tuple[int, int, int, int]]:
    return [
        quads[i] for i in range(len(quads))
        if mat[int(arr3_slice[i]) - 1] == material
    ]


def _build_face_list_and_bcs(
    data: bytes,
    mat: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int]]]:
    """
    Build the NGON face list and BC PointList index ranges.

    Returns ``(faces, bc_plan)`` where each BC entry is
    ``(name, start_index_0based, count)`` into *faces*.
    """
    sec_start = find_section(data, "LS_SurfaceGeometryArray")
    if sec_start < 0:
        return [], []
    sec_end = section_end(data, sec_start)
    blocks = list(iter_data_blocks(data, sec_start, sec_end))
    if len(blocks) < 6:
        return [], []

    meta1 = [
        read_i32_be(data, blocks[1][0] + i * 4)
        for i in range(min(18, blocks[1][1] // 4))
    ]
    while len(meta1) < 15:
        meta1.append(0)

    arr3 = np.frombuffer(
        data, dtype=">i4", count=blocks[3][1] // 4, offset=blocks[3][0],
    )
    arr5 = np.frombuffer(
        data, dtype=">i4", count=blocks[5][1] // 4, offset=blocks[5][0],
    )
    quads = [tuple(arr5[i : i + 4]) for i in range(0, len(arr5), 4)]

    c_entb, c_entf, c_mom, c_parts = meta1[2], meta1[3], meta1[7], meta1[10]
    c_xmax, c_xmin, c_ymax, c_surf = meta1[12], meta1[13], meta1[14], meta1[11]
    c_ymin = meta1[15] if len(meta1) > 15 else 0
    c_zmax = meta1[16] if len(meta1) > 16 else 0
    c_zmin = meta1[17] if len(meta1) > 17 else 0

    off = 0
    slices: list[slice] = []
    for c in [c_entb, c_entf, c_mom, c_parts, c_xmax, c_xmin, c_ymax, c_surf, c_ymin, c_zmax, c_zmin]:
        slices.append(slice(off, off + c))
        off += c

    mat1_idx = [i for i in range(len(arr3)) if mat[arr3[i] - 1] == 1]
    mat2_idx = [i for i in range(len(arr3)) if mat[arr3[i] - 1] == 2]
    qm1 = [quads[i] for i in mat1_idx]
    qm2 = [quads[i] for i in mat2_idx]

    entb_m2 = len([
        q for i, q in enumerate(quads[slices[0]])
        if mat[int(arr3[slices[0]][i]) - 1] == 2
    ])
    entb_m1 = c_entb - entb_m2
    parts_m1 = _filter_by_mat(quads[slices[3]], arr3[slices[3]], mat, 1)
    parts_m2 = _filter_by_mat(quads[slices[3]], arr3[slices[3]], mat, 2)
    ymax_m2_n = len(qm2) - entb_m2 - 2 * len(parts_m2)
    ymax_m1_n = c_ymax - ymax_m2_n

    seg1 = sum([list(quads[s]) for s in slices], [])
    seg2 = (
        qm1[:entb_m1] + qm2[:entb_m2]
        + parts_m1 + parts_m2 + parts_m1 + parts_m2
        + qm1[-ymax_m1_n:] + qm2[-ymax_m2_n:]
    )
    faces = seg1 + seg2

    # BC names from surface section (18-byte ASCII blocks after block 8)
    bc_names: list[str] = []
    for p, bc in blocks[8:]:
        if bc == 18:
            bc_names.append(data[p : p + bc].decode("ascii", errors="replace").strip())

    def _pick_name(prefix: str, default: str) -> str:
        for n in bc_names:
            if n == prefix or n.startswith(prefix):
                return n
        return default

    ymax_name = _pick_name("Ymax", "Ymax")
    # Face slices: entb, entf, mom, parts, xmax, xmin, ymax, surf(duplicate PARTS).
    # BC PointList order: entb, entf, mom, parts, surface, xmax, xmin, ymax.
    seg1_counts = [c_entb, c_entf, c_mom, c_parts, c_surf, c_xmax, c_xmin, c_ymax]
    seg1_bc_names = [
        "@UNDEFINEDENTB",
        "@UNDEFINEDENTF",
        "@UNDEFINEDMOM",
        "PARTS",
        "SURFACE",
        _pick_name("Xmax", "Xmax"),
        _pick_name("Xmin", "Xmin"),
        ymax_name,
    ]

    bc_plan: list[tuple[str, int, int]] = []
    idx = 0
    for name, cnt in zip(seg1_bc_names, seg1_counts):
        bc_plan.append((name, idx, cnt))
        idx += cnt

    mat_bc_names = [
        "@UNDEFINEDENTB(MAT1)", "@UNDEFINEDENTB(MAT2)",
        "PARTS(MAT1)", "PARTS(MAT2)",
        "SURFACE(MAT1)", "SURFACE(MAT2)",
        f"{ymax_name}(MAT1)", f"{ymax_name}(MAT2)",
    ]
    mat_counts = [
        entb_m1, entb_m2, len(parts_m1), len(parts_m2),
        len(parts_m1), len(parts_m2), ymax_m1_n, ymax_m2_n,
    ]
    seg2_start = len(seg1)
    for name, cnt in zip(mat_bc_names, mat_counts):
        bc_plan.append((name, seg2_start, cnt))
        seg2_start += cnt

    return faces, bc_plan


def parse_fld(filepath: str) -> dict[str, Any]:
    """Parse an FLD file into a structured mesh + solution dict."""
    with open_fld_buffer(filepath) as data:
        result: dict[str, Any] = {
            "file_size": len(data),
            "vertices": None,
            "n_vertices": 0,
            "cell_conn": None,
            "material": None,
            "n_cells": 0,
            "faces": [],
            "bc_plan": [],
            "volume_names": [],
            "fields": {},
        }

        xyz, n_verts = _parse_ls_nodes(data)
        cell_conn, mat = _parse_hex_cells(data)
        if xyz is not None:
            result["vertices"] = xyz
            result["n_vertices"] = n_verts
        if cell_conn is not None and mat is not None:
            result["cell_conn"] = cell_conn
            result["material"] = mat
            result["n_cells"] = int(cell_conn.shape[0])

        result["volume_names"] = _parse_volume_names(data)
        if mat is not None:
            faces, bc_plan = _build_face_list_and_bcs(data, mat)
            result["faces"] = faces
            result["bc_plan"] = bc_plan

        # Solution fields (vertex-centred, length n_vertices).
        n = n_verts or 0
        temp_blocks = _f64_field_blocks(data, "Temperature")
        cn01_blocks = _f64_field_blocks(data, "CN01")
        pres_blocks = _f64_field_blocks(data, "Pressure")
        vect_blocks = _f64_field_blocks(data, "VECT")
        hvec_blocks = _f64_field_blocks(data, "HVEC")

        fields: dict[str, np.ndarray] = {}
        if pres_blocks and pres_blocks[0].size == n:
            fields["PRES"] = pres_blocks[0]
        if temp_blocks:
            if temp_blocks[0].size == n:
                fields["TEMP"] = temp_blocks[0]
                fields["ATMS"] = temp_blocks[0].copy()
            if len(temp_blocks) > 3 and temp_blocks[3].size == n:
                fields["TURK"] = temp_blocks[3]
            if len(temp_blocks) > 6 and temp_blocks[6].size == n:
                fields["TEPS"] = temp_blocks[6]
        if cn01_blocks:
            if cn01_blocks[0].size == n:
                fields["CN01"] = cn01_blocks[0]
            if len(cn01_blocks) > 3 and cn01_blocks[3].size == n:
                fields["HTRC"] = cn01_blocks[3]
            if len(cn01_blocks) > 6 and cn01_blocks[6].size == n:
                fields["SURT"] = cn01_blocks[6]
            if len(cn01_blocks) > 9 and cn01_blocks[9].size == n:
                fields["HTFX"] = cn01_blocks[9]
        if len(vect_blocks) >= 3 and all(a.size == n for a in vect_blocks[:3]):
            fields["VECTX"] = vect_blocks[0]
            fields["VECTY"] = vect_blocks[1]
            fields["VECTZ"] = vect_blocks[2]
        if len(hvec_blocks) >= 3 and all(a.size == n for a in hvec_blocks[:3]):
            fields["HVECX"] = hvec_blocks[0]
            fields["HVECY"] = hvec_blocks[1]
            fields["HVECZ"] = hvec_blocks[2]

        result["fields"] = fields
        return result


def describe_fld_sections(filepath: str) -> list[dict[str, Any]]:
    """Return a section layout summary for format inspection."""
    with open_fld_buffer(filepath) as data:
        names = [
            "FileRevision", "Application", "LS_CoordinateSystem", "Pressure",
            "Temperature", "CN01", "VECT", "HVEC", "LS_Nodes",
            "LS_MatOfElements", "LS_Elements", "LS_VolumeGeometryArray",
            "LS_SurfaceGeometryArray", "LS_SFile", "OverlapEnd",
        ]
        found = []
        for name in names:
            off = find_section(data, name)
            if off >= 0:
                found.append((off, name))
        found.sort()
        layout = []
        for i, (off, name) in enumerate(found):
            end = found[i + 1][0] if i + 1 < len(found) else len(data)
            layout.append({
                "offset_hex": f"0x{off:04X}",
                "size": end - off,
                "name": name,
            })
        return layout
