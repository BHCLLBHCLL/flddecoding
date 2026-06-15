#!/usr/bin/env python3
"""
Write FLD (CRDL-FLD) binary files.

Mesh / surface sections are taken from a template FLD (or source mesh FLD).
Field arrays and embedded SDAT are updated from parsed .s settings.
"""

import struct
from pathlib import Path
from typing import Optional

import numpy as np

from fld_model import (
    fld_cell_count,
    find_section,
    iter_data_blocks,
    read_i32_be,
    section_end,
)
from surface_builder import surface_meta_counts


def _write_section_block(payload: bytes) -> bytes:
    """One data block: [12, bc][payload][bc]."""
    bc = len(payload)
    return struct.pack(">ii", 12, bc) + payload + struct.pack(">i", bc)


def _field_section_preamble(n_vertices: int) -> bytes:
    """48-byte vendor header before f64 field blocks (required by scPOST)."""
    return (
        struct.pack(">iiii", 12, 4, 1, 1)
        + struct.pack(">iiii", 12, 4, n_vertices, 4)
        + struct.pack(">iiii", 12, 8, n_vertices, 1)
    )


def _preamble_chunk3(n_vertices: int) -> bytes:
    """16-byte vendor chunk between consecutive vector f64 blocks."""
    return struct.pack(">iiii", 12, 8, n_vertices, 1)


def _field_f64_pad() -> bytes:
    """16-byte pad after scalar f64 blocks."""
    return struct.pack(">iiii", 12, 0, 0, 0)


def _field_section_trailer() -> bytes:
    """48-byte trailer between field metadata labels."""
    return (
        struct.pack(">iiii", 12, 4, 1, 1)
        + struct.pack(">iiii", 12, 4, 2, 4)
        + struct.pack(">iiii", 12, 1, 32, 1)
    )


def _field_section_trailer_link() -> bytes:
    """48-byte trailer before the next field section (e.g. CN01 -> VECT)."""
    return (
        struct.pack(">iiii", 12, 4, 1, 1)
        + struct.pack(">iiii", 12, 4, 2, 4)
        + struct.pack(">iiii", 12, 4, 1, 1)
    )


def _section_suffix() -> bytes:
    return struct.pack(">i", 12)


def _write_meta_label(label: str) -> bytes:
    text = label[:32].ljust(32).encode("ascii")
    return _write_section_block(text)


def _write_f64_scalar(arr: np.ndarray) -> bytes:
    payload = np.ascontiguousarray(arr, dtype=">f8").tobytes()
    return _write_section_block(payload) + _field_f64_pad()


def _write_f64_vector(arr: np.ndarray, n_vertices: int) -> bytes:
    payload = np.ascontiguousarray(arr, dtype=">f8").tobytes()
    return _write_section_block(payload) + _preamble_chunk3(n_vertices)


def _write_field_section_end() -> bytes:
    return _field_section_trailer() + _section_suffix()


def _write_linked_section_end() -> bytes:
    """Tail for CN01 / VECT before the next field section."""
    return (
        _write_section_block(struct.pack(">i", 0))
        + struct.pack(">iiii", 12, 1, 32, 1)
        + _section_suffix()
    )


def _mesh_base_preamble() -> bytes:
    """48-byte header common to mesh sections (LS_Nodes, LS_Elements, ...)."""
    return (
        struct.pack(">iiii", 12, 4, 1, 1)
        + struct.pack(">iiii", 12, 4, 1, 4)
        + struct.pack(">iiii", 12, 4, 1, 1)
    )


def _mesh_vertices_tail(n_vertices: int) -> bytes:
    return (
        struct.pack(">iii", 12, 4, n_vertices)
        + struct.pack(">i", 4)
        + struct.pack(">ii", 12, 8)
        + struct.pack(">i", n_vertices)
        + struct.pack(">i", 1)
    )


def _mesh_nodes_preamble(n_vertices: int) -> bytes:
    return _mesh_base_preamble() + _mesh_vertices_tail(n_vertices)


def _mesh_cells_tail(n_cells: int) -> bytes:
    return (
        struct.pack(">iii", 12, 4, n_cells)
        + struct.pack(">i", 4)
        + struct.pack(">iii", 12, 4, n_cells)
        + struct.pack(">i", 1)
    )


def _mesh_cells_preamble(n_cells: int) -> bytes:
    return _mesh_base_preamble() + _mesh_cells_tail(n_cells)


def _mesh_section_tail() -> bytes:
    """20-byte pad + suffix at end of mesh i32/f64 sections."""
    return _field_f64_pad() + _section_suffix()


def _mesh_i32_block_sep1() -> bytes:
    """16-byte separator between LS_Elements blocks."""
    return struct.pack(">iiii", 12, 4, 1, 1)


def _mesh_i32_block_sep2(count_val: int) -> bytes:
    """16-byte separator before connectivity block (carries flat conn size)."""
    return struct.pack(">iiii", 12, 4, count_val, 1)


def _mesh_geom_sep(val: int) -> bytes:
    """16-byte separator between geometry array blocks."""
    return struct.pack(">iiii", 12, 4, val, 1)


def _mesh_geometry_preamble(count_val: int, first_block_bc: int) -> bytes:
    """112-byte preamble before first block in geometry array sections."""
    return (
        _mesh_base_preamble()
        + struct.pack(">iii", 12, 4, count_val)
        + struct.pack(">i", 4)
        + struct.pack(">iiii", 12, 4, 1, 1)
        + struct.pack(">iii", 12, 4, 256)
        + struct.pack(">i", 4)
        + struct.pack(">iiii", 12, 1, first_block_bc, 1)
    )


def _geometry_preamble_from_template(
    data: bytes,
    section_name: str,
    count_val: int,
    first_block_bc: int,
) -> bytes:
    sec = find_section(data, section_name)
    if sec < 0:
        return _mesh_geometry_preamble(count_val, first_block_bc)
    inner = sec + 40
    sec_end = section_end(data, sec)
    blocks = list(iter_data_blocks(data, sec, sec_end))
    if not blocks:
        return _mesh_geometry_preamble(count_val, first_block_bc)
    pre_len = blocks[0][0] - inner - 8
    if pre_len <= 0:
        return _mesh_geometry_preamble(count_val, first_block_bc)
    pre = bytearray(data[inner:inner + pre_len])
    if len(pre) >= 60:
        struct.pack_into(">i", pre, 56, count_val)
    if len(pre) >= 108:
        struct.pack_into(">i", pre, 104, first_block_bc)
    return bytes(pre)


def _write_named_section(name: str, inner: bytes) -> bytes:
    """Section with [32][name 32B][32] prefix."""
    name_padded = name.ljust(32).encode("ascii")
    return struct.pack(">i", 32) + name_padded + struct.pack(">i", 32) + inner


def _build_f64_section(
    section_name: str,
    arrays: list[np.ndarray],
    n_vertices: int,
) -> bytes:
    """Build a field section with alternating f64 payloads and 4-float metadata blocks."""
    inner = _field_section_preamble(n_vertices)
    for arr in arrays:
        payload = np.ascontiguousarray(arr, dtype=">f8").tobytes()
        inner += _write_section_block(payload)
        if arr.size > 4:
            # Vendor interleaves 4-element metadata blocks between large arrays.
            meta = np.zeros(4, dtype=">f8").tobytes()
            inner += _write_section_block(meta)
    return _write_named_section(section_name, inner)


def _sdat_basename(s_text: str) -> Optional[str]:
    """Parse basename from SDAT POST block (e.g. ex4_e)."""
    lines = s_text.replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "POST" and i + 1 < len(lines):
            name = lines[i + 1].strip()
            return name if name else None
    return None


def fld_cell_count_file(path: str) -> Optional[int]:
    """Return cell count from an FLD file, or None."""
    data = Path(path).read_bytes()
    return fld_cell_count(data)


def resolve_template_fld(
    n_cells: int,
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """
    Find a reference FLD for scPOST-compatible header / geometry blocks.

    Prefers *explicit* when provided and cell count matches. Otherwise searches
    next to the .s file and in ``tests/`` for ``{stem}_63.fld``, ``{stem}_100.fld``,
    etc., and returns the first file whose cell count equals *n_cells*.
    """
    if explicit:
        exp = Path(explicit)
        if not exp.is_file():
            raise FileNotFoundError(f"Template FLD not found: {exp}")
        exp_cells = fld_cell_count_file(str(exp))
        if exp_cells is not None and exp_cells != n_cells:
            raise ValueError(
                f"Template {exp} has {exp_cells} cells, mesh has {n_cells} cells"
            )
        return str(exp)

    stems: list[str] = []
    parents: list[Path] = []
    if s_basename:
        stems.append(s_basename)
    if s_path:
        s_file = Path(s_path)
        stems.append(s_file.stem)
        parents.append(s_file.parent)
    if mesh_file:
        stems.append(Path(mesh_file).stem)
    tests_dir = Path(__file__).resolve().parent / "tests"
    parents.append(tests_dir)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in parents:
        for stem in stems:
            for suffix in ("_63", "_100", "_0", ""):
                c = parent / f"{stem}{suffix}.fld"
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)

    for c in candidates:
        if not c.is_file():
            continue
        n = fld_cell_count_file(str(c))
        if n == n_cells:
            return str(c)
    return None


def _patch_section(data: bytearray, section_name: str, new_inner: bytes) -> None:
    """Replace section payload (after 40-byte header) in place."""
    sec_start = find_section(data, section_name)
    if sec_start < 0:
        raise ValueError(f"Section {section_name} not found in template")
    sec_end = section_end(data, sec_start)
    new_section = _write_named_section(section_name, new_inner)
    if len(new_section) != sec_end - sec_start:
        raise ValueError(
            f"Section {section_name} size mismatch: template {sec_end - sec_start} "
            f"vs new {len(new_section)} — use a matching template FLD"
        )
    data[sec_start:sec_end] = new_section


def patch_cycle(fld_path: str, cycle: int) -> None:
    """Set cycle number in the FLD Cycle header section."""
    data = bytearray(Path(fld_path).read_bytes())
    sec_start = find_section(data, "Cycle")
    if sec_start < 0:
        return
    sec_end = section_end(data, sec_start)
    packed = struct.pack(">i", cycle)
    needle = struct.pack(">i", 100)
    idx = data.find(needle, sec_start, sec_end)
    if idx >= 0:
        data[idx : idx + 4] = packed
        Path(fld_path).write_bytes(data)

def _patch_f64_single(section_name: str, arr: np.ndarray, data: bytearray) -> None:
    """Patch the first float64 payload block in a section."""
    sec_start = find_section(data, section_name)
    sec_end = section_end(data, sec_start)
    for p, bc in iter_data_blocks(bytes(data), sec_start, sec_end):
        if bc % 8 == 0 and bc >= 8 and bc // 8 == arr.size:
            packed = np.ascontiguousarray(arr, dtype=">f8").tobytes()
            data[p : p + bc] = packed
            return
    raise ValueError(f"Could not patch {section_name}: no matching f64 block")


def _patch_f64_multi(
    section_name: str,
    arrays: list[np.ndarray],
    data: bytearray,
) -> None:
    """Patch float64 payload blocks in order (skips 4-element metadata blocks)."""
    sec_start = find_section(data, section_name)
    sec_end = section_end(data, sec_start)
    idx = 0
    for p, bc in iter_data_blocks(bytes(data), sec_start, sec_end):
        if bc % 8 != 0 or bc < 8:
            continue
        n = bc // 8
        if n == 4:
            continue
        if idx >= len(arrays):
            break
        if arrays[idx].size != n:
            raise ValueError(
                f"{section_name} block {idx}: size {n} != expected {arrays[idx].size}"
            )
        packed = np.ascontiguousarray(arrays[idx], dtype=">f8").tobytes()
        data[p : p + bc] = packed
        idx += 1
    if idx != len(arrays):
        raise ValueError(f"{section_name}: patched {idx} blocks, expected {len(arrays)}")


def _patch_sfile(s_text: str, data: bytearray) -> None:
    """Embed SDAT text in LS_SFile (block 1)."""
    sec_start = find_section(data, "LS_SFile")
    sec_end = section_end(data, sec_start)
    blocks = list(iter_data_blocks(bytes(data), sec_start, sec_end))
    if len(blocks) < 2:
        raise ValueError("LS_SFile needs at least 2 blocks")
    p, bc = blocks[1]
    # Normalize to LF without BOM for vendor compatibility.
    normalized = s_text.replace("\r\n", "\n").lstrip("\ufeff")
    if not normalized.startswith("SDAT"):
        normalized = "SDAT\n" + normalized
    payload = normalized.encode("utf-8")
    if len(payload) > bc:
        raise ValueError(
            f"SDAT text ({len(payload)} B) exceeds LS_SFile slot ({bc} B)"
        )
    data[p : p + len(payload)] = payload
    if len(payload) < bc:
        data[p + len(payload) : p + bc] = b"\x00" * (bc - len(payload))


def compose_fld(
    template_path: str,
    out_path: str,
    fields: dict[str, np.ndarray],
    s_text: str,
) -> None:
    """
    Write FLD by copying *template_path* and patching field sections.

    *fields* keys: PRES, TEMP, TURK, TEPS, CN01, SURT, HTFX, HTRC,
    VECTX, VECTY, VECTZ, HVECX, HVECY, HVECZ, ATMS (subset allowed).
    """
    data = bytearray(Path(template_path).read_bytes())

    n = None
    if "TEMP" in fields:
        n = fields["TEMP"].size
    elif "PRES" in fields:
        n = fields["PRES"].size

    if "PRES" in fields:
        _patch_f64_single("Pressure", fields["PRES"], data)

    if "TEMP" in fields:
        temp_arrays = [fields["TEMP"]]
        if "TURK" in fields:
            temp_arrays.append(fields["TURK"])
        if "TEPS" in fields:
            temp_arrays.append(fields["TEPS"])
        _patch_f64_multi("Temperature", temp_arrays, data)

    if "CN01" in fields:
        cn_arrays = [fields["CN01"]]
        if "HTRC" in fields:
            cn_arrays.append(fields["HTRC"])
        if "SURT" in fields:
            cn_arrays.append(fields["SURT"])
        if "HTFX" in fields:
            cn_arrays.append(fields["HTFX"])
        _patch_f64_multi("CN01", cn_arrays, data)

    if any(k in fields for k in ("VECTX", "VECTY", "VECTZ")):
        vx = fields.get("VECTX", np.zeros(n or 0))
        vy = fields.get("VECTY", np.zeros_like(vx))
        vz = fields.get("VECTZ", np.zeros_like(vx))
        _patch_f64_multi("VECT", [vx, vy, vz], data)

    if any(k in fields for k in ("HVECX", "HVECY", "HVECZ")):
        hx = fields.get("HVECX", np.zeros(n or 0))
        hy = fields.get("HVECY", np.zeros_like(hx))
        hz = fields.get("HVECZ", np.zeros_like(hx))
        _patch_f64_multi("HVEC", [hx, hy, hz], data)

    _patch_sfile(s_text, data)
    Path(out_path).write_bytes(data)


def _write_i32_section(name: str, arrays: list[np.ndarray]) -> bytes:
    inner = b""
    for arr in arrays:
        payload = np.ascontiguousarray(arr, dtype=">i4").tobytes()
        inner += _write_section_block(payload)
    return _write_named_section(name, inner)


def _write_f64_axes_section(
    name: str,
    axes: list[np.ndarray],
    n_vertices: int,
) -> bytes:
    inner = _mesh_nodes_preamble(n_vertices)
    for i, arr in enumerate(axes):
        payload = np.ascontiguousarray(arr, dtype=">f8").tobytes()
        inner += _write_section_block(payload)
        if i < len(axes) - 1:
            inner += _preamble_chunk3(n_vertices)
        else:
            inner += _field_f64_pad() + _section_suffix()
    return _write_named_section(name, inner)


def _fld_prefix() -> bytes:
    """CRDL-FLD file prefix required by scFLOW / scPOST (marker before sections)."""
    return struct.pack(">i", 8) + b"CRDL-FLD" + struct.pack(">i", 8) + struct.pack(">iii", 4, 4, 4)


def _section_bytes(data: bytes, name: str) -> bytes:
    start = find_section(data, name)
    if start < 0:
        return b""
    return bytes(data[start:section_end(data, start)])


def _vendor_header_bytes(template_path: Optional[str]) -> bytes:
    """Copy vendor header sections through LS_CoordinateSystem from a reference FLD."""
    if not template_path or not Path(template_path).is_file():
        return _build_minimal_header()
    data = Path(template_path).read_bytes()
    parts = [
        _section_bytes(data, "FileRevision"),
        _section_bytes(data, "Application"),
        _section_bytes(data, "ApplicationVersion"),
        _section_bytes(data, "ReleaseDate"),
        _section_bytes(data, "GridType"),
        _section_bytes(data, "Dimension"),
        _section_bytes(data, "Bias"),
        _section_bytes(data, "Date"),
        _section_bytes(data, "Comments"),
        _section_bytes(data, "Cycle"),
        _section_bytes(data, "Unused"),
        _section_bytes(data, "Encoding"),
        _section_bytes(data, "HeaderDataEnd"),
        _section_bytes(data, "OverlapStart_0"),
        _section_bytes(data, "LS_CoordinateSystem"),
    ]
    return b"".join(p for p in parts if p)


def _build_minimal_header() -> bytes:
    """Fallback header when no template FLD is available."""
    body = _write_named_section("FileRevision", _write_section_block(struct.pack(">i", 1)))
    app = b"scFLOW FLD Writer".ljust(64, b"\x00")
    body += _write_named_section("Application", _write_section_block(app))
    body += _write_named_section("OverlapStart_0", _write_section_block(struct.pack(">i", 0)))
    body += _write_named_section("LS_CoordinateSystem", _write_section_block(struct.pack(">i", 1)))
    return body


def _write_volume_geometry_array(
    volume_names: list[str],
    n_cells: int,
    template_path: Optional[str] = None,
) -> bytes:
    """Write LS_VolumeGeometryArray (names + per-cell vendor block)."""
    block0 = b"\x00" * 16384
    block1 = b"\x00" * 256
    cell_block: Optional[bytes] = None
    preamble = _mesh_geometry_preamble(64, len(block0))
    if template_path and Path(template_path).is_file():
        data = Path(template_path).read_bytes()
        sec = find_section(data, "LS_VolumeGeometryArray")
        if sec >= 0:
            blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
            if blocks:
                preamble = _geometry_preamble_from_template(
                    data, "LS_VolumeGeometryArray", 64, blocks[0][1],
                )
                p0, bc0 = blocks[0]
                block0 = bytes(data[p0:p0 + bc0])
            if len(blocks) > 1:
                p1, bc1 = blocks[1]
                block1 = bytes(data[p1:p1 + bc1])
            if len(blocks) > 2:
                p2, bc2 = blocks[2]
                if bc2 == n_cells * 8:
                    cell_block = bytes(data[p2:p2 + bc2])
    inner = preamble
    inner += _write_section_block(block0)
    inner += _mesh_geom_sep(64)
    inner += _write_section_block(block1)
    conn_int_count = n_cells * 8
    inner += _mesh_geom_sep(conn_int_count * 4)
    if cell_block is None:
        ids = np.arange(1, 2 * n_cells + 1, dtype=">i4")
        cell_block = ids.tobytes()
    inner += _write_section_block(cell_block)
    inner += _mesh_section_tail()
    return _write_named_section("LS_VolumeGeometryArray", inner)


def _write_surface_geometry_array(
    surface_cats: dict,
    ymax_name: str = "Ymax",
    template_path: Optional[str] = None,
) -> bytes:
    """Write LS_SurfaceGeometryArray for scPOST."""
    seg1_order = [
        "@UNDEFINEDENTB", "@UNDEFINEDENTF", "@UNDEFINEDENTS", "@UNDEFINEDENTX",
        "@UNDEFINEDMOM", "@UNDEFINEDVFWL", "PARTS", "SURFACE",
        "Xmax", "Xmin", ymax_name, "Ymin", "Zmax", "Zmin",
    ]
    seg1: list[tuple[tuple[int, int, int, int], int]] = []
    for key in seg1_order:
        seg1.extend(surface_cats.get(key, []))

    n_faces = len(seg1)
    meta = surface_meta_counts(surface_cats, ymax_name=ymax_name)
    arr2 = np.full(n_faces, 134, dtype=np.int32)
    arr3 = np.array([flat + 1 for _, flat in seg1], dtype=np.int32)
    arr5 = np.array([list(q) for q, _ in seg1], dtype=np.int32).reshape(-1)

    block0 = b"\x00" * 4608
    preamble = _mesh_geometry_preamble(len(meta), len(block0))
    block4 = b"\x00" * 72
    tpl_data: Optional[bytes] = None
    if template_path and Path(template_path).is_file():
        tpl_data = Path(template_path).read_bytes()
        sec = find_section(tpl_data, "LS_SurfaceGeometryArray")
        if sec >= 0:
            blocks = list(iter_data_blocks(tpl_data, sec, section_end(tpl_data, sec)))
            if blocks:
                preamble = _geometry_preamble_from_template(
                    tpl_data,
                    "LS_SurfaceGeometryArray",
                    len(meta),
                    blocks[0][1],
                )
                p0, bc0 = blocks[0]
                block0 = bytes(tpl_data[p0:p0 + bc0])
            if len(blocks) > 4:
                p4, bc4 = blocks[4]
                block4 = bytes(tpl_data[p4:p4 + bc4])

    face_bc = n_faces * 4
    meta_count = len(meta)

    inner = preamble
    inner += _write_section_block(block0)
    inner += _mesh_geom_sep(meta_count)
    inner += _write_section_block(meta.astype(">i4", copy=False).tobytes())
    inner += _mesh_geom_sep(face_bc)
    inner += _write_section_block(np.ascontiguousarray(arr2, dtype=">i4").tobytes())
    inner += _mesh_geom_sep(face_bc)
    inner += _write_section_block(np.ascontiguousarray(arr3, dtype=">i4").tobytes())
    inner += _mesh_geom_sep(meta_count)
    inner += _write_section_block(block4)
    inner += _mesh_geom_sep(face_bc)
    inner += _write_section_block(np.ascontiguousarray(arr5, dtype=">i4").tobytes())

    if tpl_data is not None:
        sec = find_section(tpl_data, "LS_SurfaceGeometryArray")
        if sec >= 0:
            tpl_blocks = list(iter_data_blocks(tpl_data, sec, section_end(tpl_data, sec)))
            if len(tpl_blocks) >= 6:
                link_start = tpl_blocks[5][0] + tpl_blocks[5][1] + 4
                link_end = section_end(tpl_data, sec) - 20
                inner += bytes(tpl_data[link_start:link_end])

    inner += _mesh_section_tail()
    return _write_named_section("LS_SurfaceGeometryArray", inner)


def write_fld_from_mesh(
    out_path: str,
    vertices: np.ndarray,
    cell_conn: np.ndarray,
    material: np.ndarray,
    fields: dict[str, np.ndarray],
    s_text: str,
    volume_names: list[str],
    cycle: int = 0,
    surface_cats: Optional[dict] = None,
    template_fld: Optional[str] = None,
    s_path: Optional[str] = None,
    mesh_file: Optional[str] = None,
) -> None:
    """Write a complete FLD from mesh arrays."""
    n_verts = vertices.shape[0]
    n_cells = cell_conn.shape[0]

    if template_fld is None:
        template_fld = resolve_template_fld(
            n_cells,
            s_path=s_path,
            s_basename=_sdat_basename(s_text),
            mesh_file=mesh_file,
        )

    body = _vendor_header_bytes(template_fld)

    if "PRES" in fields:
        inner = _field_section_preamble(n_verts)
        inner += _write_f64_scalar(fields["PRES"])
        inner += _write_meta_label("LS_Scalar:TEMP")
        inner += _write_field_section_end()
        body += _write_named_section("Pressure", inner)

    if "TEMP" in fields:
        temp_inner = _field_section_preamble(n_verts)
        temp_inner += _write_f64_scalar(fields["TEMP"])
        temp_inner += _write_meta_label("LS_Scalar:TURK") + _field_section_trailer()
        temp_inner += _write_meta_label("Turbulence K") + _field_section_preamble(n_verts)
        if "TURK" in fields:
            temp_inner += _write_f64_scalar(fields["TURK"])
        temp_inner += _write_meta_label("LS_Scalar:TEPS") + _field_section_trailer()
        temp_inner += _write_meta_label("Turbulence E") + _field_section_preamble(n_verts)
        if "TEPS" in fields:
            temp_inner += _write_f64_scalar(fields["TEPS"])
        temp_inner += _write_meta_label("LS_Scalar:CN01")
        temp_inner += _write_field_section_end()
        body += _write_named_section("Temperature", temp_inner)

    cn_arrays: list[tuple[np.ndarray, str, str]] = []
    if "CN01" in fields:
        cn_arrays.append((fields["CN01"], "LS_Scalar:HTRC", "HEAT TRANSFER COEF."))
    if "HTRC" in fields:
        cn_arrays.append((fields["HTRC"], "LS_Scalar:SURT", "WALL TEMPERATURE"))
    if "SURT" in fields:
        cn_arrays.append((fields["SURT"], "LS_Scalar:HTFX", "WALL HEAT FLUX"))
    if "HTFX" in fields:
        cn_arrays.append((fields["HTFX"], "LS_Vector:VECT", ""))
    if cn_arrays:
        cn_inner = _field_section_preamble(n_verts)
        for idx, (arr, ls_label, desc_label) in enumerate(cn_arrays):
            cn_inner += _write_f64_scalar(arr)
            if desc_label:
                cn_inner += _write_meta_label(ls_label) + _field_section_trailer()
                cn_inner += _write_meta_label(desc_label) + _field_section_preamble(n_verts)
            else:
                cn_inner += _write_meta_label(ls_label) + _field_section_trailer_link()
                cn_inner += _write_linked_section_end()
        body += _write_named_section("CN01", cn_inner)

    vx = fields.get("VECTX", np.zeros(n_verts))
    vy = fields.get("VECTY", np.zeros(n_verts))
    vz = fields.get("VECTZ", np.zeros(n_verts))
    vect_inner = _field_section_preamble(n_verts)
    vect_inner += _write_f64_vector(vx, n_verts)
    vect_inner += _write_f64_vector(vy, n_verts)
    vect_inner += _write_f64_vector(vz, n_verts)
    vect_inner += _write_meta_label("LS_Vector:HVEC") + _field_section_trailer_link()
    vect_inner += _write_linked_section_end()
    body += _write_named_section("VECT", vect_inner)

    hx = fields.get("HVECX", np.zeros(n_verts))
    hy = fields.get("HVECY", np.zeros(n_verts))
    hz = fields.get("HVECZ", np.zeros(n_verts))
    hvec_inner = _field_section_preamble(n_verts)
    hvec_inner += _write_f64_vector(hx, n_verts)
    hvec_inner += _write_f64_vector(hy, n_verts)
    hvec_inner += _write_f64_vector(hz, n_verts)
    hvec_inner += _section_suffix()
    body += _write_named_section("HVEC", hvec_inner)

    if template_fld and Path(template_fld).is_file():
        tpl = Path(template_fld).read_bytes()
        body += _section_bytes(tpl, "LS_STREAMcoc")
        body += _section_bytes(tpl, "LS_STREAMmultiblock")

    body += _write_f64_axes_section(
        "LS_Nodes",
        [vertices[:, 0], vertices[:, 1], vertices[:, 2]],
        n_verts,
    )
    mat_inner = _mesh_cells_preamble(n_cells)
    mat_inner += _write_section_block(
        np.ascontiguousarray(material, dtype=">i4").tobytes()
    )
    mat_inner += _mesh_section_tail()
    body += _write_named_section("LS_MatOfElements", mat_inner)

    elem_meta = np.full(n_cells, 38, dtype=np.int32)
    conn_flat = cell_conn.astype(np.int32).reshape(-1)
    elem_inner = _mesh_cells_preamble(n_cells)
    elem_inner += _write_section_block(np.ascontiguousarray(elem_meta, dtype=">i4").tobytes())
    elem_inner += _mesh_i32_block_sep1()
    elem_inner += _write_section_block(struct.pack(">i", conn_flat.size))
    elem_inner += _mesh_i32_block_sep2(conn_flat.size)
    elem_inner += _write_section_block(
        np.ascontiguousarray(conn_flat, dtype=">i4").tobytes()
    )
    elem_inner += _mesh_section_tail()
    body += _write_named_section("LS_Elements", elem_inner)

    labels = " ".join(volume_names).encode("ascii")
    body += _write_volume_geometry_array(volume_names, n_cells, template_fld)

    if surface_cats:
        body += _write_surface_geometry_array(surface_cats, template_path=template_fld)

    normalized = s_text.replace("\r\n", "\n").lstrip("\ufeff")
    if not normalized.startswith("SDAT"):
        normalized = "SDAT\n" + normalized

    if template_fld and Path(template_fld).is_file():
        tpl = bytearray(Path(template_fld).read_bytes())
        _patch_sfile(normalized, tpl)
        body += _section_bytes(tpl, "LS_SFile")
    else:
        sfile_payload = normalized.encode("utf-8")
        sfile_inner = _write_section_block(struct.pack(">d", 1.0))
        slot = max(len(sfile_payload) + 64, 6144)
        sfile_inner += _write_section_block(
            sfile_payload + b"\x00" * (slot - len(sfile_payload))
        )
        body += _write_named_section("LS_SFile", sfile_inner)

    body += _write_named_section("OverlapEnd", b"")

    out = _fld_prefix() + body
    Path(out_path).write_bytes(out)


def default_initial_fields(
    n_vertices: int,
    temp: np.ndarray,
    ambient: float = 20.0,
) -> dict[str, np.ndarray]:
    """Build a minimal initial field set matching vendor FLD layout."""
    pres = np.full(n_vertices, ambient, dtype=np.float64)
    atms = temp.copy()
    cn01 = np.zeros(n_vertices, dtype=np.float64)
    turk = np.full(n_vertices, 0.68931361, dtype=np.float64)
    teps = np.full(n_vertices, 646.51688989, dtype=np.float64)
    sent = np.full(n_vertices, 1.0e20, dtype=np.float64)
    zeros = np.zeros(n_vertices, dtype=np.float64)
    return {
        "PRES": pres,
        "TEMP": temp,
        "ATMS": atms,
        "TURK": turk,
        "TEPS": teps,
        "CN01": cn01,
        "HTRC": sent.copy(),
        "SURT": sent.copy(),
        "HTFX": sent.copy(),
        "VECTX": zeros.copy(),
        "VECTY": zeros.copy(),
        "VECTZ": zeros.copy(),
        "HVECX": zeros.copy(),
        "HVECY": zeros.copy(),
        "HVECZ": zeros.copy(),
    }
