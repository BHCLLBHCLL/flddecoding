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
    find_section,
    iter_data_blocks,
    read_i32_be,
    section_end,
)


def _write_section_block(payload: bytes) -> bytes:
    """One data block: [12, bc][payload][bc]."""
    bc = len(payload)
    return struct.pack(">ii", 12, bc) + payload + struct.pack(">i", bc)


def _write_named_section(name: str, inner: bytes) -> bytes:
    """Section with [32][name 32B][32] prefix."""
    name_padded = name.ljust(32).encode("ascii")
    return struct.pack(">i", 32) + name_padded + struct.pack(">i", 32) + inner


def _build_f64_section(section_name: str, arrays: list[np.ndarray]) -> bytes:
    """Build a field section with alternating f64 payloads and 4-float metadata blocks."""
    inner = b""
    for arr in arrays:
        payload = np.ascontiguousarray(arr, dtype=">f8").tobytes()
        inner += _write_section_block(payload)
        if arr.size > 4:
            # Vendor interleaves 4-element metadata blocks between large arrays.
            meta = np.zeros(4, dtype=">f8").tobytes()
            inner += _write_section_block(meta)
    return _write_named_section(section_name, inner)


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


def _write_f64_axes_section(name: str, axes: list[np.ndarray]) -> bytes:
    inner = b""
    for arr in axes:
        payload = np.ascontiguousarray(arr, dtype=">f8").tobytes()
        inner += _write_section_block(payload)
    return _write_named_section(name, inner)


def _build_header() -> bytes:
    """Minimal CRDL-FLD header (cycle 0)."""
    parts = []
    parts.append(_write_named_section("FileRevision", _write_section_block(struct.pack(">i", 1))))
    app = b"scFLOW FLD Writer".ljust(64, b"\x00")
    parts.append(_write_named_section("Application", _write_section_block(app)))
    parts.append(_write_named_section("LS_CoordinateSystem", _write_section_block(struct.pack(">i", 1))))
    return b"".join(parts)


def write_fld_from_mesh(
    out_path: str,
    vertices: np.ndarray,
    cell_conn: np.ndarray,
    material: np.ndarray,
    fields: dict[str, np.ndarray],
    s_text: str,
    volume_names: list[str],
    cycle: int = 0,
) -> None:
    """Write a complete FLD from mesh arrays (no template)."""
    n_verts = vertices.shape[0]
    n_cells = cell_conn.shape[0]

    body = _build_header()

    if "PRES" in fields:
        inner = _write_section_block(np.ascontiguousarray(fields["PRES"], dtype=">f8").tobytes())
        inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        body += _write_named_section("Pressure", inner)

    if "TEMP" in fields:
        temp_inner = _write_section_block(np.ascontiguousarray(fields["TEMP"], dtype=">f8").tobytes())
        temp_inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        temp_inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        if "TURK" in fields:
            temp_inner += _write_section_block(np.ascontiguousarray(fields["TURK"], dtype=">f8").tobytes())
        temp_inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        temp_inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        if "TEPS" in fields:
            temp_inner += _write_section_block(np.ascontiguousarray(fields["TEPS"], dtype=">f8").tobytes())
        temp_inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        body += _write_named_section("Temperature", temp_inner)

    cn_arrays = []
    if "CN01" in fields:
        cn_arrays.append(fields["CN01"])
    if "HTRC" in fields:
        cn_arrays.append(fields["HTRC"])
    if "SURT" in fields:
        cn_arrays.append(fields["SURT"])
    if "HTFX" in fields:
        cn_arrays.append(fields["HTFX"])
    if cn_arrays:
        cn_inner = b""
        for arr in cn_arrays:
            cn_inner += _write_section_block(np.ascontiguousarray(arr, dtype=">f8").tobytes())
            if arr.size > 4:
                cn_inner += _write_section_block(np.zeros(4, dtype=">f8").tobytes())
        body += _write_named_section("CN01", cn_inner)

    vx = fields.get("VECTX", np.zeros(n_verts))
    vy = fields.get("VECTY", np.zeros(n_verts))
    vz = fields.get("VECTZ", np.zeros(n_verts))
    vect_inner = (
        _write_section_block(np.ascontiguousarray(vx, dtype=">f8").tobytes())
        + _write_section_block(np.ascontiguousarray(vy, dtype=">f8").tobytes())
        + _write_section_block(np.ascontiguousarray(vz, dtype=">f8").tobytes())
        + _write_section_block(np.zeros(4, dtype=">f8").tobytes())
    )
    body += _write_named_section("VECT", vect_inner)

    hx = fields.get("HVECX", np.zeros(n_verts))
    hy = fields.get("HVECY", np.zeros(n_verts))
    hz = fields.get("HVECZ", np.zeros(n_verts))
    hvec_inner = (
        _write_section_block(np.ascontiguousarray(hx, dtype=">f8").tobytes())
        + _write_section_block(np.ascontiguousarray(hy, dtype=">f8").tobytes())
        + _write_section_block(np.ascontiguousarray(hz, dtype=">f8").tobytes())
    )
    body += _write_named_section("HVEC", hvec_inner)

    body += _write_f64_axes_section(
        "LS_Nodes",
        [vertices[:, 0], vertices[:, 1], vertices[:, 2]],
    )
    body += _write_i32_section("LS_MatOfElements", [material.astype(np.int32)])
    elem_meta = np.full(n_cells, 38, dtype=np.int32)
    body += _write_i32_section(
        "LS_Elements",
        [elem_meta, cell_conn.astype(np.int32).reshape(-1)],
    )

    labels = " ".join(volume_names).encode("ascii")
    vol_inner = _write_section_block(labels.ljust(1024, b"\x00")[:1024])
    body += _write_named_section("LS_VolumeGeometryArray", vol_inner)

    normalized = s_text.replace("\r\n", "\n").lstrip("\ufeff")
    if not normalized.startswith("SDAT"):
        normalized = "SDAT\n" + normalized
    sfile_payload = normalized.encode("utf-8")
    sfile_inner = _write_section_block(struct.pack(">d", 1.0))
    slot = max(len(sfile_payload) + 64, 6144)
    sfile_inner += _write_section_block(sfile_payload + b"\x00" * (slot - len(sfile_payload)))
    body += _write_named_section("LS_SFile", sfile_inner)

    body += _write_named_section("OverlapEnd", _write_section_block(struct.pack(">i", 0)))

    magic = b"CRDL-FLD"
    out = magic + body
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
