#!/usr/bin/env python3
"""
Write FLD (CRDL-FLD) binary files.

Mesh / surface sections are taken from a template FLD (or source mesh FLD).
Field arrays and embedded SDAT are updated from parsed .s settings.

scPOST-critical paths include LS_SFile preamble, Volume/Surface geometry preambles
(offset 104), ex3 10-slot volume names / 5-bucket block1, vol-flag block2 generation,
and material id 0→1 on export. See docs/DEVELOPMENT_SUMMARY.md §4.4.
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
    _parse_volume_names,
)
from surface_builder import (
    surface_meta_counts,
    surface_meta_counts_ex3,
    surface_seg1_order_ex3,
)


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


_GENERATED_FLD_MARKERS = ("_from_sxemt", "_from_s", "_0_from_")


def _is_generated_fld(path: Path) -> bool:
    """Skip converter output files when searching for vendor reference FLDs."""
    return any(marker in path.stem for marker in _GENERATED_FLD_MARKERS)


def _template_fld_candidates(
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
) -> list[Path]:
    """Build ordered candidate paths for reference FLD lookup."""
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
            for suffix in ("_151", "_63", "_100", "_0", ""):
                c = parent / f"{stem}{suffix}.fld"
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)

    official_pool: list[Path] = []
    other_pool: list[Path] = []
    for c in sorted(tests_dir.glob("*.fld")):
        if c in seen or _is_generated_fld(c):
            continue
        seen.add(c)
        if c.stem.endswith("_63") or c.stem.endswith("_100") or c.stem.endswith("_151"):
            official_pool.append(c)
        else:
            other_pool.append(c)
    candidates.extend(official_pool)
    candidates.extend(other_pool)
    return candidates


def _has_vendor_header(path: str) -> bool:
    """True when FLD has full scPOST file header (not minimal writer fallback)."""
    data = Path(path).read_bytes()
    return find_section(data, "ApplicationVersion") >= 0


def _has_vendor_sfile(path: str) -> bool:
    """True when LS_SFile has the 48-byte vendor preamble before block0."""
    data = Path(path).read_bytes()
    sec = find_section(data, "LS_SFile")
    if sec < 0:
        return False
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    if len(blocks) < 2:
        return False
    pre_len = blocks[0][0] - (sec + 40) - 8
    return pre_len >= 48


def _sfile_text_slot(path: str) -> Optional[int]:
    """Byte capacity of LS_SFile SDAT text block (block1), or None."""
    data = Path(path).read_bytes()
    sec = find_section(data, "LS_SFile")
    if sec < 0:
        return None
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    if len(blocks) < 2:
        return None
    return blocks[1][1]


def _prefer_stems(
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
) -> list[str]:
    stems: list[str] = []
    if s_basename:
        stems.append(s_basename)
    if s_path:
        stems.append(Path(s_path).stem)
    if mesh_file:
        stems.append(Path(mesh_file).stem)
    return stems


def _pick_reference_fld(
    candidates: list[Path],
    *,
    require_header: bool = False,
    require_sfile: bool = False,
    min_sfile_slot: int = 0,
    n_cells: Optional[int] = None,
    prefer_stems: Optional[list[str]] = None,
) -> Optional[str]:
    """Pick best vendor reference; prefers same-case stem (e.g. ex3_e_151 for ex3_e.s)."""
    pool: list[Path] = []
    for c in candidates:
        if not c.is_file() or _is_generated_fld(c):
            continue
        p = str(c)
        if require_header and not _has_vendor_header(p):
            continue
        if require_sfile and not _has_vendor_sfile(p):
            continue
        if min_sfile_slot:
            slot = _sfile_text_slot(p)
            if slot is None or slot < min_sfile_slot:
                continue
        if n_cells is not None:
            nc = fld_cell_count_file(p)
            if nc != n_cells:
                continue
        pool.append(c)
    if not pool:
        return None

    def score(path: Path) -> tuple[int, int, int]:
        stem = path.stem
        stem_match = 0
        if prefer_stems:
            for s in prefer_stems:
                if stem == s or stem.startswith(s + "_"):
                    stem_match = 2
                    break
        official = 1 if any(
            stem.endswith(s) for s in ("_151", "_63", "_100")
        ) else 0
        return (stem_match, official, path.stat().st_size)

    return str(max(pool, key=score))


def resolve_header_template_fld(
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """
    Reference FLD for file header sections (independent of cell count).

    scPOST rejects the 312-byte minimal header; vendor headers are ~1.8 KiB through
    LS_CoordinateSystem and include ApplicationVersion, Encoding, etc.
    """
    if explicit:
        exp = Path(explicit)
        if not exp.is_file():
            raise FileNotFoundError(f"Template FLD not found: {exp}")
        return str(exp)

    return _pick_reference_fld(
        _template_fld_candidates(s_path, s_basename, mesh_file),
        require_header=True,
        prefer_stems=_prefer_stems(s_path, s_basename, mesh_file),
    )


def resolve_sfile_template_fld(
    s_text: str,
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Reference FLD with vendor LS_SFile layout and large enough SDAT text slot."""
    min_slot = len(s_text.encode("utf-8"))
    if explicit:
        exp = Path(explicit)
        if not exp.is_file():
            raise FileNotFoundError(f"Template FLD not found: {exp}")
        if not _has_vendor_sfile(str(exp)):
            raise ValueError(f"Template {exp} has no vendor LS_SFile section")
        slot = _sfile_text_slot(str(exp))
        if slot is not None and slot < min_slot:
            raise ValueError(
                f"Template {exp} LS_SFile slot ({slot} B) < SDAT text ({min_slot} B)"
            )
        return str(exp)

    return _pick_reference_fld(
        _template_fld_candidates(s_path, s_basename, mesh_file),
        require_header=True,
        require_sfile=True,
        min_sfile_slot=min_slot,
        prefer_stems=_prefer_stems(s_path, s_basename, mesh_file),
    )


def resolve_geometry_layout_fld(
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """
    Reference FLD for volume/surface geometry preamble and block0/block1 layout.

    Unlike ``resolve_template_fld``, does not require matching cell count (needed for
    block1 metadata and block2 vol-flag templates when counts differ).
    """
    if explicit and Path(explicit).is_file():
        return str(Path(explicit))

    return _pick_reference_fld(
        _template_fld_candidates(s_path, s_basename, mesh_file),
        prefer_stems=_prefer_stems(s_path, s_basename, mesh_file),
    )


def resolve_template_fld(
    n_cells: int,
    s_path: Optional[str] = None,
    s_basename: Optional[str] = None,
    mesh_file: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """
    Find a reference FLD whose mesh matches *n_cells* (geometry / SFile slots).

    Prefers *explicit* when provided and cell count matches. Otherwise searches
    candidates for the first file whose cell count equals *n_cells*.
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

    return _pick_reference_fld(
        _template_fld_candidates(s_path, s_basename, mesh_file),
        n_cells=n_cells,
        prefer_stems=_prefer_stems(s_path, s_basename, mesh_file),
    )


def _write_sfile_section(s_text: str, template_path: Optional[str] = None) -> bytes:
    """Write LS_SFile by copying vendor section layout and embedding SDAT text."""
    normalized = s_text.replace("\r\n", "\n").lstrip("\ufeff")
    if not normalized.startswith("SDAT"):
        normalized = "SDAT\n" + normalized
    if template_path and Path(template_path).is_file():
        tpl = bytearray(Path(template_path).read_bytes())
        _patch_sfile(normalized, tpl)
        return _section_bytes(tpl, "LS_SFile")
    payload = normalized.encode("utf-8")
    slot = max(len(payload) + 64, 6144)
    inner = (
        struct.pack(">iiii", 12, 4, 1, 1)
        + struct.pack(">iiii", 12, 4, 1, 4)
        + struct.pack(">iiii", 12, 8, 1, 1)
        + _write_section_block(struct.pack(">d", 1.0))
        + struct.pack(">iiii", 12, 1, slot, 1)
        + _write_section_block(payload + b"\x00" * (slot - len(payload)))
        + _field_f64_pad()
        + _section_suffix()
    )
    return _write_named_section("LS_SFile", inner)


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


_VOLUME_NAME_SLOT = 256


def _build_volume_block0(volume_names: list[str], block0_size: int) -> bytes:
    """Vendor block0: one 256-byte ASCII slot per volume name."""
    slots = b"".join(
        name.encode("ascii")[: _VOLUME_NAME_SLOT].ljust(_VOLUME_NAME_SLOT, b"\x00")
        for name in volume_names
    )
    if len(slots) < block0_size:
        slots += b"\x00" * (block0_size - len(slots))
    return slots[:block0_size]


def _volume_block1_half_counts(
    material: np.ndarray,
    half: int,
    cell_part: Optional[np.ndarray] = None,
    *,
    ex3_buckets: bool = False,
) -> list[int]:
    """
    Build per-volume count metadata (first half of block1).

    Vendor ex3: five buckets ``[main, p2, p3, p5, p6]`` duplicated in block1.
    """
    mat = material.astype(np.int64, copy=False)
    n_cells = int(mat.size)
    m1 = int(np.sum(mat == 1))
    if cell_part is not None and half >= 5:
        cp = cell_part.astype(np.int64, copy=False)
        part_ids = sorted(int(p) for p in set(cp.tolist()) if p > 0)
        subs = [int(np.sum(cp == pid)) for pid in part_ids if pid != 1]
        if ex3_buckets and len(subs) >= 5:
            # Vendor ex3: bucket0 holds the main block; tail is parts 2,3,5,6
            # (part 4 stays in bucket0 — not a separate block1 tail slot).
            tail = [subs[0], subs[1], subs[3], subs[4]]
            m1 = n_cells - sum(tail)
            core = [m1] + tail
        else:
            core = [m1] + subs
    else:
        m2 = int(np.sum(mat == 2))
        m3 = int(np.sum(mat == 3))
        m0 = int(np.sum(mat == 0))
        if half >= 5 and m3 > 0:
            subs = [m0, m2, m3] if m0 else [m2, m3]
            while len(subs) < half - 1:
                subs.append(0)
            core = [m1] + subs[: half - 1]
        else:
            core = [m1, int(np.sum(mat == 0)), m2, m3]
    counts = core[:half]
    while len(counts) < half:
        counts.append(0)
    counts = counts[:half]
    delta = n_cells - sum(counts)
    if delta and counts:
        counts[0] += delta
    return counts


def _volume_export_layout(
    volume_names: list[str],
    template_path: Optional[str],
) -> tuple[int, list[str], int, bool]:
    """
    Return (count_val, names, block0_size, ex3_buckets) for LS_VolumeGeometryArray.

    ex3 vendor files use 10 fixed 256-byte name slots and five bucket counts in block1.
    """
    ex3 = _volume_geometry_ex3_style(template_path)
    if ex3 and template_path and Path(template_path).is_file():
        tpl_names = _parse_volume_names(Path(template_path).read_bytes())
        if tpl_names:
            return len(tpl_names), tpl_names, len(tpl_names) * _VOLUME_NAME_SLOT, True
        return 10, volume_names[:10], 2560, True
    count_val = max(len(volume_names), 1)
    block0_size = max(count_val * _VOLUME_NAME_SLOT, 2560)
    return count_val, volume_names, block0_size, False


def _load_template_vol_flag_buckets(
    template_path: str,
) -> Optional[tuple[list[int], list[np.ndarray]]]:
    """Return (half_counts, pair buckets) from a vendor volume-geometry section."""
    data = Path(template_path).read_bytes()
    sec = find_section(data, "LS_VolumeGeometryArray")
    if sec < 0:
        return None
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    if len(blocks) < 3:
        return None
    p1, bc1 = blocks[1]
    p2, bc2 = blocks[2]
    half = bc1 // 8
    counts = [read_i32_be(data, p1 + i * 4) for i in range(half)]
    pairs = np.frombuffer(
        data[p2:p2 + bc2], dtype=">i4",
    ).reshape(-1, 2).astype(np.int64)
    buckets: list[np.ndarray] = []
    pos = 0
    for count in counts:
        buckets.append(pairs[pos:pos + count].copy())
        pos += count
    return counts, buckets


def _bucket_vol_flag_endpoints(
    bucket_idx: int,
    count: int,
    prev_hi: int,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Return (start_pair, end_pair, new_prev_hi) for one volume bucket."""
    if bucket_idx == 0:
        start = (3, 4)
        end = (count - 1, count)
        return start, end, count
    start = (prev_hi + 1, prev_hi + 2)
    end = (prev_hi + 2 * count - 1, prev_hi + 2 * count)
    return start, end, prev_hi + 2 * count


def _resample_vol_flag_bucket(
    tpl_pairs: Optional[np.ndarray],
    out_n: int,
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    """
    Resample one bucket of vol-flag pairs, preserving vendor wrap deltas.

    Walks the template lo-delta pattern at proportional indices while pinning
    the first and last pairs to the bucket boundary values.
    """
    if out_n <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    if out_n == 1:
        return np.array([end], dtype=np.int64)

    out = np.zeros((out_n, 2), dtype=np.int64)
    out[0] = start
    out[-1] = end
    if out_n == 2:
        return out

    if tpl_pairs is None or len(tpl_pairs) < 2:
        for j in range(1, out_n - 1):
            lo = start[0] + (end[0] - start[0]) * j // (out_n - 1)
            out[j] = (lo, lo + 1)
        return out

    tpl = np.asarray(tpl_pairs, dtype=np.int64).reshape(-1, 2)
    tpl_lo = tpl[:, 0]
    tpl_hi_off = tpl[:, 1] - tpl[:, 0]
    t_max = len(tpl) - 1
    for j in range(1, out_n - 1):
        t = int(j * t_max / (out_n - 1))
        t = min(t, t_max - 1)
        delta = int(tpl_lo[t + 1] - tpl_lo[t])
        hi_off = int(tpl_hi_off[t + 1])
        if delta < -100:
            lo = int(tpl_lo[t + 1])
        else:
            lo = int(out[j - 1, 0]) + delta
        out[j, 0] = lo
        out[j, 1] = lo + hi_off
    return out


def _linear_vol_flag_bucket(
    count: int,
    start_lo: int,
    start_hi: int,
) -> np.ndarray:
    """Simple ``+2`` lo/hi pairs for one bucket."""
    out = np.zeros((count, 2), dtype=np.int64)
    lo, hi = start_lo, start_hi
    for i in range(count):
        out[i] = (lo, hi)
        lo += 2
        hi += 2
    return out


def _offset_vol_flag_bucket(tpl_pairs: np.ndarray, offset: int) -> np.ndarray:
    """Rebase a template bucket by adding *offset* to both pair values."""
    tpl = np.asarray(tpl_pairs, dtype=np.int64).reshape(-1, 2)
    return tpl + np.array([offset, offset], dtype=np.int64)


def _scale_vol_flag_bucket_from_template(
    tpl_pairs: np.ndarray,
    out_n: int,
    tpl_n_cells: int,
    out_n_cells: int,
) -> np.ndarray:
    """
    Resample bucket0 vol-flag pairs by proportional index + amplitude scale.

    Keeps vendor wrap structure while ensuring ``lo <= out_n_cells - 1``.
    """
    if out_n <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    tpl = np.asarray(tpl_pairs, dtype=np.int64).reshape(-1, 2)
    if out_n == 1:
        return np.array([[out_n - 1, out_n]], dtype=np.int64)

    out = np.zeros((out_n, 2), dtype=np.int64)
    out[0] = (3, 4)
    out[-1] = (out_n - 1, out_n)
    if out_n == 2:
        return out

    t_max = len(tpl) - 1
    lo_scale = (out_n_cells - 1) / max(tpl_n_cells - 1, 1)
    for j in range(1, out_n - 1):
        t = int(j * t_max / (out_n - 1))
        lo = int(round(tpl[t, 0] * lo_scale))
        hi = int(round(tpl[t, 1] * lo_scale))
        lo = max(1, min(out_n_cells - 1, lo))
        hi = max(lo + 1, min(out_n_cells, hi))
        out[j, 0] = lo
        out[j, 1] = hi
    return out


def _generate_vol_flag_buckets(
    bucket_counts: list[int],
    template_path: Optional[str] = None,
    *,
    n_cells: int = 0,
    tpl_n_cells: int = 0,
) -> bytes:
    """Build per-cell vol-flag pairs grouped by block1 buckets."""
    tpl_buckets: Optional[list[np.ndarray]] = None
    if template_path and Path(template_path).is_file():
        loaded = _load_template_vol_flag_buckets(template_path)
        if loaded is not None:
            tpl_buckets = loaded[1]

    chunks: list[np.ndarray] = []
    prev_hi = 2
    for bi, count in enumerate(bucket_counts):
        if count <= 0:
            continue
        tpl = tpl_buckets[bi] if tpl_buckets and bi < len(tpl_buckets) else None

        if bi == 0 and tpl is not None and n_cells > 0 and tpl_n_cells > 0:
            chunk = _scale_vol_flag_bucket_from_template(
                tpl, count, tpl_n_cells, n_cells,
            )
            chunks.append(chunk)
            prev_hi = int(chunk[-1, 1])
            continue

        if bi in (1, 2):
            start = (prev_hi + 1, prev_hi + 2)
            chunk = _linear_vol_flag_bucket(count, start[0], start[1])
            chunks.append(chunk)
            prev_hi = int(chunk[-1, 1])
            continue

        if bi == 3 and tpl is not None and len(tpl) == count:
            tpl_b2_hi = int(tpl_buckets[2][-1, 1]) if tpl_buckets else prev_hi
            offset = prev_hi - tpl_b2_hi
            chunk = _offset_vol_flag_bucket(tpl, offset)
            chunks.append(chunk)
            prev_hi = int(chunk[-1, 1])
            continue

        if bi == 4 and tpl is not None and len(tpl) == count:
            chunks.append(np.asarray(tpl, dtype=np.int64).reshape(-1, 2).copy())
            continue

        start, end, prev_hi = _bucket_vol_flag_endpoints(bi, count, prev_hi)
        chunks.append(_resample_vol_flag_bucket(tpl, count, start, end))
        prev_hi = int(chunks[-1][-1, 1])

    if not chunks:
        return b""
    return np.vstack(chunks).astype(">i4").tobytes()


def _build_volume_block2(
    half_counts: list[int],
    n_cells: int,
    template_path: Optional[str] = None,
    *,
    ex3_buckets: bool = False,
) -> bytes:
    """Per-cell vol-flag pairs; ex3 uses five vendor buckets with boundary encoding."""
    if template_path and Path(template_path).is_file():
        data = Path(template_path).read_bytes()
        sec = find_section(data, "LS_VolumeGeometryArray")
        if sec >= 0:
            blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
            if len(blocks) > 2 and blocks[2][1] == n_cells * 8:
                return bytes(data[blocks[2][0]:blocks[2][0] + blocks[2][1]])

    bucket_counts = [c for c in half_counts if c > 0]
    if ex3_buckets:
        while len(bucket_counts) < 5:
            bucket_counts.append(0)
        bucket_counts = bucket_counts[:5]
        body = _generate_vol_flag_buckets(
            bucket_counts, template_path, n_cells=n_cells,
            tpl_n_cells=fld_cell_count_file(template_path) or 0,
        )
        if body and len(body) // 8 == n_cells:
            return body
        bucket_counts = [c for c in bucket_counts if c > 0]

    flat: list[int] = []
    cur = 3
    for bucket in bucket_counts if ex3_buckets else half_counts:
        for _ in range(bucket):
            flat.extend((cur, cur + 1))
            cur += 2
    if len(flat) // 2 != n_cells:
        flat = list(range(3, 3 + 2 * n_cells))
    return np.asarray(flat, dtype=">i4").tobytes()


def _volume_geometry_ex3_style(template_path: Optional[str]) -> bool:
    """True when template uses ex3-style five-bucket volume metadata."""
    if not template_path or not Path(template_path).is_file():
        return False
    data = Path(template_path).read_bytes()
    sec = find_section(data, "LS_VolumeGeometryArray")
    if sec < 0:
        return False
    inner = sec + 40
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    if not blocks:
        return False
    pre_len = blocks[0][0] - inner - 8
    if pre_len < 60:
        return False
    count_val = read_i32_be(data[inner:inner + pre_len], 56)
    return count_val == 10


def _build_volume_block1(
    count_val: int,
    material: Optional[np.ndarray],
    template_path: Optional[str],
    cell_part: Optional[np.ndarray] = None,
    *,
    ex3_buckets: bool = False,
) -> bytes:
    """Volume block1: ``count_val`` big-endian i32 values (vendor duplicates half)."""
    half = max(count_val // 2, 1)
    if material is not None:
        counts = _volume_block1_half_counts(
            material, half, cell_part, ex3_buckets=ex3_buckets,
        )
    else:
        counts = [0] * half
    vals = counts + counts
    while len(vals) < count_val:
        vals.append(0)
    return np.asarray(vals[:count_val], dtype=">i4").tobytes()


def _write_volume_geometry_array(
    volume_names: list[str],
    n_cells: int,
    template_path: Optional[str] = None,
    material: Optional[np.ndarray] = None,
    cell_part: Optional[np.ndarray] = None,
) -> bytes:
    """Write LS_VolumeGeometryArray (names + per-cell vendor block)."""
    count_val, vol_names, block0_size, ex3 = _volume_export_layout(
        volume_names, template_path,
    )
    block0 = _build_volume_block0(vol_names, block0_size)
    block1 = _build_volume_block1(
        count_val, material, template_path, cell_part, ex3_buckets=ex3,
    )
    half = max(count_val // 2, 1)
    half_counts = (
        _volume_block1_half_counts(
            material, half, cell_part, ex3_buckets=ex3,
        )
        if material is not None
        else [0] * half
    )
    cell_block: Optional[bytes] = None
    preamble = _mesh_geometry_preamble(count_val, block0_size)
    if template_path and Path(template_path).is_file():
        data = Path(template_path).read_bytes()
        sec = find_section(data, "LS_VolumeGeometryArray")
        if sec >= 0:
            blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
            if blocks and ex3:
                tpl_block0 = blocks[0][1]
                block0_size = max(tpl_block0, block0_size)
                block0 = _build_volume_block0(vol_names, block0_size)
            if blocks:
                preamble = _geometry_preamble_from_template(
                    data, "LS_VolumeGeometryArray", count_val, block0_size,
                )
            if len(blocks) > 2:
                p2, bc2 = blocks[2]
                if bc2 == n_cells * 8:
                    cell_block = bytes(data[p2:p2 + bc2])
    inner = preamble
    inner += _write_section_block(block0)
    inner += _mesh_geom_sep(count_val)
    inner += _write_section_block(block1)
    inner += _mesh_geom_sep(n_cells * 2)
    if cell_block is None:
        cell_block = _build_volume_block2(
            half_counts, n_cells, template_path, ex3_buckets=ex3,
        )
    inner += _write_section_block(cell_block)
    inner += _mesh_section_tail()
    return _write_named_section("LS_VolumeGeometryArray", inner)


def _surface_geometry_meta_count(template_path: Optional[str]) -> int:
    """Return geometry preamble count_val from a vendor surface section (16 ex3, 18 ex4)."""
    if not template_path or not Path(template_path).is_file():
        return 18
    data = Path(template_path).read_bytes()
    sec = find_section(data, "LS_SurfaceGeometryArray")
    if sec < 0:
        return 18
    inner = sec + 40
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    if not blocks:
        return 18
    pre_len = blocks[0][0] - inner - 8
    if pre_len >= 60:
        return read_i32_be(data[inner:inner + pre_len], 56)
    return 18


def _build_surface_block4(
    n_faces: int,
    template_path: Optional[str],
    meta_count: int,
) -> bytes:
    """Build surface block4; scale vendor template values to *n_faces* when counts differ."""
    nbytes = 64 if meta_count == 16 else 72
    zeros = b"\x00" * nbytes
    if not template_path or not Path(template_path).is_file():
        return zeros
    data = Path(template_path).read_bytes()
    sec = find_section(data, "LS_SurfaceGeometryArray")
    if sec < 0:
        return zeros
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    if len(blocks) <= 4:
        return zeros
    p4, bc4 = blocks[4]
    tpl = bytes(data[p4:p4 + bc4])
    if len(tpl) != nbytes:
        tpl = tpl[:nbytes].ljust(nbytes, b"\x00")
    tpl_faces = blocks[5][1] // 16 if len(blocks) > 5 else 0
    if tpl_faces <= 0 or tpl_faces == n_faces:
        return tpl
    scale = n_faces / tpl_faces
    vals = np.frombuffer(tpl, dtype=">i4").astype(np.float64)
    scaled = np.array([int(round(v * scale)) if v else 0 for v in vals], dtype=">i4")
    return scaled.tobytes()


def _write_surface_geometry_array(
    surface_cats: dict,
    ymax_name: str = "Ymax",
    template_path: Optional[str] = None,
) -> bytes:
    """Write LS_SurfaceGeometryArray for scPOST."""
    meta_count = _surface_geometry_meta_count(template_path)
    if meta_count == 16:
        seg1_keys = surface_seg1_order_ex3(ymax_name=ymax_name)
        meta = surface_meta_counts_ex3(surface_cats, ymax_name=ymax_name)
    else:
        seg1_keys = [
            "@UNDEFINEDENTB", "@UNDEFINEDENTF", "@UNDEFINEDENTS", "@UNDEFINEDENTX",
            "@UNDEFINEDMOM", "@UNDEFINEDVFWL", "PARTS", "SURFACE",
            "Xmax", "Xmin", ymax_name, "Ymin", "Zmax", "Zmin",
        ]
        meta = surface_meta_counts(surface_cats, ymax_name=ymax_name)
    seg1: list[tuple[tuple[int, int, int, int], int]] = []
    for key in seg1_keys:
        seg1.extend(surface_cats.get(key, []))

    n_faces = len(seg1)
    arr2 = np.full(n_faces, 134, dtype=np.int32)
    arr3 = np.array([flat + 1 for _, flat in seg1], dtype=np.int32)
    arr5 = np.array([list(q) for q, _ in seg1], dtype=np.int32).reshape(-1)

    block0 = b"\x00" * 4608
    preamble = _mesh_geometry_preamble(meta_count, len(block0))
    block4 = _build_surface_block4(n_faces, template_path, meta_count)
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
                    meta_count,
                    blocks[0][1],
                )
                p0, bc0 = blocks[0]
                block0 = bytes(tpl_data[p0:p0 + bc0])

    face_bc = n_faces * 4

    inner = preamble
    inner += _write_section_block(block0)
    inner += _mesh_geom_sep(meta_count)
    inner += _write_section_block(meta.astype(">i4", copy=False).tobytes())
    # Vendor sep before arr2/arr3: face count; before arr5: byte size of arr2.
    inner += _mesh_geom_sep(n_faces)
    inner += _write_section_block(np.ascontiguousarray(arr2, dtype=">i4").tobytes())
    inner += _mesh_geom_sep(n_faces)
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
    cell_part: Optional[np.ndarray] = None,
) -> None:
    """Write a complete FLD from mesh arrays."""
    n_verts = vertices.shape[0]
    n_cells = cell_conn.shape[0]
    s_basename = _sdat_basename(s_text)

    header_tpl = resolve_header_template_fld(
        s_path=s_path,
        s_basename=s_basename,
        mesh_file=mesh_file,
        explicit=template_fld,
    )

    layout_tpl = resolve_geometry_layout_fld(
        s_path=s_path,
        s_basename=s_basename,
        mesh_file=mesh_file,
        explicit=template_fld,
    )

    geom_tpl: Optional[str] = None
    if template_fld and Path(template_fld).is_file():
        exp_cells = fld_cell_count_file(template_fld)
        if exp_cells == n_cells:
            geom_tpl = template_fld
        elif exp_cells is not None:
            geom_tpl = resolve_template_fld(
                n_cells,
                s_path=s_path,
                s_basename=s_basename,
                mesh_file=mesh_file,
            )
    else:
        geom_tpl = resolve_template_fld(
            n_cells,
            s_path=s_path,
            s_basename=s_basename,
            mesh_file=mesh_file,
        )

    vol_geom_tpl = geom_tpl or layout_tpl
    surf_geom_tpl = geom_tpl or layout_tpl

    normalized = s_text.replace("\r\n", "\n").lstrip("\ufeff")
    if not normalized.startswith("SDAT"):
        normalized = "SDAT\n" + normalized

    sfile_tpl = resolve_sfile_template_fld(
        normalized,
        s_path=s_path,
        s_basename=s_basename,
        mesh_file=mesh_file,
        explicit=template_fld,
    )

    body = _vendor_header_bytes(header_tpl)

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

    if header_tpl and Path(header_tpl).is_file():
        tpl = Path(header_tpl).read_bytes()
        body += _section_bytes(tpl, "LS_STREAMcoc")
        body += _section_bytes(tpl, "LS_STREAMmultiblock")

    body += _write_f64_axes_section(
        "LS_Nodes",
        [vertices[:, 0], vertices[:, 1], vertices[:, 2]],
        n_verts,
    )
    mat_inner = _mesh_cells_preamble(n_cells)
    mat_export = np.ascontiguousarray(
        np.where(np.asarray(material) == 0, 1, material), dtype=">i4",
    )
    mat_inner += _write_section_block(mat_export.tobytes())
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
    body += _write_volume_geometry_array(
        volume_names, n_cells, vol_geom_tpl, material=material, cell_part=cell_part,
    )

    if surface_cats:
        body += _write_surface_geometry_array(surface_cats, template_path=surf_geom_tpl)

    body += _write_sfile_section(normalized, sfile_tpl)

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
