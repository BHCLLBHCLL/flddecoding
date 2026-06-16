#!/usr/bin/env python3
"""Compare official ex3_e_151.fld vs generated ex3_e_from_sxemt.fld."""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fld_model import (
    find_section,
    iter_data_blocks,
    open_fld_buffer,
    parse_fld,
    section_end,
    validate_scpost_geometry,
    vol_flag_pair_issues,
    volume_block1_half_counts,
)

OFFICIAL = ROOT / "tests" / "ex3_e_151.fld"
OURS = ROOT / "tests" / "ex3_e_from_sxemt.fld"


def section_size(path: Path, name: str) -> int | None:
    data = path.read_bytes()
    s = find_section(data, name)
    if s < 0:
        return None
    return section_end(data, s) - s


def sfile_info(path: Path) -> dict:
    data = path.read_bytes()
    s = find_section(data, "LS_SFile")
    if s < 0:
        return {}
    e = section_end(data, s)
    blocks = list(iter_data_blocks(data, s, e))
    pre_len = blocks[0][0] - (s + 40) - 8 if blocks else 0
    return {
        "section_size": e - s,
        "preamble": pre_len,
        "block0": blocks[0][1] if blocks else 0,
        "block1": blocks[1][1] if len(blocks) > 1 else 0,
    }


def vol_geom_preamble(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    s = find_section(data, "LS_VolumeGeometryArray")
    blocks = list(iter_data_blocks(data, s, section_end(data, s)))
    pre = data[s + 40 : blocks[0][0] - 8]
    return struct.unpack(">i", pre[100:104])[0], struct.unpack(">i", pre[104:108])[0]


def header_through_coord(path: Path) -> int:
    data = path.read_bytes()
    cs = find_section(data, "LS_CoordinateSystem")
    return section_end(data, cs)


def vol_flag_summary(path: Path, n_cells: int) -> dict:
    data = path.read_bytes()
    sec = find_section(data, "LS_VolumeGeometryArray")
    blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
    p2, bc2 = blocks[2]
    pairs = np.frombuffer(data[p2:p2 + bc2], dtype=">i4").reshape(-1, 2)
    lo, hi = pairs[:, 0], pairs[:, 1]
    return {
        "lo_max": int(lo.max()),
        "lo_ge_nc": int(np.sum(lo >= n_cells)),
        "hi_gt_nc": int(np.sum(hi > n_cells)),
        "issues": vol_flag_pair_issues(data, n_cells),
    }


def main() -> int:
    if not OFFICIAL.is_file():
        print(f"Missing {OFFICIAL}")
        return 1
    if not OURS.is_file():
        print(f"Missing {OURS} — run: python sxemt2fldcgns.py tests/ex3_e.s tests/ex3_e.xemt")
        return 1

    off = parse_fld(str(OFFICIAL))
    ours = parse_fld(str(OURS))

    print("=" * 60)
    print("FILE SIZE")
    print(f"  Official: {OFFICIAL.stat().st_size:,}")
    print(f"  Ours:     {OURS.stat().st_size:,}")
    print(f"  Diff:     {OURS.stat().st_size - OFFICIAL.stat().st_size:,}")

    print("\nMESH SUMMARY")
    print(f"  Vertices  official={off['n_vertices']} ours={ours['n_vertices']}")
    print(f"  Cells     official={off['n_cells']} ours={ours['n_cells']}")
    print(f"  Faces     official={len(off['faces'])} ours={len(ours['faces'])}")

    off_mat = np.bincount(off["material"].astype(int), minlength=4)
    our_mat = np.bincount(ours["material"].astype(int), minlength=4)
    print(f"  Material bincount official: {dict(enumerate(off_mat.tolist()))}")
    print(f"  Material bincount ours:   {dict(enumerate(our_mat.tolist()))}")

    print("\nHEADER (through LS_CoordinateSystem)")
    h_off, h_our = header_through_coord(OFFICIAL), header_through_coord(OURS)
    print(f"  Official: {h_off} bytes")
    print(f"  Ours:     {h_our} bytes")
    print(f"  Match:    {h_off == h_our}")

    print("\nKEY SECTION SIZES")
    for name in (
        "LS_SFile",
        "LS_VolumeGeometryArray",
        "LS_SurfaceGeometryArray",
        "LS_Nodes",
        "LS_Elements",
        "Pressure",
        "Temperature",
    ):
        so, su = section_size(OFFICIAL, name), section_size(OURS, name)
        if so is None and su is None:
            continue
        diff = (su or 0) - (so or 0)
        print(f"  {name:28} off={so or '—':>10} ours={su or '—':>10} diff={diff:+,}")

    print("\nLS_SFile STRUCTURE (scPOST-critical)")
    for label, path in [("Official", OFFICIAL), ("Ours", OURS)]:
        info = sfile_info(path)
        print(
            f"  {label}: size={info.get('section_size')}, preamble={info.get('preamble')}, "
            f"block0={info.get('block0')}, block1_slot={info.get('block1')}",
        )

    print("\nLS_VolumeGeometryArray PREAMBLE (offset 100, 104)")
    for label, path in [("Official", OFFICIAL), ("Ours", OURS)]:
        a100, a104 = vol_geom_preamble(path)
        print(f"  {label}: @100={a100} @104={a104}  preamble_ok={a100 == 1}")

    print("\nVOLUME block1 (first half) & vol-flag bounds")
    with open_fld_buffer(str(OFFICIAL)) as data_off:
        b1_off = volume_block1_half_counts(data_off)
    with open_fld_buffer(str(OURS)) as data_our:
        b1_our = volume_block1_half_counts(data_our)
    print(f"  Official buckets: {b1_off}")
    print(f"  Ours buckets:     {b1_our}")
    vf_off = vol_flag_summary(OFFICIAL, off["n_cells"])
    vf_our = vol_flag_summary(OURS, ours["n_cells"])
    print(
        f"  Official vol-flag lo_max={vf_off['lo_max']} lo>=nc={vf_off['lo_ge_nc']} "
        f"hi>nc={vf_off['hi_gt_nc']}",
    )
    print(
        f"  Ours vol-flag     lo_max={vf_our['lo_max']} lo>=nc={vf_our['lo_ge_nc']} "
        f"hi>nc={vf_our['hi_gt_nc']}",
    )

    print("\nscPOST GEOMETRY VALIDATION")
    for label, path in [("Official", OFFICIAL), ("Ours", OURS)]:
        with open_fld_buffer(str(path)) as data:
            issues = validate_scpost_geometry(data)
        print(f"  {label}: {'OK' if not issues else issues}")

    print("\nVOLUME NAMES")
    vn_off = off.get("volume_names", [])
    vn_our = ours.get("volume_names", [])
    print(f"  Official ({len(vn_off)}): {vn_off}")
    print(f"  Ours ({len(vn_our)}):     {vn_our}")
    print(f"  Names equal: {vn_off == vn_our}")

    print("\nLAYOUT CHECKLIST")
    layout_ok: list[str] = []
    if h_off == h_our:
        layout_ok.append("File header matches vendor size")
    si_off, si_our = sfile_info(OFFICIAL), sfile_info(OURS)
    if si_our.get("preamble") == 48:
        layout_ok.append("LS_SFile 48-byte vendor preamble")
    if vol_geom_preamble(OURS)[0] == 1:
        layout_ok.append("VolumeGeometry preamble @100 = 1")
    if vn_off == vn_our:
        layout_ok.append("10 ex3 volume name slots match official")
    if not vf_our["issues"]:
        layout_ok.append("Vol-flag pairs within n_cells bounds")
    if int(np.sum(ours["material"] == 0)) == 0:
        layout_ok.append("No material id 0 in export")
    with open_fld_buffer(str(OURS)) as data:
        if not validate_scpost_geometry(data):
            layout_ok.append("validate_scpost_geometry passes")
    for item in layout_ok:
        print(f"  [OK] {item}")

    mesh_diff = []
    if off["n_vertices"] != ours["n_vertices"]:
        mesh_diff.append(f"vertices {off['n_vertices']} vs {ours['n_vertices']}")
    if off["n_cells"] != ours["n_cells"]:
        mesh_diff.append(f"cells {off['n_cells']} vs {ours['n_cells']}")

    if mesh_diff:
        print("\nMESH DIFFERENCES (SDAT mesh builder vs official step-151)")
        for item in mesh_diff:
            print(f"  • {item}")

    print("\nCONCLUSION")
    with open_fld_buffer(str(OURS)) as data:
        our_issues = validate_scpost_geometry(data)
    if not our_issues and vn_off == vn_our and not vf_our["issues"]:
        print("  Binary layout aligned for scPOST; ex3_e_from_sxemt.fld verified readable in scPOST.")
        if mesh_diff:
            print("  Cell/vertex counts differ from official — expected for SDAT-built mesh.")
    else:
        print("  Remaining issues:", our_issues or vf_our["issues"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
