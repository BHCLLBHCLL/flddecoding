#!/usr/bin/env python3
"""Verify ex2_e .s+.xemt FLD export: scPOST geometry and vol-flag bounds."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fld_model import (
    open_fld_buffer,
    parse_fld,
    validate_scpost_geometry,
    vol_flag_pair_issues,
    volume_block1_half_counts,
)
from sxemt2fldcgns import convert

S_PATH = ROOT / "tests" / "ex2_e.s"
XEMT_PATH = ROOT / "tests" / "ex2_e.xemt"
OUT_FLD = ROOT / "tests" / "ex2_e_from_sxemt.fld"
OUT_CGNS = ROOT / "tests" / "ex2_e_from_sxemt.cgns"


def compare_ex2() -> list[str]:
    errors: list[str] = []
    info = convert(str(S_PATH), str(XEMT_PATH), str(OUT_FLD), str(OUT_CGNS))
    parsed = parse_fld(str(OUT_FLD))

    if parsed["n_cells"] != info["cells"]:
        errors.append(f"FLD cells {parsed['n_cells']} != mesh {info['cells']}")
    if parsed["n_cells"] != 409188:
        errors.append(f"expected 409188 cells, got {parsed['n_cells']}")

    expected_names = [
        "PARTS1", "PARTS2", "PARTS3", "PARTS4", "PARTS5",
        "Domain(cuboid)", "CASE", "PCB", "CPU", "FIN",
    ]
    if parsed.get("volume_names") != expected_names:
        errors.append(f"volume names mismatch: {parsed.get('volume_names')}")

    with open_fld_buffer(str(OUT_FLD)) as data:
        issues = validate_scpost_geometry(data)
        if issues:
            errors.extend(issues)
        half = volume_block1_half_counts(data)
        if half is None or sum(half) != parsed["n_cells"]:
            errors.append(f"block1 half sum invalid: {half}")
        flag_issues = vol_flag_pair_issues(data, parsed["n_cells"])
        if flag_issues:
            errors.extend(flag_issues)

        from fld_model import find_section, iter_data_blocks, section_end, read_i32_be
        sec = find_section(data, "LS_SurfaceGeometryArray")
        if sec >= 0:
            blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
            if len(blocks) >= 5:
                meta = [read_i32_be(data, blocks[1][0] + i * 4) for i in range(blocks[1][1] // 4)]
                b4 = [read_i32_be(data, blocks[4][0] + i * 4) for i in range(blocks[4][1] // 4)]
                for i, (m, v) in enumerate(zip(meta, b4)):
                    if v != m * 4:
                        errors.append(f"surface block4 slot {i}: {v} != meta*4 ({m * 4})")
            if len(blocks) < 6:
                errors.append(f"surface missing link region: only {len(blocks)} blocks")

    off_fld = ROOT / "tests" / "ex2_e_67.fld"
    if off_fld.is_file():
        off = parse_fld(str(off_fld))
        if not np.array_equal(parsed["material"], off["material"]):
            errors.append("material mismatch vs ex2_e_67.fld")
        geo = sum(
            1 for i in range(parsed["n_cells"])
            if np.allclose(
                np.sort(off["vertices"][off["cell_conn"][i] - 1], axis=0),
                np.sort(parsed["vertices"][parsed["cell_conn"][i] - 1], axis=0),
                atol=1e-9,
            )
        )
        if geo != parsed["n_cells"]:
            errors.append(f"volume geometry match {geo}/{parsed['n_cells']} vs official")

    return errors


def main() -> int:
    errors = compare_ex2()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: ex2 FLD scPOST geometry and vol-flag bounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
