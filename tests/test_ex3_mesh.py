#!/usr/bin/env python3
"""Verify ex3_e .s+.xemt FLD export: scPOST geometry, volume layout, vol flags."""

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

REF_FLD = ROOT / "tests" / "ex3_e_151.fld"
S_PATH = ROOT / "tests" / "ex3_e.s"
XEMT_PATH = ROOT / "tests" / "ex3_e.xemt"
OUT_FLD = ROOT / "tests" / "ex3_e_from_sxemt.fld"
OUT_CGNS = ROOT / "tests" / "ex3_e_from_sxemt.cgns"

EX3_VOLUME_NAMES = [
    "PARTS1", "PARTS2", "PARTS3", "PARTS5", "PARTS6",
    "Interior", "Table", "Table2", "Wall", "Window",
]


def compare_ex3() -> list[str]:
    errors: list[str] = []
    if not REF_FLD.is_file():
        errors.append(f"missing reference {REF_FLD}")
        return errors

    ref = parse_fld(str(REF_FLD))
    info = convert(str(S_PATH), str(XEMT_PATH), str(OUT_FLD), str(OUT_CGNS))
    parsed = parse_fld(str(OUT_FLD))

    if parsed["n_cells"] != info["cells"]:
        errors.append(f"FLD cells {parsed['n_cells']} != mesh {info['cells']}")

    if parsed.get("volume_names") != EX3_VOLUME_NAMES:
        errors.append(
            f"volume names {parsed.get('volume_names')} != ex3 template {EX3_VOLUME_NAMES}",
        )

    mat = parsed["material"]
    if int(np.sum(mat == 0)) > 0:
        errors.append("FLD still contains material id 0")

    with open_fld_buffer(str(OUT_FLD)) as data:
        issues = validate_scpost_geometry(data)
        if issues:
            errors.extend(issues)
        half = volume_block1_half_counts(data)
        if half is None:
            errors.append("could not read volume block1 half counts")
        elif sum(half) != parsed["n_cells"]:
            errors.append(f"block1 half sum {sum(half)} != n_cells {parsed['n_cells']}")
        elif len(half) != 5:
            errors.append(f"expected 5 ex3 buckets in block1, got {half}")
        flag_issues = vol_flag_pair_issues(data, parsed["n_cells"])
        if flag_issues:
            errors.extend(flag_issues)

    if ref.get("volume_names") != EX3_VOLUME_NAMES:
        errors.append("reference ex3_e_151 volume names changed — update EX3_VOLUME_NAMES")

    return errors


def main() -> int:
    errors = compare_ex3()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: ex3 FLD scPOST geometry and volume layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
