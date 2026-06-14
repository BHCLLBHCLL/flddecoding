#!/usr/bin/env python3
"""Verify sxemt2fldcgns: mesh stats, TEMP, CGNS readable."""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fld_model import parse_fld
from mesh_builder import build_mesh_from_sdat
from s_model import parse_sdat_file, vertex_temperature_field
from sxemt2fldcgns import convert
from xemt_model import parse_xemt_file, volume_labels


def test_sxemt_pipeline() -> list[str]:
    errors: list[str] = []
    s_path = ROOT / "tests" / "ex1_e.s"
    xemt_path = ROOT / "tests" / "ex1_e.xemt"
    fld_out = ROOT / "tests" / "ex1_e_from_sxemt_run.fld"
    cgns_out = ROOT / "tests" / "ex1_e_from_sxemt_run.cgns"

    info = convert(str(s_path), str(xemt_path), str(fld_out), str(cgns_out))

    if info["vertices"] != 21145:
        errors.append(f"expected 21145 vertices, got {info['vertices']}")
    if info["cells"] != 18240:
        errors.append(f"expected 18240 cells, got {info['cells']}")

    model = parse_sdat_file(str(s_path))
    xemt = parse_xemt_file(str(xemt_path))
    built = build_mesh_from_sdat(model, volume_names=list(volume_labels(xemt)))
    temp = vertex_temperature_field(
        model, built.vertices, built.cell_conn, built.material, built.volume_names,
    )
    if not np.any(np.isclose(temp, 20.0)) or not np.any(np.isclose(temp, 100.0)):
        errors.append("TEMP missing 20C / 100C regions")

    parsed = parse_fld(str(fld_out))
    if parsed["n_vertices"] != info["vertices"]:
        errors.append("FLD parse vertex count mismatch")
    if not np.allclose(parsed["fields"]["TEMP"], temp):
        errors.append("FLD TEMP does not match INIT_REGION mapping")

    with h5py.File(cgns_out, "r") as f:
        zone = f["Base/FluidZone"]
        nx = int(zone[" data"][0, 0])
        if nx != info["vertices"]:
            errors.append(f"CGNS vertex count {nx} != {info['vertices']}")

    return errors


def main() -> int:
    errors = test_sxemt_pipeline()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: sxemt2fldcgns pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
