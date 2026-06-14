#!/usr/bin/env python3
"""Compare ex4 mesh from .s+.xemt against official ex4_e_63 reference."""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fld_model import parse_fld, _parse_hex_cells, open_fld_buffer
from mesh_builder import build_mesh_from_sdat
from s_model import parse_sdat_file
from sxemt2fldcgns import convert
from xemt_model import volume_names_for_sdat


def _parse_hex_conn(conn: np.ndarray) -> np.ndarray:
    rows: list[list[int]] = []
    i = 0
    while i < len(conn):
        if conn[i] == 17:
            rows.append([int(x) for x in conn[i + 1 : i + 9]])
            i += 9
        elif conn[i] == 20:
            i += 1 + int(conn[i + 1]) + 1
        else:
            i += 1
    return np.array(rows, dtype=np.int64)


def compare_ex4() -> list[str]:
    errors: list[str] = []
    s_path = ROOT / "tests" / "ex4_e.s"
    xemt_path = ROOT / "tests" / "ex4_e.xemt"
    ref_fld = ROOT / "tests" / "ex4_e_63.fld"
    ref_cgns = ROOT / "tests" / "ex4_e_63_orig.cgns"
    fld_out = ROOT / "tests" / "ex4_e_from_sxemt.fld"
    cgns_out = ROOT / "tests" / "ex4_e_from_sxemt.cgns"

    if not ref_fld.is_file():
        errors.append(f"missing reference {ref_fld}")
        return errors

    model = parse_sdat_file(str(s_path))
    vol_names = volume_names_for_sdat([p.name for p in model.parts])
    built = build_mesh_from_sdat(model, volume_names=vol_names)

    ref = parse_fld(str(ref_fld))
    if built.vertices.shape[0] != ref["n_vertices"]:
        errors.append(
            f"vertices {built.vertices.shape[0]} != ref {ref['n_vertices']} "
            f"(delta {ref['n_vertices'] - built.vertices.shape[0]})"
        )
    if built.cell_conn.shape[0] != 1470392:
        errors.append(f"cells {built.cell_conn.shape[0]} != 1470392")

    our_mat = np.bincount(built.material.astype(int))
    with open_fld_buffer(str(ref_fld)) as data:
        _, ref_mat = _parse_hex_cells(data)
    if ref_mat is not None:
        ref_binc = np.bincount(ref_mat.astype(int))
        if not np.array_equal(our_mat, ref_binc):
            errors.append(f"material bincount {our_mat} != {ref_binc}")

    # part cell counts vs official PARTS1..PARTS32
    if ref_cgns.is_file():
        with h5py.File(ref_cgns, "r") as f:
            zone = f["Base/FluidZone"]
            for pid in range(1, len(built.part_names) + 1):
                name = f"PARTS{pid}"
                if name not in zone:
                    errors.append(f"missing ref section {name}")
                    continue
                ref_cells = _parse_hex_conn(zone[name]["ElementConnectivity/ data"][()])
                our_cells = built.cell_conn[built.cell_part == pid]
                if ref_cells.shape != our_cells.shape:
                    errors.append(f"{name} count {our_cells.shape[0]} != ref {ref_cells.shape[0]}")

    info = convert(str(s_path), str(xemt_path), str(fld_out), str(cgns_out))
    parsed = parse_fld(str(fld_out))
    if parsed["n_vertices"] != built.vertices.shape[0]:
        errors.append("written FLD vertex count mismatch")
    if parsed["n_cells"] != built.cell_conn.shape[0]:
        errors.append(f"written FLD cells {parsed['n_cells']} != {built.cell_conn.shape[0]}")
    if parsed.get("volume_names") != vol_names:
        errors.append("FLD volume_names mismatch with vendor layout")

    if ref_cgns.is_file() and cgns_out.is_file():
        with h5py.File(ref_cgns, "r") as ref, h5py.File(cgns_out, "r") as gen:
            rz, gz = ref["Base/FluidZone"], gen["Base/FluidZone"]
            ref_secs = {
                k for k in rz.keys()
                if k not in ("ZoneType", "GridCoordinates", "ZoneBC", "FlowSolution", " data",
                             "GridElements_Faces", "NotRegistered")
            }
            gen_secs = {
                k for k in gz.keys()
                if k not in ("ZoneType", "GridCoordinates", "ZoneBC", "FlowSolution", " data",
                             "GridElements_Faces", "NotRegistered")
            }
            if ref_secs != gen_secs:
                errors.append(f"CGNS element sections differ missing={ref_secs-gen_secs} extra={gen_secs-ref_secs}")

    if len(vol_names) != 64:
        errors.append(f"expected 64 volume labels, got {len(vol_names)}")

    return errors


def main() -> int:
    errors = compare_ex4()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: ex4 mesh matches official reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
