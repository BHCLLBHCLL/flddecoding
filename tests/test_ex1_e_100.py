#!/usr/bin/env python3
"""Compare fld2cgns output against vendor reference CGNS."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _parse_face_conn(conn: np.ndarray) -> list[tuple[int, int, int, int]]:
    faces: list[tuple[int, int, int, int]] = []
    i = 0
    while i < len(conn):
        if conn[i] == 7:
            faces.append(tuple(int(x) for x in conn[i + 1 : i + 5]))
            i += 5
        else:
            i += 1
    return faces


def _parse_hex_conn(conn: np.ndarray) -> np.ndarray:
    rows: list[list[int]] = []
    i = 0
    while i < len(conn):
        if conn[i] == 17:
            rows.append([int(x) for x in conn[i + 1 : i + 9]])
            i += 9
        else:
            i += 1
    return np.array(rows, dtype=np.int64)


def compare_cgns(gen_path: Path, ref_path: Path) -> list[str]:
    errors: list[str] = []
    with h5py.File(ref_path, "r") as ref, h5py.File(gen_path, "r") as gen:
        rz = ref["Base/FluidZone"]
        gz = gen["Base/FluidZone"]

        for axis in ("CoordinateX", "CoordinateY", "CoordinateZ"):
            r = rz[f"GridCoordinates/{axis}/ data"][()]
            g = gz[f"GridCoordinates/{axis}/ data"][()]
            if not np.allclose(r, g):
                errors.append(f"coords {axis} mismatch")

        ref_faces = _parse_face_conn(
            rz["GridElements_Faces/ElementConnectivity/ data"][()],
        )
        gen_faces = _parse_face_conn(
            gz["GridElements_Faces/ElementConnectivity/ data"][()],
        )
        if ref_faces != gen_faces:
            errors.append(f"faces mismatch: {len(ref_faces)} vs {len(gen_faces)}")

        for name in rz.keys():
            if name in (
                "ZoneType", "GridCoordinates", "ZoneBC", "FlowSolution",
                " data", "GridElements_Faces", "NotRegistered",
            ):
                continue
            if name not in gz:
                errors.append(f"missing element section {name}")
                continue
            if "ElementConnectivity" not in rz[name]:
                continue
            rc = _parse_hex_conn(rz[name]["ElementConnectivity/ data"][()])
            gc = _parse_hex_conn(gz[name]["ElementConnectivity/ data"][()])
            if rc.shape != gc.shape or not np.array_equal(rc, gc):
                errors.append(f"element section {name} connectivity mismatch")

        ref_bcs = set(rz["ZoneBC"].keys())
        gen_bcs = set(gz["ZoneBC"].keys())
        if ref_bcs != gen_bcs:
            errors.append(f"BC names differ: missing={ref_bcs-gen_bcs} extra={gen_bcs-ref_bcs}")
        for bc in ref_bcs:
            rpl = rz["ZoneBC"][bc].get("PointList")
            gpl = gz["ZoneBC"][bc].get("PointList")
            if rpl is None and gpl is None:
                continue
            if rpl is None or gpl is None:
                errors.append(f"BC {bc} PointList presence mismatch")
                continue
            r_ids = rpl[" data"][()]
            g_ids = gpl[" data"][()]
            if not np.array_equal(r_ids, g_ids):
                errors.append(f"BC {bc} PointList mismatch")

        rfs = rz["FlowSolution"]
        gfs = gz["FlowSolution"]
        for name in rfs.keys():
            if name == "GridLocation":
                continue
            if name not in gfs:
                errors.append(f"missing FlowSolution field {name}")
                continue
            r = rfs[name][" data"][()]
            g = gfs[name][" data"][()]
            if not np.allclose(r, g):
                errors.append(f"field {name} mismatch")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", nargs="?", default="tests/ex1_e_100_out.cgns")
    parser.add_argument("reference", nargs="?", default="tests/ex1_e_100_orig.cgns")
    args = parser.parse_args()

    gen = ROOT / args.generated
    ref = ROOT / args.reference
    if not gen.is_file():
        print(f"Error: {gen} not found")
        sys.exit(1)
    if not ref.is_file():
        print(f"Error: {ref} not found")
        sys.exit(1)

    errors = compare_cgns(gen, ref)
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"PASS: {gen.name} matches {ref.name}")


if __name__ == "__main__":
    main()
