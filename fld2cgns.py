#!/usr/bin/env python3
"""
FLD to CGNS Converter.

Reads FLD (CRDL-FLD) field files from Software Cradle scFLOW and writes
CGNS/HDF5 matching the vendor FLDUTIL exporter layout
(see tests/ex1_e_100_orig.cgns).

Requires: numpy, h5py.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from fld_model import parse_fld

try:
    import h5py
except ImportError:
    print("Error: h5py is required. Install with: pip install h5py numpy")
    sys.exit(1)

# NGON face elements start at this index (after volume element sections).
_FACE_ELEM_START = 36481
# CGNS element type codes used in connectivity arrays.
_HEX_TYPE = 17
_QUAD_TYPE = 7
_NGON_TYPE = 20


def _create_group(parent, name: str):
    return parent.create_group(name, track_order=False)


def _set_cgns_attrs(grp, name: str, label: str, type_str: str) -> None:
    grp.attrs.create("flags", np.array([1], dtype=np.int32))
    grp.attrs.create("label", np.bytes_(label), dtype=h5py.string_dtype(length=33))
    grp.attrs.create("name", np.bytes_(name), dtype=h5py.string_dtype(length=33))
    grp.attrs.create("type", np.bytes_(type_str), dtype=h5py.string_dtype(length=3))


def _cgns_node(parent, name: str, label: str, type_str: str):
    grp = _create_group(parent, name)
    _set_cgns_attrs(grp, name, label, type_str)
    return grp


def _bytes_dataset(parent, payload: bytes) -> None:
    parent.create_dataset(" data", data=np.frombuffer(payload, dtype=np.int8))


def _i32_dataset(parent, arr) -> None:
    parent.create_dataset(" data", data=np.asarray(arr, dtype=np.int32))


def _write_grid_coordinates(zone, vertices: np.ndarray) -> None:
    gc = _cgns_node(zone, "GridCoordinates", "GridCoordinates_t", "MT")
    for axis, cname in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
        cd = _cgns_node(gc, cname, "DataArray_t", "R8")
        cd.create_dataset(
            " data",
            data=np.ascontiguousarray(vertices[:, axis], dtype=np.float64),
        )


def _write_hex_elements(zone, name: str, cells: np.ndarray, elem_start: int) -> int:
    """Write NGON_t section with HEXA_8 connectivity (type 17 prefix). Returns next elem index."""
    n_cells = int(cells.shape[0])
    el = _cgns_node(zone, name, "Elements_t", "I4")
    _i32_dataset(el, [_NGON_TYPE, n_cells])

    er = _cgns_node(el, "ElementRange", "IndexRange_t", "I4")
    _i32_dataset(er, [elem_start, elem_start + n_cells - 1])

    conn: list[int] = []
    for row in cells:
        conn.append(_HEX_TYPE)
        conn.extend(int(x) for x in row)
    ec = _cgns_node(el, "ElementConnectivity", "DataArray_t", "I4")
    _i32_dataset(ec, conn)
    return elem_start + n_cells


def _write_face_elements(zone, faces: list[tuple[int, int, int, int]]) -> None:
    el = _cgns_node(zone, "GridElements_Faces", "Elements_t", "I4")
    n_faces = len(faces)
    _i32_dataset(el, [_NGON_TYPE, n_faces])

    er = _cgns_node(el, "ElementRange", "IndexRange_t", "I4")
    _i32_dataset(er, [_FACE_ELEM_START, _FACE_ELEM_START + n_faces - 1])

    conn: list[int] = []
    for face in faces:
        conn.append(_QUAD_TYPE)
        conn.extend(face)
    ec = _cgns_node(el, "ElementConnectivity", "DataArray_t", "I4")
    _i32_dataset(ec, conn)


def _write_zone_bc(zone, bc_plan: list[tuple[str, int, int]]) -> None:
    zbc = _cgns_node(zone, "ZoneBC", "ZoneBC_t", "MT")
    used_bc: set[str] = set()
    for bc_name, start_idx, count in bc_plan:
        if count <= 0:
            continue
        name = _unique_cgns_name(bc_name, used_bc)
        bc = _cgns_node(zbc, name, "BC_t", "C1")
        _bytes_dataset(bc, b"Null")
        gl = _cgns_node(bc, "GridLocation", "GridLocation_t", "C1")
        _bytes_dataset(gl, b"FaceCenter")
        pl = _cgns_node(bc, "PointList", "IndexArray_t", "I4")
        ids = np.arange(
            _FACE_ELEM_START + start_idx,
            _FACE_ELEM_START + start_idx + count,
            dtype=np.int32,
        )
        _i32_dataset(pl, ids)


def _write_flow_solution(zone, fields: dict[str, np.ndarray]) -> None:
    fs = _cgns_node(zone, "FlowSolution", "FlowSolution_t", "MT")
    order = [
        "PRES", "TEMP", "TURK", "TEPS", "CN01", "HTRC", "SURT", "HTFX", "ATMS",
        "VECTX", "VECTY", "VECTZ", "HVECX", "HVECY", "HVECZ",
    ]
    for name in order:
        if name not in fields:
            continue
        node = _cgns_node(fs, name, "DataArray_t", "R8")
        node.create_dataset(" data", data=np.ascontiguousarray(fields[name], dtype=np.float64))


def _volume_element_names(volume_names: list[str], mat: np.ndarray) -> tuple[str, str, str, str]:
    """Return PARTS1, PARTS2, domain name, iron name labels."""
    parts1 = "PARTS1"
    parts2 = "PARTS2"
    domain = "Domain(cuboid)"
    iron = "Iron"
    if len(volume_names) >= 4:
        parts1, parts2, domain, iron = volume_names[0], volume_names[1], volume_names[2], volume_names[3]
    elif len(volume_names) == 2:
        domain, iron = volume_names[0], volume_names[1]
    return parts1, parts2, domain, iron


def _unique_cgns_name(name: str, used: set[str]) -> str:
    """Return *name* unique within *used* (HDF5 group names)."""
    base = name.strip() or "UNNAMED"
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}_{n}" in used:
        n += 1
    unique = f"{base}_{n}"
    used.add(unique)
    return unique


def write_cgns(mesh: dict, outpath: str, zone_name: str = "FluidZone") -> None:
    vertices = mesh["vertices"]
    cell_conn = mesh["cell_conn"]
    material = mesh["material"]
    faces = mesh["faces"]
    bc_plan = mesh["bc_plan"]
    fields = mesh.get("fields", {})
    cell_part = mesh.get("cell_part")
    part_names = mesh.get("part_names", [])

    if vertices is None or cell_conn is None or material is None:
        raise ValueError("Incomplete mesh data")
    if not faces:
        raise ValueError("No surface faces parsed from LS_SurfaceGeometryArray")

    n_vertex = int(vertices.shape[0])
    n_cells = int(cell_conn.shape[0])

    element_sections: list[tuple[str, np.ndarray]] = []
    if cell_part is not None and part_names:
        n_parts = len(part_names)
        for pid in range(1, n_parts + 1):
            mask = cell_part == pid
            if np.any(mask):
                element_sections.append((f"PARTS{pid}", cell_conn[mask]))
        for pid, pname in enumerate(part_names, start=1):
            mask = cell_part == pid
            if np.any(mask):
                element_sections.append((pname, cell_conn[mask]))
    else:
        parts1, parts2, domain_name, iron_name = _volume_element_names(
            mesh.get("volume_names", []), material,
        )
        cells_m1 = cell_conn[material == 1]
        cells_m2 = cell_conn[material == 2]
        for label, cells in [
            (parts1, cells_m1),
            (parts2, cells_m2),
            (domain_name, cells_m1),
            (iron_name, cells_m2),
        ]:
            if cells.shape[0] > 0:
                element_sections.append((label, cells))

    with h5py.File(outpath, "w", libver=("earliest", "v108")) as f:
        f.attrs.create("label", np.bytes_("Root Node of HDF5 File"),
                       dtype=h5py.string_dtype(length=33))
        f.attrs.create("name", np.bytes_("HDF5 MotherNode"),
                       dtype=h5py.string_dtype(length=33))
        f.attrs.create("type", np.bytes_("MT"), dtype=h5py.string_dtype(length=3))
        f.create_dataset(" hdf5version", data=np.zeros(33, dtype=np.int8))
        f.create_dataset(
            " format",
            data=np.frombuffer(b"IEEE_LITTLE_32\0", dtype=np.int8),
        )

        lv = _cgns_node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4")
        lv.create_dataset(" data", data=np.array([3.21], dtype=np.float32))

        base = _cgns_node(f, "Base", "CGNSBase_t", "I4")
        _i32_dataset(base, [3, 3])

        rs = _cgns_node(base, "ReferenceState", "ReferenceState_t", "MT")
        rsd = _cgns_node(rs, "ReferenceStateDescription", "Descriptor_t", "C1")
        _bytes_dataset(rsd, b"Software Cradle FLDUTIL")

        zone = _cgns_node(base, zone_name, "Zone_t", "I4")
        zone.create_dataset(
            " data",
            data=np.array([[n_vertex], [n_cells], [0]], dtype=np.int32),
        )
        zt = _cgns_node(zone, "ZoneType", "ZoneType_t", "C1")
        _bytes_dataset(zt, b"Unstructured")

        _write_grid_coordinates(zone, vertices)

        elem_next = 1
        used_elem_names: set[str] = set()
        for label, cells in element_sections:
            name = _unique_cgns_name(label, used_elem_names)
            elem_next = _write_hex_elements(zone, name, cells, elem_next)

        # Empty NotRegistered section (matches vendor exporter).
        el = _cgns_node(zone, "NotRegistered", "Elements_t", "I4")
        _i32_dataset(el, [_NGON_TYPE, 0])
        er = _cgns_node(el, "ElementRange", "IndexRange_t", "I4")
        _i32_dataset(er, [_FACE_ELEM_START, _FACE_ELEM_START - 1])
        ec = _cgns_node(el, "ElementConnectivity", "DataArray_t", "I4")
        _i32_dataset(ec, [])

        _write_face_elements(zone, faces)
        _write_zone_bc(zone, bc_plan)
        _write_flow_solution(zone, fields)


def main():
    parser = argparse.ArgumentParser(description="Convert FLD file to CGNS format")
    parser.add_argument("fld_file", nargs="?", default="tests/ex1_e_100.fld",
                        help="Input FLD file")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Output CGNS file (default: input basename with .cgns)")
    args = parser.parse_args()

    fld_path = Path(args.fld_file)
    if not fld_path.exists():
        print(f"Error: file not found: {fld_path}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else fld_path.with_suffix(".cgns")

    print(f"Reading: {fld_path}")
    mesh = parse_fld(str(fld_path))
    print(f"  Vertices : {mesh['n_vertices']}")
    print(f"  Cells    : {mesh['n_cells']}")
    print(f"  Faces    : {len(mesh['faces'])}")
    print(f"  BC groups: {len(mesh['bc_plan'])}")
    print(f"  Fields   : {sorted(mesh['fields'].keys())}")
    if mesh.get("volume_names"):
        print(f"  Volumes  : {mesh['volume_names']}")

    if mesh["vertices"] is None:
        print("Error: could not parse mesh from FLD.")
        sys.exit(1)

    print(f"Writing: {out_path}")
    write_cgns(mesh, str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
