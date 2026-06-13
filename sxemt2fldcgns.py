#!/usr/bin/env python3
"""
Convert scFLOW SDAT (.s) + EMT (.xemt) to FLD and CGNS without .r or template FLD.

Mesh is built from CXYZ spacing and PARTS iron box in the .s file.
Part/material names come from the .xemt file.
"""

import argparse
import sys
from pathlib import Path

from fld2cgns import write_cgns
from fld_model import parse_fld
from fld_writer import default_initial_fields, write_fld_from_mesh
from mesh_builder import build_mesh_from_sdat, mesh_to_fld_dict
from s_model import parse_sdat_file, vertex_temperature_field
from xemt_model import parse_xemt_file, volume_labels


def convert(
    s_path: str,
    xemt_path: str,
    fld_out: str,
    cgns_out: str | None = None,
) -> dict:
    s_file = Path(s_path)
    xemt_file = Path(xemt_path)
    model = parse_sdat_file(str(s_file))
    xemt = parse_xemt_file(str(xemt_file))
    vol_names = list(volume_labels(xemt))

    built = build_mesh_from_sdat(model, volume_names=vol_names)
    temp = vertex_temperature_field(
        model,
        built.vertices,
        built.cell_conn,
        built.material,
        built.volume_names,
    )
    fields = default_initial_fields(built.vertices.shape[0], temp, ambient=model.ambient_temp)
    s_text = s_file.read_text(encoding="utf-8-sig")

    write_fld_from_mesh(
        fld_out,
        built.vertices,
        built.cell_conn,
        built.material,
        fields,
        s_text,
        built.volume_names,
    )

    mesh_dict = mesh_to_fld_dict(built, fields)
    if cgns_out:
        write_cgns(mesh_dict, cgns_out)

    return {
        "fld": fld_out,
        "cgns": cgns_out,
        "vertices": built.vertices.shape[0],
        "cells": built.cell_conn.shape[0],
        "faces": len(built.faces),
        "temp_range": (float(temp.min()), float(temp.max())),
        "volume_names": vol_names,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert .s + .xemt to FLD and CGNS (no .r / template FLD)"
    )
    parser.add_argument("s_file", help="SDAT solver file (.s)")
    parser.add_argument("xemt_file", help="EMT metadata file (.xemt)")
    parser.add_argument("-o", "--fld-output", help="Output .fld path")
    parser.add_argument("--cgns", metavar="FILE", help="Also write CGNS to this path")
    parser.add_argument(
        "--verify-parse",
        action="store_true",
        help="Re-parse written FLD and print summary",
    )
    args = parser.parse_args(argv)

    s_path = Path(args.s_file)
    xemt_path = Path(args.xemt_file)
    if not s_path.is_file():
        print(f"Error: {s_path} not found", file=sys.stderr)
        return 1
    if not xemt_path.is_file():
        print(f"Error: {xemt_path} not found", file=sys.stderr)
        return 1

    fld_out = args.fld_output or str(s_path.parent / f"{s_path.stem}_0.fld")
    cgns_out = args.cgns or str(Path(fld_out).with_suffix(".cgns"))

    try:
        info = convert(str(s_path), str(xemt_path), fld_out, cgns_out)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"FLD written: {info['fld']}")
    print(f"CGNS written: {info['cgns']}")
    print(f"Mesh: {info['vertices']} vertices, {info['cells']} cells, {info['faces']} faces")
    print(f"TEMP range: {info['temp_range'][0]:.4g} .. {info['temp_range'][1]:.4g}")
    print(f"Volumes: {info['volume_names']}")

    if args.verify_parse:
        m = parse_fld(fld_out)
        print(f"Parse check: {m['n_vertices']} verts, {m['n_cells']} cells, fields={sorted(m['fields'].keys())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
