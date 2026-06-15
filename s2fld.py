#!/usr/bin/env python3
"""
Convert scFLOW / SCTpre SDAT solver file (.s) to FLD field file.

When --xemt is given (recommended), mesh is built from CXYZ + PARTS box without
.r or template FLD. Otherwise falls back to template FLD copy (--template).
"""

import argparse
import sys
from pathlib import Path

from fld_model import parse_fld
from fld_writer import (
    compose_fld,
    default_initial_fields,
    patch_cycle,
    resolve_template_fld,
    write_fld_from_mesh,
)
from mesh_builder import build_mesh_from_sdat, mesh_to_fld_dict
from s_model import parse_sdat_file, summarize_sdat, vertex_temperature_field
from xemt_model import parse_xemt_file, volume_names_for_sdat


def _resolve_template(s_path: Path, template: str | None, mesh: str | None) -> Path:
    if template:
        p = Path(template)
        if not p.is_file():
            raise FileNotFoundError(f"Template FLD not found: {p}")
        return p
    if mesh:
        p = Path(mesh)
        if not p.is_file():
            raise FileNotFoundError(f"Mesh FLD not found: {p}")
        return p
    stem = s_path.stem
    parent = s_path.parent
    candidates = [
        parent / f"{stem}_100.fld",
        parent / f"{stem}_0.fld",
        parent / f"{stem}.fld",
    ]
    model = parse_sdat_file(str(s_path))
    if model.mesh_file:
        r_stem = Path(model.mesh_file).stem
        candidates.insert(0, parent / f"{r_stem}.fld")
        candidates.insert(1, parent / f"{r_stem}_100.fld")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No template FLD found for {s_path}. "
        "Use --xemt to build mesh from .s, or pass --template."
    )


def convert_s_to_fld(
    s_path: str,
    out_path: str,
    xemt_path: str | None = None,
    template: str | None = None,
    mesh: str | None = None,
    cycle: int | None = 0,
) -> dict:
    s_file = Path(s_path)
    model = parse_sdat_file(str(s_file))
    s_text = s_file.read_text(encoding="utf-8-sig")

    if xemt_path:
        xemt = parse_xemt_file(xemt_path)
        vol_names = volume_names_for_sdat([p.name for p in model.parts])
        built = build_mesh_from_sdat(model, volume_names=vol_names)
        temp = vertex_temperature_field(
            model,
            built.vertices,
            built.cell_conn,
            built.material,
            built.volume_names,
        )
        fields = default_initial_fields(built.vertices.shape[0], temp, ambient=model.ambient_temp)
        tpl = resolve_template_fld(
            built.cell_conn.shape[0],
            s_path=str(s_file),
            mesh_file=model.mesh_file,
            explicit=template or mesh,
        )
        write_fld_from_mesh(
            out_path,
            built.vertices,
            built.cell_conn,
            built.material,
            fields,
            s_text,
            built.volume_names,
            surface_cats=built.surface_cats,
            template_fld=tpl,
            s_path=str(s_file),
            mesh_file=model.mesh_file,
        )
        n_verts = built.vertices.shape[0]
        n_cells = built.cell_conn.shape[0]
        tpl = None
    else:
        tpl = _resolve_template(s_file, template, mesh)
        fld_mesh = parse_fld(str(tpl))
        n_verts = fld_mesh["n_vertices"]
        if not n_verts:
            raise ValueError(f"Template {tpl} has no vertices")
        temp = vertex_temperature_field(
            model,
            fld_mesh["vertices"],
            fld_mesh["cell_conn"],
            fld_mesh["material"],
            fld_mesh.get("volume_names", []),
        )
        fields = default_initial_fields(n_verts, temp, ambient=model.ambient_temp)
        compose_fld(str(tpl), out_path, fields, s_text)

    if cycle is not None:
        patch_cycle(out_path, cycle)

    return {
        "s_file": str(s_file),
        "template": str(tpl) if tpl else None,
        "xemt": xemt_path,
        "out": out_path,
        "dims": (model.ni, model.nj, model.nk),
        "vertices": n_verts,
        "cells": n_cells if xemt_path else parse_fld(out_path)["n_cells"],
        "temp_range": (float(temp.min()), float(temp.max())),
        "cycle": cycle,
        "summary": summarize_sdat(model),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert scFLOW SDAT (.s) to CRDL-FLD (.fld)")
    parser.add_argument("s_file", help="Input .s solver definition")
    parser.add_argument("-o", "--output", help="Output .fld path")
    parser.add_argument("--xemt", metavar="FILE", help="EMT file — build mesh without template FLD")
    parser.add_argument(
        "--template",
        help="Reference FLD for scPOST header/geometry (with --xemt: matched by cell count)",
    )
    parser.add_argument("--mesh", help="Mesh FLD alias for --template")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--no-cycle-patch", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    s_path = Path(args.s_file)
    if not s_path.is_file():
        print(f"Error: {s_path} not found", file=sys.stderr)
        return 1
    if args.xemt and not Path(args.xemt).is_file():
        print(f"Error: {args.xemt} not found", file=sys.stderr)
        return 1

    out_path = args.output or str(s_path.parent / f"{s_path.stem}_0.fld")
    cycle = None if args.no_cycle_patch else args.cycle

    try:
        info = convert_s_to_fld(
            str(s_path),
            out_path,
            xemt_path=args.xemt,
            template=args.template,
            mesh=args.mesh,
            cycle=cycle,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Written: {info['out']}")
    if info["template"]:
        print(f"Template: {info['template']}")
    if info["xemt"]:
        print(f"XEMT: {info['xemt']}")
    print(f"Mesh: {info['vertices']} vertices, {info['cells']} cells")
    print(f"TEMP range: {info['temp_range'][0]:.4g} .. {info['temp_range'][1]:.4g}")
    if info["cycle"] is not None:
        print(f"Cycle: {info['cycle']}")
    if args.verbose:
        print("SDAT summary:", info["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
