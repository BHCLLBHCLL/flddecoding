#!/usr/bin/env python3
"""
SDAT (.s) solver definition parser for Software Cradle scFLOW / SCTpre.

The .s file is a text SDAT stream containing structured mesh spacing (CXYZ),
part/material definitions, boundary regions, and initial / boundary conditions.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class SdatPart:
    """Volume part (material region)."""
    part_id: int
    material_id: int
    fraction: float
    name: str
    box: Optional[tuple[int, int, int, int, int, int]] = None  # i1,i2,j1,j2,k1,k2 1-based


@dataclass
class SdatInit:
    """Initial condition on a region name."""
    variable: str
    value: float
    region: str


@dataclass
class SdatFlux:
    """Boundary flux (velocity / pressure) on a region."""
    kind: str  # velocity | pressure
    values: list[float]
    region: str


@dataclass
class SdatModel:
    """Parsed SDAT model from a .s file."""
    raw_text: str
    basename: str = ""
    mesh_file: str = ""  # RO line, typically .r
    title: str = ""
    ni: int = 0
    nj: int = 0
    nk: int = 0
    cxyz: list[np.ndarray] = field(default_factory=list)
    parts: list[SdatPart] = field(default_factory=list)
    init_regions: list[SdatInit] = field(default_factory=list)
    flux_regions: list[SdatFlux] = field(default_factory=list)
    cycle_max: int = 0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ambient_temp: float = 20.0
    region_names: list[str] = field(default_factory=list)


def _read_floats(line: str) -> list[float]:
    return [float(x) for x in line.split()]


def _read_ints(line: str) -> list[int]:
    return [int(x) for x in line.split()]


def parse_sdat(text: str) -> SdatModel:
    """Parse SDAT text into :class:`SdatModel`."""
    model = SdatModel(raw_text=text)
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("!"):
            i += 1
            continue

        if line == "POST":
            i += 1
            if i < len(lines):
                model.basename = lines[i].strip()
                i += 1
            if i < len(lines) and lines[i].strip() == "RO":
                i += 1
                if i < len(lines):
                    model.mesh_file = lines[i].strip()
                    i += 1
            continue

        if line.startswith("Basic"):
            model.title = line
            i += 1
            continue

        if line == "EQUA":
            i += 1
            continue

        if line == "GRAV":
            i += 1
            if i < len(lines):
                vals = _read_floats(lines[i])
                if len(vals) >= 3:
                    model.gravity = (vals[0], vals[1], vals[2])
                if len(vals) >= 4:
                    model.ambient_temp = vals[3]
            i += 1
            continue

        if line == "CYCT":
            i += 1
            if i < len(lines):
                vals = _read_ints(lines[i])
                if vals:
                    model.cycle_max = vals[1] if len(vals) > 1 else vals[0]
            i += 1
            continue

        if line == "CXYZ":
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if sub in ("PARTS", "REGION", "/", ""):
                    break
                if sub == "0":
                    i += 1
                    block: list[float] = []
                    while i < len(lines):
                        row = lines[i].strip()
                        if not row or row == "0":
                            break
                        if row in ("PARTS", "REGION", "/"):
                            break
                        try:
                            block.extend(_read_floats(row))
                        except ValueError:
                            break
                        i += 1
                    if block:
                        model.cxyz.append(np.array(block, dtype=np.float64))
                    if i < len(lines) and lines[i].strip() == "0":
                        continue
                    break
                i += 1
            continue

        if line == "PARTS":
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if sub in ("REGION", "/", "") or sub.startswith("REGION"):
                    break
                if sub.startswith("!"):
                    i += 1
                    continue
                try:
                    box_vals = _read_ints(sub)
                except ValueError:
                    box_vals = []
                if len(box_vals) == 6 and model.parts:
                    model.parts[-1].box = tuple(box_vals)
                    i += 1
                    continue
                m = re.match(r"^\s*(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+(.*)$", sub)
                if m and re.search(r"[A-Za-z_(@]", m.group(4)):
                    model.parts.append(
                        SdatPart(
                            part_id=int(m.group(1)),
                            material_id=int(m.group(2)),
                            fraction=float(m.group(3)),
                            name=m.group(4).strip(),
                        )
                    )
                    i += 1
                    continue
                i += 1
            continue

        if line == "REGION":
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if sub in ("INIT_REGION", "FLUX_REGION", "/", ""):
                    break
                if sub.endswith("!") or (sub and not sub.startswith("@") and "!" in sub):
                    name = sub.split("!")[0].strip()
                    if name:
                        model.region_names.append(name)
                i += 1
            continue

        if line == "INIT_REGION":
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if sub in ("FLUX_REGION", "AMOM_REGION"):
                    break
                if sub in ("/", ""):
                    i += 1
                    continue
                if sub in ("TEMP", "PRES", "VECT"):
                    var = sub
                    i += 1
                    val = float(lines[i].strip())
                    i += 1
                    region = lines[i].strip()
                    model.init_regions.append(SdatInit(var, val, region))
                    i += 1
                else:
                    i += 1
            continue

        if line == "FLUX_REGION":
            i += 1
            while i < len(lines):
                sub = lines[i].strip()
                if sub in ("AMOM_REGION", "AUTOFIXP", "/", ""):
                    break
                if sub.startswith("velocity") or sub.startswith("pressure"):
                    kind = "velocity" if sub.startswith("velocity") else "pressure"
                    i += 1
                    vals = _read_floats(lines[i])
                    i += 2  # skip integer flags line
                    region = ""
                    while i < len(lines):
                        r = lines[i].strip()
                        if r == "/" or r.startswith("AMOM") or r == "":
                            break
                        if r and not r[0].isdigit() and not r.startswith("-"):
                            region = r.split("!")[0].strip()
                            i += 1
                            break
                        i += 1
                    if region:
                        model.flux_regions.append(SdatFlux(kind, vals, region))
                else:
                    i += 1
            continue

        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            ni, nj, nk = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if ni > 1 and nj > 1 and nk > 1:
                model.ni, model.nj, model.nk = ni, nj, nk
        i += 1

    return model


def parse_sdat_file(path: str) -> SdatModel:
    text = Path(path).read_text(encoding="utf-8-sig")
    model = parse_sdat(text)
    if not model.basename:
        model.basename = Path(path).stem
    return model


def build_structured_coords(model: SdatModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) coordinate axes from CXYZ."""
    if len(model.cxyz) != 3:
        raise ValueError("CXYZ must contain three axis blocks")
    return model.cxyz[0], model.cxyz[1], model.cxyz[2]


def vertex_temperature_field(
    model: SdatModel,
    vertices: np.ndarray,
    cell_conn: np.ndarray,
    material: np.ndarray,
    part_names: list[str],
) -> np.ndarray:
    """Build vertex-centred TEMP from INIT_REGION entries."""
    temp = np.full(len(vertices), model.ambient_temp, dtype=np.float64)
    name_to_mat: dict[str, int] = {}
    for part in model.parts:
        name_to_mat[part.name] = part.material_id
    for init in model.init_regions:
        if init.variable != "TEMP":
            continue
        mat_id = name_to_mat.get(init.region)
        if mat_id is None:
            for p in model.parts:
                if p.name == init.region or init.region in p.name:
                    mat_id = p.material_id
                    break
        if mat_id is None:
            continue
        mask = material == mat_id
        if not mask.any():
            continue
        used_nodes = set()
        for nodes in cell_conn[mask]:
            used_nodes.update(int(n) - 1 for n in nodes)
        for nid in used_nodes:
            if 0 <= nid < len(temp):
                temp[nid] = init.value
    return temp


def summarize_sdat(model: SdatModel) -> dict[str, Any]:
    return {
        "basename": model.basename,
        "mesh_file": model.mesh_file,
        "title": model.title,
        "dims": (model.ni, model.nj, model.nk),
        "parts": [(p.name, p.material_id, p.box) for p in model.parts],
        "init": [(x.variable, x.value, x.region) for x in model.init_regions],
        "flux": [(x.kind, x.region) for x in model.flux_regions],
        "cxyz_lens": [len(a) for a in model.cxyz],
        "cycle_max": model.cycle_max,
    }
