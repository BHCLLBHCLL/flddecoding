#!/usr/bin/env python3
"""Parse scFLOW / SCTpre EMT (.xemt) material and part metadata."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class XemtMaterial:
    no: int
    name: str


@dataclass
class XemtPart:
    no: int
    name: str
    material_id: int


@dataclass
class XemtModel:
    materials: list[XemtMaterial] = field(default_factory=list)
    parts: list[XemtPart] = field(default_factory=list)
    fluid_name: str = ""
    fluid_mat: int = 1


def parse_xemt(path: str) -> XemtModel:
    root = ET.parse(path).getroot()
    model = XemtModel()
    for mat in root.findall(".//Material/mat"):
        model.materials.append(
            XemtMaterial(int(mat.attrib["no"]), mat.attrib.get("name", ""))
        )
    fluid = root.find(".//Parts/fluid")
    if fluid is not None:
        model.fluid_name = fluid.attrib.get("name", "")
        model.fluid_mat = int(fluid.attrib.get("mat", "1"))
    for part in root.findall(".//part"):
        model.parts.append(
            XemtPart(
                int(part.attrib["no"]),
                part.attrib.get("name", ""),
                int(part.attrib.get("mat", "1")),
            )
        )
    for panel in root.findall(".//panel"):
        model.parts.append(
            XemtPart(
                int(panel.attrib["no"]),
                panel.attrib.get("name", ""),
                int(panel.attrib.get("mat", "1")),
            )
        )
    return model


def parse_xemt_file(path: str) -> XemtModel:
    return parse_xemt(path)


def part_material_map(model: XemtModel) -> dict[str, int]:
    """Map part name to material id (xemt + fluid domain)."""
    out: dict[str, int] = {}
    if model.fluid_name:
        out[model.fluid_name] = model.fluid_mat
    for p in model.parts:
        out[p.name] = p.material_id
    return out


def volume_labels(model: XemtModel) -> tuple[str, str, str, str]:
    """PARTS1, PARTS2, domain label, solid label for CGNS element sections."""
    domain = model.fluid_name or "Domain"
    solid = ""
    for p in model.parts:
        if p.material_id != model.fluid_mat and p.name != domain:
            solid = p.name
            break
    if not solid and model.parts:
        for p in model.parts:
            if p.material_id != model.fluid_mat:
                solid = p.name
                break
    if not solid:
        solid = "PARTS2"
    if solid == domain:
        solid = "SOLID"
    return "PARTS1", "PARTS2", domain, solid
