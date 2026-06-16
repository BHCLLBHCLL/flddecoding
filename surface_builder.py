#!/usr/bin/env python3
"""Build vendor-style LS_SurfaceGeometryArray face lists and BC plans."""

from typing import Callable

import numpy as np

FaceQuad = tuple[int, int, int, int]
FaceRec = tuple[FaceQuad, int]  # quad, 0-based cell flat index


def _flat_cell(i: int, j: int, k: int, ni: int, nj: int) -> int:
    return i + j * ni + k * ni * nj


def _classify_interface(
    m1: int, m2: int, p1: int, p2: int,
) -> str | None:
    """Return ENT category for an internal face, or None if not a BC face."""
    if m1 != m2:
        if m1 == 1 or m2 == 1:
            return "@UNDEFINEDENTF"
        if min(m1, m2) >= 3 and max(m1, m2) >= 4:
            return "@UNDEFINEDENTX"
        return "@UNDEFINEDENTS"
    if p1 != p2:
        return "@UNDEFINEDENTS"
    return None


def build_vendor_surfaces(
    ni: int, nj: int, nk: int,
    cmat: np.ndarray,
    cpart: np.ndarray,
    face_quad: Callable[..., FaceQuad],
) -> tuple[list[FaceRec], dict[str, list[FaceRec]]]:
    """
    Extract boundary / interface quads grouped by vendor BC category.

    *face_quad(ci, cj, ck, axis, side)* returns four vertex ids.
    """
    cats: dict[str, list[FaceRec]] = {
        "@UNDEFINEDENTB": [],
        "@UNDEFINEDENTF": [],
        "@UNDEFINEDENTS": [],
        "@UNDEFINEDENTX": [],
        "@UNDEFINEDMOM": [],
        "@UNDEFINEDVFWL": [],
        "PARTS": [],
        "SURFACE": [],
        "Xmin": [],
        "Xmax": [],
        "Ymin": [],
        "Ymax": [],
        "Zmin": [],
        "Zmax": [],
    }

    def add(cat: str, quad: FaceQuad, ci: int, cj: int, ck: int) -> None:
        cats[cat].append((quad, _flat_cell(ci, cj, ck, ni, nj)))

    # Domain box faces.
    for k in range(nk):
        for j in range(nj):
            add("Xmin", face_quad(0, j, k, "x", 0), 0, j, k)
            add("Xmax", face_quad(ni - 1, j, k, "x", 1), ni - 1, j, k)
    for k in range(nk):
        for i in range(ni):
            add("Ymin", face_quad(i, 0, k, "y", 0), i, 0, k)
            add("Ymax", face_quad(i, nj - 1, k, "y", 1), i, nj - 1, k)
    for j in range(nj):
        for i in range(ni):
            add("Zmin", face_quad(i, j, 0, "z", 0), i, j, 0)
            add("Zmax", face_quad(i, j, nk - 1, "z", 1), i, j, nk - 1)

    # Internal faces: classify by material / part change.
    for k in range(nk):
        for j in range(nj):
            for i in range(ni - 1):
                m1, m2 = int(cmat[i, j, k]), int(cmat[i + 1, j, k])
                p1, p2 = int(cpart[i, j, k]), int(cpart[i + 1, j, k])
                quad = face_quad(i, j, k, "x", 1)
                ent = _classify_interface(m1, m2, p1, p2)
                if ent:
                    add(ent, quad, i, j, k)
                if p1 != p2 or m1 != m2:
                    add("PARTS", quad, i, j, k)
                    add("SURFACE", quad, i, j, k)
            for i in range(ni):
                if j < nj - 1:
                    m1, m2 = int(cmat[i, j, k]), int(cmat[i, j + 1, k])
                    p1, p2 = int(cpart[i, j, k]), int(cpart[i, j + 1, k])
                    quad = face_quad(i, j, k, "y", 1)
                    ent = _classify_interface(m1, m2, p1, p2)
                    if ent:
                        add(ent, quad, i, j, k)
                    if p1 != p2 or m1 != m2:
                        add("PARTS", quad, i, j, k)
                        add("SURFACE", quad, i, j, k)
                if k < nk - 1:
                    m1, m2 = int(cmat[i, j, k]), int(cmat[i, j, k + 1])
                    p1, p2 = int(cpart[i, j, k]), int(cpart[i, j, k + 1])
                    quad = face_quad(i, j, k, "z", 1)
                    ent = _classify_interface(m1, m2, p1, p2)
                    if ent:
                        add(ent, quad, i, j, k)
                    if p1 != p2 or m1 != m2:
                        add("PARTS", quad, i, j, k)
                        add("SURFACE", quad, i, j, k)

    # MOM on all domain walls.
    for name in ("Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax"):
        for quad, flat in cats[name]:
            cats["@UNDEFINEDMOM"].append((quad, flat))

    # Vendor ex4 meta has zero ENTB faces; do not populate @UNDEFINEDENTB.
    for rec in cats["PARTS"]:
        flat = rec[1]
        k = flat // (ni * nj)
        rem = flat % (ni * nj)
        j = rem // ni
        i = rem % ni
        if int(cmat[i, j, k]) == 1:
            cats["@UNDEFINEDVFWL"].append(rec)

    seg1_order = [
        "@UNDEFINEDENTB", "@UNDEFINEDENTF", "@UNDEFINEDENTS", "@UNDEFINEDENTX",
        "@UNDEFINEDMOM", "@UNDEFINEDVFWL", "PARTS", "SURFACE",
        "Xmax", "Xmin", "Ymax", "Ymin", "Zmax", "Zmin",
    ]
    seg1: list[FaceRec] = []
    for key in seg1_order:
        seg1.extend(cats[key])

    return seg1, cats


def vendor_bc_plan_from_categories(
    cats: dict[str, list[FaceRec]],
    material: np.ndarray,
    ymax_name: str = "Ymax",
) -> tuple[list[FaceQuad], list[tuple[str, int, int]]]:
    """Build face list and BC plan matching vendor FLD/CGNS layout (ex4 multi-mat)."""
    seg1_order = [
        "@UNDEFINEDENTB", "@UNDEFINEDENTF", "@UNDEFINEDENTS", "@UNDEFINEDENTX",
        "@UNDEFINEDMOM", "@UNDEFINEDVFWL", "PARTS", "SURFACE",
        "Xmax", "Xmin", ymax_name, "Ymin", "Zmax", "Zmin",
    ]
    seg1_names = [
        "@UNDEFINEDENTB", "@UNDEFINEDENTF", "@UNDEFINEDENTS", "@UNDEFINEDENTX",
        "@UNDEFINEDMOM", "@UNDEFINEDVFWL", "PARTS", "SURFACE",
        "Xmax", "Xmin", ymax_name, "Ymin", "Zmax", "Zmin",
    ]

    faces: list[FaceQuad] = []
    bc_plan: list[tuple[str, int, int]] = []
    for name, key in zip(seg1_names, seg1_order):
        quads = [q for q, _ in cats[key]]
        if quads:
            bc_plan.append((name, len(faces), len(quads)))
            faces.extend(quads)

    mats = sorted({int(m) for m in np.unique(material)})
    seg2_faces: list[FaceQuad] = []
    seg2_plan: list[tuple[str, int, int]] = []
    seg2_base = len(faces)
    for m in mats:
        groups = [
            (f"PARTS(MAT{m})", [q for q, flat in cats["PARTS"] if int(material[flat]) == m]),
            (f"SURFACE(MAT{m})", [q for q, flat in cats["SURFACE"] if int(material[flat]) == m]),
        ]
        ents = [q for q, flat in cats["@UNDEFINEDENTS"] if int(material[flat]) == m]
        if ents and m in (2, 3, 5, 6):
            groups.append((f"@UNDEFINEDENTS(MAT{m})", ents))
        for name, quads in groups:
            if quads:
                seg2_plan.append((name, seg2_base + len(seg2_faces), len(quads)))
                seg2_faces.extend(quads)

    faces.extend(seg2_faces)
    bc_plan.extend(seg2_plan)
    return faces, bc_plan


def surface_meta_counts(cats: dict[str, list[FaceRec]], ymax_name: str = "Ymax") -> np.ndarray:
    """18-int meta block for LS_SurfaceGeometryArray (vendor ex4 layout)."""
    return np.array([
        0, 0, 0,
        len(cats["@UNDEFINEDENTF"]),
        len(cats["@UNDEFINEDENTS"]),
        len(cats["@UNDEFINEDENTX"]),
        0,
        len(cats["@UNDEFINEDMOM"]),
        0,
        len(cats["@UNDEFINEDVFWL"]),
        len(cats["PARTS"]),
        len(cats["SURFACE"]),
        len(cats["Xmax"]),
        len(cats["Xmin"]),
        len(cats[ymax_name]),
        len(cats["Ymin"]),
        len(cats["Zmax"]),
        len(cats["Zmin"]),
    ], dtype=np.int32)


def surface_seg1_order_ex3(ymax_name: str = "Ymax") -> list[str]:
    """Face block order for ex3 (16-slot meta); must match meta field order."""
    return [
        "@UNDEFINEDENTB",
        "@UNDEFINEDENTF",
        "@UNDEFINEDENTS",
        "@UNDEFINEDENTX",
        "@UNDEFINEDVFWL",
        "@UNDEFINEDMOM",
        "Zmax",
        "Zmin",
        "PARTS",
        "SURFACE",
        "Xmax",
        "Xmin",
        ymax_name,
        "Ymin",
    ]


def surface_meta_counts_ex3(cats: dict[str, list[FaceRec]], ymax_name: str = "Ymax") -> np.ndarray:
    """16-int meta block for LS_SurfaceGeometryArray (vendor ex3 layout)."""
    return np.array([
        0, 0,
        len(cats["@UNDEFINEDENTB"]),
        len(cats["@UNDEFINEDENTF"]),
        len(cats["@UNDEFINEDENTS"]),
        len(cats["@UNDEFINEDENTX"]),
        len(cats["@UNDEFINEDVFWL"]),
        len(cats["@UNDEFINEDMOM"]),
        len(cats["Zmax"]),
        len(cats["Zmin"]),
        len(cats["PARTS"]),
        len(cats["SURFACE"]),
        len(cats["Xmax"]),
        len(cats["Xmin"]),
        len(cats[ymax_name]),
        len(cats["Ymin"]),
    ], dtype=np.int32)
