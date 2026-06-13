#!/usr/bin/env python3
"""Verify s2fld conversion: mesh preserved, initial TEMP from .s, SDAT embedded."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fld_model import open_fld_buffer, find_section, section_end, iter_data_blocks, parse_fld
from s2fld import convert_s_to_fld
from s_model import parse_sdat_file, vertex_temperature_field


def _read_sfile_text(fld_path: str) -> str:
    data = Path(fld_path).read_bytes()
    sec_start = find_section(data, "LS_SFile")
    sec_end = section_end(data, sec_start)
    blocks = list(iter_data_blocks(data, sec_start, sec_end))
    if len(blocks) < 2:
        return ""
    p, bc = blocks[1]
    return data[p : p + bc].decode("utf-8", errors="replace").lstrip("\x00").lstrip("\ufeff")


def test_s2fld_ex1_e() -> list[str]:
    errors: list[str] = []
    s_path = ROOT / "tests" / "ex1_e.s"
    template = ROOT / "tests" / "ex1_e_100.fld"
    out_path = ROOT / "tests" / "ex1_e_0_from_s.fld"

    convert_s_to_fld(
        str(s_path),
        str(out_path),
        template=str(template),
        cycle=0,
    )

    tpl = parse_fld(str(template))
    gen = parse_fld(str(out_path))
    model = parse_sdat_file(str(s_path))

    if gen["n_vertices"] != tpl["n_vertices"]:
        errors.append("vertex count mismatch")
    if gen["n_cells"] != tpl["n_cells"]:
        errors.append("cell count mismatch")
    if not np.array_equal(gen["cell_conn"], tpl["cell_conn"]):
        errors.append("cell connectivity mismatch")
    if not np.array_equal(gen["material"], tpl["material"]):
        errors.append("material mismatch")
    if not np.allclose(gen["vertices"], tpl["vertices"]):
        errors.append("vertex coordinates mismatch")

    expected_temp = vertex_temperature_field(
        model,
        tpl["vertices"],
        tpl["cell_conn"],
        tpl["material"],
        tpl.get("volume_names", []),
    )
    if not np.allclose(gen["fields"]["TEMP"], expected_temp):
        errors.append("TEMP field mismatch vs .s INIT_REGION")

    s_text = s_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").lstrip("\ufeff")
    embedded = _read_sfile_text(str(out_path)).replace("\r\n", "\n")
    if "INIT_REGION" not in embedded or "Domain(cuboid)" not in embedded:
        errors.append("LS_SFile missing SDAT content")
    if "ex1_e" not in embedded:
        errors.append("LS_SFile missing basename")

    temps = gen["fields"]["TEMP"]
    if not np.any(np.isclose(temps, 20.0)) or not np.any(np.isclose(temps, 100.0)):
        errors.append("expected 20C and 100C regions in TEMP")

    return errors


def main() -> int:
    errors = test_s2fld_ex1_e()
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: s2fld ex1_e conversion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
