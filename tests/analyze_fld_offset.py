#!/usr/bin/env python3
"""Compare FLD layout at a byte offset (scPOST error diagnosis)."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fld_model import find_section, iter_data_blocks, open_fld_buffer, read_i32_be, section_end


def dump_at(data: bytes, pos: int, n: int = 48) -> None:
    chunk = data[pos : pos + n]
    print(f"  at {pos}:")
    print(f"    hex: {chunk[:32].hex()}")
    if pos >= 4:
        print(f"    i32@pos: {read_i32_be(data, pos)}")
    if pos + 12 <= len(data):
        print(f"    block hdr: 12,{read_i32_be(data, pos+4)} bc={read_i32_be(data, pos+4)}")


def analyze(path: str, err_pos: int = 1928) -> None:
    print(f"=== {path} ===")
    with open_fld_buffer(path) as data:
        print(f"size {len(data)}")
        for name in [
            "FileRevision", "OverlapStart_0", "LS_CoordinateSystem", "Pressure",
            "Temperature", "LS_STREAMcoc", "LS_Nodes", "LS_SurfaceGeometryArray",
        ]:
            off = find_section(data, name)
            if off >= 0:
                print(f"  {name}: {off} len {section_end(data, off) - off}")

        dump_at(data, err_pos - 8)
        dump_at(data, err_pos)

        # Pressure section blocks
        ps = find_section(data, "Pressure")
        if ps >= 0:
            blocks = list(iter_data_blocks(data, ps, section_end(data, ps)))
            print(f"  Pressure blocks ({len(blocks)}):")
            for i, (p, bc) in enumerate(blocks[:5]):
                print(f"    [{i}] payload@{p} bc={bc} pre={data[p-8:p].hex() if p>=8 else '?'}")

        # LS_CoordinateSystem inner
        cs = find_section(data, "LS_CoordinateSystem")
        if cs >= 0:
            pe = cs + 40
            print(f"  LS_CoordinateSystem inner start @ {pe}: {data[pe:pe+24].hex()}")


if __name__ == "__main__":
    err = int(sys.argv[1]) if len(sys.argv) > 1 else 1928
    for p in ["tests/ex4_e_63.fld", "tests/ex4_e_from_sxemt.fld"]:
        if Path(p).exists():
            analyze(p, err)
