#!/usr/bin/env python3
"""Probe byte offset from scPOST error."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fld_model import (
    describe_fld_sections,
    find_section,
    iter_data_blocks,
    open_fld_buffer,
    read_i32_be,
    section_end,
)

err = int(sys.argv[1]) if len(sys.argv) > 1 else 22981767
path = sys.argv[2] if len(sys.argv) > 2 else "tests/ex3_e_from_sxemt.fld"

with open_fld_buffer(path) as data:
    print(f"=== {path} err={err} size={len(data)} ===")
    for s in describe_fld_sections(path):
        start = int(s["offset_hex"], 16)
        end = start + s["size"]
        mark = " ** ERROR **" if start <= err < end else ""
        print(f"  {s['offset_hex']} ({start}-{end}) {s['name']}{mark}")
        if start <= err < end:
            print(f"    offset in section: +{err - start}")

    sec = find_section(data, "LS_Elements")
    if sec >= 0:
        blocks = list(iter_data_blocks(data, sec, section_end(data, sec)))
        print(f"LS_Elements blocks: {blocks}")
        for i, (p, bc) in enumerate(blocks):
            end_p = p + bc
            if p <= err < end_p:
                rel = err - p
                print(f"  in block[{i}] rel={rel} cell={rel // 32} corner={(rel % 32) // 4}")

    print("i32 window at error:")
    for off in range(err - 16, err + 20, 4):
        if 0 <= off < len(data) - 3:
            v = read_i32_be(data, off)
            print(f"  {off}: {v}")
