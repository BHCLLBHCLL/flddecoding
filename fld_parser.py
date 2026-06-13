#!/usr/bin/env python3
"""FLD format inspector — prints section layout and mesh summary."""

import sys
from pathlib import Path

from fld_model import describe_fld_sections, parse_fld


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/ex1_e_100.fld")
    if not path.is_file():
        print(f"Error: {path} not found")
        sys.exit(1)

    mesh = parse_fld(str(path))
    print(f"File: {path}")
    print(f"Size: {mesh['file_size']} bytes")
    print(f"Vertices: {mesh['n_vertices']}")
    print(f"Cells: {mesh['n_cells']}")
    print(f"Faces: {len(mesh['faces'])}")
    print(f"Volume names: {mesh.get('volume_names', [])}")
    print(f"Fields: {sorted(mesh['fields'].keys())}")
    print(f"BC groups: {len(mesh.get('bc_plan', []))}")
    print()
    print("Sections:")
    for sec in describe_fld_sections(str(path)):
        print(f"  {sec['offset_hex']} ({sec['size']:6} B) {sec['name']}")


if __name__ == "__main__":
    main()
