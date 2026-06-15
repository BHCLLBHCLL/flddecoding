#!/usr/bin/env python3
"""FLD format inspector — prints section layout and mesh summary."""

import argparse
import sys
from pathlib import Path

from fld_model import (
    describe_fld_sections,
    open_fld_buffer,
    parse_fld,
    validate_scpost_geometry,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect CRDL-FLD section layout")
    parser.add_argument(
        "fld_file",
        nargs="?",
        default="tests/ex1_e_100.fld",
        help="Input FLD file",
    )
    parser.add_argument(
        "--validate-scpost",
        action="store_true",
        help="Check geometry preambles and volume block for known scPOST issues",
    )
    args = parser.parse_args()

    path = Path(args.fld_file)
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

    if args.validate_scpost:
        with open_fld_buffer(str(path)) as data:
            issues = validate_scpost_geometry(data)
        if issues:
            print("\nscPOST geometry issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nscPOST geometry check: OK")

    print()
    print("Sections:")
    for sec in describe_fld_sections(str(path)):
        print(f"  {sec['offset_hex']} ({sec['size']:6} B) {sec['name']}")


if __name__ == "__main__":
    main()
