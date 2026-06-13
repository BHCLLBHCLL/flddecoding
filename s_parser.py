#!/usr/bin/env python3
"""SDAT (.s) format inspector."""

import json
import sys
from pathlib import Path

from s_model import parse_sdat_file, summarize_sdat


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/ex1_e.s")
    if not path.is_file():
        print(f"Error: {path} not found")
        return 1
    model = parse_sdat_file(str(path))
    info = summarize_sdat(model)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
