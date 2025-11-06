#!/usr/bin/env python3
"""extract_attributes.py

Read every CSV in data_cleaning/data (except mars_landing_boxes_1deg.csv),
keep only columns long_East -> long_east, lat_North -> lat_north, topography -> altitude,
and save outputs with the same filenames into data_clean_attributes/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "data"
DST_DIR = BASE_DIR / "data_clean_attributes"
EXCLUDE = {"mars_landing_boxes_1deg.csv"}

KEEP_MAP = [
    ("long_East", "long_east"),
    ("lat_North", "lat_north"),
    ("topography", "altitude"),
]


def find_col(cols, target: str):
    """Return actual column name in cols matching target.

    Try exact match first, then case-insensitive match.
    """
    if target in cols:
        return target
    lower_map = {c.lower(): c for c in cols}
    return lower_map.get(target.lower())


def process_file(path: Path) -> bool:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: failed to read {path.name}: {e}", file=sys.stderr)
        return False

    cols = df.columns.tolist()
    mapping = {}
    missing = []
    for src, dst in KEEP_MAP:
        found = find_col(cols, src)
        if found:
            mapping[found] = dst
        else:
            missing.append(src)

    if missing:
        print(f"WARNING: skipping {path.name} — missing columns: {missing}")
        return False

    out = df[list(mapping.keys())].rename(columns=mapping)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DST_DIR / path.name
    out.to_csv(out_path, index=False)
    print(f"WROTE: {out_path}")
    return True


def main() -> int:
    if not SRC_DIR.exists():
        print(f"Source directory not found: {SRC_DIR}", file=sys.stderr)
        return 2

    files = sorted([p for p in SRC_DIR.iterdir() if p.suffix == ".csv"])
    if not files:
        print(f"No CSV files found in {SRC_DIR}")
        return 0

    success = 0
    for p in files:
        if p.name in EXCLUDE:
            print(f"SKIP (excluded): {p.name}")
            continue
        ok = process_file(p)
        if ok:
            success += 1

    print(f"Done — processed {success}/{len(files)} files (excluded: {list(EXCLUDE)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
