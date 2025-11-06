#!/usr/bin/env python3
"""prepare_hard_negatives.py

Clean the CSVs in data_cleaning/hard_negatives/ to keep only the columns used in
`data_complete` and combine them into one `hard_negatives.csv` file.

Kept columns (in order):
- long_east_deg, lat_north_deg, altitude_m, radius_m, slope_deg, roughness_rms_m

If slope_deg or roughness_rms_m are missing they will be created and filled with NaN.

Usage:
    python3 prepare_hard_negatives.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
IN_DIR = SCRIPT_DIR
OUT_PATH = IN_DIR / "hard_negatives.csv"

KEEP_ORDER = ["long_east_deg", "lat_north_deg", "altitude_m", "radius_m", "slope_deg", "roughness_rms_m"]

# Mapping from possible original column names (strip whitespace and case-insensitive)
SRC_TO_DST = {
    "long_east": "long_east_deg",
    "longevast": "long_east_deg",  # fallback typo-safe
    "lat_north": "lat_north_deg",
    "topography": "altitude_m",
    "planet_rad": "radius_m",
    "planet radius": "radius_m",
    "radius": "radius_m",
    "slope_deg": "slope_deg",
    "roughness_rms_m": "roughness_rms_m",
}


def normalize_columns(cols):
    """Return a mapping from actual col name to normalized key (lower/stripped)"""
    mapping = {}
    for c in cols:
        key = c.strip().lower()
        mapping[c] = key
    return mapping


def pick_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    col_key = {c: c.strip().lower() for c in df.columns}

    rename_map = {}
    for orig, key in col_key.items():
        if key in SRC_TO_DST:
            rename_map[orig] = SRC_TO_DST[key]
        else:
            # try to match by removing spaces/underscores
            k2 = key.replace(" ", "_")
            if k2 in SRC_TO_DST:
                rename_map[orig] = SRC_TO_DST[k2]

    # Apply rename for columns we know
    df = df.rename(columns=rename_map)

    # Ensure all KEEP_ORDER columns exist; create if missing
    for col in KEEP_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    # Select only desired columns in order
    df_out = df[KEEP_ORDER].copy()
    return df_out


def main():
    files = sorted(IN_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {IN_DIR}")
        return 1

    tables = []
    for p in files:
        print(f"Reading {p.name}...")
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  ERROR reading {p}: {e}")
            continue
        df_clean = pick_and_rename(df)
        # Add a source column to track origin (optional)
        df_clean["source_file"] = p.name
        tables.append(df_clean)

    if not tables:
        print("No tables processed")
        return 1

    combined = pd.concat(tables, ignore_index=True)
    # Remove rows that are entirely NaN for coordinate fields
    combined = combined.dropna(subset=["long_east_deg", "lat_north_deg", "altitude_m", "radius_m"], how='any')

    # Save combined file
    combined.to_csv(OUT_PATH, index=False)
    print(f"WROTE: {OUT_PATH} ({len(combined):,} rows)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
