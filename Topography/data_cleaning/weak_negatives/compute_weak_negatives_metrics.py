#!/usr/bin/env python3
"""compute_weak_negatives_metrics.py

Process weak_negatives CSVs to:
1. Compute slope_deg and roughness_rms_m using the same algorithms as mission data
2. Combine all weak_negative_*.csv files into one weak_negatives.csv

Usage:
    python3 compute_weak_negatives_metrics.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to import shared_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_utils import (
    meters_per_degree,
    lonlat_to_local_xy,
    fit_plane,
    plane_slope_from_coeffs,
    plane_residuals,
    rms_roughness
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "weak_negatives.csv"

# Processing parameters (same as mission data)
BASELINE_M = 1000.0
MIN_NEIGHBORS = 8


def compute_metrics(df, baseline_m=BASELINE_M, min_neighbors=MIN_NEIGHBORS):
    """
    Compute both slope and roughness for each point.
    
    Returns DataFrame with added slope_deg and roughness_rms_m columns.
    """
    lon = df["long_east_deg"].to_numpy()
    lat = df["lat_north_deg"].to_numpy()
    alt = df["altitude_m"].to_numpy()
    rad = df["radius_m"].to_numpy()
    
    slope = np.full(len(df), np.nan)
    rough = np.full(len(df), np.nan)
    
    print(f"  Computing metrics for {len(df):,} points...")
    for i in range(len(df)):
        if i % 5000 == 0 and i > 0:
            print(f"    Progress: {i:,}/{len(df):,} ({100*i/len(df):.1f}%)")
        
        lon0, lat0, z0, r0 = lon[i], lat[i], alt[i], rad[i]
        
        # Calculate angular search window
        m_per_deg_lon0, m_per_deg_lat0 = meters_per_degree(lat0, r0)
        dlon_deg = baseline_m / m_per_deg_lon0
        dlat_deg = baseline_m / m_per_deg_lat0
        
        # Find neighbors
        mask = (
            (np.abs(lon - lon0) <= dlon_deg) &
            (np.abs(lat - lat0) <= dlat_deg)
        )
        idx = np.where(mask)[0]
        
        if len(idx) < min_neighbors:
            continue
        
        # Convert to local coordinates
        x, y = lonlat_to_local_xy(lon[idx], lat[idx], lon0, lat0, r0)
        z = alt[idx]
        
        try:
            # Fit plane and compute both metrics
            a, b, c = fit_plane(x, y, z)
            slope[i] = plane_slope_from_coeffs(a, b)
            resid = plane_residuals(x, y, z, a, b, c)
            rough[i] = rms_roughness(resid)
        except Exception:
            pass
    
    df["slope_deg"] = slope
    df["roughness_rms_m"] = rough
    return df


def process_file(path: Path):
    """Read, clean, and compute metrics for one file."""
    print(f"\n{'='*70}")
    print(f"Processing: {path.name}")
    print(f"{'='*70}")
    
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Rename columns to standard names
    rename_map = {
        "long_East": "long_east_deg",
        "lat_North": "lat_north_deg",
        "topography": "altitude_m",
        "planet_rad": "radius_m"
    }
    df = df.rename(columns=rename_map)
    
    # Check required columns exist
    required = ["long_east_deg", "lat_north_deg", "altitude_m", "radius_m"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return None
    
    # Keep only required base columns
    df = df[required].copy()
    
    # Remove rows with missing values in base columns
    initial = len(df)
    df = df.dropna(subset=required)
    if len(df) < initial:
        print(f"  Removed {initial - len(df):,} rows with missing base data")
    
    # Compute slope and roughness
    df = compute_metrics(df, baseline_m=BASELINE_M, min_neighbors=MIN_NEIGHBORS)
    
    # Statistics
    valid_slope = df["slope_deg"].notna().sum()
    valid_rough = df["roughness_rms_m"].notna().sum()
    print(f"  Valid slope: {valid_slope:,}/{len(df):,} ({100*valid_slope/len(df):.1f}%)")
    print(f"  Valid roughness: {valid_rough:,}/{len(df):,} ({100*valid_rough/len(df):.1f}%)")
    
    if valid_slope > 0:
        print(f"  Slope - Min: {df['slope_deg'].min():.2f}°, Max: {df['slope_deg'].max():.2f}°, Mean: {df['slope_deg'].mean():.2f}°")
    if valid_rough > 0:
        print(f"  Roughness - Min: {df['roughness_rms_m'].min():.2f}m, Max: {df['roughness_rms_m'].max():.2f}m, Mean: {df['roughness_rms_m'].mean():.2f}m")
    
    return df


def main():
    print(f"{'='*70}")
    print(f"WEAK NEGATIVES METRICS COMPUTATION")
    print(f"{'='*70}")
    print(f"Baseline distance: {BASELINE_M}m")
    print(f"Minimum neighbors: {MIN_NEIGHBORS}")
    print(f"Output: {OUT_PATH}")
    print(f"{'='*70}")
    
    # Find all weak_negative_*.csv files
    csv_files = sorted(SCRIPT_DIR.glob("weak_negative_*.csv"))
    
    if not csv_files:
        print("\nERROR: No weak_negative_*.csv files found")
        return 1
    
    print(f"\nFound {len(csv_files)} file(s)")
    
    tables = []
    for path in csv_files:
        df = process_file(path)
        if df is not None:
            tables.append(df)
    
    if not tables:
        print("\nERROR: No tables to combine")
        return 1
    
    # Combine all tables
    combined = pd.concat(tables, ignore_index=True)
    
    # Remove rows with missing slope or roughness
    initial = len(combined)
    combined = combined.dropna(subset=["slope_deg", "roughness_rms_m"])
    removed = initial - len(combined)
    
    print(f"\n{'='*70}")
    print(f"COMBINED DATASET")
    print(f"{'='*70}")
    print(f"Total rows: {len(combined):,}")
    print(f"Removed incomplete: {removed:,}")
    
    print(f"\nOverall statistics:")
    print(combined[["altitude_m", "slope_deg", "roughness_rms_m"]].describe())
    
    # Save
    combined.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Saved: {OUT_PATH}")
    print(f"  Columns: {list(combined.columns)}")
    
    # Show sample
    print(f"\nSample (first 3 rows):")
    print(combined.head(3).to_string(index=False))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
