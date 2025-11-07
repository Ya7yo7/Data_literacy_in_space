"""slope_gen.py

Calculate surface slope from topographic data using local plane fitting.

For each coordinate point:
- Uses the actual local radius (planet_rad) instead of a constant
- Identifies nearby points within a baseline distance
- Fits a plane to the local neighborhood
- Computes slope angle from the plane gradient

Adds a new column: slope_deg (degrees)

Usage:
    python3 slope_gen.py
    
Set DEBUG_MODE = False to write output files to data_with_slopes/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from shared_utils import (
    meters_per_degree,
    lonlat_to_local_xy,
    fit_plane,
    plane_slope_from_coeffs
)

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_clean_attributes"
OUTPUT_DIR = SCRIPT_DIR / "data_with_slopes"

# DEBUG MODE: Set to False to actually write files
DEBUG_MODE = False

# Processing parameters
BASELINE_M = 1000.0      # Search radius for neighbors (meters)
MIN_NEIGHBORS = 8        # Minimum points needed for reliable plane fit


def compute_slope(df, baseline_m=BASELINE_M, min_neighbors=MIN_NEIGHBORS):
    """
    Compute slope per point using adaptive local neighborhood plane fitting.
    
    Args:
        df: DataFrame with columns long_east_deg, lat_north_deg, altitude_m, radius_m
        baseline_m: Radius in meters for finding nearby points
        min_neighbors: Minimum number of points needed for plane fitting
    
    Returns:
        DataFrame with added 'slope_deg' column
    """
    lon = df["long_east_deg"].to_numpy()
    lat = df["lat_north_deg"].to_numpy()
    alt = df["altitude_m"].to_numpy()
    rad = df["radius_m"].to_numpy()
    
    slope = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        lon0, lat0, z0, r0 = lon[i], lat[i], alt[i], rad[i]
        
        # Calculate angular search window based on baseline distance
        m_per_deg_lon0, m_per_deg_lat0 = meters_per_degree(lat0, r0)
        dlon_deg = baseline_m / m_per_deg_lon0
        dlat_deg = baseline_m / m_per_deg_lat0
        
        # Find neighbors within the search window
        mask = (
            (np.abs(lon - lon0) <= dlon_deg) &
            (np.abs(lat - lat0) <= dlat_deg)
        )
        idx = np.where(mask)[0]
        
        if len(idx) < min_neighbors:
            continue
        
        # Convert to local x,y coordinates in meters
        x, y = lonlat_to_local_xy(lon[idx], lat[idx], lon0, lat0, r0)
        z = alt[idx]
        
        try:
            # Fit plane and compute slope
            a, b, c = fit_plane(x, y, z)
            slope[i] = plane_slope_from_coeffs(a, b)
        except Exception:
            # Fitting failed (e.g., collinear points)
            pass
    
    df["slope_deg"] = slope
    return df


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"SLOPE GENERATION")
    print(f"{'='*70}")
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Baseline distance: {BASELINE_M}m")
    print(f"Minimum neighbors: {MIN_NEIGHBORS}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"{'='*70}\n")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        exit(1)
    
    csv_files = sorted(DATA_DIR.glob("*_topography.csv"))
    
    if not csv_files:
        print(f"ERROR: No *_topography.csv files found in {DATA_DIR}")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s)\n")
    
    if not DEBUG_MODE:
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    for path in csv_files:
        name = path.stem
        print(f"{'='*70}")
        print(f"Processing: {name}")
        print(f"{'='*70}")
        
        try:
            df = pd.read_csv(path)
            print(f"  Loaded {len(df):,} rows")
            
            # Check for required columns
            required = ["long_east_deg", "lat_north_deg", "altitude_m", "radius_m"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                print(f"  ERROR: Missing columns: {missing}")
                continue
            
            print(f"  Computing slopes...")
            df_out = compute_slope(df, baseline_m=BASELINE_M, min_neighbors=MIN_NEIGHBORS)
            
            # Statistics
            valid = df_out["slope_deg"].notna().sum()
            print(f"  Valid slope values: {valid:,}/{len(df):,} ({100*valid/len(df):.1f}%)")
            
            if valid > 0:
                print(f"  Slope statistics:")
                print(f"    Min:    {df_out['slope_deg'].min():.2f}°")
                print(f"    Max:    {df_out['slope_deg'].max():.2f}°")
                print(f"    Mean:   {df_out['slope_deg'].mean():.2f}°")
                print(f"    Median: {df_out['slope_deg'].median():.2f}°")
                print(f"    Std:    {df_out['slope_deg'].std():.2f}°")
            
            # Show sample
            print(f"\n  Sample results (first 5 valid rows):")
            sample = df_out[df_out["slope_deg"].notna()].head(5)
            if len(sample) > 0:
                print(sample[["long_east_deg", "lat_north_deg", "altitude_m", "slope_deg"]].to_string(index=False))
            
            if DEBUG_MODE:
                print(f"\n  [DEBUG MODE] Skipping file write")
            else:
                out_path = OUTPUT_DIR / f"{name}.csv"
                df_out.to_csv(out_path, index=False)
                print(f"\n  ✓ Saved: {out_path}")
                
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"{'='*70}")
    if DEBUG_MODE:
        print("DEBUG MODE: No files were written")
        print("Set DEBUG_MODE = False in the script to write output files")
    else:
        print(f"✓ All files processed and saved to {OUTPUT_DIR}")
    print(f"{'='*70}")
