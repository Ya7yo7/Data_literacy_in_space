"""combine_metrics.py

Combine slope and roughness data into complete topographic datasets.

This script:
1. Reads slope data from data_with_slopes/
2. Reads roughness data from data_with_roughness/
3. Merges them on coordinate columns (long_east_deg, lat_north_deg)
4. Removes rows with missing values in any metric column
5. Saves complete datasets to data_complete/

Output columns:
- long_east_deg, lat_north_deg, altitude_m, radius_m
- slope_deg (from slope_gen.py)
- roughness_rms_m (from roughness_gen.py)

Only rows with all metrics present are included in the final output.

Usage:
    python3 combine_metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
SLOPE_DIR = SCRIPT_DIR / "data_with_slopes"
ROUGHNESS_DIR = SCRIPT_DIR / "data_with_roughness"
OUTPUT_DIR = SCRIPT_DIR / "data_complete"

# Expected columns in output
BASE_COLS = ["long_east_deg", "lat_north_deg", "altitude_m", "radius_m"]
METRIC_COLS = ["slope_deg", "roughness_rms_m"]
ALL_COLS = BASE_COLS + METRIC_COLS


def merge_mission_data(mission_name):
    """
    Merge slope and roughness data for a single mission.
    
    Args:
        mission_name: Name of the mission (without .csv extension)
    
    Returns:
        DataFrame with complete data (no missing values) or None if merge fails
    """
    slope_path = SLOPE_DIR / f"{mission_name}.csv"
    rough_path = ROUGHNESS_DIR / f"{mission_name}.csv"
    
    # Check files exist
    if not slope_path.exists():
        print(f"  WARNING: Slope file not found: {slope_path.name}")
        return None
    if not rough_path.exists():
        print(f"  WARNING: Roughness file not found: {rough_path.name}")
        return None
    
    # Read data
    try:
        df_slope = pd.read_csv(slope_path)
        df_rough = pd.read_csv(rough_path)
    except Exception as e:
        print(f"  ERROR reading files: {e}")
        return None
    
    print(f"  Loaded slope data: {len(df_slope):,} rows")
    print(f"  Loaded roughness data: {len(df_rough):,} rows")
    
    # Check required columns
    required_slope = BASE_COLS + ["slope_deg"]
    required_rough = BASE_COLS + ["roughness_rms_m"]
    
    missing_slope = [col for col in required_slope if col not in df_slope.columns]
    missing_rough = [col for col in required_rough if col not in df_rough.columns]
    
    if missing_slope:
        print(f"  ERROR: Missing columns in slope data: {missing_slope}")
        return None
    if missing_rough:
        print(f"  ERROR: Missing columns in roughness data: {missing_rough}")
        return None
    
    # Merge on coordinate columns
    # Use inner join to keep only rows present in both datasets
    merge_on = ["long_east_deg", "lat_north_deg", "altitude_m", "radius_m"]
    
    df_merged = pd.merge(
        df_slope[required_slope],
        df_rough[required_rough],
        on=merge_on,
        how="inner",
        suffixes=("", "_dup")
    )
    
    print(f"  After merge: {len(df_merged):,} rows")
    
    # Remove rows with any missing values
    initial_count = len(df_merged)
    df_clean = df_merged[ALL_COLS].dropna()
    removed = initial_count - len(df_clean)
    
    if removed > 0:
        print(f"  Removed {removed:,} rows with missing values")
    
    print(f"  Final clean data: {len(df_clean):,} rows ({100*len(df_clean)/initial_count:.1f}% complete)")
    
    return df_clean


def print_statistics(df, mission_name):
    """Print summary statistics for the complete dataset."""
    print(f"\n  Statistics for {mission_name}:")
    print(f"  {'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for col in ["altitude_m", "slope_deg", "roughness_rms_m"]:
        if col in df.columns:
            print(f"  {col:<20} {df[col].min():>10.2f} {df[col].max():>10.2f} "
                  f"{df[col].mean():>10.2f} {df[col].median():>10.2f}")


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"COMBINE SLOPE AND ROUGHNESS METRICS")
    print(f"{'='*70}")
    print(f"Slope directory:     {SLOPE_DIR}")
    print(f"Roughness directory: {ROUGHNESS_DIR}")
    print(f"Output directory:    {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Check input directories exist
    if not SLOPE_DIR.exists():
        print(f"ERROR: Slope directory not found: {SLOPE_DIR}")
        print("Run slope_gen.py with DEBUG_MODE=False first")
        exit(1)
    
    if not ROUGHNESS_DIR.exists():
        print(f"ERROR: Roughness directory not found: {ROUGHNESS_DIR}")
        print("Run roughness_gen.py with DEBUG_MODE=False first")
        exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all mission files (use slope directory as reference)
    slope_files = sorted(SLOPE_DIR.glob("*_topography.csv"))
    
    if not slope_files:
        print(f"ERROR: No *_topography.csv files found in {SLOPE_DIR}")
        exit(1)
    
    print(f"Found {len(slope_files)} mission file(s)\n")
    
    successful = 0
    total_original = 0
    total_complete = 0
    
    for slope_path in slope_files:
        mission_name = slope_path.stem
        print(f"{'='*70}")
        print(f"Processing: {mission_name}")
        print(f"{'='*70}")
        
        try:
            df_complete = merge_mission_data(mission_name)
            
            if df_complete is None or len(df_complete) == 0:
                print(f"  SKIPPED: No complete data available")
                continue
            
            # Print statistics
            print_statistics(df_complete, mission_name)
            
            # Save to output
            out_path = OUTPUT_DIR / f"{mission_name}.csv"
            df_complete.to_csv(out_path, index=False)
            print(f"\n  ✓ Saved: {out_path}")
            print(f"  ✓ Columns: {list(df_complete.columns)}")
            
            successful += 1
            total_complete += len(df_complete)
            
            # Show sample
            print(f"\n  Sample (first 3 rows):")
            sample = df_complete.head(3)
            print(sample.to_string(index=False))
            
        except Exception as e:
            print(f"  ERROR processing {mission_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Missions processed: {len(slope_files)}")
    print(f"Successful merges:  {successful}")
    print(f"Total complete rows: {total_complete:,}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")
    
    if successful == 0:
        print("\nWARNING: No files were successfully merged!")
        print("Make sure both slope_gen.py and roughness_gen.py have been run")
        print("with DEBUG_MODE=False to generate the required input files.")
