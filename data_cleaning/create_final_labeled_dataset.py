#!/usr/bin/env python3
"""create_final_labeled_dataset.py

Combine all datasets with labels:
- hard_negatives: label = -1 (very poor landing sites)
- weak_negatives: label = 0 (marginal landing sites)
- data_complete (mission sites): label = 1 (good landing sites)

Output: final_labeled_dataset.csv in the data_cleaning directory

Usage:
    python3 create_final_labeled_dataset.py
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CLEANING_DIR = SCRIPT_DIR  # Script is in data_cleaning directory

# Input paths
HARD_NEG_PATH = DATA_CLEANING_DIR / "hard_negatives" / "hard_negatives.csv"
WEAK_NEG_PATH = DATA_CLEANING_DIR / "weak_negatives" / "weak_negatives.csv"
DATA_COMPLETE_DIR = DATA_CLEANING_DIR / "data_complete"

# Output path
OUT_PATH = DATA_CLEANING_DIR / "final_labeled_dataset.csv"

# Expected columns (consistent across all datasets)
REQUIRED_COLS = ["long_east_deg", "lat_north_deg", "altitude_m", "radius_m", "slope_deg", "roughness_rms_m"]


def load_and_label(path: Path, label: int, description: str):
    """Load a dataset and add the label column."""
    print(f"\nLoading {description}...")
    print(f"  Path: {path}")
    
    if not path.exists():
        print(f"  ERROR: File not found")
        return None
    
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows")
    
    # Check if required columns exist
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        # Try to continue with available columns
    
    # Keep only required columns (if they exist)
    available_cols = [col for col in REQUIRED_COLS if col in df.columns]
    df = df[available_cols].copy()
    
    # Add label column
    df["label"] = label
    
    print(f"  Columns: {list(df.columns)}")
    print(f"  Label: {label}")
    
    return df


def load_mission_data(label: int):
    """Load all mission topography files from data_complete and combine."""
    print(f"\nLoading mission data (positive examples)...")
    print(f"  Directory: {DATA_COMPLETE_DIR}")
    
    if not DATA_COMPLETE_DIR.exists():
        print(f"  ERROR: Directory not found")
        return None
    
    csv_files = sorted(DATA_COMPLETE_DIR.glob("*_topography.csv"))
    
    if not csv_files:
        print(f"  ERROR: No *_topography.csv files found")
        return None
    
    print(f"  Found {len(csv_files)} mission file(s)")
    
    tables = []
    for path in csv_files:
        mission_name = path.stem
        print(f"    - {mission_name}")
        df = pd.read_csv(path)
        
        # Keep only required columns
        available_cols = [col for col in REQUIRED_COLS if col in df.columns]
        df = df[available_cols].copy()
        
        # Add mission name as a column (optional, for reference)
        df["mission"] = mission_name.replace("_topography", "")
        
        tables.append(df)
    
    # Combine all missions
    combined = pd.concat(tables, ignore_index=True)
    print(f"  Total mission rows: {len(combined):,}")
    
    # Add label
    combined["label"] = label
    
    return combined


def main():
    print(f"{'='*70}")
    print(f"CREATE FINAL LABELED DATASET")
    print(f"{'='*70}")
    print(f"Label scheme:")
    print(f"  -1 = hard_negatives (very poor landing sites)")
    print(f"   0 = weak_negatives (marginal landing sites)")
    print(f"   1 = mission sites (good landing sites)")
    print(f"{'='*70}")
    
    datasets = []
    
    # Load hard negatives
    df_hard = load_and_label(HARD_NEG_PATH, -1, "hard_negatives")
    if df_hard is not None:
        # Remove the binary label column if it exists (from previous processing)
        if "label" in df_hard.columns and df_hard.columns.tolist().count("label") > 1:
            # Keep only the last label column we just added
            df_hard = df_hard.loc[:, ~df_hard.columns.duplicated(keep='last')]
        datasets.append(df_hard)
    
    # Load weak negatives
    df_weak = load_and_label(WEAK_NEG_PATH, 0, "weak_negatives")
    if df_weak is not None:
        datasets.append(df_weak)
    
    # Load mission data (positives)
    df_missions = load_mission_data(1)
    if df_missions is not None:
        datasets.append(df_missions)
    
    if not datasets:
        print("\nERROR: No datasets loaded")
        return 1
    
    # Combine all datasets
    print(f"\n{'='*70}")
    print(f"COMBINING DATASETS")
    print(f"{'='*70}")
    
    final = pd.concat(datasets, ignore_index=True)
    
    print(f"Total rows: {len(final):,}")
    print(f"\nLabel distribution:")
    print(final["label"].value_counts().sort_index())
    
    # Remove rows with missing values in key columns
    initial = len(final)
    final = final.dropna(subset=REQUIRED_COLS)
    removed = initial - len(final)
    if removed > 0:
        print(f"\nRemoved {removed:,} rows with missing values")
        print(f"Final row count: {len(final):,}")
        print(f"\nUpdated label distribution:")
        print(final["label"].value_counts().sort_index())
    
    # Statistics by label
    print(f"\n{'='*70}")
    print(f"STATISTICS BY LABEL")
    print(f"{'='*70}")
    for label in sorted(final["label"].unique()):
        subset = final[final["label"] == label]
        label_name = {-1: "hard_negatives", 0: "weak_negatives", 1: "mission_sites"}.get(label, f"label_{label}")
        print(f"\n{label_name} (label={label}):")
        print(f"  Count: {len(subset):,}")
        print(f"  Altitude - Mean: {subset['altitude_m'].mean():.2f}m, Std: {subset['altitude_m'].std():.2f}m")
        print(f"  Slope - Mean: {subset['slope_deg'].mean():.2f}°, Std: {subset['slope_deg'].std():.2f}°")
        print(f"  Roughness - Mean: {subset['roughness_rms_m'].mean():.2f}m, Std: {subset['roughness_rms_m'].std():.2f}m")
    
    # Save
    final.to_csv(OUT_PATH, index=False)
    print(f"\n{'='*70}")
    print(f"✓ SAVED: {OUT_PATH}")
    print(f"{'='*70}")
    print(f"Columns: {list(final.columns)}")
    print(f"Total rows: {len(final):,}")
    
    # Show sample from each label
    print(f"\nSample rows from each label:")
    for label in sorted(final["label"].unique()):
        print(f"\nLabel {label}:")
        sample = final[final["label"] == label].head(2)
        print(sample[REQUIRED_COLS + ["label"]].to_string(index=False))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
