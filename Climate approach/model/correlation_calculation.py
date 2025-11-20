'''Author: Siddharth Bhowmik
Created: November 2025'''

import pandas as pd

# === FILE NAMES ===
csv_files = [
    "candidates_mahalanobis.csv",
    "candidates_range.csv",
    "candidates_zscore.csv"
]
 
variable = "atm_density"
 
column_names = [
    'lattitude', 'longitude',
    'atm_pressure', 'atm_density',
    'temperature', 'zonal_wind', 'meridional_wind'
]
 
# === STEP 1: Load datasets ===
dfs = []
for file in csv_files:
    df = pd.read_csv(file, sep=';', header=None, engine='python', skiprows=1, usecols=range(7))
    df.columns = column_names
    df.columns = df.columns.str.strip().str.lower()
 
    # convert to numeric
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
 
    df['lattitude'] = df['lattitude'].round(3)
    df['longitude'] = df['longitude'].round(3)
    dfs.append(df)
 
# === STEP 2: Merge with explicit suffixes ===
merged = dfs[0].merge(dfs[1], on=['lattitude', 'longitude'], suffixes=('_1', '_2'), how='inner')
merged = merged.merge(dfs[2], on=['lattitude', 'longitude'], how='inner')
 
# Rename third file columns manually
for col in column_names:
    if col not in ['lattitude', 'longitude']:
        merged.rename(columns={col: f"{col}_3"}, inplace=True)
 
print(f" Merged dataset shape: {merged.shape}")
 
# === STEP 3: Compute correlations ===
corr_12 = merged[f"{variable}_1"].corr(merged[f"{variable}_2"])
corr_13 = merged[f"{variable}_1"].corr(merged[f"{variable}_3"])
corr_23 = merged[f"{variable}_2"].corr(merged[f"{variable}_3"])
 
print("\n Correlation Results:")
print(f"{variable}_1 vs {variable}_2: {corr_12:.4f}")
print(f"{variable}_1 vs {variable}_3: {corr_13:.4f}")
print(f"{variable}_2 vs {variable}_3: {corr_23:.4f}")