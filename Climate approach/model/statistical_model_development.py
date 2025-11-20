'''Author: Siddharth Bhowmik
Created: November 2025'''

import pandas as pd
import numpy as np
import glob
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import os
 
# === FOLDER WITH ALL 24 CSV FILES ===
folder_path = "C:/Users/Abhinav Bhowmik/Downloads/GCM every month/every_month"
 
# === FIND ALL CSV FILES IN THE FOLDER ===
csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
print(f" Found {len(csv_files)} CSV files")
 
# === READ AND CONCATENATE ONLY FIRST 7 COLUMNS FROM EACH FILE ===
dataframes = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file, sep=';', header=None, usecols=range(7), engine='python')
        dataframes.append(df_temp)
    except Exception as e:
        print(f" Skipping {os.path.basename(file)} due to error: {e}")
 
# Merge all 24 files
df = pd.concat(dataframes, ignore_index=True)
print(" Combined dataset shape:", df.shape)
 
# === RENAME THE FIRST 7 COLUMNS ===
# === RENAME THE FIRST 7 COLUMNS ===
df.columns = [
    'lattitude', 'longitude',
    'atm_pressure', 'atm_density',
    'temperature', 'zonal_wind', 'meridional_wind'
]
 
# === FORCE NUMERIC CONVERSION FOR ALL 7 COLUMNS ===
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
 
# Drop rows where lat/lon conversion failed
df = df.dropna(subset=['lattitude', 'longitude'])

print(f"Cleaned data types:\n{df.dtypes}")

# === DEFINE VARIABLES TO USE ===
core_vars = ['atm_pressure', 'atm_density', 'temperature', 'zonal_wind', 'meridional_wind']

# === KNOWN LANDING SITES (lat, lon in degrees) ===
landing_sites = [
    (-4, 137),# Gale Crater
    (5, 136),# Gusev Crater
    (-3, 355),  # Jezero Crater
    (18, 78),# Meridiani Planum
    (69, 235), # Elysium Planitia
    (-14, 175) # InSight
]
 
 
 
 
# === GET DATA FOR THE 6 LANDING SITES ===
def find_nearest(df, lat, lon):
    dist = np.sqrt((df['lattitude'] - lat)**2 + (df['longitude'] - lon)**2)
    idx = dist.idxmin()
    return df.loc[idx, core_vars]
 
landing_df = pd.DataFrame([find_nearest(df, lat, lon) for lat, lon in landing_sites])
 
# Force numeric conversion for all core variables
for c in core_vars:
    landing_df[c] = pd.to_numeric(landing_df[c], errors='coerce')
 
print("\n Landing site data (cleaned):")
print(landing_df.dtypes)
print("\nPreview:")
print(landing_df.head())
 
 
# === COMPUTE REFERENCE STATS ===
means = landing_df.mean()
stds = landing_df.std()
mins = landing_df.min()
maxs = landing_df.max()
cov_inv = np.linalg.inv(np.cov(landing_df[core_vars].T))
 
# === RANGE FILTER ===
mask_range = np.ones(len(df), dtype=bool)
for c in core_vars:
    mask_range &= (df[c] >= mins[c]) & (df[c] <= maxs[c])
 
df_range = df.loc[mask_range, ['lattitude', 'longitude'] + core_vars]
df_range.to_csv("candidates_range.csv", index=False, sep=';')
print(f"Saved {len(df_range)} points within range → candidates_range.csv")
 
# === Z-SCORE FILTER ===
df_z = df.copy()
for c in core_vars:
    df_z[c + '_z'] = (df[c] - means[c]) / stds[c]
 
mask_z = df_z[[c + '_z' for c in core_vars]].abs().le(2).all(axis=1)
df_zscore = df_z.loc[mask_z, ['lattitude', 'longitude'] + core_vars]
df_zscore.to_csv("candidates_zscore.csv", index=False, sep=';')
print(f" Saved {len(df_zscore)} points within ±2σ → candidates_zscore.csv")
 
# === 3️⃣ MAHALANOBIS DISTANCE ===
X = df[core_vars].values
mean_vec = means.values
 
mahal_dists = [mahalanobis(x, mean_vec, cov_inv) for x in X]
df['mahalanobis2'] = np.square(mahal_dists)
 
threshold = np.percentile(df['mahalanobis2'], 10)  # Keep 10% most similar points
df_maha = df.loc[df['mahalanobis2'] <= threshold, ['lattitude', 'longitude', 'mahalanobis2'] + core_vars]
df_maha.to_csv("candidates_mahalanobis.csv", index=False, sep=';')
print(f" Saved {len(df_maha)} statistically closest points → candidates_mahalanobis.csv")
 
print("\n All 3 CSVs generated successfully from 24 Mars climate files!")
 
 
 
csv_files = [
    "candidates_mahalanobis.csv",
    "candidates_range.csv",
    "candidates_zscore.csv"
]
