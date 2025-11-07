"""mola_data_1.py

Load, clean, and explore MOLA (Mars Orbiter Laser Altimeter) PEDR data.

This script reads raw MOLA CSV data, filters for valid surface returns
(c==1 and A==0), and provides basic statistics on topography and other
measurements. Useful for understanding the structure and quality of MOLA
datasets before further analysis.

Column descriptions:
- long_East: planetocentric east longitude (degrees 0-360° E)
- lat_North: planetocentric latitude (degrees -90 S to +90 N)
- topography: elevation relative to Mars areoid (meters)
- MOLArange: spacecraft-to-surface distance (meters)
- planet_rad: Mars center to surface distance (meters)
- c: data quality flag (1 = valid, 0 = invalid)
- A: additional status flag (usually 0 for normal surface)
- Ephemeris Time: seconds past J2000
- UTC: ISO 8601 timestamp
- Orbit: MGS orbit number
"""

'''| Column name        | Description                                                                                                                                           | Units                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **long_East**      | Planetocentric east longitude of the measurement location.                                                                                            | degrees (0–360° E)       |
| **lat_North**      | Planetocentric latitude of the measurement location.                                                                                                  | degrees (−90 S to +90 N) |
| **topography**     | Elevation relative to the **Mars areoid** (a gravity-based reference surface). Positive = above areoid; negative = below.                             | meters                   |
| **MOLArange**      | Distance from the spacecraft to the surface determined by the laser time-of-flight.                                                                   | meters                   |
| **planet_rad**     | Distance from Mars' center to the measured surface point (areoid + topography). Essentially the local planetary radius.                               | meters                   |
| **c**              | MOLA data quality flag (channel indicator). `1` = valid altimetry return; `0` = no valid surface return or noise.                                     | integer                  |
| **A**              | Additional acquisition or altimetry status flag; often used internally (e.g. atmospheric return or crossover). Usually `0` for normal surface echoes. | integer                  |
| **Ephemeris Time** | Spacecraft time of observation in **seconds past J2000** (the standard SPICE ephemeris time).                                                         | seconds                  |
| **UTC**            | Coordinated Universal Time corresponding to the measurement (derived from Ephemeris Time).                                                            | ISO 8601                 |
| **Orbit**          | Mars Global Surveyor orbit number during which the shot was taken.                                                                                    | integer                  |
'''

import pandas as pd
import matplotlib.pyplot as plt

# --- 1. File path ---
# Update this path if your file is in another directory
file_path = "/Users/edwind/Desktop/winter_school/project/PEDR_133E139E_5p59S5N_csv_table.csv"

# --- 2. Read the CSV ---
print("Loading MOLA PEDR data...")
df = pd.read_csv(file_path, low_memory=False)
print(f"Loaded {len(df):,} rows")

# --- 3. Clean and prepare ---
# Strip spaces from column names
df.columns = [c.strip() for c in df.columns]

# Print actual column names to debug
print("\nActual columns in file:")
print(df.columns.tolist())

# Ensure numeric columns are correct types
numeric_cols = ['long_East','lat_North','topography','MOLArange','planet_rad','c','A','Ephemeris Time','Orbit']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter valid surface returns only (c==1 and A==0)
df_valid = df[(df['c'] == 1) & (df['A'] == 0)]

# --- 4. Basic info ---
print("\n--- Basic Dataset Info ---")
print(df_valid.info())
print("\n--- Descriptive Statistics ---")
print(df_valid[['topography','MOLArange','planet_rad']].describe())