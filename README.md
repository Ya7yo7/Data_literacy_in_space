# Data Literacy in Space

A collection of Python tools for cleaning and analyzing Mars topographic data from MOLA (Mars Orbiter Laser Altimeter) to create labeled datasets of potential landing sites based on elevation, slope, and surface roughness.

## Project Overview

This project processes MOLA PEDR (Precision Experiment Data Record) data to:
- Extract topographic information around historical Mars landing sites
- Compute surface metrics (slope, roughness) using local plane fitting
- Generate labeled datasets (positive sites, weak negatives, hard negatives)
- Combine multiple missions into a unified training dataset

## Project Structure

```
Data_literacy_in_space/
├── Topography/
│   └── data_cleaning/              # Data processing pipeline
│       ├── *.py                    # Cleaning and analysis scripts
│       ├── data/                   # Raw MOLA topographic data
│       ├── data_clean_attributes/  # Extracted columns (lon, lat, alt, radius)
│       ├── data_with_slopes/       # Slope metrics added
│       ├── data_with_roughness/    # Roughness metrics added
│       ├── data_complete/          # All metrics combined
│       ├── hard_negatives/         # Known unsuitable terrain
│       ├── weak_negatives/         # Random Mars surface samples
│       └── final_labeled_dataset.csv  # Combined labeled dataset
├── considered_regions.txt          # Data source regions
└── candidate_data.csv              # Unlabeled candidate sites (add manually)
```

## Data Cleaning Pipeline

### 1. Data Extraction (`extract_attributes.py`)
Batch processes raw MOLA CSV files to extract essential columns:
- `long_east_deg` — East longitude (0–360°)
- `lat_north_deg` — Planetocentric latitude
- `altitude_m` — Elevation relative to Mars areoid
- `radius_m` — Local planetary radius

### 2. Coordinate Search (`coordinate_search.py`)
Generates bounding boxes (±0.5° lat/lon) around historical landing sites for querying additional data from NASA's ODE (Orbital Data Explorer).

### 3. Surface Slope Computation (`slope_gen.py`)
Calculates local surface slope using adaptive plane fitting:
- Finds neighbors within 1 km baseline distance
- Fits a plane using least-squares regression
- Computes slope angle from plane gradient
- Accounts for Mars' ellipsoidal shape using local radius

Output: `slope_deg` column added to each dataset.

### 4. Surface Roughness Computation (`roughness_gen.py`)
Calculates terrain roughness as RMS deviation from local plane:
- Uses same adaptive neighborhood as slope computation
- Measures vertical scatter around best-fit plane
- Indicates small-scale topographic variability

Output: `roughness_rms_m` column added to each dataset.

### 5. Dataset Combination (`combine_metrics.py`)
Merges slope and roughness data into complete topographic profiles for each mission.

### 6. Labeled Dataset Creation (`create_final_labeled_dataset.py`)
Combines all missions with labels:
- **Positive samples (label=1)**: Historical landing sites (Curiosity, InSight, Opportunity, Perseverance, Phoenix, Spirit)
- **Weak negatives (label=0)**: Random Mars surface points (no known hazards)
- **Hard negatives (label=-1)**: Known unsuitable terrain (steep slopes, rough surfaces)

Output: `final_labeled_dataset.csv` with ~160K labeled points.

## Usage

### Complete Pipeline

Run the full data cleaning pipeline:

```bash
cd Topography/data_cleaning

# 1. Extract attributes from raw MOLA data
python3 extract_attributes.py

# 2. Compute slope metrics
python3 slope_gen.py

# 3. Compute roughness metrics
python3 roughness_gen.py

# 4. Combine all metrics
python3 combine_metrics.py

# 5. Generate final labeled dataset
python3 create_final_labeled_dataset.py
```

### Individual Scripts

Generate landing site bounding boxes:
```bash
python3 coordinate_search.py
```

Analyze feasible landing region between sites:
```bash
python3 feasible_region.py
```

Explore raw MOLA data:
```bash
python3 mola_data_1.py
```

Generate visualization heatmap:
```bash
python3 heatmap_1.py
```

## Data Sources

- **MOLA PEDR**: Mars Global Surveyor laser altimetry data from NASA's Planetary Data System (PDS)
- **Landing Site Coordinates**: Planetocentric lat/lon from mission documentation

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib

Install dependencies:

```bash
pip install pandas numpy matplotlib
```

## Mars Coordinate Systems

This project uses:
- **Planetocentric latitude**: Angle from the equatorial plane (not surface normal)
- **East longitude**: 0–360° measured eastward from the prime meridian
- **Areoid reference**: Mars' gravity-based reference surface (analogous to Earth's geoid)

## Contributing

This project is part of a data literacy workshop exploring planetary science datasets.

## License

Educational/Research Use