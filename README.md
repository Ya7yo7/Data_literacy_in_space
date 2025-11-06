# Data Literacy in Space

A collection of Python tools for analyzing Mars topographic data from MOLA (Mars Orbiter Laser Altimeter) and assessing potential landing sites based on elevation, slope, and surface roughness.

## Project Overview

This project processes MOLA PEDR (Precision Experiment Data Record) data to:
- Extract topographic information around historical Mars landing sites
- Analyze feasible landing regions between multiple sites
- Generate surface metrics (slope, roughness) for safety assessment
- Visualize topographic data and landing constraints

## Project Structure

```
Data_literacy_in_space/
├── data_cleaning/              # Main analysis scripts
│   ├── data/                   # Raw topographic CSV data
│   │   ├── curiosity_topography.csv
│   │   ├── insight_topography.csv
│   │   ├── opportunity_topography.csv
│   │   ├── perseverance_topography.csv
│   │   ├── phoenix_topography.csv
│   │   ├── spirit_topography.csv
│   │   └── mars_landing_boxes_1deg.csv
│   ├── data_clean_attributes/  # Processed data (long_east, lat_north, altitude)
│   └── *.py                    # Analysis scripts (see below)
└── data_files/                 # Additional data resources
```

## Scripts

### Data Preparation

- **`coordinate_search.py`** — Generates bounding boxes (±0.5° lat/lon) around Mars landing sites for data extraction from ODE (Orbital Data Explorer).

- **`extract_attributes.py`** — Batch processes raw topography CSV files to extract and rename key columns (`long_east`, `lat_north`, `altitude`) for downstream analysis.

### Topographic Analysis

- **`mola_data_1.py`** — Loads and cleans raw MOLA PEDR data, filters valid surface returns (quality flags), and provides basic statistics on elevation and measurements.

- **`feasible_region.py`** — Calculates an elliptical feasible landing region between two sites (InSight and Curiosity) using haversine distance on Mars' surface. Outputs extreme coordinates for data queries.

### Visualization

- **`heatmap_1.py`** — Generates a synthetic heatmap with a clear elliptical cutout to visualize regions of interest or landing constraints.

### Future Work (TODO)

- **`slope_gen.py`** — Will implement surface slope calculation using finite difference or Horn's method for gradient analysis.

- **`roughness_gen.py`** — Will compute terrain roughness metrics (RMS height, TRI, Hurst exponent) for landing hazard assessment.

## Usage

### 1. Extract Clean Attributes from Raw Data

Process all topography CSV files to keep only essential columns:

```bash
cd data_cleaning
python3 extract_attributes.py
```

Output: cleaned CSV files in `data_clean_attributes/` with columns `long_east`, `lat_north`, `altitude`.

### 2. Generate Landing Site Bounding Boxes

Create coordinate boxes for data queries:

```bash
python3 coordinate_search.py
```

Output: `mars_landing_boxes_1deg.csv` with lat/lon bounds for each mission.

### 3. Analyze Feasible Landing Region

Compute elliptical region between two sites:

```bash
python3 feasible_region.py
```

Output: Prints extreme lat/lon coordinates and generates visualization.

### 4. Explore MOLA Data

Load and inspect raw MOLA PEDR data (update file path in script):

```bash
python3 mola_data_1.py
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