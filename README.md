# Data Literacy in Space

A comprehensive system for Mars landing site evaluation combining data cleaning, topographic analysis, and three-model machine learning comparison.

## Project Overview

This project processes MOLA (Mars Orbiter Laser Altimeter) data to:
- Extract topographic information around historical Mars landing sites
- Compute surface metrics (slope, roughness) using local plane fitting
- Generate labeled datasets (positive sites, weak negatives, hard negatives)
- Evaluate candidate landing sites using three complementary approaches:
  1. **NASA Constraints** (deterministic)
  2. **Random Forest** (supervised ML)
  3. **Similarity KDE** (unsupervised ML)

## Project Structure

```
Data_literacy_in_space/
├── data/                           # Data files
│   ├── processed/                  # Cleaned datasets (excluded from git)
│   │   ├── final_labeled_dataset.csv   # 160K labeled training sites
│   │   └── candidate_data.csv          # 758K unlabeled candidates
│   └── previews/                   # First 1000 rows (tracked in git)
│       ├── final_labeled_dataset_preview.csv
│       └── candidate_data_preview.csv
├── Topography/
│   └── data_cleaning/              # Data processing pipeline
│       ├── *.py                    # Cleaning and analysis scripts
│       ├── hard_negatives/         # Known unsuitable terrain
│       └── weak_negatives/         # Random Mars surface samples
├── modeling/                       # Three-model evaluation system
│   ├── data_loader.py              # Training data loading and scaling
│   ├── nasa_classifier.py          # Deterministic constraints
│   ├── random_forest_model.py      # Supervised ensemble learning
│   ├── similarity_model.py         # Unsupervised KDE similarity
│   ├── evaluate_candidates.py      # Main evaluation pipeline
│   ├── visualize_results.py        # Comparison plots
│   ├── EVALUATION_REPORT.md        # Detailed findings
│   └── output/
│       ├── models/                 # Trained models
│       ├── results/                # Evaluation results (101 MB)
│       └── plots/                  # Visualization figures
├── outputs/
│   ├── figures/                    # Analysis plots
│   └── intermediate/               # Temporary files
└── considered_regions.txt          # Data source regions
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

### Data Cleaning Pipeline

Run the full data cleaning pipeline to generate labeled datasets:

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

### Model Evaluation Pipeline

Evaluate 758K candidate landing sites with three models:

```bash
cd modeling

# Train models and evaluate all candidates
python3 evaluate_candidates.py

# Generate comparison visualizations
python3 visualize_results.py
```

**Output**: 
- Trained models saved to `output/models/`
- Evaluation results (101 MB) in `output/results/candidate_evaluations.csv`
- Comparison plots in `output/plots/`
- Detailed report in `EVALUATION_REPORT.md`

### Individual Data Cleaning Scripts

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

## Evaluation Results Summary

**Region**: 320-330°E, ±5°N (Equatorial Mars)  
**Candidates Evaluated**: 758,108 sites

### Model Performance

| Model | Philosophy | Suitable Sites | % Suitable |
|-------|-----------|----------------|------------|
| NASA Constraints | Physics-based rules | 388,348 | 51.2% |
| Random Forest | Supervised ML | 2,240 | 0.3% |
| Similarity KDE | Unsupervised ML | 340,910 | 45.0% |

### Consensus Analysis

- **All Accept (3/3)**: 0 sites (0.0%)
- **Majority Accept (2/3)**: 301,959 sites (39.8%) — **Recommended**
- **Minority Accept (1/3)**: 127,580 sites (16.8%)
- **All Reject (0/3)**: 328,569 sites (43.3%)

**Key Finding**: Random Forest is extremely conservative (0.3% acceptance), likely due to:
- 2x weighting of hard negatives during training
- Perfect training accuracy suggesting overfitting
- Strong dependence on altitude and geographic features (82% combined importance)

See `modeling/EVALUATION_REPORT.md` for detailed analysis.

## Data Sources

- **MOLA PEDR**: Mars Global Surveyor laser altimetry data from NASA's Planetary Data System (PDS)
- **Landing Site Coordinates**: Planetocentric lat/lon from mission documentation
- **Training Data**: 160,777 labeled sites from 6 Mars missions (Curiosity, InSight, Opportunity, Perseverance, Phoenix, Spirit)

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
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