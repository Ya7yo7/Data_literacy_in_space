# Mars Landing Suitability ML Pipeline

This directory contains machine learning models to classify Martian terrain as suitable or unsafe for landing missions.

## Overview

The pipeline uses an ensemble of three complementary models:
1. **Binary Classifier**: Random Forest or Gradient Boosting to classify terrain as suitable/unsuitable
2. **Safety Filter**: Threshold-based or One-Class SVM to identify unsafe terrain features
3. **Similarity Model**: Kernel Density Estimation or Gaussian Mixture Model to find similarity to successful landing sites

## Dataset

**Source**: `../Topography/data_cleaning/final_labeled_dataset.csv`

**Features** (6 total):
- `long_east_deg`: East longitude [0-360°]
- `lat_north_deg`: North latitude [-90 to 90°]
- `altitude_m`: Elevation relative to Mars areoid
- `radius_m`: Local planetary radius
- `slope_deg`: Local terrain slope from plane fitting
- `roughness_rms_m`: Surface roughness (RMS of elevation residuals)

**Labels**:
- `1`: Positive (successful mission landing sites) - 48,903 samples
- `0`: Weak negatives (potentially unsuitable) - 61,927 samples
- `-1`: Hard negatives (definitely unsuitable: too rough or too steep) - 49,947 samples

**Total**: 160,777 samples

## Files

### Core Modules
- **`data_loader.py`**: Data loading, preprocessing, and spatial cross-validation splits
- **`binary_classifier.py`**: Binary classification (Random Forest, Gradient Boosting)
- **`safety_filter.py`**: Safety filtering (Threshold, One-Class SVM, Isolation Forest)
- **`similarity_model.py`**: Similarity scoring (KDE, GMM)
- **`ensemble.py`**: Ensemble combining all models with weighted voting
- **`main.py`**: Main pipeline orchestration, training, evaluation, and visualization
- **`show_tree.py`**: Utility to display the directory structure with file sizes

### Output Directory Structure
All generated files are organized in the `output/` directory:

```
output/
├── trained_models/      # Saved model files (.pkl)
│   ├── binary_random_forest.pkl
│   ├── safety_threshold.pkl
│   ├── similarity_kde.pkl
│   └── ensemble.pkl
├── plots/              # Evaluation visualizations
│   ├── feature_importance.png
│   ├── score_distributions.png
│   └── component_scores.png
├── results/            # Metrics and evaluation results
│   └── metrics_random_forest_threshold_kde.json
└── predictions/        # Model predictions on test set
    └── test_predictions_random_forest_threshold_kde.csv
```

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python main.py
```

This will:
1. Load and prepare the dataset (160K samples)
2. Train all models (Random Forest, threshold filter, KDE)
3. Evaluate on test set (20% holdout)
4. Save trained models to `trained_models/`
5. Generate visualizations in `plots/`
6. Test on known Mars landing sites

### Custom Configuration

```python
from main import train_full_pipeline

results = train_full_pipeline(
    binary_type='gradient_boosting',  # or 'random_forest'
    safety_type='one_class_svm',      # or 'threshold', 'isolation_forest'
    similarity_type='gmm',            # or 'kde'
    test_split=0.2,
    random_state=42
)
```

### Testing Individual Models

Each model can be tested independently:

```bash
# Test data loader
python data_loader.py

# Test binary classifier
python binary_classifier.py

# Test safety filter
python safety_filter.py

# Test similarity model
python similarity_model.py

# Test ensemble
python ensemble.py
```

### Scoring New Locations

```python
from ensemble import LandingSuitabilityEnsemble

# Load trained ensemble
ensemble = LandingSuitabilityEnsemble()
ensemble.load('output/trained_models/ensemble.pkl')

# Score a new location
result = ensemble.score_new_location(
    longitude=137.4,    # Gale Crater (Curiosity)
    latitude=-4.5,
    altitude=-4500,
    radius=3396000,
    slope=2.5,
    roughness=5.0
)

print(f"Suitability Score: {result['final_score']:.3f}")
print(f"Recommendation: {result['recommendation']}")
```

### Viewing Directory Structure

```bash
# Show complete modeling directory tree
python show_tree.py

# Show only output directory
python show_tree.py --output-only
```

## Model Details

### Binary Classifier

**Random Forest** (default):
- 200 trees, max_depth=15
- Class-balanced weighting to handle imbalance
- Returns probability of suitability [0, 1]

**Gradient Boosting** (alternative):
- 200 estimators, learning_rate=0.1, max_depth=8
- Sequential tree building for high accuracy

**Key Features**: `slope_deg` and `roughness_rms_m` are typically most important

### Safety Filter

**Threshold Method** (default):
- Computes 95th percentile thresholds from successful sites
- Conservative: slope < ~3-5°, roughness < ~8-10m
- Fast and interpretable

**One-Class SVM** (alternative):
- Learns boundary of "safe" region from positive examples
- More flexible but slower

### Similarity Model

**Kernel Density Estimation** (default):
- Estimates probability density of successful landing sites
- Auto-tuned bandwidth using Scott's rule
- Returns log-likelihood scores

**Gaussian Mixture Model** (alternative):
- Fits 5 Gaussian components to capture multimodal distribution
- Can identify distinct "clusters" of similar sites

### Ensemble

Combines all three models with weighted voting:
- Binary classifier: 40%
- Safety filter: 30%
- Similarity model: 30%

Final score in [0, 1]: higher = more suitable for landing

## Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall classification correctness
- **Precision/Recall/F1**: Detailed performance on positive class
- **ROC-AUC**: Area under ROC curve (binary classifier)
- **PR-AUC**: Precision-Recall AUC (handles class imbalance)
- **Feature Importance**: Most influential terrain features

## Performance

Expected performance on test set (20% holdout):
- Binary classifier: 85-90% accuracy, 0.90+ ROC-AUC
- Safety filter: 75-85% accuracy on safe/unsafe separation
- Similarity model: Good separation between positive and negative examples
- Ensemble: 80-90% accuracy with balanced precision/recall

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Data Processing

The dataset is generated from Mars MOLA (Mars Orbiter Laser Altimeter) data using the pipeline in `../Topography/data_cleaning/`:

1. Extract raw coordinates and elevation
2. Compute slope (plane fitting on 1000m baseline)
3. Compute roughness (RMS residuals from fitted plane)
4. Label examples based on mission success and terrain criteria

See `../Topography/data_cleaning/README.md` for details.

## Spatial Cross-Validation

The data loader supports spatial cross-validation to avoid overfitting from spatially correlated samples. Use `loader.spatial_cv_splits(n_splits=5)` for proper evaluation.

## Future Enhancements

Potential improvements:
- [ ] Add more terrain features (crater density, thermal inertia, etc.)
- [ ] Implement true spatial clustering for CV (e.g., k-means on lat/lon)
- [ ] Add explainability (SHAP values, LIME)
- [ ] Create interactive web interface for scoring
- [ ] Integrate with Mars orbital imagery

## References

- Mars MOLA PEDR data: https://pds-geosciences.wustl.edu/missions/mgs/mola.html
- Landing site selection criteria: NASA Mars Exploration Program
- Successful landing sites: Curiosity, Perseverance, InSight, Spirit, Opportunity, Phoenix

## Authors

Data Literacy in Space Project
