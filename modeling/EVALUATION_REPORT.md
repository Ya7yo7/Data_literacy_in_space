# Mars Landing Site Evaluation - Three-Model Comparison

**Date**: 2024  
**Region**: 320-330°E, ±5°N (Equatorial Mars)  
**Candidates Evaluated**: 758,108 sites  

---

## Executive Summary

Three fundamentally different approaches were used to evaluate Mars landing sites:

1. **NASA Constraints** (Deterministic): Physics-based rules with soft thresholds
2. **Random Forest** (Supervised ML): Learns from labeled mission data with emphasis on hazards
3. **Similarity KDE** (Unsupervised ML): Identifies sites similar to successful missions

### Key Findings

- **No perfect consensus**: 0 sites accepted by all three models
- **Strong disagreement**: 43.3% of sites rejected by all models
- **Moderate agreement**: 39.8% accepted by 2 of 3 models (majority consensus)
- **Model selectivity**: RF (0.3%) ≪ SIM (45.0%) ≈ NASA (51.2%)

The Random Forest model is **extremely conservative**, accepting only 2,240 sites (0.3%), suggesting it learned very strict criteria from the training data, particularly emphasizing hard negative examples.

---

## Model Descriptions

### 1. NASA Constraints Classifier
**Type**: Deterministic rule-based  
**Philosophy**: Enforce engineering constraints with soft scoring  

**Constraints**:
- Slope < 5°
- Roughness < 8m RMS
- Altitude < -1300m

**Scoring**: Logistic smoothing around thresholds, geometric mean aggregation

**Result**: 388,348 suitable sites (51.2%)

---

### 2. Random Forest Classifier
**Type**: Supervised ensemble learning  
**Training**: 160,777 labeled sites (30.4% positive, 38.5% weak negative, 31.1% hard negative)  

**Configuration**:
- 200 trees
- Balanced class weights
- Hard negatives weighted 2x
- Features: longitude, latitude, altitude, radius, slope, roughness

**Performance**:
- Validation accuracy: 100%
- Validation AUC: 1.000

**Feature Importance**:
1. Altitude (42.0%)
2. Latitude (22.1%)
3. Longitude (17.7%)
4. Radius (15.3%)
5. Roughness (1.6%)
6. Slope (remaining)

**Result**: 2,240 suitable sites (0.3%)

**Analysis**: Unexpectedly conservative. The 2x weighting of hard negatives and perfect validation accuracy suggest potential overfitting to training distribution. May be capturing subtle patterns in the labeled data that make it extremely selective.

---

### 3. Similarity Model (KDE)
**Type**: Unsupervised density estimation  
**Training**: 48,903 positive examples only  

**Configuration**:
- Kernel: Gaussian
- Bandwidth: 0.4281 (auto-tuned via 3-fold CV)
- Feature space: Scaled 6D (all features)

**Philosophy**: Score sites by similarity to known successful landing sites, ignoring negative examples entirely.

**Result**: 340,910 suitable sites (45.0%)

---

## Score Distributions

### NASA Scores
- Mean: 0.449
- Median: 0.522
- Range: [0.000, 1.000]
- Distribution: Bimodal (peaks near 0 and 1)

### Random Forest Scores
- Mean: 0.136
- Median: 0.075
- Range: [0.035, 0.625]
- Distribution: Heavily skewed toward low scores
- **Note**: Maximum score only 0.625 (well below typical "confident" threshold)

### Similarity (KDE) Scores
- Mean: 0.191
- Median: 0.355
- Range: [-0.555, 0.948]
- Distribution: Wide spread, slight negative skew

---

## Model Agreement Analysis

### Pairwise Agreement

**NASA vs Random Forest**:
- Agreement: 48.8% (both reject or both accept)
- Disagreement: 51.2%
- NASA accepts / RF rejects: 386,782 sites (51.0%)

**NASA vs Similarity**:
- Agreement: 83.2%
- Disagreement: 16.8%
- NASA accepts / SIM rejects: 87,955 sites (11.6%)
- NASA rejects / SIM accepts: 40,517 sites (5.3%)

**Random Forest vs Similarity**:
- Agreement: 54.8%
- Disagreement: 45.2%
- RF rejects / SIM accepts: 340,910 sites (45.0%)
- **RF never accepts when SIM rejects** (RF is strictly more conservative)

### Consensus Categories

| Category | Count | Percentage |
|----------|-------|------------|
| All Accept (3/3) | 0 | 0.0% |
| Majority Accept (2/3) | 301,959 | 39.8% |
| Minority Accept (1/3) | 127,580 | 16.8% |
| All Reject (0/3) | 328,569 | 43.3% |

---

## Spatial Patterns

### Geographic Distribution
- Longitude range: 320-330°E (10° span)
- Latitude range: -5° to 5°N (10° span)
- **Observation**: All models show spatial clustering, suggesting terrain-dependent suitability

### Feature Space Analysis (Altitude-Slope)

**NASA**: Clear decision boundaries at slope=5° and altitude=-1300m  
**Random Forest**: Extremely restrictive, accepts only sites with very low slope AND low altitude  
**Similarity**: More diffuse acceptance region, following positive example distribution

---

## Interpretation & Recommendations

### Why is Random Forest so conservative?

1. **Perfect training accuracy** suggests overfitting to training distribution
2. **2x hard negative weighting** emphasizes learning from known hazards
3. **Maximum score of 0.625** indicates the model never achieves high confidence
4. **Feature importance** shows altitude and geographic location dominate (42% + 22% + 18% = 82%)

**Hypothesis**: The RF learned that the training positives occupy a very specific region of feature space, and it's applying those learned geographic constraints strictly. This could be appropriate if the labeled missions were in a geographically limited area.

### Model Comparison

- **NASA**: Fast, interpretable, physics-based. Good baseline.
- **Random Forest**: Potentially overly conservative. May need:
  - Reduced hard negative weighting (try 1.5x instead of 2x)
  - More regularization (max_depth limit)
  - Geographic features removed or downweighted
- **Similarity KDE**: Moderate selectivity, purely data-driven. No awareness of hazards (only learns from successes).

### Recommended Approach

For **safe, validated sites**, use **majority consensus (2/3 models)**:
- **301,959 sites** meet this criterion
- Balances conservatism (rejects >50% of candidates) with practicality
- Reduces false positives from overly permissive models

For **exploratory missions** accepting higher risk:
- Use NASA + Similarity agreement (300,393 sites)
- Excludes RF conservatism

For **maximum safety** (conservative):
- Investigate the 2,240 RF-accepted sites more closely
- These may represent the safest sites if RF's conservatism is justified

---

## Files Generated

### Models
- `output/models/nasa_constraints.txt` - NASA constraint parameters
- `output/models/random_forest.pkl` - Trained RF model (serialized)
- `output/models/similarity_kde.pkl` - Trained KDE model (serialized)

### Results
- `output/results/candidate_evaluations.csv` - Full results (101.6 MB, 758K rows)
- `output/results/candidate_evaluations_preview.csv` - First 10K rows for preview

### Visualizations
- `output/plots/score_distributions.png` - Score histograms for all models
- `output/plots/consensus_breakdown.png` - Bar chart and pie chart of consensus
- `output/plots/consensus_map.png` - Geographic map colored by consensus count
- `output/plots/individual_predictions_map.png` - Three geographic maps (one per model)
- `output/plots/score_correlations.png` - Pairwise score scatter plots
- `output/plots/feature_space_altitude_slope.png` - Decision boundaries in altitude-slope space

---

## Conclusion

The three-model evaluation successfully identified **301,959 candidate landing sites** with majority consensus (2 of 3 models agree). The Random Forest model's extreme conservatism (0.3% acceptance rate) is notable and warrants further investigation—it may reflect overfitting to the training data's geographic distribution or appropriately strict learned criteria from hard negative examples.

**Next steps**:
1. Investigate RF's low acceptance rate (retrain with adjusted hyperparameters?)
2. Examine the 2,240 RF-accepted sites in detail (what makes them special?)
3. Validate majority-consensus sites against independent mission planning criteria
4. Consider ensemble approaches (e.g., weighted voting rather than hard threshold)
