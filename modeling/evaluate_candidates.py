"""evaluate_candidates.py

Score all candidate landing sites with three different models:
1. NASA Constraints (deterministic)
2. Random Forest (supervised ML)
3. Similarity KDE (unsupervised ML)

Generates comprehensive results CSV with all scores and classifications.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, Tuple

from data_loader import MarsDataLoader
from nasa_classifier import NASAConstraintsClassifier
from random_forest_model import RandomForestModel
from similarity_model import SimilarityModel


def train_models(loader: MarsDataLoader) -> Tuple[NASAConstraintsClassifier, RandomForestModel, SimilarityModel]:
    """
    Train all three models on labeled data.
    
    Returns:
        (nasa_model, rf_model, sim_model)
    """
    print("="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # NASA Constraints (no training needed)
    nasa_model = NASAConstraintsClassifier()
    print("\n✓ NASA Constraints: Initialized (deterministic, no training)")
    
    # Random Forest
    print("\n" + "-"*70)
    X, y_binary, sample_weights = loader.get_binary_dataset(scale=True, use_sample_weights=True)
    feature_names = loader.get_feature_names()
    
    rf_model = RandomForestModel(n_estimators=200, random_state=42)
    rf_model.train(X, y_binary, sample_weight=sample_weights, feature_names=feature_names, validation_split=0.2)
    
    # Similarity Model (KDE)
    print("\n" + "-"*70)
    X_positive = loader.get_positive_only(scale=True)
    
    sim_model = SimilarityModel(bandwidth=None, kernel='gaussian')
    sim_model.train(X_positive, cv_folds=3)
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED")
    print("="*70)
    
    return nasa_model, rf_model, sim_model


def save_models(nasa_model, rf_model, sim_model, output_dir: Path) -> None:
    """Save trained models to disk."""
    print("\nSaving models...")
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # NASA (no parameters to save, just documentation)
    with open(models_dir / 'nasa_constraints.txt', 'w') as f:
        f.write("NASA Constraints Model\n")
        f.write("======================\n")
        f.write(f"Max slope: {nasa_model.max_slope}°\n")
        f.write(f"Max roughness: {nasa_model.max_roughness}m RMS\n")
        f.write(f"Max altitude: {nasa_model.max_altitude}m\n")
    
    rf_model.save(str(models_dir / 'random_forest.pkl'))
    sim_model.save(str(models_dir / 'similarity_kde.pkl'))
    
    print(f"  ✓ Models saved to: {models_dir}")


def evaluate_candidates(
    candidate_df: pd.DataFrame,
    loader: MarsDataLoader,
    nasa_model,
    rf_model,
    sim_model,
    threshold: float = 0.5,
    batch_size: int = 50000
) -> pd.DataFrame:
    """
    Score all candidates with all three models.
    
    Args:
        candidate_df: DataFrame with candidate sites
        loader: Data loader (for feature scaling)
        nasa_model: NASA constraints classifier
        rf_model: Random Forest model
        sim_model: Similarity (KDE) model
        threshold: Classification threshold (default 0.5)
        batch_size: Process in batches to manage memory
        
    Returns:
        DataFrame with added score/prediction columns
    """
    print("\n" + "="*70)
    print("EVALUATING CANDIDATES")
    print("="*70)
    print(f"  Total candidates: {len(candidate_df):,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Classification threshold: {threshold}")
    
    # Extract features
    feature_cols = ['long_east_deg', 'lat_north_deg', 'altitude_m', 'radius_m', 'slope_deg', 'roughness_rms_m']
    X_raw = candidate_df[feature_cols].values
    
    # Scale features
    X_scaled = loader.transform_new_data(X_raw)
    
    # Initialize result arrays
    n_candidates = len(candidate_df)
    nasa_scores = np.zeros(n_candidates)
    rf_scores = np.zeros(n_candidates)
    sim_scores = np.zeros(n_candidates)
    
    # Process in batches
    n_batches = int(np.ceil(n_candidates / batch_size))
    
    print(f"\nProcessing {n_batches} batch(es)...")
    start_time = time.time()
    
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, n_candidates)
        
        X_batch_raw = X_raw[batch_start:batch_end]
        X_batch_scaled = X_scaled[batch_start:batch_end]
        
        # NASA scores (uses raw features)
        nasa_scores[batch_start:batch_end] = nasa_model.predict_proba(X_batch_raw)
        
        # RF scores (uses scaled features)
        rf_scores[batch_start:batch_end] = rf_model.predict_proba(X_batch_scaled)
        
        # Similarity scores (uses scaled features)
        sim_scores[batch_start:batch_end] = sim_model.predict_proba(X_batch_scaled)
        
        elapsed = time.time() - start_time
        rate = (batch_end / elapsed) if elapsed > 0 else 0
        eta = (n_candidates - batch_end) / rate if rate > 0 else 0
        
        print(f"  Batch {i+1}/{n_batches}: {batch_end:,}/{n_candidates:,} "
              f"({100*batch_end/n_candidates:.1f}%) | "
              f"Rate: {rate:.0f} sites/sec | ETA: {eta:.0f}s")
    
    elapsed_total = time.time() - start_time
    print(f"\n✓ Evaluation complete in {elapsed_total:.1f}s ({n_candidates/elapsed_total:.0f} sites/sec)")
    
    # Add scores to dataframe
    result_df = candidate_df.copy()
    result_df['nasa_score'] = nasa_scores
    result_df['rf_score'] = rf_scores
    result_df['sim_score'] = sim_scores
    
    # Binary classifications
    result_df['nasa_pred'] = (nasa_scores >= threshold).astype(int)
    result_df['rf_pred'] = (rf_scores >= threshold).astype(int)
    result_df['sim_pred'] = (sim_scores >= threshold).astype(int)
    
    # Consensus: how many models predict suitable (0-3)
    result_df['consensus_count'] = (
        result_df['nasa_pred'] + result_df['rf_pred'] + result_df['sim_pred']
    )
    
    # Consensus categories
    result_df['consensus_category'] = result_df['consensus_count'].map({
        0: 'all_reject',
        1: 'minority_accept',
        2: 'majority_accept',
        3: 'all_accept'
    })
    
    return result_df


def print_summary(result_df: pd.DataFrame) -> None:
    """Print evaluation summary statistics."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📊 Score Statistics:")
    for model in ['nasa', 'rf', 'sim']:
        scores = result_df[f'{model}_score']
        print(f"\n{model.upper()} Scores:")
        print(f"  Mean:   {scores.mean():.3f}")
        print(f"  Median: {scores.median():.3f}")
        print(f"  Std:    {scores.std():.3f}")
        print(f"  Min:    {scores.min():.3f}")
        print(f"  Max:    {scores.max():.3f}")
    
    print("\n" + "-"*70)
    print("🎯 Classification Results (threshold=0.5):")
    for model in ['nasa', 'rf', 'sim']:
        n_suitable = result_df[f'{model}_pred'].sum()
        pct = 100 * n_suitable / len(result_df)
        print(f"  {model.upper()}: {n_suitable:,} suitable ({pct:.1f}%)")
    
    print("\n" + "-"*70)
    print("🤝 Consensus Analysis:")
    consensus_counts = result_df['consensus_category'].value_counts()
    for category in ['all_accept', 'majority_accept', 'minority_accept', 'all_reject']:
        count = consensus_counts.get(category, 0)
        pct = 100 * count / len(result_df)
        print(f"  {category:20s}: {count:7,} ({pct:5.1f}%)")
    
    print("\n" + "-"*70)
    print("🔍 Agreement Matrix:")
    print("\nNASA vs RF:")
    print(pd.crosstab(result_df['nasa_pred'], result_df['rf_pred'], rownames=['NASA'], colnames=['RF'], margins=True))
    
    print("\nNASA vs Similarity:")
    print(pd.crosstab(result_df['nasa_pred'], result_df['sim_pred'], rownames=['NASA'], colnames=['SIM'], margins=True))
    
    print("\nRF vs Similarity:")
    print(pd.crosstab(result_df['rf_pred'], result_df['sim_pred'], rownames=['RF'], colnames=['SIM'], margins=True))


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("MARS LANDING SITE EVALUATION")
    print("Three-Model Comparison")
    print("="*70)
    
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir = Path(__file__).parent / 'output'
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print("\n📂 Loading training data...")
    loader = MarsDataLoader()
    loader.load()
    
    # Train models
    nasa_model, rf_model, sim_model = train_models(loader)
    
    # Save models
    save_models(nasa_model, rf_model, sim_model, output_dir)
    
    # Load candidate data
    print("\n📂 Loading candidate sites...")
    candidate_path = data_dir / 'candidate_data.csv'
    candidate_df = pd.read_csv(candidate_path)
    print(f"  ✓ Loaded {len(candidate_df):,} candidate sites")
    
    # Evaluate all candidates
    result_df = evaluate_candidates(
        candidate_df,
        loader,
        nasa_model,
        rf_model,
        sim_model,
        threshold=0.5,
        batch_size=50000
    )
    
    # Print summary
    print_summary(result_df)
    
    # Save results
    print("\n💾 Saving results...")
    results_path = results_dir / 'candidate_evaluations.csv'
    result_df.to_csv(results_path, index=False)
    print(f"  ✓ Saved to: {results_path}")
    print(f"  Size: {results_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save preview (first 10k rows for quick inspection)
    preview_path = results_dir / 'candidate_evaluations_preview.csv'
    result_df.head(10000).to_csv(preview_path, index=False)
    print(f"  ✓ Preview saved to: {preview_path}")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
