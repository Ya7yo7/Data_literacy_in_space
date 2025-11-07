"""
Main file from where we can run each model version, print results, print figures etc 
choosing model or choosing data set as we see fit. In particular, we can comment 
things out as we see fit.

This orchestrates the complete ML pipeline for Mars landing suitability classification:
1. Load and prepare data with spatial cross-validation
2. Train binary classifier (Random Forest / Gradient Boosting)
3. Train safety filter (threshold-based or One-Class SVM)
4. Train similarity model (KDE or GMM)
5. Combine into ensemble
6. Evaluate and visualize results
7. Save trained models for inference
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from data_loader import load_data
from binary_classifier import BinaryLandingSuitabilityClassifier
from safety_filter import SafetyFilter
from similarity_model import SimilarityModel
from ensemble import LandingSuitabilityEnsemble


# Directory structure
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS_DIR = OUTPUT_DIR / "trained_models"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Create all output directories
for dir_path in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR, PREDICTIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def train_full_pipeline(binary_type='random_forest', safety_type='threshold', 
                       similarity_type='kde', test_split=0.2, random_state=42):
    """
    Train complete ML pipeline.
    
    Args:
        binary_type: 'random_forest' or 'gradient_boosting'
        safety_type: 'threshold', 'one_class_svm', or 'isolation_forest'
        similarity_type: 'kde' or 'gmm'
        test_split: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    print("="*80)
    print("MARS LANDING SUITABILITY ML PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Binary classifier: {binary_type}")
    print(f"  Safety filter:     {safety_type}")
    print(f"  Similarity model:  {similarity_type}")
    print(f"  Test split:        {test_split}")
    print("="*80)
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    loader = load_data()
    
    # Get datasets
    X, y_multi = loader.get_multiclass_dataset(scale=True)
    y_binary = loader.y_binary
    feature_names = loader.get_feature_names()
    
    # Stratified train/test split to ensure balanced classes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_binary, y_test_binary, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multi, test_size=test_split, random_state=random_state, stratify=y_binary
    )
    
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set:  {len(X_test):,} samples")
    
    # =========================================================================
    # STEP 2: TRAIN BINARY CLASSIFIER
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: TRAINING BINARY CLASSIFIER")
    print("="*80)
    
    binary_clf = BinaryLandingSuitabilityClassifier(
        model_type=binary_type, 
        random_state=random_state
    )
    binary_clf.train(X_train, y_train_binary, feature_names=feature_names)
    
    print("\nEvaluating binary classifier on test set...")
    binary_metrics = binary_clf.evaluate(X_test, y_test_binary)
    binary_importances = binary_clf.get_feature_importance()
    
    # =========================================================================
    # STEP 3: TRAIN SAFETY FILTER
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING SAFETY FILTER")
    print("="*80)
    
    # Split training data by label
    X_safe_train = X_train[y_train_multi == 1]
    X_unsafe_train = X_train[y_train_multi == -1]
    
    # For test: get safe and unsafe examples
    X_safe_test = X_test[y_test_multi == 1]
    X_unsafe_test = X_test[y_test_multi == -1]
    
    safety = SafetyFilter(method=safety_type, random_state=random_state)
    safety.train(X_safe_train, X_unsafe_train)
    
    print("\nEvaluating safety filter on test set...")
    if len(X_safe_test) > 0 and len(X_unsafe_test) > 0:
        safety_metrics = safety.evaluate(X_safe_test, X_unsafe_test)
    else:
        print("  Warning: Not enough test examples for safety evaluation")
        safety_metrics = {}
    
    # =========================================================================
    # STEP 4: TRAIN SIMILARITY MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING SIMILARITY MODEL")
    print("="*80)
    
    similarity = SimilarityModel(method=similarity_type, random_state=random_state)
    similarity.train(X_safe_train)
    
    print("\nEvaluating similarity model on test set...")
    # Evaluate on positive vs all negative test examples
    X_negative_test = X_test[y_test_multi != 1]
    
    if len(X_safe_test) > 0 and len(X_negative_test) > 0:
        # Sample negatives for faster evaluation
        n_neg_sample = min(10000, len(X_negative_test))
        X_neg_sample = X_negative_test[:n_neg_sample]
        similarity_metrics = similarity.evaluate(X_safe_test, X_neg_sample)
    else:
        print("  Warning: Not enough test examples for similarity evaluation")
        similarity_metrics = {}
    
    # =========================================================================
    # STEP 5: CREATE ENSEMBLE
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: CREATING ENSEMBLE MODEL")
    print("="*80)
    
    ensemble = LandingSuitabilityEnsemble(
        binary_classifier=binary_clf,
        safety_filter=safety,
        similarity_model=similarity,
        weights={'binary': 0.4, 'safety': 0.3, 'similarity': 0.3}
    )
    
    print("\nEvaluating ensemble on test set...")
    ensemble_metrics = ensemble.evaluate(X_test, y_test_binary, verbose=True)
    
    # =========================================================================
    # STEP 6: SAVE MODELS AND RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING MODELS AND RESULTS")
    print("="*80)
    
    # Save trained models
    binary_clf.save(str(MODELS_DIR / f"binary_{binary_type}.pkl"))
    safety.save(str(MODELS_DIR / f"safety_{safety_type}.pkl"))
    similarity.save(str(MODELS_DIR / f"similarity_{similarity_type}.pkl"))
    ensemble.save(str(MODELS_DIR / "ensemble.pkl"))
    
    print(f"\n✓ Models saved to: {MODELS_DIR}")
    
    # Save predictions on test set
    predictions_df = pd.DataFrame({
        'longitude': X_test[:, 0],
        'latitude': X_test[:, 1],
        'altitude': X_test[:, 2],
        'radius': X_test[:, 3],
        'slope': X_test[:, 4],
        'roughness': X_test[:, 5],
        'true_label': y_test_binary,
        'predicted_label': ensemble_metrics['y_pred'],
        'final_score': ensemble_metrics['final_score'],
        'binary_score': ensemble_metrics['predictions']['binary_score'],
        'safety_score': ensemble_metrics['predictions']['safety_score'],
        'similarity_score': ensemble_metrics['predictions']['similarity_score'],
        'recommendation': ensemble_metrics['predictions']['recommendation']
    })
    
    predictions_file = PREDICTIONS_DIR / f"test_predictions_{binary_type}_{safety_type}_{similarity_type}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Test predictions saved to: {predictions_file}")
    
    # Save metrics summary
    metrics_summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'binary_type': binary_type,
            'safety_type': safety_type,
            'similarity_type': similarity_type,
            'test_split': test_split,
            'random_state': random_state
        },
        'dataset_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_names),
            'features': feature_names
        },
        'binary_classifier_metrics': {
            'accuracy': float(binary_metrics['accuracy']),
            'roc_auc': float(binary_metrics['roc_auc']),
            'pr_auc': float(binary_metrics['pr_auc'])
        },
        'safety_filter_metrics': {
            'safe_accuracy': float(safety_metrics.get('safe_accuracy', 0)),
            'unsafe_accuracy': float(safety_metrics.get('unsafe_accuracy', 0)),
            'overall_accuracy': float(safety_metrics.get('overall_accuracy', 0))
        } if safety_metrics else {},
        'similarity_model_metrics': {
            'separation': float(similarity_metrics.get('separation', 0)),
            'mean_positive': float(similarity_metrics.get('mean_positive', 0)),
            'mean_negative': float(similarity_metrics.get('mean_negative', 0))
        } if similarity_metrics else {},
        'ensemble_metrics': {
            'accuracy': float(ensemble_metrics['accuracy']),
            'precision': float(ensemble_metrics['precision']),
            'recall': float(ensemble_metrics['recall']),
            'f1': float(ensemble_metrics['f1']),
            'confusion_matrix': {
                'tp': int(ensemble_metrics['tp']),
                'fp': int(ensemble_metrics['fp']),
                'tn': int(ensemble_metrics['tn']),
                'fn': int(ensemble_metrics['fn'])
            }
        },
        'feature_importance': binary_importances
    }
    
    metrics_file = RESULTS_DIR / f"metrics_{binary_type}_{safety_type}_{similarity_type}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Metrics summary saved to: {metrics_file}")
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    results = {
        'loader': loader,
        'binary_classifier': binary_clf,
        'safety_filter': safety,
        'similarity_model': similarity,
        'ensemble': ensemble,
        'binary_metrics': binary_metrics,
        'safety_metrics': safety_metrics,
        'similarity_metrics': similarity_metrics,
        'ensemble_metrics': ensemble_metrics,
        'binary_importances': binary_importances,
        'X_test': X_test,
        'y_test_binary': y_test_binary,
        'y_test_multi': y_test_multi
    }
    
    return results


def plot_feature_importance(importances_dict, save_path=None):
    """Plot feature importance from binary classifier."""
    features = list(importances_dict.keys())
    importances = list(importances_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.title('Feature Importance (Binary Classifier)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Feature importance plot saved: {Path(save_path).name}")
        plt.close()
    else:
        plt.show()


def plot_score_distributions(results, save_path=None):
    """Plot score distributions for suitable vs unsuitable sites."""
    ensemble_metrics = results['ensemble_metrics']
    predictions = ensemble_metrics['predictions']
    y_test = results['y_test_binary']
    
    suitable_scores = predictions['final_score'][y_test == 1]
    unsuitable_scores = predictions['final_score'][y_test == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(unsuitable_scores, bins=50, alpha=0.7, label='Unsuitable (y=0)', color='#e74c3c', edgecolor='black')
    plt.hist(suitable_scores, bins=50, alpha=0.7, label='Suitable (y=1)', color='#27ae60', edgecolor='black')
    plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Suitability Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Score Distribution by Class', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Score distribution plot saved: {Path(save_path).name}")
        plt.close()
    else:
        plt.show()


def plot_component_scores(results, sample_size=1000, save_path=None):
    """Plot individual component scores vs final ensemble score."""
    ensemble_metrics = results['ensemble_metrics']
    predictions = ensemble_metrics['predictions']
    y_test = results['y_test_binary']
    
    # Sample for readability
    if len(y_test) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(y_test), sample_size, replace=False)
    else:
        indices = np.arange(len(y_test))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Binary score
    scatter = axes[0, 0].scatter(predictions['binary_score'][indices], 
                      predictions['final_score'][indices],
                      c=y_test[indices], cmap='RdYlGn', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Binary Classifier Score', fontsize=11)
    axes[0, 0].set_ylabel('Final Ensemble Score', fontsize=11)
    axes[0, 0].set_title('Binary Classifier vs Ensemble', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Safety score
    axes[0, 1].scatter(predictions['safety_score'][indices], 
                      predictions['final_score'][indices],
                      c=y_test[indices], cmap='RdYlGn', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Safety Filter Score', fontsize=11)
    axes[0, 1].set_ylabel('Final Ensemble Score', fontsize=11)
    axes[0, 1].set_title('Safety Filter vs Ensemble', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Similarity score
    axes[1, 0].scatter(predictions['similarity_score'][indices], 
                      predictions['final_score'][indices],
                      c=y_test[indices], cmap='RdYlGn', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Similarity Model Score', fontsize=11)
    axes[1, 0].set_ylabel('Final Ensemble Score', fontsize=11)
    axes[1, 0].set_title('Similarity Model vs Ensemble', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # All three components
    scatter2 = axes[1, 1].scatter(predictions['binary_score'][indices], 
                      predictions['safety_score'][indices],
                      c=predictions['final_score'][indices], 
                      cmap='RdYlGn', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Binary Score', fontsize=11)
    axes[1, 1].set_ylabel('Safety Score', fontsize=11)
    axes[1, 1].set_title('Binary vs Safety (colored by Final Score)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1, 1], label='Final Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Component scores plot saved: {Path(save_path).name}")
        plt.close()
    else:
        plt.show()


def test_known_locations(ensemble):
    """Test ensemble on known Mars landing sites."""
    print("\n" + "="*80)
    print("TESTING ON KNOWN LANDING SITES")
    print("="*80)
    
    # Known successful landing sites (approximate coordinates and terrain)
    test_sites = [
        {
            'name': 'Curiosity (Gale Crater)',
            'longitude': 137.4,
            'latitude': -4.5,
            'altitude': -4500,
            'radius': 3396000,
            'slope': 2.5,
            'roughness': 5.0
        },
        {
            'name': 'Perseverance (Jezero Crater)',
            'longitude': 77.4,
            'latitude': 18.4,
            'altitude': -2500,
            'radius': 3396000,
            'slope': 3.0,
            'roughness': 6.0
        },
        {
            'name': 'InSight (Elysium Planitia)',
            'longitude': 135.6,
            'latitude': 4.5,
            'altitude': -2600,
            'radius': 3396000,
            'slope': 1.5,
            'roughness': 3.0
        },
        {
            'name': 'Hypothetical Unsafe Site',
            'longitude': 100.0,
            'latitude': -30.0,
            'altitude': 5000,
            'radius': 3396000,
            'slope': 25.0,  # Very steep
            'roughness': 30.0  # Very rough
        }
    ]
    
    for site in test_sites:
        result = ensemble.score_new_location(
            longitude=site['longitude'],
            latitude=site['latitude'],
            altitude=site['altitude'],
            radius=site['radius'],
            slope=site['slope'],
            roughness=site['roughness']
        )
        
        print(f"\n{site['name']}:")
        print(f"  Coordinates: ({site['longitude']:.1f}°E, {site['latitude']:.1f}°N)")
        print(f"  Terrain: slope={site['slope']:.1f}°, roughness={site['roughness']:.1f}m")
        print(f"  Final Score: {result['final_score']:.3f}")
        print(f"    Binary:     {result['binary_score']:.3f}")
        print(f"    Safety:     {result['safety_score']:.3f}")
        print(f"    Similarity: {result['similarity_score']:.3f}")
        print(f"  → Recommendation: {result['recommendation'].upper()}")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print(" MARS LANDING SUITABILITY ML PIPELINE ".center(80))
    print("="*80)
    
    # Train pipeline with default configuration
    results = train_full_pipeline(
        binary_type='random_forest',
        safety_type='threshold',
        similarity_type='kde',
        test_split=0.2,
        random_state=42
    )
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\nSaving plots to:", PLOTS_DIR)
    
    plot_feature_importance(
        results['binary_importances'],
        save_path=PLOTS_DIR / "feature_importance.png"
    )
    
    plot_score_distributions(
        results,
        save_path=PLOTS_DIR / "score_distributions.png"
    )
    
    plot_component_scores(
        results,
        save_path=PLOTS_DIR / "component_scores.png"
    )
    
    # Test on known locations
    test_known_locations(results['ensemble'])
    
    # Print directory tree
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Directory Structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── trained_models/")
    print(f"   │   ├── binary_random_forest.pkl")
    print(f"   │   ├── safety_threshold.pkl")
    print(f"   │   ├── similarity_kde.pkl")
    print(f"   │   └── ensemble.pkl")
    print(f"   ├── plots/")
    print(f"   │   ├── feature_importance.png")
    print(f"   │   ├── score_distributions.png")
    print(f"   │   └── component_scores.png")
    print(f"   ├── results/")
    print(f"   │   └── metrics_random_forest_threshold_kde.json")
    print(f"   └── predictions/")
    print(f"       └── test_predictions_random_forest_threshold_kde.csv")
    
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    
    return results


if __name__ == "__main__":
    main()
