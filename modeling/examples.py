#!/usr/bin/env python3
"""
Example usage of the Mars Landing Suitability ML Pipeline.

This script demonstrates various ways to use the trained models:
1. Running the complete pipeline
2. Loading and using a trained ensemble
3. Scoring individual locations
4. Batch processing multiple coordinates
"""

from pathlib import Path
import pandas as pd
import numpy as np
from ensemble import LandingSuitabilityEnsemble


def example_1_run_pipeline():
    """Example 1: Run the complete training pipeline."""
    print("="*80)
    print("EXAMPLE 1: Running Complete Pipeline")
    print("="*80)
    
    from main import train_full_pipeline
    
    # Train with custom configuration
    results = train_full_pipeline(
        binary_type='random_forest',
        safety_type='threshold',
        similarity_type='kde',
        test_split=0.15,  # Faster training with smaller test set
        random_state=42
    )
    
    print(f"\n✓ Pipeline complete!")
    print(f"   Accuracy: {results['ensemble_metrics']['accuracy']:.3f}")
    print(f"   Models saved in: output/trained_models/")


def example_2_load_and_score():
    """Example 2: Load trained model and score new locations."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Loading Trained Model")
    print("="*80)
    
    # Load the ensemble
    ensemble = LandingSuitabilityEnsemble()
    model_path = Path(__file__).parent / "output" / "trained_models" / "ensemble.pkl"
    
    if not model_path.exists():
        print("❌ Model not found. Run main.py first to train models.")
        return
    
    ensemble.load(str(model_path))
    print("✓ Model loaded successfully")
    
    # Score known successful landing sites
    test_sites = {
        'Curiosity (Gale Crater)': {
            'longitude': 137.4, 'latitude': -4.5,
            'altitude': -4500, 'radius': 3396000,
            'slope': 2.5, 'roughness': 5.0
        },
        'Perseverance (Jezero Crater)': {
            'longitude': 77.4, 'latitude': 18.4,
            'altitude': -2500, 'radius': 3396000,
            'slope': 3.0, 'roughness': 6.0
        },
        'InSight (Elysium Planitia)': {
            'longitude': 135.6, 'latitude': 4.5,
            'altitude': -2600, 'radius': 3396000,
            'slope': 1.5, 'roughness': 3.0
        }
    }
    
    print("\n" + "-"*80)
    print("Scoring Known Landing Sites:")
    print("-"*80)
    
    for name, site in test_sites.items():
        result = ensemble.score_new_location(**site)
        
        print(f"\n{name}:")
        print(f"  📍 Coordinates: {site['longitude']:.1f}°E, {site['latitude']:.1f}°N")
        print(f"  🏔️  Terrain: slope={site['slope']:.1f}°, roughness={site['roughness']:.1f}m")
        print(f"  📊 Final Score: {result['final_score']:.3f}")
        print(f"     → Binary:     {result['binary_score']:.3f}")
        print(f"     → Safety:     {result['safety_score']:.3f}")
        print(f"     → Similarity: {result['similarity_score']:.3f}")
        print(f"  {'✅' if result['recommendation'] == 'suitable' else '❌'} {result['recommendation'].upper()}")


def example_3_batch_processing():
    """Example 3: Batch process multiple candidate locations."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Processing Candidate Sites")
    print("="*80)
    
    # Load model
    ensemble = LandingSuitabilityEnsemble()
    model_path = Path(__file__).parent / "output" / "trained_models" / "ensemble.pkl"
    
    if not model_path.exists():
        print("❌ Model not found. Run main.py first.")
        return
    
    ensemble.load(str(model_path))
    
    # Create candidate sites
    np.random.seed(42)
    n_candidates = 10
    
    candidates = pd.DataFrame({
        'longitude': np.random.uniform(0, 360, n_candidates),
        'latitude': np.random.uniform(-60, 60, n_candidates),
        'altitude': np.random.uniform(-5000, 2000, n_candidates),
        'radius': np.full(n_candidates, 3396000),
        'slope': np.random.uniform(0.5, 15, n_candidates),
        'roughness': np.random.uniform(1, 20, n_candidates)
    })
    
    print(f"\nProcessing {n_candidates} candidate sites...")
    
    # Score all candidates
    results = []
    for idx, row in candidates.iterrows():
        result = ensemble.score_new_location(
            longitude=row['longitude'],
            latitude=row['latitude'],
            altitude=row['altitude'],
            radius=row['radius'],
            slope=row['slope'],
            roughness=row['roughness']
        )
        results.append(result)
    
    # Compile results
    results_df = pd.DataFrame([
        {
            'site_id': i+1,
            'longitude': r['location']['longitude'],
            'latitude': r['location']['latitude'],
            'slope': r['location']['slope'],
            'roughness': r['location']['roughness'],
            'final_score': r['final_score'],
            'recommendation': r['recommendation']
        }
        for i, r in enumerate(results)
    ])
    
    # Sort by score
    results_df = results_df.sort_values('final_score', ascending=False)
    
    print("\n" + "-"*80)
    print("Top 5 Candidates by Suitability Score:")
    print("-"*80)
    print(results_df.head().to_string(index=False))
    
    # Save results
    output_path = Path(__file__).parent / "output" / "predictions" / "candidate_sites_batch.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Batch results saved to: {output_path}")


def example_4_load_predictions():
    """Example 4: Load and analyze saved predictions."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Analyzing Saved Predictions")
    print("="*80)
    
    pred_path = Path(__file__).parent / "output" / "predictions"
    csv_files = list(pred_path.glob("test_predictions_*.csv"))
    
    if not csv_files:
        print("❌ No prediction files found. Run main.py first.")
        return
    
    # Load latest predictions
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    print(f"\n✓ Loaded predictions from: {latest_file.name}")
    print(f"   Total samples: {len(df):,}")
    
    # Analyze predictions
    print("\n" + "-"*80)
    print("Prediction Analysis:")
    print("-"*80)
    
    accuracy = (df['true_label'] == df['predicted_label']).mean()
    print(f"\nOverall Accuracy: {accuracy:.3%}")
    
    print("\nPrediction Distribution:")
    print(df['recommendation'].value_counts())
    
    print("\nScore Statistics:")
    print(df['final_score'].describe())
    
    # Find highest and lowest scored sites
    print("\n" + "-"*80)
    print("Most Suitable Site (Highest Score):")
    print("-"*80)
    best = df.loc[df['final_score'].idxmax()]
    print(f"  Location: {best['longitude']:.2f}°E, {best['latitude']:.2f}°N")
    print(f"  Score: {best['final_score']:.3f}")
    print(f"  Terrain: slope={best['slope']:.2f}°, roughness={best['roughness']:.2f}m")
    
    print("\n" + "-"*80)
    print("Least Suitable Site (Lowest Score):")
    print("-"*80)
    worst = df.loc[df['final_score'].idxmin()]
    print(f"  Location: {worst['longitude']:.2f}°E, {worst['latitude']:.2f}°N")
    print(f"  Score: {worst['final_score']:.3f}")
    print(f"  Terrain: slope={worst['slope']:.2f}°, roughness={worst['roughness']:.2f}m")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" MARS LANDING SUITABILITY - USAGE EXAMPLES ".center(80))
    print("="*80)
    
    import sys
    
    # Check if model exists
    model_path = Path(__file__).parent / "output" / "trained_models" / "ensemble.pkl"
    
    if not model_path.exists():
        print("\n⚠️  No trained model found!")
        print("   Run this first: python main.py")
        print("\nOr run Example 1 to train the model:")
        if input("\n   Train model now? (y/n): ").lower() == 'y':
            example_1_run_pipeline()
        else:
            print("\n   Exiting. Run 'python main.py' to train models.")
            return
    
    # Run examples
    example_2_load_and_score()
    example_3_batch_processing()
    example_4_load_predictions()
    
    print("\n" + "="*80)
    print("✨ All examples complete!")
    print("="*80)
    print("\nFor more information:")
    print("  • See README.md for full documentation")
    print("  • Run 'python show_tree.py' to view directory structure")
    print("  • Check output/ folder for all generated files")
    print()


if __name__ == "__main__":
    main()
