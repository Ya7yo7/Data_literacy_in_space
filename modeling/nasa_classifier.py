"""nasa_classifier.py

Deterministic NASA constraints-based classifier.

NASA landing site constraints:
- Slope < 5 degrees
- Roughness < 8 meters RMS
- Altitude < -1300 meters (relative to MOLA areoid)

Uses soft scoring with logistic smoothing around thresholds.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class NASAConstraintsClassifier:
    """NASA constraints-based landing site classifier."""
    
    def __init__(self):
        # NASA constraints
        self.max_slope = 5.0  # degrees
        self.max_roughness = 8.0  # meters RMS
        self.max_altitude = -1300.0  # meters
        
        # Logistic smoothing parameters (steepness around threshold)
        self.slope_k = 2.0
        self.roughness_k = 1.0
        self.altitude_k = 0.002
        
    def score_slope(self, slope: np.ndarray) -> np.ndarray:
        """Score based on slope constraint (lower is better)."""
        # Logistic function: 1 / (1 + exp(k * (x - threshold)))
        # Clip to prevent overflow (exp(x) overflows for x > ~700)
        exponent = np.clip(self.slope_k * (slope - self.max_slope), -700, 700)
        return 1.0 / (1.0 + np.exp(exponent))
    
    def score_roughness(self, roughness: np.ndarray) -> np.ndarray:
        """Score based on roughness constraint (lower is better)."""
        # Clip to prevent overflow
        exponent = np.clip(self.roughness_k * (roughness - self.max_roughness), -700, 700)
        return 1.0 / (1.0 + np.exp(exponent))
    
    def score_altitude(self, altitude: np.ndarray) -> np.ndarray:
        """Score based on altitude constraint (lower is better)."""
        # Clip to prevent overflow
        exponent = np.clip(self.altitude_k * (altitude - self.max_altitude), -700, 700)
        return 1.0 / (1.0 + np.exp(exponent))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict suitability probabilities for landing sites.
        
        Args:
            X: Feature matrix with columns [long, lat, alt, radius, slope, roughness]
            
        Returns:
            Array of suitability scores [0, 1]
        """
        slope = X[:, 4]  # slope_deg
        roughness = X[:, 5]  # roughness_rms_m
        altitude = X[:, 2]  # altitude_m
        
        # Individual constraint scores
        slope_score = self.score_slope(slope)
        roughness_score = self.score_roughness(roughness)
        altitude_score = self.score_altitude(altitude)
        
        # Overall suitability: geometric mean of individual scores
        suitability_score = (slope_score * roughness_score * altitude_score) ** (1/3)
        
        return suitability_score
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary suitability classification.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions (1=suitable, 0=unsuitable)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def classify_detailed(self, slope: np.ndarray, roughness: np.ndarray, altitude: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Classify landing sites with detailed constraint breakdown.
        
        Args:
            slope: Surface slope in degrees
            roughness: Surface roughness in meters RMS
            altitude: Altitude in meters relative to MOLA areoid
            
        Returns:
            Dictionary with individual scores, overall score, and categories
        """
        # Individual constraint scores
        slope_score = self.score_slope(slope)
        roughness_score = self.score_roughness(roughness)
        altitude_score = self.score_altitude(altitude)
        
        # Overall suitability: geometric mean of individual scores
        suitability_score = (slope_score * roughness_score * altitude_score) ** (1/3)
        
        # Categorize
        category = np.where(
            suitability_score >= 0.7, 'suitable',
            np.where(suitability_score >= 0.3, 'marginal', 'unsuitable')
        )
        
        return {
            'suitability_score': suitability_score,
            'category': category,
            'slope_score': slope_score,
            'roughness_score': roughness_score,
            'altitude_score': altitude_score
        }
    
    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a dataframe with landing site data.
        
        Args:
            df: DataFrame with columns 'slope_deg', 'roughness_rms_m', 'altitude_m'
            
        Returns:
            DataFrame with added NASA classification columns
        """
        results = self.classify_detailed(
            df['slope_deg'].values,
            df['roughness_rms_m'].values,
            df['altitude_m'].values
        )
        
        df_out = df.copy()
        for key, value in results.items():
            df_out[f'nasa_{key}'] = value
        
        return df_out
    
    def summarize(self, categories: np.ndarray) -> None:
        """Print summary statistics."""
        unique, counts = np.unique(categories, return_counts=True)
        print("\nNASA Classification Summary:")
        for cat, count in zip(unique, counts):
            print(f"  {cat:12s}: {count:6,} ({100*count/len(categories):.1f}%)")


if __name__ == "__main__":
    # Test the classifier
    print("="*70)
    print("TESTING NASA CONSTRAINTS CLASSIFIER")
    print("="*70)
    
    # Create test data
    np.random.seed(42)
    n = 1000
    test_data = pd.DataFrame({
        'slope_deg': np.random.uniform(0, 10, n),
        'roughness_rms_m': np.random.uniform(0, 15, n),
        'altitude_m': np.random.uniform(-3000, 0, n)
    })
    
    classifier = NASAConstraintsClassifier()
    results = classifier.evaluate_dataframe(test_data)
    
    print(f"\nEvaluated {len(results)} sites")
    classifier.summarize(results['nasa_category'].values)
    
    print(f"\nSample results:")
    print(results[['slope_deg', 'roughness_rms_m', 'altitude_m', 'nasa_suitability_score', 'nasa_category']].head(10))
    
    print("\n✓ NASA classifier test complete")
