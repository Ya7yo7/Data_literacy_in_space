"""similarity_model.py

KDE-based similarity model for Mars landing site suitability.

Philosophy: Instead of learning what's unsuitable, learn what successful landing sites look like.
Trains ONLY on positive examples (historical successful missions).
Scores candidates by similarity to known safe sites using Kernel Density Estimation.
"""
import numpy as np
import pickle
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from typing import Optional


class SimilarityModel:
    """KDE-based similarity model (trains on positive examples only)."""
    
    def __init__(self, bandwidth: Optional[float] = None, kernel: str = 'gaussian'):
        """
        Args:
            bandwidth: KDE bandwidth parameter (None = auto-tune via CV)
            kernel: Kernel type ('gaussian', 'tophat', 'epanechnikov')
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
        self.is_trained = False
        self.log_density_range = None  # For normalizing scores to [0, 1]
        
    def train(self, X_positive: np.ndarray, cv_folds: int = 5) -> None:
        """
        Train KDE on positive examples only.
        
        Args:
            X_positive: Feature matrix of successful landing sites (scaled)
            cv_folds: Number of CV folds for bandwidth selection (if bandwidth=None)
        """
        print(f"Training Similarity Model (KDE)...")
        print(f"  Positive examples: {len(X_positive):,}")
        print(f"  Kernel: {self.kernel}")
        
        if self.bandwidth is None:
            # Auto-tune bandwidth via cross-validation
            print(f"  Auto-tuning bandwidth via {cv_folds}-fold CV...")
            bandwidths = np.logspace(-1, 1, 20)
            grid = GridSearchCV(
                KernelDensity(kernel=self.kernel),
                {'bandwidth': bandwidths},
                cv=cv_folds,
                n_jobs=-1
            )
            grid.fit(X_positive)
            self.bandwidth = grid.best_params_['bandwidth']
            self.kde = grid.best_estimator_
            print(f"  Best bandwidth: {self.bandwidth:.4f}")
        else:
            # Use provided bandwidth
            self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
            self.kde.fit(X_positive)
            print(f"  Using bandwidth: {self.bandwidth:.4f}")
        
        # Compute log density range for normalization
        log_densities = self.kde.score_samples(X_positive)
        self.log_density_range = (log_densities.min(), log_densities.max())
        print(f"  Log density range: [{self.log_density_range[0]:.2f}, {self.log_density_range[1]:.2f}]")
        
        self.is_trained = True
        print(f"  ✓ Training complete")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict similarity scores (probability-like, but not true probabilities).
        
        Higher score = more similar to successful landing sites.
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Array of similarity scores [0, 1]
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Compute log densities
        log_densities = self.kde.score_samples(X)
        
        # Normalize to [0, 1] range based on training data range
        min_log_dens, max_log_dens = self.log_density_range
        
        # Clip to avoid extrapolation issues
        log_densities = np.clip(log_densities, min_log_dens - 5, max_log_dens + 1)
        
        # Normalize
        if max_log_dens > min_log_dens:
            scores = (log_densities - min_log_dens) / (max_log_dens - min_log_dens)
        else:
            scores = np.ones_like(log_densities) * 0.5
        
        return scores
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary suitability classification.
        
        Args:
            X: Feature matrix (scaled)
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions (1=suitable, 0=unsuitable)
        """
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'kde': self.kde,
                'bandwidth': self.bandwidth,
                'kernel': self.kernel,
                'log_density_range': self.log_density_range,
                'is_trained': self.is_trained
            }, f)
        print(f"  ✓ Saved model to: {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.kde = data['kde']
        self.bandwidth = data['bandwidth']
        self.kernel = data['kernel']
        self.log_density_range = data['log_density_range']
        self.is_trained = data['is_trained']
        print(f"  ✓ Loaded model from: {path}")


if __name__ == "__main__":
    # Test with dummy data
    print("="*70)
    print("TESTING SIMILARITY MODEL (KDE)")
    print("="*70)
    
    from data_loader import load_data
    
    # Load positive examples only
    loader = load_data()
    X_positive = loader.get_positive_only(scale=True)
    
    # Train model
    sim_model = SimilarityModel(bandwidth=None, kernel='gaussian')
    sim_model.train(X_positive, cv_folds=3)
    
    # Test predictions on positive data (should score high)
    scores = sim_model.predict_proba(X_positive[:100])
    print(f"\nSimilarity scores for positive examples (first 100):")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Min:  {scores.min():.3f}")
    print(f"  Max:  {scores.max():.3f}")
    
    print("\n✓ Similarity model test complete")
