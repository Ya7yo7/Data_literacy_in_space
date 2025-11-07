"""similarity_model.py

Similarity-based model to find locations similar to successful landing sites.
Uses Kernel Density Estimation or Gaussian Mixture Models.
"""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional
import pickle
from pathlib import Path


class SimilarityModel:
    """Model to score similarity to successful landing sites."""
    
    def __init__(self, method: str = 'kde', n_components: int = 5, 
                 bandwidth: Optional[float] = None, random_state: int = 42):
        """
        Args:
            method: 'kde' (Kernel Density Estimation) or 'gmm' (Gaussian Mixture)
            n_components: Number of components for GMM
            bandwidth: KDE bandwidth (auto-tuned if None)
            random_state: Random seed
        """
        self.method = method
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.random_state = random_state
        
        if method == 'kde':
            # Will set bandwidth during training if None
            self.model = None  # KDE model will be created during training
        elif method == 'gmm':
            self.model = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=random_state,
                max_iter=200
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train(self, X_positive: np.ndarray) -> 'SimilarityModel':
        """
        Train similarity model on successful landing sites.
        
        Args:
            X_positive: Features from successful landing sites
        """
        print(f"\nTraining similarity model (method={self.method})...")
        print(f"  Positive examples: {len(X_positive):,}")
        
        if self.method == 'kde':
            # Auto-tune bandwidth using Scott's rule if not provided
            if self.bandwidth is None:
                n, d = X_positive.shape
                self.bandwidth = n ** (-1.0 / (d + 4))
                print(f"  Auto-tuned bandwidth: {self.bandwidth:.4f}")
            
            self.model = KernelDensity(
                bandwidth=self.bandwidth,
                kernel='gaussian',
                metric='euclidean'
            )
            self.model.fit(X_positive)
        elif self.method == 'gmm':
            self.model.fit(X_positive)
            if hasattr(self.model, 'converged_'):
                print(f"  Converged: {self.model.converged_}")
            if hasattr(self.model, 'n_iter_'):
                print(f"  Iterations: {self.model.n_iter_}")
        
        print("  ✓ Training complete")
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood scores (higher = more similar to positive sites).
        
        Args:
            X: Features to score
            
        Returns:
            Log-likelihood scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.score_samples(X)
    
    def get_similarity_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get normalized similarity score in [0, 1] range.
        
        Higher scores indicate greater similarity to successful landing sites.
        """
        log_likelihood = self.score(X)
        
        # Normalize using sigmoid to [0, 1]
        # Adjust scaling factor based on typical log-likelihood range
        similarity = 1 / (1 + np.exp(-log_likelihood / 10.0))
        
        return similarity
    
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        For GMM, predict which cluster each point belongs to.
        
        Returns:
            Cluster assignments (0 to n_components-1)
        """
        if self.method != 'gmm':
            raise ValueError("Cluster prediction only available for GMM method")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_positive: np.ndarray, X_negative: np.ndarray) -> Dict:
        """
        Evaluate similarity model.
        
        Positive examples should get higher scores than negative examples.
        """
        score_pos = self.score(X_positive)
        score_neg = self.score(X_negative)
        
        mean_pos = np.mean(score_pos)
        mean_neg = np.mean(score_neg)
        std_pos = np.std(score_pos)
        std_neg = np.std(score_neg)
        
        # Separation metric: how well separated are the distributions?
        separation = (mean_pos - mean_neg) / np.sqrt(std_pos**2 + std_neg**2)
        
        print(f"\nSIMILARITY MODEL EVALUATION ({self.method})")
        print("="*60)
        print(f"Positive examples (successful sites):")
        print(f"  Mean log-likelihood: {mean_pos:.2f} ± {std_pos:.2f}")
        print(f"  Min: {np.min(score_pos):.2f}, Max: {np.max(score_pos):.2f}")
        
        print(f"\nNegative examples:")
        print(f"  Mean log-likelihood: {mean_neg:.2f} ± {std_neg:.2f}")
        print(f"  Min: {np.min(score_neg):.2f}, Max: {np.max(score_neg):.2f}")
        
        print(f"\nSeparation metric: {separation:.2f}")
        print(f"  (Higher is better; >2.0 is good separation)")
        
        # Compute percentile threshold
        threshold_90 = np.percentile(score_pos, 10)  # 90% of positives above this
        pos_above = (score_pos >= threshold_90).sum()
        neg_above = (score_neg >= threshold_90).sum()
        
        print(f"\nAt 90th percentile threshold ({threshold_90:.2f}):")
        print(f"  Positives above: {pos_above:,} / {len(X_positive):,} ({100*pos_above/len(X_positive):.1f}%)")
        print(f"  Negatives above: {neg_above:,} / {len(X_negative):,} ({100*neg_above/len(X_negative):.1f}%)")
        
        metrics = {
            'mean_positive': mean_pos,
            'mean_negative': mean_neg,
            'std_positive': std_pos,
            'std_negative': std_neg,
            'separation': separation,
            'threshold_90': threshold_90,
            'score_positive': score_pos,
            'score_negative': score_neg
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_obj = {
            'method': self.method,
            'model': self.model,
            'n_components': self.n_components,
            'bandwidth': self.bandwidth,
            'random_state': self.random_state
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f"Similarity model saved to: {filepath}")
    
    def load(self, filepath: str) -> 'SimilarityModel':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.method = save_obj['method']
        self.model = save_obj['model']
        self.n_components = save_obj['n_components']
        self.bandwidth = save_obj['bandwidth']
        self.random_state = save_obj['random_state']
        
        print(f"Similarity model loaded from: {filepath}")
        return self


if __name__ == "__main__":
    print("="*70)
    print("TESTING SIMILARITY MODEL")
    print("="*70)
    
    from data_loader import load_data
    
    # Load data
    loader = load_data()
    X, y = loader.get_multiclass_dataset(scale=True)
    
    # Split by label
    X_positive = X[y == 1]  # Successful sites
    X_negative = X[y != 1]  # All negatives
    
    # Train/test split for positives
    from sklearn.model_selection import train_test_split
    X_pos_train, X_pos_test = train_test_split(
        X_positive, test_size=0.2, random_state=42
    )
    
    # Test negatives (sample for speed)
    n_neg_test = min(10000, len(X_negative))
    X_neg_test = X_negative[:n_neg_test]
    
    # Test KDE
    print("\n" + "="*70)
    print("KERNEL DENSITY ESTIMATION")
    print("="*70)
    kde_model = SimilarityModel(method='kde')
    kde_model.train(X_pos_train)
    kde_metrics = kde_model.evaluate(X_pos_test, X_neg_test)
    
    # Test GMM
    print("\n" + "="*70)
    print("GAUSSIAN MIXTURE MODEL")
    print("="*70)
    gmm_model = SimilarityModel(method='gmm', n_components=5)
    gmm_model.train(X_pos_train)
    gmm_metrics = gmm_model.evaluate(X_pos_test, X_neg_test)
    
    print("\n✓ Similarity model test complete")
