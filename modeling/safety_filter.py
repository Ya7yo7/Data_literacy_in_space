"""safety_filter.py

Safety filter to identify unsafe terrain for Mars landing.
Uses One-Class SVM or threshold-based approach on hard negatives.
"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional
import pickle
from pathlib import Path


class SafetyFilter:
    """Filter to detect unsafe landing terrain."""
    
    def __init__(self, method: str = 'one_class_svm', contamination: float = 0.1, 
                 random_state: int = 42):
        """
        Args:
            method: 'one_class_svm', 'isolation_forest', or 'threshold'
            contamination: Expected proportion of outliers (for isolation_forest)
            random_state: Random seed
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.thresholds = None  # For threshold method
        
        if method == 'one_class_svm':
            self.model = OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=0.1  # Upper bound on training errors
            )
        elif method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1
            )
        elif method == 'threshold':
            # Will compute thresholds during training
            pass
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train(self, X_safe: np.ndarray, X_unsafe: Optional[np.ndarray] = None) -> 'SafetyFilter':
        """
        Train safety filter.
        
        Args:
            X_safe: Safe examples (positive landing sites)
            X_unsafe: Unsafe examples (hard negatives) - optional for threshold method
        """
        print(f"\nTraining safety filter (method={self.method})...")
        print(f"  Safe examples: {len(X_safe):,}")
        if X_unsafe is not None:
            print(f"  Unsafe examples: {len(X_unsafe):,}")
        
        if self.method == 'threshold':
            # Compute threshold-based boundaries from safe examples
            # Unsafe regions have extreme slope or roughness
            self._compute_thresholds(X_safe, X_unsafe)
        else:
            # Train on safe examples only (novelty detection)
            self.model.fit(X_safe)
        
        print("  ✓ Training complete")
        return self
    
    def _compute_thresholds(self, X_safe: np.ndarray, X_unsafe: Optional[np.ndarray] = None):
        """Compute safety thresholds based on safe and unsafe distributions."""
        # Assume features are: [long, lat, altitude, radius, slope, roughness]
        # Focus on slope (index 4) and roughness (index 5)
        
        safe_slope = X_safe[:, 4]
        safe_roughness = X_safe[:, 5]
        
        # Conservative thresholds: mean + 2*std of safe examples
        self.thresholds = {
            'slope_max': np.mean(safe_slope) + 2 * np.std(safe_slope),
            'roughness_max': np.mean(safe_roughness) + 2 * np.std(safe_roughness),
            'slope_percentile_95': np.percentile(safe_slope, 95),
            'roughness_percentile_95': np.percentile(safe_roughness, 95)
        }
        
        print(f"\n  Computed thresholds:")
        print(f"    Slope max (mean+2σ):    {self.thresholds['slope_max']:.2f}°")
        print(f"    Roughness max (mean+2σ): {self.thresholds['roughness_max']:.2f}m")
        print(f"    Slope 95th percentile:   {self.thresholds['slope_percentile_95']:.2f}°")
        print(f"    Roughness 95th percentile: {self.thresholds['roughness_percentile_95']:.2f}m")
        
        if X_unsafe is not None:
            unsafe_slope = X_unsafe[:, 4]
            unsafe_roughness = X_unsafe[:, 5]
            print(f"\n  Unsafe statistics:")
            print(f"    Slope mean:    {np.mean(unsafe_slope):.2f}° (std: {np.std(unsafe_slope):.2f}°)")
            print(f"    Roughness mean: {np.mean(unsafe_roughness):.2f}m (std: {np.std(unsafe_roughness):.2f}m)")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict safety: 1 = safe, -1 = unsafe.
        
        Args:
            X: Features to evaluate
            
        Returns:
            Array of predictions (1 = safe, -1 = unsafe)
        """
        if self.method == 'threshold':
            if self.thresholds is None:
                raise ValueError("Thresholds not computed. Call train() first.")
            
            # Use 95th percentile thresholds
            slope = X[:, 4]
            roughness = X[:, 5]
            
            safe_mask = (slope <= self.thresholds['slope_percentile_95']) & \
                       (roughness <= self.thresholds['roughness_percentile_95'])
            
            predictions = np.where(safe_mask, 1, -1)
            return predictions
        else:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            return self.model.predict(X)
    
    def get_safety_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get safety score in [0, 1] range (1 = safe, 0 = unsafe).
        
        For threshold method, this is a soft score based on distance from thresholds.
        For ML methods, uses decision function.
        """
        if self.method == 'threshold':
            if self.thresholds is None:
                raise ValueError("Thresholds not computed. Call train() first.")
            
            slope = X[:, 4]
            roughness = X[:, 5]
            
            # Normalize by thresholds
            slope_score = np.clip(1 - slope / self.thresholds['slope_percentile_95'], 0, 1)
            roughness_score = np.clip(1 - roughness / self.thresholds['roughness_percentile_95'], 0, 1)
            
            # Combined score (geometric mean)
            safety_score = np.sqrt(slope_score * roughness_score)
            return safety_score
        else:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Use decision function (higher = more normal/safe)
            decision = self.model.decision_function(X)
            
            # Normalize to [0, 1] using sigmoid-like transformation
            safety_score = 1 / (1 + np.exp(-decision))
            return safety_score
    
    def evaluate(self, X_safe_test: np.ndarray, X_unsafe_test: np.ndarray) -> Dict:
        """Evaluate safety filter on test sets."""
        pred_safe = self.predict(X_safe_test)
        pred_unsafe = self.predict(X_unsafe_test)
        
        # Compute accuracy
        safe_correct = (pred_safe == 1).sum()
        unsafe_correct = (pred_unsafe == -1).sum()
        
        safe_acc = safe_correct / len(X_safe_test)
        unsafe_acc = unsafe_correct / len(X_unsafe_test)
        overall_acc = (safe_correct + unsafe_correct) / (len(X_safe_test) + len(X_unsafe_test))
        
        print(f"\nSAFETY FILTER EVALUATION ({self.method})")
        print("="*60)
        print(f"Safe set (should predict 1):")
        print(f"  Correct: {safe_correct:,} / {len(X_safe_test):,} ({100*safe_acc:.1f}%)")
        print(f"\nUnsafe set (should predict -1):")
        print(f"  Correct: {unsafe_correct:,} / {len(X_unsafe_test):,} ({100*unsafe_acc:.1f}%)")
        print(f"\nOverall accuracy: {100*overall_acc:.1f}%")
        
        metrics = {
            'safe_accuracy': safe_acc,
            'unsafe_accuracy': unsafe_acc,
            'overall_accuracy': overall_acc,
            'pred_safe': pred_safe,
            'pred_unsafe': pred_unsafe
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save trained model to disk."""
        save_obj = {
            'method': self.method,
            'model': self.model,
            'thresholds': self.thresholds,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f"Safety filter saved to: {filepath}")
    
    def load(self, filepath: str) -> 'SafetyFilter':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.method = save_obj['method']
        self.model = save_obj['model']
        self.thresholds = save_obj['thresholds']
        self.contamination = save_obj['contamination']
        self.random_state = save_obj['random_state']
        
        print(f"Safety filter loaded from: {filepath}")
        return self


if __name__ == "__main__":
    print("="*70)
    print("TESTING SAFETY FILTER")
    print("="*70)
    
    from data_loader import load_data
    
    # Load data
    loader = load_data()
    X, y = loader.get_multiclass_dataset(scale=True)
    
    # Split by label
    X_positive = X[y == 1]  # Safe sites
    X_hard_neg = X[y == -1]  # Unsafe sites
    
    # Stratified train/test split for both
    from sklearn.model_selection import train_test_split
    X_safe_train, X_safe_test = train_test_split(
        X_positive, test_size=0.2, random_state=42
    )
    X_unsafe_train, X_unsafe_test = train_test_split(
        X_hard_neg, test_size=0.2, random_state=42
    )
    
    # Test threshold method
    print("\n" + "="*70)
    print("THRESHOLD METHOD")
    print("="*70)
    thresh_filter = SafetyFilter(method='threshold')
    thresh_filter.train(X_safe_train, X_unsafe_train)
    thresh_metrics = thresh_filter.evaluate(X_safe_test, X_unsafe_test)
    
    # Test One-Class SVM
    print("\n" + "="*70)
    print("ONE-CLASS SVM")
    print("="*70)
    svm_filter = SafetyFilter(method='one_class_svm')
    svm_filter.train(X_safe_train)
    svm_metrics = svm_filter.evaluate(X_safe_test, X_unsafe_test)
    
    print("\n✓ Safety filter test complete")
