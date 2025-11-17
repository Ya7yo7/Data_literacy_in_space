"""random_forest_model.py

Random Forest classifier for Mars landing site suitability.

Trains on labeled data (positive, weak negatives, hard negatives) with sample weighting.
Hard negatives receive 2x weight to emphasize learning from known hazards.
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from typing import Optional, Dict


class RandomForestModel:
    """Random Forest classifier for landing site suitability."""
    
    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None, random_state: int = 42):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced'  # Handle class imbalance
        )
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None, validation_split: float = 0.2) -> Dict:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix (scaled)
            y: Binary labels (1=suitable, 0=unsuitable)
            sample_weight: Optional sample weights (hard negatives weighted 2x)
            feature_names: Names of features for interpretability
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data (including sample weights if provided)
        if sample_weight is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X, y, sample_weight, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            sw_train, sw_val = None, None
        
        print(f"Training Random Forest...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Features: {len(self.feature_names)}")
        
        # Train
        self.model.fit(X_train, y_train, sample_weight=sw_train)
        self.is_trained = True
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_acc = (y_train_pred == y_train).mean()
        val_acc = (y_val_pred == y_val).mean()
        
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        print(f"\n  Training accuracy: {train_acc:.3f}")
        print(f"  Validation accuracy: {val_acc:.3f}")
        print(f"  Validation AUC: {val_auc:.3f}")
        
        # Feature importance
        print(f"\n  Top 5 important features:")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(importances))):
            idx = indices[i]
            print(f"    {i+1}. {self.feature_names[idx]:20s}: {importances[idx]:.3f}")
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_auc': val_auc,
            'feature_importances': dict(zip(self.feature_names, importances))
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict suitability probabilities.
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Array of probabilities for class 1 (suitable)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary suitability classification.
        
        Args:
            X: Feature matrix (scaled)
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions (1=suitable, 0=unsuitable)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances as dictionary."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }, f)
        print(f"  ✓ Saved model to: {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        print(f"  ✓ Loaded model from: {path}")


if __name__ == "__main__":
    # Test with dummy data
    print("="*70)
    print("TESTING RANDOM FOREST MODEL")
    print("="*70)
    
    from data_loader import load_data
    
    # Load training data
    loader = load_data()
    X, y, weights = loader.get_binary_dataset(scale=True, use_sample_weights=True)
    feature_names = loader.get_feature_names()
    
    # Train model
    rf = RandomForestModel(n_estimators=100, max_depth=20)
    metrics = rf.train(X, y, sample_weight=weights, feature_names=feature_names, validation_split=0.2)
    
    print("\n✓ Random Forest model test complete")
