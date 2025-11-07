"""binary_classifier.py

Binary classifier for Mars landing suitability (suitable vs unsuitable).
Uses ensemble methods (Random Forest, Gradient Boosting) for robust predictions.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Dict, Tuple, Optional, List
import pickle
from pathlib import Path


class BinaryLandingSuitabilityClassifier:
    """Binary classifier: suitable (1) vs unsuitable (0) landing sites."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                class_weight='balanced',  # Handle class imbalance
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> 'BinaryLandingSuitabilityClassifier':
        """
        Train the binary classifier.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (0=unsuitable, 1=suitable)
            feature_names: List of feature names for interpretability
        """
        print(f"\nTraining {self.model_type} binary classifier...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Positive (suitable): {(y_train == 1).sum():,} ({100*(y_train==1).sum()/len(y_train):.1f}%)")
        print(f"  Negative (unsuitable): {(y_train == 0).sum():,} ({100*(y_train==0).sum()/len(y_train):.1f}%)")
        
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = (train_pred == y_train).mean()
        print(f"  Training accuracy: {train_acc:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns shape (n_samples, 2)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_suitability_score(self, X: np.ndarray) -> np.ndarray:
        """Get suitability score (probability of class 1). Returns shape (n_samples,)."""
        return self.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate classifier on test set.
        
        Returns:
            Dictionary with metrics: accuracy, roc_auc, pr_auc, etc.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        # Classification report
        print(f"\n{self.model_type.upper()} EVALUATION")
        print("="*60)
        print(classification_report(y_test, y_pred, 
                                   target_names=['unsuitable', 'suitable'],
                                   digits=3))
        
        print(f"ROC-AUC Score: {roc_auc:.3f}")
        print(f"PR-AUC Score:  {pr_auc:.3f}")
        
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            importance_dict = dict(zip(self.feature_names, importances))
        else:
            importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True))
        
        print("\nFeature Importances:")
        for feature, importance in importance_dict.items():
            print(f"  {feature:20s}: {importance:.4f}")
        
        return importance_dict
    
    def save(self, filepath: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_obj = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f"Model saved to: {filepath}")
    
    def load(self, filepath: str) -> 'BinaryLandingSuitabilityClassifier':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.model = save_obj['model']
        self.model_type = save_obj['model_type']
        self.feature_names = save_obj['feature_names']
        self.random_state = save_obj['random_state']
        
        print(f"Model loaded from: {filepath}")
        return self


if __name__ == "__main__":
    # Test with synthetic data
    print("="*70)
    print("TESTING BINARY CLASSIFIER")
    print("="*70)
    
    from data_loader import load_data
    
    # Load data
    loader = load_data()
    X, y = loader.get_binary_dataset(scale=True)
    
    # Stratified split to ensure both classes in train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test Random Forest
    print("\n" + "="*70)
    print("RANDOM FOREST")
    print("="*70)
    rf_clf = BinaryLandingSuitabilityClassifier(model_type='random_forest')
    rf_clf.train(X_train, y_train, feature_names=loader.get_feature_names())
    rf_metrics = rf_clf.evaluate(X_test, y_test)
    rf_clf.get_feature_importance()
    
    # Test Gradient Boosting
    print("\n" + "="*70)
    print("GRADIENT BOOSTING")
    print("="*70)
    gb_clf = BinaryLandingSuitabilityClassifier(model_type='gradient_boosting')
    gb_clf.train(X_train, y_train, feature_names=loader.get_feature_names())
    gb_metrics = gb_clf.evaluate(X_test, y_test)
    gb_clf.get_feature_importance()
    
    print("\n✓ Binary classifier test complete")
