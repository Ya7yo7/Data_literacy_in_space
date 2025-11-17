"""data_loader.py

Load and prepare Mars landing suitability dataset for modeling.

Dataset: final_labeled_dataset.csv
Features: long_east_deg, lat_north_deg, altitude_m, radius_m, slope_deg, roughness_rms_m
Labels: 
  - 1: positive (successful landing sites)
  - 0: weak negatives (random Mars terrain)
  - -1: hard negatives (known unsuitable terrain)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class MarsDataLoader:
    """Load and prepare Mars landing suitability dataset."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Path to final_labeled_dataset.csv. If None, auto-detect.
        """
        if data_path is None:
            # Auto-detect data file
            script_dir = Path(__file__).resolve().parent
            possible_paths = [
                script_dir.parent / "data" / "processed" / "final_labeled_dataset.csv",
                script_dir / "data" / "processed" / "final_labeled_dataset.csv",
            ]
            for p in possible_paths:
                if p.exists():
                    data_path = str(p)
                    break
            
            if data_path is None:
                raise FileNotFoundError("Could not find final_labeled_dataset.csv")
        
        self.data_path = data_path
        self.feature_cols = [
            "long_east_deg", "lat_north_deg", "altitude_m", 
            "radius_m", "slope_deg", "roughness_rms_m"
        ]
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        
    def load(self) -> 'MarsDataLoader':
        """Load the dataset and prepare features."""
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(self.df):,} rows")
        
        # Check for required columns
        missing = [col for col in self.feature_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if 'label' not in self.df.columns:
            raise ValueError("Missing 'label' column")
        
        # Extract features and labels
        self.X = self.df[self.feature_cols].values
        self.y = self.df['label'].values
        
        print(f"\nLabel distribution:")
        for label in sorted(np.unique(self.y)):
            count = (self.y == label).sum()
            label_name = {-1: "hard_negative", 0: "weak_negative", 1: "positive"}.get(label, f"label_{label}")
            print(f"  {label_name:15s} (label={label:2d}): {count:6,} ({100*count/len(self.y):.1f}%)")
        
        return self
    
    def get_features_scaled(self) -> np.ndarray:
        """Get scaled features (fit scaler on all data)."""
        if self.X is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.scaler.fit_transform(self.X)
    
    def get_binary_dataset(self, scale: bool = True, use_sample_weights: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get dataset for binary classification (suitable vs unsuitable).
        
        Suitable: label == 1
        Unsuitable: label == 0 or label == -1 (weak + hard negatives)
        
        Args:
            scale: Whether to scale features
            use_sample_weights: If True, return weights with hard negatives weighted 2x
            
        Returns:
            (X, y_binary, sample_weights) where y_binary is 0/1
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Binary labels: 1 = suitable, 0 = unsuitable
        y_binary = (self.y == 1).astype(int)
        
        # Sample weights: hard negatives get 2x weight of weak negatives
        weights = None
        if use_sample_weights:
            weights = np.ones(len(self.y))
            weights[self.y == -1] = 2.0  # Hard negatives: 2x weight
            weights[self.y == 0] = 1.0   # Weak negatives: 1x weight
            weights[self.y == 1] = 1.0   # Positives: 1x weight
        
        X = self.get_features_scaled() if scale else self.X
        
        print(f"\nBinary dataset:")
        print(f"  Suitable (1):   {(y_binary == 1).sum():6,} ({100*(y_binary == 1).sum()/len(y_binary):.1f}%)")
        print(f"  Unsuitable (0): {(y_binary == 0).sum():6,} ({100*(y_binary == 0).sum()/len(y_binary):.1f}%)")
        if weights is not None:
            print(f"  Using sample weights (hard negatives: 2x)")
        
        return X, y_binary, weights
    
    def get_positive_only(self, scale: bool = True) -> np.ndarray:
        """Get only positive examples (successful landing sites) for similarity modeling."""
        if self.X is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        pos_mask = self.y == 1
        X_pos = self.X[pos_mask]
        
        if scale:
            X_pos = self.scaler.fit_transform(X_pos)
        
        print(f"\nPositive examples: {len(X_pos):,}")
        return X_pos
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.feature_cols
    
    def transform_new_data(self, X_new: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted scaler.
        
        Args:
            X_new: New data to transform (must have same features in same order)
            
        Returns:
            Scaled features
        """
        return self.scaler.transform(X_new)


def load_data(data_path: Optional[str] = None) -> MarsDataLoader:
    """Convenience function to load data."""
    loader = MarsDataLoader(data_path)
    loader.load()
    return loader


if __name__ == "__main__":
    # Test the data loader
    print("="*70)
    print("TESTING DATA LOADER")
    print("="*70)
    
    loader = load_data()
    
    # Test binary dataset
    X_bin, y_bin, weights = loader.get_binary_dataset(scale=True, use_sample_weights=True)
    print(f"\nBinary dataset shape: {X_bin.shape}, {y_bin.shape}")
    print(f"Sample weights shape: {weights.shape if weights is not None else None}")
    
    # Test positive-only dataset
    X_pos = loader.get_positive_only(scale=True)
    print(f"Positive-only dataset shape: {X_pos.shape}")
    
    print("\n✓ Data loader test complete")
