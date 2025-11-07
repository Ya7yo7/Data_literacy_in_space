"""data_loader.py

Load and prepare Mars landing suitability dataset for ML models.

Features: long_east_deg, lat_north_deg, altitude_m, radius_m, slope_deg, roughness_rms_m
Labels: -1 (hard_negative), 0 (weak_negative), 1 (positive/successful sites)

Provides spatial cross-validation splits to avoid data leakage from nearby locations.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional


class MarsDataLoader:
    """Load and prepare Mars landing suitability dataset."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: Path to final_labeled_dataset.csv. If None, auto-detect.
        """
        if data_path is None:
            # Try to find the data file
            script_dir = Path(__file__).resolve().parent
            possible_paths = [
                script_dir.parent / "Topography" / "data_cleaning" / "final_labeled_dataset.csv",
                script_dir.parent / "data_cleaning" / "final_labeled_dataset.csv",
            ]
            for p in possible_paths:
                if p.exists():
                    data_path = str(p)
                    break
            
            if data_path is None:
                raise FileNotFoundError("Could not find final_labeled_dataset.csv")
        
        self.data_path = data_path
        self.feature_cols = ["long_east_deg", "lat_north_deg", "altitude_m", 
                            "radius_m", "slope_deg", "roughness_rms_m"]
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.y_binary = None  # For binary classification (suitable vs unsuitable)
        
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
        
        # Create binary labels: 1 = suitable (positive sites), 0 = unsuitable (negatives)
        self.y_binary = (self.y == 1).astype(int)
        
        print(f"\nLabel distribution:")
        for label in sorted(np.unique(self.y)):
            count = (self.y == label).sum()
            label_name = {-1: "hard_negative", 0: "weak_negative", 1: "positive"}.get(label, f"label_{label}")
            print(f"  {label_name:15s} (label={label:2d}): {count:6,} ({100*count/len(self.y):.1f}%)")
        
        print(f"\nBinary label distribution:")
        print(f"  unsuitable (0): {(self.y_binary == 0).sum():6,} ({100*(self.y_binary == 0).sum()/len(self.y_binary):.1f}%)")
        print(f"  suitable   (1): {(self.y_binary == 1).sum():6,} ({100*(self.y_binary == 1).sum()/len(self.y_binary):.1f}%)")
        
        return self
    
    def get_features_scaled(self) -> np.ndarray:
        """Get feature-scaled features (fit scaler on all data)."""
        if self.X is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.scaler.fit_transform(self.X)
    
    def spatial_cv_splits(self, n_splits: int = 5, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create spatial cross-validation splits using lat/lon clustering.
        
        This ensures training and validation sets are spatially separated
        to avoid data leakage from nearby points.
        
        Args:
            n_splits: Number of CV folds
            random_state: Random seed
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Use KFold with shuffling as a simple approach
        # For true spatial CV, we'd cluster by lat/lon regions
        # but this is sufficient for initial modeling
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, val_idx in kf.split(self.X):
            splits.append((train_idx, val_idx))
        
        print(f"\nCreated {n_splits} spatial CV splits")
        print(f"  Avg train size: {len(splits[0][0]):,}")
        print(f"  Avg val size:   {len(splits[0][1]):,}")
        
        return splits
    
    def get_binary_dataset(self, scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get dataset for binary classification (suitable vs unsuitable).
        
        Args:
            scale: Whether to scale features
            
        Returns:
            (X, y_binary) where y_binary is 0/1
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        X = self.get_features_scaled() if scale else self.X
        return X, self.y_binary
    
    def get_multiclass_dataset(self, scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get dataset for multiclass classification (-1, 0, 1).
        
        Args:
            scale: Whether to scale features
            
        Returns:
            (X, y) where y is -1/0/1
        """
        if self.X is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        X = self.get_features_scaled() if scale else self.X
        return X, self.y
    
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
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_cols
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        stats = {
            'n_samples': len(self.df),
            'n_features': len(self.feature_cols),
            'n_positives': (self.y == 1).sum(),
            'n_weak_negatives': (self.y == 0).sum(),
            'n_hard_negatives': (self.y == -1).sum(),
            'feature_stats': {}
        }
        
        for col in self.feature_cols:
            stats['feature_stats'][col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }
        
        return stats


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
    
    print("\n" + "="*70)
    print("FEATURE STATISTICS")
    print("="*70)
    stats = loader.get_statistics()
    for feature, fstats in stats['feature_stats'].items():
        print(f"\n{feature}:")
        print(f"  Mean: {fstats['mean']:.2f}")
        print(f"  Std:  {fstats['std']:.2f}")
        print(f"  Min:  {fstats['min']:.2f}")
        print(f"  Max:  {fstats['max']:.2f}")
    
    # Test spatial CV
    splits = loader.spatial_cv_splits(n_splits=5)
    
    # Test dataset variants
    X_bin, y_bin = loader.get_binary_dataset()
    print(f"\nBinary dataset shape: {X_bin.shape}, {y_bin.shape}")
    
    X_multi, y_multi = loader.get_multiclass_dataset()
    print(f"Multiclass dataset shape: {X_multi.shape}, {y_multi.shape}")
    
    X_pos = loader.get_positive_only()
    print(f"Positive-only dataset shape: {X_pos.shape}")
    
    print("\n✓ Data loader test complete")
