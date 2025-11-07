"""ensemble.py

Ensemble model combining binary classifier, safety filter, and similarity model
to produce final landing suitability scores.
"""
import numpy as np
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path


class LandingSuitabilityEnsemble:
    """Ensemble model for Mars landing suitability assessment."""
    
    def __init__(self, binary_classifier=None, safety_filter=None, 
                 similarity_model=None, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            binary_classifier: Trained binary classifier
            safety_filter: Trained safety filter
            similarity_model: Trained similarity model
            weights: Dictionary with keys 'binary', 'safety', 'similarity'
        """
        self.binary_classifier = binary_classifier
        self.safety_filter = safety_filter
        self.similarity_model = similarity_model
        
        # Default weights: equal weighting
        if weights is None:
            weights = {'binary': 0.4, 'safety': 0.3, 'similarity': 0.3}
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        print(f"Ensemble weights: {self.weights}")
    
    def predict_suitability(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict landing suitability scores for input locations.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Dictionary containing:
                - 'final_score': Combined score in [0, 1]
                - 'binary_score': Binary classifier probability
                - 'safety_score': Safety filter score
                - 'similarity_score': Similarity model score
                - 'recommendation': 'suitable' or 'unsuitable' based on threshold
        """
        results = {}
        
        # 1. Binary classifier score
        if self.binary_classifier is not None:
            binary_score = self.binary_classifier.get_suitability_score(X)
            results['binary_score'] = binary_score
        else:
            binary_score = np.ones(len(X)) * 0.5  # Neutral
            results['binary_score'] = binary_score
        
        # 2. Safety filter score
        if self.safety_filter is not None:
            safety_score = self.safety_filter.get_safety_score(X)
            results['safety_score'] = safety_score
        else:
            safety_score = np.ones(len(X)) * 0.5  # Neutral
            results['safety_score'] = safety_score
        
        # 3. Similarity model score
        if self.similarity_model is not None:
            similarity_score = self.similarity_model.get_similarity_score(X)
            results['similarity_score'] = similarity_score
        else:
            similarity_score = np.ones(len(X)) * 0.5  # Neutral
            results['similarity_score'] = similarity_score
        
        # 4. Combine scores with weights
        final_score = (
            self.weights['binary'] * binary_score +
            self.weights['safety'] * safety_score +
            self.weights['similarity'] * similarity_score
        )
        
        results['final_score'] = final_score
        
        # 5. Make recommendation (threshold at 0.5)
        results['recommendation'] = np.where(final_score >= 0.5, 'suitable', 'unsuitable')
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 verbose: bool = True) -> Dict:
        """
        Evaluate ensemble on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels (1 = suitable, 0 = unsuitable)
            verbose: Print detailed results
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict_suitability(X_test)
        final_score = predictions['final_score']
        
        # Convert recommendations to binary
        y_pred = (predictions['recommendation'] == 'suitable').astype(int)
        
        # Compute metrics
        accuracy = (y_pred == y_test).mean()
        
        # True/False positives/negatives
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if verbose:
            print("\n" + "="*70)
            print("ENSEMBLE EVALUATION")
            print("="*70)
            print(f"Test samples: {len(X_test):,}")
            print(f"  Suitable (y=1):   {(y_test == 1).sum():,}")
            print(f"  Unsuitable (y=0): {(y_test == 0).sum():,}")
            
            print(f"\nPredictions:")
            print(f"  Predicted suitable:   {(y_pred == 1).sum():,}")
            print(f"  Predicted unsuitable: {(y_pred == 0).sum():,}")
            
            print(f"\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"                 Pos   Neg")
            print(f"  Actual  Pos   {tp:5d} {fn:5d}")
            print(f"          Neg   {fp:5d} {tn:5d}")
            
            print(f"\nMetrics:")
            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1 Score:  {f1:.3f}")
            
            print(f"\nScore Statistics:")
            print(f"  Final score range: [{final_score.min():.3f}, {final_score.max():.3f}]")
            print(f"  Mean: {final_score.mean():.3f} ± {final_score.std():.3f}")
            print(f"  Suitable (y=1) mean score:   {final_score[y_test == 1].mean():.3f}")
            print(f"  Unsuitable (y=0) mean score: {final_score[y_test == 0].mean():.3f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'final_score': final_score,
            'y_pred': y_pred,
            'predictions': predictions
        }
        
        return metrics
    
    def score_new_location(self, longitude: float, latitude: float, 
                          altitude: float, radius: float, 
                          slope: float, roughness: float) -> Dict:
        """
        Score a single new location.
        
        Args:
            longitude: East longitude in degrees [0, 360]
            latitude: North latitude in degrees [-90, 90]
            altitude: Altitude in meters
            radius: Local radius in meters
            slope: Slope in degrees
            roughness: Surface roughness RMS in meters
            
        Returns:
            Dictionary with scores and recommendation
        """
        # Construct feature vector
        X = np.array([[longitude, latitude, altitude, radius, slope, roughness]])
        
        # Get predictions
        results = self.predict_suitability(X)
        
        # Extract single values
        output = {
            'location': {
                'longitude': longitude,
                'latitude': latitude,
                'altitude': altitude,
                'radius': radius,
                'slope': slope,
                'roughness': roughness
            },
            'final_score': float(results['final_score'][0]),
            'binary_score': float(results['binary_score'][0]),
            'safety_score': float(results['safety_score'][0]),
            'similarity_score': float(results['similarity_score'][0]),
            'recommendation': results['recommendation'][0]
        }
        
        return output
    
    def save(self, filepath: str):
        """Save ensemble model to disk."""
        save_obj = {
            'binary_classifier': self.binary_classifier,
            'safety_filter': self.safety_filter,
            'similarity_model': self.similarity_model,
            'weights': self.weights
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f"Ensemble model saved to: {filepath}")
    
    def load(self, filepath: str) -> 'LandingSuitabilityEnsemble':
        """Load ensemble model from disk."""
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)
        
        self.binary_classifier = save_obj['binary_classifier']
        self.safety_filter = save_obj['safety_filter']
        self.similarity_model = save_obj['similarity_model']
        self.weights = save_obj['weights']
        
        print(f"Ensemble model loaded from: {filepath}")
        return self


if __name__ == "__main__":
    print("="*70)
    print("TESTING ENSEMBLE MODEL")
    print("="*70)
    
    from data_loader import load_data
    from binary_classifier import BinaryLandingSuitabilityClassifier
    from safety_filter import SafetyFilter
    from similarity_model import SimilarityModel
    
    # Load data
    loader = load_data()
    X, y_multi = loader.get_multiclass_dataset(scale=True)
    y_binary = loader.y_binary
    
    # Stratified split to ensure balanced test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multi, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Train individual models
    print("\n[1/3] Training binary classifier...")
    binary_clf = BinaryLandingSuitabilityClassifier('random_forest')
    binary_clf.train(X_train, y_train_bin, loader.get_feature_names())
    
    print("\n[2/3] Training safety filter...")
    X_safe = X_train[y_train_multi == 1]
    X_unsafe = X_train[y_train_multi == -1]
    safety = SafetyFilter('threshold')
    safety.train(X_safe, X_unsafe)
    
    print("\n[3/3] Training similarity model...")
    similarity = SimilarityModel('kde')
    similarity.train(X_safe)
    
    # Create ensemble
    print("\n" + "="*70)
    print("CREATING ENSEMBLE")
    print("="*70)
    ensemble = LandingSuitabilityEnsemble(
        binary_classifier=binary_clf,
        safety_filter=safety,
        similarity_model=similarity,
        weights={'binary': 0.4, 'safety': 0.3, 'similarity': 0.3}
    )
    
    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test_bin)
    
    # Test single location scoring
    print("\n" + "="*70)
    print("TESTING SINGLE LOCATION")
    print("="*70)
    result = ensemble.score_new_location(
        longitude=137.4,  # Gale Crater (Curiosity)
        latitude=-4.5,
        altitude=-4500,
        radius=3396000,
        slope=2.5,
        roughness=5.0
    )
    print(f"Location: {result['location']}")
    print(f"Final Score: {result['final_score']:.3f}")
    print(f"  Binary:     {result['binary_score']:.3f}")
    print(f"  Safety:     {result['safety_score']:.3f}")
    print(f"  Similarity: {result['similarity_score']:.3f}")
    print(f"Recommendation: {result['recommendation']}")
    
    print("\n✓ Ensemble model test complete")
