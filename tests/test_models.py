"""
Unit tests for detection models
"""
import pytest
import numpy as np
from sklearn.datasets import make_blobs

from src.anomaly_detection.models.isolation_forest import IsolationForestDetector
from src.anomaly_detection.models.autoencoder import AutoencoderDetector
from src.anomaly_detection.models.ensemble_detector import EnsembleDetector

class TestModels:
    """Test cases for detection models"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        # Normal data
        X_normal = np.random.normal(0, 1, (100, 5))
        
        # Anomalous data
        X_anomaly = np.random.normal(10, 1, (10, 5))
        
        X = np.vstack([X_normal, X_anomaly])
        y = np.array([0] * 100 + [1] * 10)
        
        return X, y
    
    def test_isolation_forest(self, sample_data):
        """Test Isolation Forest detector"""
        X, y = sample_data
        
        config = {
            'n_estimators': 100,
            'contamination': 0.1,
            'random_state': 42
        }
        
        detector = IsolationForestDetector(config)
        detector.fit(X)
        
        scores = detector.predict_scores(X)
        predictions = detector.predict(X)
        
        assert len(scores) == len(X)
        assert len(predictions) == len(X)
        assert np.all((predictions == -1) | (predictions == 1))
    
    def test_autoencoder(self, sample_data):
        """Test Autoencoder detector"""
        X, y = sample_data
        
        config = {
            'encoding_dim': 3,
            'layers': [10, 5, 3, 5, 10],
            'epochs': 10,
            'batch_size': 16
        }
        
        detector = AutoencoderDetector(config)
        detector.fit(X)
        
        scores = detector.predict_scores(X)
        
        assert len(scores) == len(X)
        assert np.all(scores >= 0)
    
    def test_ensemble_detector(self, sample_data):
        """Test Ensemble detector"""
        X, y = sample_data
        
        # Create component models
        models = {
            'isolation_forest': IsolationForestDetector({
                'n_estimators': 50,
                'contamination': 0.1
            }),
            'autoencoder': AutoencoderDetector({
                'encoding_dim': 3,
                'epochs': 5
            })
        }
        
        # Train component models
        for model in models.values():
            model.fit(X)
        
        config = {
            'voting': 'soft',
            'threshold': 0.5
        }
        
        ensemble = EnsembleDetector(models, config)
        
        scores = ensemble.predict_scores(X)
        predictions = ensemble.predict(X)
        
        assert len(scores) == len(X)
        assert len(predictions) == len(X