"""
Unit tests for anomaly detector
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from app.inference.detector import AnomalyDetector
from src.anomaly_detection.models.isolation_forest import IsolationForestDetector

class TestAnomalyDetector:
    """Test cases for AnomalyDetector"""
    
    @pytest.fixture
    def config(self):
        """Sample configuration"""
        return {
            'model_path': 'artifacts/trained_models/ensemble_model.joblib',
            'scaler_path': 'artifacts/scalers/standard_scaler.joblib',
            'threshold_path': 'artifacts/thresholds/thresholds.json',
            'models': {
                'isolation_forest': {
                    'n_estimators': 100,
                    'contamination': 0.1
                }
            }
        }
    
    @pytest.fixture
    def detector(self, config):
        """Create detector instance"""
        return AnomalyDetector(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample data point"""
        return {
            'timestamp': datetime.now(),
            'bytes_in': 1024.5,
            'bytes_out': 2048.0,
            'packets_in': 100,
            'packets_out': 150,
            'flows': 5,
            'duration': 10.5
        }
    
    @pytest.mark.asyncio
    async def test_detect_single(self, detector, sample_data):
        """Test single detection"""
        result = await detector.detect(sample_data)
        
        assert 'is_anomaly' in result
        assert 'score' in result
        assert 'confidence' in result
        assert isinstance(result['is_anomaly'], bool)
        assert 0 <= result['score'] <= 1
        assert 0 <= result['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_detect_batch(self, detector):
        """Test batch detection"""
        data = pd.DataFrame({
            'bytes_in': np.random.exponential(1000, 10),
            'bytes_out': np.random.exponential(2000, 10),
            'packets_in': np.random.poisson(50, 10),
            'packets_out': np.random.poisson(75, 10),
            'flows': np.random.poisson(10, 10),
            'duration': np.random.exponential(5, 10)
        })
        
        results = await detector.detect_batch(data)
        
        assert len(results) == len(data)
        for result in results:
            assert 'is_anomaly' in result
            assert 'score' in result
    
    def test_calculate_confidence(self, detector):
        """Test confidence calculation"""
        # Score of 0.9 should have high confidence
        confidence = detector._calculate_confidence(0.9)
        assert confidence >= 0.9
        
        # Score of 0.3 should have low confidence
        confidence = detector._calculate_confidence(0.3)
        assert confidence <= 0.3