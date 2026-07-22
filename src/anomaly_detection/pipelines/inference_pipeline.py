"""
Inference Pipeline - End-to-end inference workflow
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from ..utils.logger import setup_logger
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.ensemble_detector import EnsembleDetector
from ..evaluation.explainability import ModelExplainer

logger = setup_logger(__name__)

class InferencePipeline:
    """
    End-to-end inference pipeline for anomaly detection
    """
    
    def __init__(self, model: EnsembleDetector, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.explainer = ModelExplainer(model, config) if config.get('explainability') else None
        
        self.is_initialized = False
        self.pipeline_metrics = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a single data point
        """
        start_time = datetime.now()
        
        try:
            # Validate data
            if not self._validate_data(data):
                return {'error': 'Invalid data format'}
            
            # Preprocess
            preprocessed = await self._preprocess(data)
            
            # Extract features
            features = await self._extract_features(preprocessed)
            
            # Run model inference
            result = await self._infer(features)
            
            # Add explanation if available
            if self.explainer and result.get('is_anomaly', False):
                explanation = await self._explain(features, result)
                result['explanation'] = explanation
            
            # Update metrics
            self._update_metrics(start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {'error': str(e)}
    
    async def run_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of data
        """
        results = []
        
        for _, row in data.iterrows():
            result = await self.run(row.to_dict())
            results.append(result)
        
        return results
    
    async def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Apply preprocessing
        processed = self.preprocessor.transform(df)
        
        # Convert back to dict
        return processed[0] if len(processed) > 0 else {}
    
    async def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from data"""
        features = self.feature_engineer.extract_features(data)
        return features.reshape(1, -1)
    
    async def _infer(self, features: np.ndarray) -> Dict[str, Any]:
        """Run model inference"""
        result = self.model.predict(features)
        return result
    
    async def _explain(self, features: np.ndarray, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get explanation for prediction"""
        if self.explainer:
            explanation = self.explainer.explain(features)
            return explanation
        return {}
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Check required fields
        required_fields = self.config.get('required_fields', [])
        if required_fields:
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False
        
        # Check data types
        for key, value in data.items():
            if key in self.config.get('numerical_fields', []):
                if not isinstance(value, (int, float)):
                    logger.error(f"Field {key} should be numerical")
                    return False
        
        return True
    
    def _update_metrics(self, start_time: datetime):
        """Update pipeline metrics"""
        inference_time = (datetime.now() - start_time).total_seconds()
        
        self.pipeline_metrics['total_inferences'] += 1
        self.pipeline_metrics['total_time'] += inference_time
        self.pipeline_metrics['avg_time'] = self.pipeline_metrics['total_time'] / self.pipeline_metrics['total_inferences']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.pipeline_metrics