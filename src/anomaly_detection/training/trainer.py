"""
Model Training Module - Training and managing models
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from ..utils.logger import setup_logger
from ..utils.config import load_config
from ..models.isolation_forest import IsolationForestDetector
from ..models.autoencoder import AutoencoderDetector
from ..models.lstm_autoencoder import LSTMAutoencoderDetector
from ..models.ensemble_detector import EnsembleDetector
from ..evaluation.metrics import AnomalyMetrics

logger = setup_logger(__name__)

class ModelTrainer:
    """
    Model training and management class
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('models', {})
        self.models = {}
        self.metrics = {}
        self.artifacts_path = config.get('artifacts_path', 'artifacts/')
        
        # Create artifacts directories
        os.makedirs(f"{self.artifacts_path}/trained_models", exist_ok=True)
        os.makedirs(f"{self.artifacts_path}/scalers", exist_ok=True)
        os.makedirs(f"{self.artifacts_path}/thresholds", exist_ok=True)
    
    def train_isolation_forest(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """
        Train Isolation Forest model
        """
        logger.info("Training Isolation Forest...")
        
        config = self.model_config.get('isolation_forest', {})
        detector = IsolationForestDetector(config)
        
        start_time = datetime.now()
        detector.fit(X_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        scores = detector.predict_scores(X_val)
        metrics = self._evaluate_model(detector, X_val, scores)
        
        return {
            'model': detector,
            'metrics': metrics,
            'training_time': training_time
        }
    
    def train_autoencoder(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """
        Train Autoencoder model
        """
        logger.info("Training Autoencoder...")
        
        config = self.model_config.get('autoencoder', {})
        detector = AutoencoderDetector(config)
        
        start_time = datetime.now()
        detector.fit(X_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        scores = detector.predict_scores(X_val)
        metrics = self._evaluate_model(detector, X_val, scores)
        
        return {
            'model': detector,
            'metrics': metrics,
            'training_time': training_time
        }
    
    def train_lstm_autoencoder(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """
        Train LSTM Autoencoder model
        """
        logger.info("Training LSTM Autoencoder...")
        
        config = self.model_config.get('lstm_autoencoder', {})
        detector = LSTMAutoencoderDetector(config)
        
        start_time = datetime.now()
        detector.fit(X_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        scores = detector.predict_scores(X_val)
        metrics = self._evaluate_model(detector, X_val, scores)
        
        return {
            'model': detector,
            'metrics': metrics,
            'training_time': training_time
        }
    
    def train_ensemble(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """
        Train Ensemble model
        """
        logger.info("Training Ensemble...")
        
        # Train individual models if not already trained
        models = {}
        
        if 'isolation_forest' not in self.models:
            result = self.train_isolation_forest(X_train, X_val)
            models['isolation_forest'] = result['model']
        else:
            models['isolation_forest'] = self.models['isolation_forest']
        
        if 'autoencoder' not in self.models:
            result = self.train_autoencoder(X_train, X_val)
            models['autoencoder'] = result['model']
        else:
            models['autoencoder'] = self.models['autoencoder']
        
        config = self.model_config.get('ensemble', {})
        ensemble = EnsembleDetector(models, config)
        
        start_time = datetime.now()
        # Ensemble doesn't require fitting, but we evaluate it
        _, scores = ensemble.predict_with_scores(X_val)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        metrics = self._evaluate_model(ensemble, X_val, scores)
        
        return {
            'model': ensemble,
            'metrics': metrics,
            'training_time': training_time
        }
    
    def _evaluate_model(self, model, X_val: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        metrics = AnomalyMetrics()
        results = metrics.compute_all(scores, X_val)
        
        logger.info(f"Model evaluation: {results}")
        return results
    
    def train_all(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """
        Train all models
        """
        results = {}
        
        # Train each model
        model_names = self.model_config.get('models_to_train', 
                                            ['isolation_forest', 'autoencoder', 'ensemble'])
        
        for model_name in model_names:
            if model_name == 'isolation_forest':
                results['isolation_forest'] = self.train_isolation_forest(X_train, X_val)
            elif model_name == 'autoencoder':
                results['autoencoder'] = self.train_autoencoder(X_train, X_val)
            elif model_name == 'lstm_autoencoder':
                results['lstm_autoencoder'] = self.train_lstm_autoencoder(X_train, X_val)
            elif model_name == 'ensemble':
                results['ensemble'] = self.train_ensemble(X_train, X_val)
        
        # Store models
        for name, result in results.items():
            self.models[name] = result['model']
            self.metrics[name] = result['metrics']
        
        return results
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.models.items():
            path = f"{self.artifacts_path}/trained_models/{name}_model.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved {name} model to {path}")
        
        # Save metrics
        with open(f"{self.artifacts_path}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info("All models saved successfully")
    
    def load_models(self):
        """Load all trained models"""
        model_names = self.model_config.get('models_to_train', 
                                            ['isolation_forest', 'autoencoder', 'ensemble'])
        
        for name in model_names:
            path = f"{self.artifacts_path}/trained_models/{name}_model.joblib"
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                logger.info(f"Loaded {name} model from {path}")
        
        # Load metrics
        metrics_path = f"{self.artifacts_path}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        
        logger.info("All models loaded successfully")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on F1 score
        """
        best_model = None
        best_score = -1
        best_name = None
        
        for name, metrics in self.metrics.items():
            f1 = metrics.get('f1', 0)
            if f1 > best_score:
                best_score = f1
                best_name = name
                best_model = self.models[name]
        
        if best_model is None:
            raise ValueError("No models available")
        
        logger.info(f"Best model: {best_name} (F1: {best_score:.3f})")
        
        return best_name, best_model