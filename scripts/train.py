"""
Model Training Script
"""
import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly_detection.utils.logger import setup_logger
from src.anomaly_detection.utils.config import load_config
from src.anomaly_detection.training.trainer import ModelTrainer
from src.anomaly_detection.models.isolation_forest import IsolationForestDetector
from src.anomaly_detection.models.autoencoder import AutoencoderDetector
from src.anomaly_detection.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.anomaly_detection.models.ensemble_detector import EnsembleDetector
from src.anomaly_detection.evaluation.metrics import AnomalyMetrics
from src.anomaly_detection.data.preprocessing import DataPreprocessor

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train anomaly detection models')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model.yaml',
        help='Model config file path'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/cesnet_processed.parquet',
        help='Processed data path'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['isolation_forest', 'autoencoder', 'lstm_autoencoder', 'ensemble'],
        help='Models to train'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state'
    )
    parser.add_argument(
        '--mlflow',
        action='store_true',
        help='Enable MLflow tracking'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='anomaly_detection',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save trained models'
    )
    return parser.parse_args()

def load_data(data_path: str) -> pd.DataFrame:
    """Load processed data"""
    logger.info(f"Loading data from {data_path}")
    
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    return data

def train_isolation_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    save_path: str = None
) -> dict:
    """Train Isolation Forest model"""
    logger.info("Training Isolation Forest...")
    
    # Get config
    if_config = config.get('models', {}).get('isolation_forest', {})
    
    # Create detector
    detector = IsolationForestDetector(if_config)
    
    # Train
    start_time = datetime.now()
    detector.fit(X_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate on validation set
    scores = detector.predict_scores(X_val)
    predictions = detector.predict(X_val)
    
    # Save model
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(detector, f"{save_path}/isolation_forest_model.joblib")
    
    return {
        'model': detector,
        'scores': scores,
        'predictions': predictions,
        'training_time': training_time
    }

def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    save_path: str = None
) -> dict:
    """Train Autoencoder model"""
    logger.info("Training Autoencoder...")
    
    # Get config
    ae_config = config.get('models', {}).get('autoencoder', {})
    
    # Create detector
    detector = AutoencoderDetector(ae_config)
    
    # Train
    start_time = datetime.now()
    detector.fit(X_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate on validation set
    scores = detector.predict_scores(X_val)
    predictions = detector.predict(X_val)
    
    # Save model
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        detector.save(f"{save_path}/autoencoder_model")
    
    return {
        'model': detector,
        'scores': scores,
        'predictions': predictions,
        'training_time': training_time
    }

def train_lstm_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    save_path: str = None
) -> dict:
    """Train LSTM Autoencoder model"""
    logger.info("Training LSTM Autoencoder...")
    
    # Get config
    lstm_config = config.get('models', {}).get('lstm_autoencoder', {})
    
    # Create detector
    detector = LSTMAutoencoderDetector(lstm_config)
    
    # Train
    start_time = datetime.now()
    detector.fit(X_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate on validation set
    scores = detector.predict_scores(X_val)
    predictions = detector.predict(X_val)
    
    # Save model
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        detector.save(f"{save_path}/lstm_autoencoder_model")
    
    return {
        'model': detector,
        'scores': scores,
        'predictions': predictions,
        'training_time': training_time
    }

def train_ensemble(
    models: dict,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    save_path: str = None
) -> dict:
    """Train Ensemble model"""
    logger.info("Training Ensemble...")
    
    # Get config
    ensemble_config = config.get('models', {}).get('ensemble', {})
    
    # Create ensemble
    ensemble = EnsembleDetector(models, ensemble_config)
    
    # Train (ensemble doesn't need training, just uses component models)
    # Evaluate on validation set
    start_time = datetime.now()
    predictions, scores = ensemble.predict_with_scores(X_val)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save model
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(ensemble, f"{save_path}/ensemble_model.joblib")
    
    return {
        'model': ensemble,
        'scores': scores,
        'predictions': predictions,
        'training_time': training_time
    }

def main():
    """Main entry point"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    data = load_data(args.data)
    
    # Preprocess data
    preprocessor = DataPreprocessor(config)
    
    # Separate features and labels if available
    X = data.select_dtypes(include=[np.number]).values
    
    # Split data
    X_train, X_temp = train_test_split(
        X,
        test_size=args.test_size + 0.1,
        random_state=args.random_state
    )
    X_val, X_test = train_test_split(
        X_temp,
        test_size=args.test_size / (args.test_size + 0.1),
        random_state=args.random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Setup MLflow
    if args.mlflow:
        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'config': args.config
        })
    
    # Training results
    trained_models = {}
    all_scores = {}
    all_predictions = {}
    
    # Create save directory
    save_path = "artifacts/trained_models"
    os.makedirs(save_path, exist_ok=True)
    
    # Train selected models
    for model_name in args.models:
        if model_name == 'isolation_forest':
            result = train_isolation_forest(
                X_train, X_val, config, save_path
            )
            trained_models['isolation_forest'] = result['model']
            all_scores['isolation_forest'] = result['scores']
            all_predictions['isolation_forest'] = result['predictions']
            
            if args.mlflow:
                mlflow.log_params({
                    'isolation_forest_training_time': result['training_time']
                })
        
        elif model_name == 'autoencoder':
            result = train_autoencoder(
                X_train, X_val, config, save_path
            )
            trained_models['autoencoder'] = result['model']
            all_scores['autoencoder'] = result['scores']
            all_predictions['autoencoder'] = result['predictions']
            
            if args.mlflow:
                mlflow.log_params({
                    'autoencoder_training_time': result['training_time']
                })
        
        elif model_name == 'lstm_autoencoder':
            result = train_lstm_autoencoder(
                X_train, X_val, config, save_path
            )
            trained_models['lstm_autoencoder'] = result['model']
            all_scores['lstm_autoencoder'] = result['scores']
            all_predictions['lstm_autoencoder'] = result['predictions']
            
            if args.mlflow:
                mlflow.log_params({
                    'lstm_autoencoder_training_time': result['training_time']
                })
    
    # Train ensemble if requested
    if 'ensemble' in args.models and trained_models:
        result = train_ensemble(
            trained_models, X_train, X_val, config, save_path
        )
        trained_models['ensemble'] = result['model']
        all_scores['ensemble'] = result['scores']
        all_predictions['ensemble'] = result['predictions']
        
        if args.mlflow:
            mlflow.log_params({
                'ensemble_training_time': result['training_time']
            })
    
    # Evaluate on test set
    logger.info("Evaluating models on test set...")
    
    metrics = AnomalyMetrics()
    
    for model_name, model in trained_models.items():
        if model_name == 'ensemble':
            _, scores = model.predict_with_scores(X_test)
        else:
            scores = model.predict_scores(X_test)
        
        # Compute metrics
        metric_results = metrics.compute_all(scores, X_test)
        
        logger.info(f"{model_name} Test Metrics:")
        for metric_name, value in metric_results.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        if args.mlflow:
            for metric_name, value in metric_results.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)
    
    # Save preprocessor and scaler
    scaler_path = "artifacts/scalers/standard_scaler.joblib"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Fit and save scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_names_path = "artifacts/feature_store/feature_names.json"
    os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
    with open(feature_names_path, 'w') as f:
        json.dump(list(data.select_dtypes(include=[np.number]).columns), f)
    logger.info(f"Saved feature names to {feature_names_path}")
    
    # End MLflow run
    if args.mlflow:
        mlflow.end_run()
        logger.info("MLflow run completed")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()