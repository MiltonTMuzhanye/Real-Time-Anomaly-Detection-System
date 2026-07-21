"""
Detection Script for Batch and Real-time Detection
"""
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly_detection.utils.logger import setup_logger
from src.anomaly_detection.utils.config import load_config
from src.anomaly_detection.pipelines.inference_pipeline import InferencePipeline
from app.inference.detector import AnomalyDetector
from app.inference.anomaly_engine import AnomalyEngine
from app.inference.alert_engine import AlertEngine

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run anomaly detection')
    parser.add_argument(
        '--mode',
        type=str,
        default='batch',
        choices=['batch', 'stream', 'api'],
        help='Detection mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input data path (for batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/anomalies.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='artifacts/trained_models/ensemble_model.joblib',
        help='Model path'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection threshold'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Limit number of samples (for batch)'
    )
    return parser.parse_args()

def load_model(model_path: str):
    """Load trained model"""
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def run_batch_detection(args):
    """Run batch detection"""
    logger.info("Running batch detection...")
    
    # Load data
    if args.input:
        if args.input.endswith('.parquet'):
            data = pd.read_parquet(args.input)
        elif args.input.endswith('.csv'):
            data = pd.read_csv(args.input)
        else:
            raise ValueError(f"Unsupported file format: {args.input}")
        
        # Limit samples
        if args.limit:
            data = data.head(args.limit)
        
        logger.info(f"Loaded {len(data)} samples")
    else:
        # Use sample data
        logger.info("No input provided, using sample data")
        data = pd.DataFrame({
            'bytes_in': np.random.exponential(1000, 100),
            'bytes_out': np.random.exponential(2000, 100),
            'packets_in': np.random.poisson(50, 100),
            'packets_out': np.random.poisson(75, 100),
            'flows': np.random.poisson(10, 100),
            'duration': np.random.exponential(5, 100)
        })
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.model)
    
    # Create detector
    detector = AnomalyDetector(config)
    detector.model = model
    detector.thresholds['detection_threshold'] = args.threshold
    
    # Run detection
    results = asyncio.run(detector.detect_batch(data))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add original data
    results_df = pd.concat([data.reset_index(drop=True), results_df], axis=1)
    
    # Add timestamp
    results_df['timestamp'] = datetime.now()
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    
    # Summary
    anomalies = results_df[results_df['is_anomaly'] == True]
    
    logger.info(f"Detection complete!")
    logger.info(f"Total samples: {len(results_df)}")
    logger.info(f"Anomalies detected: {len(anomalies)}")
    logger.info(f"Anomaly rate: {len(anomalies)/len(results_df)*100:.2f}%")
    logger.info(f"Results saved to {args.output}")
    
    return results_df

def run_stream_detection(args):
    """Run streaming detection"""
    logger.info("Starting streaming detection...")
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.model)
    
    # Create engine
    engine = AnomalyEngine(config)
    engine.ensemble = model
    
    # Create alert engine
    alert_engine = AlertEngine(config)
    
    # Simulate streaming data
    logger.info("Streaming data... (Press Ctrl+C to stop)")
    
    try:
        import time
        count = 0
        anomalies_count = 0
        
        while True:
            # Generate sample data point
            data = {
                'timestamp': datetime.now(),
                'bytes_in': np.random.exponential(1000),
                'bytes_out': np.random.exponential(2000),
                'packets_in': np.random.poisson(50),
                'packets_out': np.random.poisson(75),
                'flows': np.random.poisson(10),
                'duration': np.random.exponential(5)
            }
            
            # Add occasional anomaly
            if np.random.random() < 0.05:
                data['bytes_in'] *= 10
                data['flows'] *= 5
            
            # Detect
            result = asyncio.run(engine.detect(data))
            
            count += 1
            
            if result['is_anomaly']:
                anomalies_count += 1
                logger.warning(f"Anomaly detected! Score: {result['score']:.3f}")
                
                # Send alert
                asyncio.run(alert_engine.process_anomaly({
                    'score': result['score'],
                    'data': data,
                    'detector_scores': result.get('detector_scores', {})
                }))
            
            # Log progress
            if count % 100 == 0:
                logger.info(f"Processed {count} samples, {anomalies_count} anomalies")
            
            time.sleep(0.1)  # Simulate real-time
    
    except KeyboardInterrupt:
        logger.info("Streaming stopped")
    
    return count, anomalies_count

def main():
    """Main entry point"""
    args = parse_args()
    
    if args.mode == 'batch':
        run_batch_detection(args)
    elif args.mode == 'stream':
        run_stream_detection(args)
    elif args.mode == 'api':
        logger.info("Starting API server...")
        logger.info("Run: uvicorn app.api.main:app --reload")
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()