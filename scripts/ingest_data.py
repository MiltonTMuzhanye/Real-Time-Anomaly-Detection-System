"""
Data Ingestion Script
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import tarfile
import zipfile
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly_detection.utils.logger import setup_logger
from src.anomaly_detection.data.ingestion import DataIngestor
from src.anomaly_detection.utils.config import load_config

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ingest CESNET-TimeSeries24 dataset')
    parser.add_argument(
        '--source',
        type=str,
        default='zenodo',
        choices=['zenodo', 'huggingface', 'local'],
        help='Data source'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset if not available'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data (smaller file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Config file path'
    )
    return parser.parse_args()

def download_from_zenodo(sample: bool = True) -> str:
    """
    Download dataset from Zenodo
    """
    logger.info("Downloading CESNET-TimeSeries24 from Zenodo...")
    
    # Zenodo record ID for CESNET-TimeSeries24
    # (You'll need to get the actual record ID from Zenodo)
    record_id = "xxxxxxx"  # Replace with actual ID
    
    # File names
    if sample:
        filename = "ip_addresses_sample.tar.gz"
    else:
        filename = "ip_addresses_full.tar.gz"
    
    # Construct download URL
    url = f"https://zenodo.org/record/{record_id}/files/{filename}"
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    filepath = f"data/raw/{filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    logger.info(f"Downloaded to {filepath}")
    return filepath

def download_from_huggingface(sample: bool = True) -> str:
    """
    Download dataset from Hugging Face
    """
    logger.info("Downloading CESNET-TimeSeries24 from Hugging Face...")
    
    try:
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset("CESNET/cesnet-timeseries24")
        
        # Save to local
        output_dir = "data/raw/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to pandas and save
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            df.to_parquet(f"{output_dir}/cesnet_{split}.parquet")
            
        logger.info(f"Dataset saved to {output_dir}")
        
        return output_dir
        
    except ImportError:
        logger.error("Hugging Face datasets library not installed")
        logger.info("Install with: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"Error downloading from Hugging Face: {e}")
        return None

def extract_tar_gz(filepath: str, output_dir: str = "data/raw/"):
    """
    Extract tar.gz file
    """
    logger.info(f"Extracting {filepath}...")
    
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(output_dir)
    
    logger.info(f"Extracted to {output_dir}")

def main():
    """Main entry point"""
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Download if requested
    if args.download:
        if args.source == 'zenodo':
            filepath = download_from_zenodo(args.sample)
            if filepath and filepath.endswith('.tar.gz'):
                extract_tar_gz(filepath, args.output)
        elif args.source == 'huggingface':
            download_from_huggingface(args.sample)
        elif args.source == 'local':
            logger.info("Using local files")
    
    # Initialize data ingestor
    ingestor = DataIngestor(config)
    
    # Ingest data
    logger.info("Ingesting data...")
    data = ingestor.ingest(args.output)
    
    logger.info(f"Ingested {len(data)} rows")
    logger.info(f"Columns: {data.columns.tolist()}")
    logger.info(f"Shape: {data.shape}")
    
    # Save processed data
    processed_path = "data/processed/cesnet_processed.parquet"
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    data.to_parquet(processed_path)
    logger.info(f"Saved processed data to {processed_path}")

if __name__ == "__main__":
    main()