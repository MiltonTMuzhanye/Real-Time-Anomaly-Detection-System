"""
Data Ingestion Module - Loading and preprocessing raw data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import glob
import json
from datetime import datetime
import dask.dataframe as dd

from ..utils.logger import setup_logger
from ..utils.exceptions import DataIngestionError

logger = setup_logger(__name__)

class DataIngestor:
    """
    Data ingestion class for loading and preprocessing data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        
    def ingest(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Ingest data from various sources
        
        Args:
            data_path: Path to data file or directory
            
        Returns:
            pandas DataFrame with ingested data
        """
        try:
            data_path = Path(data_path)
            
            if not data_path.exists():
                raise DataIngestionError(f"Data path not found: {data_path}")
            
            # Determine data type and load
            if data_path.is_file():
                data = self._load_file(data_path)
            elif data_path.is_dir():
                data = self._load_directory(data_path)
            else:
                raise DataIngestionError(f"Invalid path: {data_path}")
            
            logger.info(f"Ingested {len(data)} rows from {data_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            raise DataIngestionError(f"Failed to ingest data: {e}")
    
    def _load_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single file"""
        if filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
        elif filepath.suffix == '.feather':
            return pd.read_feather(filepath)
        elif filepath.suffix == '.hdf5':
            return pd.read_hdf(filepath)
        else:
            raise DataIngestionError(f"Unsupported file format: {filepath.suffix}")
    
    def _load_directory(self, directory: Path) -> pd.DataFrame:
        """Load all files in a directory"""
        dataframes = []
        
        for filepath in directory.glob('*'):
            if filepath.is_file():
                try:
                    df = self._load_file(filepath)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
        
        if not dataframes:
            raise DataIngestionError(f"No valid files found in {directory}")
        
        return pd.concat(dataframes, ignore_index=True)
    
    def load_cesnet_data(self, path: str, sample: bool = True) -> pd.DataFrame:
        """
        Load CESNET-TimeSeries24 data
        
        Args:
            path: Path to CESNET data
            sample: Whether to use sample data
            
        Returns:
            pandas DataFrame with CESNET data
        """
        logger.info(f"Loading CESNET data from {path}")
        
        try:
            if sample:
                file_pattern = "ip_addresses_sample.csv"
            else:
                file_pattern = "ip_addresses_full.csv"
            
            # Find the file
            files = glob.glob(str(Path(path) / "**" / file_pattern), recursive=True)
            
            if not files:
                raise DataIngestionError(f"No CESNET data found in {path}")
            
            # Load data
            data = pd.read_csv(files[0])
            
            logger.info(f"Loaded {len(data)} rows of CESNET data")
            
            # Parse timestamp
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading CESNET data: {e}")
            raise DataIngestionError(f"Failed to load CESNET data: {e}")
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate ingested data
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for required columns
        required_columns = self.data_config.get('required_columns', [])
        if required_columns:
            missing = set(required_columns) - set(data.columns)
            if missing:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing columns: {missing}")
        
        # Check for null values
        null_counts = data.isnull().sum()
        if null_counts.any():
            validation_results['warnings'].append(
                f"Null values found: {null_counts[null_counts > 0].to_dict()}"
            )
        
        # Check data types
        for col, dtype in data.dtypes.items():
            if dtype == 'object':
                validation_results['warnings'].append(
                    f"Column '{col}' has object type, consider converting"
                )
        
        # Generate stats
        validation_results['stats'] = {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum() / (1024**2),  # MB
            'date_range': {
                'min': data['timestamp'].min() if 'timestamp' in data else None,
                'max': data['timestamp'].max() if 'timestamp' in data else None
            }
        }
        
        logger.info(f"Data validation: {validation_results['is_valid']}")
        
        return validation_results