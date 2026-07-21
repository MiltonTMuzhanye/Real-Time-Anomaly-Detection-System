"""
Data Validation Module - Validating data quality and consistency
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    """
    Data validation class for ensuring data quality
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = config.get('validation', {})
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against rules
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {},
            'quality_score': 1.0
        }
        
        # Check data quality
        quality_checks = []
        
        # Check for missing values
        missing_ratio = data.isnull().sum() / len(data)
        if (missing_ratio > 0.1).any():
            results['warnings'].append(
                f"High missing ratio: {missing_ratio[missing_ratio > 0.1].to_dict()}"
            )
            quality_checks.append(0.9)
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate rows")
            quality_checks.append(0.95)
        
        # Check data ranges
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in self.validation_rules.get('ranges', {}):
                min_val, max_val = self.validation_rules['ranges'][col]
                if (data[col] < min_val).any() or (data[col] > max_val).any():
                    results['warnings'].append(
                        f"Column '{col}' has values outside range [{min_val}, {max_val}]"
                    )
                    quality_checks.append(0.9)
        
        # Check for outliers (Z-score > 3)
        for col in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outliers = (z_scores > 3).sum()
            if outliers > 0:
                results['warnings'].append(
                    f"Found {outliers} outliers in column '{col}'"
                )
                quality_checks.append(0.95)
        
        # Compute quality score
        if quality_checks:
            results['quality_score'] = np.mean(quality_checks)
        
        # Check if valid
        if results['errors']:
            results['is_valid'] = False
        
        # Generate stats
        results['stats'] = {
            'rows': len(data),
            'columns': len(data.columns),
            'missing_count': data.isnull().sum().sum(),
            'duplicate_count': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum() / (1024**2)
        }
        
        logger.info(f"Data validation complete: {results['is_valid']}")
        
        return results
    
    def check_schema(self, data: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
        """
        Check if data matches expected schema
        
        Args:
            data: DataFrame to validate
            expected_schema: Dictionary of expected columns and types
            
        Returns:
            True if schema matches, False otherwise
        """
        is_valid = True
        
        for col, expected_type in expected_schema.items():
            if col not in data.columns:
                logger.error(f"Missing column: {col}")
                is_valid = False
                continue
            
            actual_type = str(data[col].dtype)
            if not self._type_matches(actual_type, expected_type):
                logger.error(f"Column '{col}' has type {actual_type}, expected {expected_type}")
                is_valid = False
        
        return is_valid
    
    def _type_matches(self, actual: str, expected: str) -> bool:
        """Check if actual type matches expected type"""
        type_mapping = {
            'int': ['int', 'int64', 'int32', 'int16', 'int8'],
            'float': ['float', 'float64', 'float32'],
            'object': ['object', 'string'],
            'datetime': ['datetime64', 'datetime'],
            'bool': ['bool', 'boolean']
        }
        
        expected_types = type_mapping.get(expected, [])
        return any(expected_type in actual.lower() for expected_type in expected_types)
    
    def validate_streaming_data(self, data: pd.DataFrame, window_size: int = 100) -> bool:
        """
        Validate streaming data in real-time
        
        Args:
            data: New data point or batch
            window_size: Size of sliding window
            
        Returns:
            True if data is valid for streaming
        """
        # Check if data is empty
        if len(data) == 0:
            logger.warning("Empty data received")
            return False
        
        # Check for required columns
        required = ['timestamp', 'value']  # Minimum required for streaming
        if not all(col in data.columns for col in required):
            logger.error("Missing required columns for streaming")
            return False
        
        # Check timestamp format
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except:
                logger.error("Invalid timestamp format")
                return False
        
        # Check for negative values
        for col in data.select_dtypes(include=[np.number]).columns:
            if (data[col] < 0).any():
                logger.warning(f"Negative values found in column '{col}'")
        
        return True