"""
Feature Engineering Module - Creating features for anomaly detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """
    Feature engineering for anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_config = config.get('features', {})
        self.numerical_features = []
        self.categorical_features = []
        self.derived_features = []
        self.selected_features = []
        self.pca = None
        self.feature_selector = None
        self.is_fitted = False
        
        # Initialize based on config
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize feature configurations"""
        self.numerical_features = self.feature_config.get('numerical', [])
        self.categorical_features = self.feature_config.get('categorical', [])
        
        logger.info(f"Initialized with {len(self.numerical_features)} numerical and "
                   f"{len(self.categorical_features)} categorical features")
    
    def extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a single data point
        """
        features = []
        
        # Extract numerical features
        for feature in self.numerical_features:
            value = data.get(feature, 0.0)
            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                features.append(0.0)
        
        # Extract categorical features (one-hot encoding)
        if self.categorical_features:
            for feature in self.categorical_features:
                value = data.get(feature, 'unknown')
                # One-hot encode (simple mapping)
                encoded = self._one_hot_encode(feature, str(value))
                features.extend(encoded)
        
        # Create derived features
        derived = self.create_derived_features(data)
        for feature in derived:
            if isinstance(derived[feature], (int, float)):
                features.append(float(derived[feature]))
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def extract_features_batch(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from a batch of data
        """
        feature_list = []
        
        for _, row in data.iterrows():
            features = self.extract_features(row.to_dict())
            feature_list.append(features)
        
        return np.array(feature_list)
    
    def create_derived_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create derived features from raw data
        """
        derived = {}
        
        try:
            # Ratio features
            if 'bytes_in' in data and 'packets_in' in data:
                if data['packets_in'] > 0:
                    derived['bytes_per_packet_in'] = data['bytes_in'] / data['packets_in']
                else:
                    derived['bytes_per_packet_in'] = 0
            
            if 'bytes_out' in data and 'packets_out' in data:
                if data['packets_out'] > 0:
                    derived['bytes_per_packet_out'] = data['bytes_out'] / data['packets_out']
                else:
                    derived['bytes_per_packet_out'] = 0
            
            # Total features
            if 'bytes_in' in data and 'bytes_out' in data:
                derived['total_bytes'] = data['bytes_in'] + data['bytes_out']
                derived['byte_ratio'] = data['bytes_in'] / max(1, data['bytes_out'])
            
            # Rate features
            if 'duration' in data and data['duration'] > 0:
                derived['bytes_rate'] = derived.get('total_bytes', 0) / data['duration']
                derived['packets_rate'] = (data.get('packets_in', 0) + data.get('packets_out', 0)) / data['duration']
            
            # Time features
            if 'timestamp' in data:
                timestamp = data['timestamp']
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                derived['hour_of_day'] = timestamp.hour
                derived['day_of_week'] = timestamp.dayofweek
                derived['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
                derived['is_business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
                derived['minute_of_hour'] = timestamp.minute
            
            # Protocol features
            if 'protocol' in data:
                protocol = str(data['protocol']).lower()
                derived['is_tcp'] = 1 if protocol == 'tcp' else 0
                derived['is_udp'] = 1 if protocol == 'udp' else 0
                derived['is_icmp'] = 1 if protocol == 'icmp' else 0
            
        except Exception as e:
            logger.warning(f"Error creating derived features: {e}")
        
        return derived
    
    def create_rolling_features(self, data: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Create rolling window features
        """
        if len(data) < 2:
            return data
        
        df = data.copy()
        
        for col in self.numerical_features:
            if col in df.columns:
                # Rolling statistics
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
                df[f'{col}_rolling_max'] = df[col].rolling(window=window_size).max()
                df[f'{col}_rolling_min'] = df[col].rolling(window=window_size).min()
                df[f'{col}_rolling_median'] = df[col].rolling(window=window_size).median()
                df[f'{col}_rolling_q25'] = df[col].rolling(window=window_size).quantile(0.25)
                df[f'{col}_rolling_q75'] = df[col].rolling(window=window_size).quantile(0.75)
        
        return df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from the data
        """
        if len(data) < 2:
            return data
        
        df = data.copy()
        
        for col in self.numerical_features:
            if col in df.columns:
                # Statistical features
                df[f'{col}_mean'] = df[col].mean()
                df[f'{col}_std'] = df[col].std()
                df[f'{col}_skew'] = df[col].skew()
                df[f'{col}_kurtosis'] = df[col].kurtosis()
                df[f'{col}_min'] = df[col].min()
                df[f'{col}_max'] = df[col].max()
                df[f'{col}_range'] = df[col].max() - df[col].min()
                df[f'{col}_iqr'] = df[col].quantile(0.75) - df[col].quantile(0.25)
        
        return df
    
    def _one_hot_encode(self, feature: str, value: str) -> List[float]:
        """One-hot encode a categorical feature"""
        # In a real implementation, you'd use sklearn's OneHotEncoder
        # This is a simplified version
        categories = ['tcp', 'udp', 'icmp', 'unknown']
        return [1.0 if value.lower() == cat else 0.0 for cat in categories]
    
    def select_features(self, X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Select top k features using ANOVA F-test
        """
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        self.is_fitted = True
        
        logger.info(f"Selected {k} features")
        
        return X_selected
    
    def reduce_dimensions(self, X: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        Reduce dimensionality using PCA
        """
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        logger.info(f"Reduced from {X.shape[1]} to {n_components} dimensions")
        
        return X_reduced