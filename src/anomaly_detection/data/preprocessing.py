"""
Data Preprocessing Module - Cleaning and preparing data for models
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """
    Data preprocessing pipeline for anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocess_config = config.get('preprocessing', {})
        
        # Initialize transformers
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.is_fitted = False
        
        # Initialize based on config
        self._initialize_transformers()
    
    def _initialize_transformers(self):
        """Initialize preprocessing transformers"""
        # Numerical preprocessing
        num_strategy = self.preprocess_config.get('numerical', {}).get('imputation', 'mean')
        scaling_method = self.preprocess_config.get('numerical', {}).get('scaling', 'standard')
        
        if num_strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif num_strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif num_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy='mean')
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        self.numerical_transformer = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler)
        ])
        
        # Categorical preprocessing
        cat_strategy = self.preprocess_config.get('categorical', {}).get('encoding', 'onehot')
        
        if cat_strategy == 'onehot':
            self.categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        else:
            self.categorical_transformer = 'passthrough'
        
        logger.info("Preprocessing transformers initialized")
    
    def fit(self, data: pd.DataFrame):
        """Fit preprocessor on training data"""
        try:
            # Separate columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create column transformer
            transformers = []
            
            if numerical_cols:
                transformers.append(
                    ('num', self.numerical_transformer, numerical_cols)
                )
            
            if categorical_cols and self.categorical_transformer is not None:
                transformers.append(
                    ('cat', self.categorical_transformer, categorical_cols)
                )
            
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )
            
            # Fit preprocessor
            self.preprocessor.fit(data)
            self.is_fitted = True
            
            logger.info(f"Preprocessor fitted on {len(data)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting preprocessor: {e}")
            raise
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            logger.warning("Preprocessor not fitted. Fitting on data.")
            self.fit(data)
            return self.transform(data)
        
        try:
            transformed = self.preprocessor.transform(data)
            
            # Convert to dense if sparse
            if hasattr(transformed, 'toarray'):
                transformed = transformed.toarray()
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            raise
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Inverse transform data back to original space"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
        
        try:
            # Inverse transform
            inverse_data = self.preprocessor.inverse_transform(data)
            
            # Get feature names
            feature_names = self.preprocessor.get_feature_names_out()
            
            return pd.DataFrame(inverse_data, columns=feature_names)
            
        except Exception as e:
            logger.error(f"Error inverse transforming data: {e}")
            raise
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in data"""
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'mode':
            return data.fillna(data.mode().iloc[0])
        elif strategy == 'drop':
            return data.dropna()
        elif strategy == 'interpolate':
            return data.interpolate(method='linear', limit_direction='both')
        else:
            return data