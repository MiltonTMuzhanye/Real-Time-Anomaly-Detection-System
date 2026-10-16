class AnomalyDetectionError(Exception):
    """Base exception for the project."""

class DataValidationError(AnomalyDetectionError):
    """Raised when input data fails validation."""

class ModelNotFittedError(AnomalyDetectionError):
    """Raised when a detector is used before fit()."""

class ThresholdError(AnomalyDetectionError):
    """Raised for invalid threshold configuration."""