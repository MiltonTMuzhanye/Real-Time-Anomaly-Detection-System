from sklearn.preprocessing import StandardScaler
import numpy as np


class FeatureScaler:
    """Wraps sklearn StandardScaler with save/load."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)