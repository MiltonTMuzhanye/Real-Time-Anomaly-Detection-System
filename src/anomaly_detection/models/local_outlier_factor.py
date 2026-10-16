import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .base import BaseAnomalyDetector


class LOFDetector(BaseAnomalyDetector):
    name = "local_outlier_factor"

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.01):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model.fit(X)
        self._fitted = True
        return self

    def score(self, x: np.ndarray) -> float:
        raw = self.model.decision_function(x.reshape(1, -1))[0]
        return float(1.0 / (1.0 + np.exp(raw * 2.0)))