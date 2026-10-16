import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, contamination: float = 0.01, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X)
        self._fitted = True
        return self

    def score(self, x: np.ndarray) -> float:
        if not self._fitted:
            raise RuntimeError("Detector not fitted")
        raw = self.model.decision_function(x.reshape(1, -1))[0]
        # sigmoid squash → [0,1], higher = more anomalous
        return float(1.0 / (1.0 + np.exp(raw * 4.0)))

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.decision_function(X)
        return (1.0 / (1.0 + np.exp(raw * 4.0))).astype(np.float32)