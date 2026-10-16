import numpy as np

from .base import BaseAnomalyDetector


class ZScoreDetector(BaseAnomalyDetector):
    """Statistical baseline: |(x - mean)/std| > k."""
    name = "zscore"

    def __init__(self, k: float = 3.0):
        self.k = k
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        # use last value column only, since features are [N, F]
        self.mean_ = float(X[:, 0].mean())
        self.std_ = float(X[:, 0].std() + 1e-8)
        return self

    def score(self, x: np.ndarray) -> float:
        z = abs((x[0] - self.mean_) / self.std_)
        return float(min(z / self.k, 3.0) / 3.0)