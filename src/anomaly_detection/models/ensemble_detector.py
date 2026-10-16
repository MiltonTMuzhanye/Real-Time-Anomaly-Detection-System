from typing import List
import numpy as np

from .base import BaseAnomalyDetector


class EnsembleDetector(BaseAnomalyDetector):
    name = "ensemble"

    def __init__(self, detectors: List[BaseAnomalyDetector], weights=None,
                 aggregation: str = "weighted_mean"):
        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)
        self.aggregation = aggregation

    def fit(self, X: np.ndarray) -> "EnsembleDetector":
        for d in self.detectors:
            d.fit(X)
        return self

    def score(self, x: np.ndarray) -> float:
        scores = np.array([d.score(x) for d in self.detectors])
        if self.aggregation == "max":
            return float(scores.max())
        if self.aggregation == "vote":
            return float((scores > 0.5).mean())
        # weighted_mean
        return float(np.average(scores, weights=self.weights))