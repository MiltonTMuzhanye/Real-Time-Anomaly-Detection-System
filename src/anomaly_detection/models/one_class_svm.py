import numpy as np
from sklearn.svm import OneClassSVM

from .base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    name = "one_class_svm"

    def __init__(self, kernel: str = "rbf", nu: float = 0.01, gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        self.model.fit(X)
        self._fitted = True
        return self

    def score(self, x: np.ndarray) -> float:
        raw = self.model.decision_function(x.reshape(1, -1))[0]
        return float(1.0 / (1.0 + np.exp(raw * 2.0)))