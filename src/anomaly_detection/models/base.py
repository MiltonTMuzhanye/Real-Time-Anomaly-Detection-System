from abc import ABC, abstractmethod
import numpy as np


class BaseAnomalyDetector(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomalyDetector":
        ...

    @abstractmethod
    def score(self, x: np.ndarray) -> float:
        ...

    def predict(self, x: np.ndarray, threshold: float) -> bool:
        return self.score(x) > threshold