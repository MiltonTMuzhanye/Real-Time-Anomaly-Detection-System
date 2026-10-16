from typing import Iterable, List
import numpy as np


def batch_scores(detector, X: np.ndarray) -> np.ndarray:
    return np.asarray([detector.score(x) for x in X], dtype=np.float32)


def normalize_scores(scores: Iterable[float]) -> List[float]:
    arr = np.asarray(list(scores), dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return [0.0] * len(arr)
    return ((arr - lo) / (hi - lo)).tolist()