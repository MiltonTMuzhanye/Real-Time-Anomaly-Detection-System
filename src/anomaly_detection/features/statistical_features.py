import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - m) / s


def rolling_entropy(x: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())