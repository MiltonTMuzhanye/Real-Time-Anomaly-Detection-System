import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def make_windows(values: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Return [n_windows, window_size] sliding windows."""
    n = len(values)
    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32)
    idx = np.arange(0, n - window_size + 1, stride)
    return np.stack([values[i : i + window_size] for i in idx]).astype(np.float32)


def window_to_features(windows: np.ndarray) -> np.ndarray:
    """Convert [N, W] → [N, F] with rolling statistical features."""
    eps = 1e-8
    last = windows[:, -1]
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    z = (last - mean) / (std + eps)
    rng = windows.max(axis=1) - windows.min(axis=1)
    med = np.median(windows, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    q75 = np.percentile(windows, 75, axis=1)
    return np.stack([last, mean, std, z, windows.min(axis=1), windows.max(axis=1),
                     rng, med, q25, q75], axis=1).astype(np.float32)


def train_val_test_split(values: np.ndarray, train_r: float, val_r: float):
    n = len(values)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return values[:n_train], values[n_train : n_train + n_val], values[n_train + n_val :]