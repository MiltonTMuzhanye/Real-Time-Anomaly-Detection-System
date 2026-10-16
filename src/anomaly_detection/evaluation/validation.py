import numpy as np


def time_series_cv_splits(n: int, n_splits: int = 5):
    """Yield (train_idx, val_idx) with expanding window, no leakage."""
    fold = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold * k
        val_end = fold * (k + 1)
        yield np.arange(0, train_end), np.arange(train_end, val_end)