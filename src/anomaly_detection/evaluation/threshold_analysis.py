from typing import Callable, List
import numpy as np

from .metrics import pointwise_metrics


def sweep_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: List[float] | None = None,
) -> List[dict]:
    if thresholds is None:
        thresholds = list(np.linspace(0.05, 0.99, 30))
    out = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        m = pointwise_metrics(y_true, preds)
        m["threshold"] = float(t)
        out.append(m)
    return out


def pick_threshold_by_f1(scores: np.ndarray, y_true: np.ndarray) -> float:
    results = sweep_thresholds(scores, y_true)
    best = max(results, key=lambda r: r["f1"])
    return float(best["threshold"])