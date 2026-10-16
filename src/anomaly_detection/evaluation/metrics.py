"""NAB scoring + classic pointwise metrics.

NAB reward: earlier detection within the anomaly window earns more credit.
"""
from typing import List, Tuple
import numpy as np


def pointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": f1, "false_positive_rate": fpr}


def detection_delay(true_windows: List[Tuple[int, int]], predicted_idx: List[int]) -> float:
    """For each true window, delay = (first predicted idx in window) - window_start.
    Windows with no prediction are ignored (missed). Returns mean delay, or NaN."""
    delays = []
    for (start, end) in true_windows:
        preds = [p for p in predicted_idx if start <= p <= end]
        if preds:
            delays.append(min(preds) - start)
    return float(np.mean(delays)) if delays else float("nan")


def nab_score(
    true_windows: List[Tuple[int, int]],
    predicted_idx: List[int],
    n: int,
    alpha: float = 0.2,
) -> float:
    """Simplified NAB score.

    For each true window [s,e]:
      - find first prediction p in [s,e]
      - if p == s (immediate): reward = 1
      - if p > s: reward = alpha^(p - s)  (earlier is better)
      - if no prediction: reward = 0
    Normalized by number of true windows.
    False positives (predictions outside any window) receive a small penalty.
    """
    window_lookup = np.full(n, -1, dtype=int)
    for i, (s, e) in enumerate(true_windows):
        window_lookup[s : e + 1] = i

    rewards = [0.0] * len(true_windows)
    for p in predicted_idx:
        wid = window_lookup[p] if 0 <= p < n else -1
        if wid == -1:
            continue  # counted as false positive below
        s, e = true_windows[wid]
        offset = p - s
        reward = 1.0 if offset == 0 else alpha ** offset
        if reward > rewards[wid]:
            rewards[wid] = reward

    fp = sum(1 for p in predicted_idx if not (0 <= p < n and window_lookup[p] != -1))
    raw = sum(rewards) - 0.1 * fp / max(n, 1)
    return float(raw / max(len(true_windows), 1))


def labels_to_windows(anomaly_indices: List[int]) -> List[Tuple[int, int]]:
    """Collapse a list of indices into contiguous windows."""
    if not anomaly_indices:
        return []
    idx = sorted(anomaly_indices)
    windows = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            windows.append((s, prev))
            s = i
        prev = i
    windows.append((s, prev))
    return windows