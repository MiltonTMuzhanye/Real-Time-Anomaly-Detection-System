import numpy as np
from src.anomaly_detection.evaluation.metrics import pointwise_metrics, nab_score


def test_pointwise_metrics_perfect():
    y = np.array([0, 1, 1, 0, 0])
    p = np.array([0, 1, 1, 0, 0])
    m = pointwise_metrics(y, p)
    assert m["f1"] == 1.0
    assert m["false_positive_rate"] == 0.0


def test_nab_score_early_reward():
    # Two windows; prediction at start → full credit
    windows = [(10, 20)]
    score_early = nab_score(windows, [10], n=100)
    score_late = nab_score(windows, [19], n=100)
    assert score_early > score_late