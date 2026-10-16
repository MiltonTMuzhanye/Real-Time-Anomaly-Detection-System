from src.anomaly_detection.features.rolling_features import SlidingWindowFeatureExtractor
from src.anomaly_detection.models.zscore_baseline import ZScoreDetector
from src.anomaly_detection.models.isolation_forest import IsolationForestDetector
from app.inference.anomaly_engine import AnomalyEngine
import numpy as np


def _make_engine():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 10)).astype("float32")
    det = IsolationForestDetector(contamination=0.05).fit(X)
    ex = SlidingWindowFeatureExtractor(window_size=10, stride=1)
    return AnomalyEngine(ex, det, threshold=0.5)


def test_engine_streams():
    eng = _make_engine()
    rng = np.random.default_rng(1)
    for i in range(60):
        eng.ingest({"timestamp": f"t{i}", "value": float(rng.normal())})
    assert eng.stats["total_events"] == 60