import numpy as np
from src.anomaly_detection.features.rolling_features import SlidingWindowFeatureExtractor


def test_extractor_returns_none_until_full():
    ex = SlidingWindowFeatureExtractor(window_size=5)
    for i in range(4):
        assert ex.update(i) is None
    assert ex.update(4) is not None


def test_extractor_shape():
    ex = SlidingWindowFeatureExtractor(window_size=8, stride=1)
    out = None
    for i in range(10):
        out = ex.update(float(i))
    assert out is not None
    assert out.shape == (10,)