from functools import lru_cache
from pathlib import Path

import joblib

from ...anomaly_detection.features.rolling_features import SlidingWindowFeatureExtractor
from ...anomaly_detection.utils.config import load_configs
from ..inference.anomaly_engine import AnomalyEngine
from ..inference.alert_engine import AlertEngine


@lru_cache(maxsize=1)
def get_engine() -> AnomalyEngine:
    """Load once at process start. Requires a trained model artifact."""
    cfg = load_configs()
    model_path = Path(cfg["config"]["paths"]["artifacts_dir"]) / "trained_models" / "isolation_forest.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Run `python scripts/train.py` first."
        )
    detector = joblib.load(model_path)
    extractor = SlidingWindowFeatureExtractor(
        window_size=cfg["config"]["data"]["window_size"],
        stride=cfg["config"]["data"]["stride"],
    )
    alert_engine = AlertEngine(cooldown_events=cfg["streaming"]["stream"]["cooldown_events"])
    threshold = cfg["config"]["thresholds"]["fixed"]
    return AnomalyEngine(extractor, detector, threshold, alert_engine)