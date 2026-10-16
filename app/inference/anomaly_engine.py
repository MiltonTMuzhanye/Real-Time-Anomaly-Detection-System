from typing import Optional, Callable, Dict, Any
import numpy as np

from ...anomaly_detection.features.rolling_features import SlidingWindowFeatureExtractor
from ...anomaly_detection.models.base import BaseAnomalyDetector
from .alert_engine import AlertEngine


class AnomalyEngine:
    """Orchestrates feature extraction → scoring → thresholding → alerting."""

    def __init__(
        self,
        extractor: SlidingWindowFeatureExtractor,
        detector: BaseAnomalyDetector,
        threshold: float,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.extractor = extractor
        self.detector = detector
        self.threshold = threshold
        self.alert_engine = alert_engine or AlertEngine()
        self._total_events = 0
        self._total_alerts = 0
        self._recent: list[dict] = []

    def ingest(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._total_events += 1
        value = float(event["value"])
        features = self.extractor.update(value)
        if features is None:
            return None

        score = self.detector.score(features)
        is_anomaly = score > self.threshold
        result = {
            "timestamp": event["timestamp"],
            "value": value,
            "score": round(score, 6),
            "threshold": self.threshold,
            "is_anomaly": is_anomaly,
        }
        if self.alert_engine.notify(result):
            result["alert_id"] = self.alert_engine.history[-1]["alert_id"]
            self._total_alerts += 1

        self._recent.append(result)
        if len(self._recent) > 500:
            self._recent = self._recent[-500:]
        return result

    @property
    def stats(self) -> Dict[str, int]:
        return {"total_events": self._total_events, "total_alerts": self._total_alerts}

    @property
    def recent(self) -> list[dict]:
        return list(self._recent)