from collections import deque
from typing import Callable, Optional
import time

from ...utils.logger import get_logger

logger = get_logger(__name__)


class AlertEngine:
    """Deduplicates alerts via cooldown and streams them to a sink."""

    def __init__(self, cooldown_events: int = 10, sink: Optional[Callable[[dict], None]] = None):
        self.cooldown_events = cooldown_events
        self.sink = sink
        self.history: deque = deque(maxlen=1000)
        self._since_last = 10**9

    def notify(self, result: dict) -> bool:
        self._since_last += 1
        if not result.get("is_anomaly"):
            return False
        if self._since_last < self.cooldown_events:
            return False
        self._since_last = 0
        alert = {
            "alert_id": f"alert-{len(self.history)+1}",
            "timestamp": result["timestamp"],
            "value": result["value"],
            "score": result["score"],
            "threshold": result["threshold"],
            "emitted_at": time.time(),
        }
        self.history.append(alert)
        if self.sink:
            self.sink(alert)
        logger.info(f"ALERT {alert['alert_id']} ts={alert['timestamp']} score={alert['score']:.4f}")
        return True

    @property
    def alerts(self) -> list[dict]:
        return list(self.history)