"""Simulated real-time stream from NAB with callback per event."""
import time
from typing import Callable, Iterator, Optional

from ..data.ingestion import load_nab_series


class ReplayStream:
    def __init__(self, series_path: str, replay_speed: float = 20.0):
        self.df = load_nab_series(series_path)
        self.speed = max(replay_speed, 1.0)
        self._deltas = self.df["timestamp"].diff().dt.total_seconds().fillna(0).values

    def __iter__(self) -> Iterator[dict]:
        for i, row in self.df.iterrows():
            if i > 0:
                time.sleep(min(self._deltas[i] / self.speed, 0.5))
            yield {"timestamp": row["timestamp"].isoformat(), "value": float(row["value"])}

    def run(self, on_event: Callable[[dict, Optional[dict]], None], engine) -> None:
        for event in self:
            result = engine.ingest(event)
            on_event(event, result)