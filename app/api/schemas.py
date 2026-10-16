from typing import Optional
from pydantic import BaseModel, Field


class EventIn(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    value: float


class DetectionOut(BaseModel):
    timestamp: str
    value: float
    score: float
    threshold: float
    is_anomaly: bool
    alert_id: Optional[str] = None


class StatsOut(BaseModel):
    total_events: int
    total_alerts: int


class AlertOut(BaseModel):
    alert_id: str
    timestamp: str
    value: float
    score: float
    threshold: float
    emitted_at: float