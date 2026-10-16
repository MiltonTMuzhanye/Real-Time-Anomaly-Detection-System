from fastapi import APIRouter, HTTPException

from .schemas import EventIn, DetectionOut, StatsOut, AlertOut
from .dependencies import get_engine

router = APIRouter()


@router.post("/ingest", response_model=DetectionOut)
def ingest_event(event: EventIn):
    try:
        engine = get_engine()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    result = engine.ingest(event.model_dump())
    if result is None:
        # Not enough buffered events yet
        return DetectionOut(
            timestamp=event.timestamp,
            value=event.value,
            score=0.0,
            threshold=engine.threshold,
            is_anomaly=False,
        )
    return DetectionOut(**result)


@router.get("/stats", response_model=StatsOut)
def stats():
    engine = get_engine()
    return StatsOut(**engine.stats)


@router.get("/alerts", response_model=list[AlertOut])
def alerts(limit: int = 50):
    engine = get_engine()
    return [AlertOut(**a) for a in engine.alert_engine.alerts[-limit:]]


@router.post("/reset")
def reset():
    engine = get_engine()
    engine.extractor.reset()
    return {"status": "reset"}