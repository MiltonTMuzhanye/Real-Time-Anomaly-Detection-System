import numpy as np


def coerce_event(event: dict) -> tuple[str, float]:
    """Return (timestamp, value) from dict; raise on malformed input."""
    if "timestamp" not in event or "value" not in event:
        raise ValueError(f"Malformed event: {event}")
    return str(event["timestamp"]), float(event["value"])