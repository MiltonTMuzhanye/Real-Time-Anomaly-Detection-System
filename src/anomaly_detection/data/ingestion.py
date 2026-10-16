import json
from pathlib import Path
from typing import Dict, List, Iterator, Any

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_nab_series(filepath: str | Path) -> pd.DataFrame:
    """Load one NAB CSV (columns: timestamp,value)."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["value"] = df["value"].astype(float)
    return df


def load_nab_labels(nab_root: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load NAB labels.json → {series_path: [windows]}."""
    labels_path = Path(nab_root) / "labels" / "combined_labels.json"
    if not labels_path.exists():
        labels_path = Path(nab_root) / "labels" / "combined_windows.json"
    with open(labels_path) as f:
        return json.load(f)


def discover_nab_series(nab_root: str | Path, max_series: int = 20) -> List[Path]:
    """Return up to max_series NAB CSV files."""
    root = Path(nab_root) / "data"
    files = sorted(root.rglob("*.csv"))
    logger.info(f"Discovered {len(files)} NAB series, using {min(max_series, len(files))}")
    return files[:max_series]


def stream_nab_series(filepath: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield one event dict per row."""
    df = load_nab_series(filepath)
    for _, row in df.iterrows():
        yield {
            "timestamp": row["timestamp"].isoformat(),
            "value": float(row["value"]),
        }