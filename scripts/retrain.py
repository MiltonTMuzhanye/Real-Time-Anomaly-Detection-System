"""Retrain pipeline with new data. Same as train.py but logs a run to reports."""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.pipelines.training_pipeline import run_training
from anomaly_detection.utils.helpers import save_json


if __name__ == "__main__":
    results = run_training()
    save_json(
        {"timestamp": datetime.utcnow().isoformat(), "results": results},
        Path("reports/metrics") / "retrain_log.json",
    )
    print("Retrain complete.")
