"""Train all detectors on NAB."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.pipelines.training_pipeline import run_training


if __name__ == "__main__":
    results = run_training()
    print("Trained models:")
    for k, v in results.items():
        print(f"  {k}: {v['path']}")