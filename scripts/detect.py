"""Detect anomalies in a single NAB CSV and print results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.pipelines.inference_pipeline import run_inference


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/trained_models/isolation_forest.joblib"
    series_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/nab/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
    result = run_inference(model_path, series_path)
    print(f"Threshold: {result['threshold']}")
    print(f"Detected {len(result['pred_indices'])} anomalous points")
    print(f"First 10 indices: {result['pred_indices'][:10]}")