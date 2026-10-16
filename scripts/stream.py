"""Simulate real-time streaming with AnomalyEngine. Prints detections live."""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.features.rolling_features import SlidingWindowFeatureExtractor
from anomaly_detection.utils.config import load_configs
from app.inference.anomaly_engine import AnomalyEngine
from app.inference.alert_engine import AlertEngine
from anomaly_detection.pipelines.streaming_pipeline import ReplayStream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/trained_models/isolation_forest.joblib")
    parser.add_argument("--series", default="data/raw/nab/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv")
    parser.add_argument("--speed", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    detector = joblib.load(args.model)
    cfg = load_configs()
    extractor = SlidingWindowFeatureExtractor(
        window_size=cfg["config"]["data"]["window_size"],
        stride=cfg["config"]["data"]["stride"],
    )
    alert_engine = AlertEngine(cooldown_events=10)
    engine = AnomalyEngine(extractor, detector, args.threshold, alert_engine)

    def on_event(event, result):
        if result is None:
            return
        flag = "🚨" if result["is_anomaly"] else "  "
        print(f"{flag} {result['timestamp']}  value={result['value']:.4f}  score={result['score']:.4f}")

    stream = ReplayStream(args.series, replay_speed=args.speed)
    stream.run(on_event, engine)

    print(f"\nDone. {engine.stats}")


if __name__ == "__main__":
    main()