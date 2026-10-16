"""Evaluate a trained detector on a NAB series with ground-truth labels."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.data.ingestion import load_nab_series, load_nab_labels
from anomaly_detection.data.preprocessing import make_windows, window_to_features
from anomaly_detection.evaluation.metrics import (
    pointwise_metrics, detection_delay, nab_score, labels_to_windows,
)
from anomaly_detection.evaluation.threshold_analysis import pick_threshold_by_f1
import joblib


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/trained_models/isolation_forest.joblib"
    series_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/nab/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"

    model = joblib.load(model_path)
    df = load_nab_series(series_path)
    vals = df["value"].values
    window, stride = 128, 1
    windows = make_windows(vals, window, stride)
    feats = window_to_features(windows)
    scores = np.asarray([model.score(f) for f in feats])

    # Label alignment
    labels = load_nab_labels("data/raw/nab")
    key = str(Path(series_path)).replace(str(Path("data/raw/nab")), "").lstrip("/")
    key = key.replace("\\", "/")
    windows_true = []
    for label_key, windows_list in labels.items():
        if label_key.endswith(key):
            for w in windows_list:
                s = df[df["timestamp"] >= w[0]].index.min()
                e = df[df["timestamp"] <= w[1]].index.max()
                if s is not None and e is not None:
                    windows_true.append((int(s), int(e)))
            break

    # Ground truth vector aligned to feature indices
    y_true = np.zeros(len(scores), dtype=int)
    for (s, e) in windows_true:
        for i, w_start in enumerate(range(0, len(vals) - window + 1, stride)):
            w_end = w_start + window - 1
            if not (w_end < s or w_start > e):
                y_true[i] = 1

    threshold = pick_threshold_by_f1(scores, y_true)
    preds = (scores >= threshold).astype(int)

    pm = pointwise_metrics(y_true, preds)
    pred_idx = np.where(preds == 1)[0].tolist()
    dd = detection_delay(windows_true, pred_idx) if windows_true else float("nan")
    nab = nab_score(windows_true, pred_idx, len(scores)) if windows_true else 0.0

    print(json.dumps({
        "series": series_path,
        "n_features": int(len(scores)),
        "threshold": threshold,
        "pointwise": pm,
        "detection_delay": dd,
        "nab_score": nab,
        "n_true_windows": len(windows_true),
    }, indent=2))


if __name__ == "__main__":
    main()