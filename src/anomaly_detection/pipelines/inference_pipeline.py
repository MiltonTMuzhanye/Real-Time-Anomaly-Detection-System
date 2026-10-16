import numpy as np
import joblib

from ..data.ingestion import load_nab_series
from ..data.preprocessing import make_windows, window_to_features
from ..evaluation.metrics import pointwise_metrics
from ..evaluation.threshold_analysis import pick_threshold_by_f1


def run_inference(model_path: str, series_path: str, window: int = 128,
                  stride: int = 1, threshold: float | None = None) -> dict:
    model = joblib.load(model_path)
    df = load_nab_series(series_path)
    vals = df["value"].values
    windows = make_windows(vals, window, stride)
    feats = window_to_features(windows)

    scores = np.asarray([model.score(f) for f in feats], dtype=np.float32)
    if threshold is None:
        threshold = 0.65

    preds = (scores >= threshold).astype(int)
    # Align predictions back to original time index
    pred_indices = [window - 1 + i * stride for i in range(len(preds)) if preds[i] == 1]
    return {
        "scores": scores.tolist(),
        "threshold": threshold,
        "pred_indices": pred_indices,
        "n_events": len(vals),
    }