import numpy as np

from ..data.preprocessing import make_windows, window_to_features


def batch_detect(model, values: np.ndarray, window: int, stride: int,
                 threshold: float) -> dict:
    windows = make_windows(values, window, stride)
    feats = window_to_features(windows)
    scores = np.asarray([model.score(f) for f in feats])
    preds = (scores >= threshold).astype(int)
    return {
        "scores": scores,
        "preds": preds,
        "pred_indices": [window - 1 + i * stride for i, p in enumerate(preds) if p],
    }
