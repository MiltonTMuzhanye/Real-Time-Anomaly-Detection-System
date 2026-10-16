import numpy as np
import pandas as pd

from .statistical_features import rolling_entropy


def build_feature_frame(df: pd.DataFrame, window: int = 128) -> pd.DataFrame:
    """Batch feature engineering for training."""
    v = df["value"].values.astype(np.float64)
    out = pd.DataFrame({"timestamp": df["timestamp"], "value": v})

    s = pd.Series(v)
    out["roll_mean"] = s.rolling(window, min_periods=window).mean()
    out["roll_std"] = s.rolling(window, min_periods=window).std()
    out["roll_min"] = s.rolling(window, min_periods=window).min()
    out["roll_max"] = s.rolling(window, min_periods=window).max()
    out["roll_z"] = (v - out["roll_mean"]) / (out["roll_std"] + 1e-8)
    out["roll_range"] = out["roll_max"] - out["roll_min"]

    # entropy over rolling window (slower, use stride for training only)
    ent = np.full(len(v), np.nan)
    for i in range(window, len(v)):
        ent[i] = rolling_entropy(v[i - window : i])
    out["roll_entropy"] = ent

    return out.dropna().reset_index(drop=True)