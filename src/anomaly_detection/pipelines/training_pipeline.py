from pathlib import Path

import joblib
import numpy as np

from ..data.ingestion import discover_nab_series, load_nab_series
from ..data.preprocessing import make_windows, window_to_features, train_val_test_split
from ..models.zscore_baseline import ZScoreDetector
from ..models.isolation_forest import IsolationForestDetector
from ..models.autoencoder import AutoencoderDetector
from ..evaluation.metrics import pointwise_metrics, labels_to_windows, nab_score
from ..evaluation.threshold_analysis import pick_threshold_by_f1
from ..utils.config import load_configs
from ..utils.helpers import ensure_dir, set_seed, save_json
from ..utils.logger import get_logger

logger = get_logger(__name__)


def run_training(config_dir: str = "configs") -> dict:
    cfg = load_configs(config_dir)
    set_seed(cfg["config"]["project"]["seed"])

    window = cfg["config"]["data"]["window_size"]
    stride = cfg["config"]["data"]["stride"]
    nab_root = cfg["config"]["paths"]["nab_root"]
    max_series = cfg["config"]["data"]["max_series"]

    series_files = discover_nab_series(nab_root, max_series)

    all_feats = []
    for f in series_files:
        df = load_nab_series(f)
        vals = df["value"].values
        windows = make_windows(vals, window, stride)
        if len(windows) == 0:
            continue
        feats = window_to_features(windows)
        all_feats.append(feats)
    X = np.vstack(all_feats)
    logger.info(f"Training matrix: {X.shape}")

    artifacts = ensure_dir(Path(cfg["config"]["paths"]["artifacts_dir"]) / "trained_models")
    results = {}

    z = ZScoreDetector(k=cfg["config"]["thresholds"]["k_sigma"]).fit(X)
    joblib.dump(z, artifacts / "zscore.joblib")
    results["zscore"] = {"path": str(artifacts / "zscore.joblib")}

    iso_cfg = cfg["model"]["isolation_forest"]
    iso = IsolationForestDetector(**iso_cfg).fit(X)
    joblib.dump(iso, artifacts / "isolation_forest.joblib")
    results["isolation_forest"] = {"path": str(artifacts / "isolation_forest.joblib")}

    ae_cfg = cfg["model"]["autoencoder"]
    ae = AutoencoderDetector(**ae_cfg).fit(X)
    joblib.dump(ae, artifacts / "autoencoder.joblib")
    results["autoencoder"] = {"path": str(artifacts / "autoencoder.joblib")}

    save_json(results, Path(cfg["config"]["paths"]["reports_dir"]) / "metrics" / "training_manifest.json")
    return results