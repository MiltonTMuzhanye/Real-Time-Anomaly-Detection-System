from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

from ..utils.helpers import ensure_dir
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, artifacts_dir: str = "artifacts/trained_models"):
        self.artifacts_dir = ensure_dir(artifacts_dir)

    def train(self, detector, X: np.ndarray, name: str) -> Dict[str, Any]:
        logger.info(f"Training {name} on {X.shape}")
        detector.fit(X)
        path = Path(self.artifacts_dir) / f"{name}.joblib"
        joblib.dump(detector, path)
        logger.info(f"Saved {name} → {path}")
        return {"name": name, "path": str(path), "n_samples": int(X.shape[0])}