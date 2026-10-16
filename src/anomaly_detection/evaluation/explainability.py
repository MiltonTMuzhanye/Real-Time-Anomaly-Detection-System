import numpy as np


FEATURE_NAMES = [
    "last", "mean", "std", "z_score", "min",
    "max", "range", "median", "q25", "q75",
]


def feature_contribution(model, x: np.ndarray, baseline: np.ndarray) -> dict:
    """Simple finite-difference attribution for tree detectors."""
    base_score = model.score(x)
    contribs = {}
    for i, name in enumerate(FEATURE_NAMES):
        x_perturbed = x.copy()
        x_perturbed[i] = baseline[i]
        delta = base_score - model.score(x_perturbed)
        contribs[name] = float(delta)
    return contribs