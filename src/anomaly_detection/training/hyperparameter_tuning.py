"""Grid-search utilities for detector hyperparameters."""
from itertools import product
import numpy as np


def grid_search(build_fn, param_grid: dict, X_train, X_val, scorer):
    keys = list(param_grid.keys())
    best = {"score": -np.inf, "params": None}
    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        model = build_fn(**params).fit(X_train)
        score = scorer(model, X_val)
        if score > best["score"]:
            best = {"score": score, "params": params}
    return best