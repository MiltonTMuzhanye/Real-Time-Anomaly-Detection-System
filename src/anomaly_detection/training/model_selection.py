import numpy as np


def select_best(results: dict, metric: str = "f1") -> str:
    """results: {model_name: {metric: value}}"""
    return max(results.items(), key=lambda kv: kv[1].get(metric, -np.inf))[0]