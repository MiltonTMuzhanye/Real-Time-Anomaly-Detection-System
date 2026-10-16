from collections import deque
from typing import Optional

import numpy as np


class SlidingWindowFeatureExtractor:
    """Ring buffer → 10 statistical features. Streaming-safe."""

    def __init__(self, window_size: int = 128, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self.buffer: deque = deque(maxlen=window_size)
        self._counter = 0

    def update(self, value: float) -> Optional[np.ndarray]:
        self.buffer.append(float(value))
        self._counter += 1
        if len(self.buffer) < self.window_size:
            return None
        if self._counter % self.stride != 0:
            return None
        arr = np.asarray(self.buffer, dtype=np.float64)
        return self._compute(arr)

    @staticmethod
    def _compute(arr: np.ndarray) -> np.ndarray:
        eps = 1e-8
        mean = arr.mean()
        std = arr.std()
        return np.asarray(
            [
                arr[-1],
                mean,
                std,
                (arr[-1] - mean) / (std + eps),
                arr.min(),
                arr.max(),
                arr.max() - arr.min(),
                float(np.median(arr)),
                float(np.percentile(arr, 25)),
                float(np.percentile(arr, 75)),
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.buffer.clear()
        self._counter = 0