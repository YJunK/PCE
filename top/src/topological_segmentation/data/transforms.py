from __future__ import annotations

import numpy as np


def to_probability_map(mask: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Convert a binary mask to a probability-like map.

    This is a convenience function used in early toy experiments. A binary object
    is converted into a float array in [0, 1], which may later be used as a
    probability input for conventional overlap metrics or as a filtration input.
    """
    arr = np.asarray(mask, dtype=float)
    return np.clip(arr, low, high)
