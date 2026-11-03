from __future__ import annotations

import numpy as np


def persistence_diagram_to_array(diagram):
    """Convert a persistence diagram object to a NumPy array of shape (n, 2).

    The simplest representation is a list of (birth, death) pairs. This helper is
    useful for downstream visualization or table export in small experimental
    scripts.
    """
    if diagram is None:
        return np.empty((0, 2), dtype=float)
    return np.asarray([[birth, death] for _, birth, death in diagram], dtype=float)
