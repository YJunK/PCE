from __future__ import annotations

import numpy as np


def probability_filtration(prob_map: np.ndarray) -> np.ndarray:
    """Create a filtration from a probability map.

    Filtration convention used here:

    filtration = 1.0 - probability_map

    Why this choice?
    - Standard sublevel-set persistence adds low function values first.
    - In segmentation, higher probability usually means higher foreground
      confidence.
    - Therefore, to align the interpretation with standard persistence, we map
      foreground confidence to a decreasing filtration value.

    This convention is important: if it is reversed silently, one can easily get
    results that look mathematically valid but have wrong topological meaning.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    if prob_map.min() < 0.0 or prob_map.max() > 1.0:
        raise ValueError("probability maps should lie in [0, 1]")
    return 1.0 - prob_map


def scalar_field_filtration(scalar_field: np.ndarray) -> np.ndarray:
    """Return a generic scalar-field filtration.

    This placeholder helps keep the pipeline modular. Future work may include
    thickness filtration, distance-transform filtration, signed-distance fields,
    or other scalar functions derived from image or mesh data.
    """
    return np.asarray(scalar_field, dtype=float)
