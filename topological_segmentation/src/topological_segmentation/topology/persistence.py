from __future__ import annotations

import numpy as np
import gudhi # TDA library for computing persistent homology

from .filtrations import probability_filtration


def compute_cubical_persistence(prob_map: np.ndarray, homology_dimensions=(0, 1)):
    """Compute cubical persistent homology using GUDHI.

    This is the core topological primitive for the 2D toy segmentation setting.
    We intentionally keep the implementation small and explicit:

    1. convert a prob_map to a filtration,
    2. build a CubicalComplex,
    3. compute persistence pairs,
    4. optionally filter by homology dimension.

    In this project, the probability map is treated as a foreground-confidence
    field, so the filtration uses 1.0 - prob_map. This ensures the filtration is
    aligned with the standard sublevel-set convention used by persistent homology.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    filtration = probability_filtration(prob_map) # 1 - prob_map to align with sublevel-set convention

    cpx = gudhi.CubicalComplex(
        dimensions=prob_map.shape,
        top_dimensional_cells=filtration,
    )
    persistence = cpx.persistence()

    if not homology_dimensions:
        return persistence
    return [entry for entry in persistence if entry[0] in homology_dimensions]


def get_persistence_diagram(prob_map: np.ndarray, homology_dim: int):
    """Return persistence entries for one homology dimension.

    We keep this wrapper intentionally thin so future changes in the persistence
    implementation can be isolated without touching experiment scripts.
    """
    return compute_cubical_persistence(prob_map, homology_dimensions=(homology_dim,))


def compute_betti_numbers_from_persistence(prob_map: np.ndarray):
    """Convenience function for retrieving persistence entries.

    This is not a full Betti-number implementation; it is only a quick access
    tool for inspecting the persistence output in experiment scripts.
    """
    return compute_cubical_persistence(prob_map)
