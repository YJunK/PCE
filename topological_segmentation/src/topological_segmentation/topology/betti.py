from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def compute_betti_numbers(binary_mask: np.ndarray) -> tuple[int, int]:
    """Compute Betti numbers (beta_0, beta_1) for a binary 2D mask.

    This function is meant to serve as a compact and interpretable sanity check for
    the persistence pipeline. It is not a replacement for persistent homology; it
    is a manually computed baseline for small synthetic shapes.

    Interpretation:
    - beta_0: number of connected foreground components.
    - beta_1: number of holes (background components not touching the border).

    Examples:
    - filled disk -> (1, 0)
    - ring -> (1, 1)
    - two disconnected disks -> (2, 0)
    """
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.size == 0:
        return (0, 0)

    structure = np.ones((3, 3), dtype=np.uint8)
    _, n_comp = label(mask, structure=structure)
    beta_0 = int(n_comp)

    background = ~mask
    bg_labels, n_bg = label(background, structure=structure)

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    outside_ids = set(np.unique(bg_labels[border]))
    holes = 0
    for label_id in range(1, n_bg + 1):
        if label_id not in outside_ids:
            holes += 1

    beta_1 = holes
    return (beta_0, beta_1)
