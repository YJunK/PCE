"""Topological segmentation prototype package.

The package is intentionally kept small and modular. The current code focuses on
2D synthetic experiments and the topology pipeline:

- synthetic data generation,
- deterministic perturbations,
- scalar-field filtration,
- persistent homology for cubical complexes,
- overlap-based metrics,
- topology sanity checks for toy experiments.

Future work can extend these building blocks for 3D volumes, cortical meshes,
scalar-field filtrations, and topology-aware training losses without needing to
rewrite the entire project structure.
"""

from .data.synthetic_2d import make_disk, make_ring, make_two_disks
from .losses.segmentation import binary_cross_entropy, dice_loss, dice_score
from .topology.betti import compute_betti_numbers

__all__ = [
    "make_disk",
    "make_ring",
    "make_two_disks",
    "compute_betti_numbers",
    "dice_score",
    "dice_loss",
    "binary_cross_entropy",
]
