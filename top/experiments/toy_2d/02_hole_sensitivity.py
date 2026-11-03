from __future__ import annotations

from topological_segmentation.data.synthetic_2d import make_disk
from topological_segmentation.data.perturbations import add_hole
from topological_segmentation.losses.segmentation import binary_cross_entropy, dice_score
from topological_segmentation.topology.betti import compute_betti_numbers


def main():
    """Sweep the hole radius and inspect how it affects overlap and topology.

    This is a simple sensitivity analysis to show that a small topology-breaking
    perturbation can incrementally change the Betti number while still producing
    comparable overlap metrics in some regimes.
    """
    gt = make_disk(size=64, radius=20)

    for radius in range(0, 11):
        pred = add_hole(gt, radius=radius) if radius > 0 else gt.copy()
        beta_0, beta_1 = compute_betti_numbers(pred)
        print(
            {
                "hole_radius": radius,
                "dice": dice_score(pred, gt),
                "bce": binary_cross_entropy(pred.astype(float), gt.astype(float)),
                "beta_0": beta_0,
                "beta_1": beta_1,
            }
        )


if __name__ == "__main__":
    main()
