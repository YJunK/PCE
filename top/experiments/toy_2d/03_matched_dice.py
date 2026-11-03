from __future__ import annotations

from topological_segmentation.data.synthetic_2d import make_disk
from topological_segmentation.data.perturbations import add_hole, erode_boundary
from topological_segmentation.losses.segmentation import binary_cross_entropy, dice_score
from topological_segmentation.topology.betti import compute_betti_numbers


def main():
    """Search for a matched-Dice setting.

    This experiment tries to find a boundary perturbation and a hole perturbation
    whose Dice scores are close but whose topologies differ. This is the most
    direct toy analog of the central research question: similar overlap metrics
    can hide different topology.
    """
    gt = make_disk(size=64, radius=20)

    for amount in range(1, 20):
        boundary = erode_boundary(gt, amount=amount)
        hole = add_hole(gt, radius=amount)

        dice_a = dice_score(boundary, gt)
        dice_b = dice_score(hole, gt)

        if abs(dice_a - dice_b) < 0.04:
            print(
                {
                    "amount": amount,
                    "dice_boundary": dice_a,
                    "dice_hole": dice_b,
                    "bce_boundary": binary_cross_entropy(boundary.astype(float), gt.astype(float)),
                    "bce_hole": binary_cross_entropy(hole.astype(float), gt.astype(float)),
                    "beta_boundary": compute_betti_numbers(boundary),
                    "beta_hole": compute_betti_numbers(hole),
                }
            )
            break


if __name__ == "__main__":
    main()
