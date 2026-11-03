from __future__ import annotations

from topological_segmentation.data.synthetic_2d import make_disk
from topological_segmentation.data.perturbations import (
    add_component,
    add_hole,
    erode_boundary,
    split_object,
)
from topological_segmentation.losses.segmentation import binary_cross_entropy, dice_score
from topological_segmentation.topology.betti import compute_betti_numbers


def main():
    """Sanity-check the main toy topology examples.

    This script creates a ground-truth disk and a small set of controlled
    predictions: perfect, boundary-deformed, hole-inserted, split, and extra-
    component. The script prints deterministic metrics so the user can inspect
    how overlap-based loss and topological summaries diverge.
    """
    gt = make_disk(size=64, radius=20)

    preds = {
        "perfect": gt,
        "boundary": erode_boundary(gt, amount=2),
        "hole": add_hole(gt, radius=6),
        "split": split_object(gt, width=2),
        "extra_component": add_component(gt, radius=8, center=(20, 20)),
    }

    rows = []
    for name, pred in preds.items():
        beta_0, beta_1 = compute_betti_numbers(pred)
        rows.append(
            {
                "name": name,
                "dice": float(dice_score(pred, gt)),
                "bce": float(binary_cross_entropy(pred.astype(float), gt.astype(float))),
                "beta_0": beta_0,
                "beta_1": beta_1,
            }
        )
    print(rows)


if __name__ == "__main__":
    main()
