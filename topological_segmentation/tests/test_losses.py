import numpy as np

from topological_segmentation.data.synthetic_2d import make_disk
from topological_segmentation.losses.segmentation import binary_cross_entropy, dice_loss, dice_score


def test_dice_identity():
    mask = make_disk(size=32, radius=10)
    assert np.isclose(dice_score(mask, mask), 1.0)
    assert np.isclose(dice_loss(mask, mask), 0.0)


def test_binary_cross_entropy_probability_input():
    mask = make_disk(size=32, radius=10).astype(float)
    probs = np.clip(mask, 0.1, 0.9)
    value = binary_cross_entropy(probs, probs)
    assert np.isfinite(value)
    assert value >= 0.0
