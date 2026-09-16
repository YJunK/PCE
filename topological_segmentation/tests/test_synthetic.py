import numpy as np

from topological_segmentation.data.synthetic_2d import make_disk


def test_make_disk_shape_and_values():
    mask = make_disk(size=64, radius=20)
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    assert mask[32, 32] == 1
