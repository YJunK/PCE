import numpy as np

from topological_segmentation.data.perturbations import add_component, add_hole, erode_boundary, split_object
from topological_segmentation.data.synthetic_2d import make_disk
from topological_segmentation.topology.betti import compute_betti_numbers


def test_hole_increases_h1():
    gt = make_disk(size=64, radius=20)
    with_hole = add_hole(gt, radius=6)
    assert compute_betti_numbers(with_hole) == (1, 1)


def test_split_changes_beta0():
    gt = make_disk(size=64, radius=20)
    split = split_object(gt, width=2)
    assert compute_betti_numbers(split)[0] >= 2


def test_boundary_perturbation_preserves_topology():
    gt = make_disk(size=64, radius=20)
    deformed = erode_boundary(gt, amount=2)
    assert compute_betti_numbers(deformed) == (1, 0)


def test_add_component_creates_extra_component():
    gt = make_disk(size=64, radius=20)
    extra = add_component(gt, radius=8, center=(20, 20))
    assert compute_betti_numbers(extra)[0] >= 2
