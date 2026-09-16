import numpy as np

from topological_segmentation.data.perturbations import add_hole, split_object
from topological_segmentation.data.synthetic_2d import make_disk
from topological_segmentation.topology.persistence import get_persistence_diagram
from topological_segmentation.topology.pce import (
    finite_birth_persistence,
    pce_from_diagrams,
)


def _diagram(mask, dimension):
    entries = get_persistence_diagram(mask.astype(float), dimension)
    return finite_birth_persistence(entries)


def test_pce_self_comparison_has_zero_excess():
    diagram = np.array([[0.0, 1.0], [0.2, 0.5]])
    report = pce_from_diagrams(diagram, diagram)

    assert report.status == "ok"
    assert np.isclose(report.entropy_excess, 0.0)
    assert np.isclose(report.unexplained_mass, 0.0)


def test_pce_probability_difference_is_nonnegative():
    reference = np.array([[0.0, 1.0]])
    candidate = np.array([[0.5, 1.0]])
    report = pce_from_diagrams(reference, candidate)

    assert report.entropy_excess >= 0.0
    assert report.unexplained_mass >= 0.0


def test_hole_is_detected_in_reverse_h1_direction():
    gt = make_disk(size=64, radius=20)
    hole = add_hole(gt, radius=6)
    gt_diagram = _diagram(gt, dimension=1)
    hole_diagram = _diagram(hole, dimension=1)

    report = pce_from_diagrams(hole_diagram, gt_diagram)

    assert len(gt_diagram) == 0
    assert len(hole_diagram) == 1
    assert report.status == "ok"
    assert report.entropy_excess > 0.0


def test_split_is_detected_in_reverse_h0_direction():
    gt = make_disk(size=64, radius=20)
    split = split_object(gt, width=2)
    gt_diagram = _diagram(gt, dimension=0)
    split_diagram = _diagram(split, dimension=0)

    report = pce_from_diagrams(split_diagram, gt_diagram)

    assert len(gt_diagram) == 0
    assert len(split_diagram) == 1
    assert report.status == "ok"
    assert report.entropy_excess > 0.0


def test_empty_diagram_status_is_explicit():
    report = pce_from_diagrams(np.empty((0, 2)), np.empty((0, 2)))

    assert report.status == "both_empty"
    assert report.pce == 0.0
    assert report.entropy_excess == 0.0
    assert report.unexplained_mass == 0.0
