from topological_segmentation.data.synthetic_2d import make_disk, make_ring, make_two_disks
from topological_segmentation.topology.betti import compute_betti_numbers


def test_disk_betti_numbers():
    assert compute_betti_numbers(make_disk(size=64, radius=20)) == (1, 0)


def test_ring_betti_numbers():
    ring = make_ring(size=64, radius=20, hole_radius=8)
    assert compute_betti_numbers(ring) == (1, 1)


def test_two_disks_betti_numbers():
    disks = make_two_disks(size=64, radius=10)
    assert compute_betti_numbers(disks) == (2, 0)
