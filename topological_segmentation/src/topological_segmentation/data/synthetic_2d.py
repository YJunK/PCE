from __future__ import annotations

import numpy as np


def make_disk(size: int = 64, radius: int = 20, center=None) -> np.ndarray:
    """Create a filled binary disk mask.

    This is the core synthetic primitive for the toy research setup. The mask is
    represented as a 2D integer array with values in {0, 1}. We keep the object
    centered in the image by default, but allow a custom center in case a user
    wants to create shifted examples for more controlled experiments.

    Parameters
    ----------
    size : int
        Square image size. The returned mask has shape (size, size).
    radius : int
        Radius of the disk in pixels.
    center : tuple[int, int] | None
        Optional center (row, col) position. If None, the disk is centered at the
        image midpoint.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W) with uint8 values.
    """
    if center is None:
        center = (size // 2, size // 2)

    yy, xx = np.ogrid[:size, :size]
    mask = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2).astype(np.uint8)
    return mask


def make_ring(size: int = 64, radius: int = 20, hole_radius: int = 8, center=None) -> np.ndarray:
    """Create a disk with a hole.

    This is small but important: it is a binary mask that should have a single
    connected foreground component but a nontrivial first Betti number. This is
    a standard sanity-check shape for persistent homology because it should have
    one loop-like hole and one connected component.
    """
    disk = make_disk(size=size, radius=radius, center=center)
    if center is None:
        center = (size // 2, size // 2)

    yy, xx = np.ogrid[:size, :size]
    hole = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= hole_radius**2)
    ring = disk.copy()
    ring[hole] = 0
    return ring.astype(np.uint8)


def make_two_disks(size: int = 64, radius: int = 10, center=None) -> np.ndarray:
    """Create two disconnected foreground components.

    This is intentionally used to sanity-check the interpretation of beta_0 and
    the connected-component counting logic. The two disk examples are useful for
    checking that a split object is treated as multiple connected components.
    """
    if center is None:
        center = (size // 2, size // 2)

    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]

    left_center = (center[0], center[1] - 15)
    right_center = (center[0], center[1] + 15)

    left = ((xx - left_center[1]) ** 2 + (yy - left_center[0]) ** 2 <= radius**2)
    right = ((xx - right_center[1]) ** 2 + (yy - right_center[0]) ** 2 <= radius**2)

    mask[left | right] = 1
    return mask


def make_scalar_field(size: int = 64, radius: int = 20, center=None) -> np.ndarray:
    """Create a simple scalar field for future filtration experiments.

    This is not yet a full synthetic scalar field pipeline, but it provides a
    bridge from binary masks to probability-like scalar maps.
    """
    return make_disk(size=size, radius=radius, center=center).astype(float)
