from __future__ import annotations

import numpy as np
from skimage import morphology


def add_hole(mask: np.ndarray, radius: int = 5, center=None) -> np.ndarray:
    """Insert an interior hole into a binary mask.

    This perturbation is important for topology-aware analysis because it changes
    the first Betti number by creating a loop in the foreground. Practically, it
    removes a disk-shaped interior region from the object, which should make the
    object look like a ring.

    The implementation is fully deterministic: the same mask, radius, and center
    always produce the same result.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if center is None:
        center = (mask.shape[0] // 2, mask.shape[1] // 2)

    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    hole = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2)

    result = mask.copy()
    result[hole] = 0
    return result.astype(np.uint8)


def erode_boundary(mask: np.ndarray, amount: int = 2) -> np.ndarray:
    """Shrink the object boundary with a small morphological erosion.

    This is meant to approximate a boundary deformation that changes overlap but
    should preserve the topology of a single connected object when the erosion is
    small enough. In a clean disk-like mask, a modest erosion should not create a
    hole or split the object; it should mainly change the boundary geometry.
    """
    mask = np.asarray(mask, dtype=bool)
    footprint = morphology.disk(amount)
    eroded = morphology.erosion(mask, footprint=footprint)
    return eroded.astype(np.uint8)


def split_object(mask: np.ndarray, width: int = 1, axis: str = "vertical") -> np.ndarray:
    """Split one object into two disconnected components.

    This function removes a narrow slice from the mask. The result is a topology-
    breaking perturbation relative to the one-component base object. It is used as
    a simple synthetic example of beta_0 increasing from 1 to 2.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    result = mask.copy()

    if axis == "vertical":
        center = result.shape[1] // 2
        start = max(0, center - width // 2)
        end = min(result.shape[1], start + width)
        result[:, start:end] = 0
    elif axis == "horizontal":
        center = result.shape[0] // 2
        start = max(0, center - width // 2)
        end = min(result.shape[0], start + width)
        result[start:end, :] = 0
    else:
        raise ValueError("axis must be 'vertical' or 'horizontal'")

    return result.astype(np.uint8)


def add_component(mask: np.ndarray, radius: int = 8, center=None) -> np.ndarray:
    """Add a disconnected extra component to the foreground.

    This is a topology-changing perturbation that creates an additional connected
    component. It is conceptually similar to a split object, but it does not
    sever the original object; instead, it adds a separate blob elsewhere.

    If the requested center overlaps the existing object, we move the new disk to
    a deterministic far-away location so that the result genuinely creates a
    second disconnected component rather than just thickening the original one.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if center is None:
        center = (mask.shape[0] // 2 + 20, mask.shape[1] // 2 + 20)

    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    component = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2)

    if np.any(component & (mask > 0)):
        coordinates = np.argwhere(mask > 0)
        object_center = coordinates.mean(axis=0)
        offset = max(radius + 10, 20)
        candidate = (
            int(np.clip(round(object_center[0] + 2 * offset), 0, mask.shape[0] - 1)),
            int(np.clip(round(object_center[1] + 2 * offset), 0, mask.shape[1] - 1)),
        )
        center = candidate
        component = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2)

    result = mask.copy().astype(np.uint8)
    result[component] = 1
    return result.astype(np.uint8)
