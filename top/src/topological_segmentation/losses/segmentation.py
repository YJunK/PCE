from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the Dice score for binary masks or probability maps.

    This function accepts input arrays containing either binary masks or soft
    probability values in the range [0, 1]. The interpretation is that values
    close to 1 indicate foreground and values close to 0 indicate background.

    Parameters
    ----------
    pred : np.ndarray
        Prediction mask or probability map.
    target : np.ndarray
        Ground-truth mask or probability map.
    eps : float
        Small numerical stabilizer to avoid division by zero.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    if pred.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    intersection = np.sum(pred * target)
    denominator = np.sum(pred) + np.sum(target)
    return (2.0 * intersection + eps) / (denominator + eps) # denominator + eps to avoid division by zero in case both pred and target are empty


def dice_loss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute 1 - Dice score.

    This is a common overlap-based loss used for segmentation tasks. In an ideal
    identity case, comparing a mask with itself yields Dice = 1 and Dice loss = 0.
    """
    return 1.0 - dice_score(pred, target, eps=eps)


def binary_cross_entropy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute binary cross entropy for probability maps.

    This implementation is intentionally written for NumPy arrays and assumes the
    input values are in [0, 1]. It is compatible with soft probability maps and
    binary masks after conversion to floats.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    pred = np.clip(pred, eps, 1.0 - eps) # Clip predictions to avoid log(0) and log(1) issues
    return -np.mean(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred)) # Compute the mean binary cross-entropy loss across all pixels
