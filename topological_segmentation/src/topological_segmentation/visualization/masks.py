from __future__ import annotations

import matplotlib.pyplot as plt


def plot_masks(mask, target=None, ax=None):
    """Plot one or two masks for visual inspection.

    The function returns Matplotlib figure and axes so a caller can save the image
    or modify styling later. The design is intentionally lightweight and suited
    for toy experiments rather than a full plotting API.
    """
    if ax is None:
        if target is None:
            _, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(mask, cmap='gray')
            ax.set_title('mask')
            return ax.figure, ax
        _, ax = plt.subplots(1, 2, figsize=(8, 4))

    if target is None:
        ax.imshow(mask, cmap='gray')
        ax.set_title('mask')
        return ax.figure, ax

    ax[0].imshow(mask, cmap='gray')
    ax[0].set_title('prediction')
    ax[1].imshow(target, cmap='gray')
    ax[1].set_title('target')
    return ax[0].figure, ax
