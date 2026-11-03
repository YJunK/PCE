from __future__ import annotations

import matplotlib.pyplot as plt


def plot_persistence_diagram(diagram, ax=None):
    """Plot a simple persistence diagram.

    The diagram is expected to be a list of tuples of the form
    (dimension, birth, death). This helper keeps the plotting interface extremely
    light so a user can quickly inspect the persistence structure during toy
    experiments.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    xs = [birth for _, birth, _ in diagram]
    ys = [death for _, _, death in diagram]
    ax.scatter(xs, ys, s=12)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('birth')
    ax.set_ylabel('death')
    return ax.figure, ax
