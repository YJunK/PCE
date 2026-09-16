from __future__ import annotations


def combined_loss(*args, **kwargs):
    """Placeholder for a future combined segmentation + topology loss.

    This module is reserved for later research work in which a weighted sum such
    as CE + Dice + lambda * L_topological is explored. At the moment, we avoid
    premature abstraction and keep the training-layer design out of scope.
    """
    raise NotImplementedError
