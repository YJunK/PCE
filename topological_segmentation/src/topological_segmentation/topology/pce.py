from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCEReport:
    """Directional PCE summary for two persistence diagrams.

    The first diagram is the reference event space. ``candidate`` is used to
    explain the reference features, matching the direction in the paper.
    """

    pce: float
    entropy_excess: float
    unexplained_mass: float
    status: str


def finite_birth_persistence(entries) -> np.ndarray:
    """Convert GUDHI entries to finite ``(birth, lifetime)`` coordinates.

    Infinite death intervals are excluded because the paper defines persistence
    probabilities on finite off-diagonal diagram points. Multiplicity is kept.
    """
    points = []
    for _, (birth, death) in entries:
        if not np.isfinite(birth) or not np.isfinite(death):
            continue
        lifetime = float(death - birth)
        if lifetime > 0.0:
            points.append((float(birth), lifetime))
    return np.asarray(points, dtype=float).reshape((-1, 2))


def persistence_probability(diagram: np.ndarray) -> np.ndarray:
    """Assign probability proportional to persistence lifetime."""
    diagram = np.asarray(diagram, dtype=float).reshape((-1, 2))
    if len(diagram) == 0:
        return np.empty(0, dtype=float)
    lifetimes = diagram[:, 1]
    total = float(np.sum(lifetimes))
    if total <= 0.0:
        raise ValueError("diagram must contain positive total persistence")
    return lifetimes / total


def _persistence_weight(lifetime: np.ndarray) -> np.ndarray:
    """Use the paper's bounded persistence weighting function g(l)=l/(1+l)."""
    return lifetime / (1.0 + lifetime)


def _similarity(reference: np.ndarray, candidate: np.ndarray, sigma_birth: float, sigma_lifetime: float) -> np.ndarray:
    """Compute the anisotropic Gaussian similarity from the paper."""
    if sigma_birth <= 0.0 or sigma_lifetime <= 0.0:
        raise ValueError("similarity scales must be positive")
    birth_gap = (reference[:, None, 0] - candidate[None, :, 0]) / sigma_birth
    lifetime_gap = (reference[:, None, 1] - candidate[None, :, 1]) / sigma_lifetime
    return np.exp(-0.5 * (birth_gap**2 + lifetime_gap**2))


def pce_from_diagrams(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sigma_birth: float = 0.5,
    sigma_lifetime: float = 0.5,
    tau: float = 1.0,
) -> PCEReport:
    """Compute directional persistent cross entropy.

    ``reference`` is D_X and ``candidate`` is D_Y in the paper. The returned
    PCE is H(p_X, p_XY); ``entropy_excess`` subtracts H(p_X), and therefore
    isolates the discrepancy introduced by using the candidate diagram.

    Empty diagrams are reported explicitly because the paper assumes nonempty
    finite diagrams. Two empty diagrams are an exact zero comparison; an empty
    reference with a nonempty candidate is undefined in this direction.
    """
    reference = np.asarray(reference, dtype=float).reshape((-1, 2))
    candidate = np.asarray(candidate, dtype=float).reshape((-1, 2))
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if len(reference) == 0 and len(candidate) == 0:
        return PCEReport(0.0, 0.0, 0.0, "both_empty")
    if len(reference) == 0:
        return PCEReport(np.nan, np.nan, np.nan, "empty_reference")

    reference_probability = persistence_probability(reference)
    reference_weight = _persistence_weight(reference[:, 1])
    self_score = _similarity(reference, reference, sigma_birth, sigma_lifetime) @ reference_weight
    candidate_score = np.zeros(len(reference), dtype=float)
    if len(candidate) > 0:
        candidate_score = _similarity(reference, candidate, sigma_birth, sigma_lifetime) @ _persistence_weight(candidate[:, 1])

    difference = candidate_score - self_score
    explained_fraction = np.exp(-(difference**2) / (2.0 * tau**2))
    induced_probability = reference_probability * explained_fraction
    unexplained_mass = float(np.sum(reference_probability * (1.0 - explained_fraction)))

    pce = float(-np.sum(reference_probability * np.log(induced_probability)))
    reference_entropy = float(-np.sum(reference_probability * np.log(reference_probability)))
    entropy_excess = float(pce - reference_entropy)
    return PCEReport(pce, entropy_excess, unexplained_mass, "ok")


def pce_from_entries(reference_entries, candidate_entries, **kwargs) -> PCEReport:
    """Compute directional PCE directly from GUDHI persistence entries."""
    return pce_from_diagrams(
        finite_birth_persistence(reference_entries),
        finite_birth_persistence(candidate_entries),
        **kwargs,
    )
