"""Region-level (cluster-mass) permutation inference for differential systems.

Per-position p-values from correlated overlapping windows cannot be trusted position-by-position, and
with few donors the asymptotic NB Wald p-values are only approximately calibrated. Cluster-mass
permutation fixes both at once, at the level we actually report -- the **region**:

1. Regions are the contiguous runs of positions passing a fixed **cluster-forming threshold**
   (a primary per-position p), each scored by its **mass** = sum of ``-log10(p)`` over its positions.
2. The labels are permuted **at the donor level** (which donors are case vs control), the model is
   refit, and the **maximum** cluster mass is recorded per permutation.
3. A region's significance is the fraction of permutations whose max mass matches or exceeds it --
   an FWER-controlled region p-value that absorbs the window correlation, the cluster-forming step,
   and the few-donor miscalibration.

Permutations are **exhaustive** when the number of distinct donor relabelings
``C(n_donors, n_case)`` is small (few-donor cohorts), else a random sample is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sctranscriptogramer.tl.clusters import identify_clusters

__all__ = ["region_permutation", "RegionPermutationResult"]


@dataclass
class RegionPermutationResult:
    """Cluster-mass permutation result."""

    regions: list[tuple[int, int]]
    mass: NDArray[np.float64]
    region_pvalue: NDArray[np.float64]
    null_max_mass: NDArray[np.float64]
    n_perm: int
    exact: bool
    cluster_forming_p: float
    params: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "start": [lo for lo, _ in self.regions],
                "end": [hi for _, hi in self.regions],
                "mass": self.mass,
                "region_pvalue": self.region_pvalue,
            }
        )


def _suprathreshold_regions(
    pval: NDArray, present: NDArray, thr: float, radius: int, n_positions: int
) -> tuple[list[tuple[int, int]], NDArray[np.float64]]:
    """Contiguous supra-threshold clusters and their masses (sum of -log10 p over supra positions)."""
    sig = present & np.isfinite(pval) & (pval < thr)
    clusters = identify_clusters(
        np.flatnonzero(sig), radius=radius, n_positions=n_positions, periodic=True
    )
    masses = []
    for lo, hi in clusters.intervals:
        seg = np.arange(lo, hi + 1)
        seg_sig = sig[seg]
        p_seg = np.clip(pval[seg][seg_sig], 1e-300, 1.0)
        masses.append(float(np.sum(-np.log10(p_seg))))
    return clusters.intervals, np.asarray(masses, dtype=np.float64)


def _donor_condition_map(condition: NDArray, donor: NDArray) -> dict:
    """Majority condition per donor (the label reassigned under permutation)."""
    mapping = {}
    for d in np.unique(donor):
        vals, counts = np.unique(condition[donor == d], return_counts=True)
        mapping[d] = vals[counts.argmax()]
    return mapping


def _condition_vectors(
    donor: NDArray, donor_map: dict, case, control, max_exact: int, n_permutations: int, rng
) -> tuple[list[NDArray], bool]:
    """Donor-level relabelings (excluding the observed one). Exhaustive if small, else random."""
    donors = np.array(list(donor_map))
    case_donors = frozenset(d for d in donors if donor_map[d] == case)
    n_case = len(case_donors)
    n_total = len(donors)
    total = comb(n_total, n_case)

    def to_vec(case_set: frozenset) -> NDArray:
        return np.where(np.isin(donor, list(case_set)), case, control)

    exact = (total - 1) <= max_exact
    vectors = []
    if exact:
        for combo in combinations(donors.tolist(), n_case):
            cs = frozenset(combo)
            if cs != case_donors:  # exclude the observed labelling
                vectors.append(to_vec(cs))
    else:
        seen = {case_donors}
        while len(vectors) < n_permutations and len(seen) < total:
            cs = frozenset(rng.choice(donors, size=n_case, replace=False).tolist())
            if cs not in seen:
                seen.add(cs)
                vectors.append(to_vec(cs))
    return vectors, exact


def region_permutation(
    pvalue_fn: Callable[[NDArray], NDArray],
    condition: NDArray,
    donor: NDArray,
    *,
    case,
    control,
    radius: int,
    present_mask: NDArray,
    n_positions: int,
    cluster_forming_p: float = 0.01,
    max_exact: int = 500,
    n_permutations: int = 200,
    random_state: int = 0,
) -> RegionPermutationResult:
    """Cluster-mass permutation test of differential-system regions.

    Parameters
    ----------
    pvalue_fn
        Callable mapping a per-observation condition vector to a length-``n_positions`` array of
        per-position p-values (NaN where untested). Each call is one model refit.
    condition, donor
        Per-observation condition labels (values in ``{case, control}``) and donor ids. ``condition``
        must be donor-level (a donor's label is taken as its majority).
    case, control
        The two condition values.
    radius, present_mask, n_positions
        Region-forming parameters (as elsewhere in the package).
    cluster_forming_p
        Primary per-position p threshold that defines candidate regions (fixed across real and null).
    max_exact
        Enumerate all donor relabelings exhaustively when ``C(n_donors, n_case) - 1 <= max_exact``.
    n_permutations
        Number of random permutations when not exhaustive.
    random_state
        Seed.

    Returns
    -------
    RegionPermutationResult
    """
    present = np.asarray(present_mask, dtype=bool)
    condition = np.asarray(condition)
    donor = np.asarray(donor)

    donor_map = _donor_condition_map(condition, donor)
    labels = set(donor_map.values())
    if not {case, control} <= labels or len(labels) < 2:
        raise ValueError("condition must contain both case and control at the donor level")
    if len(donor_map) < 2:
        raise ValueError("need >= 2 donors for permutation inference")

    regions, masses = _suprathreshold_regions(
        pvalue_fn(condition), present, cluster_forming_p, radius, n_positions
    )

    rng = np.random.default_rng(random_state)
    perm_vectors, exact = _condition_vectors(
        donor, donor_map, case, control, max_exact, n_permutations, rng
    )

    null_max = np.zeros(len(perm_vectors), dtype=np.float64)
    for i, cond_vec in enumerate(perm_vectors):
        _, m = _suprathreshold_regions(
            pvalue_fn(cond_vec), present, cluster_forming_p, radius, n_positions
        )
        null_max[i] = m.max() if m.size else 0.0

    n_perm = len(perm_vectors)
    region_p = np.array(
        [(1 + int(np.sum(null_max >= mass))) / (n_perm + 1) for mass in masses], dtype=np.float64
    )
    return RegionPermutationResult(
        regions=regions,
        mass=masses,
        region_pvalue=region_p,
        null_max_mass=null_max,
        n_perm=n_perm,
        exact=exact,
        cluster_forming_p=cluster_forming_p,
        params={"radius": radius, "max_exact": max_exact, "random_state": random_state},
    )
