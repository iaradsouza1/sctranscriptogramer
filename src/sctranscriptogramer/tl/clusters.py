"""Identify system-level clusters from significant positions.

Ported from the R ``differentiallyExpressed`` cluster logic:

1. **Runs**: significant positions are grouped into contiguous runs; a new run starts whenever the
   gap between consecutive significant positions exceeds ``radius``. Each run spans (first, last)
   significant position.
2. **Periodic padding/merge** (optional, default on): each run is padded by ``radius`` on both sides
   (the window footprint), overlapping padded intervals are merged, and intervals that fall off
   either end of the ordering wrap around to the other end (setting ``pbc=True``, meaning the first
   and last clusters are one cluster across the boundary).

Positions falling inside any cluster interval get that cluster's label (1-based); with ``pbc`` the
last cluster is relabelled to 1 to reflect the wrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["identify_clusters", "ClusterResult"]


@dataclass
class ClusterResult:
    """Result of cluster identification.

    Attributes
    ----------
    intervals
        List of ``(start, end)`` position ranges (inclusive, within ``0..n_positions-1``).
    labels
        Per-position cluster label (length ``n_positions``); ``0`` = not in any cluster. With
        ``pbc``, the wrapped last interval shares label ``1`` with the first.
    pbc
        Whether periodic boundary wrapping merged the first and last clusters.
    n_clusters
        Number of distinct clusters (counting a wrapped pair once).
    """

    intervals: list[tuple[int, int]]
    labels: NDArray[np.int64]
    pbc: bool
    n_clusters: int


def _runs(sig: NDArray[np.int64], radius: int) -> list[list[int]]:
    """Contiguous runs of significant positions; split where the gap exceeds ``radius``."""
    runs: list[list[int]] = []
    start = sig[0]
    prev = sig[0]
    for p in sig[1:]:
        if (p - prev) > radius:
            runs.append([start, prev])
            start = p
        prev = p
    runs.append([start, prev])
    return runs


def _merge_padded(runs: list[list[int]], radius: int) -> list[list[int]]:
    """Pad each run by +/- radius and merge overlapping/touching intervals (may exceed bounds)."""
    padded = [[lo - radius, hi + radius] for lo, hi in runs]
    merged: list[list[int]] = [padded[0]]
    for lo, hi in padded[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _apply_periodic(merged: list[list[int]], lo_bound: int, hi_bound: int) -> tuple[list[list[int]], bool]:
    """Wrap intervals that run off either end, mirroring the R boundary handling."""
    pbc = False
    if merged[0][0] < lo_bound:
        pbc = True
        x = hi_bound + 1 + merged[0][0] - lo_bound
        merged[0][0] = lo_bound
        if x <= merged[-1][1]:
            merged[-1][1] = hi_bound
        else:
            merged.append([x, hi_bound])
    elif merged[-1][1] > hi_bound:
        pbc = True
        x = merged[-1][1] % hi_bound + lo_bound - 1
        merged[-1][1] = hi_bound
        if x >= merged[0][0]:
            merged[0][0] = lo_bound
        else:
            merged.insert(0, [lo_bound, x])
    # clamp any residual overhang
    for iv in merged:
        iv[0] = max(iv[0], lo_bound)
        iv[1] = min(iv[1], hi_bound)
    return merged, pbc


def identify_clusters(
    sig_positions: NDArray,
    *,
    radius: int,
    n_positions: int,
    periodic: bool = True,
) -> ClusterResult:
    """Group significant positions into system clusters.

    Parameters
    ----------
    sig_positions
        Positions (0-based) called significant.
    radius
        Window radius used to split runs and (when ``periodic``) pad clusters.
    n_positions
        Total number of positions in the ordering.
    periodic
        Apply periodic padding/merge/wrap (default). If False, clusters are the raw unpadded runs.

    Returns
    -------
    ClusterResult
    """
    labels = np.zeros(n_positions, dtype=np.int64)
    sig = np.unique(np.asarray(sig_positions, dtype=np.int64))
    if sig.size == 0:
        return ClusterResult(intervals=[], labels=labels, pbc=False, n_clusters=0)

    runs = _runs(sig, radius)

    if not periodic:
        intervals = [(int(lo), int(hi)) for lo, hi in runs]
        for i, (lo, hi) in enumerate(intervals, start=1):
            labels[lo : hi + 1] = i
        n_clusters = int(np.count_nonzero(np.unique(labels)))
        return ClusterResult(
            intervals=intervals, labels=labels, pbc=False, n_clusters=n_clusters
        )

    merged, pbc = _apply_periodic(_merge_padded(runs, radius), 0, n_positions - 1)
    intervals = [(int(lo), int(hi)) for lo, hi in merged]

    # Label positions: 1-based; with pbc the last interval joins cluster 1.
    for i, (lo, hi) in enumerate(intervals, start=1):
        label = 1 if (pbc and i == len(intervals)) else i
        labels[lo : hi + 1] = label

    # Count distinct clusters from the labels (a wrapped pair shares one label -> counted once).
    n_clusters = int(np.count_nonzero(np.unique(labels)))
    return ClusterResult(intervals=intervals, labels=labels, pbc=pbc, n_clusters=n_clusters)
