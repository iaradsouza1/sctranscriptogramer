"""Multiple-testing correction helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["benjamini_hochberg", "acat"]


def acat(pvalues: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Aggregated Cauchy Association Test (ACAT) omnibus combination of p-values.

    Combines several p-values into one along ``axis`` using the Cauchy transform. It controls the
    type-I error under the global null even when the combined p-values are dependent (so it does not
    need their covariance) -- which is why it is used here to combine the per-level Wald tests of a
    multi-level factor into a single "does any level differ" omnibus test.

    NaN inputs (untested) are ignored; a row that is entirely NaN returns NaN.

    Parameters
    ----------
    pvalues
        Array of p-values; combined along ``axis`` (default last).
    axis
        Axis to combine over.

    Returns
    -------
    numpy.ndarray
        Combined p-values with ``axis`` removed.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    # Clip away from 0/1 so tan() stays finite.
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    stat = np.tan((0.5 - p) * np.pi)  # Cauchy transform
    with np.errstate(invalid="ignore"):
        mean_stat = np.nanmean(stat, axis=axis)
    return 0.5 - np.arctan(mean_stat) / np.pi


def benjamini_hochberg(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg FDR adjustment.

    NaN p-values (e.g. positions that were not tested) are propagated as NaN and excluded from the
    correction, so the number of tests equals the number of finite p-values.

    Parameters
    ----------
    pvalues
        1D array of p-values, possibly containing NaN.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values (q-values), same shape, NaN where the input was NaN.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return out

    vals = p[finite]
    order = np.argsort(vals, kind="mergesort")
    ranked = vals[order]
    n = vals.size
    factors = n / np.arange(1, n + 1)
    adj = ranked * factors
    # enforce monotonicity from the largest p-value downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = adj
    out[finite] = result
    return out
