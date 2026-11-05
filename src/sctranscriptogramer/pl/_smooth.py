"""Shared plotting helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline


def smooth_spline(x: NDArray, y: NDArray, s: float | None = None) -> NDArray[np.float64]:
    """Smoothing-spline fit of ``y`` over ``x`` (mirrors R ``smooth.spline`` for the plots).

    ``s`` is scipy's smoothing factor; if ``None`` a moderate value proportional to the series
    length and variance is used, giving a visually similar amount of smoothing to the R default.
    NaNs in ``y`` are ignored in the fit and the spline is evaluated on all ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    if finite.sum() < 4:
        return y.copy()
    if s is None:
        s = finite.sum() * float(np.nanvar(y)) * 0.5
    spline = UnivariateSpline(x[finite], y[finite], s=s)
    return spline(x)
