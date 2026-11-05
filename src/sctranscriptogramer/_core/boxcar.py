"""Circular boxcar (uniform sliding-window) smoothing along the protein ordering.

This is the mathematical heart of the transcriptogram. The R package computes, for every
ordered protein at position ``p``, an average of expression over the window
``[p - radius, p + radius]`` with **periodic (wrap-around) boundaries** -- i.e. a circular
boxcar convolution with window width ``w = 2 * radius + 1``. The reference implementation does
this with a per-position Python-style loop; here we compute it in one vectorized pass using
cumulative sums, giving ``O(N_positions * n_units)`` instead of ``O(N_positions * w * n_units)``.

Three output modes share the same window-sum primitive:

``"sum"``
    Raw window sum. Used for the differential-expression path, where the window sum of integer
    counts is a per-position "pathway pseudo-count" fed to a negative-binomial model (with the
    unit's total counts as the library-size offset).

``"fixed"``
    Window sum divided by the full window width ``w``. This matches the R package's
    ``assignExpression`` (``sum(x) / windowSize``), which divides by ``w`` regardless of how many
    proteins are actually present in the window.

``"masked"``
    Window sum divided by the number of *present* (non-missing) proteins in the window. This is
    the dropout-aware default for single-cell data: positions with no measured expression (genes
    absent from the ordering's mapping, or dropped out) do not shrink the average of their
    neighbours. Requires a ``present_mask``; missing entries contribute 0 to the numerator.

All functions operate on an array shaped ``(n_positions, n_units)`` -- positions along axis 0 are
the axis that is smoothed; units (metacells / spots / samples) along axis 1 are independent.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Mode = Literal["masked", "fixed", "sum"]

__all__ = ["circular_boxcar", "window_sum", "window_counts"]


def _validate_radius(radius: int, n_positions: int) -> int:
    if not isinstance(radius, (int, np.integer)):
        raise TypeError(f"radius must be an integer, got {type(radius).__name__}")
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if radius >= n_positions:
        raise ValueError(
            f"radius ({radius}) must be smaller than the number of positions ({n_positions})"
        )
    return radius


def _as_2d(values: ArrayLike) -> NDArray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"values must be 2D (n_positions, n_units), got ndim={arr.ndim}")
    return arr


def window_sum(values: ArrayLike, radius: int) -> NDArray:
    """Circular window sum over the position axis (axis 0).

    For each position ``p`` returns the sum of ``values`` over ``[p - radius, p + radius]`` with
    periodic boundaries. Window width is ``w = 2 * radius + 1``.

    Parameters
    ----------
    values
        Array of shape ``(n_positions, n_units)``.
    radius
        Non-negative window radius, strictly less than ``n_positions``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``values``; integer dtype is preserved (as int64) so the ``"sum"`` mode
        keeps exact counts, otherwise the result is floating point.
    """
    arr = _as_2d(values)
    n = arr.shape[0]
    r = _validate_radius(radius, n)

    # Preserve integer counts exactly; otherwise accumulate in float64 for numerical stability.
    if np.issubdtype(arr.dtype, np.integer):
        work = arr.astype(np.int64, copy=False)
        zero_dtype = np.int64
    else:
        work = arr.astype(np.float64, copy=False)
        zero_dtype = np.float64

    if r == 0:
        return work.copy()

    w = 2 * r + 1
    # When the window spans the whole circle every position sees every element exactly once.
    if w >= n:
        total = work.sum(axis=0, keepdims=True)
        return np.repeat(total, n, axis=0)

    # Periodic extension: [last r rows][all rows][first r rows] -> length n + 2r.
    # Original position p sits at ext index p + r, and its window is ext[p : p + w].
    ext = np.concatenate([work[n - r :], work, work[:r]], axis=0)
    cs = np.cumsum(ext, axis=0)
    zero_row = np.zeros((1, arr.shape[1]), dtype=zero_dtype)
    cs = np.concatenate([zero_row, cs], axis=0)  # prepend 0 so cs[k] = sum(ext[:k])
    return cs[w : w + n] - cs[:n]


def window_counts(present_mask: ArrayLike, radius: int) -> NDArray:
    """Number of present (non-missing) proteins in each circular window.

    ``present_mask`` is a boolean / 0-1 array of shape ``(n_positions, n_units)`` (or
    ``(n_positions, 1)`` to broadcast the same mask across all units). Returns an int64 array of
    per-window occupancy counts, aligned to positions.
    """
    mask = _as_2d(present_mask)
    counts = window_sum(mask.astype(np.int64, copy=False), radius)
    return counts.astype(np.int64, copy=False)


def _boxcar_block(
    values: NDArray,
    radius: int,
    mode: Mode,
    present_mask: NDArray | None,
) -> NDArray:
    sums = window_sum(values, radius)

    if mode == "sum":
        return sums

    if mode == "fixed":
        w = 2 * radius + 1
        return sums.astype(np.float64, copy=False) / w

    if mode == "masked":
        if present_mask is None:
            mask = np.ones_like(values, dtype=np.int64)
        else:
            mask = present_mask.astype(np.int64, copy=False)
        counts = window_sum(mask, radius).astype(np.float64)
        num = sums.astype(np.float64, copy=False)
        out = np.zeros_like(num, dtype=np.float64)
        np.divide(num, counts, out=out, where=counts > 0)
        return out

    raise ValueError(f"unknown mode {mode!r}; expected one of 'masked', 'fixed', 'sum'")


def circular_boxcar(
    values: ArrayLike,
    radius: int,
    *,
    mode: Mode = "masked",
    present_mask: ArrayLike | None = None,
    chunk_size: int | None = None,
) -> NDArray:
    """Circular boxcar smoothing along the ordering (position axis).

    Parameters
    ----------
    values
        Array of shape ``(n_positions, n_units)``. For ``mode="sum"`` these are typically raw
        integer counts; for ``"fixed"``/``"masked"`` they are (normalized) expression values.
    radius
        Non-negative window radius (window width ``2 * radius + 1``), strictly ``< n_positions``.
    mode
        ``"masked"`` (default), ``"fixed"`` or ``"sum"`` -- see the module docstring.
    present_mask
        Boolean / 0-1 array marking measured positions. Shape ``(n_positions, n_units)`` for a
        per-unit mask, or ``(n_positions, 1)`` to apply one mask to every unit. Used only by
        ``mode="masked"``; in that mode missing entries are forced to 0 in the numerator so a
        non-zero value at a "missing" position never leaks into the average.
    chunk_size
        If given, process units in column blocks of this size to bound peak memory. Results are
        identical to the unchunked computation (positions are never chunked -- they are the
        reduction axis).

    Returns
    -------
    numpy.ndarray
        Smoothed array of shape ``(n_positions, n_units)``. ``"sum"`` preserves integer dtype;
        ``"fixed"`` and ``"masked"`` return float64.
    """
    arr = _as_2d(values)
    n_positions, n_units = arr.shape
    _validate_radius(radius, n_positions)

    mask_arr: NDArray | None = None
    if present_mask is not None:
        mask_arr = _as_2d(present_mask)
        if mask_arr.shape[0] != n_positions:
            raise ValueError(
                "present_mask and values must have the same number of positions "
                f"({mask_arr.shape[0]} != {n_positions})"
            )
        if mask_arr.shape[1] == 1 and n_units != 1:
            mask_arr = np.broadcast_to(mask_arr, (n_positions, n_units))
        elif mask_arr.shape[1] != n_units:
            raise ValueError(
                "present_mask must have 1 or n_units columns "
                f"({mask_arr.shape[1]} vs n_units={n_units})"
            )
        # In masked mode, missing entries must not contribute to the numerator.
        if mode == "masked":
            arr = np.where(mask_arr.astype(bool), arr, 0)

    if chunk_size is None or chunk_size >= n_units:
        return _boxcar_block(arr, radius, mode, mask_arr)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    out_dtype = np.int64 if (mode == "sum" and np.issubdtype(arr.dtype, np.integer)) else np.float64
    out = np.empty((n_positions, n_units), dtype=out_dtype)
    for start in range(0, n_units, chunk_size):
        stop = min(start + chunk_size, n_units)
        block_mask = None if mask_arr is None else mask_arr[:, start:stop]
        out[:, start:stop] = _boxcar_block(arr[:, start:stop], radius, mode, block_mask)
    return out
