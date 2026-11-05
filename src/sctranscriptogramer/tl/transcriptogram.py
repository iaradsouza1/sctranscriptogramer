"""The transcriptogram transform (the R ``transcriptogramStep2``).

Applies the circular boxcar along the ordering to a projected AnnData (from
:func:`~sctranscriptogramer.pp.map_ids.project_to_ordering`), producing:

- ``X`` -- the **embedding**: masked (dropout-aware) window mean of the expression, per unit.
- ``layers['window_counts']`` -- window **sums of raw counts** (pathway pseudo-counts) for the
  negative-binomial differential-systems path, if the projection carried a ``counts`` layer.

Both are stored on the units x positions AnnData in place.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from sctranscriptogramer._core.boxcar import circular_boxcar

__all__ = ["transcriptogram"]


def transcriptogram(
    tg: AnnData,
    radius: int = 80,
    *,
    mode: str = "masked",
    chunk_size: int | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Compute the transcriptogram (windowed smoothing) in place.

    Parameters
    ----------
    tg
        Transcriptogram AnnData from :func:`project_to_ordering` (obs = units, var = positions).
        Must carry ``uns['sctg']`` with a ``present_mask``.
    radius
        Window radius; window width is ``2 * radius + 1``. Must be ``< n_positions``. The default of
        ``80`` is a good general choice (matching the original transcriptogramer and the broad optimum
        found by :func:`~sctranscriptogramer.tl.select_radius`); use ``select_radius`` to tune it to a
        specific dataset/contrast.
    mode
        Denominator for the embedding smoothing: ``"masked"`` (default, dropout-aware) or
        ``"fixed"`` (divide by the full window width, matching the R package).
    chunk_size
        Optional unit-column chunk size to bound memory (positions are never chunked).
    copy
        If True, operate on and return a copy instead of mutating ``tg``.

    Returns
    -------
    AnnData or None
        The updated AnnData if ``copy=True``, otherwise ``None`` (``tg`` is modified in place).
    """
    if "sctg" not in tg.uns or "present_mask" not in tg.uns["sctg"]:
        raise ValueError(
            "tg does not look like a projected transcriptogram AnnData "
            "(missing uns['sctg']['present_mask']); run pp.project_to_ordering first"
        )

    out = tg.copy() if copy else tg
    n_positions = out.n_vars
    present_mask = np.asarray(out.uns["sctg"]["present_mask"], dtype=bool)
    if present_mask.shape[0] != n_positions:
        raise ValueError("present_mask length does not match number of positions")

    # boxcar operates on (n_positions, n_units); AnnData layers are (n_units, n_positions).
    expr = np.asarray(out.layers["step1"] if "step1" in out.layers else out.X, dtype=np.float64).T
    mask_col = present_mask.reshape(-1, 1)

    smoothed = circular_boxcar(
        expr, radius, mode=mode, present_mask=mask_col, chunk_size=chunk_size
    )
    out.X = smoothed.T
    out.layers["transcriptogram"] = out.X.copy()

    if out.uns["sctg"].get("has_counts") and "counts" in out.layers:
        counts = np.asarray(out.layers["counts"]).T  # (positions, units)
        window_counts = circular_boxcar(counts, radius, mode="sum", chunk_size=chunk_size)
        out.layers["window_counts"] = window_counts.T

    out.uns["sctg"] = {
        **out.uns["sctg"],
        "radius": int(radius),
        "window_size": int(2 * radius + 1),
        "mode": mode,
    }
    return out if copy else None
