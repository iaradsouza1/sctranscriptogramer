"""Difference-of-means-along-the-ordering plot (the R ``differentiallyExpressed`` plot).

Plots ``mean(case) - mean(control)`` of the (log2-CPM) transcriptogram embedding at each position,
with an optional smoothing spline and the differential-systems clusters shaded. On the log2-CPM
scale this difference reads as an average log2 fold-change of the system, on the same axis as the
differential-systems ``logFC``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from sctranscriptogramer.pl._smooth import smooth_spline

__all__ = ["difference_of_means"]


def _params(tg: AnnData, diffsystems_key: str) -> dict:
    return tg.uns.get(diffsystems_key, {}).get("params", {})


def difference_of_means(
    tg: AnnData,
    contrast: str | None = None,
    *,
    values: str = "logFC",
    condition_key: str | None = None,
    case=None,
    control=None,
    group_key: str | None = None,
    layer: str | None = None,
    smooth: bool = False,
    spline_s: float | None = None,
    shade_clusters: bool = True,
    diffsystems_key: str = "diffsystems",
    ax: "plt.Axes | None" = None,
):
    """Plot the case-vs-control signal along the ordering, with clusters shaded.

    Parameters not given fall back to the parameters recorded by
    :func:`~sctranscriptogramer.tl.diffsystems.differential_systems`.

    Parameters
    ----------
    tg
        Transcriptogram AnnData (obs = units, var = positions), smoothed by ``tl.transcriptogram``.
    contrast
        Cell-type label; ``None`` uses the ``"all"`` contrast / all metacells.
    values
        What to plot along the ordering:

        ``"logFC"`` (default)
            The per-position model ``logFC`` from ``differential_systems`` (TMM-normalized,
            donor-adjusted) -- the **same quantity the significance shading is derived from**, so
            peaks/valleys and shaded clusters correspond. Significant positions are marked.
        ``"difference"``
            The descriptive ``mean(case) - mean(control)`` of the embedding. A **bulk-style** view:
            it averages *metacells* (which are unequal-sized donor pseudoreplicates), not donors, so
            it is **not recommended for single-cell/metacell data** -- prefer ``"logFC"``, the
            donor-aware model estimate. Also composition-sensitive unless the embedding used
            ``normalization="tmm_log2cpm"``.
    condition_key, case, control, group_key
        Override the contrast definition (``"difference"`` mode); default to the stored params.
    layer
        Embedding layer for ``"difference"`` mode; defaults to ``layers['transcriptogram']`` else ``X``.
    smooth
        Overlay a smoothing spline (off by default; the window already smooths).
    spline_s
        scipy smoothing factor for the spline.
    shade_clusters
        Shade the contrast's cluster intervals.
    diffsystems_key
        uns key holding the differential-systems results.
    ax
        Existing matplotlib Axes; a new one is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    contrast_key = str(contrast) if contrast is not None else "all"
    positions = (
        tg.var["position"].to_numpy() if "position" in tg.var else np.arange(tg.n_vars)
    )
    info = tg.uns.get(diffsystems_key, {}).get("contrasts", {}).get(contrast_key, {})

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    ax.axhline(0.0, color="0.6", lw=0.8)

    if values == "logFC":
        varm_key = info.get("varm_key")
        if not varm_key or varm_key not in tg.varm:
            raise ValueError(
                f"no differential-systems result for contrast {contrast_key!r}; run "
                "tl.differential_systems first, or pass values='difference'"
            )
        table = tg.varm[varm_key]
        y = table["logFC"].to_numpy(dtype=float)
        is_de = table["is_de"].to_numpy().astype(bool)
        ax.plot(positions, y, color="0.65", lw=0.8, label="logFC")
        if smooth:
            ax.plot(positions, smooth_spline(positions, y, spline_s), color="k", lw=1.2)
        ax.scatter(positions[is_de], y[is_de], s=7, color="crimson", zorder=3,
                   label="significant")
        ax.set_ylabel("logFC (case vs control, log2)")
    elif values == "difference":
        params = _params(tg, diffsystems_key)
        condition_key = condition_key or params.get("condition_key")
        case = case if case is not None else params.get("case")
        control = control if control is not None else params.get("control")
        group_key = group_key if group_key is not None else params.get("group_key")
        if condition_key is None or case is None or control is None:
            raise ValueError(
                "condition_key/case/control not provided and not found in the diffsystems params"
            )
        mat = (
            tg.layers["transcriptogram"]
            if (layer is None and "transcriptogram" in tg.layers)
            else (tg.layers[layer] if layer is not None else tg.X)
        )
        mat = np.asarray(mat)
        cond = tg.obs[condition_key].to_numpy()
        sel = np.ones(tg.n_obs, dtype=bool)
        if contrast is not None and group_key is not None:
            sel &= tg.obs[group_key].to_numpy() == contrast
        case_mask = sel & (cond == case)
        control_mask = sel & (cond == control)
        if case_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("no case or control metacells selected for this contrast")
        diff = mat[case_mask].mean(axis=0) - mat[control_mask].mean(axis=0)
        ax.plot(positions, diff, color="0.75", lw=0.8, label="difference of means")
        if smooth:
            ax.plot(positions, smooth_spline(positions, diff, spline_s), color="k", lw=1.4,
                    label="smoothed")
        ax.set_ylabel("Difference of means (case - control)")
    else:
        raise ValueError(f"values must be 'logFC' or 'difference', got {values!r}")

    if shade_clusters:
        cmap = plt.get_cmap("tab10")
        for i, (lo, hi) in enumerate(info.get("clusters", [])):
            ax.axvspan(lo, hi, color=cmap(i % 10), alpha=0.15)

    ax.set_xlabel("Gene position")
    title = "Differential systems" + (f" - {contrast}" if contrast is not None else "")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax
