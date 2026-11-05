"""Variable / divergent systems along the ordering.

Two complementary tools:

- :func:`highly_variable_systems` -- **unsupervised**: which positions vary across metacells more than
  the technical mean-variance trend predicts (the transcriptogram analog of highly variable genes).
- :func:`divergent_systems` -- **supervised/multi-group**: which positions differ across the levels of a
  factor (e.g. cell state), via an NB gamma mixed model (edgePython / NEBULA-LN) with a donor random
  effect; the ``k-1`` level Wald tests are combined into one omnibus p-value with ACAT. The factor must
  vary **within** donors (be crossed with donor), otherwise it is confounded with the random effect.

Both group the flagged/significant positions into contiguous **regions** (the unit of inference, since
positions are overlapping windows).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from sctranscriptogramer.stats.nb_mixed import fit_nb_factor
from sctranscriptogramer.tl.clusters import identify_clusters

__all__ = ["divergent_systems", "highly_variable_systems"]


def _mv_trend(mean: np.ndarray, var: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean-variance trend: median variance in mean-quantile bins, interpolated (numpy-only).

    A dependency-light stand-in for a loess fit: bin positions by their mean, take the median variance
    per bin, and interpolate the trend at each position's mean. This is the "technical" variance
    expected at a given expression level.
    """
    edges = np.unique(np.quantile(mean, np.linspace(0.0, 1.0, n_bins + 1)))
    centers, med_var = [], []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (mean >= lo) & (mean <= hi) if b == len(edges) - 2 else (mean >= lo) & (mean < hi)
        if m.any():
            centers.append(np.median(mean[m]))
            med_var.append(np.median(var[m]))
    if not centers:
        return np.full_like(mean, np.median(var))
    centers = np.asarray(centers)
    med_var = np.asarray(med_var)
    order = np.argsort(centers)
    return np.interp(mean, centers[order], med_var[order])


def highly_variable_systems(
    tg: AnnData,
    *,
    layer: str | None = "transcriptogram",
    n_top: int | None = None,
    quantile: float = 0.90,
    n_bins: int = 20,
    present_only: bool = True,
    radius: int | None = None,
    periodic: bool = True,
    key_added: str = "highly_variable_systems",
    copy: bool = False,
) -> AnnData | None:
    """Flag positions that vary across metacells beyond the mean-variance trend (unsupervised).

    Per position it computes the mean and variance across metacells, fits a mean-variance **trend**
    (the technical expectation), and takes the **biological variance** = observed - trend. The most
    biologically-variable positions are flagged and grouped into contiguous **regions**.

    Results are written to ``tg.var`` (including a scanpy-style boolean ``highly_variable`` column, so
    ``sc.pp.pca(tg, use_highly_variable=True)`` works) and ``tg.uns[key_added]``.

    Parameters
    ----------
    tg
        Transcriptogram AnnData (obs = metacells, var = positions).
    layer
        Matrix to use: ``"transcriptogram"`` (default, the smoothed embedding) or ``None`` for ``X``.
    n_top
        Flag the ``n_top`` positions with the highest biological variance. If ``None``, use
        ``quantile`` instead.
    quantile
        When ``n_top`` is ``None``, flag positions with biological variance at or above this quantile
        (default 0.90 = top 10%).
    n_bins
        Number of mean-quantile bins for the trend fit.
    present_only
        Restrict to positions with a mapped gene (``uns['sctg']['present_mask']``) when available.
    radius
        Window radius for grouping into regions; defaults to ``uns['sctg']['radius']``.
    periodic
        Periodic region padding/wrap (default on).
    key_added, copy
        uns key for the results; operate on a copy if ``copy=True``.

    Returns
    -------
    AnnData or None
        Writes ``var['hvs_mean' | 'hvs_variance' | 'hvs_trend' | 'hvs_biological_variance' |
        'highly_variable' | 'hvs_region']`` and ``uns[key_added]`` (regions, counts, params).
    """
    out = tg.copy() if copy else tg

    mat = out.X if (layer is None or layer not in out.layers) else out.layers[layer]
    X = np.asarray(mat.todense()) if sp.issparse(mat) else np.asarray(mat, dtype=np.float64)
    n_pos = out.n_vars

    mask = out.uns.get("sctg", {}).get("present_mask")
    present = (
        np.asarray(mask, dtype=bool)
        if (present_only and mask is not None)
        else np.ones(n_pos, dtype=bool)
    )
    idx = np.flatnonzero(present)
    if idx.size == 0:
        raise ValueError("no present positions to analyse")

    mean_ = X.mean(axis=0)
    var_ = X.var(axis=0)
    trend = np.full(n_pos, np.nan)
    trend[idx] = _mv_trend(mean_[idx], var_[idx], n_bins)
    bio = var_ - trend

    hv = np.zeros(n_pos, dtype=bool)
    bio_present = bio[idx]
    if n_top is not None:
        k = int(min(n_top, idx.size))
        top = idx[np.argsort(bio_present)[::-1][:k]]
        hv[top] = True
    else:
        thr = np.nanquantile(bio_present, quantile)
        hv[idx[bio_present >= thr]] = True

    if radius is None:
        radius = out.uns.get("sctg", {}).get("radius")
        if radius is None:
            raise ValueError("radius not given and not found in uns['sctg']['radius']")
    radius = int(radius)
    regions = identify_clusters(
        np.flatnonzero(hv), radius=radius, n_positions=n_pos, periodic=periodic
    )

    out.var["hvs_mean"] = mean_
    out.var["hvs_variance"] = var_
    out.var["hvs_trend"] = trend
    out.var["hvs_biological_variance"] = bio
    out.var["highly_variable"] = hv
    out.var["hvs_region"] = regions.labels

    out.uns[key_added] = {
        "n_highly_variable": int(hv.sum()),
        "n_regions": regions.n_clusters,
        "regions": regions.intervals,
        "pbc": regions.pbc,
        "params": {
            "layer": layer,
            "n_top": n_top,
            "quantile": None if n_top is not None else quantile,
            "n_bins": n_bins,
            "present_only": present_only,
            "radius": radius,
            "periodic": periodic,
        },
    }
    return out if copy else None


def divergent_systems(
    tg: AnnData,
    *,
    group_key: str,
    donor_key: str,
    covariates: list[str] | None = None,
    counts_layer: str = "window_counts",
    offset: str | np.ndarray | None = "total_counts",
    norm_method: str = "TMM",
    radius: int | None = None,
    p_value: float = 0.05,
    adjust: str = "BH",
    periodic: bool = True,
    key_added: str = "divergent_systems",
    copy: bool = False,
    **fit_kwargs,
) -> AnnData | None:
    """Find regions of the ordering that diverge across the levels of ``group_key``.

    Parameters
    ----------
    tg
        Transcriptogram AnnData with ``layers[counts_layer]`` (window pseudo-counts) and the metadata
        columns below in ``obs``.
    group_key
        The factor whose levels are compared (>= 2 levels), e.g. ``"cell_state"``. Must vary within
        donors.
    donor_key
        Column identifying donors/subjects -- the random effect (>= 2 donors).
    covariates
        Optional ``obs`` columns added as fixed effects.
    counts_layer
        Layer of integer window pseudo-counts (default ``"window_counts"``).
    offset
        Per-metacell size factors: an ``obs`` column name (default ``"total_counts"``), an array, or
        ``None`` to let edgePython normalize via ``norm_method``.
    radius
        Window radius for region splitting/padding; defaults to ``uns['sctg']['radius']``.
    p_value
        Adjusted-p threshold for calling a position divergent.
    adjust
        Multiple-testing method (``"BH"``).
    periodic
        Periodic region padding/wrap (default on).
    key_added
        Base key for the stored results.
    copy
        Operate on and return a copy instead of mutating ``tg``.
    **fit_kwargs
        Extra arguments forwarded to :func:`fit_nb_factor` (``robust``, ``cpc``, ``mincp``,
        ``n_jobs``, ``verbose``).

    Returns
    -------
    AnnData or None
        Writes per-position results to ``varm['<key>_<group_key>']`` (``omnibus_p``, ``pAdj``,
        ``max_abs_logFC``, ``tested``, ``is_divergent``, ``region``) and region/metadata to
        ``uns['<key>']``.
    """
    out = tg.copy() if copy else tg

    if counts_layer not in out.layers:
        raise KeyError(
            f"layer {counts_layer!r} not found; run tl.transcriptogram with a counts layer first"
        )
    for key in [group_key, donor_key] + list(covariates or []):
        if key not in out.obs:
            raise KeyError(f"column {key!r} not found in tg.obs")

    if radius is None:
        radius = out.uns.get("sctg", {}).get("radius")
        if radius is None:
            raise ValueError("radius not given and not found in uns['sctg']['radius']")
    radius = int(radius)

    counts_all = np.asarray(out.layers[counts_layer])
    if not np.issubdtype(counts_all.dtype, np.integer):
        rounded = np.rint(counts_all)
        if not np.allclose(rounded, counts_all):
            warnings.warn(
                f"layer {counts_layer!r} is not integer-valued; rounding for the NB model",
                stacklevel=2,
            )
        counts_all = rounded.astype(np.int64)

    # Drop metacells with a missing factor value.
    factor_full = out.obs[group_key].to_numpy()
    keep = pd.notna(factor_full)
    y = counts_all[keep].T  # positions x metacells
    factor = factor_full[keep]
    donor = out.obs[donor_key].to_numpy()[keep]
    cov_df = out.obs.loc[keep, covariates].reset_index(drop=True) if covariates else None

    if isinstance(offset, str):
        off = out.obs[offset].to_numpy()[keep].astype(float)
    elif offset is None:
        off = None
    else:
        off = np.asarray(offset)[keep].astype(float)

    result = fit_nb_factor(
        y, factor, donor, covariates=cov_df, offset=off, norm_method=norm_method,
        adjust=adjust, **fit_kwargs,
    )

    is_divergent = result.tested & (result.padj <= p_value)
    regions = identify_clusters(
        np.flatnonzero(is_divergent), radius=radius, n_positions=out.n_vars, periodic=periodic
    )

    df = pd.DataFrame(
        {
            "omnibus_p": result.omnibus_p,
            "pAdj": result.padj,
            "max_abs_logFC": result.max_abs_logFC,
            "tested": result.tested,
            "is_divergent": is_divergent,
            "region": regions.labels,
        },
        index=out.var_names,
    )
    varm_key = f"{key_added}_{group_key}"
    out.varm[varm_key] = df

    out.uns[key_added] = {
        "group_key": group_key,
        "levels": [str(lv) for lv in result.levels],
        "n_per_level": result.n_per_level,
        "n_divergent": int(is_divergent.sum()),
        "n_regions": regions.n_clusters,
        "regions": regions.intervals,
        "pbc": regions.pbc,
        "varm_key": varm_key,
        "params": {
            "donor_key": donor_key,
            "covariates": list(covariates or []),
            "counts_layer": counts_layer,
            "norm_method": norm_method,
            "radius": radius,
            "p_value": p_value,
            "adjust": adjust,
            "periodic": periodic,
        },
    }
    return out if copy else None
