"""Spatial binning for high-resolution platforms (e.g. Visium HD).

High-resolution assays sample a fine, near-single-molecule lattice: Visium HD is a 2 um grid, so a
capture area is millions of bins, each with only a handful of UMIs. Two problems follow for the
transcriptogram: (1) a per-bin profile at 2 um is almost all dropout, so its systems signal is noise;
(2) the raw object is far too large to smooth and test bin-by-bin.

The standard remedy -- and what 10x itself ships 8 um / 16 um bins for -- is to **aggregate onto a
coarser square lattice** before analysis. :func:`bin_spatial` sums counts within square bins of a
chosen edge length, producing a compact bins x genes AnnData with bin-centre coordinates that feeds
the normal pipeline (:func:`~sctranscriptogramer.pp.map_ids.project_to_ordering` ->
:func:`~sctranscriptogramer.tl.transcriptogram` -> the spatial tools). The default 16 um matches the
usual working resolution for Visium HD.

For an object whose units are already meaningful (a coarser bin object, or HD segmented into cells),
skip this and go straight to the pipeline -- the spatial tools are chunked and scale to those sizes.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree

from sctranscriptogramer.pp.metacells import _as_csr, _normalize, _sum_by_labels

__all__ = ["bin_spatial"]


def _infer_spacing(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance -- the source lattice spacing."""
    n = coords.shape[0]
    if n < 2:
        return 1.0
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)
    d = dist[:, 1]
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def bin_spatial(
    adata: AnnData,
    *,
    bin_size: float = 16.0,
    factor: int | None = None,
    coords_key: str = "spatial",
    counts_layer: str | None = None,
    carry_obs: Sequence[str] | None = None,
    normalize: bool = True,
    normalization: str = "tmm_log2cpm",
    prior_count: float = 2.0,
    target_sum: float = 1e4,
    min_counts: int = 1,
    key_added: str = "spatial",
) -> AnnData:
    """Aggregate a high-resolution spatial AnnData onto a coarser square lattice.

    Cells/bins are grouped into non-overlapping square bins of edge ``bin_size`` (in the units of
    ``obsm[coords_key]``) and their counts are summed, giving a compact bins x genes object with
    bin-centre coordinates -- ready for
    :func:`~sctranscriptogramer.pp.map_ids.project_to_ordering`. This both denoises (each super-bin
    carries many more UMIs than a native 2 um bin) and shrinks the object by ``(bin_size / source)^2``.

    Parameters
    ----------
    adata
        High-resolution spatial AnnData (raw counts in ``counts_layer`` or ``X``) with coordinates in
        ``obsm[coords_key]``.
    bin_size
        Square-bin edge length in the coordinate units. The default ``16.0`` targets 16 um for Visium
        HD **when the coordinates are in microns** -- the recommended working resolution. Ignored when
        ``factor`` is given.
    factor
        Alternative to ``bin_size``: merge ``factor`` x ``factor`` source bins, i.e. set
        ``bin_size = factor * (inferred source spacing)``. Use this when the coordinates are not in
        microns (e.g. array indices or image pixels) -- e.g. ``factor=8`` turns a native 2 um grid into
        16 um bins regardless of units.
    coords_key
        ``obsm`` key with the 2D coordinates.
    counts_layer
        Layer holding raw counts; defaults to ``adata.X``.
    carry_obs
        ``obs`` columns to carry onto bins: numeric columns are averaged, categorical columns take the
        majority value within the bin.
    normalize, normalization, prior_count, target_sum
        Embedding normalization of ``X``, exactly as in
        :func:`~sctranscriptogramer.pp.aggregate_metacells` (``layers['counts']`` always holds the raw
        summed counts for the differential path).
    min_counts
        Drop bins whose summed counts fall below this (default 1 keeps every occupied bin).
    key_added
        ``obsm`` key for the bin-centre coordinates on the output (default ``"spatial"``).

    Returns
    -------
    AnnData
        Bins x genes, with ``layers['counts']`` (summed raw counts), ``obsm[key_added]`` (bin
        centres), ``obs['n_cells']``/``obs['total_counts']``/``obs['bin_x']``/``obs['bin_y']`` (integer
        lattice indices, so exact rook/queen weights are available via k-NN), and ``uns['sctg_bins']``.
    """
    if coords_key not in adata.obsm:
        raise KeyError(f"obsm[{coords_key!r}] not found; spatial coordinates are required")
    carry_obs = list(carry_obs or [])
    missing = [k for k in carry_obs if k not in adata.obs]
    if missing:
        raise KeyError(f"columns not found in adata.obs: {missing}")

    coords = np.asarray(adata.obsm[coords_key], dtype=np.float64)[:, :2]

    if factor is not None:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        spacing = _infer_spacing(coords)
        bin_size = float(factor) * spacing
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    origin = coords.min(axis=0)
    bxy = np.floor((coords - origin) / bin_size).astype(np.int64)
    bx, by = bxy[:, 0], bxy[:, 1]
    # linear bin id -> compact labels over occupied bins only
    lin = bx * (by.max() + 1) + by
    labels, uniq = pd.factorize(lin, sort=True)
    labels = labels.astype(np.int64)
    n_bins = len(uniq)

    counts = _as_csr(adata.layers[counts_layer] if counts_layer is not None else adata.X)
    bin_counts = _sum_by_labels(counts, labels, n_bins)
    n_cells = np.asarray(
        _sum_by_labels(sp.csr_matrix(np.ones((counts.shape[0], 1))), labels, n_bins).todense()
    ).ravel()
    total_counts = np.asarray(bin_counts.sum(axis=1)).ravel()

    # lattice index of each bin (constant within a bin) via the first member cell
    first_of_bin = pd.Series(np.arange(len(labels))).groupby(labels).first().to_numpy()
    bin_bx = bx[first_of_bin]
    bin_by = by[first_of_bin]
    centres = np.column_stack(
        [origin[0] + (bin_bx + 0.5) * bin_size, origin[1] + (bin_by + 0.5) * bin_size]
    )

    obs = pd.DataFrame(index=[f"bin{i}" for i in range(n_bins)])
    for key in carry_obs:
        series = adata.obs[key]
        if pd.api.types.is_numeric_dtype(series):
            agg = pd.Series(series.to_numpy(), index=labels).groupby(level=0).mean()
            obs[key] = agg.reindex(range(n_bins)).to_numpy()
        else:
            modes = pd.Series(series.to_numpy()).groupby(labels).agg(
                lambda s: s.value_counts().index[0]
            )
            obs[key] = modes.reindex(range(n_bins)).to_numpy()
    obs["bin_x"] = bin_bx
    obs["bin_y"] = bin_by
    obs["n_cells"] = n_cells.astype(np.int64)
    obs["total_counts"] = total_counts

    keep = np.ones(n_bins, bool) if min_counts <= 1 else total_counts >= min_counts
    if not keep.all():
        warnings.warn(
            f"dropping {int((~keep).sum())} bins with < {min_counts} total counts", stacklevel=2
        )
        bin_counts = bin_counts[keep]
        obs = obs.iloc[keep].copy()
        centres = centres[keep]
        n_bins = int(keep.sum())

    X = (
        _normalize(bin_counts, normalization, prior_count=prior_count, target_sum=target_sum)
        if normalize
        else np.asarray(bin_counts.todense())
    )
    out = AnnData(X=X, obs=obs, var=adata.var.copy())
    out.layers["counts"] = np.asarray(bin_counts.todense())
    out.obsm[key_added] = centres
    out.uns["sctg_bins"] = {
        "bin_size": float(bin_size),
        "factor": int(factor) if factor is not None else None,
        "coords_key": coords_key,
        "n_bins": int(n_bins),
        "n_input_units": int(adata.n_obs),
        "median_cells_per_bin": float(np.median(n_cells)),
        "normalized": normalize,
        "normalization": normalization if normalize else None,
    }
    return out
