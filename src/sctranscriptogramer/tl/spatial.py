"""Spatial associations: which systems are spatially organized.

In spatial transcriptomics each unit (spot / cell / neighborhood) has a location, so each
transcriptogram position is a spatial field over the tissue. :func:`spatially_variable_systems`
scores every position by **Moran's I** (spatial autocorrelation) across units, flags the significantly
spatially-structured positions, and groups them into **regions** -- the spatial analog of
``differential_systems`` / ``divergent_systems``.

Moran's I is computed with a lightweight scipy k-nearest-neighbour spatial-weights matrix (no squidpy
dependency), with analytic p-values under the normality assumption. An existing spatial graph (e.g.
from ``squidpy.gr.spatial_neighbors``, stored in ``obsp``) can be used instead.

Both tools are **chunked over positions and sparse-aware**, so they scale to high-resolution objects
(hundreds of thousands of units) without densifying the whole transcriptogram. For a native Visium HD
2 um object, aggregate first with :func:`~sctranscriptogramer.pp.bin_spatial` (default 16 um bins) --
per-bin profiles at 2 um are too shallow to carry a systems signal. On the resulting regular lattice,
``n_neighs=4`` gives exact rook contiguity and ``n_neighs=8`` gives queen.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree
from scipy.stats import norm
from scipy.stats import t as student_t

from sctranscriptogramer.stats.multitest import benjamini_hochberg
from sctranscriptogramer.tl.clusters import identify_clusters

__all__ = [
    "spatially_variable_systems",
    "spatial_domains",
    "annotate_spatial_feature",
    "spatial_feature_association",
]


def _knn_weights(coords: np.ndarray, n_neighs: int) -> sp.csr_matrix:
    """Row-standardised, symmetric k-NN spatial weights from point coordinates."""
    n = coords.shape[0]
    k = min(n_neighs, n - 1)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # first neighbour is the point itself
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    adj = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)  # symmetrise
    return _row_standardise(adj)


def _row_standardise(adj: sp.spmatrix) -> sp.csr_matrix:
    adj = adj.tocsr().astype(np.float64)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ adj


def _weight_constants(W: sp.csr_matrix) -> tuple[float, float, float]:
    """Moran's I weight sums S0, S1, S2."""
    S0 = float(W.sum())
    Wt = W.T.tocsr()
    S1 = 0.5 * float((W + Wt).multiply(W + Wt).sum())
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    col_sum = np.asarray(W.sum(axis=0)).ravel()
    S2 = float(np.sum((row_sum + col_sum) ** 2))
    return S0, S1, S2


def _polynomial_basis(coords: np.ndarray, order: int) -> np.ndarray:
    """Design matrix for a spatial polynomial trend of the given order (2D coords)."""
    x, y = coords[:, 0], coords[:, 1]
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


def _moran_scores(
    X,
    W: sp.csr_matrix,
    present: np.ndarray,
    *,
    coords: np.ndarray | None = None,
    detrend_order: int | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Moran's I per *present* column of X (units x positions), chunked and sparse-aware.

    Only present columns are computed (others stay NaN). Positions are processed in chunks so the
    full (units x positions) matrix is never densified -- essential for high-resolution objects with
    hundreds of thousands of units. Detrending, when requested, is applied within each chunk (residual
    after a least-squares polynomial in the coordinates) so the residual matrix is never materialised
    whole.
    """
    n = X.shape[0]
    S0 = float(W.sum())
    result = np.full(X.shape[1], np.nan)
    cols = np.flatnonzero(present)
    if detrend_order is not None:
        if coords is None:
            raise ValueError("detrend requires coordinates")
        B = _polynomial_basis(coords, int(detrend_order))
        Binv = np.linalg.pinv(B)
    for start in range(0, cols.size, chunk_size):
        idx = cols[start : start + chunk_size]
        block = X[:, idx]
        block = (
            np.asarray(block.todense(), dtype=np.float64)
            if sp.issparse(block)
            else np.asarray(block, dtype=np.float64)
        )
        if detrend_order is not None:
            block = block - B @ (Binv @ block)
        Z = block - block.mean(axis=0, keepdims=True)
        WZ = W @ Z
        num = np.einsum("ij,ij->j", Z, WZ)
        den = np.einsum("ij,ij->j", Z, Z)
        with np.errstate(invalid="ignore", divide="ignore"):
            I = (n / S0) * num / den
        I[den == 0] = np.nan
        result[idx] = I
    return result


def spatially_variable_systems(
    tg: AnnData,
    *,
    coords_key: str = "spatial",
    connectivity_key: str | None = None,
    n_neighs: int = 6,
    layer: str | None = "transcriptogram",
    present_only: bool = True,
    detrend: int | None = None,
    alternative: str = "greater",
    p_value: float = 0.05,
    adjust: str = "BH",
    min_moran: float | None = None,
    top_quantile: float | None = None,
    radius: int | None = None,
    periodic: bool = True,
    chunk_size: int = 2048,
    key_added: str = "spatially_variable_systems",
    copy: bool = False,
) -> AnnData | None:
    """Score each position by spatial autocorrelation (Moran's I) and find spatial regions.

    Parameters
    ----------
    tg
        Transcriptogram AnnData (obs = spatial units, var = positions) with unit coordinates in
        ``obsm[coords_key]`` (carried through from the spatial AnnData by ``project_to_ordering``).
    coords_key
        Key in ``obsm`` holding the 2D/3D coordinates (default ``"spatial"``).
    connectivity_key
        Optional key in ``obsp`` holding a precomputed spatial graph (e.g. squidpy's
        ``"spatial_connectivities"``) to use instead of building a k-NN graph.
    n_neighs
        Number of spatial neighbours for the k-NN weights (when ``connectivity_key`` is ``None``). On
        a regular lattice (e.g. binned Visium HD) k-NN reduces to exact lattice contiguity: ``4`` gives
        rook neighbours, ``8`` gives queen.
    layer
        Transcriptogram layer to score (default ``"transcriptogram"``; ``None`` = ``X``).
    present_only
        Restrict to positions with a mapped gene (``uns['sctg']['present_mask']``).
    detrend
        If set (``1`` linear, ``2`` quadratic), regress each system's spatial field on a polynomial of
        the coordinates and compute Moran's I on the **residual** -- so significance reflects
        *system-specific* spatial structure (domains) rather than broad tissue-wide gradients (under
        which almost every system is trivially autocorrelated).
    alternative
        ``"greater"`` (default) tests for positive spatial autocorrelation (clustered systems);
        ``"two-sided"`` also flags dispersed/checkerboard patterns.
    p_value
        Adjusted-p threshold for calling a position spatially variable. Note: in structured tissue
        almost every system is significantly autocorrelated, so significance alone is rarely
        discriminating -- combine it with ``min_moran`` or ``top_quantile`` to find the *most*
        spatially organized systems.
    adjust
        Multiple-testing method (``"BH"``).
    min_moran
        If given, additionally require Moran's I at or above this value (effect-size threshold).
    top_quantile
        If given (e.g. ``0.90``), additionally keep only positions whose Moran's I is at or above this
        quantile among present positions -- the top spatially-structured systems.
    radius
        Window radius for region grouping; defaults to ``uns['sctg']['radius']``.
    periodic
        Periodic region padding/wrap.
    chunk_size
        Number of positions scored per block. Bounds peak memory to ``n_units x chunk_size`` so the
        function scales to high-resolution objects (hundreds of thousands of units) without densifying
        the whole transcriptogram.
    key_added, copy
        uns key for the results; operate on a copy if ``copy=True``.

    Returns
    -------
    AnnData or None
        Writes ``varm['<key>']`` (``moran_I``, ``pval``, ``pAdj``, ``is_spatial``, ``region``) and
        ``uns[key_added]`` (regions, counts, params, Moran's null moments).
    """
    out = tg.copy() if copy else tg

    # spatial weights
    if connectivity_key is not None:
        if connectivity_key not in out.obsp:
            raise KeyError(f"obsp[{connectivity_key!r}] not found")
        W = _row_standardise(out.obsp[connectivity_key])
    else:
        if coords_key not in out.obsm:
            raise KeyError(
                f"obsm[{coords_key!r}] not found; spatial coordinates are required "
                "(carried through project_to_ordering from the spatial AnnData)"
            )
        W = _knn_weights(np.asarray(out.obsm[coords_key], dtype=np.float64), n_neighs)

    n = out.n_obs
    mat = out.X if (layer is None or layer not in out.layers) else out.layers[layer]

    mask = out.uns.get("sctg", {}).get("present_mask")
    present = (
        np.asarray(mask, bool) if (present_only and mask is not None) else np.ones(out.n_vars, bool)
    )

    # Optionally remove a smooth global spatial trend so Moran's I measures system-specific
    # structure (real domains) rather than broad, tissue-wide gradients. The scoring is chunked over
    # positions and keeps the transcriptogram sparse, so it scales to high-resolution objects.
    coords = None
    if detrend is not None:
        if coords_key not in out.obsm:
            raise KeyError(f"detrend needs coordinates in obsm[{coords_key!r}]")
        coords = np.asarray(out.obsm[coords_key], dtype=np.float64)

    moran = _moran_scores(
        mat, W, present, coords=coords, detrend_order=detrend, chunk_size=chunk_size
    )

    # analytic inference under normality (constant variance across features)
    S0, S1, S2 = _weight_constants(W)
    EI = -1.0 / (n - 1)
    varI = (n * n * S1 - n * S2 + 3 * S0 * S0) / (S0 * S0 * (n * n - 1)) - EI * EI
    varI = max(varI, 1e-12)
    z = (moran - EI) / np.sqrt(varI)
    if alternative == "greater":
        pval = norm.sf(z)
    elif alternative == "two-sided":
        pval = 2.0 * norm.sf(np.abs(z))
    else:
        raise ValueError(f"alternative must be 'greater' or 'two-sided', got {alternative!r}")
    pval[~present] = np.nan

    if adjust.upper() != "BH":
        raise ValueError(f"unsupported adjust method {adjust!r}; only 'BH' is implemented")
    padj = benjamini_hochberg(pval)
    is_spatial = present & (padj <= p_value)
    if min_moran is not None:
        is_spatial &= moran >= min_moran
    if top_quantile is not None:
        thr = np.nanquantile(moran[present], top_quantile)
        is_spatial &= moran >= thr

    if radius is None:
        radius = out.uns.get("sctg", {}).get("radius")
        if radius is None:
            raise ValueError("radius not given and not found in uns['sctg']['radius']")
    radius = int(radius)
    regions = identify_clusters(
        np.flatnonzero(is_spatial), radius=radius, n_positions=out.n_vars, periodic=periodic
    )

    out.varm[key_added] = pd.DataFrame(
        {
            "moran_I": moran,
            "pval": pval,
            "pAdj": padj,
            "is_spatial": is_spatial,
            "region": regions.labels,
        },
        index=out.var_names,
    )
    out.uns[key_added] = {
        "n_spatial": int(is_spatial.sum()),
        "n_regions": regions.n_clusters,
        "regions": regions.intervals,
        "pbc": regions.pbc,
        "moran_expected": float(EI),
        "moran_var": float(varI),
        "params": {
            "coords_key": coords_key,
            "connectivity_key": connectivity_key,
            "n_neighs": n_neighs,
            "layer": layer,
            "detrend": detrend,
            "alternative": alternative,
            "p_value": p_value,
            "adjust": adjust,
            "min_moran": min_moran,
            "top_quantile": top_quantile,
            "radius": radius,
            "periodic": periodic,
            "chunk_size": chunk_size,
        },
    }
    return out if copy else None


def _domain_signature(
    mat, labels: np.ndarray, domains: list, present: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-domain mean profile (all positions) and between/within-domain F (present positions).

    Sparse-aware and allocation-light: the per-domain means come from a single indicator matmul, so
    the full (units x positions) matrix is never densified -- the F score is derived from column sums
    via the ANOVA identity ``SSW = SST - SSB``. Ranks the domain-defining systems.
    """
    n = mat.shape[0]
    code = {d: i for i, d in enumerate(domains)}
    codes = np.array([code[x] for x in labels], dtype=np.int64)
    k = len(domains)
    ind = sp.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(k, n))
    counts = np.asarray(ind.sum(axis=1)).ravel()
    safe = np.where(counts == 0, 1.0, counts)
    group_sum = ind @ mat  # k x n_vars
    group_sum = np.asarray(group_sum.todense()) if sp.issparse(group_sum) else np.asarray(group_sum)
    group_mean = group_sum / safe[:, None]
    meanmat = group_mean.T  # n_vars x k

    sub = mat[:, present]
    gm = group_mean[:, present]
    if sp.issparse(sub):
        tot = np.asarray(sub.sum(axis=0)).ravel()
        sq = np.asarray(sub.multiply(sub).sum(axis=0)).ravel()
    else:
        sub = np.asarray(sub, dtype=np.float64)
        tot = sub.sum(axis=0)
        sq = (sub * sub).sum(axis=0)
    overall = tot / n
    sst = sq - n * overall**2
    ssb = (counts[:, None] * (gm - overall[None, :]) ** 2).sum(axis=0)
    ssw = np.maximum(sst - ssb, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / max(k - 1, 1)) / (ssw / max(n - k, 1))
    return meanmat, F


def spatial_domains(
    tg: AnnData,
    *,
    layer: str | None = "transcriptogram",
    use_spatial_systems: bool = True,
    coords_key: str = "spatial",
    smooth: bool = True,
    n_neighs_smooth: int = 6,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 0,
    key_added: str = "spatial_domain",
    copy: bool = False,
) -> AnnData | None:
    """Cluster spots into spatial domains on the transcriptogram and score domain-defining systems.

    Spots are clustered in **systems space** (their transcriptogram), optionally restricted to the
    spatially-variable systems and spatially smoothed so the domains are spatially coherent. Each
    domain then has an interpretable **system signature**: its mean transcriptogram, plus a
    per-position between-domain variance score naming the systems that separate the domains.

    Parameters
    ----------
    tg
        Transcriptogram AnnData (obs = spots, var = positions) with ``obsm[coords_key]``.
    layer
        Transcriptogram layer to cluster on (default ``"transcriptogram"``; ``None`` = ``X``).
    use_spatial_systems
        If a ``spatially_variable_systems`` result is present, cluster on those positions only.
    coords_key
        ``obsm`` key with the spot coordinates.
    smooth
        Blend each spot's profile with its spatial neighbours before clustering (promotes coherent
        domains).
    n_neighs_smooth
        Spatial neighbours used for the smoothing blend.
    n_pcs, n_neighbors, resolution, random_state
        Standard scanpy PCA / neighbours / Leiden parameters.
    key_added, copy
        obs/uns key for the domains; operate on a copy if ``copy=True``.

    Returns
    -------
    AnnData or None
        Writes ``obs[key_added]`` (domain labels), ``varm['<key>_mean']`` (positions x domains mean
        profile), ``var['<key>_score']`` (domain-defining score) and ``uns[key_added]``.
    """
    import scanpy as sc

    out = tg.copy() if copy else tg
    if coords_key not in out.obsm:
        raise KeyError(f"obsm[{coords_key!r}] not found; spatial coordinates are required")

    mat = out.X if (layer is None or layer not in out.layers) else out.layers[layer]

    present = np.asarray(
        out.uns.get("sctg", {}).get("present_mask", np.ones(out.n_vars, bool)), dtype=bool
    )
    svs = out.varm.get("spatially_variable_systems")
    if use_spatial_systems and svs is not None:
        feats = present & svs["is_spatial"].to_numpy().astype(bool)
        if feats.sum() < 2:
            feats = present
    else:
        feats = present

    # Densify only the clustering features (the spatially-variable subset is small); the full
    # transcriptogram stays sparse for the signature/score below.
    sub = mat[:, feats]
    X = np.asarray(sub.todense()) if sp.issparse(sub) else np.asarray(sub, dtype=np.float64)
    coords = np.asarray(out.obsm[coords_key], dtype=np.float64)
    if smooth:
        W = _knn_weights(coords, n_neighs_smooth)
        X = 0.5 * X + 0.5 * np.asarray(W @ X)

    dom = AnnData(X=X.astype(np.float32))
    sc.pp.scale(dom, max_value=10)
    sc.pp.pca(dom, n_comps=min(n_pcs, dom.n_obs - 1, dom.n_vars - 1))
    sc.pp.neighbors(dom, n_neighbors=min(n_neighbors, dom.n_obs - 1))
    sc.tl.leiden(dom, flavor="igraph", n_iterations=2, resolution=resolution,
                 random_state=random_state)
    labels = dom.obs["leiden"].to_numpy()
    out.obs[key_added] = pd.Categorical(labels)

    domains = sorted(set(labels), key=int)
    meanmat, f_present = _domain_signature(mat, labels, domains, present)
    out.varm[f"{key_added}_mean"] = pd.DataFrame(
        meanmat, index=out.var_names, columns=[f"domain_{d}" for d in domains]
    )
    score = np.full(out.n_vars, np.nan)
    score[present] = f_present
    out.var[f"{key_added}_score"] = score

    out.uns[key_added] = {
        "n_domains": len(domains),
        "sizes": {str(d): int(np.sum(labels == d)) for d in domains},
        "params": {
            "layer": layer,
            "use_spatial_systems": bool(use_spatial_systems and svs is not None),
            "smooth": smooth,
            "n_neighs_smooth": n_neighs_smooth,
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
        },
    }
    return out if copy else None


def annotate_spatial_feature(
    tg: AnnData,
    feature_coords: np.ndarray,
    feature_values: np.ndarray | None = None,
    *,
    coords_key: str = "spatial",
    method: str = "radius",
    radius: float | None = None,
    agg: str = "max",
    fill: float = 0.0,
    key_added: str = "spatial_feature",
    copy: bool = False,
) -> AnnData | None:
    """Map an externally annotated spatial feature onto the transcriptogram units.

    The feature (e.g. amyloid-plaque locations) is given as its own set of coordinates -- typically the
    grid points where a 0/1 mask is 1. This assigns a per-unit feature value to every spot/bin so it can
    be tested with :func:`spatial_feature_association`.

    Parameters
    ----------
    tg
        Transcriptogram AnnData with unit coordinates in ``obsm[coords_key]``.
    feature_coords
        ``(m, 2)`` coordinates of the feature annotation, in the **same coordinate system** as
        ``obsm[coords_key]``.
    feature_values
        Optional per-point values aligned to ``feature_coords`` (e.g. intensities). If ``None``, every
        supplied point counts as ``1`` (a pure presence mask).
    method
        ``"radius"`` (default): aggregate the feature points within ``radius`` of each unit -- the right
        choice for a presence mask (a unit is positive if any feature point is near it). ``"nearest"``:
        take the value of the single nearest feature point (for a dense/continuous field).
    radius
        Neighbourhood radius for ``method="radius"``; defaults to the median nearest-neighbour distance
        between units (roughly one unit spacing).
    agg
        Aggregation of the feature values within the radius: ``"max"`` (presence, default), ``"mean"``
        (local density/intensity) or ``"sum"``.
    fill
        Value for units with no feature point in the radius (default ``0.0``).
    key_added
        ``obs`` column to write the per-unit feature to.
    copy
        Operate on a copy if ``True``.

    Returns
    -------
    AnnData or None
        Writes ``obs[key_added]`` (per-unit feature value).
    """
    out = tg.copy() if copy else tg
    if coords_key not in out.obsm:
        raise KeyError(f"obsm[{coords_key!r}] not found; spatial coordinates are required")
    units = np.asarray(out.obsm[coords_key], dtype=np.float64)[:, :2]
    fcoords = np.asarray(feature_coords, dtype=np.float64)[:, :2]
    if fcoords.shape[0] == 0:
        raise ValueError("feature_coords is empty")
    fvals = (
        np.ones(fcoords.shape[0]) if feature_values is None
        else np.asarray(feature_values, dtype=np.float64).ravel()
    )
    if fvals.shape[0] != fcoords.shape[0]:
        raise ValueError("feature_values must align with feature_coords")

    tree = cKDTree(fcoords)
    if method == "nearest":
        _, idx = tree.query(units, k=1)
        vals = fvals[idx]
    elif method == "radius":
        if radius is None:
            utree = cKDTree(units)
            dist, _ = utree.query(units, k=min(2, units.shape[0]))
            nn = dist[:, 1] if dist.ndim == 2 and dist.shape[1] > 1 else np.array([1.0])
            nn = nn[nn > 0]
            radius = float(np.median(nn)) if nn.size else 1.0
        neighbours = tree.query_ball_point(units, r=radius)
        reducer = {"max": np.max, "mean": np.mean, "sum": np.sum}.get(agg)
        if reducer is None:
            raise ValueError(f"agg must be 'max', 'mean' or 'sum', got {agg!r}")
        vals = np.full(units.shape[0], float(fill))
        for i, nb in enumerate(neighbours):
            if nb:
                vals[i] = reducer(fvals[nb])
    else:
        raise ValueError(f"method must be 'radius' or 'nearest', got {method!r}")

    out.obs[key_added] = vals
    return out if copy else None


def _assoc_corr(mat, present: np.ndarray, f: np.ndarray, chunk_size: int):
    """Per-position correlation and regression effect of each system field on the feature ``f``.

    ``corr`` is the (point-biserial) correlation; ``effect`` is the OLS slope of the system on ``f``,
    which for a 0/1 feature equals mean(feature=1) - mean(feature=0). Chunked and sparse-aware.
    """
    fc = f - f.mean()
    fss = float(fc @ fc)
    if fss <= 0:
        raise ValueError("feature is constant; no association can be computed")
    corr = np.full(mat.shape[1], np.nan)
    effect = np.full(mat.shape[1], np.nan)
    cols = np.flatnonzero(present)
    for s in range(0, cols.size, chunk_size):
        idx = cols[s : s + chunk_size]
        block = mat[:, idx]
        block = (
            np.asarray(block.todense(), dtype=np.float64)
            if sp.issparse(block)
            else np.asarray(block, dtype=np.float64)
        )
        xc = block - block.mean(axis=0, keepdims=True)
        num = fc @ xc
        xnorm = np.sqrt(np.einsum("ij,ij->j", xc, xc))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr[idx] = num / (np.sqrt(fss) * xnorm)
            effect[idx] = num / fss
        corr[idx[xnorm == 0]] = np.nan
    return corr, effect


def _detect_lattice(out: AnnData, grid_key, coords_key: str):
    """Integer lattice indices (gx, gy) per unit and whether the geometry is a regular lattice.

    Uses ``grid_key``/``bin_x``,``bin_y`` when present (always treated as a lattice); otherwise snaps
    the coordinates to a grid at the median-spacing and checks regularity -- that cells are (mostly)
    uniquely occupied and units sit close to their snapped centres. This lets Visium (hex) and binned
    HD data use the well-calibrated toroidal null automatically, while flagging genuinely irregular
    point patterns (e.g. imaging-based single cells) where a toroidal shift is not appropriate.

    Returns ``(gx, gy, is_regular)``; ``(None, None, False)`` if no coordinates are available.
    """
    if grid_key is None and {"bin_x", "bin_y"} <= set(out.obs.columns):
        grid_key = ("bin_x", "bin_y")
    if grid_key is not None:
        gx = out.obs[grid_key[0]].to_numpy()
        gy = out.obs[grid_key[1]].to_numpy()
        gx = (gx - gx.min()).astype(np.int64)
        gy = (gy - gy.min()).astype(np.int64)
        return gx, gy, True
    if coords_key not in out.obsm:
        return None, None, False
    coords = np.asarray(out.obsm[coords_key], dtype=np.float64)[:, :2]
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=min(2, coords.shape[0]))
    nn = dist[:, 1] if dist.ndim == 2 and dist.shape[1] > 1 else np.array([1.0])
    nn = nn[nn > 0]
    spacing = float(np.median(nn)) if nn.size else 1.0
    origin = coords.min(axis=0)
    gx = np.rint((coords[:, 0] - origin[0]) / spacing).astype(np.int64)
    gy = np.rint((coords[:, 1] - origin[1]) / spacing).astype(np.int64)
    lin = gx * (int(gy.max()) + 1) + gy
    unique_frac = np.unique(lin).size / lin.size
    resid = np.hypot(
        coords[:, 0] - (origin[0] + gx * spacing), coords[:, 1] - (origin[1] + gy * spacing)
    )
    is_regular = bool(unique_frac > 0.95 and np.median(resid) < 0.25 * spacing)
    return gx, gy, is_regular


def _toroidal_fields(f: np.ndarray, gx, gy, n_perm: int, rng) -> np.ndarray:
    """Feature fields under random toroidal (cyclic) shifts of the lattice -- preserves the feature's
    own spatial autocorrelation while breaking its alignment with any system."""
    Gx, Gy = int(gx.max()) + 1, int(gy.max()) + 1
    grid = np.full((Gx, Gy), f.mean())
    grid[gx, gy] = f
    max_shifts = Gx * Gy - 1
    B = min(n_perm, max_shifts)
    fields = np.empty((f.size, B))
    seen: set = set()
    count = 0
    guard = 0
    while count < B and guard < 50 * B + 100:
        guard += 1
        sx = int(rng.integers(0, Gx))
        sy = int(rng.integers(0, Gy))
        if (sx, sy) == (0, 0) or (sx, sy) in seen:
            continue
        seen.add((sx, sy))
        rolled = np.roll(np.roll(grid, sx, axis=0), sy, axis=1)
        fields[:, count] = rolled[gx, gy]
        count += 1
    return fields[:, :count]


def _toroidal_pval(mat, present, fields, corr_obs, alternative, chunk_size):
    """Two-sided/one-sided permutation p per position from toroidally-shifted feature fields."""
    Fc = fields - fields.mean(axis=0, keepdims=True)
    Fnorm = np.sqrt(np.einsum("ij,ij->j", Fc, Fc))
    Fnorm[Fnorm == 0] = np.nan
    B = fields.shape[1]
    exceed = np.zeros(mat.shape[1])
    cols = np.flatnonzero(present)
    for s in range(0, cols.size, chunk_size):
        idx = cols[s : s + chunk_size]
        block = mat[:, idx]
        block = (
            np.asarray(block.todense(), dtype=np.float64)
            if sp.issparse(block)
            else np.asarray(block, dtype=np.float64)
        )
        xc = block - block.mean(axis=0, keepdims=True)
        xnorm = np.sqrt(np.einsum("ij,ij->j", xc, xc))
        xnorm[xnorm == 0] = np.nan
        rn = (xc.T @ Fc) / (xnorm[:, None] * Fnorm[None, :])  # (chunk, B)
        ro = corr_obs[idx][:, None]
        if alternative == "two-sided":
            ex = np.nansum(np.abs(rn) >= np.abs(ro), axis=1)
        elif alternative == "greater":
            ex = np.nansum(rn >= ro, axis=1)
        else:
            ex = np.nansum(rn <= ro, axis=1)
        exceed[idx] = ex
    pval = np.full(mat.shape[1], np.nan)
    pval[cols] = (1.0 + exceed[cols]) / (B + 1.0)
    return pval, B


def spatial_feature_association(
    tg: AnnData,
    feature,
    *,
    coords_key: str = "spatial",
    connectivity_key: str | None = None,
    n_neighs: int = 6,
    layer: str | None = "transcriptogram",
    present_only: bool = True,
    method: str = "auto",
    grid_key: tuple[str, str] | None = None,
    n_perm: int = 1000,
    alternative: str = "two-sided",
    p_value: float = 0.05,
    adjust: str = "BH",
    min_abs_corr: float | None = None,
    radius: int | None = None,
    periodic: bool = True,
    chunk_size: int = 2048,
    random_state: int = 0,
    key_added: str = "spatial_feature_association",
    copy: bool = False,
) -> AnnData | None:
    """Test which systems are spatially associated with an annotated feature (e.g. plaques).

    For each transcriptogram position (a system's spatial field over the tissue) this measures how
    strongly the system co-localizes with ``feature`` and gives a p-value that **accounts for spatial
    autocorrelation** -- without which the test is badly anticonservative, since both the system field
    and the feature are spatially smooth. Contiguous associated positions are grouped into regions, ready
    for :func:`~sctranscriptogramer.tl.cluster_enrichment`.

    Parameters
    ----------
    tg
        Transcriptogram AnnData (obs = spatial units, var = positions).
    feature
        The per-unit feature: an ``obs`` column name, or an array of length ``n_obs``. A 0/1 mask (see
        :func:`annotate_spatial_feature` to build one from external coordinates) or a continuous field.
    coords_key, connectivity_key, n_neighs
        Spatial-weights specification (only used by ``method="moran_corrected"``), as in
        :func:`spatially_variable_systems`.
    layer
        Transcriptogram layer to test (default ``"transcriptogram"``; ``None`` = ``X``).
    present_only
        Restrict to positions with a mapped gene.
    method
        How the null accounts for autocorrelation. This matters enormously: on signal-free but
        autocorrelated systems a naive correlation test gives ~70% false positives at 0.05, so the
        correction is not optional.

        - ``"auto"`` (default): ``"toroidal"`` when the geometry is a regular lattice (``grid_key``,
          ``bin_x``/``bin_y`` from :func:`~sctranscriptogramer.pp.bin_spatial`, or coordinates detected
          to lie on a grid -- including Visium's hex grid and binned HD); otherwise
          ``"moran_corrected"`` with a warning.
        - ``"toroidal"``: permutation null from random cyclic shifts of the feature over the lattice --
          preserves the feature's own autocorrelation and breaks only its alignment with each system.
          **Well-calibrated** (empirically ~nominal false-positive rate) and recommended whenever a
          lattice exists.
        - ``"moran_corrected"``: analytic test that shrinks the effective sample size by the
          autocorrelation (Moran's I) of both fields (first-order Clifford--Richardson). Fast and works
          on any geometry, but only a **partial** correction -- it under-corrects strong/long-range
          autocorrelation, so its p-values are optimistic. Use it for ranking (with ``|corr|`` effect
          size) rather than as a strict significance filter when no lattice is available.
    grid_key
        ``(row_col, col_col)`` obs columns giving integer lattice indices for ``"toroidal"``.
    n_perm
        Number of toroidal shifts (capped at the number of distinct non-trivial shifts).
    alternative
        ``"two-sided"`` (default), ``"greater"`` (systems up where the feature is), or ``"less"``.
    p_value, adjust
        Adjusted-p threshold and multiple-testing method (``"BH"``).
    min_abs_corr
        If given, additionally require ``|corr|`` at or above this effect-size threshold.
    radius, periodic
        Region grouping (defaults to ``uns['sctg']['radius']``).
    chunk_size
        Positions scored per block (bounds memory for high-resolution objects).
    random_state
        Seed for the toroidal shifts.
    key_added, copy
        Output key; operate on a copy if ``True``.

    Returns
    -------
    AnnData or None
        Writes ``varm['<key>']`` (``corr``, ``effect``, ``pval``, ``pAdj``, ``is_associated``,
        ``region``) and ``uns[key_added]`` (regions, counts, method, params). ``effect`` is the OLS slope
        of the system on the feature -- for a 0/1 feature, mean(feature=1) - mean(feature=0).
    """
    out = tg.copy() if copy else tg

    if isinstance(feature, str):
        if feature not in out.obs:
            raise KeyError(f"feature column {feature!r} not in tg.obs")
        f = out.obs[feature].to_numpy(dtype=np.float64)
    else:
        f = np.asarray(feature, dtype=np.float64).ravel()
        if f.shape[0] != out.n_obs:
            raise ValueError(f"feature length {f.shape[0]} != n_obs {out.n_obs}")
    if not np.all(np.isfinite(f)):
        raise ValueError("feature contains non-finite values")

    mat = out.X if (layer is None or layer not in out.layers) else out.layers[layer]
    mask = out.uns.get("sctg", {}).get("present_mask")
    present = (
        np.asarray(mask, bool) if (present_only and mask is not None) else np.ones(out.n_vars, bool)
    )

    corr, effect = _assoc_corr(mat, present, f, chunk_size)

    gx, gy, is_regular = _detect_lattice(out, grid_key, coords_key)
    if method == "auto":
        method_used = "toroidal" if is_regular else "moran_corrected"
        if not is_regular:
            warnings.warn(
                "no regular lattice detected; falling back to method='moran_corrected', whose "
                "analytic p-values under-correct strong spatial autocorrelation (they are "
                "optimistic). Prefer effect size (|corr|) for ranking, or supply a lattice "
                "(grid_key / bin_x,bin_y from pp.bin_spatial) to use the calibrated toroidal null.",
                stacklevel=2,
            )
    else:
        method_used = method

    n = out.n_obs
    n_perm_used = None
    if method_used == "toroidal":
        if gx is None:
            raise ValueError("method='toroidal' needs coordinates or a grid_key")
        if not is_regular:
            warnings.warn(
                "coordinates do not form a regular lattice; the toroidal shift snaps them to a grid "
                "and may collide units -- check calibration or use method='moran_corrected'",
                stacklevel=2,
            )
        rng = np.random.default_rng(random_state)
        fields = _toroidal_fields(f, gx, gy, n_perm, rng)
        if fields.shape[1] < 1:
            raise ValueError("could not generate toroidal shifts; lattice is degenerate")
        pval, n_perm_used = _toroidal_pval(mat, present, fields, corr, alternative, chunk_size)
    elif method_used == "moran_corrected":
        if connectivity_key is not None:
            if connectivity_key not in out.obsp:
                raise KeyError(f"obsp[{connectivity_key!r}] not found")
            W = _row_standardise(out.obsp[connectivity_key])
        else:
            if coords_key not in out.obsm:
                raise KeyError(f"obsm[{coords_key!r}] not found; spatial coordinates are required")
            W = _knn_weights(np.asarray(out.obsm[coords_key], dtype=np.float64), n_neighs)
        Ix = _moran_scores(mat, W, present, chunk_size=chunk_size)
        If = float(_moran_scores(f.reshape(-1, 1), W, np.array([True]))[0])
        prod = np.clip(Ix * If, -0.999, 0.999)
        n_eff = np.clip(n * (1.0 - prod) / (1.0 + prod), 4.0, float(n))
        with np.errstate(invalid="ignore", divide="ignore"):
            tstat = corr * np.sqrt((n_eff - 2.0) / (1.0 - corr**2))
        df = n_eff - 2.0
        if alternative == "two-sided":
            pval = 2.0 * student_t.sf(np.abs(tstat), df)
        elif alternative == "greater":
            pval = student_t.sf(tstat, df)
        elif alternative == "less":
            pval = student_t.cdf(tstat, df)
        else:
            raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
        pval = np.where(present, pval, np.nan)
    else:
        raise ValueError(
            f"method must be 'auto', 'toroidal' or 'moran_corrected', got {method!r}"
        )

    if adjust.upper() != "BH":
        raise ValueError(f"unsupported adjust method {adjust!r}; only 'BH' is implemented")
    padj = benjamini_hochberg(pval)
    is_assoc = present & (padj <= p_value)
    if min_abs_corr is not None:
        is_assoc &= np.abs(corr) >= min_abs_corr

    if radius is None:
        radius = out.uns.get("sctg", {}).get("radius")
        if radius is None:
            raise ValueError("radius not given and not found in uns['sctg']['radius']")
    radius = int(radius)
    regions = identify_clusters(
        np.flatnonzero(is_assoc), radius=radius, n_positions=out.n_vars, periodic=periodic
    )

    out.varm[key_added] = pd.DataFrame(
        {
            "corr": corr,
            "effect": effect,
            "pval": pval,
            "pAdj": padj,
            "is_associated": is_assoc,
            "region": regions.labels,
        },
        index=out.var_names,
    )
    out.uns[key_added] = {
        "method": method_used,
        "n_associated": int(is_assoc.sum()),
        "n_regions": regions.n_clusters,
        "regions": regions.intervals,
        "pbc": regions.pbc,
        "n_perm": n_perm_used,
        "params": {
            "feature": feature if isinstance(feature, str) else "<array>",
            "coords_key": coords_key,
            "connectivity_key": connectivity_key,
            "n_neighs": n_neighs,
            "layer": layer,
            "method": method,
            "alternative": alternative,
            "p_value": p_value,
            "adjust": adjust,
            "min_abs_corr": min_abs_corr,
            "radius": radius,
            "periodic": periodic,
            "chunk_size": chunk_size,
        },
    }
    return out if copy else None
