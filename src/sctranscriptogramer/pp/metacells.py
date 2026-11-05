"""Aggregate cells into metacells, respecting biological grouping.

The single most important guarantee here: **metacells are pure in the grouping columns**. Cells are
*partitioned first* by ``group_keys`` (e.g. condition/disease, and optionally cell type and donor),
and metacells are built only *within* each partition -- never across. This means:

- a metacell never mixes case and control cells (so the downstream case/control contrast is clean);
- with a cell-type key, a metacell never mixes cell types;
- with a donor key, every metacell belongs to exactly one donor, which is what the differential
  systems model needs to use donor as a random effect.

Within each partition, the subdivision into metacells comes from one of three sources:

``engine``
    A metacell algorithm (``"metacells2"`` or any callable) run independently per partition.
``assignment_key``
    A precomputed per-cell metacell label already in ``adata.obs`` (e.g. from an external
    metacells2 / SEACells run). Final metacells are ``group_keys`` x this label, so purity is
    enforced even if the external labels crossed groups.
neither
    Pseudobulk: each partition collapses to a single aggregate.

The result is a metacells x genes AnnData carrying summed ``counts`` (for the differential-systems
path), a normalized ``X`` (for the embedding), the pure grouping metadata, ``n_cells`` and
``total_counts`` per metacell -- ready for :func:`~sctranscriptogramer.pp.map_ids.project_to_ordering`.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

__all__ = ["aggregate_metacells", "suggest_metacell_size"]


def suggest_metacell_size(
    adata: AnnData,
    *,
    group_keys,
    counts_layer: str | None = None,
    min_metacells_per_group: int = 3,
    min_umis_per_metacell: float = 10_000,
    engine_min: int = 12,
) -> dict:
    """Suggest a ``target_metacell_size`` (cells) from the data, as a cheap diagnostic.

    Metacell size for this method is a downstream-power decision: metacells must have enough UMIs to
    give stable window pseudo-counts, yet be small enough to leave several metacells per (donor x
    group) partition for the mixed model. This function inspects the ``group_keys`` partitions and the
    per-cell depth and returns the **smallest** metacell size that still clears the UMI and engine
    floors (which maximises replication), then checks how many metacells that yields per partition.

    No metacells are built -- this only reads ``obs`` and count totals.

    Parameters
    ----------
    adata
        Cells x genes AnnData.
    group_keys
        The partition columns you will pass to :func:`aggregate_metacells` (the replication units,
        e.g. ``["condition", "cell_state", "donor"]``).
    counts_layer
        Layer with raw counts for per-cell UMIs; defaults to ``adata.X``.
    min_metacells_per_group
        Target number of metacells per partition for adequate within-donor replication (feasibility
        check; a warning is issued if the typical partition falls short).
    min_umis_per_metacell
        Minimum UMIs a metacell should carry for stable pseudo-counts.
    engine_min
        The engine's minimum metacell size in cells (metacells2 ``min_metacell_size``, default 12).

    Returns
    -------
    dict
        ``target_metacell_size`` plus diagnostics: partition sizes, median cell UMIs, expected
        metacells per partition at the suggestion, the binding floor, and any notes.
    """
    if isinstance(group_keys, str):
        group_keys = [group_keys]
    group_keys = list(group_keys)
    missing = [k for k in group_keys if k not in adata.obs]
    if missing:
        raise KeyError(f"columns not found in adata.obs: {missing}")

    counts = adata.layers[counts_layer] if counts_layer is not None else adata.X
    cell_umis = (
        np.asarray(counts.sum(axis=1)).ravel() if sp.issparse(counts)
        else np.asarray(counts).sum(axis=1)
    )
    median_umis = float(np.median(cell_umis))

    sizes = adata.obs.groupby(group_keys, observed=True).size().to_numpy()
    umi_floor_size = int(np.ceil(min_umis_per_metacell / max(median_umis, 1.0)))
    suggested = int(max(umi_floor_size, engine_min))

    expected = np.maximum(1, np.round(sizes / suggested)).astype(int)
    med_expected = int(np.median(expected))
    n_too_small = int((sizes < engine_min).sum())

    notes = []
    if med_expected < min_metacells_per_group:
        notes.append(
            f"the typical partition yields ~{med_expected} metacells at size {suggested} "
            f"(< target {min_metacells_per_group}); partitions are cell-limited — consider coarser "
            "group_keys, more cells, or accept reduced within-donor replication"
        )
    if n_too_small:
        notes.append(
            f"{n_too_small}/{len(sizes)} partitions have < {engine_min} cells and will collapse to "
            "one metacell or be dropped"
        )
    if umi_floor_size > engine_min:
        notes.append(
            f"size is UMI-limited: cells are shallow (median {int(median_umis)} UMIs), so "
            f"{suggested} cells are needed to reach {int(min_umis_per_metacell)} UMIs/metacell"
        )
    for n in notes:
        warnings.warn(n, stacklevel=2)

    return {
        "target_metacell_size": suggested,
        "n_partitions": int(len(sizes)),
        "cells_per_partition": {
            "min": int(sizes.min()),
            "q25": float(np.quantile(sizes, 0.25)),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
        },
        "median_cell_umis": median_umis,
        "expected_metacells_per_partition": {"median": med_expected, "min": int(expected.min())},
        "binding_floor": "umis" if umi_floor_size > engine_min else "engine_min",
        "engine_min": engine_min,
        "notes": notes,
    }

# An engine maps a per-partition AnnData to integer metacell labels (one per cell);
# label < 0 marks an outlier cell to be dropped.
Engine = Callable[[AnnData], np.ndarray]


def _as_csr(x) -> sp.csr_matrix:
    return x.tocsr() if sp.issparse(x) else sp.csr_matrix(np.asarray(x))


def _partition_codes(obs: pd.DataFrame, group_keys: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Integer code per cell for its group combination, plus a table of the combos."""
    if not group_keys:
        codes = np.zeros(len(obs), dtype=np.int64)
        combos = pd.DataFrame(index=[0])
        return codes, combos
    combo = obs[group_keys].astype("object")
    tuples = pd.MultiIndex.from_frame(combo)
    codes, uniques = pd.factorize(tuples, sort=True)
    combos = pd.DataFrame(list(uniques), columns=group_keys)
    return codes.astype(np.int64), combos


def _within_labels(
    adata: AnnData,
    counts: sp.csr_matrix,
    partition_codes: np.ndarray,
    assignment_key: str | None,
    engine: Engine | str | None,
    engine_kwargs: dict,
) -> np.ndarray:
    """Per-cell within-partition label; -1 marks a cell to drop (outlier).

    The engine is fed a sub-AnnData whose ``X`` is the **raw counts** for that partition (metacell
    algorithms such as metacells2 expect UMIs), with the original ``obs``/``var`` attached.
    """
    n = adata.n_obs
    if assignment_key is not None:
        raw = adata.obs[assignment_key].to_numpy()
        return pd.factorize(raw, sort=True)[0].astype(np.int64)

    if engine is None:
        return np.zeros(n, dtype=np.int64)  # pseudobulk: one metacell per partition

    engine_fn = _resolve_engine(engine)
    labels = np.full(n, -1, dtype=np.int64)
    for code in np.unique(partition_codes):
        idx = np.flatnonzero(partition_codes == code)
        sub = AnnData(X=counts[idx].copy(), obs=adata.obs.iloc[idx].copy(), var=adata.var.copy())
        sub_labels = np.asarray(engine_fn(sub, **engine_kwargs), dtype=np.int64)
        if sub_labels.shape[0] != idx.shape[0]:
            raise ValueError("engine returned the wrong number of labels for a partition")
        labels[idx] = sub_labels
    return labels


def _resolve_engine(engine: Engine | str) -> Engine:
    if callable(engine):
        return engine
    if engine == "metacells2":
        from sctranscriptogramer.pp._metacells2 import metacells2_assign

        return metacells2_assign
    raise ValueError(f"unknown engine {engine!r}; pass a callable or 'metacells2'")


def _sum_by_labels(counts: sp.csr_matrix, labels: np.ndarray, n_labels: int) -> sp.csr_matrix:
    """Sum rows of ``counts`` (cells x genes) within each label -> (n_labels x genes)."""
    n_cells = counts.shape[0]
    indicator = sp.csr_matrix(
        (np.ones(n_cells), (labels, np.arange(n_cells))),
        shape=(n_labels, n_cells),
    )
    return indicator @ counts


def _log2cpm(counts: np.ndarray, prior_count: float, lib: np.ndarray | None = None) -> np.ndarray:
    """edgeR-style log2 CPM (``cpm(log=TRUE, prior.count=...)``).

    The prior count is scaled by each metacell's (effective) library size and added to both
    numerator and, doubled, to the library size -- matching edgeR. Putting the embedding on this
    scale means the descriptive difference-of-means curve and edgePython's ``logFC`` share the same
    log2-CPM axis. Pass ``lib`` to use effective (TMM-normalized) library sizes instead of raw sums.
    """
    if lib is None:
        lib = counts.sum(axis=1)  # per metacell
    mean_lib = lib.mean()
    if mean_lib <= 0:
        mean_lib = 1.0
    prior_scaled = prior_count * lib / mean_lib  # (n_metacells,)
    adj_lib = lib + 2.0 * prior_scaled
    adj_lib[adj_lib == 0] = 1.0
    cpm = (counts + prior_scaled[:, None]) / adj_lib[:, None] * 1e6
    return np.log2(cpm)


def _tmm_effective_lib(counts: np.ndarray) -> np.ndarray:
    """Effective library sizes = raw library size x TMM normalization factor (per metacell).

    Uses edgePython's ``calc_norm_factors`` (the exact TMM the differential-systems model uses), so
    the embedding is normalized composition-robustly and on the same footing as the DE. edgePython
    expects a genes x samples matrix.
    """
    import edgepython as ep

    lib = counts.sum(axis=1)
    factors = np.asarray(ep.calc_norm_factors(counts.T, method="TMM"), dtype=np.float64)
    if factors.shape[0] != lib.shape[0] or not np.all(np.isfinite(factors) & (factors > 0)):
        raise ValueError("TMM produced invalid normalization factors")
    return lib * factors


def _log1p_cp(counts: np.ndarray, target_sum: float) -> np.ndarray:
    """scanpy-style CP-target-sum followed by natural-log ``log1p``."""
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(counts / totals * target_sum)


def _normalize(
    counts: sp.csr_matrix, method: str, *, prior_count: float, target_sum: float
) -> np.ndarray:
    """Normalize summed metacell counts for the embedding (dense metacell x genes)."""
    dense = np.asarray(counts.todense(), dtype=np.float64)
    if method == "tmm_log2cpm":
        try:
            lib = _tmm_effective_lib(dense)
        except ImportError:
            warnings.warn(
                "normalization='tmm_log2cpm' needs edgePython (`pip install "
                "sctranscriptogramer[de]`); falling back to 'log2cpm'",
                stacklevel=2,
            )
            return _log2cpm(dense, prior_count)
        except Exception as exc:  # TMM can fail on degenerate/tiny inputs
            warnings.warn(
                f"TMM normalization failed ({exc}); falling back to 'log2cpm'", stacklevel=2
            )
            return _log2cpm(dense, prior_count)
        return _log2cpm(dense, prior_count, lib=lib)
    if method == "log2cpm":
        return _log2cpm(dense, prior_count)
    if method == "log1p":
        return _log1p_cp(dense, target_sum)
    raise ValueError(
        f"unknown normalization {method!r}; expected 'tmm_log2cpm', 'log2cpm' or 'log1p'"
    )


def aggregate_metacells(
    adata: AnnData,
    *,
    group_keys: str | Sequence[str] | None = None,
    assignment_key: str | None = None,
    engine: Engine | str | None = None,
    counts_layer: str | None = None,
    carry_obs: Sequence[str] | None = None,
    normalize: bool = True,
    normalization: str = "tmm_log2cpm",
    prior_count: float = 2.0,
    target_sum: float = 1e4,
    min_cells: int = 1,
    engine_kwargs: dict | None = None,
) -> AnnData:
    """Aggregate cells into group-pure metacells.

    Parameters
    ----------
    adata
        Cells x genes AnnData with raw counts (in ``counts_layer`` or ``X``).
    group_keys
        Column(s) in ``adata.obs`` defining the partitions cells may be aggregated within.
        Cells are **never** aggregated across different values of these columns. Pass the
        condition/disease column to keep case and control separate; add a cell-type column to keep
        cell types separate; add a donor/sample column so each metacell belongs to one donor.
    assignment_key
        Optional precomputed per-cell metacell label in ``adata.obs``. Final metacells are the
        cross of ``group_keys`` and this label (purity is still enforced).
    engine
        Metacell algorithm run within each partition: ``"metacells2"`` or a callable
        ``(sub_adata, **engine_kwargs) -> labels`` (negative label = outlier, dropped). If both
        ``engine`` and ``assignment_key`` are ``None``, each partition collapses to one pseudobulk.
    counts_layer
        Layer holding raw counts; defaults to ``adata.X``.
    carry_obs
        Extra ``adata.obs`` columns to carry onto metacells for use as model covariates. Numeric
        columns (e.g. ``percent.mt``) are **averaged** within the metacell; categorical columns take
        the majority value (a warning is issued if a metacell is not pure in one). Donor-level
        covariates (sex, age, PMI) are constant within a metacell, so they carry over exactly.
    normalize
        If True, ``X`` is normalized for the embedding (see ``normalization``); otherwise ``X``
        holds raw summed counts. ``layers['counts']`` always holds raw counts for the DE path.
    normalization
        Embedding normalization when ``normalize=True``:
        ``"tmm_log2cpm"`` (default) -- edgeR-style ``log2(CPM + prior)`` on **TMM effective library
        sizes**, i.e. the same composition-robust normalization the differential-systems model uses,
        so the embedding does not carry a systemic compositional shift (needs edgePython; falls back
        to ``"log2cpm"`` if unavailable);
        ``"log2cpm"`` -- the same but on raw library sizes (composition-sensitive);
        ``"log1p"`` -- scanpy-style CP-``target_sum`` + natural-log ``log1p``.
    prior_count
        Prior count for the log2CPM methods (edgeR default 2.0), stabilising low-abundance genes
        and avoiding ``log2(0)``.
    target_sum
        Target total for the ``"log1p"`` normalization.
    min_cells
        Drop metacells backed by fewer than this many cells (after outlier removal).
    engine_kwargs
        Extra keyword arguments forwarded to the engine.

    Returns
    -------
    AnnData
        Metacells x genes, with ``layers['counts']`` (summed raw counts), pure ``group_keys``
        metadata, ``obs['n_cells']`` and ``obs['total_counts']``, and ``uns['sctg_metacells']``.
    """
    if isinstance(group_keys, str):
        group_keys = [group_keys]
    group_keys = list(group_keys or [])
    carry_obs = list(carry_obs or [])
    engine_kwargs = dict(engine_kwargs or {})

    missing = [k for k in [*group_keys, *carry_obs] if k not in adata.obs]
    if assignment_key is not None and assignment_key not in adata.obs:
        missing.append(assignment_key)
    if missing:
        raise KeyError(f"columns not found in adata.obs: {missing}")

    counts = _as_csr(adata.layers[counts_layer] if counts_layer is not None else adata.X)

    partition_codes, _ = _partition_codes(adata.obs, group_keys)
    within = _within_labels(
        adata, counts, partition_codes, assignment_key, engine, engine_kwargs
    )

    keep = within >= 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.warn(f"dropping {n_dropped} outlier cells flagged by the engine", stacklevel=2)

    # Final metacell = (partition, within-partition label), factorized over kept cells.
    pair = np.stack([partition_codes[keep], within[keep]], axis=1)
    mc_codes, mc_pairs = pd.factorize(pd.MultiIndex.from_arrays(pair.T), sort=True)
    mc_codes = mc_codes.astype(np.int64)
    n_mc = len(mc_pairs)

    kept_counts = counts[keep]
    kept_obs = adata.obs.iloc[keep]

    mc_counts = _sum_by_labels(kept_counts, mc_codes, n_mc)
    n_cells = np.asarray(
        _sum_by_labels(sp.csr_matrix(np.ones((kept_counts.shape[0], 1))), mc_codes, n_mc).todense()
    ).ravel()
    total_counts = np.asarray(mc_counts.sum(axis=1)).ravel()

    # Build pure metadata: group_keys are constant within a metacell by construction.
    obs = pd.DataFrame(index=[f"mc{i}" for i in range(n_mc)])
    first_cell_of_mc = pd.Series(np.arange(len(mc_codes))).groupby(mc_codes).first().to_numpy()
    for key in group_keys:
        obs[key] = kept_obs[key].to_numpy()[first_cell_of_mc]
    for key in carry_obs:
        series = kept_obs[key]
        if pd.api.types.is_numeric_dtype(series):
            # continuous covariate (e.g. percent.mt): average within the metacell.
            agg = pd.Series(series.to_numpy(), index=mc_codes).groupby(level=0).mean()
            obs[key] = agg.reindex(range(n_mc)).to_numpy()
        else:
            # categorical: majority value; warn if a metacell is not pure in it.
            by_mc = pd.Series(series.to_numpy()).groupby(mc_codes)
            modes = by_mc.agg(lambda s: s.value_counts().index[0])
            if (by_mc.nunique() > 1).any():
                warnings.warn(
                    f"carry_obs column {key!r} is not constant within some metacells; "
                    "using the majority value",
                    stacklevel=2,
                )
            obs[key] = modes.to_numpy()
    obs["n_cells"] = n_cells.astype(np.int64)
    obs["total_counts"] = total_counts

    if min_cells > 1:
        big = n_cells >= min_cells
        if not big.all():
            warnings.warn(
                f"dropping {int((~big).sum())} metacells with < {min_cells} cells", stacklevel=2
            )
            mc_counts = mc_counts[big]
            obs = obs.iloc[big].copy()
            n_mc = int(big.sum())

    X = (
        _normalize(mc_counts, normalization, prior_count=prior_count, target_sum=target_sum)
        if normalize
        else np.asarray(mc_counts.todense())
    )
    mc = AnnData(X=X, obs=obs, var=adata.var.copy())
    mc.layers["counts"] = np.asarray(mc_counts.todense())
    mc.uns["sctg_metacells"] = {
        "group_keys": group_keys,
        "assignment_key": assignment_key,
        "engine": engine if isinstance(engine, str) else getattr(engine, "__name__", "callable"),
        "n_metacells": int(n_mc),
        "n_input_cells": int(adata.n_obs),
        "n_outliers_dropped": n_dropped,
        "normalized": normalize,
        "normalization": normalization if normalize else None,
        "prior_count": (
            prior_count if (normalize and normalization in ("tmm_log2cpm", "log2cpm")) else None
        ),
    }
    return mc
