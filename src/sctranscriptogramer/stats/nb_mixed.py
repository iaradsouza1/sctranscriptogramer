"""Negative-binomial mixed-model differential testing via edgePython (NEBULA-LN).

This adapter runs edgePython's single-cell NB gamma mixed model
(``glm_sc_fit`` -> ``shrink_sc_disp`` -> ``glm_sc_test``) for a two-group contrast with a random
effect (donor/subject) and optional fixed-effect covariates. It is used to test each transcriptogram
position (a window pseudo-count) for a case-vs-control difference.

The random effect is the key to correct statistics on metacells: metacells from the same donor are
not independent, and modelling donor as a random effect absorbs that correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sctranscriptogramer.stats.multitest import benjamini_hochberg

__all__ = ["fit_nb_mixed", "NBMixedResult"]


@dataclass
class NBMixedResult:
    """Per-feature results of the NB mixed-model contrast, aligned to all input features.

    Arrays have length ``n_features`` (positions). Features dropped by edgePython's abundance filter
    have ``tested=False`` and ``NaN`` statistics.
    """

    logFC: NDArray[np.float64]
    pvalue: NDArray[np.float64]
    padj: NDArray[np.float64]
    tested: NDArray[np.bool_]
    converged: NDArray[np.bool_]
    n_case: int
    n_control: int
    coef_name: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "logFC": self.logFC,
                "pValue": self.pvalue,
                "pAdj": self.padj,
                "tested": self.tested,
                "converged": self.converged,
            }
        )


def _build_design(
    condition01: NDArray[np.int_], covariates: pd.DataFrame | None
) -> tuple[NDArray[np.float64], list[str], int]:
    """Design matrix with intercept, the condition term (coef of interest), then covariates.

    Numeric covariates enter as-is; categorical/string covariates are one-hot encoded (drop-first).
    Returns the design, the column names, and the index of the condition coefficient (always 1).
    """
    n = condition01.shape[0]
    cols = [np.ones(n), condition01.astype(np.float64)]
    names = ["intercept", "condition"]
    _append_covariates(cols, names, covariates)
    design = np.column_stack(cols)
    return design, names, 1


def _append_covariates(
    cols: list[NDArray], names: list[str], covariates: pd.DataFrame | None
) -> None:
    """Append numeric covariates as-is and categorical ones one-hot (drop-first), in place."""
    if covariates is None or covariates.shape[1] == 0:
        return
    for name in covariates.columns:
        series = covariates[name]
        if pd.api.types.is_numeric_dtype(series):
            cols.append(series.to_numpy(dtype=np.float64))
            names.append(str(name))
        else:
            dummies = pd.get_dummies(series.astype("category"), drop_first=True)
            for dcol in dummies.columns:
                cols.append(dummies[dcol].to_numpy(dtype=np.float64))
                names.append(f"{name}[{dcol}]")


def fit_nb_mixed(
    counts: NDArray,
    condition: NDArray,
    donor: NDArray,
    *,
    case,
    control,
    covariates: pd.DataFrame | None = None,
    offset: NDArray | None = None,
    norm_method: str = "TMM",
    adjust: str = "BH",
    robust: bool = True,
    cpc: float = 0.005,
    mincp: int = 5,
    n_jobs: int = 1,
    verbose: bool = False,
) -> NBMixedResult:
    """Fit the NB mixed model and test ``case`` vs ``control`` per feature.

    Parameters
    ----------
    counts
        Integer count matrix, **features x observations** (positions x metacells).
    condition
        Length-``n_observations`` condition labels.
    donor
        Length-``n_observations`` subject/donor labels (the random effect).
    case, control
        The two values of ``condition`` to contrast; ``logFC`` is ``case`` relative to ``control``
        (log2). Observations with any other condition value are dropped.
    covariates
        Optional per-observation fixed-effect covariates (rows aligned to ``counts`` columns).
    offset
        Optional positive per-observation size factors (e.g. total counts). If ``None``,
        edgePython computes an offset via ``norm_method``.
    norm_method
        Normalization used when ``offset`` is ``None`` (``"TMM"`` or ``"none"``).
    adjust
        Multiple-testing method for ``padj`` (currently ``"BH"``).
    robust
        Robust empirical-Bayes dispersion shrinkage.
    cpc, mincp
        edgePython abundance filters (min mean counts per cell; min non-zero cells).
    n_jobs
        Parallel workers for the fit.
    verbose
        Forward edgePython progress messages.

    Returns
    -------
    NBMixedResult
    """
    import edgepython as ep

    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D (features x observations)")
    condition = np.asarray(condition)
    donor = np.asarray(donor)
    n_features, n_obs = counts.shape
    if condition.shape[0] != n_obs or donor.shape[0] != n_obs:
        raise ValueError("condition and donor must have length n_observations (counts.shape[1])")

    keep_obs = np.isin(condition, [case, control])
    if keep_obs.sum() < 2:
        raise ValueError("need at least one case and one control observation")
    y = counts[:, keep_obs]
    cond = condition[keep_obs]
    don = donor[keep_obs]
    cov = covariates.iloc[keep_obs].reset_index(drop=True) if covariates is not None else None

    condition01 = (cond == case).astype(np.int64)
    n_case = int(condition01.sum())
    n_control = int((condition01 == 0).sum())
    if n_case == 0 or n_control == 0:
        raise ValueError(f"contrast needs both groups; got case={n_case}, control={n_control}")
    if len(np.unique(don)) < 2:
        raise ValueError(
            "the random effect needs >= 2 donors/subjects; got 1 "
            "(a mixed model cannot be identified with a single subject)"
        )

    design, names, coef = _build_design(condition01, cov)
    cell_meta = pd.DataFrame({"__sample__": don})

    fit = ep.glm_sc_fit(
        y,
        cell_meta=cell_meta,
        design=design,
        sample="__sample__",
        offset=None if offset is None else np.asarray(offset)[keep_obs].astype(np.float64),
        norm_method=norm_method,
        cpc=cpc,
        mincp=mincp,
        ncore=n_jobs,
        verbose=verbose,
    )
    fit = ep.shrink_sc_disp(fit, counts=y, robust=robust)
    res = ep.glm_sc_test(fit, coef=coef)
    table = res["table"]

    gene_mask = np.asarray(fit.get("gene_mask"))
    if gene_mask is None or gene_mask.shape[0] != n_features:
        # No filtering info: assume all features present, in order.
        gene_mask = np.ones(n_features, dtype=bool)
    kept_idx = np.flatnonzero(gene_mask)
    if kept_idx.shape[0] != table.shape[0]:
        raise RuntimeError(
            "edgePython result length does not match the gene_mask; cannot realign to positions"
        )

    logFC = np.full(n_features, np.nan)
    pvalue = np.full(n_features, np.nan)
    converged = np.zeros(n_features, dtype=bool)
    logFC[kept_idx] = table["logFC"].to_numpy()
    pvalue[kept_idx] = table["PValue"].to_numpy()
    if "converged" in table:
        converged[kept_idx] = table["converged"].to_numpy().astype(bool)
    else:
        converged[kept_idx] = True

    if adjust.upper() != "BH":
        raise ValueError(f"unsupported adjust method {adjust!r}; only 'BH' is implemented")
    padj = benjamini_hochberg(pvalue)

    tested = gene_mask.copy()
    return NBMixedResult(
        logFC=logFC,
        pvalue=pvalue,
        padj=padj,
        tested=tested,
        converged=converged,
        n_case=n_case,
        n_control=n_control,
        coef_name=f"{case}-{control}",
    )


@dataclass
class NBFactorResult:
    """Per-feature results of the NB mixed-model omnibus factor test, aligned to all features.

    ``omnibus_p`` is the ACAT-combined p-value that *any* level of the factor differs from the
    reference (donor-adjusted); ``max_abs_logFC`` is the largest absolute level effect (an effect
    size). Features dropped by the abundance filter have ``tested=False`` and NaN statistics.
    """

    omnibus_p: NDArray[np.float64]
    padj: NDArray[np.float64]
    max_abs_logFC: NDArray[np.float64]
    tested: NDArray[np.bool_]
    levels: list
    n_per_level: dict


def _build_factor_design(
    factor: NDArray, covariates: pd.DataFrame | None
) -> tuple[NDArray[np.float64], list[str], list[int], list]:
    """Design for a multi-level factor: intercept, k-1 treatment dummies, then covariates.

    Returns the design, the column names, the indices of the k-1 factor coefficients (to test), and
    the ordered factor levels.
    """
    cat = pd.Categorical(factor)
    levels = list(cat.categories)
    n = len(cat)
    cols: list[NDArray] = [np.ones(n)]
    names = ["intercept"]
    group_idx: list[int] = []
    dummies = pd.get_dummies(cat, drop_first=True)
    for dcol in dummies.columns:
        cols.append(dummies[dcol].to_numpy(dtype=np.float64))
        names.append(f"group[{dcol}]")
        group_idx.append(len(cols) - 1)
    _append_covariates(cols, names, covariates)
    return np.column_stack(cols), names, group_idx, levels


def fit_nb_factor(
    counts: NDArray,
    factor: NDArray,
    donor: NDArray,
    *,
    covariates: pd.DataFrame | None = None,
    offset: NDArray | None = None,
    norm_method: str = "TMM",
    adjust: str = "BH",
    robust: bool = True,
    cpc: float = 0.005,
    mincp: int = 5,
    n_jobs: int = 1,
    verbose: bool = False,
) -> NBFactorResult:
    """Omnibus NB mixed-model test that a multi-level ``factor`` affects each feature.

    Fits one NB gamma mixed model per feature (donor random effect + optional covariates) with the
    factor treatment-coded, Wald-tests each of the ``k-1`` level coefficients, and combines them with
    ACAT into a single omnibus p-value per feature ("does any level differ from the reference").

    Parameters
    ----------
    counts
        Integer count matrix, **features x observations** (positions x metacells).
    factor
        Length-``n_observations`` factor labels (>= 2 levels), e.g. cell state.
    donor
        Length-``n_observations`` subject/donor labels (the random effect). The factor must vary
        **within** donors (be crossed with donor), else it is confounded with the random effect.
    covariates
        Optional per-observation fixed-effect covariates.
    offset, norm_method, adjust, robust, cpc, mincp, n_jobs, verbose
        As in :func:`fit_nb_mixed`.

    Returns
    -------
    NBFactorResult
    """
    import edgepython as ep

    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D (features x observations)")
    factor = np.asarray(factor)
    donor = np.asarray(donor)
    n_features, n_obs = counts.shape
    if factor.shape[0] != n_obs or donor.shape[0] != n_obs:
        raise ValueError("factor and donor must have length n_observations (counts.shape[1])")

    cat = pd.Categorical(factor)
    levels = list(cat.categories)
    if len(levels) < 2:
        raise ValueError("factor needs >= 2 levels")
    if len(np.unique(donor)) < 2:
        raise ValueError("the random effect needs >= 2 donors/subjects")
    n_per_level = {str(lv): int((cat == lv).sum()) for lv in levels}
    if min(n_per_level.values()) < 2:
        raise ValueError(f"every factor level needs >= 2 observations; got {n_per_level}")

    design, _, group_idx, _ = _build_factor_design(factor, covariates)
    cell_meta = pd.DataFrame({"__sample__": donor})

    fit = ep.glm_sc_fit(
        counts,
        cell_meta=cell_meta,
        design=design,
        sample="__sample__",
        offset=None if offset is None else np.asarray(offset).astype(np.float64),
        norm_method=norm_method,
        cpc=cpc,
        mincp=mincp,
        ncore=n_jobs,
        verbose=verbose,
    )
    fit = ep.shrink_sc_disp(fit, counts=counts, robust=robust)

    # Wald-test each level coefficient; collect p-values and logFCs (kept features x (k-1)).
    pmat, lmat = [], []
    for gi in group_idx:
        table = ep.glm_sc_test(fit, coef=gi)["table"]
        pmat.append(table["PValue"].to_numpy())
        lmat.append(table["logFC"].to_numpy())
    pmat = np.column_stack(pmat)
    lmat = np.column_stack(lmat)

    from sctranscriptogramer.stats.multitest import acat, benjamini_hochberg

    omnibus_kept = acat(pmat, axis=1)
    max_abs_kept = np.max(np.abs(lmat), axis=1)

    gene_mask = np.asarray(fit.get("gene_mask"))
    if gene_mask is None or gene_mask.shape[0] != n_features:
        gene_mask = np.ones(n_features, dtype=bool)
    kept_idx = np.flatnonzero(gene_mask)
    if kept_idx.shape[0] != omnibus_kept.shape[0]:
        raise RuntimeError("edgePython result length does not match gene_mask")

    omnibus = np.full(n_features, np.nan)
    max_abs = np.full(n_features, np.nan)
    omnibus[kept_idx] = omnibus_kept
    max_abs[kept_idx] = max_abs_kept

    if adjust.upper() != "BH":
        raise ValueError(f"unsupported adjust method {adjust!r}; only 'BH' is implemented")
    padj = benjamini_hochberg(omnibus)

    return NBFactorResult(
        omnibus_p=omnibus,
        padj=padj,
        max_abs_logFC=max_abs,
        tested=gene_mask.copy(),
        levels=levels,
        n_per_level=n_per_level,
    )
