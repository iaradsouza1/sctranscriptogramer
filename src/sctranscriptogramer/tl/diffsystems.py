"""Differential systems: per-cell-type case-vs-control testing on the transcriptogram.

Runs the negative-binomial mixed model (:func:`~sctranscriptogramer.stats.nb_mixed.fit_nb_mixed`)
on the window pseudo-counts, once per cell type, contrasting condition with a donor random effect and
optional covariates, then groups significant positions into system clusters
(:func:`~sctranscriptogramer.tl.clusters.identify_clusters`).

Results are written onto the transcriptogram AnnData:

- ``varm['<key>_<group>']`` -- per-position DataFrame (``logFC``, ``pValue``, ``pAdj``, ``tested``,
  ``is_de``, ``cluster``) for each cell type.
- ``uns['<key>']`` -- per-contrast metadata (cluster intervals, ``pbc``, counts, parameters).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from sctranscriptogramer.stats.nb_mixed import fit_nb_mixed
from sctranscriptogramer.stats.permutation import region_permutation
from sctranscriptogramer.tl.clusters import identify_clusters

__all__ = ["differential_systems"]


def _resolve_offset(tg: AnnData, mask: np.ndarray, offset):
    if offset is None:
        return None
    if isinstance(offset, str):
        if offset not in tg.obs:
            raise KeyError(f"offset column {offset!r} not in tg.obs")
        return tg.obs[offset].to_numpy()[mask].astype(float)
    return np.asarray(offset)[mask].astype(float)


def differential_systems(
    tg: AnnData,
    *,
    condition_key: str,
    case,
    control,
    donor_key: str,
    group_key: str | None = None,
    covariates: list[str] | None = None,
    counts_layer: str = "window_counts",
    offset: str | np.ndarray | None = "total_counts",
    norm_method: str = "TMM",
    radius: int | None = None,
    p_value: float = 0.05,
    adjust: str = "BH",
    periodic: bool = True,
    inference: str = "asymptotic",
    cluster_forming_p: float = 0.01,
    n_permutations: int = 200,
    max_exact: int = 500,
    perm_n_jobs: int = 4,
    perm_random_state: int = 0,
    key_added: str = "diffsystems",
    copy: bool = False,
    **fit_kwargs,
) -> AnnData | None:
    """Test differential systems per cell type and identify clusters.

    Parameters
    ----------
    tg
        Transcriptogram AnnData with ``layers[counts_layer]`` (window pseudo-counts) and the
        metadata columns below in ``obs``.
    condition_key, case, control
        Condition column and the two levels to contrast (``logFC`` = case vs control, log2).
    donor_key
        Column identifying subjects/donors -- the random effect (needs >= 2 donors per contrast).
    group_key
        Cell-type (or other) column; the contrast is run independently within each value. If
        ``None``, a single contrast over all metacells is run (labelled ``"all"``).
    covariates
        Optional ``obs`` columns added as fixed effects.
    counts_layer
        Layer of integer window pseudo-counts (default ``"window_counts"``).
    offset
        Per-metacell size factors: an ``obs`` column name (default ``"total_counts"``), an array,
        or ``None`` to let edgePython normalize via ``norm_method``.
    radius
        Window radius for cluster splitting/padding; defaults to ``uns['sctg']['radius']``.
    p_value
        Adjusted-p threshold for calling a position differential.
    adjust
        Multiple-testing method (``"BH"``).
    periodic
        Periodic cluster padding/wrap (default on).
    inference
        ``"asymptotic"`` (default) uses the NB Wald p-values + BH per position. ``"permutation"`` adds
        a **region-level cluster-mass permutation test** (donor-level label permutation): regions are
        formed at ``cluster_forming_p`` and given an FWER-controlled ``region_pvalue`` in
        ``uns[...]['contrasts'][group]['permutation']``. Robust to few donors and to the window
        correlation, at the cost of one model refit per permutation.
    cluster_forming_p
        Primary per-position p threshold defining candidate regions for the permutation test.
    n_permutations
        Number of random permutations when the donor relabelings are too many to enumerate.
    max_exact
        Enumerate all donor relabelings exhaustively when ``C(n_donors, n_case) - 1 <= max_exact``
        (few-donor cohorts get an exact, deterministic null for free).
    perm_n_jobs
        Parallel workers per permutation refit (4 is the empirical sweet spot).
    perm_random_state
        Seed for the random permutations.
    key_added
        Base key for the stored results.
    copy
        Operate on and return a copy instead of mutating ``tg``.
    **fit_kwargs
        Extra arguments forwarded to :func:`fit_nb_mixed` (e.g. ``robust``, ``cpc``, ``mincp``,
        ``n_jobs``, ``verbose``).

    Returns
    -------
    AnnData or None
        The updated AnnData if ``copy=True``, else ``None``.
    """
    out = tg.copy() if copy else tg

    if counts_layer not in out.layers:
        raise KeyError(
            f"layer {counts_layer!r} not found; run tl.transcriptogram with a counts layer first"
        )
    for key in [condition_key, donor_key] + ([group_key] if group_key else []) + list(
        covariates or []
    ):
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

    n_positions = out.n_vars
    present_mask = np.asarray(
        out.uns.get("sctg", {}).get("present_mask", np.ones(n_positions, bool)), dtype=bool
    )
    if inference not in ("asymptotic", "permutation"):
        raise ValueError(f"inference must be 'asymptotic' or 'permutation', got {inference!r}")
    groups = ["all"] if group_key is None else [
        g for g in pd.unique(out.obs[group_key]) if pd.notna(g)
    ]

    contrasts: dict[str, dict] = {}
    for g in groups:
        mask = np.ones(out.n_obs, dtype=bool) if group_key is None else (
            out.obs[group_key].to_numpy() == g
        )
        cond = out.obs[condition_key].to_numpy()[mask]
        keep = np.isin(cond, [case, control])
        n_case = int((cond[keep] == case).sum())
        n_control = int((cond[keep] == control).sum())
        donors = out.obs[donor_key].to_numpy()[mask][keep]

        if n_case == 0 or n_control == 0:
            warnings.warn(
                f"group {g!r}: skipped (case={n_case}, control={n_control})", stacklevel=2
            )
            contrasts[str(g)] = {"skipped": "missing a condition group"}
            continue
        if len(np.unique(donors)) < 2:
            warnings.warn(
                f"group {g!r}: skipped (needs >= 2 donors for the random effect)", stacklevel=2
            )
            contrasts[str(g)] = {"skipped": "fewer than 2 donors"}
            continue

        y = counts_all[mask].T  # positions x metacells(group)
        cov_df = (
            out.obs.loc[mask, covariates].reset_index(drop=True) if covariates else None
        )
        try:
            result = fit_nb_mixed(
                y,
                condition=out.obs[condition_key].to_numpy()[mask],
                donor=out.obs[donor_key].to_numpy()[mask],
                case=case,
                control=control,
                covariates=cov_df,
                offset=_resolve_offset(out, mask, offset),
                norm_method=norm_method,
                adjust=adjust,
                **fit_kwargs,
            )
        except (ValueError, RuntimeError) as exc:
            # e.g. a rare cell type where no position survives the abundance filter
            warnings.warn(f"group {g!r}: skipped (model fit failed: {exc})", stacklevel=2)
            contrasts[str(g)] = {"skipped": f"fit failed: {exc}"}
            continue

        is_de = result.tested & (result.padj <= p_value)
        clusters = identify_clusters(
            np.flatnonzero(is_de), radius=radius, n_positions=n_positions, periodic=periodic
        )

        df = result.to_frame()
        df["is_de"] = is_de
        df["cluster"] = clusters.labels
        df.index = out.var_names
        out.varm[f"{key_added}_{g}"] = df

        info = {
            "n_case": result.n_case,
            "n_control": result.n_control,
            "n_de": int(is_de.sum()),
            "n_clusters": clusters.n_clusters,
            "clusters": clusters.intervals,
            "pbc": clusters.pbc,
            "varm_key": f"{key_added}_{g}",
        }

        if inference == "permutation":
            cond_g = out.obs[condition_key].to_numpy()[mask]
            donor_g = out.obs[donor_key].to_numpy()[mask]
            off_g = _resolve_offset(out, mask, offset)
            cc = np.isin(cond_g, [case, control])
            perm_kwargs = {**fit_kwargs, "n_jobs": perm_n_jobs, "verbose": False}

            def pvalue_fn(cond_vec, _y=y[:, cc], _donor=donor_g[cc],
                          _cov=(cov_df.iloc[cc].reset_index(drop=True) if cov_df is not None else None),
                          _off=(off_g[cc] if off_g is not None else None)):
                return fit_nb_mixed(
                    _y, cond_vec, _donor, case=case, control=control, covariates=_cov,
                    offset=_off, norm_method=norm_method, **perm_kwargs,
                ).pvalue

            rp = region_permutation(
                pvalue_fn, cond_g[cc], donor_g[cc], case=case, control=control, radius=radius,
                present_mask=present_mask, n_positions=n_positions,
                cluster_forming_p=cluster_forming_p, max_exact=max_exact,
                n_permutations=n_permutations, random_state=perm_random_state,
            )
            info["permutation"] = {
                "cluster_forming_p": rp.cluster_forming_p,
                "n_perm": rp.n_perm,
                "exact": rp.exact,
                "regions": rp.regions,
                "mass": rp.mass,
                "region_pvalue": rp.region_pvalue,
                "n_significant": int(np.sum(rp.region_pvalue <= p_value)),
            }

        contrasts[str(g)] = info

    out.uns[key_added] = {
        "contrasts": contrasts,
        "params": {
            "condition_key": condition_key,
            "case": case,
            "control": control,
            "donor_key": donor_key,
            "group_key": group_key,
            "covariates": list(covariates or []),
            "counts_layer": counts_layer,
            "norm_method": norm_method,
            "radius": radius,
            "p_value": p_value,
            "adjust": adjust,
            "periodic": periodic,
            "inference": inference,
            "cluster_forming_p": cluster_forming_p if inference == "permutation" else None,
        },
    }
    return out if copy else None
