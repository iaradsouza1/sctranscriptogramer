"""Gene-set over-representation of the differential-systems clusters, via GSEApy.

For each cluster of a contrast, the cluster's proteins are tested for over-representation against a
background (the ordering by default) using GSEApy's hypergeometric over-representation analysis
(:func:`gseapy.enrich` offline, or :func:`gseapy.enrichr` online). We do not re-implement the
statistics -- this is a thin adapter that assembles the per-cluster gene lists, calls GSEApy, and
collects the results back onto the AnnData.

Identifier spaces must match between the cluster genes and the gene sets. By default the ordering's
protein ids are used; set ``gene_symbol_col`` to a ``var`` column (e.g. mapped gene symbols) to use
standard symbol-keyed Enrichr libraries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from sctranscriptogramer._core.ordering import _strip_prefix

__all__ = ["cluster_enrichment"]

_RENAME = {
    "Term": "term",
    "Overlap": "overlap",
    "P-value": "pValue",
    "Adjusted P-value": "pAdj",
    "Odds Ratio": "odds_ratio",
    "Combined Score": "combined_score",
    "Genes": "genes",
}


def _gene_ids(tg: AnnData, gene_symbol_col: str | None, strip_prefix: bool) -> np.ndarray:
    if gene_symbol_col is not None:
        if gene_symbol_col not in tg.var:
            raise KeyError(f"gene_symbol_col {gene_symbol_col!r} not in tg.var")
        return tg.var[gene_symbol_col].to_numpy().astype(str)
    proteins = tg.var["protein"].to_numpy().astype(str)
    if strip_prefix:
        proteins = np.array([_strip_prefix(p) for p in proteins])
    return proteins


def cluster_enrichment(
    tg: AnnData,
    gene_sets,
    *,
    contrasts: list[str] | None = None,
    universe: list[str] | None = None,
    use: str = "cluster",
    gene_symbol_col: str | None = None,
    strip_prefix: bool = True,
    online: bool = False,
    organism: str = "human",
    p_value: float = 0.05,
    diffsystems_key: str = "diffsystems",
    key_added: str = "enrichment",
    copy: bool = False,
    **enrich_kwargs,
) -> AnnData | None:
    """Over-representation of clusters from :func:`differential_systems`, using GSEApy.

    Parameters
    ----------
    tg
        Transcriptogram AnnData carrying ``uns[diffsystems_key]``, the per-contrast ``varm`` tables,
        and ``var['protein']``.
    gene_sets
        Passed straight to GSEApy: a ``dict`` (``term -> [genes]``), a ``.gmt`` path, or -- when
        ``online=True`` -- Enrichr library name(s).
    contrasts
        Which contrasts (cell types) to enrich; defaults to all with a stored ``varm`` table.
    universe
        Background gene ids; defaults to all genes in the ordering (in the chosen id space).
    use
        ``"cluster"`` (default) uses every position in a cluster; ``"de"`` only ``is_de`` positions.
    gene_symbol_col
        ``var`` column to use as gene identifiers instead of the protein ids (e.g. mapped symbols),
        so standard symbol-keyed libraries can be used.
    strip_prefix
        Strip the taxonomy prefix from protein ids (ignored if ``gene_symbol_col`` is set).
    online
        Use :func:`gseapy.enrichr` (queries Enrichr) instead of the offline :func:`gseapy.enrich`.
    organism
        Organism for the online path.
    p_value
        Keep terms with GSEApy ``Adjusted P-value`` at or below this.
    diffsystems_key, key_added
        uns keys to read results from and write enrichment to.
    copy
        Operate on and return a copy instead of mutating ``tg``.

    Returns
    -------
    AnnData or None
        Results stored in ``uns[key_added][contrast]`` as a DataFrame (GSEApy columns, renamed to
        ``term/overlap/pValue/pAdj/odds_ratio/combined_score/genes``, plus ``cluster``).
    """
    import gseapy

    out = tg.copy() if copy else tg
    if diffsystems_key not in out.uns:
        raise KeyError(f"uns[{diffsystems_key!r}] not found; run tl.differential_systems first")
    if "protein" not in out.var:
        raise KeyError("tg.var['protein'] not found; project onto an ordering first")
    if use not in {"cluster", "de"}:
        raise ValueError(f"use must be 'cluster' or 'de', got {use!r}")

    gene_ids = _gene_ids(out, gene_symbol_col, strip_prefix)
    background = list(universe) if universe is not None else [g for g in pd.unique(gene_ids) if g]

    all_contrasts = out.uns[diffsystems_key]["contrasts"]
    selected = contrasts if contrasts is not None else [
        g for g, info in all_contrasts.items() if "varm_key" in info
    ]

    def _run(study: list[str]) -> pd.DataFrame | None:
        try:
            if online:
                res = gseapy.enrichr(
                    gene_list=study, gene_sets=gene_sets, organism=organism,
                    background=background, outdir=None, no_plot=True, verbose=False,
                    **enrich_kwargs,
                )
            else:
                res = gseapy.enrich(
                    gene_list=study, gene_sets=gene_sets, background=background,
                    outdir=None, no_plot=True, verbose=False, **enrich_kwargs,
                )
        except Exception as exc:  # GSEApy raises on empty overlaps etc.
            import warnings

            warnings.warn(f"gseapy enrichment failed for a cluster: {exc}", stacklevel=2)
            return None
        return res.results if res is not None else None

    results: dict[str, pd.DataFrame] = {}
    for g in selected:
        info = all_contrasts.get(g, {})
        varm_key = info.get("varm_key")
        if varm_key is None or varm_key not in out.varm:
            continue
        table = out.varm[varm_key]
        clusters = table["cluster"].to_numpy()
        is_de = table["is_de"].to_numpy().astype(bool)

        frames = []
        for c in sorted(x for x in np.unique(clusters) if x > 0):
            in_cluster = clusters == c
            if use == "de":
                in_cluster &= is_de
            study = sorted({s for s in gene_ids[in_cluster] if s and s != "nan"})
            if not study:
                continue
            res = _run(study)
            if res is None or res.empty:
                continue
            res = res.rename(columns=_RENAME)
            if "pAdj" in res:
                res = res[res["pAdj"] <= p_value]
            if res.empty:
                continue
            res.insert(0, "cluster", int(c))
            frames.append(res)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            sort_cols = [c for c in ["cluster", "pAdj"] if c in combined]
            results[str(g)] = combined.sort_values(sort_cols).reset_index(drop=True)

    out.uns[key_added] = results
    return out if copy else None
