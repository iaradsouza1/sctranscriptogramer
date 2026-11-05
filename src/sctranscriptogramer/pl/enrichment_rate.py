"""GO-term-rate-along-the-ordering plot (the R ``enrichmentPlot``).

For each selected term, plots the fraction of proteins within the sliding window that are annotated
to the term, along the ordering -- reusing the same circular boxcar as the transcriptogram, applied
to a 0/1 annotation vector. Clusters are shaded.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from sctranscriptogramer._core.boxcar import circular_boxcar
from sctranscriptogramer._core.ordering import _strip_prefix
from sctranscriptogramer.pl._smooth import smooth_spline

__all__ = ["enrichment_rate"]


def _term2genes(gene_sets, terms: list[str], strip: bool) -> dict[str, set[str]]:
    """Return ``term -> set(gene id)`` for the requested terms (same format as cluster_enrichment).

    ``gene_sets`` is either a dict ``term -> [genes]`` or a DataFrame with protein/term columns.
    """
    norm = _strip_prefix if strip else (lambda x: x)
    wanted = set(terms)
    t2g: dict[str, set[str]] = defaultdict(set)
    if isinstance(gene_sets, Mapping):
        for term in terms:
            for g in gene_sets.get(term, ()):
                t2g[term].add(norm(str(g)))
        return t2g
    cols = {c.lower(): c for c in gene_sets.columns}
    pcol = cols.get("protein") or cols.get("ensembl_peptide_id") or gene_sets.columns[0]
    tcol = cols.get("go_id") or cols.get("term") or gene_sets.columns[1]
    for row in gene_sets.itertuples(index=False):
        term = str(getattr(row, tcol))
        if term in wanted:
            t2g[term].add(norm(str(getattr(row, pcol))))
    return t2g


def enrichment_rate(
    tg: AnnData,
    gene_sets,
    terms: list[str],
    *,
    term_labels: Mapping[str, str] | None = None,
    gene_symbol_col: str | None = None,
    strip_prefix: bool = True,
    radius: int | None = None,
    smooth: bool = True,
    spline_s: float | None = None,
    shade_clusters: bool = True,
    contrast: str | None = None,
    diffsystems_key: str = "diffsystems",
    ax: "plt.Axes | None" = None,
):
    """Plot per-term annotation rate along the ordering.

    Parameters
    ----------
    tg
        Transcriptogram AnnData with ``var['protein']`` and ``uns['sctg']['radius']``.
    gene_sets
        Term -> genes mapping, same format as :func:`~sctranscriptogramer.tl.cluster_enrichment`:
        a dict ``term -> [genes]`` or a DataFrame with protein/term columns.
    terms
        Term ids to plot.
    term_labels
        Optional term -> legend-label mapping.
    gene_symbol_col
        ``var`` column to use as gene ids instead of the protein ids.
    strip_prefix
        Strip taxonomy prefix from protein ids when matching annotations.
    radius
        Window radius; defaults to ``uns['sctg']['radius']``.
    smooth
        Overlay a smoothing spline (the raw rate is drawn faint).
    spline_s
        scipy smoothing factor.
    shade_clusters
        Shade the contrast's cluster intervals.
    contrast
        Contrast whose clusters to shade (default ``"all"``).
    diffsystems_key
        uns key holding the differential-systems results.
    ax
        Existing matplotlib Axes; created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if radius is None:
        radius = tg.uns.get("sctg", {}).get("radius")
        if radius is None:
            raise ValueError("radius not given and not found in uns['sctg']['radius']")
    radius = int(radius)

    if gene_symbol_col is not None:
        gene_ids = tg.var[gene_symbol_col].to_numpy().astype(str)
    else:
        gene_ids = tg.var["protein"].to_numpy().astype(str)
        if strip_prefix:
            gene_ids = np.array([_strip_prefix(g) for g in gene_ids])

    t2g = _term2genes(gene_sets, terms, strip_prefix and gene_symbol_col is None)
    positions = tg.var["position"].to_numpy() if "position" in tg.var else np.arange(len(gene_ids))

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    cmap = plt.get_cmap("tab10")

    for i, term in enumerate(terms):
        members = t2g.get(term, set())
        binary = np.array([1.0 if g in members else 0.0 for g in gene_ids])
        rate = circular_boxcar(binary[:, None], radius, mode="fixed").ravel()
        label = (term_labels or {}).get(term, term)
        color = cmap(i % 10)
        if smooth:
            ax.plot(positions, rate, color=color, lw=0.6, alpha=0.35)
            ax.plot(positions, smooth_spline(positions, rate, spline_s), color=color, lw=1.6,
                    label=label)
        else:
            ax.plot(positions, rate, color=color, lw=1.2, label=label)

    if shade_clusters:
        clusters = tg.uns.get(diffsystems_key, {}).get("contrasts", {}).get(
            str(contrast) if contrast is not None else "all", {}
        ).get("clusters", [])
        for lo, hi in clusters:
            ax.axvspan(lo, hi, color="0.5", alpha=0.12)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Gene position")
    ax.set_ylabel("Term rate by window")
    ax.set_title("Enrichment rate along the ordering")
    ax.legend(loc="upper right", fontsize=8)
    return ax
