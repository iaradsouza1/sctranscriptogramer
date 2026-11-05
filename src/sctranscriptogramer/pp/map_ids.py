"""Project a units x genes AnnData onto the protein ordering (the R ``transcriptogramStep1``).

Genes are mapped to proteins (optionally via a dictionary), dropped if their protein is not in the
ordering, and aggregated per ordering position: **mean** for the continuous embedding value, **sum**
for raw counts (the differential-expression path). The result is a new AnnData whose ``var`` axis is
the ordered positions ``0..N-1``.

Missing positions (ordered proteins with no measured gene) are kept in the dense layout and flagged
in a ``present_mask`` so the downstream masked boxcar ignores them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from sctranscriptogramer._core.ordering import Ordering

__all__ = ["project_to_ordering"]


def _as_matrix(x) -> np.ndarray | sp.spmatrix:
    """Return X as a 2D dense array or scipy sparse matrix (AnnData may hold either)."""
    if sp.issparse(x):
        return x.tocsr()
    return np.asarray(x)


def _row_totals(x) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.sum(axis=1)).ravel()
    return np.asarray(x).sum(axis=1)


def _resolve_pairs(
    identifiers: np.ndarray,
    ordering: Ordering,
    dictionary: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gene_row_index, position) pairs linking a var row to an ordering position.

    Without a dictionary, ``identifiers`` are treated as protein ids directly. With a dictionary
    (columns ``protein``, ``identifier``), each var row expands to every protein it maps to.
    """
    if dictionary is None:
        pos = ordering.positions_of(identifiers)
        keep = np.flatnonzero(pos >= 0)
        return keep, pos[keep]

    if not {"protein", "identifier"}.issubset(dictionary.columns):
        raise KeyError("dictionary must have columns 'protein' and 'identifier'")

    row_of_identifier: dict[str, list[int]] = {}
    for i, ident in enumerate(identifiers):
        row_of_identifier.setdefault(str(ident), []).append(i)

    gene_rows: list[int] = []
    proteins: list[str] = []
    for ident, protein in zip(
        dictionary["identifier"].astype(str), dictionary["protein"].astype(str)
    ):
        rows = row_of_identifier.get(ident)
        if rows is None:
            continue
        for r in rows:
            gene_rows.append(r)
            proteins.append(protein)

    if not gene_rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    pos = ordering.positions_of(proteins)
    keep = np.flatnonzero(pos >= 0)
    return np.asarray(gene_rows, dtype=np.int64)[keep], pos[keep]


def project_to_ordering(
    adata: AnnData,
    ordering: Ordering,
    *,
    dictionary: pd.DataFrame | None = None,
    var_id_col: str | None = None,
    expr_layer: str | None = None,
    counts_layer: str | None = None,
) -> AnnData:
    """Map a units x genes AnnData onto the protein ordering.

    Parameters
    ----------
    adata
        Units (metacells/spots/samples) x genes AnnData.
    ordering
        The protein :class:`~sctranscriptogramer._core.ordering.Ordering`.
    dictionary
        Optional gene->protein map with columns ``protein`` and ``identifier``. If omitted,
        ``adata`` var identifiers are assumed to be protein ids already.
    var_id_col
        Column in ``adata.var`` holding the identifiers to map. Defaults to ``var_names``.
    expr_layer
        Layer holding the continuous expression used for the embedding (mean-aggregated per
        position). Defaults to ``adata.X``.
    counts_layer
        Layer holding raw integer counts for the DE path (sum-aggregated per position). If given,
        the result carries a ``counts`` layer and ``obs['total_counts']`` library sizes.

    Returns
    -------
    AnnData
        Transcriptogram AnnData with ``obs`` = units and ``var`` = ordered positions. ``X`` and
        ``layers['step1']`` hold the mean-aggregated expression; ``uns['sctg']`` records the
        ordering, the boolean ``present_mask`` over positions, and per-position gene multiplicity.
    """
    identifiers = (
        adata.var[var_id_col].to_numpy().astype(str)
        if var_id_col is not None
        else np.asarray(adata.var_names, dtype=str)
    )

    gene_rows, positions = _resolve_pairs(identifiers, ordering, dictionary)
    if gene_rows.size == 0:
        raise ValueError(
            "no genes could be mapped onto the ordering; check identifiers / dictionary / species"
        )

    n_units = adata.n_obs
    n_genes = adata.n_vars
    n_pos = ordering.n

    # Incidence S (genes x positions): 1 where a gene maps to a position.
    incidence = sp.csr_matrix(
        (np.ones(gene_rows.size), (gene_rows, positions)),
        shape=(n_genes, n_pos),
    )
    genes_per_pos = np.asarray(incidence.sum(axis=0)).ravel()  # length n_pos
    present_mask = genes_per_pos > 0

    expr_src = adata.layers[expr_layer] if expr_layer is not None else adata.X
    expr = _as_matrix(expr_src)
    # units x positions summed over mapped genes, then mean over gene multiplicity.
    expr_sum = expr @ incidence  # (units x positions)
    expr_sum = np.asarray(expr_sum.todense()) if sp.issparse(expr_sum) else np.asarray(expr_sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        expr_mean = np.zeros_like(expr_sum, dtype=np.float64)
        np.divide(expr_sum, genes_per_pos, out=expr_mean, where=present_mask)

    var = pd.DataFrame(
        {
            "position": ordering.positions,
            "protein": ordering.proteins.astype(str),
            "n_genes": genes_per_pos.astype(np.int64),
        }
    )
    var.index = ordering.positions.astype(str)

    tg = AnnData(
        X=expr_mean,
        obs=adata.obs.copy(),
        var=var,
    )
    tg.layers["step1"] = expr_mean.copy()

    # Carry per-unit embeddings (e.g. obsm['spatial'] coordinates) through to the transcriptogram,
    # so spatial-association analyses can run directly on tg.
    for key, value in adata.obsm.items():
        tg.obsm[key] = value.copy() if hasattr(value, "copy") else value

    if counts_layer is not None:
        counts = _as_matrix(adata.layers[counts_layer])
        counts_sum = counts @ incidence
        counts_sum = (
            np.asarray(counts_sum.todense()) if sp.issparse(counts_sum) else np.asarray(counts_sum)
        )
        tg.layers["counts"] = counts_sum
        tg.obs["total_counts"] = _row_totals(counts)

    tg.uns["sctg"] = {
        "ordering_name": ordering.name,
        "species": ordering.species,
        "threshold": ordering.threshold,
        "n_positions": n_pos,
        "present_mask": present_mask,
        "n_mapped_positions": int(present_mask.sum()),
        "has_counts": counts_layer is not None,
    }
    return tg
