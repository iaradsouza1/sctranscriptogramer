"""metacells2 engine wrapper.

Runs the Tanay-lab ``metacells`` (metacells2) divide-and-conquer pipeline on a single partition and
returns a per-cell metacell label (outliers -> -1). Imported lazily so ``metacells`` is only
required when this engine is actually selected.

Note: this wrapper is deliberately minimal -- it runs the standard pipeline with sensible defaults
and forwards extra keyword arguments. For production runs you will usually want to do gene/cell
cleaning (excluded/lateral/noisy genes) upstream and pass tuned parameters via ``engine_kwargs``.
"""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData

__all__ = ["metacells2_assign"]


def metacells2_assign(
    adata: AnnData,
    *,
    target_metacell_size: int | None = None,
    random_seed: int = 0,
    verbose: bool = False,
    lateral_gene_names: list[str] | None = None,
    noisy_gene_names: list[str] | None = None,
    **pipeline_kwargs: object,
) -> np.ndarray:
    """Assign each cell in ``adata`` to a metacell using metacells2.

    Parameters
    ----------
    adata
        Cells x genes AnnData whose ``X`` holds raw UMI counts for one partition.
    target_metacell_size
        Forwarded to metacells2 as ``target_metacell_size`` when given.
    random_seed
        Seed for reproducibility.
    **pipeline_kwargs
        Extra keyword arguments forwarded to ``mc.pl.divide_and_conquer_pipeline``.

    Returns
    -------
    numpy.ndarray
        Integer metacell label per cell; ``-1`` for outliers.
    """
    try:
        import metacells as mc
    except ImportError as exc:  # pragma: no cover - exercised only without metacells installed
        raise ImportError(
            "the 'metacells2' engine requires the `metacells` package; "
            "install it with `pip install 'sctranscriptogramer[sc]'` or `pip install metacells`"
        ) from exc

    if not verbose:
        logging.getLogger("metacells").setLevel(logging.WARNING)

    work = adata.copy()
    # metacells2 requires float32 UMIs (its downsampling asserts the dtype).
    if hasattr(work.X, "astype"):
        work.X = work.X.astype("float32")
    mc.ut.set_name(work, "partition")

    # Initialise the gene masks the divide-and-conquer pipeline expects. We mark genes only
    # (never drop cells) so the returned per-cell metacell labels stay aligned to the input.
    # The masks are set directly (all-False by default) rather than via mc.pl.exclude_genes,
    # whose bursty/lonely-gene detector is unstable on the small per-partition inputs here.
    import numpy as _np

    work.var["excluded_gene"] = _np.zeros(work.n_vars, dtype=bool)
    work.obs["excluded_umis"] = _np.zeros(work.n_obs, dtype="float32")
    mc.pl.mark_lateral_genes(work, lateral_gene_names=lateral_gene_names or [])
    mc.pl.mark_noisy_genes(work, noisy_gene_names=noisy_gene_names or [])

    if target_metacell_size is not None:
        pipeline_kwargs.setdefault("target_metacell_size", target_metacell_size)
    pipeline_kwargs.setdefault("random_seed", random_seed)

    mc.pl.divide_and_conquer_pipeline(work, **pipeline_kwargs)

    if "metacell" not in work.obs:
        raise RuntimeError("metacells2 did not produce an obs['metacell'] assignment")
    labels = work.obs["metacell"].to_numpy()
    return np.asarray(labels, dtype=np.int64)
