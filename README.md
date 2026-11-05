# sctranscriptogramer

System-level differential expression for **single-cell, single-nucleus, and spatial
transcriptomics**, via *transcriptograms*.

A Python port and scale-up of the R/Bioconductor
[`transcriptogramer`](https://academic.oup.com/bioinformatics/article/35/16/2875/5280729)
method, built natively for the scanpy / squidpy / AnnData ecosystem.

## The idea

The transcriptogram projects expression onto a **fixed 1D ordering of proteins** derived from
the STRING protein–protein-interaction network — arranged so that neighbours are likely to share
pathways. A **sliding window** along the ordering smooths expression, turning scattered
per-gene signal into contiguous functional *systems*. Comparing conditions per window reveals which
systems are up- or down-regulated, rather than isolated genes.

This package adapts that method to modern single-cell scale:

- **Aggregation first.** Millions of cells are collapsed into **metacells** (`metacells2`) or spatial
  neighbourhoods (`squidpy`) before the transcriptogram runs.
- **Vectorized windowing.** The sliding window is a circular boxcar computed with cumulative sums —
  O(N · units) instead of the R implementation's per-position loop.
- **Two outputs.** (A) a per-unit **system-level embedding** for clustering / UMAP / trajectory / spatial
  features, and (B) **differential systems** via a negative-binomial mixed model
  (`edgePython`) with a donor random effect and covariates.

## Status

Pre-alpha, under active development. The core windowing engine and its tests come first
(see the project plan and `tests/`).

## Development

```bash
conda activate sctranscriptogramer
pip install -e ".[dev]"
pytest
```

## License

GPL-3.0-or-later (following the original R package).
