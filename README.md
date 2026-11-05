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

## Installation

Requires Python ≥ 3.10. The package is not on PyPI yet, so install it from GitHub.

We recommend a fresh environment:

```bash
conda create -n sctranscriptogramer python=3.11
conda activate sctranscriptogramer
```

Install the full single-cell + spatial workflow:

```bash
pip install "sctranscriptogramer[all] @ git+https://github.com/iaradsouza1/sctranscriptogramer.git"
```

Or install only the core (windowing, orderings, AnnData I/O) and add the extras you need:

```bash
pip install "git+https://github.com/iaradsouza1/sctranscriptogramer.git"
```

| Extra | Installs | Needed for |
|---|---|---|
| `sc` | `scanpy`, `metacells` | metacell aggregation (`pp.aggregate_metacells`) |
| `de` | `edgepython` | differential / divergent systems (NB mixed model), TMM normalization |
| `spatial` | `squidpy` | spatial datasets and graphs (the spatial analyses themselves don't need it) |
| `enrich` | `gseapy` | GO / pathway enrichment (`tl.cluster_enrichment`) |
| `ids` | `pybiomart`, `mygene` | gene ID → protein mapping (`map_to_protein`) |
| `plot` | `matplotlib` | plotting (`pl.*`) |
| `all` | everything above | the full workflow |

For example, `pip install "sctranscriptogramer[sc,de] @ git+https://github.com/iaradsouza1/sctranscriptogramer.git"`.

The STRING protein orderings (human, mouse, rat, yeast at scores 700/800/900) ship with the package, so
no extra download is needed. Check the install:

```python
import sctranscriptogramer as sctg
sctg.datasets.human_ordering(900)   # 12,396 ordered proteins
```

Example datasets used in the quickstart notebooks are not included in the repository.

## Development

```bash
git clone https://github.com/iaradsouza1/sctranscriptogramer.git
cd sctranscriptogramer
conda create -n sctranscriptogramer python=3.11
conda activate sctranscriptogramer
pip install -e ".[all,dev]"
pytest
```

## License

GPL-3.0-or-later (following the original R package).
