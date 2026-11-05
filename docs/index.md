# sctranscriptogramer

**System-level analysis for single-cell, single-nucleus, and spatial transcriptomics — via transcriptograms.**

A Python port and scale-up of the R/Bioconductor
[`transcriptogramer`](https://academic.oup.com/bioinformatics/article/35/16/2875/5280729) method,
built natively for the scanpy / squidpy / AnnData ecosystem.

## What is a transcriptogram?

Genes are projected onto a **fixed 1-D ordering of proteins** derived from the STRING
protein–protein-interaction network — arranged so that neighbours are likely to share pathways. A
**sliding window** along that ordering smooths expression, turning scattered per-gene signal into
contiguous functional **systems**. Instead of asking "which genes changed?", you ask "which
*systems* changed, and where on the proteome?".

Every position on the ordering is interpretable (it maps to a specific protein/pathway), so the
results — differential, divergent, variable, or spatial — always come with a gene list.

## What it does

Starting from raw counts, the package builds a per-unit transcriptogram and offers four "systems"
analyses:

| Analysis | Function | Question |
|---|---|---|
| **Differential** | [`tl.differential_systems`][sctranscriptogramer.tl.differential_systems] | Which systems differ between **two conditions** (case vs control), per cell type? |
| **Divergent** | [`tl.divergent_systems`][sctranscriptogramer.tl.divergent_systems] | Which systems differ across the **levels of a factor** (e.g. cell state)? |
| **Highly variable** | [`tl.highly_variable_systems`][sctranscriptogramer.tl.highly_variable_systems] | Which systems vary across metacells **more than expected** (unsupervised)? |
| **Spatial** | [`tl.spatially_variable_systems`][sctranscriptogramer.tl.spatially_variable_systems] | Which systems are **spatially organised** across a tissue? |

The differential and divergent tests use a **negative-binomial mixed model** (edgePython / NEBULA-LN)
with a **donor random effect**, so metacells from the same donor are not treated as independent, and
support **covariate adjustment** — things the original bulk method could not do.

## Design in one paragraph

Millions of cells are first collapsed into **metacells** (`metacells2`) — pure in condition, cell
state, and donor by construction — or into spatial neighbourhoods. Each unit is projected onto the
ordering and windowed into a smooth systems profile. The result is a standard **AnnData** (units ×
positions) that flows straight into scanpy/squidpy and the analyses above.

## Next steps

- [Installation](installation.md)
- [The transcriptogram method](concepts/method.md) — the idea and why single-cell needs adapting
- [The pipeline](concepts/pipeline.md) — the end-to-end API
- [Quickstart notebooks](quickstart/differential-systems.ipynb) — worked examples on real data
- [API reference](api.md)
