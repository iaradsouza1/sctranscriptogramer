# Installation

## Requirements

- Python ≥ 3.10
- The core stack (`numpy`, `scipy`, `pandas`, `anndata`, `pyarrow`) is installed automatically.

## Install

```bash
pip install sctranscriptogramer
```

The heavier, phase-specific dependencies are grouped into optional extras so the core stays light:

| Extra | Installs | Needed for |
|---|---|---|
| `sc` | `scanpy`, `metacells` | metacell aggregation ([`pp.aggregate_metacells`][sctranscriptogramer.pp.aggregate_metacells]) |
| `de` | `edgepython` | differential / divergent systems (NB mixed model) |
| `spatial` | `squidpy` | spatial datasets & plotting (the analysis itself needs no squidpy) |
| `enrich` | `gseapy` | GO/pathway enrichment |
| `ids` | `pybiomart`, `mygene` | ID mapping ([`map_to_protein`][sctranscriptogramer.map_to_protein]) |
| `plot` | `matplotlib` | plotting |
| `all` | everything above | the full workflow |

```bash
# everything for the full single-cell + spatial workflow
pip install "sctranscriptogramer[all]"
```

!!! note "edgePython"
    The differential and divergent tests require `edgepython` (the `de` extra), which provides the
    NEBULA-LN negative-binomial mixed model. The embedding's TMM normalization also uses it when
    available, falling back to plain log2-CPM otherwise.

## Development

```bash
git clone https://github.com/iaradsouza1/sctranscriptogramer
cd sctranscriptogramer
pip install -e ".[all,dev]"
pytest
```
