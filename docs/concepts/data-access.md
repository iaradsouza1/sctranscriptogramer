# Accessing results (AnnData)

`sctranscriptogramer` stores everything in the **standard AnnData structure** of the transcriptogram
object `tg` — there is no custom result object to learn. Read results with ordinary
`anndata` / `pandas` idioms.

## The transcriptogram object `tg`

`tg` has **obs = units** (metacells / spots / samples) and **var = ordered positions**.

```python
tg.X                       # smoothed embedding (log2-CPM), == tg.layers["transcriptogram"]
tg.obs_names               # unit ids (rows)
tg.obsm["spatial"]         # unit coordinates (spatial data), carried through the pipeline
```

### The ordered positions (`tg.var`)

```python
tg.var["position"]         # 0-based position along the ordering
tg.var["protein"]          # ENSEMBL peptide id at each position
tg.var["symbol"]           # gene symbol (if you added it)
```

### Layers

| Layer | Contents |
|---|---|
| `layers["transcriptogram"]` | smoothed embedding (= `X`) |
| `layers["step1"]` | per-protein expression, pre-smoothing |
| `layers["counts"]` | summed raw counts per position |
| `layers["window_counts"]` | windowed pseudo-counts (the NB-model response) |

```python
import pandas as pd
# transcriptogram as a labelled DataFrame (units × positions):
T = pd.DataFrame(tg.X, index=tg.obs_names, columns=tg.var["position"])
```

## Analysis results

Each `*_systems` function writes a **per-position table** to `varm` and metadata (regions, params)
to `uns`.

```python
# differential systems (one table per cell type)
de = tg.varm["diffsystems_Microglia"]          # logFC, pValue, pAdj, tested, is_de, cluster
tg.uns["diffsystems"]["contrasts"]["Microglia"]["clusters"]   # [(start, end), ...]
tg.uns["diffsystems"]["params"]                # case/control/donor/covariates used

# divergent systems (across a factor)
tg.varm["divergent_systems_cell_state"]        # omnibus_p, pAdj, max_abs_logFC, is_divergent, region
tg.uns["divergent_systems"]["regions"]

# highly variable systems (unsupervised) -> written to tg.var
tg.var["highly_variable"]                      # bool (also usable by sc.pp.pca)
tg.var["hvs_biological_variance"]
tg.uns["highly_variable_systems"]["regions"]

# spatially variable systems
tg.varm["spatially_variable_systems"]          # moran_I, pval, pAdj, is_spatial, region
tg.uns["spatially_variable_systems"]["regions"]

# enrichment
tg.uns["enrichment"]["Microglia"]              # GSEApy results DataFrame
```

## Useful idioms

```python
# genes of a differential cluster
de = tg.varm["diffsystems_Microglia"]
tg.var.loc[de["cluster"] == 3, "symbol"].tolist()

# subset the transcriptogram to a system region (positions 3800–4100)
tg[:, (tg.var["position"] >= 3800) & (tg.var["position"] <= 4100)]

# cluster metacells in systems space (highly_variable flags the features)
import scanpy as sc
sctg.tl.highly_variable_systems(tg)
sc.pp.pca(tg, mask_var="highly_variable"); sc.pp.neighbors(tg); sc.tl.leiden(tg)

# write a per-region spatial score back as an obs feature for squidpy
region = slice(10105, 10461)
tg.obs["oxphos_system"] = tg.layers["transcriptogram"][:, region].mean(1)
```

Everything round-trips through `tg.write_h5ad(...)`.
