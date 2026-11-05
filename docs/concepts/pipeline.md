# The pipeline

The whole workflow is a chain of AnnData-in / AnnData-out steps. Here is the end-to-end shape; the
[quickstarts](../quickstart/differential-systems.ipynb) run each on real data.

```
raw counts (cells × genes, sparse)
   │  ① aggregate           pp.aggregate_metacells   (or pp.spatial neighbourhoods / per-spot)
   ▼
units × genes  (metacells / spots), pure in condition·state·donor
   │  ② map ids             map_to_protein            gene symbol / ENSG → ENSEMBL peptide
   │  ③ project             pp.project_to_ordering    onto the fixed ordering
   │  ④ window              tl.transcriptogram        → embedding X + window_counts
   ▼
transcriptogram AnnData  (units × positions)
   │  ⑤ analyse
   ├── tl.differential_systems     case vs control, per cell type
   ├── tl.divergent_systems        across a factor (cell state)
   ├── tl.highly_variable_systems  unsupervised
   ├── tl.spatially_variable_systems  spatial autocorrelation
   │  ⑥ interpret
   ├── tl.cluster_enrichment       GO / pathway ORA per region
   └── pl.difference_of_means / pl.enrichment_rate
```

## ① Aggregate into metacells

Cells are partitioned by the grouping columns **before** metacells are built, so a metacell never
mixes conditions, cell states, or donors — the last is what lets the models use donor as a random
effect.

```python
mc = sctg.pp.aggregate_metacells(
    adata,
    group_keys=["condition", "cell_state", "donor"],   # metacells are pure in all three
    engine="metacells2",
    counts_layer="counts",
    carry_obs=["sex", "age", "percent_mt"],            # covariates for later adjustment
    engine_kwargs={"target_metacell_size": 12},
)
```

!!! tip "Choosing the metacell size"
    `target_metacell_size` (cells per metacell) is a **power** decision, not a homogeneity one
    (metacells2 handles homogeneity itself): metacells need enough UMIs for stable pseudo-counts, yet
    be small enough to leave **several metacells per (donor × group) partition** for the mixed model.
    [`pp.suggest_metacell_size`][sctranscriptogramer.pp.suggest_metacell_size] reads the partition
    sizes and depth and recommends a value (no metacells built):

    ```python
    sctg.pp.suggest_metacell_size(adata, group_keys=["condition", "cell_state", "donor"],
                                  counts_layer="counts")["target_metacell_size"]
    ```
    metacells2's default (48) is often too coarse for donor-resolved single-cell DE — it can leave
    only one metacell per partition, removing within-donor replication.

## ② Map identifiers to the ordering

The ordering is keyed by ENSEMBL peptide IDs; expression is usually gene symbols or ENSG.

```python
dictionary = sctg.map_to_protein(mc.var_names, id_type="symbol", species="human",
                                 cache_path="symbol2ensp.parquet")
```

## ③–④ Project and window

```python
ordering = sctg.datasets.human_ordering(900)
tg = sctg.pp.project_to_ordering(mc, ordering, dictionary=dictionary, counts_layer="counts")
sctg.tl.transcriptogram(tg)   # radius defaults to 80; writes X (embedding) + layers['window_counts']
```

`tg` is now a units × positions AnnData carrying `obs` metadata (and `obsm['spatial']` for spatial
data). Everything below reads and writes standard AnnData slots — see
[Accessing results](data-access.md).

!!! tip "Choosing the radius"
    The window `radius` is the one free parameter of the transform. [`tl.select_radius`][sctranscriptogramer.tl.select_radius]
    scans candidate radii and picks the one that maximises the **null-corrected reproducibility** of
    the per-position signal across independent halves of the data — the bias–variance sweet spot
    (see [Statistics & caveats](statistics.md#choosing-the-radius)). It re-windows the stored
    `step1`, so you can run it right after projecting:

    ```python
    df = sctg.tl.select_radius(tg, group_key="cell_state", donor_key="donor")
    sctg.tl.transcriptogram(tg, radius=df.attrs["best_radius"])
    ```

## ⑤ Analyse

```python
# case vs control, per cell type, adjusted for covariates, donor random effect.
# Default inference: per-window NB test -> BH FDR -> cluster the significant windows
# (the transcriptogramer procedure, adapted to counts). inference="permutation" is an
# optional stricter region-level test for few-donor cohorts (see Statistics).
sctg.tl.differential_systems(tg, condition_key="condition", case="disease", control="healthy",
                             donor_key="donor", group_key="cell_state",
                             covariates=["sex", "age"])

# systems that differ across cell states (multi-group omnibus)
sctg.tl.divergent_systems(tg, group_key="cell_state", donor_key="donor")

# unsupervised: which systems vary across metacells beyond the mean–variance trend
sctg.tl.highly_variable_systems(tg)

# spatial: which systems are spatially organised (top decile of Moran's I)
sctg.tl.spatially_variable_systems(tg, top_quantile=0.90)
```

## ⑥ Interpret

```python
sctg.tl.cluster_enrichment(tg, gene_sets="GO_Biological_Process_2023",
                           gene_symbol_col="symbol", online=True)
sctg.pl.difference_of_means(tg, contrast="Microglia")   # plots the model logFC by default
```
