# Statistics & caveats

The statistical design is what makes the method valid on single-cell data. A few points are worth
understanding — several were learned the hard way on real data.

## Donor is the replicate, not the metacell

Metacells from the same donor are **pseudoreplicates** — highly correlated. Averaging or testing over
metacells as if they were independent dramatically inflates significance. Two safeguards:

1. **Purity by construction.** `aggregate_metacells` partitions cells by `group_keys` *before*
   building metacells, so each metacell belongs to exactly one donor (and one condition, one cell
   state).
2. **Donor random effect.** `differential_systems` and `divergent_systems` fit an NB gamma mixed
   model (edgePython / NEBULA-LN) with `donor` as a random effect, absorbing within-donor
   correlation. They require ≥ 2 donors, and for `divergent_systems` the factor must vary *within*
   donors (be crossed with donor), else it is confounded with the random effect.

## Plot the `logFC`, not the difference of means

The original bulk plot shows `mean(case) − mean(control)` of the smoothed values. On metacells this
is the **wrong object**, for two independent reasons:

- **Pseudoreplication / weighting.** Metacells are unequal-sized and unevenly distributed across
  donors, so an unweighted mean over metacells is a donor-imbalanced comparison.
- **A constant compositional offset.** A difference of *mean log2-CPM over a gene subset* carries a
  near-constant baseline offset driven by low-end dropout, which per-sample normalisation (CPM/TMM)
  cannot remove — so the descriptive curve "leans" even when there is no global effect.

The model **`logFC`** (donor-aware, TMM-normalised, count-based) has neither problem: it is centred
and is the exact quantity the significance is computed on. `pl.difference_of_means` therefore plots
`logFC` by default; `values="difference"` is kept only as a bulk-style descriptive view.

## Significance vs effect size

With many units, tests become trivially significant:

- In structured **tissue**, almost every system is significantly spatially autocorrelated
  (Moran's I ≫ 0 everywhere). `spatially_variable_systems` therefore offers `top_quantile` /
  `min_moran` to rank by **magnitude** and surface the *most* organised systems.
- `highly_variable_systems` likewise selects by biological-variance quantile, not a p-value.

Read effect sizes (`logFC`, `max_abs_logFC`, `moran_I`, biological variance) alongside adjusted p.

## Multi-group omnibus via ACAT

edgePython's mixed model provides single-coefficient Wald tests only. For a multi-level factor,
`divergent_systems` runs the `k−1` level tests and combines them with **ACAT** (Cauchy combination)
into one omnibus "does any level differ" p-value — valid under the global null without needing the
coefficients' covariance.

## Regions are the unit of inference

Positions are **overlapping windows**, so adjacent positions are strongly autocorrelated and
per-position p-values are not independent. At radius 80 the effective number of independent tests is
roughly the number of non-overlapping windows (`N / (2r+1)` ≈ 70 for the human ordering), not the
~12,000 positions. Two consequences:

- **BH stays valid but the count misleads.** Positively-correlated statistics satisfy the PRDS
  condition, so per-position BH still controls the FDR — but the number of significant *positions*
  is not the number of discoveries (one true system lights up a whole window). Report **regions**.
- **Per-position FDR ≠ region-level FDR.** Thresholding per position and then forming regions is a
  cluster-forming procedure; controlling per-position FDR does not control the fraction of reported
  *regions* that are spurious.

### Region-level permutation inference

`differential_systems(..., inference="permutation")` closes both gaps (and the few-donor
miscalibration) with a **cluster-mass permutation test**:

1. Regions are the contiguous positions passing a fixed **cluster-forming threshold**
   (`cluster_forming_p`, default 0.01), each scored by its **mass** = Σ `-log10(p)`.
2. Labels are permuted **at the donor level**; the model is refit; the **maximum** cluster mass per
   permutation forms the null.
3. A region's `region_pvalue` = fraction of permutations whose max mass ≥ its mass — an
   **FWER-controlled** region-level p that absorbs the window correlation, the cluster-forming step,
   and the small-donor inflation.

Permutations are **exhaustive** when the donor relabelings are few (`C(n_donors, n_case) - 1 ≤
max_exact`) — so few-donor cohorts get an exact, deterministic null for free — and random otherwise.
Each permutation is one model refit (~5 s at `perm_n_jobs=4`); with `K` permutations the smallest
reachable region p is `1/(K+1)`, and with few donors `K` is capped at `C(n_donors, n_case)`, so the
null is inherently coarse there. Results land in
`uns['diffsystems']['contrasts'][group]['permutation']`.

Use permutation inference when donors are few or p-values are borderline; the asymptotic BH default is
fine for a fast first pass and is approximately calibrated at q=0.05 (see below).

## Calibration of the asymptotic tests

Under a **donor-level permutation null** on a signal-free cell type, the differential test's false-
positive rate is close to nominal even with ~8 donors (`p<0.05` ≈ 0.058), with only mild
anti-conservativeness in the far tail (`p<0.01` ≈ 0.017). So BH discovery at q=0.05 is broadly honest;
treat very small p-values with mild caution, and switch to `inference="permutation"` when donors are
few or calls are borderline. (Permuting labels on a cell type that *has* real signal overstates the
false-positive rate — the real effect leaks into near-truth permutations — so calibrate on a
no-effect subset.) This should tighten further at larger donor counts.

## Choosing the radius

The window `radius` is a smoothing bandwidth: too small is noisy, too large blurs distinct systems.
[`tl.select_radius`][sctranscriptogramer.tl.select_radius] chooses it by **null-corrected split-half
reproducibility**: for each candidate radius it splits the units into two independent halves (over
donors when available), computes the per-position between-group signal (an F statistic) in each half,
and correlates them — then subtracts the same quantity under **shuffled group labels**. That
subtraction is essential: smoothing trivially inflates the correlation of *any* two vectors, and the
shuffled-label null grows the same way, so the difference isolates real, reproducible structure.

The radius maximising this score is the sweet spot. In practice the curve is **broad** — on the
microglia data any radius in roughly 60–160 is near-optimal (the argmax was ~120, and the classic
choice of 80 is essentially tied), so the result is not sensitive to the exact value. The `one_se`
rule picks the smallest radius within one standard error of the maximum, favouring resolution.

Note that an *intrinsic* criterion — the autocorrelation length of the mean profile along the
ordering — does **not** work here: that profile is near-white (the ordering groups genes by PPI, not
by expression level), so it gives no scale. Reproducibility of the *signal* is the criterion that has
a genuine optimum.

## Normalisation

- **Embedding**: `aggregate_metacells` defaults to `tmm_log2cpm` — edgeR-style `log2(CPM + prior)` on
  **TMM effective library sizes**, a composition-robust scale (falls back to plain `log2cpm` if
  edgePython is absent).
- **Differential/divergent tests**: TMM on the window pseudo-counts, with the unit's total counts as
  the offset — so the reported `logFC` is in log2-CPM units.
