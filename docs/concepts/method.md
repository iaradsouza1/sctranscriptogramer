# The transcriptogram method

## The ordering

The method rests on a **fixed 1-D ordering of proteins**, computed once (externally) from the STRING
protein–protein-interaction network so that proteins likely to participate in the same pathway sit
near each other on the line. Position `0..N-1` along this ordering is a coordinate on the "systems
axis" of the proteome.

`sctranscriptogramer` ships these orderings (human, mouse, rat, yeast at STRING score thresholds
700/800/900) and reuses them as-is — it does **not** recompute the ordering.

```python
import sctranscriptogramer as sctg
ordering = sctg.datasets.human_ordering(threshold=900)   # 12,396 ordered proteins
```

## The transcriptogram

For a given sample (or metacell, or spot), expression is:

1. **projected** onto the ordering — each protein gets the (averaged) expression of the genes mapping
   to it (`pp.project_to_ordering`); and
2. **windowed** — a circular sliding window of radius `r` replaces each position with the average of
   its neighbourhood (`tl.transcriptogram`).

The windowing is a 1-D boxcar convolution along the ordering. Because neighbours share pathways, it
denoises dropout and turns per-gene noise into a smooth **systems profile**. Contiguous stretches of
the ordering that move together are functional systems.

## Two representations, two purposes

`tl.transcriptogram` produces two aligned matrices per unit:

- **`X` / `layers['transcriptogram']`** — a masked window **mean** of (log2-CPM) expression: the
  *embedding*, used for visualisation, clustering, and the descriptive views.
- **`layers['window_counts']`** — a window **sum of raw counts** (a "pathway pseudo-count"): the
  integer response fed to the negative-binomial models, with the unit's total counts as the offset.

## Why single-cell needs adapting

The original method was built for **bulk** data, where a handful of samples are exchangeable
biological replicates. Single-cell data breaks that in three ways, which shape the whole design:

1. **Scale.** Millions of cells. Solved by aggregating into **metacells** (`metacells2`) or spatial
   neighbourhoods *before* the transcriptogram, and by a vectorized (cumulative-sum) windowing.
2. **Replication.** The biological replicate is the **donor**, not the cell or metacell — there are
   many correlated metacells per donor. Handled by a **donor random effect** in the statistical
   models (and by building metacells pure in donor).
3. **Estimation.** A descriptive *difference of means* over metacells is a biased, donor-imbalanced
   quantity; the model **`logFC`** (donor-aware, count-based) is the right effect size. See
   [Statistics & caveats](statistics.md).

## The systems analyses

Once each unit has a transcriptogram, four analyses ask different questions of it — all writing
per-position results and grouping significant positions into **regions** (the unit of inference,
since windows overlap). See [the pipeline](pipeline.md) and the
[quickstarts](../quickstart/differential-systems.ipynb).
