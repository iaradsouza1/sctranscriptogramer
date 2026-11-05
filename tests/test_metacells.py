"""Tests for pp.aggregate_metacells -- especially the group-purity guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import sctranscriptogramer as sctg


@pytest.fixture
def cells() -> AnnData:
    # 8 cells x 4 genes, two conditions x two cell types x two donors.
    rng = np.random.default_rng(0)
    counts = rng.poisson(5.0, size=(8, 4)).astype(np.int64)
    obs = pd.DataFrame(
        {
            "condition": ["case", "case", "case", "case", "ctrl", "ctrl", "ctrl", "ctrl"],
            "cell_type": ["T", "T", "B", "B", "T", "T", "B", "B"],
            "donor": ["d1", "d2", "d1", "d2", "d3", "d4", "d3", "d4"],
        },
        index=[f"c{i}" for i in range(8)],
    )
    ad = AnnData(X=counts.astype(float), obs=obs, var=pd.DataFrame(index=list("gGhH")))
    return ad


# --------------------------------------------------------------------------- pseudobulk / purity


def test_pseudobulk_one_metacell_per_group(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition", "cell_type"])
    # 2 conditions x 2 cell types = 4 pure metacells
    assert mc.n_obs == 4
    combos = set(map(tuple, mc.obs[["condition", "cell_type"]].to_numpy()))
    assert combos == {("case", "T"), ("case", "B"), ("ctrl", "T"), ("ctrl", "B")}
    assert mc.obs["n_cells"].sum() == 8


def test_metacells_never_mix_condition(cells: AnnData) -> None:
    # Even with a precomputed assignment that crosses conditions, purity must hold.
    cells.obs["ext_mc"] = ["m0", "m0", "m0", "m0", "m0", "m0", "m0", "m0"]  # all same label
    mc = sctg.pp.aggregate_metacells(
        cells, group_keys=["condition"], assignment_key="ext_mc"
    )
    # ext label is identical for all, but condition splits it into exactly 2 metacells.
    assert mc.n_obs == 2
    assert set(mc.obs["condition"]) == {"case", "ctrl"}


def test_counts_are_summed_within_metacell(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"])
    # case = first 4 cells, ctrl = last 4
    case_sum = cells.X[:4].sum(axis=0)
    ctrl_sum = cells.X[4:].sum(axis=0)
    row = {c: i for i, c in enumerate(mc.obs["condition"])}
    np.testing.assert_array_equal(mc.layers["counts"][row["case"]], case_sum)
    np.testing.assert_array_equal(mc.layers["counts"][row["ctrl"]], ctrl_sum)
    np.testing.assert_array_equal(
        mc.obs["total_counts"].to_numpy()[[row["case"], row["ctrl"]]],
        [case_sum.sum(), ctrl_sum.sum()],
    )


def test_no_group_keys_single_metacell(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(cells)
    assert mc.n_obs == 1
    assert mc.obs["n_cells"].iloc[0] == 8


def test_assignment_key_subdivides_within_group(cells: AnnData) -> None:
    cells.obs["sub"] = ["a", "b", "a", "b", "a", "b", "a", "b"]
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"], assignment_key="sub")
    # 2 conditions x 2 sub-labels = 4
    assert mc.n_obs == 4


# --------------------------------------------------------------------------- engine hook


def test_callable_engine_partitionwise(cells: AnnData) -> None:
    # Dummy engine: split each partition's cells into 2 metacells by row parity.
    def parity_engine(sub: AnnData) -> np.ndarray:
        return (np.arange(sub.n_obs) % 2).astype(np.int64)

    mc = sctg.pp.aggregate_metacells(
        cells, group_keys=["condition", "cell_type"], engine=parity_engine
    )
    # 4 partitions x up to 2 labels; each partition has 2 cells -> 2 metacells each = 8, but
    # every partition has exactly 2 cells (one even, one odd) -> 2 metacells per partition.
    assert mc.n_obs == 8
    assert (mc.obs["n_cells"] == 1).all()


def test_engine_outliers_dropped(cells: AnnData) -> None:
    def drop_first(sub: AnnData) -> np.ndarray:
        lab = np.zeros(sub.n_obs, dtype=np.int64)
        lab[0] = -1  # mark first cell of each partition as outlier
        return lab

    with pytest.warns(UserWarning, match="outlier"):
        mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"], engine=drop_first)
    # 4 case + 4 ctrl, drop 1 each -> 3 + 3 cells across 2 metacells
    assert mc.obs["n_cells"].sum() == 6


def test_engine_gets_raw_counts(cells: AnnData) -> None:
    # Put lognorm in X, counts in a layer; engine must receive counts, not X.
    cells.layers["counts"] = cells.X.astype(np.int64)
    cells.X = np.log1p(cells.X)
    seen = {}

    def spy(sub: AnnData) -> np.ndarray:
        seen["max"] = float(np.asarray(sub.X.todense() if sp.issparse(sub.X) else sub.X).max())
        return np.zeros(sub.n_obs, dtype=np.int64)

    sctg.pp.aggregate_metacells(cells, group_keys=["condition"], engine=spy, counts_layer="counts")
    assert seen["max"] > 5  # raw counts, not log1p (which would be < 5)


# --------------------------------------------------------------------------- normalization / misc


def test_default_normalization_is_tmm(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"])
    assert mc.uns["sctg_metacells"]["normalization"] == "tmm_log2cpm"
    assert np.isfinite(mc.X).all()


def test_log2cpm_matches_edger_formula(cells: AnnData) -> None:
    prior = 2.0
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"], normalization="log2cpm")
    assert mc.uns["sctg_metacells"]["normalization"] == "log2cpm"
    assert mc.uns["sctg_metacells"]["prior_count"] == prior

    counts = mc.layers["counts"].astype(float)
    lib = counts.sum(axis=1)
    prior_scaled = prior * lib / lib.mean()
    adj = lib + 2 * prior_scaled
    expected = np.log2((counts + prior_scaled[:, None]) / adj[:, None] * 1e6)
    np.testing.assert_allclose(mc.X, expected)


def test_tmm_reduces_to_log2cpm_with_uniform_composition() -> None:
    # If every metacell shares the same gene composition, TMM factors are ~1, so
    # tmm_log2cpm and log2cpm coincide.
    pytest.importorskip("edgepython")
    profile = np.array([50, 30, 15, 5], dtype=np.int64)
    counts = np.vstack([profile * k for k in (1, 2, 3, 4, 5, 6)])  # proportional -> nf ~ 1
    obs = pd.DataFrame(
        {"condition": ["a", "a", "a", "b", "b", "b"]}, index=[f"c{i}" for i in range(6)]
    )
    ad = AnnData(X=counts.astype(float), obs=obs, var=pd.DataFrame(index=list("wxyz")))
    tmm = sctg.pp.aggregate_metacells(ad, group_keys=["condition"], normalization="tmm_log2cpm")
    log2 = sctg.pp.aggregate_metacells(ad, group_keys=["condition"], normalization="log2cpm")
    np.testing.assert_allclose(tmm.X, log2.X, rtol=1e-3, atol=1e-3)


def test_log1p_normalization_option(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(
        cells, group_keys=["condition"], normalization="log1p", target_sum=1e4
    )
    approx_total = np.expm1(mc.X).sum(axis=1)
    np.testing.assert_allclose(approx_total, 1e4, rtol=1e-4)


def test_normalize_false_keeps_counts(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"], normalize=False)
    np.testing.assert_array_equal(mc.X, mc.layers["counts"])


def test_unknown_normalization_raises(cells: AnnData) -> None:
    with pytest.raises(ValueError, match="unknown normalization"):
        sctg.pp.aggregate_metacells(cells, group_keys=["condition"], normalization="bogus")


def test_carry_obs_nonconstant_warns(cells: AnnData) -> None:
    with pytest.warns(UserWarning, match="not constant"):
        # donor is not constant within a condition-only metacell
        sctg.pp.aggregate_metacells(cells, group_keys=["condition"], carry_obs=["donor"])


def test_carry_obs_numeric_is_averaged(cells: AnnData) -> None:
    # A continuous, cell-varying covariate should be averaged within each metacell (not moded).
    cells.obs["pct_mt"] = np.arange(cells.n_obs, dtype=float)
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"], carry_obs=["pct_mt"])
    expected = cells.obs.groupby("condition", observed=True)["pct_mt"].mean()
    got = mc.obs.set_index("condition")["pct_mt"]
    for cond in expected.index:
        np.testing.assert_allclose(got[cond], expected[cond])


def test_missing_column_raises(cells: AnnData) -> None:
    with pytest.raises(KeyError, match="nope"):
        sctg.pp.aggregate_metacells(cells, group_keys=["nope"])


def test_sparse_input(cells: AnnData) -> None:
    cells.X = sp.csr_matrix(cells.X)
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition"])
    assert mc.n_obs == 2


def test_uns_records_provenance(cells: AnnData) -> None:
    mc = sctg.pp.aggregate_metacells(cells, group_keys=["condition", "cell_type"])
    info = mc.uns["sctg_metacells"]
    assert info["group_keys"] == ["condition", "cell_type"]
    assert info["n_metacells"] == 4
    assert info["n_input_cells"] == 8
