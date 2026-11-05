"""Tests for the NB mixed-model adapter, BH correction, and differential_systems orchestration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg
from sctranscriptogramer.stats.multitest import benjamini_hochberg
from sctranscriptogramer.stats.nb_mixed import fit_nb_mixed

pytest.importorskip("edgepython")


# --------------------------------------------------------------------------- BH


def test_bh_basic() -> None:
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = benjamini_hochberg(p)
    # monotone non-decreasing, all <= 1
    assert np.all(np.diff(q) >= -1e-12)
    assert q.max() <= 1.0
    np.testing.assert_allclose(q[0], 0.05)  # 0.01 * 5/1


def test_bh_propagates_nan() -> None:
    p = np.array([0.01, np.nan, 0.5])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    assert np.isfinite(q[0]) and np.isfinite(q[2])


def test_bh_all_nan() -> None:
    q = benjamini_hochberg(np.array([np.nan, np.nan]))
    assert np.isnan(q).all()


# --------------------------------------------------------------------------- adapter


def _sim_counts(rng, n_pos, n_mc, cond, de_positions, fold=1.8, base=300):
    counts = rng.poisson(base, size=(n_pos, n_mc)).astype(float)
    counts[np.ix_(de_positions, cond == "case")] *= fold
    return np.rint(counts).astype(np.int64)


def test_fit_nb_mixed_recovers_injected_block() -> None:
    rng = np.random.default_rng(0)
    n_pos, n_mc = 40, 60
    cond = np.array(["ctrl", "case"]).repeat(n_mc // 2)
    donor = np.repeat([f"d{i}" for i in range(6)], 10)
    de_pos = np.arange(15, 25)
    null_pos = np.arange(0, 10)
    y = _sim_counts(rng, n_pos, n_mc, cond, de_pos, fold=2.0)

    res = fit_nb_mixed(y, cond, donor, case="case", control="ctrl", verbose=False)
    assert res.n_case == 30 and res.n_control == 30
    # most injected positions significant and up; block far more enriched than a null region
    block_hits = (res.padj[de_pos] < 0.05)
    assert block_hits.mean() >= 0.7
    assert np.nanmedian(res.logFC[de_pos]) > 0
    assert block_hits.sum() > (res.padj[null_pos] < 0.05).sum()


def test_fit_nb_mixed_requires_two_donors() -> None:
    rng = np.random.default_rng(1)
    n_pos, n_mc = 10, 20
    cond = np.array(["ctrl", "case"]).repeat(n_mc // 2)
    donor = np.array(["only"] * n_mc)
    y = _sim_counts(rng, n_pos, n_mc, cond, np.arange(3, 6))
    with pytest.raises(ValueError, match=">= 2 donors"):
        fit_nb_mixed(y, cond, donor, case="case", control="ctrl", verbose=False)


def test_fit_nb_mixed_alignment_with_filtering() -> None:
    # Zero out two positions to trigger edgePython's abundance filter; result must realign.
    rng = np.random.default_rng(2)
    n_pos, n_mc = 20, 40
    cond = np.array(["ctrl", "case"]).repeat(n_mc // 2)
    donor = np.repeat([f"d{i}" for i in range(4)], 10)
    y = _sim_counts(rng, n_pos, n_mc, cond, np.arange(8, 12))
    y[5, :] = 0
    y[13, :] = 0
    res = fit_nb_mixed(y, cond, donor, case="case", control="ctrl", cpc=0.005, mincp=5, verbose=False)
    assert res.logFC.shape == (n_pos,)
    assert not res.tested[5] and np.isnan(res.logFC[5])
    assert res.tested[8]


# --------------------------------------------------------------------------- orchestration


@pytest.fixture
def tg_with_signal() -> AnnData:
    """Transcriptogram AnnData: DE block in cell type T (case up), none in B."""
    rng = np.random.default_rng(3)
    n_pos = 60
    per = 4  # metacells per (celltype, condition, donor)
    rows = []
    for ct in ["T", "B"]:
        for cond in ["ctrl", "case"]:
            donors = ["d0", "d1", "d2"] if cond == "ctrl" else ["d3", "d4", "d5"]
            for d in donors:
                for _ in range(per):
                    rows.append((ct, cond, d))
    obs = pd.DataFrame(rows, columns=["cell_type", "condition", "donor"])
    obs.index = [f"mc{i}" for i in range(len(obs))]
    n_mc = len(obs)

    counts = rng.poisson(300, size=(n_mc, n_pos)).astype(float)
    de_pos = np.arange(20, 30)
    t_case = (obs["cell_type"] == "T").to_numpy() & (obs["condition"] == "case").to_numpy()
    counts[np.ix_(t_case, de_pos)] *= 1.8
    counts = np.rint(counts).astype(np.int64)

    tg = AnnData(X=counts.astype(float), obs=obs, var=pd.DataFrame(index=[str(i) for i in range(n_pos)]))
    tg.layers["window_counts"] = counts
    tg.obs["total_counts"] = counts.sum(axis=1)
    tg.uns["sctg"] = {"radius": 3}
    return tg


def test_differential_systems_per_celltype(tg_with_signal: AnnData) -> None:
    sctg.tl.differential_systems(
        tg_with_signal,
        condition_key="condition",
        case="case",
        control="ctrl",
        donor_key="donor",
        group_key="cell_type",
        verbose=False,
    )
    info = tg_with_signal.uns["diffsystems"]["contrasts"]
    assert set(info) == {"T", "B"}

    # Cell type T: the injected block (20..29) should be recovered as DE and clustered together.
    t = tg_with_signal.varm["diffsystems_T"]
    de_pos = np.arange(20, 30)
    block_de = t["is_de"].to_numpy()[de_pos]
    assert block_de.mean() >= 0.7
    assert (t["logFC"].to_numpy()[de_pos][block_de] > 0).all()
    de_clusters = set(t["cluster"].to_numpy()[de_pos][block_de]) - {0}
    assert len(de_clusters) == 1  # the DE positions in the block share one cluster
    assert info["T"]["n_clusters"] >= 1

    # Cell type B (no injected signal): far fewer DE calls in that block than cell type T.
    b = tg_with_signal.varm["diffsystems_B"]
    assert b["is_de"].to_numpy()[de_pos].sum() < block_de.sum()


def test_differential_systems_records_params(tg_with_signal: AnnData) -> None:
    sctg.tl.differential_systems(
        tg_with_signal,
        condition_key="condition",
        case="case",
        control="ctrl",
        donor_key="donor",
        verbose=False,
    )
    params = tg_with_signal.uns["diffsystems"]["params"]
    assert params["case"] == "case" and params["control"] == "ctrl"
    assert params["radius"] == 3
    assert "all" in tg_with_signal.uns["diffsystems"]["contrasts"]


def test_differential_systems_skips_underpowered_group(tg_with_signal: AnnData) -> None:
    # Make cell type B have a single donor -> should be skipped, not crash.
    tg = tg_with_signal
    b_mask = tg.obs["cell_type"] == "B"
    tg.obs.loc[b_mask, "donor"] = "only"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sctg.tl.differential_systems(
            tg,
            condition_key="condition",
            case="case",
            control="ctrl",
            donor_key="donor",
            group_key="cell_type",
            verbose=False,
        )
    assert "skipped" in tg.uns["diffsystems"]["contrasts"]["B"]
    assert "diffsystems_T" in tg.varm


def test_differential_systems_missing_counts_layer() -> None:
    tg = AnnData(
        X=np.ones((4, 10)),
        obs=pd.DataFrame({"c": [0, 0, 1, 1], "d": list("abcd")}, index=list("wxyz")),
        var=pd.DataFrame(index=[str(i) for i in range(10)]),
    )
    tg.uns["sctg"] = {"radius": 2}
    with pytest.raises(KeyError, match="window_counts"):
        sctg.tl.differential_systems(tg, condition_key="c", case=1, control=0, donor_key="d")
