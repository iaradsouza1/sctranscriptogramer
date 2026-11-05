"""Tests for ACAT, fit_nb_factor, and tl.divergent_systems."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg
from sctranscriptogramer.stats.multitest import acat
from sctranscriptogramer.stats.nb_mixed import fit_nb_factor

pytest.importorskip("edgepython")


# --------------------------------------------------------------------------- ACAT


def test_acat_uniform_under_null() -> None:
    rng = np.random.default_rng(0)
    # rows of independent uniforms -> combined p roughly uniform, well-calibrated at 0.05
    p = rng.uniform(size=(20000, 4))
    comb = acat(p, axis=1)
    assert 0.03 < np.mean(comb < 0.05) < 0.07
    assert comb.min() >= 0 and comb.max() <= 1


def test_acat_small_pvalue_dominates() -> None:
    # one tiny p-value should drive the combination small
    p = np.array([[1e-8, 0.5, 0.6, 0.7]])
    assert acat(p, axis=1)[0] < 1e-3


def test_acat_ignores_nan() -> None:
    p = np.array([[0.01, np.nan, 0.02]])
    assert np.isfinite(acat(p, axis=1)[0])


# --------------------------------------------------------------------------- fit_nb_factor


def _sim_factor(rng, n_pos, per_group, groups, de_positions, folds):
    labels = np.repeat(groups, per_group)
    n = labels.size
    counts = rng.poisson(300, size=(n_pos, n)).astype(float)
    for g, fold in folds.items():
        counts[np.ix_(de_positions, labels == g)] *= fold
    return np.rint(counts).astype(np.int64), labels


def test_fit_nb_factor_recovers_divergent_block() -> None:
    rng = np.random.default_rng(0)
    n_pos = 200  # enough unchanged features for TMM to anchor on
    groups = ["A", "B", "C"]
    per = 12
    donor = np.tile(np.repeat([f"d{i}" for i in range(3)], per // 3), len(groups))
    de = np.arange(90, 110)
    null = np.arange(0, 20)
    # group C is elevated over the block; A/B flat -> factor "matters" there
    y, labels = _sim_factor(rng, n_pos, per, groups, de, {"C": 2.0})

    res = fit_nb_factor(y, labels, donor, verbose=False)
    assert res.levels == ["A", "B", "C"]
    # the divergent block is strongly detected...
    assert (res.padj[de] < 0.05).mean() >= 0.7
    # ...and its level effect sizes are far larger than the (compositionally-shifted) nulls.
    assert np.nanmedian(res.max_abs_logFC[de]) > 5 * np.nanmedian(res.max_abs_logFC[null])


def test_fit_nb_factor_needs_two_levels() -> None:
    rng = np.random.default_rng(1)
    y, labels = _sim_factor(rng, 10, 12, ["A"], np.arange(3, 6), {})
    donor = np.repeat(["d0", "d1"], 6)
    with pytest.raises(ValueError, match=">= 2 levels"):
        fit_nb_factor(y, labels, donor, verbose=False)


# --------------------------------------------------------------------------- divergent_systems


@pytest.fixture
def tg_factor() -> AnnData:
    rng = np.random.default_rng(2)
    n_pos = 50
    states = ["S1", "S2", "S3"]
    per = 12  # metacells per state
    rows = []
    for st in states:
        for d in ["d0", "d1", "d2"]:  # crossed with donor
            for _ in range(per // 3):
                rows.append((st, d))
    obs = pd.DataFrame(rows, columns=["cell_state", "donor"])
    obs.index = [f"mc{i}" for i in range(len(obs))]
    n_mc = len(obs)
    counts = rng.poisson(300, size=(n_mc, n_pos)).astype(float)
    de = np.arange(20, 30)
    s3 = (obs["cell_state"] == "S3").to_numpy()
    counts[np.ix_(s3, de)] *= 1.8
    counts = np.rint(counts).astype(np.int64)
    tg = AnnData(X=counts.astype(float), obs=obs,
                 var=pd.DataFrame(index=[str(i) for i in range(n_pos)]))
    tg.layers["window_counts"] = counts
    tg.obs["total_counts"] = counts.sum(1)
    tg.uns["sctg"] = {"radius": 3}
    return tg


def test_divergent_systems_recovers_region(tg_factor: AnnData) -> None:
    sctg.tl.divergent_systems(tg_factor, group_key="cell_state", donor_key="donor", verbose=False)
    info = tg_factor.uns["divergent_systems"]
    assert info["levels"] == ["S1", "S2", "S3"]
    assert info["n_regions"] >= 1

    tbl = tg_factor.varm["divergent_systems_cell_state"]
    de = np.arange(20, 30)
    assert tbl["is_divergent"].to_numpy()[de].mean() >= 0.7
    # the divergent block shares one region
    regions_on_block = set(tbl["region"].to_numpy()[de][tbl["is_divergent"].to_numpy()[de]]) - {0}
    assert len(regions_on_block) == 1


def test_divergent_systems_missing_layer() -> None:
    tg = AnnData(
        X=np.zeros((3, 4)),
        obs=pd.DataFrame({"cs": ["a", "b", "c"], "d": ["x", "y", "z"]}, index=list("pqr")),
        var=pd.DataFrame(index=list("abcd")),
    )
    tg.uns["sctg"] = {"radius": 1}
    with pytest.raises(KeyError, match="window_counts"):
        sctg.tl.divergent_systems(tg, group_key="cs", donor_key="d")
