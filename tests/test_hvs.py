"""Tests for tl.highly_variable_systems (unsupervised variable systems)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg


@pytest.fixture
def tg() -> AnnData:
    # 40 metacells x 120 positions; inject high across-metacell variance in a block.
    rng = np.random.default_rng(0)
    n_mc, n_pos = 40, 120
    base = rng.normal(0.0, 0.1, size=(n_mc, n_pos))  # low variance everywhere
    block = np.arange(50, 70)
    base[:, block] += rng.normal(0.0, 2.0, size=(n_mc, block.size))  # high variance block
    var = pd.DataFrame(
        {"position": range(n_pos), "protein": [f"P{i}" for i in range(n_pos)]},
        index=[str(i) for i in range(n_pos)],
    )
    a = AnnData(X=base, var=var)
    a.layers["transcriptogram"] = base
    a.uns["sctg"] = {"radius": 3, "present_mask": np.ones(n_pos, dtype=bool)}
    return a


def test_flags_high_variance_block(tg: AnnData) -> None:
    sctg.tl.highly_variable_systems(tg, quantile=0.85)
    hv = tg.var["highly_variable"].to_numpy()
    block = np.arange(50, 70)
    # the injected block should be preferentially flagged vs the low-variance background
    assert hv[block].mean() > 0.7
    assert hv[np.arange(0, 20)].mean() < 0.2


def test_writes_expected_columns(tg: AnnData) -> None:
    sctg.tl.highly_variable_systems(tg)
    for col in ["hvs_mean", "hvs_variance", "hvs_trend", "hvs_biological_variance",
                "highly_variable", "hvs_region"]:
        assert col in tg.var
    info = tg.uns["highly_variable_systems"]
    assert info["n_highly_variable"] == int(tg.var["highly_variable"].sum())
    assert info["n_regions"] >= 1


def test_biological_variance_is_residual(tg: AnnData) -> None:
    sctg.tl.highly_variable_systems(tg)
    bio = tg.var["hvs_biological_variance"].to_numpy()
    v = tg.var["hvs_variance"].to_numpy()
    t = tg.var["hvs_trend"].to_numpy()
    np.testing.assert_allclose(bio, v - t)


def test_n_top_selection(tg: AnnData) -> None:
    sctg.tl.highly_variable_systems(tg, n_top=15)
    assert int(tg.var["highly_variable"].sum()) == 15


def test_regions_group_the_block(tg: AnnData) -> None:
    sctg.tl.highly_variable_systems(tg, quantile=0.85)
    hv = tg.var["highly_variable"].to_numpy()
    region = tg.var["hvs_region"].to_numpy()
    block = np.arange(50, 70)
    flagged = block[hv[block]]
    assert len(set(region[flagged]) - {0}) == 1  # the block's flagged positions share one region


def test_present_mask_respected(tg: AnnData) -> None:
    mask = np.ones(tg.n_vars, dtype=bool)
    mask[55] = False  # mark a high-variance position absent
    tg.uns["sctg"]["present_mask"] = mask
    sctg.tl.highly_variable_systems(tg, quantile=0.85)
    assert not tg.var["highly_variable"].to_numpy()[55]  # absent -> never flagged


def test_scanpy_compatible_flag(tg: AnnData) -> None:
    # the 'highly_variable' column should let scanpy PCA use it
    sc = pytest.importorskip("scanpy")
    sctg.tl.highly_variable_systems(tg, quantile=0.8)
    tg.X = tg.X.astype(np.float32)
    sc.pp.pca(tg, n_comps=5, use_highly_variable=True)
    assert "X_pca" in tg.obsm


def test_copy(tg: AnnData) -> None:
    out = sctg.tl.highly_variable_systems(tg, copy=True)
    assert out is not None
    assert "highly_variable" in out.var
    assert "highly_variable" not in tg.var
