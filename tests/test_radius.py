"""Tests for tl.select_radius (radius selection by null-corrected reproducibility)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg


def _make_tg(rng, n_pos=200, block=slice(80, 121), fold=1.0, noise=1.0):
    """Two groups; a wide block carries a consistent group difference under heavy per-position noise,
    so smoothing across the block should boost reproducibility (favouring r > 0)."""
    per = 12
    labels = np.array(["A", "B"]).repeat(per)
    donors = np.concatenate([np.repeat([f"a{i}" for i in range(4)], 3),
                             np.repeat([f"b{i}" for i in range(4)], 3)])
    n = labels.size
    step1 = rng.normal(0, noise, size=(n, n_pos))
    step1[labels == "A", block] += fold          # coherent block signal in group A
    var = pd.DataFrame({"position": range(n_pos), "protein": [f"P{i}" for i in range(n_pos)]},
                       index=[str(i) for i in range(n_pos)])
    obs = pd.DataFrame({"grp": labels, "donor": donors}, index=[f"u{i}" for i in range(n)])
    tg = AnnData(X=step1.copy(), obs=obs, var=var)
    tg.layers["step1"] = step1
    tg.uns["sctg"] = {"radius": 0, "present_mask": np.ones(n_pos, bool)}
    return tg


def test_returns_curve(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    tg = _make_tg(rng)
    df = sctg.tl.select_radius(tg, group_key="grp", radii=[0, 5, 20, 40, 80], n_splits=8)
    assert list(df.columns) == ["reproducibility", "score", "score_sd", "score_se", "null"]
    assert set(df.index) == {0, 5, 20, 40, 80}
    assert "best_radius" in df.attrs
    assert df.attrs["best_radius"] in df.index
    assert tg.uns["radius_selection"]["best_radius"] == df.attrs["best_radius"]


def test_prefers_nonzero_radius_for_block_signal() -> None:
    rng = np.random.default_rng(1)
    tg = _make_tg(rng, block=slice(80, 121), fold=1.0, noise=1.5)
    df = sctg.tl.select_radius(tg, group_key="grp", radii=[0, 5, 10, 20, 40, 80, 160], n_splits=15)
    # smoothing across the wide noisy block improves reproducibility over no smoothing
    assert df.attrs["best_radius"] > 0
    assert df.loc[df.attrs["best_radius"], "score"] > df.loc[0, "score"]


def test_null_is_near_zero() -> None:
    rng = np.random.default_rng(2)
    tg = _make_tg(rng)
    sctg.tl.select_radius(tg, group_key="grp", radii=[0, 20, 80], n_splits=15)
    # the shuffled-label reproducibility should hover around zero at every radius
    null = tg.uns["radius_selection"]["reproducibility"] - tg.uns["radius_selection"]["score"]
    assert np.all(np.abs(null) < 0.25)


def test_donor_split_used_when_enough_donors() -> None:
    rng = np.random.default_rng(3)
    tg = _make_tg(rng)
    sctg.tl.select_radius(tg, group_key="grp", donor_key="donor", radii=[0, 40], n_splits=6)
    assert tg.uns["radius_selection"]["params"]["split"] == "donor"


def test_one_se_rule_not_larger_than_argmax() -> None:
    rng = np.random.default_rng(4)
    tg = _make_tg(rng)
    radii = [0, 5, 10, 20, 40, 80, 160]
    a = sctg.tl.select_radius(tg, group_key="grp", radii=radii, n_splits=12, rule="argmax")
    b = sctg.tl.select_radius(tg, group_key="grp", radii=radii, n_splits=12, rule="one_se")
    assert b.attrs["best_radius"] <= a.attrs["best_radius"]


def test_both_halves_contain_all_groups_donor_level() -> None:
    # A donor-level condition (one condition per donor): a naive random donor split could put all
    # case donors on one side. Verify the stratified/validated split keeps both groups in both halves
    # by checking select_radius runs and uses a donor split with all splits valid.
    rng = np.random.default_rng(7)
    n_pos = 120
    # 8 donors: 4 "disease", 4 "healthy"; 3 metacells each
    donor_ids = np.repeat([f"d{i}" for i in range(8)], 3)
    cond = np.repeat(["disease"] * 4 + ["healthy"] * 4, 3)
    n = donor_ids.size
    step1 = rng.normal(0, 1, size=(n, n_pos))
    step1[cond == "disease", 50:70] += 1.0
    tg = AnnData(
        X=step1.copy(),
        obs=pd.DataFrame({"grp": cond, "donor": donor_ids}, index=[f"u{i}" for i in range(n)]),
        var=pd.DataFrame({"position": range(n_pos)}, index=[str(i) for i in range(n_pos)]),
    )
    tg.layers["step1"] = step1
    tg.uns["sctg"] = {"radius": 0, "present_mask": np.ones(n_pos, bool)}

    df = sctg.tl.select_radius(tg, group_key="grp", donor_key="donor",
                               radii=[0, 10, 20, 40], n_splits=10)
    info = tg.uns["radius_selection"]
    assert info["params"]["split"] == "donor"
    assert info["params"]["n_splits"] == 10          # all 10 splits were valid (both groups present)
    assert np.all(np.isfinite(df["score"].to_numpy()))


def test_requires_step1() -> None:
    tg = AnnData(X=np.zeros((4, 5)), obs=pd.DataFrame({"grp": list("aabb")}, index=list("wxyz")))
    with pytest.raises(KeyError, match="step1"):
        sctg.tl.select_radius(tg, group_key="grp")


def test_requires_two_groups() -> None:
    rng = np.random.default_rng(5)
    tg = _make_tg(rng)
    tg.obs["grp"] = "only"
    with pytest.raises(ValueError, match=">= 2 levels"):
        sctg.tl.select_radius(tg, group_key="grp")
