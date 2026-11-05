"""Tests for region-level cluster-mass permutation inference."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg
from sctranscriptogramer.stats.permutation import (
    _suprathreshold_regions,
    region_permutation,
)


# --------------------------------------------------------------------------- helpers


def _labels(n_donors=5, n_case=2, per=2):
    donors = np.repeat([f"d{i}" for i in range(n_donors)], per)
    cond = np.where(
        np.isin(donors, [f"d{i}" for i in range(n_case)]), "case", "ctrl"
    )
    return cond, donors


def _mock_pvalue_fn(observed, n_pos, block=slice(40, 51)):
    """Returns a strong block only for the observed labelling; weak for any permutation."""
    def fn(cond_vec):
        p = np.full(n_pos, 0.8)
        p[block] = 1e-4 if np.array_equal(cond_vec, observed) else 0.3
        return p
    return fn


# --------------------------------------------------------------------------- suprathreshold


def test_suprathreshold_regions_and_mass() -> None:
    p = np.full(60, 0.5)
    p[20:31] = 1e-3  # 11 positions below threshold 0.01
    present = np.ones(60, bool)
    regions, mass = _suprathreshold_regions(p, present, thr=0.01, radius=2, n_positions=60)
    assert len(regions) == 1
    assert 20 >= regions[0][0] - 2 and regions[0][1] >= 30  # covers the block (padded)
    np.testing.assert_allclose(mass[0], 11 * 3.0)  # 11 * -log10(1e-3)


# --------------------------------------------------------------------------- permutation core


def test_exact_enumeration_and_region_pvalue() -> None:
    cond, donors = _labels(n_donors=5, n_case=2)
    n_pos = 100
    fn = _mock_pvalue_fn(cond, n_pos)
    r = region_permutation(
        fn, cond, donors, case="case", control="ctrl", radius=3,
        present_mask=np.ones(n_pos, bool), n_positions=n_pos,
        cluster_forming_p=0.01, max_exact=500,
    )
    assert r.exact
    assert r.n_perm == comb(5, 2) - 1              # all distinct relabelings minus the observed
    assert len(r.regions) == 1                     # the observed block
    np.testing.assert_allclose(r.mass[0], 11 * 4.0)  # 11 * -log10(1e-4)
    assert np.all(r.null_max_mass == 0.0)          # no null produces a region
    np.testing.assert_allclose(r.region_pvalue[0], 1 / (r.n_perm + 1))  # most significant possible


def test_random_permutations_when_not_exhaustive() -> None:
    cond, donors = _labels(n_donors=5, n_case=2)
    n_pos = 100
    fn = _mock_pvalue_fn(cond, n_pos)
    r = region_permutation(
        fn, cond, donors, case="case", control="ctrl", radius=3,
        present_mask=np.ones(n_pos, bool), n_positions=n_pos,
        max_exact=2, n_permutations=5,
    )
    assert not r.exact
    assert r.n_perm == 5


def test_no_real_region_gives_empty() -> None:
    cond, donors = _labels()
    n_pos = 100
    fn = lambda cv: np.full(n_pos, 0.5)  # never below threshold
    r = region_permutation(
        fn, cond, donors, case="case", control="ctrl", radius=3,
        present_mask=np.ones(n_pos, bool), n_positions=n_pos,
    )
    assert len(r.regions) == 0
    assert r.region_pvalue.size == 0


def test_requires_both_groups() -> None:
    donors = np.repeat(["d0", "d1", "d2"], 2)
    cond = np.array(["case"] * 6)
    with pytest.raises(ValueError, match="both case and control"):
        region_permutation(
            lambda cv: np.full(10, 0.5), cond, donors, case="case", control="ctrl",
            radius=2, present_mask=np.ones(10, bool), n_positions=10,
        )


# --------------------------------------------------------------------------- integration


@pytest.mark.slow
def test_differential_systems_permutation_integration() -> None:
    pytest.importorskip("edgepython")
    rng = np.random.default_rng(0)
    n_pos = 200
    # 4 donors x 4 metacells; 2 case donors, 2 control -> C(4,2)-1 = 5 exact permutations
    donors = np.repeat([f"d{i}" for i in range(4)], 4)
    cond = np.where(np.isin(donors, ["d0", "d1"]), "disease", "healthy")
    n = donors.size
    counts = rng.poisson(300, size=(n, n_pos)).astype(np.int64)
    counts[cond == "disease", 90:110] = (counts[cond == "disease", 90:110] * 1.8).astype(np.int64)
    tg = AnnData(
        X=counts.astype(float),
        obs=pd.DataFrame({"status": cond, "donor": donors}, index=[f"m{i}" for i in range(n)]),
        var=pd.DataFrame({"position": range(n_pos)}, index=[str(i) for i in range(n_pos)]),
    )
    tg.layers["window_counts"] = counts
    tg.obs["total_counts"] = counts.sum(1)
    tg.uns["sctg"] = {"radius": 3, "present_mask": np.ones(n_pos, bool)}

    sctg.tl.differential_systems(
        tg, condition_key="status", case="disease", control="healthy", donor_key="donor",
        inference="permutation", cluster_forming_p=0.01, max_exact=500, perm_n_jobs=2,
        verbose=False,
    )
    perm = tg.uns["diffsystems"]["contrasts"]["all"]["permutation"]
    assert perm["exact"] is True
    assert perm["n_perm"] == comb(4, 2) - 1
    assert "region_pvalue" in perm
    assert tg.uns["diffsystems"]["params"]["inference"] == "permutation"
