"""Tests for pp.suggest_metacell_size."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg


def _adata(n_per_partition, n_groups, umis_per_cell, n_genes=50, seed=0):
    rng = np.random.default_rng(seed)
    rows, blocks = [], []
    for g in range(n_groups):
        for d in range(2):  # two donors per group
            for _ in range(n_per_partition):
                rows.append((f"g{g}", f"d{g}_{d}"))
        # counts scaled so each cell ~ umis_per_cell total
    n = len(rows)
    base = rng.random(size=(n, n_genes))
    X = np.rint(base / base.sum(1, keepdims=True) * umis_per_cell).astype(np.int64)
    obs = pd.DataFrame(rows, columns=["grp", "donor"], index=[f"c{i}" for i in range(n)])
    ad = AnnData(X=X.astype(float), obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
    ad.layers["counts"] = X
    return ad


def test_returns_suggestion_and_floors_at_engine_min() -> None:
    # deep cells (high UMIs) -> UMI floor is small -> engine_min binds
    ad = _adata(n_per_partition=60, n_groups=3, umis_per_cell=5000)
    out = sctg.pp.suggest_metacell_size(ad, group_keys=["grp", "donor"], counts_layer="counts")
    assert out["target_metacell_size"] == 12          # engine_min
    assert out["binding_floor"] == "engine_min"
    assert out["n_partitions"] == 6                   # 3 groups x 2 donors


def test_shallow_cells_are_umi_limited() -> None:
    # very shallow cells -> need many to reach the UMI floor
    ad = _adata(n_per_partition=200, n_groups=2, umis_per_cell=500)
    out = sctg.pp.suggest_metacell_size(
        ad, group_keys=["grp", "donor"], counts_layer="counts", min_umis_per_metacell=10_000
    )
    assert out["target_metacell_size"] == int(np.ceil(10_000 / out["median_cell_umis"]))
    assert out["target_metacell_size"] > 12
    assert out["binding_floor"] == "umis"


def test_underpowered_partitions_warn() -> None:
    # tiny partitions -> can't reach min_metacells_per_group -> warning
    ad = _adata(n_per_partition=8, n_groups=2, umis_per_cell=5000)
    with pytest.warns(UserWarning, match="cell-limited|collapse"):
        out = sctg.pp.suggest_metacell_size(
            ad, group_keys=["grp", "donor"], counts_layer="counts", min_metacells_per_group=3
        )
    assert out["expected_metacells_per_partition"]["median"] < 3


def test_expected_metacells_scales() -> None:
    ad = _adata(n_per_partition=120, n_groups=2, umis_per_cell=5000)
    out = sctg.pp.suggest_metacell_size(ad, group_keys=["grp", "donor"], counts_layer="counts")
    # ~120 cells / 12 -> ~10 metacells per partition
    assert out["expected_metacells_per_partition"]["median"] >= 8


def test_missing_group_keys() -> None:
    ad = _adata(20, 2, 3000)
    with pytest.raises(KeyError, match="nope"):
        sctg.pp.suggest_metacell_size(ad, group_keys=["nope"], counts_layer="counts")
