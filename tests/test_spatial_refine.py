"""Tests for the spatial refinements: detrended Moran's I and spatial_domains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg


@pytest.fixture
def tg_grid() -> AnnData:
    # 24x24 grid; one block of positions carries a smooth global GRADIENT (trend), another carries a
    # compact DOMAIN (a square patch). Detrending should keep the domain, kill the gradient.
    rng = np.random.default_rng(0)
    g = 24
    xx, yy = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    n = coords.shape[0]
    n_pos = 90
    X = rng.normal(0, 0.3, size=(n, n_pos))
    grad = (coords[:, 0] - coords[:, 0].mean()) / coords[:, 0].std()      # global linear gradient
    patch = ((coords[:, 0] < 8) & (coords[:, 1] < 8)).astype(float)       # compact corner domain
    X[:, 20:31] += grad[:, None]        # gradient systems
    X[:, 60:71] += 2 * patch[:, None]   # domain systems
    tg = AnnData(
        X=X,
        var=pd.DataFrame({"position": range(n_pos)}, index=[str(i) for i in range(n_pos)]),
    )
    tg.layers["transcriptogram"] = X
    tg.obsm["spatial"] = coords
    tg.uns["sctg"] = {"radius": 3, "present_mask": np.ones(n_pos, bool)}
    return tg


# --------------------------------------------------------------------------- detrend


def test_detrend_suppresses_global_gradient(tg_grid: AnnData) -> None:
    grad = np.arange(20, 31)
    patch = np.arange(60, 71)
    sctg.tl.spatially_variable_systems(tg_grid, n_neighs=8, key_added="raw")
    sctg.tl.spatially_variable_systems(tg_grid, n_neighs=8, detrend=1, key_added="det")
    mi_raw = tg_grid.varm["raw"]["moran_I"].to_numpy()
    mi_det = tg_grid.varm["det"]["moran_I"].to_numpy()
    # raw: both gradient and domain are strongly autocorrelated
    assert np.nanmean(mi_raw[grad]) > 0.3 and np.nanmean(mi_raw[patch]) > 0.3
    # detrended: the gradient's Moran collapses, the compact domain survives
    assert np.nanmean(mi_det[grad]) < 0.5 * np.nanmean(mi_raw[grad])
    assert np.nanmean(mi_det[patch]) > np.nanmean(mi_det[grad])


def test_detrend_records_param(tg_grid: AnnData) -> None:
    sctg.tl.spatially_variable_systems(tg_grid, detrend=2)
    assert tg_grid.uns["spatially_variable_systems"]["params"]["detrend"] == 2


# --------------------------------------------------------------------------- spatial_domains


def test_spatial_domains_recovers_patch(tg_grid: AnnData) -> None:
    pytest.importorskip("scanpy")
    sctg.tl.spatial_domains(tg_grid, resolution=1.0)
    assert "spatial_domain" in tg_grid.obs
    info = tg_grid.uns["spatial_domain"]
    assert info["n_domains"] >= 2
    # the corner patch spots should be concentrated in one domain
    coords = tg_grid.obsm["spatial"]
    in_patch = (coords[:, 0] < 8) & (coords[:, 1] < 8)
    dom = tg_grid.obs["spatial_domain"].to_numpy()
    # the modal domain among patch spots covers most of the patch
    vals, counts = np.unique(dom[in_patch], return_counts=True)
    assert counts.max() / in_patch.sum() > 0.6
    # the domain-defining score peaks on the domain systems
    score = tg_grid.var["spatial_domain_score"].to_numpy()
    assert np.nanmean(score[np.arange(60, 71)]) > np.nanmean(score[np.arange(0, 15)])


def test_spatial_domains_writes_signature(tg_grid: AnnData) -> None:
    pytest.importorskip("scanpy")
    sctg.tl.spatial_domains(tg_grid, use_spatial_systems=False)
    mean = tg_grid.varm["spatial_domain_mean"]
    assert mean.shape[0] == tg_grid.n_vars
    assert mean.shape[1] == tg_grid.uns["spatial_domain"]["n_domains"]


def test_spatial_domains_requires_coords() -> None:
    tg = AnnData(X=np.zeros((5, 4)))
    tg.uns["sctg"] = {"radius": 1, "present_mask": np.ones(4, bool)}
    with pytest.raises(KeyError, match="coordinates"):
        sctg.tl.spatial_domains(tg)
