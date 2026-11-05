"""High-resolution (Visium HD) support: bin_spatial aggregation and scale-safe spatial tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import sctranscriptogramer as sctg
from sctranscriptogramer.tl.spatial import _moran_scores, _row_standardise


@pytest.fixture
def hd_grid() -> AnnData:
    # 12x12 fine lattice at unit spacing; a handful of genes with structured counts.
    g = 12
    xx, yy = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    n = coords.shape[0]
    rng = np.random.default_rng(0)
    counts = rng.poisson(3, size=(n, 20)).astype(np.int64)
    ad = AnnData(
        X=counts.astype(np.float32),
        var=pd.DataFrame(index=[f"g{i}" for i in range(20)]),
    )
    ad.obsm["spatial"] = coords
    ad.obs["region"] = pd.Categorical(np.where(coords[:, 0] < 6, "left", "right"))
    ad.obs["depth"] = coords[:, 0]
    return ad


# --------------------------------------------------------------------------- bin_spatial


def test_bin_spatial_conserves_counts_and_shrinks(hd_grid: AnnData) -> None:
    out = sctg.pp.bin_spatial(hd_grid, bin_size=2.0, normalize=False)
    # 12x12 -> 6x6 bins, each merging a 2x2 block
    assert out.n_obs == 36
    assert out.layers["counts"].sum() == hd_grid.X.sum()
    assert (out.obs["n_cells"] == 4).all()
    assert out.uns["sctg_bins"]["bin_size"] == 2.0
    assert out.obsm["spatial"].shape == (36, 2)


def test_bin_spatial_factor_matches_bin_size(hd_grid: AnnData) -> None:
    by_size = sctg.pp.bin_spatial(hd_grid, bin_size=3.0, normalize=False)
    by_factor = sctg.pp.bin_spatial(hd_grid, factor=3, normalize=False)  # spacing 1 -> bin_size 3
    assert by_factor.n_obs == by_size.n_obs
    assert by_factor.layers["counts"].sum() == by_size.layers["counts"].sum()


def test_bin_spatial_carries_obs(hd_grid: AnnData) -> None:
    out = sctg.pp.bin_spatial(
        hd_grid, bin_size=2.0, normalize=False, carry_obs=["region", "depth"]
    )
    assert "region" in out.obs and "depth" in out.obs
    # numeric averaged, categorical majority; left-most bins stay "left"
    left_bins = out.obsm["spatial"][:, 0] < 6
    assert (out.obs["region"].to_numpy()[left_bins] == "left").all()


def test_bin_spatial_min_counts_drops_low_bins(hd_grid: AnnData) -> None:
    full = sctg.pp.bin_spatial(hd_grid, bin_size=2.0, normalize=False)
    floor = int(np.median(full.obs["total_counts"])) + 1  # remove roughly the lower half
    out = sctg.pp.bin_spatial(hd_grid, bin_size=2.0, normalize=False, min_counts=floor)
    assert 0 < out.n_obs < 36
    assert (out.obs["total_counts"] >= floor).all()


def test_bin_spatial_requires_coords() -> None:
    ad = AnnData(X=np.ones((4, 3), dtype=np.float32))
    with pytest.raises(KeyError, match="coordinates"):
        sctg.pp.bin_spatial(ad)


# --------------------------------------------------------------------------- scale-safe tools


def test_moran_sparse_dense_parity(hd_grid: AnnData) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(hd_grid.n_obs, 40))
    present = np.ones(40, bool)
    W = _row_standardise(
        sctg.tl.spatial._knn_weights(hd_grid.obsm["spatial"], 6)
    )
    dense = _moran_scores(X, W, present, chunk_size=40)
    chunked = _moran_scores(X, W, present, chunk_size=7)
    sparse = _moran_scores(sp.csr_matrix(X), W, present, chunk_size=7)
    assert np.allclose(dense, chunked, equal_nan=True)
    assert np.allclose(dense, sparse, equal_nan=True)


def test_spatially_variable_sparse_matches_dense(hd_grid: AnnData) -> None:
    # build a transcriptogram-like tg with a spatial signal in some positions
    coords = hd_grid.obsm["spatial"]
    patch = ((coords[:, 0] < 6)).astype(float)
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.2, size=(hd_grid.n_obs, 30))
    X[:, 5:10] += patch[:, None]
    var = pd.DataFrame({"position": range(30)}, index=[str(i) for i in range(30)])

    def make(mat):
        tg = AnnData(X=mat, var=var.copy())
        tg.layers["transcriptogram"] = mat
        tg.obsm["spatial"] = coords
        tg.uns["sctg"] = {"radius": 2, "present_mask": np.ones(30, bool)}
        return tg

    tg_dense = make(X.copy())
    tg_sparse = make(sp.csr_matrix(X))
    sctg.tl.spatially_variable_systems(tg_dense, n_neighs=6, chunk_size=4)
    sctg.tl.spatially_variable_systems(tg_sparse, n_neighs=6, chunk_size=4)
    md = tg_dense.varm["spatially_variable_systems"]["moran_I"].to_numpy()
    ms = tg_sparse.varm["spatially_variable_systems"]["moran_I"].to_numpy()
    assert np.allclose(md, ms, equal_nan=True)
    # the injected patch positions are the most autocorrelated
    assert np.nanmean(md[5:10]) > np.nanmean(md[15:20])


def test_spatial_domains_sparse_runs(hd_grid: AnnData) -> None:
    pytest.importorskip("scanpy")
    coords = hd_grid.obsm["spatial"]
    patch = (coords[:, 0] < 6).astype(float)
    rng = np.random.default_rng(3)
    X = rng.normal(0, 0.2, size=(hd_grid.n_obs, 24))
    X[:, 8:14] += 2 * patch[:, None]
    var = pd.DataFrame({"position": range(24)}, index=[str(i) for i in range(24)])
    tg = AnnData(X=sp.csr_matrix(X), var=var)
    tg.layers["transcriptogram"] = sp.csr_matrix(X)
    tg.obsm["spatial"] = coords
    tg.uns["sctg"] = {"radius": 2, "present_mask": np.ones(24, bool)}
    sctg.tl.spatial_domains(tg, use_spatial_systems=False, resolution=1.0)
    assert "spatial_domain" in tg.obs
    assert tg.varm["spatial_domain_mean"].shape[0] == 24
    score = tg.var["spatial_domain_score"].to_numpy()
    assert np.nanmean(score[8:14]) > np.nanmean(score[0:6])
