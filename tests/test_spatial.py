"""Tests for tl.spatially_variable_systems (Moran's I on the transcriptogram)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg


@pytest.fixture
def tg_spatial() -> AnnData:
    # 20x20 grid of units; a block of positions carries a smooth spatial gradient (autocorrelated),
    # the rest is spatial white noise.
    rng = np.random.default_rng(0)
    g = 20
    xx, yy = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)  # 400 x 2
    n = coords.shape[0]
    n_pos = 60
    X = rng.normal(size=(n, n_pos))
    block = np.arange(20, 31)
    grad = (coords[:, 0] - coords[:, 0].mean()) / coords[:, 0].std()
    X[:, block] = grad[:, None] + 0.1 * rng.normal(size=(n, block.size))  # spatial signal
    var = pd.DataFrame(
        {"position": range(n_pos), "protein": [f"P{i}" for i in range(n_pos)]},
        index=[str(i) for i in range(n_pos)],
    )
    tg = AnnData(X=X, var=var)
    tg.layers["transcriptogram"] = X
    tg.obsm["spatial"] = coords
    tg.uns["sctg"] = {"radius": 3, "present_mask": np.ones(n_pos, bool)}
    return tg


def test_flags_spatial_block(tg_spatial: AnnData) -> None:
    sctg.tl.spatially_variable_systems(tg_spatial, n_neighs=6)
    v = tg_spatial.varm["spatially_variable_systems"]
    block = np.arange(20, 31)
    null = np.arange(0, 15)
    assert v["is_spatial"].to_numpy()[block].mean() >= 0.8
    assert v["is_spatial"].to_numpy()[null].mean() < 0.3
    # the spatial block has much higher Moran's I than the noise
    assert np.nanmean(v["moran_I"].to_numpy()[block]) > 0.5
    assert np.nanmean(v["moran_I"].to_numpy()[block]) > 5 * abs(
        np.nanmean(v["moran_I"].to_numpy()[null])
    )


def test_top_quantile_selects_most_spatial(tg_spatial: AnnData) -> None:
    # top_quantile keeps only the most spatially-structured positions (effect-size selection)
    sctg.tl.spatially_variable_systems(tg_spatial, n_neighs=6, top_quantile=0.90)
    v = tg_spatial.varm["spatially_variable_systems"]
    sel = np.flatnonzero(v["is_spatial"].to_numpy())
    assert 0 < sel.size <= int(0.15 * tg_spatial.n_vars)  # ~top 10%
    # the selected positions are (almost) all inside the planted spatial block
    block = set(range(20, 31))
    assert np.mean([s in block for s in sel]) >= 0.8


def test_min_moran_threshold(tg_spatial: AnnData) -> None:
    sctg.tl.spatially_variable_systems(tg_spatial, n_neighs=6, min_moran=0.5)
    v = tg_spatial.varm["spatially_variable_systems"]
    sel = v["is_spatial"].to_numpy()
    assert np.all(v["moran_I"].to_numpy()[sel] >= 0.5)


def test_regions_cover_block(tg_spatial: AnnData) -> None:
    sctg.tl.spatially_variable_systems(tg_spatial, n_neighs=6)
    info = tg_spatial.uns["spatially_variable_systems"]
    assert info["n_regions"] >= 1
    v = tg_spatial.varm["spatially_variable_systems"]
    block = np.arange(20, 31)
    sig = v["is_spatial"].to_numpy()[block]
    regions_on_block = set(v["region"].to_numpy()[block][sig]) - {0}
    assert len(regions_on_block) == 1


def test_uns_records_moran_moments(tg_spatial: AnnData) -> None:
    sctg.tl.spatially_variable_systems(tg_spatial, n_neighs=6)
    info = tg_spatial.uns["spatially_variable_systems"]
    assert info["moran_expected"] == pytest.approx(-1 / (tg_spatial.n_obs - 1))
    assert info["moran_var"] > 0
    assert info["params"]["n_neighs"] == 6


def test_missing_coords_raises() -> None:
    tg = AnnData(X=np.zeros((4, 3)), var=pd.DataFrame(index=list("abc")))
    tg.uns["sctg"] = {"radius": 1, "present_mask": np.ones(3, bool)}
    with pytest.raises(KeyError, match="spatial"):
        sctg.tl.spatially_variable_systems(tg)


def test_connectivity_key_path(tg_spatial: AnnData) -> None:
    # build a graph and pass it via obsp instead of coordinates
    from sctranscriptogramer.tl.spatial import _knn_weights

    tg_spatial.obsp["spatial_connectivities"] = _knn_weights(
        np.asarray(tg_spatial.obsm["spatial"]), 6
    )
    sctg.tl.spatially_variable_systems(
        tg_spatial, connectivity_key="spatial_connectivities"
    )
    assert tg_spatial.uns["spatially_variable_systems"]["n_spatial"] > 0


def test_project_to_ordering_carries_spatial() -> None:
    # obsm['spatial'] must survive projection so spatial analyses work on tg
    from sctranscriptogramer._core.ordering import Ordering

    rng = np.random.default_rng(1)
    genes = [f"ENSP{i}" for i in range(5)]
    ad = AnnData(
        X=rng.poisson(3, size=(6, 5)).astype(float),
        var=pd.DataFrame(index=genes),
    )
    ad.obsm["spatial"] = rng.normal(size=(6, 2))
    o = Ordering.from_proteins(genes)
    tg = sctg.pp.project_to_ordering(ad, o)
    assert "spatial" in tg.obsm
    np.testing.assert_array_equal(tg.obsm["spatial"], ad.obsm["spatial"])
