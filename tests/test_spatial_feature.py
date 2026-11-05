"""Spatial feature association: which systems co-localize with an annotated feature (e.g. plaques)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import sctranscriptogramer as sctg


@pytest.fixture
def tg_feature() -> AnnData:
    # 20x20 lattice. A corner patch is the "feature". Positions 5:10 track the feature (associated);
    # positions 20:25 track a DIFFERENT, unrelated patch (spatially variable but NOT feature-associated);
    # the rest is noise.
    g = 20
    xx, yy = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    n = coords.shape[0]
    rng = np.random.default_rng(0)
    n_pos = 40
    X = rng.normal(0, 0.25, size=(n, n_pos))
    feature = ((coords[:, 0] < 10) & (coords[:, 1] < 10)).astype(float)  # corner plaque
    other = ((coords[:, 0] >= 10) & (coords[:, 1] >= 10)).astype(float)  # opposite corner
    X[:, 5:10] += 2.0 * feature[:, None]
    X[:, 20:25] += 2.0 * other[:, None]
    tg = AnnData(
        X=X, var=pd.DataFrame({"position": range(n_pos)}, index=[str(i) for i in range(n_pos)])
    )
    tg.layers["transcriptogram"] = X
    tg.obsm["spatial"] = coords
    tg.obs["bin_x"] = coords[:, 0].astype(int)
    tg.obs["bin_y"] = coords[:, 1].astype(int)
    tg.obs["plaque"] = feature
    tg.uns["sctg"] = {"radius": 2, "present_mask": np.ones(n_pos, bool)}
    return tg


# --------------------------------------------------------------------------- annotate_spatial_feature


def test_annotate_from_coords_builds_mask(tg_feature: AnnData) -> None:
    coords = tg_feature.obsm["spatial"]
    plaque_pts = coords[tg_feature.obs["plaque"].to_numpy() > 0]  # the 1-locations only
    sctg.tl.annotate_spatial_feature(
        tg_feature, plaque_pts, method="radius", radius=0.5, key_added="plq"
    )
    got = tg_feature.obs["plq"].to_numpy()
    assert np.array_equal(got > 0, tg_feature.obs["plaque"].to_numpy() > 0)


def test_annotate_requires_coords() -> None:
    tg = AnnData(X=np.zeros((4, 3)))
    with pytest.raises(KeyError, match="coordinates"):
        sctg.tl.annotate_spatial_feature(tg, np.zeros((2, 2)))


# --------------------------------------------------------------------------- association: toroidal


def test_toroidal_recovers_feature_systems(tg_feature: AnnData) -> None:
    sctg.tl.spatial_feature_association(
        tg_feature, "plaque", method="toroidal", n_perm=300, random_state=0
    )
    res = tg_feature.varm["spatial_feature_association"]
    assert tg_feature.uns["spatial_feature_association"]["method"] == "toroidal"
    corr = res["corr"].to_numpy()
    assoc = res["is_associated"].to_numpy()
    # the feature-tracking systems are positively correlated and flagged
    assert np.all(corr[5:10] > 0.3)
    assert assoc[5:10].sum() >= 4
    assert np.all(res["effect"].to_numpy()[5:10] > 0)
    # the unrelated spatial patch is NOT (much) associated with the feature -> specificity
    assert assoc[20:25].sum() <= 1
    # noise positions are largely not associated
    noise = np.r_[0:5, 10:20, 25:40]
    assert assoc[noise].mean() < 0.2


def test_auto_selects_toroidal_with_lattice(tg_feature: AnnData) -> None:
    sctg.tl.spatial_feature_association(tg_feature, "plaque", method="auto", n_perm=100)
    assert tg_feature.uns["spatial_feature_association"]["method"] == "toroidal"


def test_array_feature_matches_obs(tg_feature: AnnData) -> None:
    f = tg_feature.obs["plaque"].to_numpy()
    sctg.tl.spatial_feature_association(
        tg_feature, f, method="toroidal", n_perm=100, key_added="arr"
    )
    sctg.tl.spatial_feature_association(
        tg_feature, "plaque", method="toroidal", n_perm=100, key_added="col"
    )
    assert np.allclose(
        tg_feature.varm["arr"]["corr"], tg_feature.varm["col"]["corr"], equal_nan=True
    )


# --------------------------------------------------------------------------- association: analytic


def test_moran_corrected_runs_and_flags(tg_feature: AnnData) -> None:
    sctg.tl.spatial_feature_association(tg_feature, "plaque", method="moran_corrected", n_neighs=8)
    res = tg_feature.varm["spatial_feature_association"]
    assert tg_feature.uns["spatial_feature_association"]["method"] == "moran_corrected"
    assert res["is_associated"].to_numpy()[5:10].sum() >= 4
    # analytic null still discriminates the unrelated patch
    assert res["is_associated"].to_numpy()[20:25].sum() <= 2


def test_sparse_layer_matches_dense(tg_feature: AnnData) -> None:
    dense_corr = None
    for as_sparse in (False, True):
        tg = tg_feature.copy()
        if as_sparse:
            tg.layers["transcriptogram"] = sp.csr_matrix(tg.layers["transcriptogram"])
        sctg.tl.spatial_feature_association(tg, "plaque", method="toroidal", n_perm=150)
        c = tg.varm["spatial_feature_association"]["corr"].to_numpy()
        if dense_corr is None:
            dense_corr = c
        else:
            assert np.allclose(dense_corr, c, equal_nan=True)


def test_auto_falls_back_on_irregular_geometry() -> None:
    # random (non-lattice) point pattern -> auto should warn and use moran_corrected
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 50, size=(300, 2))
    X = rng.normal(size=(300, 20))
    f = (coords[:, 0] < 25).astype(float)
    tg = AnnData(X=X, var=pd.DataFrame(index=[str(i) for i in range(20)]))
    tg.layers["transcriptogram"] = X
    tg.obsm["spatial"] = coords
    tg.uns["sctg"] = {"radius": 2, "present_mask": np.ones(20, bool)}
    with pytest.warns(UserWarning, match="no regular lattice"):
        sctg.tl.spatial_feature_association(tg, f, method="auto", n_neighs=8)
    assert tg.uns["spatial_feature_association"]["method"] == "moran_corrected"


def test_constant_feature_errors(tg_feature: AnnData) -> None:
    tg_feature.obs["flat"] = np.ones(tg_feature.n_obs)
    with pytest.raises(ValueError, match="constant"):
        sctg.tl.spatial_feature_association(tg_feature, "flat", method="moran_corrected")


def test_user_radius_dilates_mask(tg_feature: AnnData) -> None:
    # a user-provided radius builds a peri-feature halo: larger radius -> more positive units
    coords = tg_feature.obsm["spatial"]
    plaque_pts = coords[tg_feature.obs["plaque"].to_numpy() > 0]
    sctg.tl.annotate_spatial_feature(tg_feature, plaque_pts, radius=0.5, key_added="core")
    sctg.tl.annotate_spatial_feature(tg_feature, plaque_pts, radius=3.0, key_added="halo")
    n_core = int((tg_feature.obs["core"].to_numpy() > 0).sum())
    n_halo = int((tg_feature.obs["halo"].to_numpy() > 0).sum())
    assert n_halo > n_core
