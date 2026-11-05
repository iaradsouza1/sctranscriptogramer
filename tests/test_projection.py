"""End-to-end tests for project_to_ordering (step 1) and transcriptogram (step 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import sctranscriptogramer as sctg
from sctranscriptogramer._core.boxcar import circular_boxcar
from sctranscriptogramer._core.ordering import Ordering


@pytest.fixture
def toy_ordering() -> Ordering:
    # 6 proteins in order p0..p5
    return Ordering.from_proteins([f"ENSP{i}" for i in range(6)], name="toy")


@pytest.fixture
def toy_adata() -> AnnData:
    # 3 units x 5 genes; gene ids overlap the ordering except geneX which is absent.
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    var = pd.DataFrame(index=["ENSP0", "ENSP1", "ENSP2", "ENSP4", "geneX"])
    ad = AnnData(X=X, var=var)
    ad.obs_names = ["u0", "u1", "u2"]
    ad.layers["counts"] = np.rint(X).astype(np.int64)
    return ad


# --------------------------------------------------------------------------- projection


def test_project_basic_shapes(toy_adata: AnnData, toy_ordering: Ordering) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering, counts_layer="counts")
    assert tg.shape == (3, 6)  # 3 units x 6 positions
    assert list(tg.var["position"]) == [0, 1, 2, 3, 4, 5]
    # geneX unmapped; ENSP3/ENSP5 have no gene -> not present.
    mask = tg.uns["sctg"]["present_mask"]
    np.testing.assert_array_equal(mask, [True, True, True, False, True, False])
    assert tg.uns["sctg"]["n_mapped_positions"] == 4


def test_project_values_scatter_correctly(toy_adata: AnnData, toy_ordering: Ordering) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering)
    # position 0 <- gene ENSP0 (col0), pos1 <- col1, pos2 <- col2, pos4 <- col3
    expected = np.zeros((3, 6))
    expected[:, 0] = toy_adata.X[:, 0]
    expected[:, 1] = toy_adata.X[:, 1]
    expected[:, 2] = toy_adata.X[:, 2]
    expected[:, 4] = toy_adata.X[:, 3]
    np.testing.assert_allclose(tg.X, expected)


def test_project_averages_multiple_genes_per_protein(toy_ordering: Ordering) -> None:
    # Two genes both map to ENSP1 -> their mean should land at position 1.
    X = np.array([[2.0, 4.0, 100.0]])  # unit0
    var = pd.DataFrame(index=["ENSP1", "ENSP1_alias", "ENSP2"])
    ad = AnnData(X=X, var=var)
    dictionary = pd.DataFrame(
        {
            "protein": ["ENSP1", "ENSP1", "ENSP2"],
            "identifier": ["ENSP1", "ENSP1_alias", "ENSP2"],
        }
    )
    tg = sctg.pp.project_to_ordering(ad, toy_ordering, dictionary=dictionary)
    assert tg.var.loc["1", "n_genes"] == 2
    np.testing.assert_allclose(tg.X[0, 1], (2.0 + 4.0) / 2)  # mean of the two aliases
    np.testing.assert_allclose(tg.X[0, 2], 100.0)


def test_project_counts_summed_and_totals(toy_adata: AnnData, toy_ordering: Ordering) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering, counts_layer="counts")
    assert "counts" in tg.layers
    # total_counts is the per-unit library size over ALL genes.
    np.testing.assert_array_equal(
        tg.obs["total_counts"].to_numpy(), toy_adata.layers["counts"].sum(axis=1)
    )


def test_project_sparse_input_matches_dense(toy_adata: AnnData, toy_ordering: Ordering) -> None:
    dense = sctg.pp.project_to_ordering(toy_adata, toy_ordering).X
    sparse_ad = toy_adata.copy()
    sparse_ad.X = sp.csr_matrix(sparse_ad.X)
    sparse_out = sctg.pp.project_to_ordering(sparse_ad, toy_ordering).X
    np.testing.assert_allclose(sparse_out, dense)


def test_project_no_mapping_raises(toy_ordering: Ordering) -> None:
    ad = AnnData(X=np.ones((2, 2)), var=pd.DataFrame(index=["nope1", "nope2"]))
    with pytest.raises(ValueError, match="no genes could be mapped"):
        sctg.pp.project_to_ordering(ad, toy_ordering)


def test_project_uses_var_id_col(toy_ordering: Ordering) -> None:
    ad = AnnData(X=np.ones((1, 2)), var=pd.DataFrame({"ens": ["ENSP0", "ENSP2"]}, index=["a", "b"]))
    tg = sctg.pp.project_to_ordering(ad, toy_ordering, var_id_col="ens")
    np.testing.assert_array_equal(tg.uns["sctg"]["present_mask"], [True, False, True, False, False, False])


# --------------------------------------------------------------------------- transcriptogram


def test_transcriptogram_embedding_matches_masked_boxcar(
    toy_adata: AnnData, toy_ordering: Ordering
) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering, counts_layer="counts")
    step1 = tg.layers["step1"].copy()
    mask = tg.uns["sctg"]["present_mask"]
    sctg.tl.transcriptogram(tg, radius=1, mode="masked")

    expected = circular_boxcar(
        step1.T, 1, mode="masked", present_mask=mask.reshape(-1, 1)
    ).T
    np.testing.assert_allclose(tg.X, expected)
    np.testing.assert_allclose(tg.layers["transcriptogram"], expected)
    assert tg.uns["sctg"]["radius"] == 1
    assert tg.uns["sctg"]["window_size"] == 3


def test_transcriptogram_window_counts_are_sums(
    toy_adata: AnnData, toy_ordering: Ordering
) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering, counts_layer="counts")
    counts = tg.layers["counts"].copy()
    sctg.tl.transcriptogram(tg, radius=1)
    expected = circular_boxcar(counts.T, 1, mode="sum").T
    np.testing.assert_array_equal(tg.layers["window_counts"], expected)


def test_transcriptogram_no_counts_layer_skips_window_counts(
    toy_adata: AnnData, toy_ordering: Ordering
) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering)  # no counts_layer
    sctg.tl.transcriptogram(tg, radius=1)
    assert "window_counts" not in tg.layers


def test_transcriptogram_copy_does_not_mutate(
    toy_adata: AnnData, toy_ordering: Ordering
) -> None:
    tg = sctg.pp.project_to_ordering(toy_adata, toy_ordering)
    before = tg.X.copy()
    out = sctg.tl.transcriptogram(tg, radius=1, copy=True)
    assert out is not None
    np.testing.assert_array_equal(tg.X, before)  # original untouched
    assert "transcriptogram" in out.layers


def test_transcriptogram_default_radius_is_80(toy_adata: AnnData, toy_ordering: Ordering) -> None:
    # toy ordering is tiny, so an explicit small radius is needed there; here just assert the
    # signature default is 80 without invoking the (too-large-for-toy) window.
    import inspect

    assert inspect.signature(sctg.tl.transcriptogram).parameters["radius"].default == 80


def test_transcriptogram_requires_projection() -> None:
    bad = AnnData(X=np.ones((2, 4)))
    with pytest.raises(ValueError, match="project_to_ordering first"):
        sctg.tl.transcriptogram(bad, radius=1)


def test_end_to_end_on_real_ordering() -> None:
    # A small unit set projected onto the real human ordering, smoothed without error.
    o = sctg.datasets.human_ordering(900)
    genes = o.bare_proteins[:200]
    rng = np.random.default_rng(1)
    X = rng.poisson(3.0, size=(4, genes.size)).astype(float)
    ad = AnnData(X=X, var=pd.DataFrame(index=genes.astype(str)))
    ad.layers["counts"] = np.rint(X).astype(np.int64)
    tg = sctg.pp.project_to_ordering(ad, o, counts_layer="counts")
    assert tg.n_vars == o.n
    sctg.tl.transcriptogram(tg, radius=50)
    assert tg.layers["transcriptogram"].shape == (4, o.n)
    assert tg.layers["window_counts"].shape == (4, o.n)
    assert np.isfinite(tg.X).all()
