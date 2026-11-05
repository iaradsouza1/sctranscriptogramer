"""Smoke tests for the plotting functions (matplotlib Agg backend)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

matplotlib.use("Agg")

pytest.importorskip("matplotlib")

import sctranscriptogramer.pl as pl  # noqa: E402


@pytest.fixture
def tg_plot() -> AnnData:
    rng = np.random.default_rng(0)
    n_pos, n_mc = 40, 8
    obs = pd.DataFrame(
        {
            "condition": ["ctrl"] * 4 + ["case"] * 4,
            "cell_type": ["T"] * 8,
        },
        index=[f"mc{i}" for i in range(n_mc)],
    )
    emb = rng.normal(size=(n_mc, n_pos))
    emb[4:, 10:21] += 2.0  # case elevated in a block
    var = pd.DataFrame(
        {"position": range(n_pos), "protein": [f"P{i}" for i in range(n_pos)]},
        index=[str(i) for i in range(n_pos)],
    )
    tg = AnnData(X=emb, obs=obs, var=var)
    tg.layers["transcriptogram"] = emb
    tg.uns["sctg"] = {"radius": 3}
    tg.uns["diffsystems"] = {
        "params": {
            "condition_key": "condition",
            "case": "case",
            "control": "ctrl",
            "group_key": "cell_type",
        },
        "contrasts": {"T": {"clusters": [(10, 20)], "varm_key": "diffsystems_T"}},
    }
    logfc = np.zeros(n_pos)
    logfc[10:21] = 1.5  # elevated block
    is_de = np.zeros(n_pos, dtype=bool)
    is_de[10:21] = True
    tg.varm["diffsystems_T"] = pd.DataFrame(
        {
            "logFC": logfc,
            "pValue": 0.0,
            "pAdj": 0.0,
            "tested": True,
            "is_de": is_de,
            "cluster": [1 if 10 <= i <= 20 else 0 for i in range(n_pos)],
        },
        index=var.index,
    )
    return tg


def _line(ax, label):
    return next(ln for ln in ax.lines if ln.get_label() == label)


def test_difference_of_means_logfc_default(tg_plot: AnnData) -> None:
    ax = pl.difference_of_means(tg_plot, contrast="T")  # values="logFC" by default
    assert ax.get_xlabel() == "Gene position"
    assert ax.get_ylabel() == "logFC (case vs control, log2)"
    # the plotted line IS the model logFC; its peak sits in the elevated block
    y = _line(ax, "logFC").get_ydata()
    assert 10 <= int(np.argmax(y)) <= 20


def test_difference_of_means_difference_mode(tg_plot: AnnData) -> None:
    ax = pl.difference_of_means(tg_plot, contrast="T", values="difference")
    assert ax.get_ylabel() == "Difference of means (case - control)"
    y = _line(ax, "difference of means").get_ydata()
    assert 5 <= int(np.argmax(y)) <= 25


def test_difference_of_means_smooth_adds_line(tg_plot: AnnData) -> None:
    n0 = len(pl.difference_of_means(tg_plot, contrast="T", smooth=False).lines)
    n1 = len(pl.difference_of_means(tg_plot, contrast="T", smooth=True).lines)
    assert n1 == n0 + 1


def test_difference_of_means_logfc_missing_varm(tg_plot: AnnData) -> None:
    # contrast without a stored varm table -> clear error suggesting difference mode
    with pytest.raises(ValueError, match="no differential-systems result"):
        pl.difference_of_means(tg_plot, contrast="B")


def test_difference_of_means_difference_missing_params() -> None:
    tg = AnnData(X=np.zeros((2, 3)), var=pd.DataFrame(index=["0", "1", "2"]))
    with pytest.raises(ValueError, match="condition_key"):
        pl.difference_of_means(tg, values="difference")


def test_difference_of_means_bad_values(tg_plot: AnnData) -> None:
    with pytest.raises(ValueError, match="values must be"):
        pl.difference_of_means(tg_plot, contrast="T", values="bogus")


def test_enrichment_rate_returns_axes(tg_plot: AnnData) -> None:
    annotations = {"termA": [f"P{i}" for i in range(10, 21)]}
    ax = pl.enrichment_rate(tg_plot, annotations, ["termA"], contrast="T")
    assert ax.get_ylabel() == "Term rate by window"
    assert ax.get_ylim() == (0.0, 1.0)
    # the term's rate should peak within its annotated block
    smoothed = ax.lines[-1].get_ydata()
    assert 5 <= int(np.argmax(smoothed)) <= 25


def test_enrichment_rate_dataframe_annotations(tg_plot: AnnData) -> None:
    ann = pd.DataFrame(
        {"protein": [f"P{i}" for i in range(10, 21)], "go_id": ["termA"] * 11}
    )
    ax = pl.enrichment_rate(tg_plot, ann, ["termA"])
    assert ax is not None
