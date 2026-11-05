"""Tests for tl.cluster_enrichment (GSEApy offline ORA wrapper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctranscriptogramer as sctg

pytest.importorskip("gseapy")


@pytest.fixture
def tg_with_clusters() -> AnnData:
    n_pos = 30
    proteins = [f"P{i}" for i in range(n_pos)]
    var = pd.DataFrame(
        {"position": range(n_pos), "protein": proteins}, index=[str(i) for i in range(n_pos)]
    )
    tg = AnnData(X=np.zeros((2, n_pos)), var=var)
    cluster = np.zeros(n_pos, dtype=int)
    cluster[5:15] = 1
    is_de = np.zeros(n_pos, dtype=bool)
    is_de[5:15] = True
    df = pd.DataFrame(
        {
            "logFC": 0.0,
            "pValue": 0.0,
            "pAdj": 0.0,
            "tested": True,
            "is_de": is_de,
            "cluster": cluster,
        },
        index=var.index,
    )
    tg.varm["diffsystems_T"] = df
    tg.uns["diffsystems"] = {
        "contrasts": {"T": {"varm_key": "diffsystems_T", "clusters": [(5, 14)]}},
        "params": {"group_key": "cell_type"},
    }
    return tg


@pytest.fixture
def gene_sets() -> dict:
    return {
        "termA": [f"P{i}" for i in range(5, 15)],  # exactly the cluster -> enriched
        "termB": [f"P{i}" for i in range(20, 25)],  # disjoint -> not enriched
    }


def test_cluster_enrichment_finds_matching_term(tg_with_clusters, gene_sets) -> None:
    sctg.tl.cluster_enrichment(tg_with_clusters, gene_sets)
    res = tg_with_clusters.uns["enrichment"]
    assert "T" in res
    tbl = res["T"]
    assert "termA" in set(tbl["term"])
    assert "termB" not in set(tbl["term"])  # no overlap -> filtered
    row = tbl[tbl["term"] == "termA"].iloc[0]
    assert row["cluster"] == 1
    assert row["pAdj"] <= 0.05


def test_cluster_enrichment_use_de(tg_with_clusters, gene_sets) -> None:
    sctg.tl.cluster_enrichment(tg_with_clusters, gene_sets, use="de")
    assert "termA" in set(tg_with_clusters.uns["enrichment"]["T"]["term"])


def test_cluster_enrichment_respects_pvalue(tg_with_clusters, gene_sets) -> None:
    # An impossible threshold keeps nothing.
    sctg.tl.cluster_enrichment(tg_with_clusters, gene_sets, p_value=0.0)
    assert tg_with_clusters.uns["enrichment"].get("T") is None


def test_cluster_enrichment_requires_diffsystems() -> None:
    tg = AnnData(
        X=np.zeros((2, 3)),
        var=pd.DataFrame({"protein": ["a", "b", "c"]}, index=["0", "1", "2"]),
    )
    with pytest.raises(KeyError, match="diffsystems"):
        sctg.tl.cluster_enrichment(tg, {"t": ["a"]})


def test_cluster_enrichment_copy(tg_with_clusters, gene_sets) -> None:
    out = sctg.tl.cluster_enrichment(tg_with_clusters, gene_sets, copy=True)
    assert out is not None
    assert "enrichment" in out.uns
    assert "enrichment" not in tg_with_clusters.uns
