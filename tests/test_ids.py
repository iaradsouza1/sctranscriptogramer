"""Tests for the ID -> ENSEMBL peptide mapping (offline parsing/caching + gated live query)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import sctranscriptogramer as sctg
from sctranscriptogramer import ids


# --------------------------------------------------------------------------- parsing


def test_extract_proteins_dict() -> None:
    assert ids._extract_proteins({"protein": ["A", "B"]}) == ["A", "B"]


def test_extract_proteins_str() -> None:
    assert ids._extract_proteins({"protein": "A"}) == ["A"]


def test_extract_proteins_list_of_dicts() -> None:
    got = ids._extract_proteins([{"protein": ["A"]}, {"protein": "B"}])
    assert got == ["A", "B"]


def test_extract_proteins_missing() -> None:
    assert ids._extract_proteins({"gene": "x"}) == []


# --------------------------------------------------------------------------- validation


def test_unknown_species() -> None:
    with pytest.raises(ValueError, match="unknown species"):
        sctg.map_to_protein(["APOE"], species="elephant")


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        sctg.map_to_protein(["APOE"], backend="wat")


def test_unsupported_id_type(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unsupported id_type"):
        sctg.map_to_protein(["x"], id_type="banana", backend="mygene")


def test_empty_ids() -> None:
    with pytest.raises(ValueError, match="no identifiers"):
        sctg.map_to_protein([])


# --------------------------------------------------------------------------- caching (mocked)


def test_cache_roundtrip(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({"protein": ["ENSP1", "ENSP2"], "identifier": ["A", "A"]})
    calls = {"n": 0}

    def fake_map(ids_, id_type, species):
        calls["n"] += 1
        return fake.copy()

    monkeypatch.setattr(ids, "_map_mygene", fake_map)
    cache = tmp_path / "map.parquet"

    out1 = sctg.map_to_protein(["A"], cache_path=cache)
    assert cache.exists()
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(out1, fake)

    # second call must hit the cache, not the backend
    out2 = sctg.map_to_protein(["A"], cache_path=cache)
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(out2, fake)

    # refresh forces a re-query
    sctg.map_to_protein(["A"], cache_path=cache, refresh=True)
    assert calls["n"] == 2


def test_result_columns_and_dedup(monkeypatch) -> None:
    dupe = pd.DataFrame(
        {"protein": ["ENSP1", "ENSP1", "ENSP2"], "identifier": ["A", "A", "A"]}
    )
    monkeypatch.setattr(ids, "_map_mygene", lambda *a, **k: dupe.copy())
    out = sctg.map_to_protein(["A"])
    assert list(out.columns) == ["protein", "identifier"]
    assert len(out) == 2  # deduplicated


# --------------------------------------------------------------------------- live (network)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("SCTG_NO_NETWORK") == "1", reason="network disabled")
def test_live_mygene_symbol() -> None:
    pytest.importorskip("mygene")
    out = sctg.map_to_protein(["APOE", "TREM2"], id_type="symbol", species="human")
    assert set(out.columns) == {"protein", "identifier"}
    assert out["protein"].str.startswith("ENSP").all()
    assert {"APOE", "TREM2"} <= set(out["identifier"])
