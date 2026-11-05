"""Tests for the Ordering container and the bundled-ordering loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sctranscriptogramer import datasets
from sctranscriptogramer._core.ordering import Ordering


# --------------------------------------------------------------------------- Ordering


def test_from_proteins_basic() -> None:
    o = Ordering.from_proteins(["9606.ENSP1", "9606.ENSP2", "9606.ENSP3"], name="toy")
    assert o.n == 3
    assert len(o) == 3
    np.testing.assert_array_equal(o.positions, [0, 1, 2])
    assert o.name == "toy"


def test_positions_of_full_and_bare() -> None:
    o = Ordering.from_proteins(["9606.ENSPa", "9606.ENSPb", "9606.ENSPc"])
    # Full ids
    np.testing.assert_array_equal(o.positions_of(["9606.ENSPb", "9606.ENSPa"]), [1, 0])
    # Bare ids resolve to the same positions
    np.testing.assert_array_equal(o.positions_of(["ENSPc", "ENSPa"]), [2, 0])
    # Missing -> -1
    np.testing.assert_array_equal(o.positions_of(["ENSPz"]), [-1])


def test_contains() -> None:
    o = Ordering.from_proteins(["9606.ENSPa", "9606.ENSPb"])
    np.testing.assert_array_equal(o.contains(["ENSPa", "ENSPz", "9606.ENSPb"]), [True, False, True])


def test_taxonomy_prefix_and_bare() -> None:
    o = Ordering.from_proteins(["9606.ENSP1", "9606.ENSP2"])
    assert o.taxonomy_prefix == "9606."
    np.testing.assert_array_equal(o.bare_proteins, ["ENSP1", "ENSP2"])


def test_no_prefix_ordering() -> None:
    o = Ordering.from_proteins(["ENSP1", "ENSP2"])
    assert o.taxonomy_prefix is None
    np.testing.assert_array_equal(o.bare_proteins, ["ENSP1", "ENSP2"])
    np.testing.assert_array_equal(o.positions_of(["ENSP2"]), [1])


def test_duplicate_protein_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Ordering.from_proteins(["9606.ENSP1", "9606.ENSP1"])


def test_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        Ordering.from_proteins([])


def test_from_frame_sorts_by_position() -> None:
    df = pd.DataFrame({"protein": ["c", "a", "b"], "position": [2, 0, 1]})
    o = Ordering.from_frame(df)
    np.testing.assert_array_equal(o.proteins.astype(str), ["a", "b", "c"])


def test_from_frame_noncontiguous_rejected() -> None:
    df = pd.DataFrame({"protein": ["a", "b"], "position": [0, 2]})
    with pytest.raises(ValueError, match="contiguous"):
        Ordering.from_frame(df)


def test_roundtrip_frame() -> None:
    o = Ordering.from_proteins(["a", "b", "c"])
    o2 = Ordering.from_frame(o.to_frame())
    np.testing.assert_array_equal(o.proteins.astype(str), o2.proteins.astype(str))


# --------------------------------------------------------------------------- loaders


def test_available_orderings_full_set() -> None:
    names = set(datasets.available_orderings())
    expected = {f"{sp}{thr}" for sp in ("Hs", "Mm", "Rn", "Sc") for thr in (700, 800, 900)}
    assert names == expected


def test_human_ordering_shape() -> None:
    o = datasets.human_ordering(900)
    assert o.n == 12396  # matches the R Hs900 length
    assert o.species == "Hs"
    assert o.threshold == 900
    assert o.name == "Hs900"
    assert o.taxonomy_prefix == "9606."


@pytest.mark.parametrize(
    "species,threshold,n",
    [
        ("Hs", 700, 17185),
        ("mouse", 900, 11141),
        ("Rn", 800, 13887),
        ("yeast", 900, 5213),
    ],
)
def test_load_ordering_sizes(species: str, threshold: int, n: int) -> None:
    assert datasets.load_ordering(species, threshold).n == n


def test_load_ordering_positions_are_contiguous() -> None:
    o = datasets.human_ordering(900)
    np.testing.assert_array_equal(o.positions, np.arange(o.n))


def test_unknown_species_rejected() -> None:
    with pytest.raises(ValueError, match="unknown species"):
        datasets.load_ordering("elephant", 900)


def test_bad_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="threshold must be"):
        datasets.load_ordering("Hs", 500)


def test_loader_is_cached() -> None:
    assert datasets.load_ordering("Hs", 900) is datasets.load_ordering("human", 900)
