"""Tests for the circular boxcar smoothing primitive.

Strategy: validate the fast cumsum implementation against a dead-simple brute-force circular
window reference, then assert the mathematical invariants the transcriptogram relies on
(sum-preservation, rotational equivariance, radius=0 identity, masked==fixed when nothing is
missing, chunking equivalence, and correct masked-denominator behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest

from sctranscriptogramer._core.boxcar import circular_boxcar, window_counts, window_sum


def brute_window_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Reference O(N*w) circular window sum; obviously-correct by construction."""
    n = values.shape[0]
    out = np.zeros_like(values, dtype=np.float64)
    for p in range(n):
        idx = [(p + off) % n for off in range(-radius, radius + 1)]
        out[p] = values[idx].sum(axis=0)
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --------------------------------------------------------------------------- window_sum


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 5])
def test_window_sum_matches_bruteforce(rng: np.random.Generator, radius: int) -> None:
    values = rng.normal(size=(17, 4))
    np.testing.assert_allclose(window_sum(values, radius), brute_window_sum(values, radius))


def test_window_sum_preserves_integer_dtype() -> None:
    values = np.arange(20, dtype=np.int32).reshape(10, 2)
    result = window_sum(values, 2)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, brute_window_sum(values, 2).astype(np.int64))


def test_window_sum_radius_zero_is_identity(rng: np.random.Generator) -> None:
    values = rng.normal(size=(8, 3))
    np.testing.assert_array_equal(window_sum(values, 0), values)


def test_window_full_circle_equals_total(rng: np.random.Generator) -> None:
    # window width >= n: every position sees the whole circle exactly once.
    values = rng.normal(size=(5, 3))
    total = values.sum(axis=0, keepdims=True)
    result = window_sum(values, radius=2)  # w = 5 == n
    np.testing.assert_allclose(result, np.repeat(total, 5, axis=0))


def test_window_sum_manual_small_case() -> None:
    # A tiny hand-computed circular example, radius=1, single column [1,2,3,4].
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    # p0: 4+1+2=7 ; p1: 1+2+3=6 ; p2: 2+3+4=9 ; p3: 3+4+1=8
    expected = np.array([[7.0], [6.0], [9.0], [8.0]])
    np.testing.assert_allclose(window_sum(values, 1), expected)


# --------------------------------------------------------------------------- invariants


def test_sum_mode_preserves_total_mass(rng: np.random.Generator) -> None:
    # Each element is counted in exactly w windows, so total sum scales by w.
    values = rng.normal(size=(13, 5))
    radius = 3
    w = 2 * radius + 1
    result = circular_boxcar(values, radius, mode="sum")
    np.testing.assert_allclose(result.sum(axis=0), values.sum(axis=0) * w)


def test_rotational_equivariance(rng: np.random.Generator) -> None:
    # Rotating the input rotates the output identically (circular symmetry).
    values = rng.normal(size=(11, 3))
    radius = 2
    shift = 4
    base = circular_boxcar(values, radius, mode="masked")
    rolled = circular_boxcar(np.roll(values, shift, axis=0), radius, mode="masked")
    np.testing.assert_allclose(rolled, np.roll(base, shift, axis=0))


def test_masked_equals_fixed_when_all_present(rng: np.random.Generator) -> None:
    values = rng.normal(size=(15, 4))
    radius = 3
    mask = np.ones_like(values)
    fixed = circular_boxcar(values, radius, mode="fixed")
    masked = circular_boxcar(values, radius, mode="masked", present_mask=mask)
    np.testing.assert_allclose(masked, fixed)


def test_fixed_mode_matches_bruteforce_mean(rng: np.random.Generator) -> None:
    values = rng.normal(size=(12, 3))
    radius = 2
    w = 2 * radius + 1
    np.testing.assert_allclose(
        circular_boxcar(values, radius, mode="fixed"),
        brute_window_sum(values, radius) / w,
    )


# --------------------------------------------------------------------------- masked semantics


def test_masked_denominator_is_window_occupancy() -> None:
    # 6 positions, single unit; positions 1 and 4 are missing.
    values = np.array([[10.0], [99.0], [30.0], [40.0], [99.0], [60.0]])
    mask = np.array([[1], [0], [1], [1], [0], [1]])
    radius = 1
    counts = window_counts(mask, radius)
    # p0 window {5,0,1}: present 5,0 -> 2 ; p1 {0,1,2}: 0,2 -> 2 ; p2 {1,2,3}: 2,3 -> 2
    # p3 {2,3,4}: 2,3 -> 2 ; p4 {3,4,5}: 3,5 -> 2 ; p5 {4,5,0}: 5,0 -> 2
    np.testing.assert_array_equal(counts.ravel(), [2, 2, 2, 2, 2, 2])

    masked = circular_boxcar(values, radius, mode="masked", present_mask=mask)
    # Missing values (99) must be ignored: p0 = (60+10)/2 = 35, p2 = (30+40)/2 ... etc.
    # p0 {5,0}: (60+10)/2=35 ; p1 {0,2}: (10+30)/2=20 ; p2 {2,3}:(30+40)/2=35
    # p3 {2,3}:(30+40)/2=35 ; p4 {3,5}:(40+60)/2=50 ; p5 {5,0}:(60+10)/2=35
    np.testing.assert_allclose(masked.ravel(), [35, 20, 35, 35, 50, 35])


def test_masked_missing_value_never_leaks() -> None:
    # A huge value at a masked position must not affect the result.
    a = np.array([[1.0], [1e9], [3.0]])
    mask = np.array([[1], [0], [1]])
    out_big = circular_boxcar(a, 1, mode="masked", present_mask=mask)
    a2 = a.copy()
    a2[1, 0] = -1234.0  # different masked value
    out_other = circular_boxcar(a2, 1, mode="masked", present_mask=mask)
    np.testing.assert_allclose(out_big, out_other)


def test_masked_all_missing_window_yields_zero() -> None:
    values = np.array([[5.0], [5.0], [5.0]])
    mask = np.zeros((3, 1))
    out = circular_boxcar(values, 1, mode="masked", present_mask=mask)
    np.testing.assert_array_equal(out, np.zeros((3, 1)))


def test_present_mask_broadcast_single_column(rng: np.random.Generator) -> None:
    values = rng.normal(size=(10, 4))
    mask_1col = np.ones((10, 1))
    mask_1col[3, 0] = 0
    mask_full = np.broadcast_to(mask_1col, (10, 4)).copy()
    a = circular_boxcar(values, 2, mode="masked", present_mask=mask_1col)
    b = circular_boxcar(values, 2, mode="masked", present_mask=mask_full)
    np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------- chunking


@pytest.mark.parametrize("mode", ["sum", "fixed", "masked"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 100])
def test_chunking_matches_unchunked(rng: np.random.Generator, mode: str, chunk_size: int) -> None:
    values = rng.normal(size=(14, 5))
    mask = (rng.random(size=(14, 5)) > 0.2).astype(np.int64)
    kwargs = {"present_mask": mask} if mode == "masked" else {}
    full = circular_boxcar(values, 3, mode=mode, **kwargs)
    chunked = circular_boxcar(values, 3, mode=mode, chunk_size=chunk_size, **kwargs)
    np.testing.assert_allclose(chunked, full)


# --------------------------------------------------------------------------- validation


def test_radius_must_be_smaller_than_n() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        circular_boxcar(np.ones((5, 2)), radius=5)


def test_negative_radius_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        circular_boxcar(np.ones((5, 2)), radius=-1)


def test_requires_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        circular_boxcar(np.ones(5), radius=1)


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        circular_boxcar(np.ones((5, 2)), radius=1, mode="bogus")  # type: ignore[arg-type]


def test_present_mask_wrong_positions_rejected() -> None:
    with pytest.raises(ValueError, match="same number of positions"):
        circular_boxcar(np.ones((5, 2)), 1, mode="masked", present_mask=np.ones((4, 2)))
