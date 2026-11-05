"""Tests for cluster identification (contiguous runs + periodic pad/merge/wrap)."""

from __future__ import annotations

import numpy as np

from sctranscriptogramer.tl.clusters import identify_clusters


def test_empty_significant() -> None:
    r = identify_clusters(np.array([], dtype=int), radius=2, n_positions=20)
    assert r.intervals == []
    assert r.n_clusters == 0
    assert not r.pbc
    assert (r.labels == 0).all()


def test_nonperiodic_runs_split_on_gap() -> None:
    sig = np.array([3, 4, 5, 12, 13])
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=False)
    assert r.intervals == [(3, 5), (12, 13)]
    assert not r.pbc
    assert list(np.flatnonzero(r.labels == 1)) == [3, 4, 5]
    assert list(np.flatnonzero(r.labels == 2)) == [12, 13]


def test_gap_equal_radius_stays_one_run() -> None:
    # gap of exactly radius does NOT split
    sig = np.array([3, 5])  # gap 2 == radius
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=False)
    assert r.intervals == [(3, 5)]


def test_gap_greater_than_radius_splits() -> None:
    sig = np.array([3, 6])  # gap 3 > radius 2
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=False)
    assert r.intervals == [(3, 3), (6, 6)]


def test_periodic_padding_no_wrap() -> None:
    sig = np.array([3, 4, 5, 12, 13])
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=True)
    # padded [1,7] and [10,15]; disjoint -> no merge, no wrap
    assert r.intervals == [(1, 7), (10, 15)]
    assert not r.pbc
    assert r.n_clusters == 2


def test_periodic_merge_overlap() -> None:
    sig = np.array([3, 8])  # gap 5 > radius 2 -> two runs, padded [1,5] and [6,10]
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=True)
    # padded intervals touch (5 and 6 within merge since 6 <= 5? no) -> check radius 3
    r3 = identify_clusters(sig, radius=3, n_positions=20, periodic=True)
    # radius 3: runs still split (gap 5 > 3); padded [0,6] and [5,11] overlap -> merge [0,11]
    assert r3.intervals == [(0, 11)]
    assert r3.n_clusters == 1
    # radius 2 stays split
    assert r.intervals == [(1, 5), (6, 10)]


def test_periodic_wrap_front() -> None:
    # significant at both ends -> wrap merges first and last into one cluster
    sig = np.array([0, 1, 18, 19])
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=True)
    assert r.pbc
    assert r.intervals == [(0, 3), (16, 19)]
    assert r.n_clusters == 1  # wrapped pair counts once
    # both wrapped intervals share label 1
    assert set(r.labels[[0, 1, 2, 3, 16, 17, 18, 19]]) == {1}


def test_periodic_wrap_back_overhang() -> None:
    sig = np.array([10, 11, 19])
    r = identify_clusters(sig, radius=2, n_positions=20, periodic=True)
    assert r.pbc
    # last run padded to [17,21] overhangs -> wraps a [0,x] segment to the front
    assert (0, 1) in r.intervals
    assert r.intervals[-1][1] == 19


def test_labels_within_bounds() -> None:
    sig = np.array([0, 1, 18, 19])
    r = identify_clusters(sig, radius=5, n_positions=20, periodic=True)
    assert r.labels.shape == (20,)
    assert r.labels.min() >= 0
