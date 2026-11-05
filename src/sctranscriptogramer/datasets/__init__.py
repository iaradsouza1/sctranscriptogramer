"""Bundled and on-demand data assets.

The protein **ordering** (the only asset reused from the original R package) is bundled in the
wheel under ``_assets`` as parquet. ID mapping (gene -> ENSEMBL peptide) and GO annotations are
implemented natively / fetched on demand and cached; they are not bundled.
"""

from __future__ import annotations

from sctranscriptogramer.datasets._orderings import (
    available_orderings,
    human_ordering,
    load_ordering,
    mouse_ordering,
    rat_ordering,
    yeast_ordering,
)

__all__ = [
    "load_ordering",
    "available_orderings",
    "human_ordering",
    "mouse_ordering",
    "rat_ordering",
    "yeast_ordering",
]
