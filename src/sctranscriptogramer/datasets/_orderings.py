"""Loaders for the bundled protein orderings.

The orderings are the only assets reused from the R ``transcriptogramer`` package. They live as
parquet under ``_assets/ordering_<Sp>_<Threshold>.parquet`` and are loaded into
:class:`~sctranscriptogramer._core.ordering.Ordering` objects.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files

import pandas as pd

from sctranscriptogramer._core.ordering import Ordering

__all__ = [
    "load_ordering",
    "available_orderings",
    "human_ordering",
    "mouse_ordering",
    "rat_ordering",
    "yeast_ordering",
]

_SPECIES = {"Hs": "human", "Mm": "mouse", "Rn": "rat", "Sc": "yeast"}
_ALIASES = {v: k for k, v in _SPECIES.items()}
_THRESHOLDS = (700, 800, 900)


def _asset_traversable(species_code: str, threshold: int):
    return files("sctranscriptogramer.datasets._assets").joinpath(
        f"ordering_{species_code}_{threshold}.parquet"
    )


def available_orderings() -> list[str]:
    """Names of the bundled orderings actually present (e.g. ``["Hs700", ..., "Sc900"]``)."""
    present = []
    for code in _SPECIES:
        for thr in _THRESHOLDS:
            if _asset_traversable(code, thr).is_file():
                present.append(f"{code}{thr}")
    return present


def _resolve_species(species: str) -> str:
    """Normalise a species argument to its two-letter code."""
    if species in _SPECIES:
        return species
    key = species.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"unknown species {species!r}; expected one of "
        f"{sorted(_SPECIES) + sorted(_ALIASES)}"
    )


@lru_cache(maxsize=None)
def _load_by_code(code: str, threshold: int) -> Ordering:
    resource = _asset_traversable(code, threshold)
    if not resource.is_file():
        raise FileNotFoundError(
            f"ordering asset for {code}{threshold} is not bundled; "
            f"available: {available_orderings()}"
        )
    with as_file(resource) as path:
        df = pd.read_parquet(path)
    return Ordering.from_frame(
        df, species=code, threshold=threshold, name=f"{code}{threshold}"
    )


def load_ordering(species: str = "human", threshold: int = 900) -> Ordering:
    """Load a bundled ordering.

    Species aliases resolve to the same cached object, so ``load_ordering("Hs")`` and
    ``load_ordering("human")`` return the identical instance.

    Parameters
    ----------
    species
        Two-letter code (``"Hs"``/``"Mm"``/``"Rn"``/``"Sc"``) or full name
        (``"human"``/``"mouse"``/``"rat"``/``"yeast"``).
    threshold
        STRING combined-score threshold the ordering was built at: 700, 800 or 900.

    Returns
    -------
    Ordering
    """
    code = _resolve_species(species)
    if threshold not in _THRESHOLDS:
        raise ValueError(f"threshold must be one of {_THRESHOLDS}, got {threshold}")
    return _load_by_code(code, threshold)


def human_ordering(threshold: int = 900) -> Ordering:
    """Load the human (Hs) ordering at the given STRING threshold."""
    return load_ordering("Hs", threshold)


def mouse_ordering(threshold: int = 900) -> Ordering:
    """Load the mouse (Mm) ordering at the given STRING threshold."""
    return load_ordering("Mm", threshold)


def rat_ordering(threshold: int = 900) -> Ordering:
    """Load the rat (Rn) ordering at the given STRING threshold."""
    return load_ordering("Rn", threshold)


def yeast_ordering(threshold: int = 900) -> Ordering:
    """Load the yeast (Sc) ordering at the given STRING threshold."""
    return load_ordering("Sc", threshold)
