"""The protein ordering: the fixed 1D arrangement onto which expression is projected.

An :class:`Ordering` wraps the ordered list of proteins (ENSEMBL peptide ids) and their 0-based
positions. Its job is to map arbitrary protein identifiers onto ordering positions so that
expression can be scattered into the dense ``(n_positions, n_units)`` layout the boxcar operates on.

Proteins in the shipped orderings are taxonomy-prefixed (e.g. ``9606.ENSP00000366179``), whereas
ID-mapping tools usually return bare peptide ids (``ENSP00000366179``). To make both work, lookups
match against the full id *and* its bare (prefix-stripped) form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["Ordering"]

_TAX_PREFIX = re.compile(r"^(\d+)\.")


def _strip_prefix(protein: str) -> str:
    return _TAX_PREFIX.sub("", protein, count=1)


@dataclass(frozen=True)
class Ordering:
    """A fixed 1D protein ordering.

    Parameters
    ----------
    proteins
        Ordered protein ids (length ``N``), position ``i`` holds ``proteins[i]``.
    species
        Optional two-letter species code (``"Hs"``, ``"Mm"``, ``"Rn"``, ``"Sc"``).
    threshold
        Optional STRING combined-score threshold the ordering was built at (700/800/900).
    name
        Optional human-readable name (e.g. ``"Hs900"``).
    """

    proteins: NDArray[np.str_]
    species: str | None = None
    threshold: int | None = None
    name: str | None = None
    _lookup: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        proteins = np.asarray(self.proteins, dtype=object)
        if proteins.ndim != 1:
            raise ValueError("proteins must be 1D")
        if proteins.size == 0:
            raise ValueError("ordering is empty")
        object.__setattr__(self, "proteins", proteins)

        # Build a combined lookup so both full and bare ids resolve to a position.
        lookup: dict[str, int] = {}
        for pos, protein in enumerate(proteins):
            protein = str(protein)
            if protein in lookup:
                raise ValueError(f"duplicate protein in ordering: {protein!r}")
            lookup[protein] = pos
            bare = _strip_prefix(protein)
            if bare != protein:
                # Bare form should also be unique; only record the first if collision-free.
                lookup.setdefault(bare, pos)
        object.__setattr__(self, "_lookup", lookup)

    # ------------------------------------------------------------------ basic properties

    @property
    def n(self) -> int:
        """Number of proteins (positions) in the ordering."""
        return int(self.proteins.size)

    def __len__(self) -> int:
        return self.n

    @property
    def positions(self) -> NDArray[np.int32]:
        """0-based positions ``0..N-1``."""
        return np.arange(self.n, dtype=np.int32)

    @property
    def taxonomy_prefix(self) -> str | None:
        """The taxonomy prefix (e.g. ``"9606."``) if proteins are prefixed, else ``None``."""
        match = _TAX_PREFIX.match(str(self.proteins[0]))
        return match.group(0) if match else None

    @property
    def bare_proteins(self) -> NDArray[np.str_]:
        """Proteins with any taxonomy prefix stripped."""
        return np.array([_strip_prefix(str(p)) for p in self.proteins], dtype=object)

    # ------------------------------------------------------------------ mapping

    def positions_of(self, protein_ids: Iterable[str]) -> NDArray[np.int64]:
        """Position of each id, or ``-1`` if absent from the ordering.

        Matches the full id first, then its bare (prefix-stripped) form, so both
        ``9606.ENSP...`` and ``ENSP...`` inputs resolve correctly.
        """
        ids = np.asarray(list(protein_ids), dtype=object)
        out = np.full(ids.shape[0], -1, dtype=np.int64)
        lookup = self._lookup
        for i, pid in enumerate(ids):
            pid = str(pid)
            pos = lookup.get(pid)
            if pos is None:
                pos = lookup.get(_strip_prefix(pid))
            if pos is not None:
                out[i] = pos
        return out

    def contains(self, protein_ids: Iterable[str]) -> NDArray[np.bool_]:
        """Boolean mask of which ids are present in the ordering."""
        return self.positions_of(protein_ids) >= 0

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_proteins(
        cls,
        proteins: Iterable[str],
        *,
        species: str | None = None,
        threshold: int | None = None,
        name: str | None = None,
    ) -> "Ordering":
        """Build from an ordered iterable of protein ids (position = iteration order)."""
        return cls(
            proteins=np.asarray(list(proteins), dtype=object),
            species=species,
            threshold=threshold,
            name=name,
        )

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        protein_col: str = "protein",
        position_col: str = "position",
        species: str | None = None,
        threshold: int | None = None,
        name: str | None = None,
    ) -> "Ordering":
        """Build from a DataFrame with protein and (0-based) position columns.

        The frame is sorted by ``position_col`` and the positions are checked to form a
        contiguous ``0..N-1`` range.
        """
        if protein_col not in df or position_col not in df:
            raise KeyError(f"frame must have columns {protein_col!r} and {position_col!r}")
        ordered = df.sort_values(position_col)
        positions = ordered[position_col].to_numpy()
        expected = np.arange(len(ordered))
        if not np.array_equal(positions, expected):
            raise ValueError("positions must be a contiguous 0..N-1 range")
        return cls(
            proteins=ordered[protein_col].to_numpy().astype(object),
            species=species,
            threshold=threshold,
            name=name,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a ``(protein, position)`` DataFrame."""
        return pd.DataFrame({"protein": self.proteins.astype(str), "position": self.positions})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        label = self.name or "Ordering"
        return f"<{label}: {self.n} proteins" + (
            f", {self.species} @ {self.threshold}>" if self.species else ">"
        )
