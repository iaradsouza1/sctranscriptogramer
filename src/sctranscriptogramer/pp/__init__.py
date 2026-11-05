"""Preprocessing (``sctranscriptogramer.pp``).

Turns a units x genes AnnData (metacells, spots, or samples) into a transcriptogram AnnData whose
features are the ordered protein positions.
"""

from __future__ import annotations

from sctranscriptogramer.pp.map_ids import project_to_ordering
from sctranscriptogramer.pp.metacells import aggregate_metacells, suggest_metacell_size
from sctranscriptogramer.pp.spatial import bin_spatial

__all__ = ["project_to_ordering", "aggregate_metacells", "suggest_metacell_size", "bin_spatial"]
