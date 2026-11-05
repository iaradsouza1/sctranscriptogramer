"""sctranscriptogramer: transcriptogram-based system-level analysis for single-cell,
single-nucleus and spatial transcriptomics.

The public API is organised scanpy-style into submodules:

- ``sctranscriptogramer.pp``  -- preprocessing (aggregation, projection onto the ordering)
- ``sctranscriptogramer.tl``  -- tools (transcriptogram, differential systems, clusters, enrichment)
- ``sctranscriptogramer.pl``  -- plotting
- ``sctranscriptogramer.datasets`` -- the bundled protein ordering(s) and on-demand annotations

These submodules are populated incrementally as the package is built; the core windowing
engine lives in :mod:`sctranscriptogramer._core`.
"""

from __future__ import annotations

from sctranscriptogramer import datasets, pp, tl
from sctranscriptogramer._core.ordering import Ordering
from sctranscriptogramer.ids import map_to_protein

__version__ = "0.0.1.dev0"

__all__ = ["__version__", "datasets", "pp", "tl", "Ordering", "map_to_protein"]
