"""Core numerical primitives for sctranscriptogramer.

These modules have no dependency beyond numpy/scipy and are the performance-critical,
heavily-tested heart of the package.
"""

from __future__ import annotations

from sctranscriptogramer._core.boxcar import circular_boxcar, window_counts, window_sum
from sctranscriptogramer._core.ordering import Ordering

__all__ = ["circular_boxcar", "window_sum", "window_counts", "Ordering"]
