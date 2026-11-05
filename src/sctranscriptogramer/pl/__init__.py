"""Plotting (``sctranscriptogramer.pl``).

Requires matplotlib (the ``plot`` optional dependency). Imported as a submodule so the core package
stays matplotlib-free.
"""

from __future__ import annotations

from sctranscriptogramer.pl.diffmeans import difference_of_means
from sctranscriptogramer.pl.enrichment_rate import enrichment_rate

__all__ = ["difference_of_means", "enrichment_rate"]
