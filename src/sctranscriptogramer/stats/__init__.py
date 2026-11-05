"""Statistics backends for sctranscriptogramer.

The differential-systems statistics use a negative-binomial mixed model (edgePython / NEBULA-LN),
which handles the metacell pseudoreplication via a donor random effect and supports covariates.
"""

from __future__ import annotations

from sctranscriptogramer.stats.multitest import benjamini_hochberg
from sctranscriptogramer.stats.nb_mixed import fit_nb_mixed

__all__ = ["benjamini_hochberg", "fit_nb_mixed"]
