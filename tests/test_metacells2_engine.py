"""Integration test for the real metacells2 engine.

Skipped automatically if the optional ``metacells`` package is not installed. Kept small so it is
cheap enough to run, while still exercising the partition-first wiring against the real algorithm.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

pytest.importorskip("metacells")

import sctranscriptogramer as sctg  # noqa: E402


@pytest.fixture(scope="module")
def structured_cells() -> AnnData:
    rng = np.random.default_rng(0)
    n_per = 120
    blocks, obs_rows = [], []
    for cond in ["case", "ctrl"]:
        for ct in ["T", "B"]:
            base = rng.gamma(0.3, size=250) * (2.0 if ct == "T" else 1.0)
            blocks.append(rng.poisson(base, size=(n_per, 250)))
            obs_rows += [(cond, ct, d) for d in rng.choice(["d1", "d2"], size=n_per)]
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame(obs_rows, columns=["condition", "cell_type", "donor"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    ad = AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(250)]))
    ad.layers["counts"] = X.astype(np.int64)
    return ad


@pytest.mark.slow
def test_metacells2_end_to_end_purity(structured_cells: AnnData) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = sctg.pp.aggregate_metacells(
            structured_cells,
            group_keys=["condition", "cell_type", "donor"],
            engine="metacells2",
            counts_layer="counts",
            engine_kwargs={"target_metacell_size": 30},
        )

    # Every metacell is pure in all grouping columns and non-empty.
    assert mc.n_obs >= 8  # at least one per (condition x cell_type x donor) partition
    assert (mc.obs["n_cells"] > 0).all()
    assert set(mc.obs["condition"]) <= {"case", "ctrl"}
    assert set(mc.obs["cell_type"]) <= {"T", "B"}
    assert set(mc.obs["donor"]) <= {"d1", "d2"}
    assert "counts" in mc.layers
    assert mc.uns["sctg_metacells"]["engine"] == "metacells2"


@pytest.mark.slow
def test_metacells2_feeds_transcriptogram(structured_cells: AnnData) -> None:
    # metacells -> project onto (a slice of) the real ordering -> transcriptogram.
    o = sctg.datasets.human_ordering(900)
    genes = o.bare_proteins[:250].astype(str)
    ad = structured_cells.copy()
    ad.var_names = genes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = sctg.pp.aggregate_metacells(
            ad,
            group_keys=["condition", "cell_type", "donor"],
            engine="metacells2",
            counts_layer="counts",
            engine_kwargs={"target_metacell_size": 30},
        )
    tg = sctg.pp.project_to_ordering(mc, o, counts_layer="counts")
    sctg.tl.transcriptogram(tg, radius=50)
    assert tg.n_vars == o.n
    assert np.isfinite(tg.X).all()
    assert "window_counts" in tg.layers
