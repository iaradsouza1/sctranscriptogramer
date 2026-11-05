#!/usr/bin/env python
"""Maintainer one-off: convert the CSVs from export_orderings_from_r.R into the parquet
ordering assets bundled in the wheel.

Each ordering ``<Sp><Threshold>`` (e.g. ``Hs900``) becomes
``src/sctranscriptogramer/datasets/_assets/ordering_<Sp>_<Threshold>.parquet`` with columns:

- ``protein``  : ENSEMBL peptide id, taxonomy-prefixed as shipped (e.g. ``9606.ENSP...``)
- ``position`` : int32, 0-based ordering position

Usage:
    python build_ordering_assets.py <csv_dir> <assets_out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SPECIES = {"Hs": "Hs", "Mm": "Mm", "Rn": "Rn", "Sc": "Sc"}
THRESHOLDS = ("700", "800", "900")


def main(csv_dir: str, out_dir: str) -> None:
    csv_path = Path(csv_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written = 0
    for sp in SPECIES:
        for thr in THRESHOLDS:
            src = csv_path / f"{sp}{thr}.csv"
            if not src.exists():
                print(f"skip (missing): {src}")
                continue
            df = pd.read_csv(src, dtype={"protein": "string"})
            df["position"] = df["position"].astype("int32")
            # Integrity: positions must be exactly 0..N-1 in order.
            n = len(df)
            expected = pd.Series(range(n), dtype="int32")
            if not df["position"].reset_index(drop=True).equals(expected):
                raise ValueError(f"{src.name}: positions are not a contiguous 0..N-1 range")
            if df["protein"].duplicated().any():
                raise ValueError(f"{src.name}: contains duplicate proteins")
            dst = out_path / f"ordering_{sp}_{thr}.parquet"
            df[["protein", "position"]].to_parquet(dst, index=False)
            print(f"wrote {dst} ({n} proteins)")
            written += 1

    if written == 0:
        raise SystemExit("no orderings converted; check the csv_dir path")
    print(f"done: {written} orderings")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_ordering_assets.py <csv_dir> <assets_out_dir>")
    main(sys.argv[1], sys.argv[2])
