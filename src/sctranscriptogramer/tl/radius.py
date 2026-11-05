"""Choosing the transcriptogram window radius by null-corrected signal reproducibility.

The radius sets the smoothing bandwidth; too small is noisy, too large blurs distinct systems.
:func:`select_radius` scans candidate radii and, for each, measures how **reproducible the per-position
signal is across two independent halves of the data** -- minus the same under shuffled group labels, to
remove the trivial "smoothing inflates any correlation" effect. The radius maximising this
null-corrected reproducibility is the bias-variance sweet spot.

The per-position signal is the between-group F statistic for ``group_key`` (works for the 2-group
contrast and multi-group factors alike). Splits are taken over donors when ``donor_key`` is given (so
the halves are independent at the replicate level), else stratified over the groups. It re-windows the
stored ``step1`` layer, so no re-aggregation is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from sctranscriptogramer._core.boxcar import circular_boxcar

__all__ = ["select_radius"]


def _per_position_F(E: np.ndarray, grp: np.ndarray, k: int) -> np.ndarray:
    """Between-group / within-group F per position (E: present_positions x units)."""
    overall = E.mean(axis=1)
    n = E.shape[1]
    ssb = np.zeros(E.shape[0])
    ssw = np.zeros(E.shape[0])
    for g in np.unique(grp):
        m = grp == g
        ng = int(m.sum())
        if ng == 0:
            continue
        mg = E[:, m].mean(axis=1)
        ssb += ng * (mg - overall) ** 2
        ssw += ((E[:, m] - mg[:, None]) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ssb / (k - 1)) / (ssw / (n - k))


def _corr_finite(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    x, y = a[ok], b[ok]
    if x.std() == 0 or y.std() == 0:  # degenerate (e.g. over-wide window -> constant)
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def select_radius(
    tg: AnnData,
    *,
    group_key: str,
    donor_key: str | None = None,
    radii=(0, 5, 10, 20, 40, 80, 160, 320),
    step1_layer: str = "step1",
    present_only: bool = True,
    n_splits: int = 20,
    rule: str = "argmax",
    random_state: int = 0,
    key_added: str = "radius_selection",
) -> pd.DataFrame:
    """Scan radii and recommend one by null-corrected split-half reproducibility.

    Parameters
    ----------
    tg
        Projected transcriptogram AnnData (needs the unwindowed ``step1`` layer from
        ``project_to_ordering``).
    group_key
        ``obs`` column whose between-group signal is used (a condition, or a multi-level factor).
    donor_key
        If given (and >= 4 donors), the two halves are split over **whole donors** (so the halves are
        independent at the replicate level), stratified by each donor's majority group; otherwise
        units are split stratified by ``group_key``. Either way every split is required to contain
        **all groups in both halves** (splits that fail are redrawn).
    radii
        Candidate radii to evaluate (values >= n_positions are skipped).
    step1_layer
        Layer holding the pre-smoothing per-position values to re-window.
    present_only
        Restrict to mapped positions (``uns['sctg']['present_mask']``).
    n_splits
        Number of random half-splits to average the reproducibility over.
    rule
        ``"argmax"`` (default) picks the radius with the highest score; ``"one_se"`` picks the
        smallest radius whose score is within one standard deviation of the maximum (favouring less
        smoothing / more resolution).
    random_state
        Seed.
    key_added
        uns key to store the result under.

    Returns
    -------
    pandas.DataFrame
        Indexed by radius with columns ``reproducibility``, ``null``, ``score`` (= reproducibility -
        null) and ``score_sd``. ``df.attrs['best_radius']`` and ``tg.uns[key_added]`` hold the
        recommendation.
    """
    if step1_layer not in tg.layers:
        raise KeyError(
            f"layer {step1_layer!r} not found; run pp.project_to_ordering first (it writes 'step1')"
        )
    if group_key not in tg.obs:
        raise KeyError(f"column {group_key!r} not found in tg.obs")

    S1 = np.asarray(tg.layers[step1_layer]).T  # positions x units
    n_pos, n_units = S1.shape
    codes, levels = pd.factorize(tg.obs[group_key].to_numpy())
    k = len(levels)
    if k < 2:
        raise ValueError(f"group_key {group_key!r} needs >= 2 levels")

    mask = tg.uns.get("sctg", {}).get("present_mask")
    present = (
        np.asarray(mask, bool) if (present_only and mask is not None) else np.ones(n_pos, bool)
    )

    donors = tg.obs[donor_key].to_numpy() if donor_key is not None else None
    use_donor = donors is not None and len(np.unique(donors)) >= 4

    # For donor splits, stratify donors by their majority group so each half gets ~half the donors of
    # every group -- this keeps all groups in both halves even for donor-level factors (condition).
    donor_group = {}
    if use_donor:
        for d in np.unique(donors):
            dg = codes[donors == d]
            donor_group[d] = int(np.bincount(dg, minlength=k).argmax())

    def make_half(rng: np.random.Generator) -> np.ndarray:
        if use_donor:
            chosen: set = set()
            for g in range(k):
                dts = [d for d in donor_group if donor_group[d] == g]
                rng.shuffle(dts)
                chosen.update(dts[: len(dts) // 2])
            return np.isin(donors, list(chosen))
        half = np.zeros(n_units, bool)
        for g in np.unique(codes):
            idx = np.flatnonzero(codes == g)
            rng.shuffle(idx)
            half[idx[: len(idx) // 2]] = True
        return half

    def valid(half: np.ndarray) -> bool:
        # both halves must be non-trivial AND contain every group
        return (
            half.sum() >= k
            and (~half).sum() >= k
            and set(np.unique(codes[half])) == set(range(k))
            and set(np.unique(codes[~half])) == set(range(k))
        )

    # skip radii whose window spans (most of) the ordering -- the transcriptogram becomes constant.
    radii = [int(r) for r in radii if 2 * int(r) + 1 <= n_pos]
    if not radii:
        raise ValueError("no candidate radius has a window smaller than the ordering")
    rng = np.random.default_rng(random_state)
    # pre-draw valid splits so every radius is scored on the same partitions
    splits = []
    attempts = 0
    while len(splits) < n_splits and attempts < n_splits * 20:
        attempts += 1
        half = make_half(rng)
        if valid(half):
            splits.append((half, rng.permutation(codes)))
    if not splits:
        raise ValueError(
            "could not draw a split with all groups present in both halves; a group may be confined "
            "to too few donors — try donor_key=None (stratified split) or fewer/larger groups"
        )

    rows = []
    for r in radii:
        E = S1 if r == 0 else circular_boxcar(S1, r, mode="masked", present_mask=present[:, None])
        Ep = E[present]
        scores = []
        for half, perm in splits:
            Fa = _per_position_F(Ep[:, half], codes[half], k)
            Fb = _per_position_F(Ep[:, ~half], codes[~half], k)
            Na = _per_position_F(Ep[:, half], perm[half], k)
            Nb = _per_position_F(Ep[:, ~half], perm[~half], k)
            scores.append(_corr_finite(Fa, Fb) - _corr_finite(Na, Nb))
        scores = np.asarray(scores, dtype=float)
        n_ok = int(np.isfinite(scores).sum())
        sd = float(np.nanstd(scores))
        rows.append(
            {
                "radius": r,
                "reproducibility": np.nanmean(
                    [
                        _corr_finite(
                            _per_position_F(Ep[:, h], codes[h], k),
                            _per_position_F(Ep[:, ~h], codes[~h], k),
                        )
                        for h, _ in splits
                    ]
                ),
                "score": float(np.nanmean(scores)),
                "score_sd": sd,
                "score_se": sd / np.sqrt(max(n_ok, 1)),
            }
        )
    df = pd.DataFrame(rows).set_index("radius")
    df["null"] = df["reproducibility"] - df["score"]

    best_i = int(np.nanargmax(df["score"].to_numpy()))
    if rule == "one_se":
        # smallest radius within one standard error of the maximum (favours less smoothing)
        thr = df["score"].iloc[best_i] - df["score_se"].iloc[best_i]
        within = df.index[df["score"] >= thr]
        best_radius = int(min(within))
    elif rule == "argmax":
        best_radius = int(df.index[best_i])
    else:
        raise ValueError(f"rule must be 'argmax' or 'one_se', got {rule!r}")

    df.attrs["best_radius"] = best_radius
    tg.uns[key_added] = {
        "radii": list(df.index),
        "score": df["score"].to_numpy(),
        "score_sd": df["score_sd"].to_numpy(),
        "score_se": df["score_se"].to_numpy(),
        "reproducibility": df["reproducibility"].to_numpy(),
        "best_radius": best_radius,
        "params": {
            "group_key": group_key,
            "donor_key": donor_key,
            "split": "donor" if use_donor else "stratified",
            "n_splits": len(splits),
            "rule": rule,
        },
    }
    return df
