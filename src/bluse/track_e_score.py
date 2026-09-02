#!/usr/bin/env python
"""
Track E -- a morphology-only RFI score for every hit.

WHAT THIS PREDICTS, EXACTLY
---------------------------
`rfi_score` is P(the hit was seen in >=32 coherent beams | its features). It is
fitted on the multi-beam spatial filter's own verdicts, which are free: a
signal illuminating most of the 64-beam field is local interference, a signal
confined to one or two beams might be on the sky.

WHY THAT IS WORTH HAVING when `n_beams` is already in the catalogue: the
spatial filter needs many beams' worth of evidence, so it cannot judge a hit
that appears in ONE beam -- and a genuine technosignature appears in one beam.
So does a weak terrestrial emitter that only clears the detection threshold at
boresight. The filter cannot separate them. This model never sees a beam count,
so it can.

The evidence that it works is that it extrapolates. Trained only on <=2 beams
and >=32 beams, scored on the untrained 3-31 range, it reproduces beam
multiplicity monotonically across every bin (docs/track-e-2026-09.md).

WHAT IT IS NOT -- read this before quoting a number
---------------------------------------------------
The labels are POSITIVE-UNLABELLED. `weak_label == 0` means *seen in <=2
beams*; it does NOT mean verified clean. Most single-beam hits are still RFI --
the spatial filter is a sieve, not a proof. Therefore:

  * A HIGH score is strong evidence of RFI. The hit is morphologically
    indistinguishable from signals independently known to be terrestrial.
  * A LOW score is NOT evidence of a technosignature. It says only that the hit
    does not resemble the multi-beam RFI in this survey. Reading it the other
    way round is the most likely way this gets misused.

It is also a PROXY: it inherits every blind spot of the filter it learned from.
And brightness is partly confounded with the label, because a signal must be
bright to be detected in 32 beams -- within-SNR-decile AUC is 0.9243 against
0.9895 overall, so brightness contributes but does not carry the result.

DEFAULT FEATURE SET
-------------------
The 12 stamp-morphology columns, not all 16. It costs 0.0004 AUC (0.9891 vs
0.9895) and buys three things: it cannot relearn the RFI frequency mask, so it
is not circular with Track A's own band cut; it is the variant that transfers
to an unseen band (held-out L-band 0.9895 stamp against 0.9834 for all 16); and
"twelve numbers computed from the pixels" is a claim that can be defended.
`--features all` ships alongside it and both are reported.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import features as F

# The label. `weak_label` is derived from `n_beams` and `beam_frac`, so any
# feature set containing one of these predicts itself. test_track_e.py asserts
# no feature set can reach them -- being careful is not a guarantee.
LABEL_COLUMNS = frozenset({"n_beams", "n_beams_formed", "beam_frac",
                           "flag_multibeam", "weak_label", "weak_label_reason"})

# Track A's hand-built flags, carried as a baseline feature set: the honest
# comparison for "is morphology worth anything" is not chance, it is the
# classical filter we already have. Measured 0.9373 against stamp's 0.9891.
FLAG_COLUMNS = ["flag_rfi_band", "flag_zero_drift", "flag_drift_high",
                "flag_snr_low", "flag_snr_high", "flag_repeat"]

# mk_sample_hits was pre-filtered before delivery (BLUSE team, brainstorming.md
# Q4) and contains zero weak_label == 1 rows, so it can only ever contribute a
# biased block of negatives. Excluded from training by default; still scored.
# Measured cost of excluding it: 0.0012 AUC.
PREFILTERED_FILES = ("mk_sample_hits",)


def _registry_columns(kind):
    """Normalised feature columns of one kind, from the registry.

    Never hard-code the list. `features.REGISTRY` is the source of truth for
    what a feature is and whether it comes from the stamp or the metadata, and
    a literal here would go stale the first time a feature is added.
    """
    return [c + "_n" for c in F.all_columns(kind=kind)
            if not c.endswith("_saturated")]


FEATURE_SETS = {
    "stamp": _registry_columns("stamp"),                       # 12, the default
    "meta": _registry_columns("meta"),                         # 4
    "all": _registry_columns("meta") + _registry_columns("stamp"),
    "flags": list(FLAG_COLUMNS),
}


def feature_columns(name):
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; "
                         f"choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])


def fit_score(df, *, features="stamp", n_splits=5, seed=0, exclude_mk=True,
              max_iter=300):
    """
    Fit group-fold models on the spatial filter's labels and score every row.

    Cross-validation here is the DELIVERY MECHANISM, not just the audit. Every
    labelled row carries its out-of-fold score, so no row is ever scored by a
    model that saw its observation; every unlabelled row carries the mean of
    the `n_splits` fold models, and was in no training fold at all. That makes
    the shipped column honest everywhere, which a single fit on everything
    would not be.

    Splits on `group_id` (the obsid). Hits from one observation share a
    pointing, an RFI environment and a calibration, so a random row split
    leaks.

    Returns (score, fold, info). `fold` is -1 for any row in no training fold.
    """
    cols = feature_columns(features)
    missing = [c for c in cols + ["weak_label", "group_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    X = df[cols].to_numpy(np.float32)
    y = df["weak_label"].to_numpy().astype(np.int8)
    groups = df["group_id"].to_numpy()

    # A column that is non-finite everywhere gives the histogram binner zero
    # bins, and the failure surfaces from inside joblib as "window shape cannot
    # be larger than input array shape" -- naming neither the column nor the
    # cause. Reachable in practice: f08_turning_bw_hz is unresolved for ~72% of
    # hits, so a small enough subset can be all-NaN in it. Drop such columns,
    # say which, and record it, rather than crash.
    usable = np.isfinite(X).any(axis=0)
    dropped = [c for c, ok in zip(cols, usable) if not ok]
    if dropped:
        print(f"  WARNING: dropping {len(dropped)} all-NaN feature column(s): "
              f"{', '.join(dropped)}")
        if not usable.any():
            raise ValueError("every feature column is non-finite")
        X = X[:, usable]
        cols = [c for c, ok in zip(cols, usable) if ok]

    train = y >= 0
    if exclude_mk and "file" in df.columns:
        train &= ~df["file"].isin(PREFILTERED_FILES).to_numpy()
    tr_idx = np.flatnonzero(train)
    if len(np.unique(y[train])) < 2:
        raise ValueError("training rows carry only one class")
    n_groups = len(np.unique(groups[train]))
    if n_groups < n_splits:
        raise ValueError(f"{n_groups} groups is fewer than n_splits={n_splits}")

    score = np.full(len(df), np.nan)
    fold = np.full(len(df), -1, dtype=np.int8)
    rest = np.flatnonzero(~train)
    acc = np.zeros(len(rest))

    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits)
                                 .split(tr_idx, y[tr_idx], groups[tr_idx])):
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, early_stopping=False, random_state=seed)
        model.fit(X[tr_idx[tr]], y[tr_idx[tr]])
        score[tr_idx[te]] = model.predict_proba(X[tr_idx[te]])[:, 1]
        fold[tr_idx[te]] = k
        if len(rest):
            acc += model.predict_proba(X[rest])[:, 1]
        folds.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te))})
    if len(rest):
        score[rest] = acc / n_splits

    info = {
        "features": features,
        "columns": cols,
        "dropped_columns": dropped,
        "n_splits": n_splits,
        "seed": seed,
        "max_iter": max_iter,
        "exclude_mk": bool(exclude_mk),
        "n_rows": int(len(df)),
        "n_train": int(train.sum()),
        "n_groups": int(n_groups),
        "n_scored_out_of_fold": int(train.sum()),
        "n_scored_by_fold_mean": int(len(rest)),
        "base_rate": float(y[train].mean()),
        "folds": folds,
    }
    return score, fold, info
