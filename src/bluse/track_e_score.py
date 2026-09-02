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

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

# Which band each delivered file observes. Used only to hold a whole band out:
# "general RFI morphology or memorised band-specific emitters?" cannot be
# answered by any split that keeps the band in training.
BANDS = {"lband_long": "L", "lband_short": "L", "uhf_long": "UHF",
         "uhf_short": "UHF", "sband_long": "S", "sband_short": "S",
         "mk_sample_hits": "MK"}

# The n_beams bins the monotonicity check reports. The gap between 2 and 32 is
# the whole point: those five bins are in NO training fold, so a monotone score
# across them is out-of-class generalisation, not a fitted boundary.
NBEAM_BINS = [0, 1, 2, 3, 5, 9, 17, 32, 49, 65]
TRAINED_BINS = {"(0, 1]", "(1, 2]", "(32, 49]", "(49, 65]"}


def _auc(y, s):
    """ROC-AUC, or NaN where it is undefined -- rather than an exception."""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    ok = np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def evaluation_mask(df, *, exclude_mk=True):
    """
    The rows a reported AUC may be computed over: labelled AND trainable.

    Not a detail. Scoring a file that was excluded from training and then
    folding it into the headline measures distribution shift and calls it
    accuracy. Measured: including mk_sample_hits in the evaluation of a model
    that never trained on it moves the stamp AUC from 0.9884 to 0.9795, because
    the model scores that pre-filtered file's confined hits as RFI. That is a
    real and useful finding -- it is reported by validate() under "ood" -- but
    it is not a statement about how well the score works on the survey.
    """
    m = df["weak_label"].to_numpy() >= 0
    if exclude_mk and "file" in df.columns:
        m &= ~df["file"].isin(PREFILTERED_FILES).to_numpy()
    return m


def validate(df, *, n_splits=5, seed=0, exclude_mk=True,
             importance_sample=20_000, verbose=True):
    """
    Reproduce every number the write-up quotes, from the shipped code.

    This exists because the result is only as good as its de-confounding. A
    0.99 AUC for predicting beam multiplicity from morphology has at least five
    boring explanations, and each block below closes one:

      ablation        is morphology worth more than Track A's own flags?
      nonzero_drift   x01 is NaN exactly at zero drift and P(RFI|NaN)=0.996,
                      a 29.6% freebie -- does the result survive without it?
      snr_stratified  a signal must be BRIGHT to reach 32 beams, so the score
                      could be a brightness meter; within an SNR decile it
                      cannot be
      signal_level    one emitter yields up to 64 near-duplicate rows, which
                      could be inflating every row-wise metric
      cross_band      hold out a whole band: general morphology, or memorised
                      emitters?
      per_file        is one file carrying the result?
      ood             what happens on a differently-filtered hit population --
                      the case the BLUSE team hits the moment they apply this
                      to new data

    Returns a dict of plain floats, ready for json.dump.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score

    def say(*a):
        if verbose:
            print(*a, flush=True)

    ev = evaluation_mask(df, exclude_mk=exclude_mk)
    y = df["weak_label"].to_numpy()[ev].astype(int)
    rep = {"n_rows": int(len(df)), "n_evaluated": int(ev.sum()),
           "n_groups": int(df["group_id"].nunique()), "exclude_mk": exclude_mk,
           "base_rate": float(y.mean()), "n_splits": n_splits, "seed": seed}

    # --- ablation ---------------------------------------------------------
    say("  ablation (4 feature sets)...")
    oof, rep["ablation"] = {}, {}
    for name in ("flags", "meta", "stamp", "all"):
        s, _, info = fit_score(df, features=name, n_splits=n_splits, seed=seed,
                               exclude_mk=exclude_mk)
        oof[name] = s
        rep["ablation"][name] = {
            "n_features": len(info["columns"]),
            "roc_auc": _auc(y, s[ev]),
            "pr_auc": float(average_precision_score(y, s[ev])),
        }
        say(f"    {name:6s} n={len(info['columns']):2d} "
            f"AUC={rep['ablation'][name]['roc_auc']:.4f}")
    primary = oof["stamp"][ev]

    # --- the zero-drift freebie -------------------------------------------
    say("  without zero-drift rows...")
    nz = ~df["flag_zero_drift"].to_numpy()[ev]
    rep["nonzero_drift"] = {
        "n": int(nz.sum()), "base_rate": float(y[nz].mean()),
        "stamp": _auc(y[nz], primary[nz]),
        "all": _auc(y[nz], oof["all"][ev][nz]),
        "flags": _auc(y[nz], oof["flags"][ev][nz]),
    }
    say(f"    stamp={rep['nonzero_drift']['stamp']:.4f} "
        f"flags={rep['nonzero_drift']['flags']:.4f}")

    # --- brightness --------------------------------------------------------
    say("  SNR-stratified...")
    snr = df["snr"].to_numpy()[ev]
    dec = pd.qcut(snr, 10, labels=False, duplicates="drop")
    rows, num, den = [], 0.0, 0
    for d in sorted(pd.unique(dec)):
        m = dec == d
        a = _auc(y[m], primary[m])
        rows.append({"decile": int(d), "n": int(m.sum()),
                     "snr_lo": float(snr[m].min()), "snr_hi": float(snr[m].max()),
                     "base_rate": float(y[m].mean()), "roc_auc": a})
        if np.isfinite(a):
            num += a * m.sum(); den += int(m.sum())
    rep["snr_stratified"] = {"deciles": rows,
                             "weighted": float(num / den) if den else float("nan")}
    say(f"    weighted within-decile AUC = {rep['snr_stratified']['weighted']:.4f}")

    # --- duplication -------------------------------------------------------
    say("  signal-level (collapsing near-duplicate rows)...")
    sig = (df["group_id"].astype(str) + "|" + df["coarseChannel"].astype(str)
           + "|" + np.round(df["frequency"].to_numpy() * 1e6).astype("int64").astype(str)
           + "|" + np.round(df["driftRate"].to_numpy(), 4).astype(str))
    sd = pd.DataFrame({"sig": sig.to_numpy()[ev], "y": y, "s": primary})
    agg = sd.groupby("sig").agg(y=("y", "max"), s=("s", "median"), n=("y", "size"))
    one = (agg.n == 1).to_numpy()
    rep["signal_level"] = {
        "n_signals": int(len(agg)),
        "rows_per_signal": float(len(sd) / len(agg)),
        "median_per_signal": _auc(agg.y.to_numpy(), agg.s.to_numpy()),
        "n_singletons": int(one.sum()),
        "singleton": _auc(agg.y.to_numpy()[one], agg.s.to_numpy()[one]),
    }
    say(f"    per-signal={rep['signal_level']['median_per_signal']:.4f} "
        f"singleton={rep['signal_level']['singleton']:.4f}")

    # --- generalisation ----------------------------------------------------
    say("  cross-band (hold out a whole band)...")
    edf = df[ev].reset_index(drop=True)
    eband = edf["file"].map(BANDS).to_numpy()
    rep["cross_band"] = {}
    for b in ("L", "UHF", "S"):
        te = eband == b
        tr = (~te) & (eband != "MK")
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        got = {}
        for name in ("stamp", "all"):
            cols = feature_columns(name)
            m = HistGradientBoostingClassifier(
                max_iter=300, early_stopping=False, random_state=seed)
            m.fit(edf.loc[tr, cols].to_numpy(np.float32), y[tr])
            got[name] = _auc(y[te], m.predict_proba(
                edf.loc[te, cols].to_numpy(np.float32))[:, 1])
        rep["cross_band"][b] = {"n_test": int(te.sum()),
                                "base_rate": float(y[te].mean()), **got}
        say(f"    held out {b:3s} stamp={got['stamp']:.4f} all={got['all']:.4f}")

    # --- per file ----------------------------------------------------------
    rep["per_file"] = {}
    files = df["file"].to_numpy()[ev]
    for f in sorted(pd.unique(files)):
        m = files == f
        rep["per_file"][str(f)] = {
            "n": int(m.sum()), "base_rate": float(y[m].mean()),
            "roc_auc": _auc(y[m], primary[m])}

    # --- out of distribution ------------------------------------------------
    # mk_sample_hits was pre-filtered before delivery, and its labels do not
    # mean what they mean elsewhere. `n_beams` is counted WITHIN a file, and
    # this file carries ~25 hits per observation against ~2,931 in the others,
    # so its beam multiplicity tops out at 9 of up to 64 formed beams. Nothing
    # in it can reach the >=32 threshold: it contributes zero positives by
    # construction, and "<=2 beams" there is not established confinement so
    # much as a file too sparse to count.
    #
    # So the two readings of the number below cannot be separated from these
    # data -- either the model is wrong about this file, or the file's labels
    # are. The ceiling at 9 makes the labels the more suspect of the two, and
    # a frequency-tolerance match against the same obsids in lband_long
    # resolves nothing: only 0.6% of mk hits have an lband_long hit within 2 Hz,
    # so this is a different hit list, not a subsample of one we hold.
    #
    # The operational conclusion is the same either way, and it is the one that
    # matters when the BLUSE team points this at new data: a score fitted to
    # how THIS survey's hits were selected does not transfer, unchecked, to a
    # hit list selected differently.
    ood = (df["file"].isin(PREFILTERED_FILES) & (df["weak_label"] >= 0)).to_numpy()
    if exclude_mk and ood.any():
        s = oof["stamp"][ood]
        rep["ood"] = {
            "file": list(PREFILTERED_FILES), "n": int(ood.sum()),
            "n_positive": int((df["weak_label"].to_numpy()[ood] == 1).sum()),
            "mean_score": float(np.nanmean(s)),
            "frac_scored_rfi": float(np.nanmean(s > 0.5)),
            "auc_including_it": _auc(
                df["weak_label"].to_numpy()[ev | ood].astype(int),
                oof["stamp"][ev | ood]),
        }
        say(f"  out-of-distribution ({', '.join(PREFILTERED_FILES)}): "
            f"{rep['ood']['frac_scored_rfi']:.1%} of known-confined hits "
            f"scored >0.5")

    # --- the monotonicity table -------------------------------------------
    b = pd.cut(df["n_beams"], NBEAM_BINS, right=True)
    t = pd.DataFrame({"bin": b.astype(str), "s": oof["stamp"]})
    g = t.groupby("bin", observed=True)["s"].agg(["size", "mean", "median"])
    g = g.reindex([str(i) for i in b.cat.categories]).dropna()
    rep["n_beams_monotonicity"] = {
        "bins": [{"bin": i, "n": int(r["size"]), "mean_score": float(r["mean"]),
                  "median_score": float(r["median"]),
                  "in_training": i in TRAINED_BINS} for i, r in g.iterrows()],
        "monotone": bool(np.all(np.diff(g["mean"].to_numpy()) > 0)),
    }
    say(f"  monotone across all {len(g)} n_beams bins: "
        f"{rep['n_beams_monotonicity']['monotone']}")

    # --- what morphology actually says RFI ---------------------------------
    say("  permutation importance...")
    cols = feature_columns("stamp")
    ei = np.flatnonzero(ev)
    tr, te = next(GroupKFold(n_splits).split(
        ei, y, df["group_id"].to_numpy()[ev]))
    m = HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                       random_state=seed)
    m.fit(df.iloc[ei[tr]][cols].to_numpy(np.float32), y[tr])
    rng = np.random.default_rng(seed)
    pick = te if len(te) <= importance_sample else rng.choice(
        te, importance_sample, replace=False)
    r = permutation_importance(m, df.iloc[ei[pick]][cols].to_numpy(np.float32),
                               y[pick], n_repeats=5, random_state=seed,
                               scoring="roc_auc", n_jobs=-1)
    rep["importance"] = sorted(
        [{"column": c, "mean": float(a), "std": float(sd_)}
         for c, a, sd_ in zip(cols, r.importances_mean, r.importances_std)],
        key=lambda d: -d["mean"])
    say(f"    top: {', '.join(d['column'] for d in rep['importance'][:3])}")
    return rep
