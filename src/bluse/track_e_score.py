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

The matrix is float64 and the score is averaged over three seeds. Neither is
decoration: HistGradientBoosting draws its 256 bin edges from a random
200,000-row subsample, so both the dtype and the seed move where a value falls
relative to a split, and a single seed's per-hit verdict is substantially
churn. See #9 of that document -- it also records the control that makes the
mechanism unambiguous.

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
bright to be detected in 32 beams -- within-SNR-decile AUC is 0.8934 against
0.9899 overall, so brightness contributes but does not carry the result.

DEFAULT FEATURE SET
-------------------
The 12 stamp-morphology columns, not all 16. It costs 0.0012 AUC (0.9899 against
0.9911 for all 16 -- the 16-feature model is the better one on this survey) and
buys the argument the headline rests on: it cannot relearn the RFI frequency
mask, so "morphology beats the classical filter's own flags" is not partly
circular with that filter's first cut. It also matters in use, since the score
is applied to Track A survivors, which sit outside the mask by construction.

An earlier draft also claimed stamp transfers better to an unseen band. At
these settings that is false -- held out, stamp against all-16 is L 0.9911 /
0.9872 but UHF 0.8976 / 0.9127 and S 0.8962 / 0.9157. One band of three.
`--features all` ships alongside and both are reported.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import features as F

# Every feature column in the parquet is float64, and every other science path
# in this package -- features.py, track_a_filter.py, track_b_cluster.py,
# matching.py, diagnostics.py, metrics.py -- asks for float64 explicitly. The
# feature matrix here stays float64 for the same reason, so that no result
# depends on where a value happens to land relative to a bin edge.
#
# HistGradientBoosting bins to 256 levels and upcasts to float64 internally, so
# the measured cost of narrowing is small -- but "small" was an assumption
# until it was measured, and the measurement is in docs/track-e-2026-09.md #9.
# One constant, so a future narrowing is a one-line change with a reason
# attached rather than five scattered casts.
FEATURE_DTYPE = np.float64

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


def fit_score(df, *, features="stamp", n_splits=5, seed=0, n_seeds=3,
              exclude_mk=True, max_iter=300):
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

    AVERAGED OVER `n_seeds` SEEDS, and that is not a refinement -- a single
    seed's per-hit verdict is substantially churn. HistGradientBoosting picks
    its 256 bin edges from a 200,000-row random subsample, so the seed moves
    where values fall relative to a split, and 300 boosting rounds amplify it.
    Measured across seeds 0-2 (docs/track-e-2026-09.md #9): ROC-AUC is stable
    to +/-0.00004, but the `contrarian` count swings 2,972-3,368 and the
    shortlist shares only 93% of its membership between two seeds.

    Averaging three seeds raises AUC from 0.9892 to 0.9899, fixes the counts,
    and halves the contrarian set -- because roughly half of it at any one seed
    was marginal hits pushed below the threshold by that seed's bin edges. It
    costs 3x 27 seconds. `n_seeds=1` reproduces the single-seed behaviour.

    Averaging does not weaken the no-leak property: every seed uses the same
    GroupKFold split, so a row's score is the mean of models none of which saw
    its observation.

    Returns (score, fold, info). `fold` is -1 for any row in no training fold.
    """
    cols = feature_columns(features)
    missing = [c for c in cols + ["weak_label", "group_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    X = df[cols].to_numpy(FEATURE_DTYPE)
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

    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    seeds = [seed + i for i in range(n_seeds)]

    score = np.zeros(len(df))
    fold = np.full(len(df), -1, dtype=np.int8)
    rest = np.flatnonzero(~train)
    acc = np.zeros(len(rest))

    splits = list(GroupKFold(n_splits).split(tr_idx, y[tr_idx], groups[tr_idx]))
    folds = [{"fold": k, "n_train": int(len(tr)), "n_test": int(len(te))}
             for k, (tr, te) in enumerate(splits)]
    for k, (_, te) in enumerate(splits):
        fold[tr_idx[te]] = k

    for sd in seeds:
        for tr, te in splits:
            model = HistGradientBoostingClassifier(
                max_iter=max_iter, early_stopping=False, random_state=sd)
            model.fit(X[tr_idx[tr]], y[tr_idx[tr]])
            score[tr_idx[te]] += model.predict_proba(X[tr_idx[te]])[:, 1]
            if len(rest):
                acc += model.predict_proba(X[rest])[:, 1]
    score[tr_idx] /= n_seeds
    if len(rest):
        score[rest] = acc / (n_seeds * n_splits)

    info = {
        "features": features,
        "columns": cols,
        "dropped_columns": dropped,
        "n_splits": n_splits,
        "seed": seed,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "dtype": np.dtype(FEATURE_DTYPE).name,
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
    that never trained on it moves the stamp AUC from 0.9899 to 0.9821, because
    the model scores that pre-filtered file's confined hits as RFI. That is a
    real and useful finding -- it is reported by validate() under "ood" -- but
    it is not a statement about how well the score works on the survey.
    """
    m = df["weak_label"].to_numpy() >= 0
    if exclude_mk and "file" in df.columns:
        m &= ~df["file"].isin(PREFILTERED_FILES).to_numpy()
    return m


def validate(df, *, n_splits=5, seed=0, n_seeds=3, exclude_mk=True,
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
           "base_rate": float(y.mean()), "n_splits": n_splits, "seed": seed,
           "n_seeds": n_seeds, "dtype": np.dtype(FEATURE_DTYPE).name}

    # --- ablation ---------------------------------------------------------
    say("  ablation (4 feature sets)...")
    oof, rep["ablation"] = {}, {}
    for name in ("flags", "meta", "stamp", "all"):
        s, _, info = fit_score(df, features=name, n_splits=n_splits, seed=seed,
                               n_seeds=n_seeds, exclude_mk=exclude_mk)
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
            Xtr = edf.loc[tr, cols].to_numpy(FEATURE_DTYPE)
            Xte = edf.loc[te, cols].to_numpy(FEATURE_DTYPE)
            # Averaged over the same seeds as the headline, so the transfer
            # number and the in-band number are measured the same way.
            p = np.zeros(int(te.sum()))
            for sd in range(seed, seed + n_seeds):
                m = HistGradientBoostingClassifier(
                    max_iter=300, early_stopping=False, random_state=sd)
                m.fit(Xtr, y[tr])
                p += m.predict_proba(Xte)[:, 1]
            got[name] = _auc(y[te], p / n_seeds)
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
    # so its beam multiplicity tops out at 9 of up to 64 formed beams. It
    # contributes zero positives by construction.
    #
    # The number below -- 94.2% of its nominally-confined hits scored as RFI --
    # is the model being RIGHT about labels that are wrong. Measured directly:
    # 8,116 hits appear in both mk_sample_hits and lband_long under the same
    # id, same frequency, same beam, and the same hit is counted in a mean of
    # 1.87 beams there against 29.71 here. Of the 6,141 that mk calls confined,
    # lband_long puts 1,813 (29.5%) in >=32 beams.
    #
    # (Testing this by matching FREQUENCIES inside all_features.parquet finds
    # nothing, because that table deduplicates on id and the overlap is
    # structurally invisible in it. The per-file Track A catalogues keep it.)
    #
    # Kept in the report because it is the mechanism behind the caveat that
    # matters operationally: a hit list selected differently from the one this
    # score was fitted on will not behave the same way -- and neither will the
    # spatial filter it learned from.
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
    # Single seed: this is a ranking, not a headline number, and permuting
    # 12 columns x 5 repeats x n_seeds would triple the run for no change
    # in the order. Recorded in the report as importance_n_seeds.
    say("  permutation importance (single seed)...")
    cols = feature_columns("stamp")
    ei = np.flatnonzero(ev)
    tr, te = next(GroupKFold(n_splits).split(
        ei, y, df["group_id"].to_numpy()[ev]))
    m = HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                       random_state=seed)
    m.fit(df.iloc[ei[tr]][cols].to_numpy(FEATURE_DTYPE), y[tr])
    rng = np.random.default_rng(seed)
    pick = te if len(te) <= importance_sample else rng.choice(
        te, importance_sample, replace=False)
    r = permutation_importance(m, df.iloc[ei[pick]][cols].to_numpy(FEATURE_DTYPE),
                               y[pick], n_repeats=5, random_state=seed,
                               scoring="roc_auc", n_jobs=-1)
    rep["importance_n_seeds"] = 1
    rep["importance"] = sorted(
        [{"column": c, "mean": float(a), "std": float(sd_)}
         for c, a, sd_ in zip(cols, r.importances_mean, r.importances_std)],
        key=lambda d: -d["mean"])
    say(f"    top: {', '.join(d['column'] for d in rep['importance'][:3])}")
    return rep


# ---------------------------------------------------------------------------
# catalogues
# ---------------------------------------------------------------------------

# Where a survivor lands. Deliberately not a single cut: the middle band is
# large and the honest thing is to say so rather than force every hit to a side.
SHORTLIST_BELOW = 0.10     # "does not look like this survey's multi-beam RFI"
PRUNED_ABOVE = 0.90        # "morphologically indistinguishable from known RFI"

# Enough to find the hit again, and enough to plot it. `row` and `file` are what
# `bluse-explore stamps --rows` needs, so a candidate list is directly
# eyeballable without a join.
PROVENANCE = ["file", "row", "id", "obsid", "sourceName", "beam", "frequency",
              "driftRate", "snr", "n_beams", "beam_frac", "weak_label"]

_SHORTLIST_WARNING = (
    "A LOW SCORE IS NOT EVIDENCE OF A TECHNOSIGNATURE. The labels are "
    "positive-unlabelled: weak_label==0 means seen in <=2 beams, not verified "
    "clean, and most single-beam hits are still RFI. A low score says only "
    "that the hit does not resemble this survey's multi-beam RFI."
)


def catalogues(df, score, *, shortlist_below=SHORTLIST_BELOW,
               pruned_above=PRUNED_ABOVE):
    """
    Turn a column of scores into the three tables that are actually actionable.

    `candidates`  every Track A survivor, ranked ascending, with a verdict.
                  The deliverable: Track A leaves ~4,565 hits to vet by eye and
                  the score says which of them look exactly like RFI.
    `contrarian`  seen in >=32 beams AND scored clean. The spatial filter and
                  morphology disagree in the direction that cannot be explained
                  by the filter being conservative, so these are either
                  instrumental or the model's blind spot. Small, and the set
                  that teaches us the most.
    `ambiguous`   3-31 beams, where the spatial filter abstains and this score
                  is the only verdict available. The reason Track E exists.
    """
    keep = [c for c in PROVENANCE if c in df.columns]
    out = df[keep].copy()
    out["rfi_score"] = score

    verdict = np.where(score < shortlist_below, "shortlist",
                       np.where(score > pruned_above, "pruned", "uncertain"))
    out["verdict"] = verdict

    surv = (out[df["pass_all"].to_numpy()].sort_values("rfi_score")
            if "pass_all" in df.columns else out.iloc[:0])
    nb = df["n_beams"].to_numpy()
    contrarian = out[(nb >= 32) & (score < shortlist_below)].sort_values("rfi_score")
    ambiguous = out[(nb > 2) & (nb < 32)].sort_values("rfi_score")
    return {"candidates": surv, "contrarian": contrarian, "ambiguous": ambiguous}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively replace non-finite floats with None, for strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # bool BEFORE int: bool is a subclass of int, so the int branch would
    # catch True and write 1. That survives json.loads and stays truthy, so it
    # looks harmless -- until a consumer does np.array(flags) and gets an int
    # array, where `arr[mask]` silently becomes fancy indexing and `~mask`
    # becomes bitwise NOT. That is exactly how the first draft of the
    # monotonicity figure drew the wrong five points.
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def main():
    import argparse
    import json
    import os
    import time

    from . import paths

    p = argparse.ArgumentParser(
        prog="bluse-score",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_workspace_arg(p)
    p.add_argument(
        "--features", default="stamp", choices=sorted(FEATURE_SETS),
        help="which columns the model may see. 'stamp' (default) is the 12 "
             "morphology features computed from the stamp pixels. It is NOT "
             "the most accurate: 'all' scores 0.9911 against its 0.9899. It is "
             "the default because it cannot relearn the RFI frequency mask, so "
             "'morphology beats the classical filter' is not partly circular "
             "with that filter's own first cut -- and because the score is "
             "applied to Track A survivors, which sit outside that mask by "
             "construction. 'all' adds frequency, |drift|, SNR and channel "
             "offset. 'meta' is those four alone and 'flags' is Track A's six "
             "hand-built flags; both exist as baselines, not as things to ship")
    p.add_argument(
        "--folds", type=int, default=5, metavar="K",
        help="group-CV folds, split on obsid (default 5). Cross-validation is "
             "how the score is DELIVERED, not just audited: a labelled hit "
             "carries its out-of-fold score and an unlabelled one the mean of "
             "the K fold models, so no hit is ever scored by a model that saw "
             "its observation")
    p.add_argument("--seed", type=int, default=0,
                   help="first RNG seed (default 0); --seeds N uses "
                        "seed..seed+N-1")
    p.add_argument(
        "--seeds", type=int, default=3, metavar="N",
        help="average the score over N seeds (default 3). Not a refinement: "
             "HistGradientBoosting picks its 256 bin edges from a random "
             "200,000-row subsample, so one seed's per-hit verdict is "
             "substantially churn -- across seeds 0-2 the ROC-AUC is stable to "
             "+/-0.00004 but the contrarian count swings 2,972-3,368. "
             "Averaging three costs 3x27 s, raises AUC to 0.9899 and halves "
             "the contrarian set. --seeds 1 reproduces single-seed behaviour")
    p.add_argument(
        "--max-iter", type=int, default=300, metavar="N",
        help="boosting iterations per fold (default 300). Below ~100 the "
             "score degrades; above ~300 it does not improve")
    p.add_argument(
        "--include-mk", action="store_true",
        help="train on mk_sample_hits too. Off by default: that file carries "
             "~25 hits per observation against ~2,931 elsewhere, so its beam "
             "multiplicity tops out at 9 of up to 64 formed beams and it can "
             "contribute no RFI examples at all -- only a biased block of "
             "negatives. It is scored either way")
    p.add_argument(
        "--no-report", action="store_true",
        help="skip validate() and just write the scores. The report is seven "
             "de-confounding measurements and about 3 minutes of the ~4 the "
             "whole run takes; skip it when iterating, not when publishing")
    p.add_argument(
        "--shortlist-below", type=float, default=SHORTLIST_BELOW, metavar="P",
        help=f"verdict 'shortlist' below this score (default "
             f"{SHORTLIST_BELOW}). NOT a detection threshold -- see the "
             f"warning this command prints")
    p.add_argument(
        "--pruned-above", type=float, default=PRUNED_ABOVE, metavar="P",
        help=f"verdict 'pruned' above this score (default {PRUNED_ABOVE}); "
             f"these are Track A survivors that look exactly like known RFI")
    p.add_argument(
        "--no-plots", action="store_true",
        help="skip the five figures. They are drawn from the report, so "
             "--no-report implies this; `bluse-score-plots` redraws them from "
             "an existing report without refitting")
    p.add_argument("--tag", default="", metavar="STR",
                   help="prefix for the output filenames, to keep two runs "
                        "side by side (default none)")
    args = p.parse_args()
    paths.set_workspace(args.workspace)
    print(paths.banner())

    t0 = time.time()
    src = os.path.join(paths.features_dir(), "all_features.parquet")
    if not os.path.exists(src):
        raise SystemExit(paths.missing_workspace_message("all_features.parquet",
                                                         src))
    need = sorted(set(FEATURE_SETS["all"] + FLAG_COLUMNS + PROVENANCE + [
        "weak_label", "group_id", "coarseChannel", "pass_all"]))
    df = pd.read_parquet(src, columns=need)
    print(f"{len(df):,} hits, {df.group_id.nunique()} observations, "
          f"{df.file.nunique()} files  [{time.time() - t0:.0f}s]")

    ev = evaluation_mask(df, exclude_mk=not args.include_mk)
    print(f"  {int(ev.sum()):,} labelled and trainable "
          f"({df.weak_label.eq(1).sum():,} RFI : "
          f"{df.weak_label.eq(0).sum():,} confined), "
          f"{int((df.weak_label < 0).sum()):,} ambiguous")

    print(f"\nfitting {args.folds} folds x {args.seeds} seed(s) on "
          f"'{args.features}' ({len(feature_columns(args.features))} features, "
          f"{np.dtype(FEATURE_DTYPE).name})...")
    score, fold, info = fit_score(
        df, features=args.features, n_splits=args.folds, seed=args.seed,
        n_seeds=args.seeds, exclude_mk=not args.include_mk,
        max_iter=args.max_iter)
    print(f"  scored {len(df):,} hits in {time.time() - t0:.0f}s")
    print(f"  out-of-fold: {info['n_scored_out_of_fold']:,}   "
          f"fold-mean: {info['n_scored_by_fold_mean']:,}")

    out = paths.scores_dir()
    tag = f"{args.tag}_" if args.tag else ""
    sp = os.path.join(out, f"{tag}all_scores.parquet")
    pd.DataFrame({"id": df["id"].to_numpy(), "file": df["file"].to_numpy(),
                  "row": df["row"].to_numpy(), "rfi_score": score,
                  "fold": fold, "model": args.features}).to_parquet(sp)
    print(f"\nwrote {sp}")

    cats = catalogues(df, score, shortlist_below=args.shortlist_below,
                      pruned_above=args.pruned_above)
    for name, tab in cats.items():
        path = os.path.join(out, f"{tag}{name}.csv")
        tab.to_csv(path, index=False)
        print(f"wrote {path}  ({len(tab):,} rows)")

    c = cats["candidates"]
    if len(c):
        v = c.verdict.value_counts()
        print(f"\nTRACK A SURVIVORS: {len(c):,}")
        print(f"  pruned    {v.get('pruned', 0):>6,} "
              f"({v.get('pruned', 0) / len(c):>5.1%})  score > {args.pruned_above}"
              f"  -- look exactly like known multi-beam RFI")
        print(f"  uncertain {v.get('uncertain', 0):>6,} "
              f"({v.get('uncertain', 0) / len(c):>5.1%})")
        print(f"  shortlist {v.get('shortlist', 0):>6,} "
              f"({v.get('shortlist', 0) / len(c):>5.1%})  score < "
              f"{args.shortlist_below}")
        print(f"\n  {_SHORTLIST_WARNING}")
        print(f"\n  10 lowest-scoring survivors:")
        cols = [c_ for c_ in ["file", "sourceName", "frequency", "snr",
                              "n_beams", "rfi_score"] if c_ in c.columns]
        print("    " + c.head(10)[cols].to_string(index=False).replace("\n", "\n    "))

    rep = {"info": info, "shortlist_below": args.shortlist_below,
           "pruned_above": args.pruned_above,
           "counts": {k: int(len(v)) for k, v in cats.items()},
           "verdicts": {k: int(n) for k, n in
                        cats["candidates"].verdict.value_counts().items()},
           "survey": {"n_hits": int(len(df)),
                      "n_survivors": int(len(cats["candidates"]))}}
    if not args.no_report:
        print("\nvalidating...")
        rep["validation"] = validate(df, n_splits=args.folds, seed=args.seed,
                                     n_seeds=args.seeds,
                                     exclude_mk=not args.include_mk)
    rp = os.path.join(out, f"{tag}report.json")
    with open(rp, "w") as fh:
        json.dump(_json_safe(rep), fh, indent=2, allow_nan=False)
    print(f"\nwrote {rp}")

    if not (args.no_plots or args.no_report):
        from . import track_e_plots
        print()
        for path in track_e_plots.all_figures(rep, paths.plots_dir()):
            print(f"wrote {path}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
