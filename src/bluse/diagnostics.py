#!/usr/bin/env python3
"""
diagnostics.py -- per-column audit of a feature matrix.

Why this exists: the Bench's feature rail shows raw interquartile ranges, which
answer "how unequal are the columns before scaling". They cannot answer "what is
each column actually contributing to the distance HDBSCAN takes", and that is
where the defects are.

Measured on sband_short at Bench defaults, post-scaling distance shares run from
1.7% (f02_abs_drift) to 24.3% (x03_channel_offset) -- a 14x spread against an
equal share of 6.7%. `robust` scaling equalises the IQR, but HDBSCAN responds to
variance, and the IQR-to-variance ratio depends on distribution shape. So robust
misrepresents the distribution in BOTH directions: f02's IQR is inflated to
5.954 by a 26.6% tie sitting at the extreme (-5.199), and x03's is deflated to
0.042 by a tie near the centre.

Everything here is a pure function of (matrix, columns). No FastAPI, no
argparse, no workspace access -- both entry points import it.
"""

from __future__ import annotations

import numpy as np

CLIP = 5.0          # scale() clips robust z-scores here
TIE_FLAG = 0.10     # flag a continuous/ordinal column tying above this
CLIP_FLAG = 0.01    # flag a column landing on the clip more often than this

# Share thresholds, as multiples of an equal share (1/n_features). Calibrated
# in docs/share-knn-threshold-2026-09.md across all seven per-file parquets:
# at these values share_knn flags a mean of 4.9 columns of 15 per file, against
# 7.4 for the same rule on share_global -- the local statistic is already the
# more selective one. SHARE_DISAGREE flags a column whose two shares differ by
# that factor either way, i.e. one where the global number misleads; at 2.5x it
# marks a mean of 2.3 columns per file, at 2.0x it eventually marks 12 of 15
# (too loose) and at 3.0x it misses f13_redness.
SHARE_HIGH = 2.0
SHARE_LOW = 0.5
SHARE_DISAGREE = 2.5

# Backstop on any single equalising weight, applied even when `cap` is None.
# Mean-normalisation already bounds the maximum by roughly the column count, so
# on real data this never binds -- the measured closed-form weights on
# sband_short span 0.430 to 1.594. It exists because the sd > 1e-12 floor only
# catches exactly-constant columns: a column with sigma = 0.001 among fifteen
# would otherwise draw a weight near 8 unnoticed, and the failure presents as
# "clustering got worse after I added a feature".
EQUALISE_MAX_WEIGHT = 5.0
# The shipped strategy, chosen by measurement in
# docs/equalising-scaling-experiment-2026-09.md. "closed" is w ∝ 1/sigma --
# winsorised standardisation. The k-NN strategy scored far higher and was
# REJECTED: its target is unattainable for any column with a large tie
# fraction (tied pairs contribute zero to that coordinate and are
# disproportionately near neighbours), so it drives f02's weight to the
# ceiling forever, collapses the metric from 15 effective dimensions to 2.2,
# and is beaten by a hand-built two-column weighting. See that write-up before
# changing these.
EQUALISE_STRATEGY = "closed"
EQUALISE_ITERS = 0
EQUALISE_CAP = None
# Ratio of largest to smallest weight above which info["spread_warning"] fires.
EQUALISE_SPREAD_WARN = 10.0


def robust_stats(X):
    """Median and IQR per column, for fitting a robust scaler on one row set."""
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    return {"median": np.median(X, axis=0), "iqr": q75 - q25}


def scale(X, how, stats=None, *, kinds=None, columns=None):
    """
    Equalise how much each feature contributes to the Euclidean distance.

    The single shared implementation. bench/app.py and track_b_cluster.py both
    call this one, so the two paths cannot drift.

      "robust"    centre on the median, divide by the IQR, clip to +/-CLIP.
      "quantile"  rank-transform every feature to a uniform distribution.
      "robust-equalised"
                  robust, then per-column weights that equalise each column's
                  contribution to the distance. Robust equalises the IQR, but
                  HDBSCAN responds to VARIANCE and the ratio between them
                  depends on distribution shape, so contributions run 1.7% to
                  24.3% globally against an equal share of 6.7%. See
                  equalising_weights, and
                  docs/equalising-scaling-experiment-2026-09.md for why the
                  shipped strategy is the closed form rather than the k-NN one
                  that scored higher.
      "none"      GLOBULAR's literal spec -- their transforms are applied
                  upstream in features.normalise(), so "none" means "the
                  paper's preprocessing and nothing further".

    `kinds`/`columns` are used only by "robust-equalised", to keep boolean and
    flag columns at weight exactly 1.0.

    `stats` is an optional {"median", "iqr"} from robust_stats(), so the fit
    can come from a different row set than the transform. The Bench uses it to
    fit on the full population and apply to its 35k sample (D-4). Without it
    the fit is on whatever is passed, which is what the CLI wants.
    """
    X = np.array(X, copy=True)
    if how in ("robust", "robust-equalised"):
        if stats is None:
            stats = robust_stats(X)
        med = np.asarray(stats["median"])
        iqr = np.asarray(stats["iqr"])
        iqr = np.where(iqr > 1e-12, iqr, 1.0)
        base = np.clip((X - med) / iqr, -CLIP, CLIP)
        if how == "robust":
            return base
        # with_info=False deliberately: the achieved-deviation diagnostic calls
        # _shares_knn, which costs 1,311 ms on sband_short against 1.7 ms for
        # the weights, and scale() runs once per clustering run, once per Bench
        # rail render and once per stability seed.
        w, _ = equalising_weights(base, kinds=kinds, columns=columns,
                                  strategy=EQUALISE_STRATEGY,
                                  iters=EQUALISE_ITERS, cap=EQUALISE_CAP,
                                  with_info=False)
        return base * w
    if how == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, len(X)),
                                 subsample=200_000, random_state=0)
        return qt.fit_transform(X)
    if how == "none":
        return X
    # Previously any unrecognised value fell through to `return X`, so a typo'd
    # mode silently meant "none" -- the quietest possible way to run the wrong
    # experiment.
    raise ValueError(f"unknown scaling {how!r}; expected one of "
                     f"robust, robust-equalised, quantile, none")


def _shares(Z, rng, n_pairs=4000):
    """Mean per-column share of the squared Euclidean distance, random pairs."""
    n = len(Z)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    per = ((Z[a] - Z[b]) ** 2).mean(axis=0)
    tot = per.sum()
    return per / tot if tot > 0 else np.full(len(per), np.nan)


def _shares_knn(Z, k, rng, sample=5000):
    """
    The same, restricted to each point's k nearest neighbours.

    Not decoration. HDBSCAN responds to core distances and mutual reachability,
    both LOCAL, and the global random-pair share is a proxy for a local
    quantity. The approximation is worst exactly where ties are: a tied column
    contributes zero to every tie-tie pair, and tie-tie pairs are
    disproportionately likely to be mutual near neighbours because they already
    agree in that coordinate. The gap between share_global and share_knn is
    therefore itself the tie diagnostic, which is why both are reported.

    THE INDEX IS BUILT ON EVERY ROW; only the QUERY points are sampled.
    Sampling the index too -- which this did originally -- thins the data, so
    each point's k nearest neighbours sit further away and the statistic drifts
    toward the global share. That is a systematic bias, not sampling noise, and
    it is much larger: measured on sband_short against the exact full-data
    value, a 5,000-row subsampled index is off by 0.66 points on average and
    2.18 at worst (f01_frequency reading 7.35% against a true 5.28%), while
    5,000 queries against the full index are off by 0.044 and 0.095. Since the
    flag threshold now keys on this number, that mattered.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(Z)
    k = int(max(2, min(k, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    q = (rng.choice(n, sample, replace=False) if n > sample
         else np.arange(n))
    _, ind = nn.kneighbors(Z[q])
    a = np.repeat(q, k)
    b = ind[:, 1:].ravel()
    per = ((Z[a] - Z[b]) ** 2).mean(axis=0)
    tot = per.sum()
    return per / tot if tot > 0 else np.full(len(per), np.nan)


def _normalise_weights(w, frozen, cap):
    """
    Mean-1 over the free columns, capped, with the frozen ones pinned at 1.0.

    NOTE the cap is applied AFTER mean-normalisation and the result is not
    re-normalised, so a capped weight vector has mean != 1. That is harmless
    for the percentile matching cut, which is scale-invariant, but
    derive_cut_quantile and any explicit `cut=` are absolute distances and
    would shift. Re-normalising would push weights back over the cap, so this
    is a deliberate choice rather than an oversight.
    """
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    w = np.asarray(w, dtype=np.float64).copy()
    w[frozen] = 1.0
    free = ~frozen
    if free.any() and w[free].mean() > 0:
        w[free] = w[free] / w[free].mean()
    # The backstop is ONE-SIDED, and deliberately. The documented hazard is a
    # low-variance column drawing a LARGE weight -- the zero-drift indicator
    # that would have drawn 10.256. A small weight merely suppresses a column,
    # which is benign, and flooring it would block legitimate equalisation: a
    # column with 5x the spread of its neighbours needs a weight near 1/5.
    # An explicit `cap` stays two-sided, because that is what the P0-2
    # weight-cap experiment measured.
    hi = EQUALISE_MAX_WEIGHT
    if cap:
        w[free] = np.clip(w[free], 1.0 / cap, cap)
        hi = min(hi, cap)
    w[free] = np.minimum(w[free], hi)
    return w


def equalising_weights(Z, *, kinds=None, columns=None, strategy="closed",
                       iters=0, damping=0.5, cap=None, min_samples=8,
                       knn_sample=20_000, seed=0, with_info=True):
    """
    Per-column weights that equalise each column's contribution to the distance.

    Robust scaling equalises the INTERQUARTILE RANGE, but HDBSCAN responds to
    VARIANCE, and the ratio between them depends on distribution shape -- so
    contributions run 1.7% to 24.3% globally against an equal share of 6.7%.
    This targets the contribution itself.

      strategy="closed"  w proportional to 1/sigma. This is the EXACT solution
                         for the GLOBAL share, since share_global_j is
                         proportional to w_j^2 var_j. One pass, no seed, 1.7 ms
                         on sband_short.

                         Read it for what it is: robust-scale, clip, then
                         divide by sigma is "winsorise at +/-5 IQR units, then
                         z-score". Dividing by the IQR and then by the standard
                         deviation of the result is dividing by the standard
                         deviation of the original, so the robust step
                         contributes nothing except WHERE THE CLIP LANDS.

      strategy="knn"     damped fixed point on the k-NN share, which is the
                         statistic HDBSCAN actually responds to (P0-3) but
                         which does NOT converge: the neighbour graph is itself
                         a function of w, so the map is not a contraction.
                         Measured undamped, the worst column is still 0.33 off
                         an equal share after eight iterations while the
                         weights diverge from 0.500-3.047 to 0.044-10.347.
                         Usable only with `iters` and `cap` fixed by
                         measurement; check info["dev_trace"] before trusting
                         it.

      strategy="hybrid"  closed form, then `iters` damped k-NN steps.

    boolean and flag columns keep weight exactly 1.0. Their low variance draws
    a huge weight -- a zero-drift indicator measured 0.33% k-NN share and would
    have drawn 10.256, twice the constant measured to destroy eom.

    `with_info=False` skips the achieved-deviation diagnostic, which calls
    _shares_knn and costs 1,311 ms on sband_short at knn_sample=20,000 against
    1.7 ms for the weights. scale() uses it: that call happens once per
    clustering run, once per Bench rail render and once per stability seed.

    Returns (weights, info). See
    docs/superpowers/specs/2026-09-02-equalising-scaling-design.md.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n_cols = Z.shape[1]
    kinds = kinds or {}
    cols = list(columns) if columns is not None else [None] * n_cols
    declared = np.array([kinds.get(c) in ("boolean", "flag") for c in cols])
    # A column with no spread contributes nothing to any distance, so its
    # weight is undefined rather than large. Freeze it at 1.0 alongside the
    # declared booleans instead of letting mean-normalisation drift it.
    degenerate = Z.std(axis=0) <= 1e-12
    frozen = declared | degenerate

    def _knn_dev(w):
        s = _shares_knn(Z * w, min_samples, np.random.default_rng(seed),
                        knn_sample)
        s = np.where(np.isfinite(s) & (s > 1e-9), s, 1.0 / n_cols)
        return s

    w = np.ones(n_cols)
    trace = []
    if strategy in ("closed", "hybrid"):
        sd = (Z * w).std(axis=0)
        w = w / np.where(sd > 1e-12, sd, 1.0)
        w = _normalise_weights(w, frozen, cap)
    if strategy in ("knn", "hybrid"):
        for _ in range(int(iters)):
            s = _knn_dev(w)
            trace.append(float(np.abs(s * n_cols - 1).max()))
            w = w * ((1.0 / n_cols) / s) ** damping
            w = _normalise_weights(w, frozen, cap)
    w = _normalise_weights(w, frozen, cap)

    lo = float(w.min())
    info = {
        "strategy": strategy,
        "iters": int(iters),
        "skipped": [c for c, f in zip(cols, declared) if f and c is not None],
        "degenerate": [i for i, f in enumerate(degenerate) if f],
        "dev_trace": trace,
        "weight_min": lo,
        "weight_max": float(w.max()),
        "spread_warning": bool(lo > 0
                               and w.max() / lo > EQUALISE_SPREAD_WARN),
    }
    if with_info:
        rng = np.random.default_rng(seed)
        sg = _shares(Z * w, rng)
        sk = _shares_knn(Z * w, min_samples, rng, knn_sample)
        info["max_dev_global"] = float(np.abs(sg * n_cols - 1).max())
        info["max_dev_knn"] = float(np.abs(sk * n_cols - 1).max())
    return w, info


def audit(raw, columns, *, scaling="robust", kinds=None, min_samples=8,
          knn_sample=20_000, seed=0):
    """
    Audit every column of `raw` (n, n_features), pre-scaling.

    Returns one dict per column. `kinds` maps column name -> value kind; any
    column absent is treated as "continuous".
    """
    raw = np.asarray(raw, dtype=np.float64)
    kinds = kinds or {}
    rng = np.random.default_rng(seed)

    # clip_frac must come from the ROBUST BASE. Under "robust-equalised" the
    # returned matrix is base * w, so a value sitting exactly on the +/-5 clip
    # is no longer there after weighting and the threshold test misses it --
    # measured, a weight of 0.661 moves a clipped row to 3.31 and clip_frac
    # reads 0.000 for a mode that clips 5% of the column. Building the base
    # here also avoids scaling twice.
    base = None
    weights = np.ones(raw.shape[1])
    if scaling in ("robust", "robust-equalised"):
        base = scale(raw, "robust")
        if scaling == "robust":
            Z = base
        else:
            weights, _ = equalising_weights(base, kinds=kinds,
                                            columns=columns,
                                            strategy=EQUALISE_STRATEGY,
                                            iters=EQUALISE_ITERS,
                                            cap=EQUALISE_CAP, with_info=False)
            Z = base * weights
    else:
        Z = scale(raw, scaling)
    q75r, q25r = np.percentile(raw, [75, 25], axis=0)
    iqr_raw = q75r - q25r
    q75s, q25s = np.percentile(Z, [75, 25], axis=0)
    iqr_scaled = q75s - q25s

    sg = _shares(Z, rng)
    sk = _shares_knn(Z, min_samples, rng, knn_sample)
    equal = 1.0 / max(len(columns), 1)

    out = []
    for i, col in enumerate(columns):
        x = raw[:, i]
        vals, counts = np.unique(x, return_counts=True)
        j = int(counts.argmax())
        tie = float(counts[j] / len(x))
        kind = kinds.get(col, "continuous")
        # Only "robust" clips. Under "none"/"quantile" this test would flag
        # any raw value with magnitude >= 5, which is not a clip at all --
        # `bluse-cluster --report --scaling none` reported false clips.
        clip_frac = (float((np.abs(base[:, i]) >= CLIP - 1e-12).mean())
                     if base is not None else 0.0)

        flags = []
        if kind in ("continuous", "ordinal") and tie > TIE_FLAG:
            flags.append("tie")
        if clip_frac > CLIP_FLAG:
            flags.append("clip")
        # share_knn CARRIES THE THRESHOLD. HDBSCAN responds to core distances
        # and mutual reachability, both local, so the local statistic is the
        # one describing what the clusterer actually sees; the global
        # random-pair share is a proxy for it. These thresholds were withheld
        # until calibrated rather than guessed -- see the P0-3 write-up for the
        # cross-file measurement behind the constants.
        #
        # share_global is now secondary, and earns its place only where it
        # DISAGREES: a large ratio says the global number misleads for this
        # column. The two cases, both measured on real files:
        #   x03_channel_offset  24.3% global / 5.9% local on sband_short --
        #     the column the original review indicted, on the strength of the
        #     statistic that overstates it.
        #   f09_temporal_skew   6.7% global / 15.3% local -- the largest local
        #     contributor, which the global rule flagged not at all.
        #
        # The two share flags are ONE observation, not two: shares sum to 1, so
        # a column at 24% mechanically depresses every other toward the lower
        # bound. Useful for pointing at a column; not independent evidence.
        if np.isfinite(sk[i]):
            if sk[i] > SHARE_HIGH * equal:
                flags.append("share-high")
            elif sk[i] < SHARE_LOW * equal:
                flags.append("share-low")
        if np.isfinite(sg[i]) and np.isfinite(sk[i]) and sk[i] > 0:
            ratio = sg[i] / sk[i]
            if ratio >= SHARE_DISAGREE or ratio <= 1 / SHARE_DISAGREE:
                flags.append("share-disagree")

        out.append({
            "col": col,
            "label": col[:-2] if col.endswith("_n") else col,
            "kind": kind,
            "n_distinct": int(len(vals)),
            "max_tie_fraction": tie,
            "tie_value": float(vals[j]),
            "iqr_raw": float(iqr_raw[i]),
            "iqr_scaled": float(iqr_scaled[i]),
            "clip_frac": clip_frac,
            "weight": float(weights[i]),
            "share_global": float(sg[i]),
            "share_knn": float(sk[i]),
            "equal_share": float(equal),
            "flags": flags,
        })
    return out


def group_index(labels):
    """
    Sort-based grouping for a label vector. Returns (ids, rows, starts).

    `rows` are the original row indices of the clustered points, ordered so
    that group `ids[j]` occupies `rows[starts[j]:starts[j+1]]`. Feed `starts`
    to np.add.reduceat / np.minimum.reduceat / np.maximum.reduceat.

    Use this instead of a mask per cluster. `X[labels == c]` inside a loop over
    k clusters is O(k*n) and it was the pattern in four places here. Measured on
    1,281,878 rows x 15 columns:

        k =  1,491   mask-per-cluster ~3.3 s     sort + reduceat 0.22 s
        k = 20,000   mask-per-cluster ~47 s      sort + reduceat 0.23 s

    The sort is O(n log n) once and flat in k, which matters because `leaf` on
    all_features.parquet produces of order 80,000 clusters -- where the mask
    pattern is minutes in centroids() alone, and worse in quality(), which
    groups seven times per call (two thresholds plus five permutations).
    """
    labels = np.asarray(labels)
    rows = np.nonzero(labels >= 0)[0]
    if not len(rows):
        return np.array([], dtype=labels.dtype), rows, np.array([], dtype=int)
    lab = labels[rows]
    order = np.argsort(lab, kind="stable")
    rows = rows[order]
    lab = lab[order]
    ids = np.unique(lab)
    starts = np.searchsorted(lab, ids)
    return ids, rows, starts
