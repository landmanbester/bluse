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


def scale(X, how):
    """
    Equalise how much each feature contributes to the Euclidean distance.

    The single shared implementation. bench/app.py and track_b_cluster.py both
    call this one, so the two paths cannot drift.

      "robust"    centre on the median, divide by the IQR, clip to +/-CLIP.
      "quantile"  rank-transform every feature to a uniform distribution.
      "none"      GLOBULAR's literal spec -- their transforms are applied
                  upstream in features.normalise(), so "none" means "the
                  paper's preprocessing and nothing further".
    """
    X = np.array(X, copy=True)
    if how == "robust":
        med = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
        return np.clip((X - med) / iqr, -CLIP, CLIP)
    if how == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, len(X)),
                                 subsample=200_000, random_state=0)
        return qt.fit_transform(X)
    return X


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
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(Z)
    Zs = Z[rng.choice(n, sample, replace=False)] if n > sample else Z
    k = int(max(2, min(k, len(Zs) - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zs)
    _, ind = nn.kneighbors(Zs)
    a = np.repeat(np.arange(len(Zs)), k)
    b = ind[:, 1:].ravel()
    per = ((Zs[a] - Zs[b]) ** 2).mean(axis=0)
    tot = per.sum()
    return per / tot if tot > 0 else np.full(len(per), np.nan)


def audit(raw, columns, *, scaling="robust", kinds=None, min_samples=8,
          knn_sample=5000, seed=0):
    """
    Audit every column of `raw` (n, n_features), pre-scaling.

    Returns one dict per column. `kinds` maps column name -> value kind; any
    column absent is treated as "continuous".
    """
    raw = np.asarray(raw, dtype=np.float64)
    kinds = kinds or {}
    rng = np.random.default_rng(seed)

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
        clip_frac = float((np.abs(Z[:, i]) >= CLIP - 1e-12).mean())

        flags = []
        if kind in ("continuous", "ordinal") and tie > TIE_FLAG:
            flags.append("tie")
        if clip_frac > CLIP_FLAG:
            flags.append("clip")
        # share_global carries the threshold because it was the statistic
        # actually measured when these rules were written. share_knn is
        # reported but NOT thresholded: no value for it existed then, and
        # inventing a bound would have been the unverified-claim pattern this
        # work exists to correct.
        #
        # SINCE MEASURED, and the flag is now known to be the weaker signal:
        # on sband_short x03_channel_offset is 24.3% global but only 7.4%
        # local, while f09_temporal_skew is 6.7% global and 15.5% local. HDBSCAN
        # responds to local density, so share_knn is the more relevant
        # statistic and should become primary once thresholds are calibrated.
        # Until then the rail caption warns readers to prefer knn where the two
        # disagree.
        #
        # The two share flags are ONE observation, not two: shares sum to 1, so
        # a column at 24% mechanically depresses every other toward the lower
        # bound. Useful for pointing at a column; not independent evidence.
        if np.isfinite(sg[i]):
            if sg[i] > 2 * equal:
                flags.append("share-high")
            elif sg[i] < 0.5 * equal:
                flags.append("share-low")

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
            "share_global": float(sg[i]),
            "share_knn": float(sk[i]),
            "equal_share": float(equal),
            "flags": flags,
        })
    return out
