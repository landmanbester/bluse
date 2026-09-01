"""
Synthetic fixtures with PLANTED defects, so every diagnostic has a known answer.

Nothing here reads the workspace. The real feature matrices are gitignored
(aug_2026_workshop/features/ and data/ both are, 0 files tracked), so a suite
built on them runs on one machine and nowhere else -- and with no CI, it would
silently stop running the first time it broke.

An earlier draft of this suite used mk_sample_hits as its fixture. That file
cannot serve, for three independent reasons, all measured: it has 0% zero-drift
(it is pre-filtered), so f02's tie fraction there is 0.4525 rather than the
0.266 measured on sband_short; it is 53.7% duplicated against lband_long, which
makes it the worst available choice for anything density-related; and it
contains no weak_label == 1 rows at all, so every label-based metric is
degenerate on it.
"""

import numpy as np
import pandas as pd

# What each planted column is for. Keep these in sync with test assertions.
TIE_FRACTION = 0.30      # of rows sit at exactly 0.0 in "tie_col"
CLIP_FRACTION = 0.05     # of rows are pushed past +/-5 IQRs in "clip_col"
ORDINAL_LEVELS = 8       # distinct values in "ordinal_col"


def synthetic_matrix(n=500, seed=0):
    """
    A feature matrix whose defects are known by construction.

    Returns (X, columns, kinds).

      tie_col     30% of rows at exactly 0.0        -> max_tie_fraction == 0.30
      clip_col    5% of rows 20 sigma out           -> clip_frac == 0.05
      ordinal_col 8 evenly spaced levels            -> n_distinct == 8
      plain_a     clean normal                      -> no flag
      plain_b     clean normal                      -> no flag
    """
    rng = np.random.default_rng(seed)

    n_tie = int(round(n * TIE_FRACTION))
    tie = rng.normal(5.0, 1.0, n)
    tie[:n_tie] = 0.0

    n_clip = int(round(n * CLIP_FRACTION))
    clip = rng.normal(0.0, 1.0, n)
    clip[:n_clip] = 20.0

    ordinal = rng.integers(0, ORDINAL_LEVELS, n).astype(float)

    X = np.column_stack([tie, clip, ordinal,
                         rng.normal(0, 1, n), rng.normal(0, 1, n)])
    columns = ["tie_col", "clip_col", "ordinal_col", "plain_a", "plain_b"]
    kinds = {"tie_col": "continuous", "clip_col": "continuous",
             "ordinal_col": "ordinal", "plain_a": "continuous",
             "plain_b": "continuous"}
    return X, columns, kinds


def synthetic_labelled(n=600, seed=0):
    """
    A labelling with a KNOWN narrow-cluster share and a known enrichment.

    30 clusters of 20 hits each. Clusters 0-5 are "narrow": all their hits sit
    within 0.2 MHz. The rest are spread over 500 MHz. So 6/30 clusters and
    120/600 hits are narrow -> narrow_frac == 0.20 at narrow_mhz=1.0.

    weak_label is 0 for every hit in clusters 0-2 and 1 elsewhere, so those
    three clusters are perfectly enriched in the minority class.
    """
    rng = np.random.default_rng(seed)
    k, per = 30, n // 30
    labels = np.repeat(np.arange(k), per).astype(np.int32)

    freq = np.empty(n)
    for c in range(k):
        m = labels == c
        if c < 6:
            freq[m] = 1000.0 + c + rng.uniform(0, 0.2, m.sum())
        else:
            freq[m] = rng.uniform(1000.0, 1500.0, m.sum())

    weak = np.where(labels < 3, 0, 1).astype(np.int64)
    df = pd.DataFrame({
        "frequency": freq,
        "obsid": rng.integers(0, 4, n),
        "weak_label": weak,
    })
    return labels, df


def synthetic_centroid_space(seed=0):
    """
    Labels plus a feature matrix whose clusters fall into THREE families.

    9 clusters of 40 points. Clusters 0-2 sit near (0,0,0), 3-5 near (10,0,0),
    6-8 near (0,10,0). Any sane matching cuts this into exactly 3 families.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0., 0., 0.]] * 3 + [[10., 0., 0.]] * 3
                       + [[0., 10., 0.]] * 3)
    per = 40
    labels = np.repeat(np.arange(9), per).astype(np.int32)
    X = np.vstack([c + rng.normal(0, 0.3, (per, 3)) for c in centres])
    return labels, X
