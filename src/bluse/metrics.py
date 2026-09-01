#!/usr/bin/env python3
"""
metrics.py -- does this clustering configuration do anything worth having?

Cluster Bench exposed every knob and explained each one, but nothing on screen
said whether a configuration was BETTER, so tuning was by eye on a scatter plot
whose geometry the docs correctly warn against over-reading.

Two things this module deliberately does NOT do, both learned by measurement:

  - It never reports stability as one number. See stability().
  - It does not treat AMI against weak_label as an objective function. Among
    labelled rows the class balance is 26,956 : 872, i.e. 31:1, and AMI's whole
    observed range across every configuration tried is 0.0017-0.0048. It also
    ranks eom above leaf, the opposite of every other signal. It ships
    captioned, alongside a metric that works.

The headline is narrow_frac: the fraction of clustered hits sitting in clusters
that span less than a threshold in frequency. It needs no labels, has a dynamic
range of 0.1% to tens of percent (measured: 0.776% for eom, 6.820% for leaf on
sband_short), and rewards physical coherence -- which is what an RFI taxonomy,
the stated Track B deliverable, actually requires.
"""

from __future__ import annotations

import numpy as np

BH_Q = 0.05                 # Benjamini-Hochberg false-discovery rate
NARROW_MHZ = (0.1, 1.0)     # report the headline at two thresholds


def _sizes(labels):
    lab = labels[labels >= 0]
    if not len(lab):
        return np.array([], dtype=int)
    counts = np.bincount(lab)
    return counts[counts > 0]


def _narrow(labels, freq, thresh):
    """(hits in clusters narrower than `thresh`, count of such clusters)."""
    hits = 0
    n_cl = 0
    for c in np.unique(labels[labels >= 0]):
        m = labels == c
        f = freq[m]
        if f.max() - f.min() < thresh:
            hits += int(m.sum())
            n_cl += 1
    return hits, n_cl


def _enrichment(labels, weak):
    """
    Fraction of clustered hits in clusters significantly enriched in
    weak_label == 0, one-sided hypergeometric, Benjamini-Hochberg at BH_Q.

    Expressed in HITS, not in clusters, so it shares a scale with narrow_frac.
    A per-cluster percentage would compare 79 clusters against 2,127 on a
    statistic whose denominator is the cluster count -- the same
    non-comparability that made AMI useless here.

    Detection floor, at the measured global rate of 872/27,828 = 3.13%: a fully
    confined cluster of 6 gives p ~ 9.4e-10 and a fully confined cluster of 4
    gives p ~ 9.6e-7, both clearing BH at ~2,000 tests; a cluster of 3 gives
    p ~ 3.1e-5 against a threshold near 2.4e-5 and is marginal. At
    min_cluster_size = 4 only a FULLY confined cluster clears -- 3-of-4 gives
    p ~ 1.2e-4 and fails. So enrichment must not be compared across
    configurations with different min_cluster_size.
    """
    from scipy.stats import hypergeom

    known = weak != -1
    if known.sum() == 0:
        return float("nan")
    m_pop = int(known.sum())
    n_pos = int((weak == 0).sum())
    if n_pos == 0 or n_pos == m_pop:
        return float("nan")

    ps, sizes = [], []
    for c in np.unique(labels[labels >= 0]):
        m = (labels == c) & known
        n_in = int(m.sum())
        if n_in == 0:
            continue
        k = int((weak[m] == 0).sum())
        sizes.append(int((labels == c).sum()))
        ps.append(float(hypergeom.sf(k - 1, m_pop, n_pos, n_in)))
    if not ps:
        return float("nan")

    ps = np.asarray(ps)
    sizes = np.asarray(sizes)
    order = np.argsort(ps)
    m_tests = len(ps)
    thresh = (np.arange(1, m_tests + 1) / m_tests) * BH_Q
    passing = ps[order] <= thresh
    n_sig = int(np.max(np.nonzero(passing)[0]) + 1) if passing.any() else 0
    return int(sizes[order][:n_sig].sum()) / max(int((labels >= 0).sum()), 1)


def quality(labels, df, *, narrow_mhz=NARROW_MHZ, n_perm=5, seed=0):
    """
    Cluster-quality statistics for one labelling.

    `df` supplies `frequency` and, when present, `weak_label`. Rows of `df`
    correspond one-to-one with `labels`.
    """
    labels = np.asarray(labels)
    n = len(labels)
    freq = df["frequency"].to_numpy(dtype=np.float64)
    sizes = _sizes(labels)
    n_clustered = int((labels >= 0).sum())

    out = {
        "n": n,
        "n_clusters": int(len(sizes)),
        "clustered_pct": 100.0 * n_clustered / n if n else 0.0,
        "largest_pct": 100.0 * sizes.max() / n if len(sizes) else 0.0,
        "median_size": int(np.median(sizes)) if len(sizes) else 0,
    }

    if not len(sizes):
        out.update(narrow_frac=float("nan"), narrow_frac_null=float("nan"),
                   narrow_enrichment=float("nan"), narrow_clusters=0,
                   median_span_mhz=float("nan"), ami=float("nan"),
                   enrichment=float("nan"),
                   narrow_frac_at={t: float("nan") for t in narrow_mhz})
        return out

    thresholds = tuple(narrow_mhz)
    at = {}
    for t in thresholds:
        hits, n_cl = _narrow(labels, freq, t)
        at[t] = hits / n_clustered
        if t == thresholds[-1]:
            out["narrow_clusters"] = n_cl
    out["narrow_frac_at"] = at
    out["narrow_frac"] = at[thresholds[-1]]

    # Null for the headline, because every other metric here has one. Small
    # clusters are narrow by chance more often than large ones, and leaf's
    # median size is 6 against eom's 11, so part of the 8.8x could be
    # arithmetic. Permuting the label VECTOR preserves the cluster size
    # distribution exactly, which is the confound being controlled for.
    rng = np.random.default_rng(seed)
    nulls = [_narrow(rng.permutation(labels), freq, thresholds[-1])[0]
             / n_clustered for _ in range(max(1, n_perm))]
    out["narrow_frac_null"] = float(np.mean(nulls))
    # 0/0 is undefined, not infinite. A permuted labelling of a wideband file
    # routinely produces NO narrow clusters at all, and reporting inf there
    # would read as "infinitely enriched" when the truth is "no signal either
    # way". Only a genuinely non-zero numerator over a zero null is inf.
    if out["narrow_frac_null"] > 0:
        out["narrow_enrichment"] = out["narrow_frac"] / out["narrow_frac_null"]
    elif out["narrow_frac"] > 0:
        out["narrow_enrichment"] = float("inf")
    else:
        out["narrow_enrichment"] = float("nan")

    spans = [float(freq[labels == c].max() - freq[labels == c].min())
             for c in np.unique(labels[labels >= 0])]
    out["median_span_mhz"] = float(np.median(spans))

    if "weak_label" in df.columns:
        from sklearn.metrics import adjusted_mutual_info_score
        weak = df["weak_label"].to_numpy()
        known = weak != -1
        out["ami"] = (float(adjusted_mutual_info_score(weak[known],
                                                       labels[known]))
                      if known.sum() > 10 else float("nan"))
        out["enrichment"] = _enrichment(labels, weak)
    else:
        out["ami"] = float("nan")
        out["enrichment"] = float("nan")
    return out
