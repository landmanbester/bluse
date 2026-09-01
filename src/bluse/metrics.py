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


def _spans(labels, freq):
    """(cluster ids, per-cluster frequency span, per-cluster size)."""
    from .diagnostics import group_index
    ids, rows, starts = group_index(labels)
    if not len(ids):
        return ids, np.array([]), np.array([], dtype=int)
    f = np.asarray(freq)[rows]
    spans = (np.maximum.reduceat(f, starts) - np.minimum.reduceat(f, starts))
    counts = np.diff(np.append(starts, len(rows)))
    return ids, spans, counts


def _narrow(labels, freq, thresh, cache=None):
    """
    (hits in clusters narrower than `thresh`, count of such clusters).

    `cache` is an optional (ids, spans, counts) triple from _spans, so the
    grouping is done once per labelling rather than once per threshold and
    once per permutation -- quality() calls this seven times.
    """
    ids, spans, counts = cache if cache is not None else _spans(labels, freq)
    if not len(ids):
        return 0, 0
    m = spans < thresh
    return int(counts[m].sum()), int(m.sum())


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

    NOTE ON DENOMINATORS: the hypergeometric test runs over the LABELLED rows
    of a cluster, `(labels == c) & known`, while the returned fraction counts
    each significant cluster's FULL size. That is deliberate -- the metric is a
    fraction of clustered hits, matching narrow_frac's scale -- but it means a
    cluster that is 90% unlabelled contributes its whole size on the strength
    of a test over a tenth of it.
    """
    from scipy.stats import hypergeom

    known = weak != -1
    if known.sum() == 0:
        return float("nan")
    m_pop = int(known.sum())
    n_pos = int((weak == 0).sum())
    if n_pos == 0 or n_pos == m_pop:
        return float("nan")

    from .diagnostics import group_index
    ids, rows, starts = group_index(labels)
    if not len(ids):
        return float("nan")
    counts = np.diff(np.append(starts, len(rows)))
    lab_known = known[rows].astype(np.int64)
    n_in = np.add.reduceat(lab_known, starts)
    n_zero = np.add.reduceat(((weak[rows] == 0) & known[rows]).astype(np.int64),
                             starts)
    keep = n_in > 0
    if not keep.any():
        return float("nan")
    ps = hypergeom.sf(n_zero[keep] - 1, m_pop, n_pos, n_in[keep])
    sizes = counts[keep]

    ps = np.asarray(ps, dtype=np.float64)
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
                   narrow_frac_at={f"{t:g}": float("nan")
                                   for t in narrow_mhz})
        return out

    # String keys: float keys survive neither a JSON round-trip nor an equality
    # comparison after one, and <tag>_metrics.json carries this dict.
    thresholds = tuple(narrow_mhz)
    span_cache = _spans(labels, freq)
    at = {}
    for t in thresholds:
        hits, n_cl = _narrow(labels, freq, t, span_cache)
        at[f"{t:g}"] = hits / n_clustered
        if t == thresholds[-1]:
            out["narrow_clusters"] = n_cl
    out["narrow_frac_at"] = at
    out["narrow_frac"] = at[f"{thresholds[-1]:g}"]

    # Null for the headline, because every other metric here has one. Small
    # clusters are narrow by chance more often than large ones, and leaf's
    # median size is 6 against eom's 11, so part of the 8.8x could be
    # arithmetic. Permuting the label VECTOR preserves the cluster size
    # distribution exactly, which is the confound being controlled for.
    rng = np.random.default_rng(seed)
    nulls = [_narrow(p, freq, thresholds[-1], _spans(p, freq))[0] / n_clustered
             for p in (rng.permutation(labels) for _ in range(max(1, n_perm)))]
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

    out["median_span_mhz"] = float(np.median(span_cache[1]))

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


def stability(run_fn, seeds=(0, 1, 2, 3, 4)):
    """
    How reproducible is this configuration across shuffle seeds?

    `run_fn(seed) -> labels`. Returns THREE separate numbers and never one
    scalar, because collapsing them is a measured error rather than a
    hypothetical one:

      ari_composite    pairwise ARI over the full label vectors
      ari_restricted   pairwise ARI over points clustered in BOTH runs
      noise_agreement  agreement on the binary (labels >= 0) vector

    sklearn's adjusted_rand_score treats -1 as an ordinary label, so a method
    that leaves half its points unclustered scores agreement for every
    within-noise pair. Measured: leaf scores composite 0.480 against eom's
    0.024 -- a 20x apparent advantage -- but restricted to cluster membership
    the two are 0.0316 and 0.0279, a 13% difference. The composite was
    measuring agreement about what is noise, not agreement about what belongs
    together.

    ari_restricted is the acceptance statistic. The larger reading of those
    numbers is that cluster membership is currently not reproducible under
    EITHER selection method -- both sit at the noise floor -- which is what
    matching (bluse.matching) exists to fix.

    noise_agreement is near-degenerate wherever a method clusters almost
    everything: eom clusters 99.9% of points and scores 0.999 by construction.
    Record it; do not gate on it.
    """
    import itertools

    from sklearn.metrics import adjusted_rand_score

    seeds = tuple(seeds)
    runs = [np.asarray(run_fn(s)) for s in seeds]
    ks = [int(len(np.unique(r[r >= 0]))) for r in runs]

    comp, rest, noise = [], [], []
    for a, b in itertools.combinations(runs, 2):
        comp.append(adjusted_rand_score(a, b))
        both = (a >= 0) & (b >= 0)
        rest.append(adjusted_rand_score(a[both], b[both])
                    if both.sum() > 1 else float("nan"))
        noise.append(float(((a >= 0) == (b >= 0)).mean()))

    return {
        "n_seeds": len(seeds),
        "ari_composite": float(np.mean(comp)) if comp else float("nan"),
        "ari_restricted": float(np.nanmean(rest)) if rest else float("nan"),
        "noise_agreement": float(np.mean(noise)) if noise else float("nan"),
        "k_mean": float(np.mean(ks)),
        "k_min": int(min(ks)),
        "k_max": int(max(ks)),
    }


def epoch_trace(alive_after, n_total):
    """
    GLOBULAR Table 1 for our runs: how much each epoch actually removed.

    `alive_after[i]` is the number of hits still unclustered after epoch i+1.
    Measured on sband_short at defaults: epoch 1 removes 87.9%, epoch 2 12.0%,
    epoch 3 0.1%, and epochs 4-8 remove nothing at all -- so the epoch budget
    is spent in a single pass and five of the eight epochs are dead.
    """
    rows = []
    prev = n_total
    for i, alive in enumerate(alive_after, start=1):
        removed = prev - alive
        rows.append({
            "epoch": i,
            "alive": int(alive),
            "removed": int(removed),
            "pct_of_original": 100.0 * removed / n_total if n_total else 0.0,
        })
        prev = alive
    return rows


def coarsening_null(cluster_runs, family_runs, seed=0):
    """
    Does coarsening inflate family ARI on its own?

    The family-level ARI is the headline result, and it was the only metric
    without a control -- the same asymmetry that let AMI through unchallenged
    for a round. The worry is concrete: matching's default cut is a granularity
    dial that returns about k/2 groups regardless of structure, so a sceptic
    would ask whether merely halving the group count raises agreement.

    It does not. Permuting the cluster -> family assignment while preserving
    the family size distribution destroys the correspondence but keeps the
    coarsening, so this returns what family ARI would be if matching grouped
    clusters arbitrarily. Independently verified on structureless data: two
    random partitions into 72 clusters, each coarsened by Ward at p50, score
    cluster ARI 0.00005 and family ARI -0.00003. ARI's chance correction
    handles it.

    Returns the mean pairwise restricted ARI over the permuted families.
    """
    import itertools

    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    shuffled = []
    for cl, fam in zip(cluster_runs, family_runs):
        cl = np.asarray(cl)
        fam = np.asarray(fam)
        ids = np.unique(cl[cl >= 0])
        if not len(ids):
            shuffled.append(fam)
            continue
        # One family id per cluster, then permute that mapping.
        per_cluster = np.array([fam[cl == c][0] for c in ids])
        rng.shuffle(per_cluster)
        out = np.full(len(cl), -1, dtype=np.int32)
        m = cl >= 0
        out[m] = per_cluster[np.searchsorted(ids, cl[m])]
        shuffled.append(out)

    vals = []
    for a, b in itertools.combinations(shuffled, 2):
        both = (a >= 0) & (b >= 0)
        if both.sum() > 1:
            vals.append(adjusted_rand_score(a[both], b[both]))
    return float(np.mean(vals)) if vals else float("nan")
