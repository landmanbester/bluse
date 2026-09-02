#!/usr/bin/env python3
"""
matching.py -- group the clusters the batching loop mints separately.

HDBSCAN mints local ids 0..k-1 in EVERY batch, and the epoch loop runs many
batches, so one physical population becomes a fresh cluster id in every batch
and every epoch it appears in. That is not cosmetic. Measured on sband_short:
two runs of an identical configuration differing only in shuffle seed agree at
ARI 0.024, while dropping a whole feature at FIXED seed leaves ARI 0.75-0.89.
Batch membership, not feature geometry, is what decides which cluster a hit
lands in -- so cluster ids as they stand are batch artefacts, and matching is a
correctness fix rather than an enhancement.

Ward linkage on cluster centroids, cut at an explicit distance. Deterministic:
no perplexity, no seed, no embedding. Measured cost at D=15:

    2,000 centroids (Bench, leaf)      0.04 s        18 MB
   20,000 centroids                    6.78 s       1.6 GB
  ~78,000 centroids (full `all`)      ~100 s        ~24 GB  -- REFUSED

MEMORY IS O(k^2), NOT O(k). An earlier version of this docstring claimed
scipy's nn-chain is O(n) in memory. That is wrong and was asserted without
being measured -- peak allocation tracks the condensed distance matrix
exactly (18 MB against 16 MB predicted at k=2,000; 162 MB against 144 MB at
k=6,000). At the ~78,000 centroids `leaf` produces on all_features.parquet
the condensed matrix alone is 24.3 GB, so `match()` refuses rather than
exhausting memory; raise MAX_CENTROIDS deliberately if you have the RAM.

A k-NN-graph approximation was measured at 75.8 s for 80,000 -- slower than
exact Ward at that size and strictly worse in quality -- and is not used, so
above the guard the honest answer is to cluster a sample or use a coarser
`min_cluster_size` rather than to approximate.

CAVEAT, and it belongs in any write-up of the first family taxonomy: the cut
is a granularity dial (see derive_cut_pct), so the number of families is a
choice rather than a discovery. Treat the first taxonomy as provisional for
that reason.

An earlier version of this caveat blamed the scaled feature space instead,
naming x03_channel_offset at 24.3% and f07_kurt_bw_corr at 13.1% of the global
distance share. That was the wrong statistic: measured on k-NN pairs -- which
is what HDBSCAN actually responds to -- x03 contributes 5.9% against an equal
share of 6.7%, i.e. it is already well behaved locally. The column the local
measurement does indict is f09_temporal_skew, at 15.3% local against 6.7%
equal, and benign on every global statistic.
"""

from __future__ import annotations

import numpy as np

from . import diagnostics

# Above this many clusters, scipy's Ward needs a condensed distance matrix
# larger than most machines have: k=40,000 is 6.4 GB, k=78,000 is 24.3 GB.
# Measured, not assumed -- see the module docstring.
MAX_CENTROIDS = 40_000


def centroids(labels, X):
    """
    (cluster ids, their centroids in the columns of X).

    Grouped by sort + reduceat rather than a mask per cluster: the latter is
    O(k*n), which is minutes at the ~80,000 clusters `leaf` produces on
    all_features.parquet. See diagnostics.group_index.
    """
    ids, rows, starts = diagnostics.group_index(labels)
    if not len(ids):
        return ids, np.zeros((0, X.shape[1]))
    counts = np.diff(np.append(starts, len(rows)))
    sums = np.add.reduceat(np.asarray(X)[rows], starts, axis=0)
    return ids, sums / counts[:, None]


def _nn_distances(C):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(C)
    d, _ = nn.kneighbors(C)
    return [float(v) for v in d[:, 1]]


def derive_cut_pct(heights, pct=50):
    """
    Cut at a percentile of the Ward merge heights. THE DEFAULT.

    READ THIS FIRST: this is a GRANULARITY DIAL, not a structure-sensitive cut.
    A dendrogram over k clusters has k-1 merges, and cutting at the p-th
    percentile of their heights performs the lowest p% of them, so it returns
    approximately k(1 - p/100) groups REGARDLESS of whether the data has any
    structure. Verified on pure 15-D Gaussian noise:

        k=72   -> p25 54, p50 36, p90 8      (predicted 54 / 36 / 7)
        k=2162 -> p25 1621, p50 1081, p90 217 (predicted 1622 / 1081 / 216)

    and our own real runs sit exactly on that line: 2,162 -> 1,081 and
    72 -> 36, both k/2 to within rounding. So `pct=50` means "halve the number
    of groups", and nothing more.

    That is a defensible default and it demonstrably works -- see the
    reproducibility table below -- but two consequences matter. It is
    scale-invariant, so the 5.26x per-file drift-lattice spread cannot move it
    and the rule transfers across files unchanged. It is also
    STRUCTURE-invariant, which is the property to worry about: it will halve
    the group count on uhf_long too, whatever that dendrogram looks like.
    Prefer `n_families=` when you know the granularity you want; `pct` is kept
    because it is what the sweep below was measured against.

    Chosen by measurement, after two other rules were tried and failed:

      nearest-neighbour percentile (the first thing specified). Wrong units.
        Ward merge heights carry a cluster-size term and are not raw distances,
        so a threshold read off the distance distribution is thresholding the
        wrong quantity. On a fixture with three well-separated families the
        median centroid NN distance is 0.070 while within-family merges run to
        0.201, so it split 3 real families into 6.

      largest gap in merge heights. Right on a clean fixture, badly wrong on
        real data. The final few merges of any dendrogram are the biggest
        jumps, so the gap statistic is dominated by the root: on sband_short it
        collapsed 2,162 leaf clusters into 4 families spanning the whole band,
        taking the narrow-cluster share from 6.820% to 0.000%.

    Percentiles behave predictably where the other two rules did not, and p50
    is where the reproducibility gain peaks for eom. Measured on sband_short,
    3 seeds, membership ARI and narrow-cluster share:

                        clusters    p25      p50      p90
        eom   ARI         0.0279  0.2137   0.5190   0.4031
        eom   narrow %     0.776   0.776    0.670    0.553
        leaf  ARI         0.0316  0.0686   0.1077   0.2383
        leaf  narrow %     6.820   5.140    4.156    1.144

    So for eom, matching at p50 buys an 18.6x improvement in reproducibility
    for a 0.11-point cost in coherence, and at p25 it is free -- 7.7x better
    ARI at an identical narrow share. Note the ARI is NOT monotone in the cut:
    it peaks near p50 and falls by p90, because past some point matching fuses
    populations that are genuinely distinct.

    Those columns compare the two methods at DIFFERENT granularities, though,
    because k differs by 30x. Matched at equal family count the picture changes
    and both apparent advantages turn out to be granularity effects pointing in
    opposite directions:

        target families    eom ARI / narrow%     leaf ARI / narrow%
                    20       0.5189 / 0.567       0.4524 / 0.000
                    36       0.5190 / 0.670       0.4888 / 0.194
                    72       0.0332 / 0.776       0.4170 / 0.774

    So the 4.8x reproducibility gap largely evaporates at matched count (0.519
    against 0.489 at 36), and leaf's 8.8x coherence advantage evaporates too --
    coarsening leaf to eom's granularity takes its narrow share from 6.820% to
    0.194%, BELOW eom's 0.670%. leaf's coherence is a property of its fine
    granularity, not of the extraction method.

    WHY p50 IS NEVERTHELESS A GOOD DEFAULT FOR eom. Sweeping the whole cut
    chain on sband_short shows family ARI is flat at its maximum across a broad
    plateau and then falls off a cliff:

        families      4      8     19     23     36     39     45     54     72
        famARI    0.477  0.403  0.519  0.519  0.519  0.519  0.261  0.199  0.033
        narrow %  0.000  0.553  0.553  0.567  0.670  0.670  0.710  0.776  0.776

    p50 picks 36 -- inside the plateau, and at its best narrow share. The
    plateau ends between 39 and 45, so the default has roughly 10% headroom
    before reproducibility collapses. That is luck rather than measurement, but
    it is why the default has held up.

    leaf has no such plateau: famARI falls monotonically (0.762 at 4 families
    to 0.032 at 2,157) while the narrow share rises monotonically, so there is
    no knee for any rule to find and the count is a genuine trade-off the
    analyst must state. Use `n_families=`.
    """
    h = np.asarray(heights, dtype=np.float64)
    if not len(h):
        return 0.0
    return float(np.percentile(h, pct))


def derive_cut_gap(heights):
    """
    Cut at the largest gap in the merge heights -- reading the dendrogram the
    way an eye would.

    NOT the default. Correct on well-separated data, dominated by the root
    merges on real data; see derive_cut_pct for the numbers. Kept because it is
    the right answer when families genuinely are separated, which is the case
    the synthetic fixture covers and the case a future scaling fix may create.

    DO NOT ADD A max_pct BOUND. Restricting the gap search to merges below a
    percentile -- so the root merges cannot dominate -- was proposed by the
    implementation review, measured, and rejected. Three results, in
    docs/matching-cut-experiment-2026-09.md:

      1. The bound IS the granularity dial, and a worse-behaved one. On
         sband_short/leaf the family count runs 4 (unbounded), 31 (p99), 113
         (p95), then 2,157 (p90) -- essentially no merging at all. A five-point
         change in an arbitrary parameter moves the answer by 19x, where `pct`
         moves smoothly and predictably.

      2. It does not respond to structure, which was the entire motivation. On
         60 clusters carrying 3 planted families, 6 planted families, or none
         at all, no bound recovers the planted count, and the counts returned
         on structureless data are indistinguishable from those on strongly
         structured data (at p90: 14 on noise, 30 on 3 families, 13 on 6).

      3. It cannot help anyway. See match(): a cut only ever chooses a family
         count, so no threshold rule can find a partition that `n_families=`
         cannot state outright.

    The gap rule's success on the fixture is also weaker evidence than it
    looks: because root merges dominate, it returns 2-4 families on almost any
    input, and lands on exactly 3 about a quarter of the time by coincidence.
    See test_gap_rule_is_not_evidence_of_structure_sensitivity.
    """
    h = np.sort(np.asarray(heights, dtype=np.float64))
    if len(h) < 2:
        return float(h[0]) if len(h) else 0.0
    gaps = np.diff(h)
    i = int(gaps.argmax())
    return float((h[i] + h[i + 1]) / 2.0)


def derive_cut_quantile(C, quantile=50):
    """
    A cut read off the centroid nearest-neighbour distribution.

    NOT the default -- wrong units, see derive_cut_pct. Kept for reproducing
    the originally specified behaviour.
    """
    if len(C) < 2:
        return 0.0
    return float(np.percentile(_nn_distances(C), quantile))


def _match_tsne(C):
    """GLOBULAR's route, for reproduction. Not the default; not deterministic."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_pc = min(6, C.shape[1], len(C) - 1)
    P = PCA(n_components=n_pc, random_state=0).fit_transform(C)
    perp = float(min(40, max(5, len(C) // 4)))
    E = TSNE(n_components=2, perplexity=perp, early_exaggeration=4,
             random_state=0, init="pca").fit_transform(P)
    lab = HDBSCAN(min_cluster_size=2, n_jobs=-1).fit_predict(E)
    # t-SNE noise gets its own family each, so the count stays honest.
    out = lab.copy()
    nxt = int(lab.max()) + 1 if (lab >= 0).any() else 0
    for i in np.nonzero(lab < 0)[0]:
        out[i] = nxt
        nxt += 1
    return out


def match(labels, X, *, cut=None, pct=50, rule="pct", n_families=None,
          method="ward"):
    """
    Group clusters into families.

    Returns (family_ids, info). `family_ids` is aligned with `labels` and is
    -1 wherever `labels` is -1.

    The cut is chosen in this order: an explicit `cut` wins; then
    `n_families`; else `rule` selects a derivation. Default is the 50th
    percentile of the Ward merge heights, which is where the measured
    reproducibility gain peaks (see derive_cut_pct). A hardcoded constant is
    not an option -- the per-file drift lattice alone spans 5.26x across our
    eight files (uhf_long 0.00204 Hz/s, sband_short 0.01071), so a value tuned
    on one file is wrong on another.

    PREFER `n_families=`. Every horizontal cut of a fixed Ward tree is
    uniquely determined by the number of families it leaves -- verified over
    1,000 thresholds on the real sband_short trees with zero exceptions, and
    pinned by test_a_distance_cut_is_exactly_a_choice_of_family_count. So a
    cut rule never selects a better partition, only a point on a fixed nested
    chain, and `n_families` says which point you want without pretending a
    threshold derived it. `cut` and `pct` are kept because the published
    sweeps were measured against them.

    method="ward"  exact Ward linkage on centroids. The default.
    method="tsne"  GLOBULAR's own route -- PCA to 6, t-SNE (perplexity 40,
                   early exaggeration 4), HDBSCAN on the 2-D embedding. Kept
                   as a reproduction path, not a default: the paper's own
                   health warnings about t-SNE are the argument for having a
                   deterministic alternative.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    labels = np.asarray(labels)
    X = np.asarray(X, dtype=np.float64)
    ids, C = centroids(labels, X)

    fam = np.full(len(labels), -1, dtype=np.int32)
    # cut_source is set here, not only on the linkage path: the early returns
    # below used to omit it, and the CLI indexes it unconditionally, so a valid
    # all-noise or single-cluster run raised KeyError instead of being written.
    info = {"cut": float(cut) if cut is not None else 0.0,
            "n_clusters": int(len(ids)), "n_families": 0,
            "nn_distances": [], "method": method,
            "cut_source": "explicit" if cut is not None else "none needed"}
    if len(ids) == 0:
        return fam, info
    if len(ids) == 1:
        fam[labels == ids[0]] = 0
        info["n_families"] = 1
        return fam, info

    if len(ids) > MAX_CENTROIDS and method != "tsne":
        raise MemoryError(
            f"{len(ids):,} clusters would need a "
            f"{len(ids) * (len(ids) - 1) // 2 * 8 / 1e9:.1f} GB condensed "
            f"distance matrix for exact Ward. Cluster a sample, raise "
            f"min_cluster_size, or raise matching.MAX_CENTROIDS deliberately.")

    info["nn_distances"] = _nn_distances(C)

    if method == "tsne":
        assign = _match_tsne(C)
    else:
        Z = linkage(C, method="ward")
        if cut is None and n_families is not None:
            # Ask for the count directly. maxclust rather than a derived
            # threshold because it is exact under tied merge heights, where
            # a height-based cut can overshoot.
            n = int(np.clip(n_families, 2, len(ids)))
            assign = fcluster(Z, t=n, criterion="maxclust")
            # Report the height that cut would have sat at, for continuity
            # with the rule paths -- rows of Z are ascending, and leaving n
            # groups means performing the lowest len(ids)-n merges.
            info["cut"] = float(Z[len(ids) - n - 1, 2]) if n < len(ids) else 0.0
            info["cut_source"] = f"n_families={n_families}"
            return _assign(fam, labels, ids, assign, info)
        if cut is not None:
            source = "explicit"
        elif rule == "gap":
            cut, source = derive_cut_gap(Z[:, 2]), "merge-height gap"
        elif rule == "nn":
            cut, source = derive_cut_quantile(C, pct), f"centroid nn p{pct}"
        else:
            cut, source = derive_cut_pct(Z[:, 2], pct), f"merge-height p{pct}"
        info["cut"] = float(cut)
        info["cut_source"] = source
        # criterion="distance" with t<=0 would put every cluster in its own
        # family; guard so a degenerate cut does not look like a real answer.
        assign = fcluster(Z, t=max(float(cut), 1e-12), criterion="distance")

    return _assign(fam, labels, ids, assign, info)


def _assign(fam, labels, ids, assign, info):
    """
    Map cluster id -> family id, vectorised. ids is sorted, so searchsorted
    gives each hit's position in `ids` in one pass.
    """
    assign = np.asarray(assign, dtype=np.int32)
    clustered = labels >= 0
    pos = np.searchsorted(ids, labels[clustered])
    fam[clustered] = assign[pos] - assign.min()
    info["n_families"] = int(len(np.unique(assign)))
    return fam, info
