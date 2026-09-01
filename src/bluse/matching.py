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
no perplexity, no seed, no embedding. Measured cost at D=15, scipy nn-chain
(O(n) memory):

    2,000 centroids (Bench, leaf)      0.04 s
   20,000 centroids                    6.78 s
  ~78,000 centroids (full `all`)      ~100 s   (extrapolated O(n^2))

so one exact implementation serves the Bench interactively and the CLI offline.
A k-NN-graph approximation was measured at 75.8 s for 80,000 -- slower than
exact Ward and strictly worse -- and is not used.

CAVEAT, and it belongs in any write-up of the first family taxonomy: Ward runs
on centroids in the SCALED feature space, and that space still carries the 14x
distance-share spread that the contribution-equalising scaling work has not yet
fixed. With x03_channel_offset at 24.3% and f07_kurt_bw_corr at 13.1%, families
are grouped substantially by channel offset and by a clipped correlation
coefficient. The first taxonomy is provisional and must be re-derived once that
scaling lands.
"""

from __future__ import annotations

import numpy as np


def centroids(labels, X):
    """(cluster ids, their centroids in the columns of X)."""
    ids = np.unique(labels[labels >= 0])
    C = (np.vstack([X[labels == c].mean(axis=0) for c in ids]) if len(ids)
         else np.zeros((0, X.shape[1])))
    return ids, C


def _nn_distances(C):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(C)
    d, _ = nn.kneighbors(C)
    return [float(v) for v in d[:, 1]]


def derive_cut_pct(heights, pct=50):
    """
    Cut at a percentile of the Ward merge heights. THE DEFAULT.

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

    Percentiles of the merge-height distribution behave, and p50 is where the
    reproducibility gain peaks. Measured on sband_short, 3 seeds, membership
    ARI and narrow-cluster share:

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

    For leaf the same operation helps much less and costs more, which is the
    real eom-vs-leaf trade-off: leaf wins on coherence (6.820% against 0.776%),
    eom-plus-matching wins on reproducibility (0.519 against 0.108).
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


def match(labels, X, *, cut=None, pct=50, rule="pct", method="ward"):
    """
    Group clusters into families.

    Returns (family_ids, info). `family_ids` is aligned with `labels` and is
    -1 wherever `labels` is -1.

    The cut is chosen in this order: an explicit `cut` wins; else `rule`
    selects a derivation. Default is the 50th percentile of the Ward merge
    heights, which is where the measured reproducibility gain peaks (see
    derive_cut_pct). A hardcoded constant is not an option -- the per-file
    drift lattice alone spans 5.26x across our eight files (uhf_long 0.00204
    Hz/s, sband_short 0.01071), so a value tuned on one file is wrong on
    another.

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
    info = {"cut": float(cut) if cut is not None else 0.0,
            "n_clusters": int(len(ids)), "n_families": 0,
            "nn_distances": [], "method": method}
    if len(ids) == 0:
        return fam, info
    if len(ids) == 1:
        fam[labels == ids[0]] = 0
        info["n_families"] = 1
        return fam, info

    info["nn_distances"] = _nn_distances(C)

    if method == "tsne":
        assign = _match_tsne(C)
    else:
        Z = linkage(C, method="ward")
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

    # Map cluster id -> family id, vectorised. ids is sorted, so searchsorted
    # gives each hit's position in `ids` in one pass.
    assign = np.asarray(assign, dtype=np.int32)
    clustered = labels >= 0
    pos = np.searchsorted(ids, labels[clustered])
    fam[clustered] = assign[pos] - assign.min()
    info["n_families"] = int(len(np.unique(assign)))
    return fam, info
