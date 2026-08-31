#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib", "h5py"]
# ///
"""
track_b_cluster.py -- HDBSCAN over the Track B features, to name the RFI.

The point is not to score hits as anomalous. It is to discover the *families*
of interference present in the data, so that "here are 200 unexplained hits"
replaces "here are some outliers". Brzycki et al. 2025 identified ~59 distinct
RFI clusters this way and cut false positives by 93.1%.

    uv run track_b_cluster.py                              # all_features.parquet
    uv run track_b_cluster.py --file sband_short
    uv run track_b_cluster.py --epochs 6 --batch 3000      # GLOBULAR's scheme
    uv run track_b_cluster.py --min-cluster-size 10

Two modes:

  epochs (default)  GLOBULAR's iterative batching: cluster ~3000 hits at a
                    time, keep only the noise points, reshuffle, repeat. Each
                    epoch strips another layer of recognisable RFI, and what
                    survives is what no batch could group.

                    The batching is not a memory workaround -- it is integral.
                    GLOBULAR's hyperparameters (min_cluster_size 4, min_samples
                    2, epsilon 0.18) are tuned for ~3000-point batches and
                    degrade badly on one large pass: 205 clusters vs 10 on
                    sband_short.

  single            one HDBSCAN pass over a sample. Kept for comparison and for
                    trying your own hyperparameters at scale.

KNOWN GAP -- CROSS-BATCH CLUSTER MATCHING
-----------------------------------------
Cluster ids are NOT comparable between batches or epochs. The same RFI family
recurs in every batch and is labelled separately each time: on sband_short,
clusters 100000/100001/100002/200000 have near-identical medians (2232.8 MHz,
drift 0.032 Hz/s, 11.4 Hz bandwidth) and are plainly one population split four
ways. GLOBULAR closed this with cross-batch matching to arrive at ~59 named RFI
families; we have not implemented it yet.

Until it exists, read the cluster table as "these hits grouped with something"
rather than as a taxonomy, and treat the UNCLUSTERED set -- which is well
defined regardless -- as the actual output.

Outputs into --outdir:
    <tag>_clusters.parquet   every input row plus cluster_label and a
                             per-cluster summary, ready to join back
    <tag>_summary.csv        one row per cluster: size, medians, purity
    <tag>_umap.png           feature space coloured by cluster
    <tag>_clusters.png       median spectrum and drift of the largest clusters
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, "features")


def feature_matrix(df, columns=None, drop_saturated=True, scaling="robust"):
    """
    Assemble the feature matrix that HDBSCAN will see.

    Saturated columns are excluded by default: f08 is unresolved for ~72% of
    hits because our spectral window is far narrower than GLOBULAR's, so
    including it mostly clusters on "hit the window edge". f12's own saturation
    flag fires for well under 1% and it is kept.

    SCALING IS PART OF THE MODEL. HDBSCAN uses Euclidean distance, so a feature
    with a wider spread contributes proportionally more. GLOBULAR's published
    transforms alone do not equalise that on our data: measured on sband_short,
    the interquartile ranges of the normalised features span 0.036
    (x03_channel_offset) to 5.88 (f02_abs_drift, whose quantile-normal transform
    throws the 33.5% zero-drift spike out to -5.2). Distance is then essentially
    drift rate and nothing else, and the result is two clusters holding 90% of
    the data.

      "robust"    (default) centre on the median, divide by the IQR, then clip
                  to +/-5. Equalises contribution without letting the heavy
                  SNR tail dominate.
      "quantile"  rank-transform every feature to a uniform distribution. The
                  most aggressive equaliser; discards magnitude entirely.
      "none"      GLOBULAR's literal spec. Kept so the difference is
                  reproducible, not because it works well here.
    """
    if columns is None:
        columns = [c + "_n" for c in F.all_columns()
                   if c + "_n" in df.columns and not c.endswith("_saturated")]
        if drop_saturated:
            columns = [c for c in columns if not c.startswith("f08_")]
    X = np.array(df[columns].to_numpy(dtype=np.float64), copy=True)
    good = np.isfinite(X).all(axis=1)

    if scaling == "robust":
        sub = X[good]
        med = np.median(sub, axis=0)
        q75, q25 = np.percentile(sub, [75, 25], axis=0)
        iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
        X = np.clip((X - med) / iqr, -5, 5)
    elif scaling == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, int(good.sum())),
                                 subsample=200_000, random_state=0)
        X[good] = qt.fit_transform(X[good])
    return X, columns, good


_EPSILON_FALLBACKS = 0


def run_hdbscan(X, args):
    """
    HDBSCAN with a fallback for a bug in sklearn's epsilon search.

    `cluster_selection_epsilon > 0` raises "only 0-dimensional arrays can be
    converted to Python scalars" from `_tree.pyx:traverse_upwards` when a batch
    collapses to a degenerate condensed tree -- which happens on the small,
    homogeneous batches of the late epochs. Retrying that batch with epsilon=0
    costs only the cluster merging, so we do that and count it rather than
    losing the batch.
    """
    global _EPSILON_FALLBACKS
    kw = dict(min_cluster_size=args.min_cluster_size,
              min_samples=args.min_samples,
              cluster_selection_method="eom", n_jobs=-1)
    try:
        return HDBSCAN(cluster_selection_epsilon=args.epsilon, **kw).fit_predict(X)
    except (TypeError, ValueError):
        _EPSILON_FALLBACKS += 1
        try:
            return HDBSCAN(cluster_selection_epsilon=0.0, **kw).fit_predict(X)
        except (TypeError, ValueError):
            return np.full(len(X), -1, dtype=np.int32)


def cluster_single(df, args):
    X, cols, good = feature_matrix(df, scaling=args.scaling)
    idx = np.where(good)[0]
    if args.sample and len(idx) > args.sample:
        idx = np.random.default_rng(args.seed).choice(idx, args.sample,
                                                      replace=False)
        idx.sort()
    print(f"  clustering {len(idx):,} hits on {len(cols)} features")
    labels = np.full(len(df), -2, dtype=np.int32)      # -2 = not clustered
    labels[idx] = run_hdbscan(X[idx], args)
    return labels, cols


def cluster_epochs(df, args):
    """
    GLOBULAR's iterative batching: cluster a batch, discard everything that
    landed in a cluster, carry the noise points forward, repeat.

    Each epoch removes another layer of RFI that some batch was able to
    recognise. What is left at the end is what no batch could group with
    anything else -- which is the interesting set.
    """
    X, cols, good = feature_matrix(df, scaling=args.scaling)
    alive = np.where(good)[0]
    rng = np.random.default_rng(args.seed)
    labels = np.full(len(df), -2, dtype=np.int32)
    epoch_of = np.full(len(df), -1, dtype=np.int32)
    print(f"  epoch 0: {len(alive):,} hits, {len(cols)} features")

    for ep in range(1, args.epochs + 1):
        rng.shuffle(alive)
        survivors = []
        for s in range(0, len(alive), args.batch):
            b = alive[s:s + args.batch]
            if len(b) < args.min_cluster_size * 2:
                survivors.append(b)
                continue
            lab = run_hdbscan(X[b], args)
            clustered = lab >= 0
            # Namespace cluster ids by epoch so they stay distinguishable.
            labels[b[clustered]] = lab[clustered] + ep * 100_000
            epoch_of[b[clustered]] = ep
            survivors.append(b[~clustered])
        alive = np.concatenate(survivors) if survivors else np.array([], int)
        pct = 100 * len(alive) / max(1, int(good.sum()))
        print(f"  epoch {ep}: {len(alive):,} remaining ({pct:.1f}%)")
        if len(alive) < args.min_cluster_size * 2:
            break
    if _EPSILON_FALLBACKS:
        print(f"  ({_EPSILON_FALLBACKS} batches retried with epsilon=0 "
              f"after the sklearn epsilon-search bug)")

    labels[alive] = -1                                  # never clustered
    epoch_of[alive] = args.epochs + 1
    return labels, cols, epoch_of


def summarise(df, labels, outdir, tag):
    df = df.copy()
    df["cluster_label"] = labels
    n = len(df)
    clustered = labels >= 0
    noise = labels == -1

    skipped = labels == -2
    print(f"\n  {int(clustered.sum()):,} hits clustered ({100*clustered.mean():.1f}%)"
          f", {int(noise.sum()):,} noise ({100*noise.mean():.1f}%)"
          f", {len(np.unique(labels[clustered])):,} clusters")
    if skipped.any():
        # A normalised feature is NaN where its log transform saw a
        # non-positive raw value -- f12_bandwidth_hz is 0 when no channel
        # clears 1% of the peak, f13_redness when the periodogram denominator
        # vanishes. Those rows cannot enter a Euclidean clusterer.
        print(f"  {int(skipped.sum()):,} hits ({100*skipped.mean():.1f}%) had a "
              f"non-finite normalised feature and were not clustered")

    rows = []
    for lab in np.unique(labels[clustered]):
        m = labels == lab
        d = df[m]
        rows.append({
            "cluster": int(lab),
            "n": int(m.sum()),
            "frac_rfi_label": float((d.weak_label == 1).mean()),
            "frac_confined": float((d.weak_label == 0).mean()),
            "median_freq_mhz": float(d.frequency.median()),
            "freq_span_mhz": float(d.frequency.max() - d.frequency.min()),
            "median_snr": float(d.snr.median()),
            "median_abs_drift": float(d.f02_abs_drift.median()),
            "median_bandwidth_hz": float(d.f12_bandwidth_hz.median()),
            "median_n_beams": float(d.n_beams.median()),
            "n_obs": int(d.obsid.nunique()),
        })
    summary = pd.DataFrame(rows).sort_values("n", ascending=False)

    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(os.path.join(outdir, f"{tag}_summary.csv"), index=False)
    df.to_parquet(os.path.join(outdir, f"{tag}_clusters.parquet"), index=False)

    if len(summary):
        print(f"\n  largest clusters:")
        print(f"  {'id':>8} {'n':>8} {'RFI%':>6} {'freq [MHz]':>12} "
              f"{'span':>9} {'SNR':>9} {'|drift|':>8} {'BW Hz':>7} {'beams':>6}")
        for _, r in summary.head(12).iterrows():
            print(f"  {int(r.cluster):>8} {int(r.n):>8,} "
                  f"{100*r.frac_rfi_label:>5.0f}% {r.median_freq_mhz:>12.4f} "
                  f"{r.freq_span_mhz:>9.4f} {r.median_snr:>9.3g} "
                  f"{r.median_abs_drift:>8.4f} {r.median_bandwidth_hz:>7.1f} "
                  f"{r.median_n_beams:>6.0f}")

    # The interesting set: never clustered, and spatially confined.
    interesting = df[noise & (df.n_beams <= 4)]
    print(f"\n  UNCLUSTERED and confined to <=4 beams: {len(interesting):,} hits")
    if len(interesting):
        cols = ["file", "row", "obsid", "sourceName", "frequency", "driftRate",
                "snr", "n_beams", "f12_bandwidth_hz"]
        p = os.path.join(outdir, f"{tag}_interesting.csv")
        interesting.sort_values("snr", ascending=False)[cols].to_csv(p, index=False)
        print(f"  wrote {p}")
    return df, summary


def plot(df, labels, cols, outdir, tag, scaling="robust"):
    X, _, good = feature_matrix(df, columns=cols, scaling=scaling)
    idx = np.where(good)[0]
    if len(idx) > 40000:
        idx = np.random.default_rng(0).choice(idx, 40000, replace=False)
    emb = PCA(n_components=2, random_state=0).fit_transform(X[idx])
    lab = labels[idx]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{tag}  --  Track B clustering", fontsize=13)

    noise = lab < 0
    ax[0].scatter(emb[noise, 0], emb[noise, 1], s=2, c="#cccccc",
                  label="noise / unclustered", rasterized=True)
    ax[0].scatter(emb[~noise, 0], emb[~noise, 1], s=2, c=lab[~noise] % 20,
                  cmap="tab20", rasterized=True)
    ax[0].set_title("feature space (PCA), coloured by cluster")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(markerscale=4)

    wl = df["weak_label"].to_numpy()[idx]
    for v, c, l in [(1, "#c0392b", "RFI (many beams)"),
                    (-1, "#bbbbbb", "ambiguous"),
                    (0, "#2980b9", "confined (<=2 beams)")]:
        m = wl == v
        ax[1].scatter(emb[m, 0], emb[m, 1], s=2, c=c, label=l, rasterized=True)
    ax[1].set_title("same space, coloured by weak label")
    ax[1].set_xlabel("PC1"); ax[1].legend(markerscale=4)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{tag}_space.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="basename, e.g. sband_short. Default: all")
    p.add_argument("--featdir", default=FEAT_DIR)
    p.add_argument("--outdir", default=os.path.join(HERE, "clusters"))
    p.add_argument("--mode", choices=["epochs", "single"], default="epochs",
                   help="epochs is the default: GLOBULAR's hyperparameters "
                        "are tuned for ~3000-point batches and degrade badly "
                        "on a single large pass (10 clusters vs 205 on "
                        "sband_short)")
    p.add_argument("--sample", type=int, default=60000,
                   help="single mode: cap on hits clustered (default 60000)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=3000, help="GLOBULAR used ~3000")
    p.add_argument("--min-cluster-size", type=int, default=4,
                   help="HDBSCAN n_pts; GLOBULAR used 4")
    p.add_argument("--min-samples", type=int, default=2,
                   help="HDBSCAN rho_pts; GLOBULAR used 2")
    p.add_argument("--epsilon", type=float, default=0.18,
                   help="HDBSCAN merge threshold; GLOBULAR used 0.18")
    p.add_argument("--scaling", choices=["robust", "quantile", "none"],
                   default="robust",
                   help="how to equalise feature contribution before the "
                        "Euclidean distance. 'none' is GLOBULAR's literal spec "
                        "and clusters poorly here -- see feature_matrix()")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    src = os.path.join(args.featdir,
                       f"{args.file}_features.parquet" if args.file
                       else "all_features.parquet")
    if not os.path.exists(src):
        sys.exit(f"Missing {src}\nRun:  python track_b_features.py")
    df = pd.read_parquet(src)
    df = df[df.feature_ok].reset_index(drop=True)
    tag = args.file or "all"
    print(f"{tag}: {len(df):,} hits with usable features")

    if args.mode == "epochs":
        labels, cols, _ = cluster_epochs(df, args)
    else:
        labels, cols = cluster_single(df, args)

    summarise(df, labels, args.outdir, tag)
    plot(df, labels, cols, args.outdir, tag, args.scaling)


if __name__ == "__main__":
    main()
