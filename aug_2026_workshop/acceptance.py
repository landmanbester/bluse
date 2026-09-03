#!/usr/bin/env python3
"""
Run the spec's acceptance criteria and dump JSON.

Exists so the numbers in clusters/acceptance-2026-09.md are reproducible rather
than transcribed by hand.

    uv run python aug_2026_workshop/acceptance.py > /tmp/acceptance.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from bluse import feature_io as FIO
from bluse import diagnostics as D
from bluse import features as F
from bluse import matching, metrics, paths
from bluse.bench import app

SEEDS = (0, 1, 2)


class DS:
    """Stand-in for bench.app.Dataset; cluster() uses .raw/.columns only."""

    def __init__(self, X, cols, df):
        self.raw, self.columns, self.df = X, cols, df


def load(name):
    path = FIO.find(name, paths.features_dir())
    if path is None:
        raise SystemExit(f"No feature matrix for {name}. Run: bluse-features")
    df = FIO.read(path)
    df = df[df.feature_ok].reset_index(drop=True)
    cols = [c + "_n" for c in F.all_columns()
            if c + "_n" in df.columns and not c.endswith("_saturated")
            and not c.startswith("f08_")]
    X = df[cols].to_numpy(dtype=np.float64)
    good = np.isfinite(X).all(axis=1)
    return DS(X[good], cols, df[good].reset_index(drop=True))


def main():
    out = {}
    ds = load("sband_short")
    kinds = {c + "_n": k for c, k in F.column_kinds().items()}

    rows = {r["col"]: r for r in D.audit(ds.raw, ds.columns, scaling="robust",
                                         kinds=kinds, min_samples=8)}
    out["diagnostics"] = {
        c: {k: rows[c][k] for k in ("n_distinct", "max_tie_fraction",
                                    "clip_frac", "share_global", "share_knn",
                                    "flags")}
        for c in rows
    }

    for method in ("eom", "leaf"):
        labels, X, _, trace = app.cluster(ds, ds.columns, "robust", "epochs",
                                          4, 8, 8, 3000, 0, method)
        q = metrics.quality(labels, ds.df)
        fam, minfo = matching.match(labels, X)
        fq = metrics.quality(fam, ds.df)

        def run_cl(seed, m=method):
            lab, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                       4, 8, 8, 3000, seed, m)
            return lab

        def run_fam(seed, m=method):
            lab, Xs, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                        4, 8, 8, 3000, seed, m)
            return matching.match(lab, Xs)[0]

        out[method] = {
            "quality": {k: v for k, v in q.items() if k != "narrow_frac_at"},
            "narrow_frac_at": {str(k): v for k, v in q["narrow_frac_at"].items()},
            "epochs": metrics.epoch_trace(trace, len(labels)),
            "stability_clusters": metrics.stability(run_cl, seeds=SEEDS),
            "stability_families": metrics.stability(run_fam, seeds=SEEDS),
            "matching": {k: v for k, v in minfo.items() if k != "nn_distances"},
            "family_quality": {"narrow_frac": fq["narrow_frac"],
                               "median_span_mhz": fq["median_span_mhz"],
                               "n_families": fq["n_clusters"]},
        }

    # P1-5: the shipped equalising mode, reported at matched family count in
    # the non-degenerate region. leaf only -- under eom equalisation collapses
    # family reproducibility, measured.
    Zeq = D.scale(ds.raw, "robust-equalised",
                  kinds={c + "_n": k for c, k in F.column_kinds().items()},
                  columns=list(ds.columns))
    eq_ds = DS(Zeq, ds.columns, ds.df)
    labels, X, _, _ = app.cluster(eq_ds, ds.columns, "none", "epochs",
                                  4, 8, 8, 3000, 0, "leaf")

    def run_fam_eq(seed):
        lab, Xs, _, _ = app.cluster(eq_ds, ds.columns, "none", "epochs",
                                    4, 8, 8, 3000, seed, "leaf")
        return matching.match(lab, Xs, n_families=36)[0]

    out["leaf_equalised"] = {
        "n_clusters": int(len(np.unique(labels[labels >= 0]))),
        "stability_families_at36": metrics.stability(run_fam_eq, seeds=SEEDS),
        "at_families": {},
    }
    for n in (16, 24, 36, 44):
        fam = matching.match(labels, X, n_families=n)[0]
        q = metrics.quality(fam, ds.df)
        out["leaf_equalised"]["at_families"][str(n)] = {
            "narrow_frac": q["narrow_frac"],
            "median_span_mhz": q["median_span_mhz"],
        }

    # The provisional-taxonomy hedge: how much does the deferred scaling work
    # plausibly move the families? Drop the two columns that dominate the
    # global distance share and see what changes.
    keep = [c for c in ds.columns if not c.startswith(("x03_", "f07_"))]
    labels, X, _, _ = app.cluster(ds, keep, "robust", "epochs",
                                  4, 8, 8, 3000, 0, "leaf")
    fam, minfo = matching.match(labels, X)
    fq = metrics.quality(fam, ds.df)
    out["leaf_without_x03_f07"] = {
        "n_clusters": minfo["n_clusters"],
        "n_families": minfo["n_families"],
        "narrow_frac": fq["narrow_frac"],
        "median_span_mhz": fq["median_span_mhz"],
    }

    json.dump(out, sys.stdout, indent=2, default=float)


if __name__ == "__main__":
    main()
