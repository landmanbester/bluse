#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy", "pandas", "pyarrow", "scipy", "scikit-learn"]
# ///
"""
track_b_features.py -- extract the Track B feature matrix.

Reads the Track A catalogue for provenance and flags, streams the stamp cubes,
runs every registered feature in `features.py`, and writes a single parquet per
input file that Track E can train on directly.

    uv run track_b_features.py                       # all files
    uv run track_b_features.py data/sband_short.h5
    uv run track_b_features.py --sample 50000        # per-file subsample
    uv run track_b_features.py --list                # show the registry

OUTPUT -- designed for Track E
------------------------------
`features/<name>_features.parquet`, one row per hit, columns in four groups:

  provenance    file row obsid sourceName beam frequency driftRate snr
                power ra dec tstart numChannels n_beams n_beams_formed
                beam_frac n_obs_at_freq  + every Track A flag_* and pass_all
  features      f01..f13, x01..x03 (raw) and <col>_n (normalised)
  weak labels   weak_label, weak_label_reason, group_id
  quality       stamp_ok, feature_ok

`weak_label` is the free supervision Track E was proposed to exploit:

     1  RFI          seen in >= --rfi-beams distinct beams. A signal in most of
                     a 64-beam field is not on the sky. High confidence.
     0  spatially    seen in <= --clean-beams beams. LOW confidence: spatial
        confined     confinement is necessary but nowhere near sufficient for a
                     signal to be astrophysical, so these are better treated as
                     UNLABELLED than as negatives. Positive-unlabelled learning
                     is the honest framing; if you train a plain binary
                     classifier on 1 vs 0, say so.

                     Deliberately NOT conditioned on surviving Track A. Doing so
                     would (a) starve the class -- 24 rows in sband_short rather
                     than 1,089 -- and (b) make Track E partly circular, since
                     it would be rewarded for rediscovering Track A's frequency
                     masks instead of learning signal morphology. Use
                     --require-pass-all if you want the strict version anyway.
    -1  ambiguous    everything in between. Score these, do not train on them.

`group_id` is the observation. Split on it -- hits from one observation share a
pointing, an RFI environment and a calibration, so a random row split leaks
badly and will flatter any classifier.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CAT_DIR = os.path.join(HERE, "catalogues")

PROVENANCE = [
    "row", "obsid", "sourceName", "beam", "frequency", "driftRate", "snr",
    "power", "ra", "dec", "tstart", "numChannels", "coarseChannel",
    "n_beams", "n_beams_formed", "beam_frac", "n_obs_at_freq",
]


def resolve_files(paths):
    if not paths:
        found = sorted(glob(os.path.join(DATA_DIR, "*.h5")))
        if not found:
            sys.exit(f"No .h5 files in {DATA_DIR}")
        return found
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob(os.path.join(p, "*.h5"))))
        elif os.path.exists(p):
            out.append(p)
        elif os.path.exists(os.path.join(DATA_DIR, p)):
            out.append(os.path.join(DATA_DIR, p))
        else:
            sys.exit(f"Not found: {p}")
    return out


def load_track_a(name):
    """Track A gives us the flags, n_beams and beam_frac. It is a prerequisite."""
    p = os.path.join(CAT_DIR, f"{name}_cat.parquet")
    if not os.path.exists(p):
        sys.exit(f"Missing {p}\nRun:  python track_a_filter.py data/{name}.h5")
    return pd.read_parquet(p)


def assign_weak_labels(df, rfi_beams, clean_beams, require_pass_all=False):
    """
    Free supervision from the spatial filter (see module docstring).

    Uses beam_frac as well as n_beams so the threshold means the same thing in
    a 49-beam observation as in a 64-beam one.
    """
    nb = df["n_beams"].to_numpy()
    frac = df["beam_frac"].to_numpy()

    is_rfi = (nb >= rfi_beams) | (frac >= rfi_beams / 64.0)
    is_clean = nb <= clean_beams
    note = f"<={clean_beams} beams"
    if require_pass_all:
        is_clean &= df["pass_all"].to_numpy()
        note += " and survived Track A"

    label = np.full(len(df), -1, dtype=np.int8)
    reason = np.full(len(df), "ambiguous", dtype=object)
    label[is_clean] = 0
    reason[is_clean] = note
    label[is_rfi] = 1                                    # RFI wins any overlap
    reason[is_rfi] = f">={rfi_beams} beams"

    df["weak_label"] = label
    df["weak_label_reason"] = reason
    df["group_id"] = df["obsid"]
    return df


def extract(path, args):
    name = os.path.basename(path).replace(".h5", "")
    print(f"\n{'=' * 80}\n{name}")
    t0 = time.time()

    cat = load_track_a(name)
    if args.sample and args.sample < len(cat):
        cat = cat.sample(args.sample, random_state=args.seed)
        print(f"  subsampled to {len(cat):,} of {len(cat):,} hits")
    cat = cat.sort_values("row").reset_index(drop=True)
    rows = cat["row"].to_numpy()
    print(f"  {len(cat):,} hits")

    # --- metadata features: one vectorised pass over the whole table --------
    meta_out = {}
    for spec in F.REGISTRY.values():
        if spec.kind == "meta":
            meta_out.update(spec.func(cat))

    # --- stamp features: streamed in batches --------------------------------
    stamp_cols = F.all_columns("stamp")
    store = {c: np.full(len(cat), np.nan) for c in stamp_cols}
    stamp_ok = np.zeros(len(cat), dtype=bool)
    n_unreadable = 0

    with h5py.File(path, "r") as h:
        data = h["data"]

        readable, bad_blocks = scan_bad_regions(data, data.shape[0])
        if bad_blocks:
            keep_mask = readable[rows]
            n_skip = int((~keep_mask).sum())
            print(f"  corrupt HDF5 region: {bad_blocks} bad blocks, "
                  f"{n_skip:,} of {len(cat):,} hits have no readable stamp")
            n_unreadable += n_skip

        bs = args.batch
        for s in range(0, len(cat), bs):
            e = min(s + bs, len(cat))
            idx = rows[s:e]
            sub = cat.iloc[s:e]
            if bad_blocks:
                m = readable[idx]
                if not m.any():
                    continue
                if not m.all():
                    idx = idx[m]
                    sub = sub.iloc[m]
                    s_pre = np.arange(s, e)[m]
                else:
                    s_pre = np.arange(s, e)
            else:
                s_pre = np.arange(s, e)
            try:
                cube = data[idx[0]:idx[-1] + 1] if _contiguous(idx) else data[idx]
                if _contiguous(idx) and len(idx) != idx[-1] - idx[0] + 1:
                    cube = cube[idx - idx[0]]
            except Exception:
                # Corrupt HDF5 regions exist in this delivery (see AGENTS.md).
                # Retry row by row so we lose only the genuinely bad stamps.
                cube, keep = _read_one_by_one(data, idx)
                n_unreadable += len(idx) - len(keep)
                if len(keep) == 0:
                    continue
                sub = sub.iloc[keep]
                s_idx = s_pre[keep]
            else:
                s_idx = s_pre

            batch = F.prepare_batch(cube, sub, crop_channels=args.crop)
            for spec in F.REGISTRY.values():
                if spec.kind != "stamp":
                    continue
                for col, val in spec.func(batch).items():
                    store[col][s_idx] = np.asarray(val, dtype=np.float64)
            stamp_ok[s_idx] = True

            if args.progress and (s // bs) % 20 == 0:
                done = e / len(cat)
                print(f"      {done:6.1%}  ({time.time() - t0:5.0f}s)",
                      end="\r", flush=True)
    if args.progress:
        print(" " * 60, end="\r")

    out = cat[[c for c in PROVENANCE if c in cat.columns]].copy()
    out.insert(0, "file", name)
    for c in cat.columns:
        if c.startswith("flag_") or c == "pass_all":
            out[c] = cat[c].to_numpy()
    for c, v in meta_out.items():
        out[c] = v
    for c in stamp_cols:
        out[c] = store[c]

    out["stamp_ok"] = stamp_ok
    feat_cols = F.all_columns()
    numeric = [c for c in feat_cols if out[c].dtype.kind == "f"]
    out["feature_ok"] = stamp_ok & np.isfinite(out[numeric].to_numpy()).all(axis=1)

    out, _ = F.normalise(out)
    out = assign_weak_labels(out, args.rfi_beams, args.clean_beams,
                             args.require_pass_all)

    if n_unreadable:
        print(f"  WARNING {n_unreadable:,} stamps unreadable "
              f"({100 * n_unreadable / len(cat):.2f}%) -- corrupt HDF5 region. "
              f"Rows kept with stamp_ok=False.")

    os.makedirs(args.outdir, exist_ok=True)
    dest = os.path.join(args.outdir, f"{name}_features.parquet")
    out.to_parquet(dest, index=False)

    ok = int(out.feature_ok.sum())
    print(f"  usable feature rows: {ok:,} / {len(out):,} ({100 * ok / len(out):.1f}%)")
    vc = out.weak_label.value_counts()
    print(f"  weak labels: RFI={int(vc.get(1, 0)):,}  "
          f"confined={int(vc.get(0, 0)):,}  "
          f"ambiguous={int(vc.get(-1, 0)):,}")
    print(f"  wrote {dest}  ({time.time() - t0:.0f}s)")
    return out


def scan_bad_regions(data, n, probe=2000):
    """
    Find unreadable regions of the stamp cube cheaply.

    This delivery contains corrupt HDF5 regions -- lband_short.h5 rows
    338,000-742,000 and uhf_long.h5 rows 264,000-270,000 (see AGENTS.md). Hitting
    one raises OSError mid-read. Probing a single row per block costs a few
    hundred reads instead of re-reading the file, and the per-row fallback in
    the main loop catches anything a block probe misses.

    Returns a boolean mask, True where the row is expected to be readable.
    """
    ok = np.ones(n, dtype=bool)
    bad_blocks = 0
    for s in range(0, n, probe):
        e = min(s + probe, n)
        try:
            _ = data[s]
            _ = data[e - 1]
        except Exception:
            ok[s:e] = False
            bad_blocks += 1
    return ok, bad_blocks


def _contiguous(idx):
    return len(idx) > 0 and idx[-1] - idx[0] + 1 <= 4 * len(idx)


def _read_one_by_one(data, idx):
    got, keep = [], []
    for k, i in enumerate(idx):
        try:
            got.append(data[i])
            keep.append(k)
        except Exception:
            pass
    if not got:
        return None, []
    return np.stack(got), np.array(keep)


def summarise(frames, outdir):
    """Print feature health, then write the combined table Track E will read."""
    all_df = pd.concat(frames, ignore_index=True)
    cols = F.all_columns()
    print(f"\n{'=' * 80}\nCOMBINED  {len(all_df):,} hits from {len(frames)} files\n")
    print(f"  {'feature':<26} {'finite%':>8} {'median':>12} {'p1':>12} {'p99':>12}")
    print("  " + "-" * 74)
    for c in cols:
        v = all_df[c].to_numpy(dtype=np.float64)
        f = np.isfinite(v)
        if f.sum() == 0:
            print(f"  {c:<26} {'0.0':>8}")
            continue
        p1, p50, p99 = np.percentile(v[f], [1, 50, 99])
        print(f"  {c:<26} {100 * f.mean():>7.1f}% {p50:>12.4g} "
              f"{p1:>12.4g} {p99:>12.4g}")

    vc = all_df.weak_label.value_counts()
    n = len(all_df)
    print(f"\n  weak labels   RFI {int(vc.get(1, 0)):>9,} ({100*vc.get(1,0)/n:5.2f}%)"
          f"   confined {int(vc.get(0, 0)):>7,} ({100*vc.get(0,0)/n:5.2f}%)"
          f"   ambiguous {int(vc.get(-1, 0)):>9,} ({100*vc.get(-1,0)/n:5.2f}%)")
    print(f"  groups (observations): {all_df.group_id.nunique():,}")

    dest = os.path.join(outdir, "all_features.parquet")
    all_df.to_parquet(dest, index=False)
    print(f"\n  wrote {dest}")
    return all_df


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*")
    p.add_argument("--outdir", default=os.path.join(HERE, "features"))
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--sample", type=int, help="subsample each file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", type=int, default=60,
                   help="channels kept about the stamp centre. Default 60 is "
                        "the global minimum numChannels, so every hit in every "
                        "file gets an identical window and no feature can "
                        "accidentally measure window width instead of signal")
    p.add_argument("--rfi-beams", type=int, default=32,
                   help="weak label 1 at or above this beam count (default 32)")
    p.add_argument("--clean-beams", type=int, default=2,
                   help="weak label 0 at or below this beam count (default 2)")
    p.add_argument("--require-pass-all", action="store_true",
                   help="additionally require label-0 rows to survive Track A. "
                        "Starves the class and makes Track E partly circular; "
                        "off by default")
    p.add_argument("--no-progress", dest="progress", action="store_false")
    p.add_argument("--list", action="store_true", help="print the registry and exit")
    args = p.parse_args()

    if args.list:
        F.describe()
        return

    frames = [extract(f, args) for f in resolve_files(args.files)]
    if frames:
        summarise(frames, args.outdir)


if __name__ == "__main__":
    main()
