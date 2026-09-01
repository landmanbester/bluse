#!/usr/bin/env python3
"""
track_a_filter.py -- the classical post-processing baseline for BLUSE hits.

Reproduces the standard technosignature filtering chain (Tremblay et al. 2026,
K2-18b with VLA + MeerKAT) over the workshop HDF5 stamp files. Reads metadata
only -- the stamp cubes are never touched, so this runs over all 2M hits in
minutes.

DESIGN: nothing is deleted. Each cut adds a boolean `flag_*` column and the
final `pass_all` is the AND of their negations. That way you can inspect what
each cut did, disagree with any of it, and re-decide without re-running.

    bluse-track-a                                   # every file in data/
    bluse-track-a data/sband_short.h5
    bluse-track-a data/sband_short.h5 --derive-mask masks/sband.csv
    bluse-track-a data/sband_short.h5 --incoherent-power incoh.csv

Outputs per input file, into --outdir (default <workspace>/catalogues):
    <name>_cat.parquet   full catalogue: metadata + flags + diagnostics
    <name>_cutflow.csv   the cut-flow table
    <name>_survivors.csv survivors only, human-readable, sorted by SNR

The cuts, in order:
    1. known-RFI frequency masks          (rfi_masks.py)
    2. zero drift rate                    -> local RFI
    3. SNR window                         -> below: false positives
                                             above: instrumental artefacts
    4. multi-beam spatial coincidence     -> field-wide == not on sky
    5. coherent/incoherent power ratio    -> only if incoherentPower available
    6. cross-epoch persistence            -> same freq+drift on many days
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

from . import paths
from .paths import resolve_files
from .rfi_masks import build_mask_table, load_empirical_mask

META_COLS = [
    "id", "index", "beam", "coarseChannel", "startChannel", "numChannels",
    "numTimesteps", "frequency", "driftRate", "driftSteps", "snr", "power",
    "incoherentPower", "ra", "dec", "fch1", "foff", "tsamp", "tstart",
]
STR_COLS = ["sourceName", "obsid"]

FLAG_ORDER = [
    ("flag_rfi_band",   "known-RFI frequency mask"),
    ("flag_zero_drift", "drift rate exactly zero"),
    ("flag_snr_low",    "SNR below window"),
    ("flag_snr_high",   "SNR above window (instrumental)"),
    ("flag_multibeam",  "multi-beam coincidence"),
    ("flag_incoherent", "coherent/incoherent ratio"),
    ("flag_repeat",     "cross-epoch persistence"),
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_catalogue(path) -> pd.DataFrame:
    """Read every scalar metadata column into a DataFrame. Stamps untouched."""
    with h5py.File(path, "r") as h:
        d = {c: h[c][:] for c in META_COLS if c in h}
        for c in STR_COLS:
            if c in h:
                d[c] = np.array([x.decode() if isinstance(x, bytes) else str(x)
                                 for x in h[c][:]])
    df = pd.DataFrame(d)
    df["row"] = np.arange(len(df))          # index back into the stamp cube
    return df


# ---------------------------------------------------------------------------
# cut 1: known-RFI frequency masks
# ---------------------------------------------------------------------------

def cut_rfi_bands(df, mask_table):
    """Flag hits whose frequency falls inside any masked range."""
    f = df["frequency"].to_numpy()
    flag = np.zeros(len(f), dtype=bool)
    which = np.full(len(f), "", dtype=object)
    for lo, hi, label, src in mask_table:
        if hi < f.min() or lo > f.max():
            continue                        # mask not relevant to this band
        m = (f >= lo) & (f <= hi)
        if m.any():
            newly = m & ~flag
            which[newly] = f"{label} [{src}]"
            flag |= m
    df["flag_rfi_band"] = flag
    df["rfi_band_label"] = which
    return df


# ---------------------------------------------------------------------------
# cut 2/3: drift and SNR
# ---------------------------------------------------------------------------

def cut_zero_drift(df):
    df["flag_zero_drift"] = df["driftRate"].to_numpy() == 0.0
    return df


def cut_snr_window(df, snr_min, snr_max):
    s = df["snr"].to_numpy()
    df["flag_snr_low"] = s < snr_min
    df["flag_snr_high"] = s > snr_max
    return df


# ---------------------------------------------------------------------------
# cut 4: multi-beam spatial coincidence
# ---------------------------------------------------------------------------

def beam_multiplicity(freq_mhz, beam, drift_steps, tol_hz=1.0, tol_steps=1):
    """
    For each hit, count the distinct beams carrying a matching hit.

    A match means |delta frequency| <= tol_hz AND |delta driftSteps| <= tol_steps,
    following Tremblay et al. 2026 (MeerKAT: +/-1 Hz, +/-1 drift step).

    A genuine sky signal is confined to one or a few coherent beams; local RFI
    illuminates the whole 64-beam field. With incoherentPower unavailable in
    these files, this is our strongest discriminant.

    Sorted sweep, O(n log n + n*w) for mean window width w.
    """
    f_hz = np.asarray(freq_mhz, dtype=np.float64) * 1e6
    beam = np.asarray(beam)
    ds = np.asarray(drift_steps)

    order = np.argsort(f_hz, kind="mergesort")
    fs, bs, dss = f_hz[order], beam[order], ds[order]

    lo = np.searchsorted(fs, fs - tol_hz, side="left")
    hi = np.searchsorted(fs, fs + tol_hz, side="right")

    out = np.empty(len(fs), dtype=np.int32)
    for i in range(len(fs)):
        a, b = lo[i], hi[i]
        win_b = bs[a:b]
        win_d = dss[a:b]
        keep = np.abs(win_d - dss[i]) <= tol_steps
        out[i] = np.unique(win_b[keep]).size if keep.any() else 1

    res = np.empty_like(out)
    res[order] = out
    return res


def cut_multibeam(df, tol_hz, tol_steps, max_beams, verbose=True):
    """Compute beam multiplicity within each observation, then flag."""
    nb = np.ones(len(df), dtype=np.int32)
    obs = df["obsid"].to_numpy()
    freq = df["frequency"].to_numpy()
    beam = df["beam"].to_numpy()
    ds = df["driftSteps"].to_numpy()

    uniq = np.unique(obs)
    t0 = time.time()
    for k, o in enumerate(uniq, 1):
        m = obs == o
        nb[m] = beam_multiplicity(freq[m], beam[m], ds[m], tol_hz, tol_steps)
        if verbose and (k % 25 == 0 or k == len(uniq)):
            print(f"      ...{k}/{len(uniq)} observations "
                  f"({time.time() - t0:.0f}s)", end="\r", flush=True)
    if verbose:
        print(" " * 70, end="\r")

    df["n_beams"] = nb

    # Beams formed in each observation. BLUSE assigns one coherent beam per
    # catalogue target in the primary field of view and fills them contiguously
    # from 0, so a sparse patch of sky forms fewer than 64 beams -- we see as
    # few as 20 at high galactic latitude. The coincidence denominator therefore
    # varies, and `beam_frac` is the denominator-aware version of `n_beams`.
    #
    # We keep the absolute threshold: switching the cut to a fixed fraction
    # (6.25%, i.e. 4/64) changes the survivor count across all seven files by
    # 7 hits out of 7,143, all of them in the pre-filtered mk_sample_hits. Not
    # worth a knob. The columns are recorded because Tracks B/E may want them.
    formed = df.groupby("obsid")["beam"].transform("max") + 1
    df["n_beams_formed"] = formed.astype(np.int32)
    df["beam_frac"] = nb / formed.to_numpy()

    df["flag_multibeam"] = nb > max_beams
    return df


# ---------------------------------------------------------------------------
# cut 5: coherent / incoherent power ratio
# ---------------------------------------------------------------------------

def cut_incoherent(df, n_ants, verbose=True):
    """
    Flag hits violating SNR_coherent <= sqrt(N_ant) * SNR_incoherent.

    A signal genuinely localised on the sky gains coherently; interference
    entering through the sidelobes does not. Tremblay et al. 2026 use exactly
    this test.

    The BLUSE team have confirmed incoherentPower was never measured for this
    dataset, so this cut is inert here and will stay that way -- it is not
    waiting on a delivery. It remains wired to --incoherent-power so the same
    code works on any future dataset that does carry the incoherent beam.
    """
    ip = df["incoherentPower"].to_numpy(dtype=np.float64)
    have = np.isfinite(ip) & (ip > 0)
    if not have.any():
        if verbose:
            print("      incoherentPower absent -- cut skipped (never "
                  "measured for this dataset; see AGENTS.md)")
        df["flag_incoherent"] = False
        df["coh_ratio"] = np.nan
        return df

    p = df["power"].to_numpy(dtype=np.float64)
    ratio = np.full(len(df), np.nan)
    ratio[have] = p[have] / ip[have]
    df["coh_ratio"] = ratio
    # Ratio above sqrt(N) is impossible for a sky signal -> local interference.
    df["flag_incoherent"] = have & (ratio > np.sqrt(n_ants))
    if verbose:
        print(f"      incoherent test applied to {int(have.sum()):,} hits "
              f"(threshold sqrt({n_ants}) = {np.sqrt(n_ants):.2f})")
    return df


def apply_external_incoherent(df, path, verbose=True):
    """
    Merge externally supplied incoherentPower values.

    The file must be CSV or parquet containing `incoherentPower` plus a key.
    Keys tried in order: ('obsid','beam','frequency'), ('obsid','id'), ('id',).
    Frequency matching is exact -- round-tripped float64 from the same source
    should be fine; if not, pre-round both sides.
    """
    ext = pd.read_parquet(path) if path.endswith(".parquet") \
        else pd.read_csv(path)
    if "incoherentPower" not in ext.columns:
        sys.exit(f"{path}: no 'incoherentPower' column "
                 f"(found: {list(ext.columns)})")

    for key in (["obsid", "beam", "frequency"], ["obsid", "id"], ["id"]):
        if all(k in ext.columns for k in key) and all(k in df.columns for k in key):
            merged = df.drop(columns=["incoherentPower"]).merge(
                ext[key + ["incoherentPower"]], on=key, how="left")
            n = int(merged["incoherentPower"].notna().sum())
            if verbose:
                print(f"      matched {n:,}/{len(df):,} hits on {key}")
            merged["incoherentPower"] = merged["incoherentPower"].fillna(0.0)
            return merged
    sys.exit(f"{path}: no usable key. Need one of "
             f"(obsid,beam,frequency) / (obsid,id) / (id)")


# ---------------------------------------------------------------------------
# cut 6: cross-epoch persistence
# ---------------------------------------------------------------------------

def cut_repeat(df, tol_hz, min_obs):
    """
    Flag frequencies recurring across many separate observations.

    A transmitter on another world drifts differently on different days as the
    geometry changes; a terrestrial or satellite emitter keeps turning up at the
    same frequency. Binning on frequency alone is deliberately blunt -- we want
    persistence, not an exact match.
    """
    f_hz = df["frequency"].to_numpy() * 1e6
    binned = np.round(f_hz / tol_hz).astype(np.int64)
    tmp = pd.DataFrame({"bin": binned, "obsid": df["obsid"].to_numpy()})
    n_obs = tmp.groupby("bin")["obsid"].transform("nunique").to_numpy()
    df["n_obs_at_freq"] = n_obs
    df["flag_repeat"] = n_obs >= min_obs
    return df


# ---------------------------------------------------------------------------
# empirical mask derivation
# ---------------------------------------------------------------------------

def derive_mask(df, out_path, bin_hz, min_beams, min_obs):
    """
    Build an RFI mask from the data themselves.

    Any frequency bin seen in many beams AND many observations is interference,
    whatever the published tables say. This is the "empirically derived RFI
    mask" of Tremblay et al. 2026, and for our S band -- barely covered by the
    SARAO table -- it matters more than the documented masks.
    """
    f_hz = df["frequency"].to_numpy() * 1e6
    b = np.round(f_hz / bin_hz).astype(np.int64)
    g = pd.DataFrame({"bin": b,
                      "beam": df["beam"].to_numpy(),
                      "obsid": df["obsid"].to_numpy()}).groupby("bin")
    agg = g.agg(n_beams=("beam", "nunique"),
                n_obs=("obsid", "nunique"),
                n_hits=("beam", "size")).reset_index()
    bad = agg[(agg.n_beams >= min_beams) & (agg.n_obs >= min_obs)].copy()
    bad["f_lo_mhz"] = (bad["bin"] * bin_hz - bin_hz / 2) / 1e6
    bad["f_hi_mhz"] = (bad["bin"] * bin_hz + bin_hz / 2) / 1e6
    bad["label"] = ("empirical: " + bad.n_beams.astype(str) + " beams, "
                    + bad.n_obs.astype(str) + " obs")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    bad[["f_lo_mhz", "f_hi_mhz", "label", "n_beams", "n_obs", "n_hits"]] \
        .sort_values("f_lo_mhz").to_csv(out_path, index=False)
    print(f"      derived {len(bad):,} masked bins covering "
          f"{int(bad.n_hits.sum()):,} hits -> {out_path}")
    return bad


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def cutflow(df, name):
    """
    Two views of every cut:
      marginal   -- how many hits this cut flags, on its own
      sequential -- how many survive after applying cuts in order

    They differ a lot when cuts overlap, and the difference is informative:
    a cut with a large marginal but tiny sequential contribution is redundant.
    """
    n = len(df)
    rows = []
    alive = np.ones(n, dtype=bool)
    for col, desc in FLAG_ORDER:
        if col not in df.columns:
            continue
        f = df[col].to_numpy()
        marginal = int(f.sum())
        alive_before = int(alive.sum())
        alive &= ~f
        alive_after = int(alive.sum())
        rows.append({
            "cut": desc,
            "flagged_alone": marginal,
            "pct_alone": 100 * marginal / n,
            "removed_here": alive_before - alive_after,
            "remaining": alive_after,
            "pct_remaining": 100 * alive_after / n,
        })
    out = pd.DataFrame(rows)

    print(f"\n  cut-flow for {name}  (start: {n:,} hits)")
    print(f"  {'cut':<34} {'alone':>10} {'removes':>10} "
          f"{'remain':>10} {'remain%':>9}")
    print("  " + "-" * 76)
    for _, r in out.iterrows():
        print(f"  {r['cut']:<34} {r['flagged_alone']:>10,} "
              f"{r['removed_here']:>10,} {r['remaining']:>10,} "
              f"{r['pct_remaining']:>8.3f}%")
    print("  " + "-" * 76)
    print(f"  {'SURVIVORS':<34} {'':>10} {'':>10} "
          f"{int(alive.sum()):>10,} {100 * alive.mean():>8.3f}%")
    return out, alive


# ---------------------------------------------------------------------------

def process(path, args, mask_table):
    name = os.path.basename(path).replace(".h5", "")
    print(f"\n{'=' * 80}\n{name}")
    t0 = time.time()

    df = load_catalogue(path)
    print(f"  loaded {len(df):,} hits  ({time.time() - t0:.1f}s)")

    if args.incoherent_power:
        print("  [external] merging incoherentPower")
        df = apply_external_incoherent(df, args.incoherent_power)

    if args.derive_mask:
        print("  [0] deriving empirical mask")
        derive_mask(df, args.derive_mask, args.mask_bin_hz,
                    args.mask_min_beams, args.mask_min_obs)
        mask_table = mask_table + load_empirical_mask(args.derive_mask)

    print("  [1] known-RFI frequency masks")
    df = cut_rfi_bands(df, mask_table)
    print("  [2] zero drift rate")
    df = cut_zero_drift(df)
    print(f"  [3] SNR window [{args.snr_min}, {args.snr_max:g}]")
    df = cut_snr_window(df, args.snr_min, args.snr_max)
    print(f"  [4] multi-beam coincidence "
          f"(+/-{args.tol_hz:g} Hz, +/-{args.tol_steps} steps, "
          f"max {args.max_beams} beams)")
    df = cut_multibeam(df, args.tol_hz, args.tol_steps, args.max_beams)
    print("  [5] coherent/incoherent ratio")
    df = cut_incoherent(df, args.n_ants)
    print(f"  [6] cross-epoch persistence (>= {args.min_obs} observations)")
    df = cut_repeat(df, args.repeat_tol_hz, args.min_obs)

    # derived quantities that every downstream track will want
    df["log_snr"] = np.log10(np.clip(df["snr"], 1e-12, None))
    df["log_power"] = np.log10(np.clip(df["power"], 1e-12, None))
    df["abs_drift"] = df["driftRate"].abs()

    flow, alive = cutflow(df, name)
    df["pass_all"] = alive

    os.makedirs(args.outdir, exist_ok=True)
    cat = os.path.join(args.outdir, f"{name}_cat.parquet")
    df.to_parquet(cat, index=False)
    flow.to_csv(os.path.join(args.outdir, f"{name}_cutflow.csv"), index=False)

    surv = df[df.pass_all].sort_values("snr", ascending=False)
    keep = ["row", "obsid", "sourceName", "beam", "frequency", "driftRate",
            "snr", "n_beams", "n_beams_formed", "beam_frac", "n_obs_at_freq",
            "ra", "dec"]
    surv_path = os.path.join(args.outdir, f"{name}_survivors.csv")
    surv[keep].to_csv(surv_path, index=False)

    print(f"\n  wrote {cat}")
    print(f"  wrote {surv_path}")
    if len(surv):
        print(f"\n  top survivors by SNR:")
        print(f"  {'row':>9} {'freq [MHz]':>14} {'SNR':>10} {'drift':>8} "
              f"{'beam':>5} {'nb':>4} source")
        for _, r in surv.head(args.show).iterrows():
            print(f"  {int(r['row']):>9} {r['frequency']:>14.6f} "
                  f"{r['snr']:>10.4g} {r['driftRate']:>+8.4f} "
                  f"{int(r['beam']):>5} {int(r['n_beams']):>4} "
                  f"{r['sourceName']}")
        print(f"\n  inspect them with:")
        print(f"    bluse-explore stamps {path} --sort snr")
    else:
        print("\n  no survivors.")
    print(f"\n  ({time.time() - t0:.1f}s total)")
    return df


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="HDF5 files (default: all in data/)")
    p.add_argument("--outdir", default=None,
                   help="default: <workspace>/catalogues")
    paths.add_workspace_arg(p)
    p.add_argument("--show", type=int, default=15, help="survivors to print")

    g = p.add_argument_group("cut parameters")
    g.add_argument("--snr-min", type=float, default=10.0,
                   help="below this is mostly false positives (default 10)")
    g.add_argument("--snr-max", type=float, default=1e6,
                   help="above this is mostly instrumental. Tremblay+2026 use "
                        "100; these data have a detached population at 1e7-1e8, "
                        "so the default is deliberately loose (default 1e6)")
    g.add_argument("--tol-hz", type=float, default=1.0,
                   help="multi-beam frequency tolerance (default 1 Hz)")
    g.add_argument("--tol-steps", type=int, default=1,
                   help="multi-beam drift-step tolerance (default 1)")
    g.add_argument("--max-beams", type=int, default=4,
                   help="reject hits seen in more beams than this (default 4)")
    g.add_argument("--n-ants", type=int, default=62,
                   help="antennas, for the sqrt(N) coherence test (default 62)")
    g.add_argument("--min-obs", type=int, default=5,
                   help="reject frequencies seen in >= this many observations")
    g.add_argument("--repeat-tol-hz", type=float, default=10.0,
                   help="binning for the persistence test (default 10 Hz)")

    g = p.add_argument_group("masks")
    g.add_argument("--no-itu", action="store_true",
                   help="use only SARAO-documented masks, drop the ITU guesses")
    g.add_argument("--dtv", action="store_true",
                   help="add the digital-TV comb. OFF by default: channels "
                        "21-68 tile contiguously over 466-858 MHz and mask the "
                        "whole UHF band. Use --derive-mask instead")
    g.add_argument("--derive-mask", metavar="CSV",
                   help="derive an empirical mask, write it here, and use it")
    g.add_argument("--mask-bin-hz", type=float, default=100.0)
    g.add_argument("--mask-min-beams", type=int, default=32)
    g.add_argument("--mask-min-obs", type=int, default=3)

    g = p.add_argument_group("external data")
    g.add_argument("--incoherent-power", metavar="FILE",
                   help="CSV/parquet with incoherentPower plus a join key")

    args = p.parse_args()
    paths.set_workspace(args.workspace)
    args.outdir = args.outdir or paths.catalogues_dir()
    print(paths.banner())
    mask_table = build_mask_table(include_itu=not args.no_itu,
                                  include_dtv=args.dtv)
    print(f"{len(mask_table)} RFI mask ranges loaded")

    for f in resolve_files(args.files):
        process(f, args, mask_table)


if __name__ == "__main__":
    main()
