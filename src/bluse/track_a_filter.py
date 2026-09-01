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

The cuts, in order (Tremblay et al. 2026 section numbers in brackets):
    1. known-RFI frequency masks          [3.1]  (rfi_masks.py)
    2. zero drift rate                    [3.2]  -> local RFI
    3. drift rate above the plausible max [3.2]  -> not a bound companion
    4. SNR window                         [3.3]  -> below: false positives
                                                    above: instrumental
    5. multi-beam spatial coincidence     [3.4]  -> field-wide == not on sky
    6. coherent/incoherent power ratio    [3.7]  -> only if incoherentPower
    7. cross-epoch persistence            [3.6]  -> same freq+drift on many days

Their [3.5] primary/secondary transit filter has no analogue here: it needs a
transit-timed planet, and our targets are catalogued stars.

See papers/Tremblay-technical-reference.md for the full specification, the
places we diverge and why, and the inconsistencies inside the paper itself --
in particular that its section 4 does not apply the drift limits its section
3.2 prescribes.
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
    ("flag_drift_high", "drift rate above plausible max"),
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


# Hz/s per MHz. Tremblay et al. 2026 section 3.2 quotes three anchor points from
# Li et al. (2022, 2023) covering 99% of plausible signals: ~0.4 Hz/s below
# 1.5 GHz, 1.879 Hz/s at 4.5 GHz, 4.177 Hz/s at 10 GHz. Doppler drift scales
# linearly with frequency, and indeed 0.4/1000, 1.879/4500 and 4.177/10000 all
# land within 5% of 4.18e-4 -- so one coefficient reproduces all three numbers.
MAX_DRIFT_COEFF = 4.18e-4


def cut_max_drift(df, coeff):
    """
    Flag hits drifting faster than a bound companion plausibly could.

    TWO HONEST CAVEATS, because this coefficient is easy to over-trust:

    1. It is K2-18-specific. It bounds Earth's rotation -- about 1.1e-4 Hz/s per
       MHz, and universal -- PLUS K2-18b's own orbital acceleration. Our targets
       are arbitrary Gaia sources whose companions we know nothing about, so
       read this as a generous envelope, not a per-target limit. Pass
       --max-drift-coeff to change it or --no-max-drift to switch it off.

    2. The paper does not apply its own prescription. Section 3.2 gives the
       frequency-scaled limits above; section 4 then applies a blanket
       +/-1.9 Hz/s -- the 4.5 GHz value -- to L, S and C band alike, and only
       X band follows the rule. See papers/Tremblay-technical-reference.md
       section 7.1.

    OFF BY DEFAULT since 2026-09. Myburgh et al. 2026 -- same group, blind Gaia
    targets, i.e. our situation rather than K2-18's -- deliberately search
    +/-50 Hz/s "as many of our targets are toward unknown planetary systems".
    Ours are too. And the limit bites inside the range seticore actually
    searched: on lband_short_clean it lands at 0.358-0.402 Hz/s against an
    observed maximum of 0.4203, with 4,257 hits sitting at the extreme
    driftSteps -- exactly where a genuinely fast-drifting signal would be. In a
    blind survey a false negative costs more than a human looking at one more
    waterfall, so the cut stays available and stays off.
    """
    if not coeff:
        df["flag_drift_high"] = np.zeros(len(df), dtype=bool)
        return df
    limit = coeff * df["frequency"].to_numpy()          # frequency is MHz
    df["max_drift_hz_s"] = limit
    df["flag_drift_high"] = df["driftRate"].abs().to_numpy() > limit
    return df


def cut_snr_window(df, snr_min, snr_max, snr_min_short=None, short_nt=16):
    """
    SNR window, with a raised floor for short integrations.

    Myburgh et al. 2026 filter 3: seticore's noise estimate degrades when there
    are few time samples to average, so a short hit needs more SNR to be
    believable. They require SNR > 15 below 16 timesteps and SNR > 10 above.
    Tremblay et al. 2026 sec 3.3 measured the same effect from the other end --
    ~80% of 8-sigma detections with few time samples were false positives -- and
    Czech et al. 2026 sec 6 separately calls beams shorter than 150 s unviable.
    Three independent statements of one problem.

    It matters here because uhf_short is 14-15 timesteps throughout: every one
    of its hits is in the regime all three papers warn about.
    """
    s = df["snr"].to_numpy()
    lo = np.full(len(df), float(snr_min))
    if snr_min_short and "numTimesteps" in df.columns:
        short = df["numTimesteps"].to_numpy() < short_nt
        lo[short] = float(snr_min_short)
    df["snr_floor"] = lo
    df["flag_snr_low"] = s < lo
    df["flag_snr_high"] = s > snr_max
    return df


# ---------------------------------------------------------------------------
# cut 4: multi-beam spatial coincidence
# ---------------------------------------------------------------------------

def beam_multiplicity(freq_mhz, beam, drift_steps, tol_hz=1.0, tol_steps=1):
    """
    For each hit, count the distinct beams carrying a matching hit.

    A match means |delta frequency| <= tol_hz AND |delta driftSteps| <= tol_steps,
    following Tremblay et al. 2026 section 3.4.

    THE RULE IS +/-1 FINE CHANNEL, NOT +/-1 Hz. They quote 1 Hz for MeerKAT
    because their channel was 1 Hz; ours are 1.013 (UHF), 1.594 (L) and 1.630
    (S). Hardcoding 1.0 matched ~37% too tightly in L and S band, under-counting
    beam multiplicity and letting multi-beam RFI through. tol_hz now defaults to
    the file's own |foff|; --tol-hz overrides it.

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

def cut_incoherent(df, n_ants, tol=2.0, verbose=True):
    """
    Flag hits whose coherent/incoherent ratio is inconsistent with a sky source.

    THE TEST IS TWO-SIDED, and an earlier version of this function got that
    wrong. Coherent summation of N antennas gains signal power as N^2 against
    noise as N; the incoherent sum gains neither. So for a source actually at
    the beam centre the ratio sits AT roughly sqrt(N) -- and interference,
    which does not phase up, falls well SHORT of it. Flagging only ratio >
    sqrt(N) therefore caught the physically impossible tail and missed the
    entire real RFI population, which is the half worth catching.

    The two source papers state it in opposite directions, which is why this
    was easy to get wrong:

      Tremblay et al. 2026 sec 3.7   "a signal of the same origin ... will have
                                     SNR_coh <= sqrt(N) * SNR_incoh"
                                     -- i.e. an inequality a REAL signal obeys
      Myburgh et al. 2026 filter 7   "if coherent_S/N <= sqrt(N) *
                                     incoherent_S/N then the signal is most
                                     likely RFI"
                                     -- i.e. the same inequality means RFI

    Read literally they contradict each other. The physics, plus Tremblay's own
    section 4.1 (they look for hits showing "an expected power ratio (4.69)",
    which is sqrt(22)) and Myburgh's Figure 5 caption ("coherent power not
    sufficiently greater than its incoherent power, marking it as non-localized
    interference"), agree that the discriminant is the ratio being NEAR
    sqrt(N). Both tails are interference. That is what we implement; `tol` is
    the factor of slack either side and is ours, not from either paper.

    CAVEAT: both papers state the relation in SNR. Our columns are `power` and
    `incoherentPower`, so we are applying an SNR relation to a power ratio.
    That substitution is unverified and must be checked against real data
    before this cut is trusted -- see papers/Myburgh-technical-reference.md.

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
    expect = np.sqrt(n_ants)
    too_high = ratio > expect * tol       # impossible for a sky signal
    too_low = ratio < expect / tol        # never phased up -> not localised
    df["flag_incoherent"] = have & (too_high | too_low)
    if verbose:
        print(f"      incoherent test applied to {int(have.sum()):,} hits "
              f"(expect ~sqrt({n_ants}) = {expect:.2f}, "
              f"accepting {expect / tol:.2f}-{expect * tol:.2f}); "
              f"{int((have & too_low).sum()):,} below, "
              f"{int((have & too_high).sum()):,} above")
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

def cut_repeat(df, tol_hz, min_obs, tol_steps=1):
    """
    Flag signals recurring at the same frequency AND drift across observations.

    Tremblay et al. 2026 section 3.6, after Li et al. (2022): a transmitter that
    moves with its planet, observed from a rotating Earth, traces a sinusoid in
    frequency, so it cannot reappear at an identical frequency and drift rate on
    a different day. One that does is terrestrial.

    ADAPTED, deliberately. They track a single target across a handful of epochs
    and treat ANY repeat as RFI. We have 100-143 observations per file, where
    "any repeat" would flag nearly everything, so we count how many distinct
    observations a (frequency, drift) cell appears in and threshold at min_obs.
    That threshold is ours, not theirs.

    Two counts are recorded and they are not interchangeable:

      n_obs_at_freq        frequency only. UNCHANGED -- Track B's provenance
                           columns and weak labels are built on it.
      n_obs_at_freq_drift  frequency AND drift, the faithful version. This is
                           what flag_repeat now uses, so the cut is strictly
                           more conservative than it was: a signal has to
                           persist in *both* to be called interference.
    """
    f_hz = df["frequency"].to_numpy() * 1e6
    fbin = np.round(f_hz / tol_hz).astype(np.int64)
    obs = df["obsid"].to_numpy()

    n_freq = pd.DataFrame({"bin": fbin, "obsid": obs}) \
        .groupby("bin")["obsid"].transform("nunique").to_numpy()
    df["n_obs_at_freq"] = n_freq

    # Drift binned at the search's own resolution, mirroring the one-drift-step
    # tolerance of section 3.4. driftSteps is already integer-valued.
    dbin = np.round(df["driftSteps"].to_numpy() / max(tol_steps, 1)).astype(np.int64)
    n_both = pd.DataFrame({"f": fbin, "d": dbin, "obsid": obs}) \
        .groupby(["f", "d"])["obsid"].transform("nunique").to_numpy()
    df["n_obs_at_freq_drift"] = n_both

    df["flag_repeat"] = n_both >= min_obs
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

    # +/-1 fine channel, per Tremblay et al. 2026 section 3.4 -- so the default
    # is the file's own channel width, not a constant. See cut_multibeam.
    if args.tol_hz is not None:
        tol_hz, tol_src = args.tol_hz, "--tol-hz"
    elif "foff" in df.columns and len(df):
        tol_hz, tol_src = abs(float(df["foff"].iloc[0])) * 1e6, "1 fine channel"
    else:
        tol_hz, tol_src = 1.0, "fallback"

    print("  [1] known-RFI frequency masks")
    df = cut_rfi_bands(df, mask_table)
    print("  [2] zero drift rate")
    df = cut_zero_drift(df)
    if args.max_drift_coeff:
        lo = args.max_drift_coeff * df["frequency"].min()
        hi = args.max_drift_coeff * df["frequency"].max()
        print(f"  [3] max drift rate ({args.max_drift_coeff:g} Hz/s per MHz "
              f"-> {lo:.3f}-{hi:.3f} Hz/s over this band)")
    else:
        print("  [3] max drift rate (disabled)")
    df = cut_max_drift(df, args.max_drift_coeff)
    nshort = int((df["numTimesteps"] < args.short_timesteps).sum())
    print(f"  [4] SNR window [{args.snr_min}, {args.snr_max:g}]"
          + (f", floor {args.snr_min_short} for the {nshort:,} hits under "
             f"{args.short_timesteps} timesteps" if args.snr_min_short and nshort
             else ""))
    df = cut_snr_window(df, args.snr_min, args.snr_max,
                        args.snr_min_short, args.short_timesteps)
    print(f"  [5] multi-beam coincidence "
          f"(+/-{tol_hz:g} Hz [{tol_src}], +/-{args.tol_steps} steps, "
          f"max {args.max_beams} beams)")
    df = cut_multibeam(df, tol_hz, args.tol_steps, args.max_beams)
    print("  [6] coherent/incoherent ratio")
    df = cut_incoherent(df, args.n_ants, args.coh_ratio_tol)
    print(f"  [7] cross-epoch persistence (>= {args.min_obs} observations "
          f"at matching frequency AND drift)")
    df = cut_repeat(df, args.repeat_tol_hz, args.min_obs, args.repeat_tol_steps)

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
            "n_obs_at_freq_drift", "ra", "dec"]
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
    g.add_argument("--snr-min-short", type=float, default=15.0,
                   help="raised SNR floor for short integrations, where "
                        "seticore's noise estimate is least reliable "
                        "(default 15, per Myburgh+2026 filter 3). Set 0 to "
                        "use --snr-min everywhere")
    g.add_argument("--short-timesteps", type=int, default=16,
                   help="a hit with fewer timesteps than this is 'short' "
                        "(default 16). uhf_short is 14-15 throughout")
    g.add_argument("--snr-max", type=float, default=1e6,
                   help="above this is mostly instrumental. Tremblay+2026 use "
                        "100; these data have a detached population at 1e7-1e8, "
                        "so the default is deliberately loose (default 1e6)")
    g.add_argument("--tol-hz", type=float, default=None,
                   help="multi-beam frequency tolerance. Default: the file's "
                        "own fine-channel width (1.013-1.630 Hz here), which "
                        "is the +/-1 fine channel of Tremblay+2026 sec 3.4 -- "
                        "NOT the 1 Hz they quote for their 1 Hz channels")
    g.add_argument("--tol-steps", type=int, default=1,
                   help="multi-beam drift-step tolerance (default 1)")
    g.add_argument("--max-drift-coeff", type=float, default=0.0,
                   help="max plausible |drift| in Hz/s per MHz of observing "
                        "frequency. OFF by default -- ours is a blind survey "
                        "of unknown systems, and Myburgh+2026 deliberately "
                        "search +/-50 Hz/s for exactly that reason. Pass "
                        f"{MAX_DRIFT_COEFF:g} to reproduce Tremblay+2026 sec "
                        "3.2, whose coefficient is K2-18-specific. See "
                        "cut_max_drift()")
    g.add_argument("--max-beams", type=int, default=4,
                   help="reject hits seen in more beams than this (default 4)")
    g.add_argument("--n-ants", type=int, default=62,
                   help="antennas, for the sqrt(N) coherence test (default 62)")
    g.add_argument("--coh-ratio-tol", type=float, default=2.0,
                   help="slack factor either side of sqrt(N) for the "
                        "coherent/incoherent test (default 2). The test is "
                        "two-sided: interference sits well below sqrt(N) as "
                        "well as impossibly above it")
    g.add_argument("--min-obs", type=int, default=5,
                   help="reject frequencies seen in >= this many observations")
    g.add_argument("--repeat-tol-hz", type=float, default=10.0,
                   help="frequency binning for the persistence test "
                        "(default 10 Hz)")
    g.add_argument("--repeat-tol-steps", type=int, default=1,
                   help="drift binning for the persistence test, in drift "
                        "steps (default 1, mirroring sec 3.4). Tremblay+2026 "
                        "sec 3.6 requires frequency AND drift to match")

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
