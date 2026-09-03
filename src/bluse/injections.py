#!/usr/bin/env python
"""
Synthetic signal injection -- the only true objective function available.

WHY THIS EXISTS
---------------
Every number in Track E measures agreement with the multi-beam spatial filter.
That filter is a good instrument and it is not truth: no hit in this survey is
confirmed clean, so "the score reproduces the filter's verdict at 0.9899" is a
statement about two instruments agreeing, not about either being right.

Injecting signals we constructed ourselves breaks that circle. We know the
answer for an injected hit, so we can ask three questions nothing else can:

  1. WHAT IS THE OPERATING POINT? `shortlist_below=0.1` is currently a
     convention. With injections it becomes "the threshold that retains N% of
     real signals at SNR >= S" -- the difference between a ranking and a cut.
  2. IS THE MONOTONICITY AN SNR GRADIENT? The strongest evidence for the score
     is that it orders beam counts it never trained on. The strongest objection
     is that faint hits are both detected in fewer beams AND have less
     structure, so the ordering could be brightness in disguise. Scoring
     injected signals at MATCHED SNR settles it (docs/track-e-2026-09.md #8,
     falsification risk 2).
  3. DID SEED-AVERAGING REMOVE NOISE OR REMOVE DETECTIONS? Unanswerable without
     ground truth (docs/TODO.md, Track E open item 2).

THE SUBSTRATE PROBLEM, AND WHAT WE DO ABOUT IT
----------------------------------------------
Every stamp in the archive is centred on a detected hit, so there is no such
thing as an empty cube to inject into. We inject into stamps from hits near the
detection floor, where the cube is mostly noise -- and then measure a SHIFT
rather than an absolute:

    control   = features(substrate as delivered)
    injected  = features(substrate + synthetic signal)

The substrate's own signal is present in both, so whatever it contributes
largely cancels out of the comparison. `run()` always scores both and reports
the pair. A result quoted without its control is not a result.
"""

import numpy as np

from . import features as F

# Gaussian sigma of the injected carrier, in Hz. Roughly two channels at L and
# S band, three at UHF -- narrow enough to be a carrier, wide enough that the
# profile is not a single-sample spike that the crop could clip.
DEFAULT_BANDWIDTH_HZ = 3.0

# The crop `bluse-features` used to build all_features.parquet. It MUST match,
# or every stamp feature here is computed through a different window than the
# ones the model trained on -- a systematic offset that would show up as an
# injection effect. 60 is the global minimum numChannels across all seven
# files, so every hit gets an identical window.
EXTRACTION_CROP = 60

# Robust noise scale: 1.4826 * MAD recovers the Gaussian sigma. The stamps are
# integrated power and not Gaussian, but we only need a stable scale to define
# SNR against, and MAD is far less sensitive to the substrate's own signal than
# a standard deviation would be -- which matters, because the substrate always
# has one.
MAD_TO_SIGMA = 1.4826


def noise_sigma(img):
    """Per-stamp robust noise scale over the real (non-pad) samples."""
    img = np.asarray(img, dtype=np.float64)
    out = np.empty(len(img))
    for i, a in enumerate(img):
        v = a[np.isfinite(a) & (a != F.PAD_VALUE)]
        out[i] = MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))) if v.size else np.nan
    return out


def drift_track(n_time, n_chan, df_hz, dt_s, drift_hz_s, centre_channel=None):
    """
    Channel index of the signal at each time step, for a linear drift.

    Anchored at the MIDDLE time step rather than the first, so a fast drift
    runs off both edges symmetrically instead of leaving the window early. That
    is also how a detection is reported: the catalogue frequency is the
    signal's frequency at a reference time, not at t=0.
    """
    c0 = (n_chan - 1) / 2.0 if centre_channel is None else float(centre_channel)
    t = (np.arange(n_time) - (n_time - 1) / 2.0) * dt_s        # seconds, centred
    return c0 + (drift_hz_s * t) / df_hz


def signal_profile(n_time, n_chan, *, df_hz, dt_s, drift_hz_s,
                   bandwidth_hz=DEFAULT_BANDWIDTH_HZ, centre_channel=None):
    """
    Unit-peak drifting narrowband carrier: a Gaussian in frequency whose centre
    moves linearly in time. Returns (T, C), values in [0, 1].
    """
    track = drift_track(n_time, n_chan, df_hz, dt_s, drift_hz_s, centre_channel)
    chans = np.arange(n_chan)[None, :]
    sigma_ch = max(float(bandwidth_hz) / df_hz, 0.5)     # never sub-half-channel
    return np.exp(-0.5 * ((chans - track[:, None]) / sigma_ch) ** 2)


def amplitude_for_snr(profile, sigma, snr):
    """
    Amplitude giving a matched-filter SNR of `snr` against noise scale `sigma`.

    A filter matched to the profile g sums signal as A*sum(g) and noise as
    sigma*sqrt(sum(g^2)), so

        SNR = A * sum(g) / (sigma * sqrt(sum(g^2)))

    which inverts to the expression below. Stating it this way -- rather than
    "peak channel over noise" -- makes the number independent of how many
    channels and time steps the signal happens to occupy, so an SNR 20
    injection means the same thing in UHF (1.01 Hz channels, 36 steps) as in
    S band (1.63 Hz, 24 steps). It is NOT guaranteed identical to seticore's
    own definition; it is a stated, monotone, dimensionless strength, and the
    calibration curve is reported against it rather than against a claim of
    equivalence.
    """
    g = np.asarray(profile, dtype=np.float64)
    s1, s2 = g.sum(), np.sqrt((g ** 2).sum())
    return 0.0 if s1 <= 0 else float(snr) * float(sigma) * s2 / s1


def inject(cube, meta, *, snr, drift_hz_s, bandwidth_hz=DEFAULT_BANDWIDTH_HZ,
           centre_channel=None):
    """
    Add a synthetic drifting carrier to each stamp of a raw `data` slice.

    Takes and returns the cube in the layout `h5py` hands over -- (B, 1, T, W)
    or (B, T, W) -- so the result drops straight into `F.prepare_batch` and
    goes through exactly the same feature code as real data. Anything else
    would measure our reimplementation rather than the pipeline.

    Padding is preserved exactly. Stamps are right-aligned in a W-channel
    buffer with a leading run of PAD_VALUE, and injecting into the pad would
    make the signal partly indistinguishable from the window edge.
    """
    arr = np.asarray(cube)
    squeeze = arr.ndim == 4
    work = (arr[:, 0] if squeeze else arr).astype(np.float64, copy=True)

    nchan = np.asarray(meta["numChannels"], dtype=int)
    df_hz = float(abs(np.asarray(meta["foff"])[0]) * 1e6)
    dt_s = float(np.asarray(meta["tsamp"])[0])
    B, T, W = work.shape
    sigma = noise_sigma(work)

    for i in range(B):
        n = min(int(nchan[i]), W)
        lo = W - n                                    # the pad leads
        if not np.isfinite(sigma[i]) or sigma[i] <= 0:
            continue
        g = signal_profile(T, n, df_hz=df_hz, dt_s=dt_s, drift_hz_s=drift_hz_s,
                           bandwidth_hz=bandwidth_hz,
                           centre_channel=centre_channel)
        work[i, :, lo:] = work[i, :, lo:] + amplitude_for_snr(g, sigma[i], snr) * g

    return work[:, None] if squeeze else work


def stamp_features(cube, meta, crop_channels=None):
    """Every registered stamp feature for one cube, through the real path."""
    batch = F.prepare_batch(cube, meta, crop_channels=crop_channels)
    out = {}
    for spec in F.REGISTRY.values():
        if spec.kind == "stamp":
            out.update(spec.func(batch))
    return out


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def normalise_like(reference, new_raw, columns, suffix="_n", by="file"):
    """
    Map new raw feature values onto the transform the pipeline ALREADY applied
    -- PER FILE, because that is how the pipeline applied it.

    The GLOBULAR transforms are monotone rank maps, and `all_features.parquet`
    stores both sides of each one. So the transform never needs refitting:
    interpolating a new raw value onto the stored (raw, raw_n) curve reproduces
    it exactly for any value inside the observed range.

    THE MAP IS PER FILE. `bluse-features` calls `normalise()` inside `extract()`,
    once per file, and `summarise()` then concatenates the per-file frames
    without re-normalising. So a `_n` value means "this hit's rank within its own
    file", not within the survey, and the same raw value maps to different `_n`
    in different files -- measured: x01_drift_residual = 0.2790005 is 0.1523 in
    lband_long and 0.0253 in mk_sample_hits. Interpolating on the pooled table
    silently mixes seven maps; interpolating on the substrate's own file is
    exact.

    Two alternatives were measured and rejected. Refitting
    `features.normalise` on the union redraws QuantileTransformer's 200,000-row
    subsample and shifts the map by **0.070 mean absolute score** on low-SNR
    substrates -- four times the seed noise, enough to swamp the effect this
    experiment measures. Interpolating on the pooled table leaves **0.127**
    residual on x01 alone. Re-extraction itself is exact: all 12 raw columns
    reproduce bit-identically against the stored table.

    Values outside a file's observed range clip to its endpoints, which is what
    the pipeline's own transform does with its tie mass.
    """
    import pandas as pd

    ref_files = set(reference[by].unique())
    out = pd.DataFrame(index=new_raw.index,
                       columns=[c + suffix for c in columns], dtype=np.float64)
    for fname, part in new_raw.groupby(by, sort=False):
        ref = reference[reference[by] == fname]
        if ref.empty:
            raise ValueError(f"{fname!r} has no rows in the reference table; "
                             f"known files: {sorted(ref_files)}")
        for c in columns:
            x = np.asarray(ref[c], dtype=np.float64)
            y = np.asarray(ref[c + suffix], dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            order = np.argsort(x[m], kind="mergesort")
            xs, ys = x[m][order], y[m][order]
            ux, first = np.unique(xs, return_index=True)
            v = np.asarray(part[c], dtype=np.float64)
            r = np.full(len(v), np.nan)
            f = np.isfinite(v)
            if len(ux):
                r[f] = np.interp(v[f], ux, ys[first])
            out.loc[part.index, c + suffix] = r
    return out.reset_index(drop=True)


def select_substrates(cat, n, *, max_snr=8.0, seed=0):
    """
    Hits near the detection floor, to inject on top of.

    Low SNR is the closest thing to an empty cube this archive has, and it is
    NOT empty: at the floor the stamps still show visible drifting carriers.
    That is why every measurement here is a control/injected pair -- see the
    module docstring.
    """
    pool = cat[cat["snr"] <= max_snr]
    if len(pool) < n:
        pool = cat.nsmallest(max(n, 1), "snr")
    return pool.sample(min(n, len(pool)), random_state=seed).sort_values("row")


# Reported per substrate, in this order, so the floor is always visible:
#   stored    the score the shipped pipeline gives this hit
#   control   re-extracted and re-normalised, NO injection
#   injected  the same, with a synthetic carrier added
#
# `stored` vs `control` is the cost of re-extraction plus refitting the rank
# transforms -- measured at mean 0.017 in absolute score, about 3x the
# seed-to-seed noise. It is a floor on this experiment and it is reported, not
# assumed away. `control` vs `injected` is the measurement, and it is clean:
# both go through ONE normalise call, so they share a transform exactly.


def run(files, *, workspace_dirs, feature_table, n_substrate=200,
        snr_grid=(5, 8, 12, 20, 35, 60, 100), drift_grid=(0.0, 0.1, 0.3),
        max_snr=8.0, bandwidth_hz=DEFAULT_BANDWIDTH_HZ, crop=EXTRACTION_CROP,
        seed=0, n_splits=5, n_seeds=3, verbose=True):
    """
    Inject, extract, score. Returns a tidy frame, one row per
    (substrate, snr, drift), plus the controls at snr=0.

    Everything goes through the real code path: `F.prepare_batch` and the
    registered stamp features, not a reimplementation. A harness that computed
    its own features would measure the harness.
    """
    import os

    import h5py
    import pandas as pd

    from . import track_e_score as E

    def say(*a):
        if verbose:
            print(*a, flush=True)

    raw_cols = [c[:-2] for c in E.FEATURE_SETS["stamp"]]
    n_cols = E.FEATURE_SETS["stamp"]

    say(f"training {n_splits} fold models x {n_seeds} seeds...")
    models, fold_of_group = E.fold_models(
        feature_table, features="stamp", n_splits=n_splits, seed=seed,
        n_seeds=n_seeds)

    rows, meta_rows = [], []
    for name in files:
        cat_path = os.path.join(workspace_dirs["catalogues"],
                                f"{name}_cat.parquet")
        h5_path = os.path.join(workspace_dirs["data"], f"{name}.h5")
        if not (os.path.exists(cat_path) and os.path.exists(h5_path)):
            say(f"  {name}: missing catalogue or HDF5, skipped")
            continue
        cat = pd.read_parquet(cat_path)
        sub = select_substrates(cat, n_substrate, max_snr=max_snr, seed=seed)
        idx = sub["row"].to_numpy().astype(int)
        with h5py.File(h5_path, "r") as h:
            cube = np.stack([h["data"][int(r)] for r in idx])
        say(f"  {name}: {len(sub)} substrates, "
            f"SNR {sub.snr.min():.1f}-{sub.snr.max():.1f}")

        variants = [(0.0, 0.0)] + [(s, d) for s in snr_grid for d in drift_grid]
        for snr, drift in variants:
            c = cube if snr == 0 else inject(
                cube, sub, snr=snr, drift_hz_s=drift, bandwidth_hz=bandwidth_hz)
            feats = stamp_features(c, sub, crop_channels=crop)
            block = pd.DataFrame({k: np.asarray(v, dtype=np.float64)
                                  for k, v in feats.items()})
            block["file"] = name          # normalise_like maps per file
            rows.append(block)
            meta_rows.append(pd.DataFrame({
                "file": name, "row": idx, "id": sub["id"].to_numpy(),
                "obsid": sub["obsid"].to_numpy(),
                "substrate_snr": sub["snr"].to_numpy(),
                "n_beams": sub["n_beams"].to_numpy(),
                "injected_snr": float(snr), "drift_hz_s": float(drift),
                "kind": "control" if snr == 0 else "injected"}))

    if not rows:
        raise SystemExit("no substrates: check the workspace has catalogues "
                         "and HDF5 files")

    new_raw = pd.concat(rows, ignore_index=True)
    out = pd.concat(meta_rows, ignore_index=True)
    say(f"normalising {len(new_raw):,} stamps on each file's own transform...")
    norm = normalise_like(feature_table, new_raw, raw_cols)

    say("scoring, each with the fold that held its observation out...")
    score, n_fallback = E.predict_held_out(
        models, fold_of_group, norm[n_cols].to_numpy(), out["obsid"].to_numpy())
    out["rfi_score"] = score
    if n_fallback:
        say(f"  WARNING: {n_fallback} rows had no held-out fold and used fold 0")

    # The floor: what the shipped pipeline says about these same hits.
    stored = feature_table.set_index("id")
    keep = out["id"].isin(stored.index)
    sc, _ = E.predict_held_out(
        models, fold_of_group,
        stored.loc[out.loc[keep, "id"], n_cols].to_numpy(),
        out.loc[keep, "obsid"].to_numpy())
    out.loc[keep, "stored_score"] = sc
    out["n_fallback"] = n_fallback
    return out
