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

    A filter matched to the profile g weights the data BY g, so it sums signal
    as A*sum(g^2) and noise as sigma*sqrt(sum(g^2)):

        SNR = A * sum(g^2) / (sigma * sqrt(sum(g^2)))
            = A * sqrt(sum(g^2)) / sigma

    which inverts to the expression below.

    THE FIRST VERSION OF THIS FUNCTION WAS WRONG and every number in the first
    injection run carried the error. It used `A*sum(g)/(sigma*sqrt(sum(g^2)))`,
    which takes its numerator from a UNIT-weight filter and its denominator from
    a matched one -- not a consistent filter at all. Monte Carlo against the
    real statistic: asking for 20 delivered 14.19, a factor of 1/sqrt(2).

    For the Gaussian profile and the grid first shipped the error was a near
    constant sqrt(2) rescale of the axis (1.4142 across S/L/UHF and drifts
    0-0.3, worst deviation 0.4%), so no qualitative conclusion moved -- but it
    is genuinely profile-dependent once bandwidth is swept: an unresolved
    carrier at 0.8 Hz gives 1.678 at zero drift and 1.389 at 0.3 Hz/s. Found by
    Copilot on PR #3.

    Stating SNR this way makes it independent of how many channels and time
    steps the signal occupies. It is NOT seticore's definition; the conversion
    to a single-channel dedoppler statistic is recorded in
    docs/injections-2026-09.md.
    """
    g = np.asarray(profile, dtype=np.float64)
    s2 = np.sqrt((g ** 2).sum())
    return 0.0 if s2 <= 0 else float(snr) * float(sigma) / s2


def dedoppler_snr(profile, amplitude, sigma):
    """
    The single-channel dedoppler SNR the same injection would show -- the
    seticore-style statistic, summing one channel per timestep along the track.

    Reported alongside the matched-filter value because every operational
    statement has to be expressible in units a user can measure on their own
    data, and the catalogue's `snr` column is of this kind. The ratio between
    the two is band-dependent, so quoting only the harness unit invites a reader
    to compare it against `snr` and be wrong.
    """
    g = np.asarray(profile, dtype=np.float64)
    T = g.shape[0]
    peak = g[np.arange(T), np.argmax(g, axis=1)]
    return float(amplitude) * float(peak.sum()) / (float(sigma) * np.sqrt(T))


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

    `all_features.parquet` stores both sides of every transform: the raw column
    and its `_n` partner. The transforms are monotone, so they never need
    refitting -- interpolating a new raw value onto the stored (raw, raw_n)
    curve reproduces the map exactly inside the observed range.

    THE MAP IS PER FILE. `bluse-features` calls `normalise()` inside
    `extract()`, once per file, and `summarise()` then concatenates without
    re-normalising. So a `_n` value is a rank within its OWN FILE, not within
    the survey, and the same raw value maps to different `_n` in different
    files -- measured: x01_drift_residual = 0.2790005 is 0.1523 in lband_long
    and 0.0253 in mk_sample_hits.

    WHY REFITTING SHIFTS THE MAP. An earlier version of this docstring blamed
    QuantileTransformer redrawing its 200,000-row subsample. That was wrong:
    **none of the 12 stamp columns uses a quantile transform.** They are `unit`
    (f04, f12), `log-unit` (f05, f08, f10, f11, f13, x01), `unit-max` (f09) and
    `none` (f06, f07, x02); only f01 and f02 are quantile, and both are META
    features that FEATURE_SETS["stamp"] does not contain. The real mechanism is
    more mundane and more dangerous: `unit` and `log-unit` are per-file MIN-MAX,
    so an injected value beyond a file's observed max moves `lo`/`hi` and
    rescales EVERY row in that file. Caught by review on PR #3.

    Measured costs of the alternatives: refitting `features.normalise` on the
    union moved scores by 0.070 mean absolute -- four times the seed noise;
    interpolating on the POOLED table left 0.127 residual. Per-file
    interpolation is exact (0.000e+00 across 3,000 rows, all 12 columns), and
    re-extraction is exact independently (all 12 raw columns bit-identical).

    Values outside a file's observed range clip to its endpoints. For the affine
    transforms here the correct behaviour is linear EXTRAPOLATION, so the
    returned diagnostic counts how many values fall outside -- see
    `normalise_like`'s second return value.
    """
    import pandas as pd

    ref_files = set(reference[by].unique())
    out = pd.DataFrame(index=new_raw.index,
                       columns=[c + suffix for c in columns], dtype=np.float64)
    oor = np.zeros(len(new_raw))
    for fname, part in new_raw.groupby(by, sort=False):
        ref = reference[reference[by] == fname]
        if ref.empty:
            raise ValueError(f"{fname!r} has no rows in the reference table; "
                             f"known files: {sorted(ref_files)}")
        n_out = np.zeros(len(part))
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
                # np.interp CLAMPS outside the fitted range, so a value beyond
                # a file's observed max is silently pinned to its endpoint --
                # and the model's 256 bins make the endpoint and far beyond it
                # indistinguishable. Count it rather than claim it cannot happen.
                n_out += (v < ux[0]) | (v > ux[-1])
            out.loc[part.index, c + suffix] = r
        oor[new_raw.index.get_indexer(part.index)] = n_out / max(len(columns), 1)
    return out.reset_index(drop=True), oor


SUBSTRATE_CLASSES = {"single": (1, 2), "mid": (3, 31), "multibeam": (32, 64)}


def substrate_class(n_beams):
    """The three populations, by the same cuts the weak labels use."""
    nb = np.asarray(n_beams)
    return np.where(nb >= 32, "multibeam", np.where(nb <= 2, "single", "mid"))


def select_substrates(cat, n, *, max_snr=8.0, beams=None, seed=0):
    """
    Hits near the detection floor, to inject on top of.

    `beams` is an inclusive (lo, hi) bound on `n_beams`, and leaving it None was
    a defect in the first version of this experiment. Low SNR does NOT imply few
    beams: filtering on `snr <= 8` alone gave 42.4% multi-beam RFI, 44.4%
    ambiguous and only 13.1% single-beam hits -- so the headline pooled a
    population the score is deployed on with one it was trained to reject, and
    the two behave in opposite directions (docs/injections-2026-09.md #3).

    Sample each class separately and report it separately. Pooling is what
    breaks.
    """
    pool = cat[cat["snr"] <= max_snr]
    if beams is not None:
        lo, hi = beams
        pool = pool[(pool["n_beams"] >= lo) & (pool["n_beams"] <= hi)]
    if pool.empty:
        return pool
    if len(pool) < n:
        n = len(pool)
    return (pool.sample(n, random_state=seed)
            .sort_values("row").reset_index(drop=True))


# Reported per substrate, in this order:
#   stored    the same hit's STORED _n columns, scored by the same fold ensemble
#   control   re-extracted and re-normalised, NO injection
#   injected  the same, with a synthetic carrier added
#
# `stored` vs `control` is a HARNESS SELF-CONSISTENCY CHECK, not a comparison
# against the shipped column -- it uses one fold's ensemble, whereas fit_score
# scores unlabelled rows with the mean of all five. It is exactly zero in the
# committed run, which is real evidence that re-extraction plus normalise_like
# reproduces the stored values. An earlier comment here claimed it measured a
# 0.017 discrepancy against the shipped pipeline; both halves were wrong.
#
# `control` vs `injected` is the measurement. Both go through ONE normalise
# call and both traverse inject(), so they share every code path except the
# amplitude.


def run(files, *, workspace_dirs, feature_table, n_per_class=150,
        classes=("single", "mid", "multibeam"),
        snr_grid=(5, 8, 12, 20, 35, 60, 100), drift_grid=(0.0, 0.1, 0.3),
        max_snr=8.0, bandwidth_hz=DEFAULT_BANDWIDTH_HZ, crop=EXTRACTION_CROP,
        seed=0, n_splits=5, n_seeds=3, verbose=True):
    """
    Inject, extract, score. One row per (substrate, snr, drift), controls
    included at snr=0.

    Substrates are sampled PER BEAM CLASS and the class is recorded, because
    pooling them was the defect that broke the first run: the score responds in
    opposite directions on single-beam and multi-beam substrates, and a pooled
    number is a composition effect (docs/injections-2026-09.md #3).

    Everything goes through the real code path -- `F.prepare_batch` and the
    registered stamp features. A harness that computed its own features would
    measure the harness.
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

    rows, meta_rows, n_unreadable = [], [], 0
    for name in files:
        cat_path = os.path.join(workspace_dirs["catalogues"],
                                f"{name}_cat.parquet")
        h5_path = os.path.join(workspace_dirs["data"], f"{name}.h5")
        if not (os.path.exists(cat_path) and os.path.exists(h5_path)):
            say(f"  {name}: missing catalogue or HDF5, skipped")
            continue
        cat = pd.read_parquet(cat_path)

        for cls in classes:
            sub = select_substrates(cat, n_per_class, max_snr=max_snr,
                                    beams=SUBSTRATE_CLASSES[cls], seed=seed)
            if sub.empty:
                say(f"  {name}/{cls}: no substrates in this class")
                continue
            idx = sub["row"].to_numpy().astype(int)
            # The documented corrupt block in uhf_long raises OSError per row,
            # so a six-file run must not die hours in on one bad stamp.
            stamps, keep = [], []
            with h5py.File(h5_path, "r") as h:
                for i, r in enumerate(idx):
                    try:
                        stamps.append(h["data"][int(r)])
                        keep.append(i)
                    except Exception:
                        n_unreadable += 1
            if not stamps:
                say(f"  {name}/{cls}: no readable stamps")
                continue
            sub = sub.iloc[keep].reset_index(drop=True)
            cube = np.stack(stamps)
            idx = sub["row"].to_numpy().astype(int)

            # Hoisted: sigma depends on the substrate, not on the variant.
            work = cube[:, 0] if cube.ndim == 4 else cube
            sigma = noise_sigma(work)
            ok = np.isfinite(sigma) & (sigma > 0)
            say(f"  {name}/{cls}: {len(sub)} substrates, "
                f"SNR {sub.snr.min():.1f}-{sub.snr.max():.1f}, "
                f"{int((~ok).sum())} with degenerate noise")

            df_hz = float(abs(sub["foff"].iloc[0]) * 1e6)
            dt_s = float(sub["tsamp"].iloc[0])
            if sub["tsamp"].nunique() > 1 or sub["foff"].nunique() > 1:
                say(f"    WARNING: {name} mixes foff/tsamp; injections use "
                    f"row 0's values (df={df_hz:.4f} Hz, dt={dt_s:.3f} s)")

            for snr, drift in [(0.0, 0.0)] + [(s_, d) for s_ in snr_grid
                                              for d in drift_grid]:
                # The control traverses inject() too -- amplitude_for_snr
                # returns 0 at snr=0 -- so no asymmetry of code path can enter
                # the one comparison the whole experiment rests on.
                c = inject(cube, sub, snr=snr, drift_hz_s=drift,
                           bandwidth_hz=bandwidth_hz)
                feats = stamp_features(c, sub, crop_channels=crop)
                block = pd.DataFrame({k: np.asarray(v, dtype=np.float64)
                                      for k, v in feats.items()})
                block["file"] = name
                rows.append(block)

                g = signal_profile(work.shape[1], min(int(sub["numChannels"].min()),
                                                      work.shape[2]),
                                   df_hz=df_hz, dt_s=dt_s, drift_hz_s=drift,
                                   bandwidth_hz=bandwidth_hz)
                ded = np.array([dedoppler_snr(g, amplitude_for_snr(g, sg, snr), sg)
                                if o else np.nan for sg, o in zip(sigma, ok)])
                meta_rows.append(pd.DataFrame({
                    "file": name, "row": idx, "id": sub["id"].to_numpy(),
                    "obsid": sub["obsid"].to_numpy(),
                    "substrate_class": cls,
                    "substrate_snr": sub["snr"].to_numpy(),
                    "n_beams": sub["n_beams"].to_numpy(),
                    "noise_sigma": sigma, "injected_ok": ok,
                    "injected_snr": float(snr), "dedoppler_snr": ded,
                    "drift_hz_s": float(drift),
                    "kind": "control" if snr == 0 else "injected"}))

    if not rows:
        raise ValueError("no substrates: check the workspace has catalogues "
                         "and HDF5 files")
    if n_unreadable:
        say(f"  {n_unreadable} stamps were unreadable and were skipped")

    new_raw = pd.concat(rows, ignore_index=True)
    out = pd.concat(meta_rows, ignore_index=True)
    say(f"normalising {len(new_raw):,} stamps on each file's own transform...")
    norm, oor = normalise_like(feature_table, new_raw, raw_cols)
    out["frac_out_of_range"] = oor

    say("scoring, each with the fold that held its observation out...")
    score, fold_used, n_fallback = E.predict_held_out(
        models, fold_of_group, norm[n_cols].to_numpy(), out["obsid"].to_numpy())
    out["rfi_score"] = score
    out["fold_used"] = fold_used
    out["fold_was_fallback"] = np.asarray(fold_used) < 0
    if n_fallback:
        say(f"  WARNING: {n_fallback} rows had no held-out fold (flagged)")

    if feature_table["id"].duplicated().any():
        raise ValueError("feature_table has duplicate ids; stored_score would "
                         "misalign")
    stored = feature_table.set_index("id")
    keep = out["id"].isin(stored.index)
    out["stored_score"] = np.nan
    if keep.any():
        sc, _, _ = E.predict_held_out(
            models, fold_of_group,
            stored.loc[out.loc[keep, "id"], n_cols].to_numpy(),
            out.loc[keep, "obsid"].to_numpy())
        out.loc[keep, "stored_score"] = sc
    return out
