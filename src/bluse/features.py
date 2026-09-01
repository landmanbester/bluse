#!/usr/bin/env python3
"""
features.py -- extensible feature registry for BLUSE hits.

Implements the 13 features of GLOBULAR clustering (Jacobson-Bell et al. 2025,
AJ 169:206, doi:10.3847/1538-3881/adb8e7, arXiv:2411.16556) plus a small number
of BLUSE-specific extras, and provides a registry so more can be added without
touching the extraction driver.

See papers/GLOBULAR-technical-reference.md for the full spec, including a table
of the three places our normalisation departs from the published one.

ADDING A FEATURE
----------------
Write a function, decorate it, done. Both kinds are batch-vectorised: never
write a per-hit Python loop.

    @meta_feature("my_column", description="...")
    def my_feature(df):
        # df: the whole catalogue as a DataFrame
        return {"my_column": df.snr.to_numpy() / df.power.to_numpy()}

    @stamp_feature("my_a", "my_b", description="...")
    def my_stamp_feature(b):
        # b: StampBatch with .img (B,T,C), .spectrum (B,C), .timeseries (B,T),
        #    .df_hz, .dt_s, .meta (DataFrame slice)
        return {"my_a": b.spectrum.max(axis=1),
                "my_b": b.timeseries.std(axis=1)}

One function may emit several columns -- use that when features share
expensive intermediate work (as f07/f08 do).

NORMALISATION IS SEPARATE. These functions return RAW values. The log /
quantile / unit-range transforms of the GLOBULAR feature table are applied
afterwards by `normalise()`, so the raw values stay available for Track E and
for any model that prefers its own scaling.

HOW THIS DIFFERS FROM GLOBULAR
------------------------------
GLOBULAR ran on GBT turboSETI hits with a 2.7 kHz spectral window and could
sweep bandwidths from 200 Hz to 100 kHz. Our stamps are 120 channels at
1.01-1.63 Hz, i.e. a total window of only ~121-196 Hz -- narrower than
GLOBULAR's *minimum* sweep bandwidth. Consequently:

  f07, f08  the kurtosis-vs-bandwidth sweep runs over ~5 Hz to ~196 Hz
            instead of 200 Hz to 100 kHz. Same construction, different regime:
            it probes the shape of the line itself rather than its spectral
            neighbourhood.
  f12       bandwidth at 1% of peak power is capped by the window, so a signal
            wider than the stamp saturates at the window width. Check
            f12_bandwidth_saturated before trusting it.
  f13       "redness" is computed over our ~150 Hz window, not GLOBULAR's
            10 kHz, so it detects fine comb structure across the line, not
            wide-band combs. We use an FFT periodogram rather than
            Lomb-Scargle: our spectrum is uniformly sampled, which is the
            case Lomb-Scargle exists to avoid needing.

None of this makes the features wrong, but they are not numerically comparable
to the published GLOBULAR values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import signal, stats

PAD_VALUE = -1.0
EPS = 1e-30


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    name: str
    func: Callable
    columns: tuple[str, ...]
    kind: str            # "meta" or "stamp"
    description: str


REGISTRY: dict[str, FeatureSpec] = {}


def _register(kind, columns, description):
    def deco(fn):
        spec = FeatureSpec(fn.__name__, fn, tuple(columns), kind,
                           description or (fn.__doc__ or "").strip())
        if spec.name in REGISTRY:
            raise ValueError(f"duplicate feature name {spec.name}")
        REGISTRY[spec.name] = spec
        return fn
    return deco


def meta_feature(*columns, description=""):
    """Register a feature computed from catalogue metadata alone."""
    return _register("meta", columns, description)


def stamp_feature(*columns, description=""):
    """Register a feature computed from the stamp cube."""
    return _register("stamp", columns, description)


def all_columns(kind=None):
    return [c for s in REGISTRY.values()
            if kind is None or s.kind == kind for c in s.columns]


@dataclass
class StampBatch:
    """A batch of cleaned stamps plus their derived 1-D projections."""
    img: np.ndarray          # (B, T, C) padding removed, cropped, NaN-free
    spectrum: np.ndarray     # (B, C)    time-integrated
    timeseries: np.ndarray   # (B, T)    frequency-averaged
    df_hz: float             # channel width
    dt_s: float              # time step
    meta: pd.DataFrame       # the matching catalogue rows


# ---------------------------------------------------------------------------
# stamp preparation
# ---------------------------------------------------------------------------

def prepare_batch(cube, meta, crop_channels=None):
    """
    Turn a raw (B, 1, T, 120) slice of `data` into a StampBatch.

    Stamps are right-aligned in a 120-channel buffer and padded with exactly
    -1. `numChannels` varies from 60 to 120 across the dataset, so unless we
    crop to a common width, every window-shape feature partly measures the
    window rather than the signal -- and a clusterer will happily find
    `numChannels` instead of physics. We therefore crop a fixed number of
    channels about the window centre, which is where the detection sits (98.7%
    of peaks fall in the central third).
    """
    arr = np.asarray(cube)
    if arr.ndim == 4:
        arr = arr[:, 0]                                   # (B, T, C)
    arr = arr.astype(np.float64, copy=True)

    nchan = meta["numChannels"].to_numpy()
    B, T, W = arr.shape
    crop = int(crop_channels or nchan.min())
    crop = max(3, min(crop, W))

    out = np.empty((B, T, crop), dtype=np.float64)
    for i in range(B):
        n = int(nchan[i])
        lo = W - n                                        # pad is leading
        centre = lo + n // 2
        a = centre - crop // 2
        a = max(lo, min(a, W - crop))
        out[i] = arr[i, :, a:a + crop]

    # Any residual pad or non-finite value would poison every moment.
    out[out == PAD_VALUE] = np.nan
    bad = ~np.isfinite(out)
    if bad.any():
        med = np.nanmedian(np.where(bad, np.nan, out), axis=(1, 2))
        med = np.where(np.isfinite(med), med, 0.0)
        out[bad] = np.broadcast_to(med[:, None, None], out.shape)[bad]

    return StampBatch(
        img=out,
        spectrum=out.mean(axis=1),
        timeseries=out.mean(axis=2),
        df_hz=float(abs(meta["foff"].iloc[0]) * 1e6),
        dt_s=float(meta["tsamp"].iloc[0]),
        meta=meta,
    )


def _norm(x):
    """Scale each row to zero median, unit MAD. Moments are then dimensionless."""
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True)
    return (x - med) / np.where(mad > EPS, mad, 1.0)


# ---------------------------------------------------------------------------
# GLOBULAR features 1-3: metadata
# ---------------------------------------------------------------------------

@meta_feature("f01_frequency",
              description="GLOBULAR #1. Observation frequency [MHz]. "
                          "Quantile-transformed to uniform by normalise(), "
                          "which removes spectrum-allocation bias while "
                          "preserving relative spectral proximity.")
def f01_frequency(df):
    return {"f01_frequency": df["frequency"].to_numpy(dtype=np.float64)}


@meta_feature("f02_abs_drift",
              description="GLOBULAR #2. |drift rate| [Hz/s]. Sign discarded; "
                          "quantile-transformed to Gaussian by normalise() to "
                          "stop the huge zero-drift spike dominating.")
def f02_abs_drift(df):
    return {"f02_abs_drift": np.abs(df["driftRate"].to_numpy(dtype=np.float64))}


@meta_feature("f03_snr",
              description="GLOBULAR #3. Signal-to-noise ratio, log-scaled by "
                          "normalise(). Spans ~7 decades here.")
def f03_snr(df):
    return {"f03_snr": df["snr"].to_numpy(dtype=np.float64)}


# ---------------------------------------------------------------------------
# GLOBULAR features 4-8: spectral shape
# ---------------------------------------------------------------------------

@stamp_feature("f04_spectral_skew",
               description="GLOBULAR #4. Skewness of the time-integrated "
                           "spectrum. High for broad, drifted or combed "
                           "signals.")
def f04_spectral_skew(b):
    return {"f04_spectral_skew": stats.skew(_norm(b.spectrum), axis=1)}


@stamp_feature("f05_spectral_kurtosis",
               description="GLOBULAR #5. Pearson kurtosis of the "
                           "time-integrated spectrum. A clean narrow line is "
                           "strongly leptokurtic.")
def f05_spectral_kurtosis(b):
    return {"f05_spectral_kurtosis":
            stats.kurtosis(_norm(b.spectrum), axis=1, fisher=False)}


@stamp_feature("f06_bimodality",
               description="GLOBULAR #6. Sarle's bimodality coefficient "
                           "b = (skew^2 + 1) / kurtosis on the spectrum. "
                           "Bounded 0-1, so normalise() leaves it alone.")
def f06_bimodality(b):
    s = _norm(b.spectrum)
    g = stats.skew(s, axis=1)
    k = stats.kurtosis(s, axis=1, fisher=False)
    return {"f06_bimodality": (g ** 2 + 1.0) / np.where(np.abs(k) > EPS, k, np.nan)}


@stamp_feature("f07_kurt_bw_corr", "f08_turning_bw_hz", "f08_turning_bw_saturated",
               description="GLOBULAR #7 and #8, from one shared sweep. "
                           "Kurtosis is measured in progressively wider "
                           "sub-windows centred on the detection; #7 is the "
                           "Pearson correlation of kurtosis against log "
                           "bandwidth, #8 is the bandwidth at which kurtosis "
                           "peaks (parabola vertex through the maximum and its "
                           "neighbours). GLOBULAR swept 200 Hz-100 kHz; our "
                           "window allows only ~5-196 Hz, which is below their "
                           "*minimum*, so for ~72% of hits kurtosis is still "
                           "rising at the window edge and the turning point is "
                           "unresolved. Those are marked by the companion "
                           "saturation flag and #8 should be read as a lower "
                           "limit for them -- filter on the flag before using "
                           "it, or drop #8 entirely on this data.")
def f07_f08_kurtosis_sweep(b, n_steps=12):
    B, C = b.spectrum.shape
    widths = np.unique(np.round(np.logspace(
        np.log10(3), np.log10(C), n_steps)).astype(int))
    widths = widths[widths >= 3]
    centre = C // 2

    kurt = np.empty((B, len(widths)))
    for j, w in enumerate(widths):
        a = max(0, centre - w // 2)
        sub = b.spectrum[:, a:a + w]
        kurt[:, j] = stats.kurtosis(_norm(sub), axis=1, fisher=False)

    log_bw = np.log10(widths * b.df_hz)

    # Pearson correlation of each row against log_bw
    x = log_bw - log_bw.mean()
    y = kurt - kurt.mean(axis=1, keepdims=True)
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum(axis=1))
    corr = np.where(denom > EPS, (y * x).sum(axis=1) / np.where(denom > EPS, denom, 1), np.nan)

    # Turning point: parabola vertex through the peak and its two neighbours.
    jmax = np.nanargmax(kurt, axis=1)
    jc = np.clip(jmax, 1, len(widths) - 2)
    i = np.arange(B)
    y0, y1, y2 = kurt[i, jc - 1], kurt[i, jc], kurt[i, jc + 1]
    x0, x1, x2 = log_bw[jc - 1], log_bw[jc], log_bw[jc + 1]
    denom2 = (y0 - 2 * y1 + y2)
    shift = np.where(np.abs(denom2) > EPS, 0.5 * (y0 - y2) / np.where(np.abs(denom2) > EPS, denom2, 1), 0.0)
    vertex = x1 + shift * (x2 - x1)
    # Never extrapolate outside the swept range.
    vertex = np.clip(vertex, log_bw[0], log_bw[-1])
    # At the sweep edges the parabola is not constrained; fall back to the grid.
    vertex = np.where((jmax == 0) | (jmax == len(widths) - 1), log_bw[jmax], vertex)

    return {"f07_kurt_bw_corr": corr,
            "f08_turning_bw_hz": 10.0 ** vertex,
            "f08_turning_bw_saturated": jmax >= len(widths) - 1}


# ---------------------------------------------------------------------------
# GLOBULAR features 9-11: temporal behaviour and scatter
# ---------------------------------------------------------------------------

@stamp_feature("f09_temporal_skew",
               description="GLOBULAR #9. Skewness of the frequency-averaged "
                           "time series -- a pulsing or intermittent emitter "
                           "is skewed. Only 15-59 samples here, so it is noisy.")
def f09_temporal_skew(b):
    return {"f09_temporal_skew": stats.skew(_norm(b.timeseries), axis=1)}


@stamp_feature("f10_timeseries_std",
               description="GLOBULAR #10. Standard deviation of the "
                           "frequency-averaged time series, relative to its "
                           "mean so it is dimensionless.")
def f10_timeseries_std(b):
    ts = b.timeseries
    mu = np.abs(ts.mean(axis=1))
    return {"f10_timeseries_std": ts.std(axis=1) / np.where(mu > EPS, mu, np.nan)}


@stamp_feature("f11_spectrum_std",
               description="GLOBULAR #11. Standard deviation of the "
                           "time-averaged spectrum, relative to its mean.")
def f11_spectrum_std(b):
    sp = b.spectrum
    mu = np.abs(sp.mean(axis=1))
    return {"f11_spectrum_std": sp.std(axis=1) / np.where(mu > EPS, mu, np.nan)}


# ---------------------------------------------------------------------------
# GLOBULAR features 12-13: bandwidth and comb structure
# ---------------------------------------------------------------------------

@stamp_feature("f12_bandwidth_hz", "f12_bandwidth_saturated",
               description="GLOBULAR #12. Width of the contiguous region "
                           "around the peak that stays above 1% of the "
                           "baseline-subtracted maximum. The companion flag is "
                           "True when the region reaches the window edge, i.e. "
                           "the true width exceeds the stamp and the value is a "
                           "lower limit.")
def f12_bandwidth(b):
    sp = b.spectrum
    base = np.median(sp, axis=1, keepdims=True)
    z = sp - base
    peak = z.max(axis=1, keepdims=True)
    above = z >= 0.01 * np.where(peak > EPS, peak, np.inf)

    B, C = sp.shape
    pk = np.argmax(z, axis=1)
    width = np.zeros(B, dtype=np.float64)
    saturated = np.zeros(B, dtype=bool)
    for i in range(B):
        j = pk[i]
        if not above[i, j]:
            width[i] = 0.0
            continue
        lo = j
        while lo > 0 and above[i, lo - 1]:
            lo -= 1
        hi = j
        while hi < C - 1 and above[i, hi + 1]:
            hi += 1
        width[i] = (hi - lo + 1)
        saturated[i] = (lo == 0) or (hi == C - 1)

    return {"f12_bandwidth_hz": width * b.df_hz,
            "f12_bandwidth_saturated": saturated}


@stamp_feature("f13_redness",
               description="GLOBULAR #13. Ratio of mean periodogram power in "
                           "the first half to the second half of the lowest "
                           "fifth of the spectrum's own periodogram -- large "
                           "when the spectrum carries slow, comb-like "
                           "structure. GLOBULAR used Lomb-Scargle over 10 kHz; "
                           "our spectrum is uniformly sampled over ~150 Hz, so "
                           "an FFT periodogram is the appropriate tool.")
def f13_redness(b):
    sp = _norm(b.spectrum)
    _, pxx = signal.periodogram(sp, axis=1, detrend="constant")
    pxx = pxx[:, 1:]                                   # drop the DC bin
    n = max(4, pxx.shape[1] // 5)
    low = pxx[:, :n]
    h = n // 2
    a = low[:, :h].mean(axis=1)
    c = low[:, h:].mean(axis=1)
    return {"f13_redness": a / np.where(c > EPS, c, np.nan)}


# ---------------------------------------------------------------------------
# BLUSE-specific extras -- not part of GLOBULAR
# ---------------------------------------------------------------------------

@stamp_feature("x01_drift_residual",
               description="EXTRA. Scatter of the per-timestep peak channel "
                           "about a straight-line fit, in channels. A coherent "
                           "drifting signal gives a small value; a wandering or "
                           "intermittent emitter does not.")
def x01_drift_residual(b):
    B, T, C = b.img.shape
    pk = np.argmax(b.img, axis=2).astype(np.float64)
    t = np.arange(T, dtype=np.float64)
    tc = t - t.mean()
    denom = (tc ** 2).sum()
    slope = ((pk - pk.mean(axis=1, keepdims=True)) * tc).sum(axis=1) / max(denom, EPS)
    fit = pk.mean(axis=1, keepdims=True) + slope[:, None] * tc[None, :]
    return {"x01_drift_residual": (pk - fit).std(axis=1)}


@stamp_feature("x02_time_occupancy",
               description="EXTRA. Fraction of timesteps whose peak exceeds "
                           "half the stamp's maximum -- an always-on carrier "
                           "sits near 1, a brief burst near 0.")
def x02_time_occupancy(b):
    mx = b.img.max(axis=2)
    thr = 0.5 * mx.max(axis=1, keepdims=True)
    return {"x02_time_occupancy": (mx >= thr).mean(axis=1)}


@meta_feature("x03_channel_offset",
              description="EXTRA. Position of the hit within its coarse "
                          "channel, 0-1. Instrumental artefacts cluster at "
                          "coarse-channel edges and centres.")
def x03_channel_offset(df):
    n = df["numChannels"].to_numpy(dtype=np.float64)
    start = df["startChannel"].to_numpy(dtype=np.float64)
    idx = df["index"].to_numpy(dtype=np.float64)
    span = np.where(n > 0, n, np.nan)
    return {"x03_channel_offset": np.clip((idx - start) / span, 0.0, 1.0)}


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

# Per the GLOBULAR feature table (Jacobson-Bell et al. 2025 sec 2), which
# specifies: log for features {3,5,8,10,11,13}; unit range for {1,3,4,5,8,10,
# 11,12,13}; unit variance for 2 after its quantile-normal transform; unit
# MAXIMUM for 9, negatives permitted; nothing for 6 and 7.
#
# "quantile-uniform"/"quantile-normal" are theirs; "log-unit" is their
# log-scale-then-normalise; "unit-max" preserves sign; "none" leaves a naturally
# bounded feature alone.
TRANSFORMS = {
    "f01_frequency":         "quantile-uniform",
    "f02_abs_drift":         "quantile-normal",
    "f03_snr":               "log-unit",
    "f04_spectral_skew":     "unit",
    "f05_spectral_kurtosis": "log-unit",
    "f06_bimodality":        "none",
    "f07_kurt_bw_corr":      "none",
    "f08_turning_bw_hz":     "log-unit",
    # Unit MAXIMUM, not unit range: temporal skew is signed, and a min-max
    # rescale to [0,1] throws that sign away. The paper keeps it deliberately
    # ("normalized to unit maximum with negative values permitted").
    "f09_temporal_skew":     "unit-max",
    "f10_timeseries_std":    "log-unit",
    "f11_spectrum_std":      "log-unit",
    # Unit range only -- f12 is NOT in the paper's log list.
    "f12_bandwidth_hz":      "unit",
    "f13_redness":           "log-unit",
    "x01_drift_residual":    "log-unit",
    "x02_time_occupancy":    "none",
    "x03_channel_offset":    "none",
}


def normalise(df, columns=None, transforms=None, suffix="_n"):
    """
    Apply the GLOBULAR preprocessing transforms, writing new `<col><suffix>`
    columns and leaving the raw values untouched.

    Fitted on whatever rows you pass, so fit on the training set and reuse the
    returned state if you care about leakage. For clustering, fitting on
    everything is the intended behaviour.
    """
    from sklearn.preprocessing import QuantileTransformer

    transforms = transforms or TRANSFORMS
    columns = columns or [c for c in transforms if c in df.columns]
    state = {}

    for col in columns:
        how = transforms.get(col, "unit")
        x = df[col].to_numpy(dtype=np.float64).copy()
        finite = np.isfinite(x)
        out = np.full(len(x), np.nan)
        if finite.sum() < 2:
            df[col + suffix] = out
            continue

        if how.startswith("log"):
            pos = finite & (x > 0)
            v = np.full(len(x), np.nan)
            v[pos] = np.log10(x[pos])
            x, finite = v, pos

        if how in ("quantile-uniform", "quantile-normal"):
            dist = "uniform" if how.endswith("uniform") else "normal"
            qt = QuantileTransformer(output_distribution=dist,
                                     n_quantiles=min(1000, int(finite.sum())),
                                     subsample=200_000, random_state=0)
            out[finite] = qt.fit_transform(x[finite].reshape(-1, 1)).ravel()
            state[col] = qt
        elif how == "unit-max":
            # Divide by the largest magnitude, so the scale is unity and the
            # sign survives. Not min-max.
            m = np.nanmax(np.abs(x[finite]))
            out[finite] = x[finite] / m if m > EPS else 0.0
            state[col] = m
        elif how == "none":
            out[finite] = x[finite]
        else:                                            # "unit", "log-unit"
            lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
            rng = hi - lo
            out[finite] = (x[finite] - lo) / rng if rng > EPS else 0.0
            state[col] = (lo, hi)

        df[col + suffix] = out

    return df, state


def describe():
    """Print the registry. `python features.py` calls this."""
    print(f"{len(REGISTRY)} feature functions, "
          f"{len(all_columns())} columns\n")
    for kind in ("meta", "stamp"):
        specs = [s for s in REGISTRY.values() if s.kind == kind]
        print(f"--- {kind} ({len(specs)} functions) ---")
        for s in specs:
            print(f"\n  {', '.join(s.columns)}")
            text = " ".join(s.description.split())
            while text:
                print(f"      {text[:76]}")
                text = text[76:]
        print()


if __name__ == "__main__":
    describe()
