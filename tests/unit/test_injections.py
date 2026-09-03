"""
Synthetic injection.

The harness is only worth anything if it measures the pipeline rather than
itself, so these tests pin the three properties that make that true: the SNR
normalisation delivers the SNR it was asked for, the padding survives, and the
normalisation maps new values onto the transform the pipeline ALREADY applied
-- per file, which is how the pipeline applied it.
"""

import numpy as np
import pandas as pd
import pytest

from bluse import features as F
from bluse import injections as I


def _meta(n=4, nchan=80, foff=1.6298e-6, tsamp=4.909):
    return pd.DataFrame({"numChannels": [nchan] * n, "foff": [foff] * n,
                         "tsamp": [tsamp] * n, "file": ["synthetic"] * n})


def _cube(n=4, T=24, W=120, nchan=80, seed=0, level=4e14):
    """A padded stamp block shaped like the real thing: pad LEADS, value -1."""
    rng = np.random.default_rng(seed)
    a = rng.gamma(shape=15.0, scale=level / 15.0, size=(n, T, W))
    a[:, :, :W - nchan] = F.PAD_VALUE
    return a


def test_amplitude_delivers_the_snr_it_was_asked_for():
    """
    The matched-filter definition is the load-bearing choice in this module: it
    is what makes 'SNR 20' mean the same thing in UHF (1.01 Hz channels, 36
    steps) as in S band (1.63 Hz, 24 steps). If the normalisation drifts, every
    calibration curve silently changes its x-axis.
    """
    g = I.signal_profile(24, 80, df_hz=1.63, dt_s=4.9, drift_hz_s=0.1)
    for snr in (5, 20, 100):
        a = I.amplitude_for_snr(g, sigma=1.0, snr=snr)
        got = a * g.sum() / np.sqrt((g ** 2).sum())
        assert abs(got - snr) < 1e-9, (snr, got)


def test_amplitude_scales_with_the_noise_it_is_measured_against():
    g = I.signal_profile(24, 80, df_hz=1.63, dt_s=4.9, drift_hz_s=0.0)
    a1 = I.amplitude_for_snr(g, sigma=1.0, snr=10)
    a2 = I.amplitude_for_snr(g, sigma=7.0, snr=10)
    assert np.isclose(a2, 7 * a1)


def test_injection_never_touches_the_pad():
    """
    Stamps are right-aligned in a 120-channel buffer with a leading run of
    exactly -1, and prepare_batch converts that pad to NaN. Injecting into it
    would put signal where the pipeline expects absence -- and f12_bandwidth and
    f08_turning_bw would then be measuring the window edge.
    """
    c = _cube(nchan=80)
    m = _meta(nchan=80)
    out = I.inject(c, m, snr=100, drift_hz_s=0.2)
    pad = slice(None), slice(None), slice(0, 120 - 80)
    assert (out[pad] == F.PAD_VALUE).all()
    assert not np.allclose(out[:, :, 40:], c[:, :, 40:]), "nothing was injected"


def test_injection_preserves_the_cube_layout():
    """h5py hands over (B, 1, T, W); prepare_batch expects it back that way."""
    c4 = _cube()[:, None]
    out = I.inject(c4, _meta(), snr=20, drift_hz_s=0.1)
    assert out.shape == c4.shape
    c3 = _cube()
    assert I.inject(c3, _meta(), snr=20, drift_hz_s=0.1).shape == c3.shape


def test_zero_drift_is_a_straight_line_and_drift_is_not():
    flat = I.drift_track(24, 80, 1.63, 4.9, drift_hz_s=0.0)
    tilt = I.drift_track(24, 80, 1.63, 4.9, drift_hz_s=0.3)
    assert np.allclose(flat, flat[0])
    assert tilt[-1] > tilt[0]
    # Anchored at the MIDDLE step, so the track is symmetric about the centre.
    assert np.isclose(tilt.mean(), flat[0])


def test_a_brighter_injection_raises_the_stamp_more():
    c, m = _cube(), _meta()
    base = np.nanmedian(np.where(c == F.PAD_VALUE, np.nan, c))
    peaks = [np.nanmax(np.where(I.inject(c, m, snr=s, drift_hz_s=0.1)
                                == F.PAD_VALUE, np.nan,
                                I.inject(c, m, snr=s, drift_hz_s=0.1)))
             for s in (5, 50, 200)]
    assert peaks[0] < peaks[1] < peaks[2]
    assert peaks[0] > base


def test_normalise_like_reproduces_the_pipeline_exactly():
    """
    The whole experiment is a control/injected comparison, so any error in this
    map is a floor under the effect being measured. Refitting
    features.normalise on the union was measured at 0.070 mean absolute score
    -- four times the seed noise. Interpolation on the stored curve is exact.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    ref = pd.DataFrame({"file": "A", "f04_spectral_skew": x,
                        "f04_spectral_skew_n": np.tanh(x)})   # any monotone map
    probe = ref.sample(200, random_state=1)
    got = I.normalise_like(ref, probe, ["f04_spectral_skew"])
    assert np.allclose(got["f04_spectral_skew_n"].to_numpy(),
                       probe["f04_spectral_skew_n"].to_numpy(), atol=1e-12)


def test_normalise_like_uses_each_file_s_own_transform():
    """
    Regression for a real defect. `bluse-features` calls normalise() inside
    extract(), once PER FILE, and summarise() concatenates without
    re-normalising -- so a `_n` value is a rank within its own file. Measured on
    the real table: x01_drift_residual = 0.2790005 is 0.1523 in lband_long and
    0.0253 in mk_sample_hits.

    Interpolating on the pooled table mixes the maps and leaves 0.127 residual.
    """
    x = np.linspace(-2, 2, 500)
    ref = pd.concat([
        pd.DataFrame({"file": "A", "f04_spectral_skew": x,
                      "f04_spectral_skew_n": x}),
        pd.DataFrame({"file": "B", "f04_spectral_skew": x,
                      "f04_spectral_skew_n": 10 * x})])       # a different map
    probe = pd.DataFrame({"file": ["A", "B"], "f04_spectral_skew": [1.0, 1.0]})
    got = I.normalise_like(ref, probe, ["f04_spectral_skew"])
    assert np.isclose(got["f04_spectral_skew_n"].iloc[0], 1.0)
    assert np.isclose(got["f04_spectral_skew_n"].iloc[1], 10.0)


def test_normalise_like_refuses_an_unknown_file():
    ref = pd.DataFrame({"file": "A", "f04_spectral_skew": [0.0, 1.0],
                        "f04_spectral_skew_n": [0.0, 1.0]})
    probe = pd.DataFrame({"file": ["Z"], "f04_spectral_skew": [0.5]})
    with pytest.raises(ValueError, match="no rows in the reference"):
        I.normalise_like(ref, probe, ["f04_spectral_skew"])


def test_substrate_selection_prefers_the_detection_floor():
    cat = pd.DataFrame({"snr": np.r_[np.full(50, 6.5), np.full(50, 500.0)],
                        "row": np.arange(100)})
    s = I.select_substrates(cat, 20, max_snr=8.0, seed=0)
    assert len(s) == 20 and (s.snr <= 8.0).all()
    assert s.row.is_monotonic_increasing, "read order must be sorted for h5py"


def test_substrate_selection_falls_back_rather_than_returning_nothing():
    cat = pd.DataFrame({"snr": np.full(30, 500.0), "row": np.arange(30)})
    s = I.select_substrates(cat, 10, max_snr=8.0, seed=0)
    assert len(s) == 10, "no hit under the cap must not mean no substrates"
