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


def test_amplitude_delivers_the_matched_filter_snr_by_simulation():
    """
    Measured against the real statistic, not against the algebraic inverse of
    the function under test -- which is what the first version of this test did,
    and which is why it could not fail when the formula was wrong.

    It WAS wrong. `A*sum(g)/(sigma*sqrt(sum(g**2)))` takes its numerator from a
    unit-weight filter and its denominator from a matched one. Asking for 20
    delivered 14.19. Caught by Copilot on PR #3, after the first injection run
    had already been published against the bad axis.
    """
    rng = np.random.default_rng(0)
    g = I.signal_profile(24, 60, df_hz=1.6298, dt_s=4.909, drift_hz_s=0.1)
    for want in (5.0, 20.0):
        a = I.amplitude_for_snr(g, sigma=1.0, snr=want)
        n = 20_000
        sig = np.mean([((a * g + rng.normal(0, 1, g.shape)) * g).sum()
                       for _ in range(n)])
        noise = np.std([(rng.normal(0, 1, g.shape) * g).sum() for _ in range(n)])
        assert abs(sig / noise - want) < 0.15 * want, (want, sig / noise)


def test_dedoppler_conversion_is_reported_and_differs_by_band():
    """
    Every operational statement has to be expressible in units a user can
    measure. The harness SNR is not seticore's, and the ratio between them is
    band-dependent, so quoting only the harness unit invites a reader to compare
    it against the catalogue's `snr` column and be wrong.
    """
    ratios = {}
    for band, df_hz, T in (("S", 1.6298, 24), ("UHF", 1.0100, 36)):
        g = I.signal_profile(T, 60, df_hz=df_hz, dt_s=4.909, drift_hz_s=0.0)
        a = I.amplitude_for_snr(g, 1.0, 100.0)
        ratios[band] = 100.0 / I.dedoppler_snr(g, a, 1.0)
    assert all(r > 1 for r in ratios.values()), ratios
    assert abs(ratios["UHF"] - ratios["S"]) > 0.1, ratios


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
    Any error in this map is a floor under the effect being measured. Refitting
    features.normalise on the union cost 0.070 mean absolute score -- four times
    the seed noise.

    Probes strictly BETWEEN stored knots as well as at them: an earlier version
    sampled the probe from the reference, so every value was a knot and
    exactness was guaranteed by construction.
    """
    x = np.linspace(-3.0, 3.0, 401)
    ref = pd.DataFrame({"file": "A", "f04_spectral_skew": x,
                        "f04_spectral_skew_n": 2.0 * x + 1.0})   # affine, exact
    mid = pd.DataFrame({"file": "A",
                        "f04_spectral_skew": (x[:-1] + x[1:]) / 2})
    got, _ = I.normalise_like(ref, mid, ["f04_spectral_skew"])
    assert np.allclose(got["f04_spectral_skew_n"].to_numpy(),
                       2.0 * mid["f04_spectral_skew"].to_numpy() + 1.0,
                       atol=1e-12)


def test_normalise_like_reports_values_outside_the_fitted_range():
    """
    np.interp CLAMPS. A value beyond a file's observed max is pinned to the
    endpoint, and HistGradientBoosting's 256 bins make the endpoint and far
    beyond it produce identical predictions -- so "inside the trained range"
    cannot be checked by looking at the normalised values, which is circular.
    The count has to come out of the mapping itself.
    """
    ref = pd.DataFrame({"file": "A", "f04_spectral_skew": [0.0, 1.0],
                        "f04_spectral_skew_n": [0.0, 1.0]})
    probe = pd.DataFrame({"file": "A", "f04_spectral_skew": [0.5, 99.0, -99.0]})
    got, oor = I.normalise_like(ref, probe, ["f04_spectral_skew"])
    assert list(oor) == [0.0, 1.0, 1.0]
    assert got["f04_spectral_skew_n"].iloc[1] == 1.0, "should clamp, not soar"


def test_normalise_like_uses_each_file_s_own_transform():
    """
    Regression for a real defect. `bluse-features` calls normalise() inside
    extract(), once PER FILE, and summarise() concatenates without
    re-normalising -- so a `_n` value is a rank within its own file. Measured on
    the real table: x01_drift_residual = 0.2790005 is 0.1523 in lband_long and
    0.0253 in mk_sample_hits. Interpolating on the pooled table mixes the maps
    and leaves 0.127 residual.
    """
    x = np.linspace(-2, 2, 500)
    ref = pd.concat([
        pd.DataFrame({"file": "A", "f04_spectral_skew": x,
                      "f04_spectral_skew_n": x}),
        pd.DataFrame({"file": "B", "f04_spectral_skew": x,
                      "f04_spectral_skew_n": 10 * x})])
    probe = pd.DataFrame({"file": ["A", "B"], "f04_spectral_skew": [1.0, 1.0]})
    got, _ = I.normalise_like(ref, probe, ["f04_spectral_skew"])
    assert np.isclose(got["f04_spectral_skew_n"].iloc[0], 1.0)
    assert np.isclose(got["f04_spectral_skew_n"].iloc[1], 10.0)


def test_normalise_like_refuses_an_unknown_file():
    ref = pd.DataFrame({"file": "A", "f04_spectral_skew": [0.0, 1.0],
                        "f04_spectral_skew_n": [0.0, 1.0]})
    probe = pd.DataFrame({"file": ["Z"], "f04_spectral_skew": [0.5]})
    with pytest.raises(ValueError, match="no rows in the reference"):
        I.normalise_like(ref, probe, ["f04_spectral_skew"])


def test_substrate_selection_filters_by_beam_class():
    """
    The defect that broke the first run. Low SNR does NOT imply few beams:
    filtering on `snr <= 8` alone gave 42.4% multi-beam RFI and only 13.1%
    single-beam, and the score responds in OPPOSITE directions on those two
    populations -- so the pooled headline was a composition effect.
    """
    cat = pd.DataFrame({"snr": np.full(120, 6.5),
                        "n_beams": np.r_[np.full(40, 1), np.full(40, 10),
                                         np.full(40, 55)],
                        "row": np.arange(120)})
    single = I.select_substrates(cat, 30, beams=I.SUBSTRATE_CLASSES["single"])
    multi = I.select_substrates(cat, 30, beams=I.SUBSTRATE_CLASSES["multibeam"])
    assert (single.n_beams <= 2).all() and len(single) == 30
    assert (multi.n_beams >= 32).all() and len(multi) == 30
    assert single.row.is_monotonic_increasing, "sorted for h5py read order"
    assert list(single.index) == list(range(30)), "index reset for iloc slicing"


def test_substrate_class_uses_the_weak_label_cuts():
    assert list(I.substrate_class([1, 2, 3, 31, 32, 64])) == \
        ["single", "single", "mid", "mid", "multibeam", "multibeam"]


def test_substrate_selection_returns_empty_rather_than_the_wrong_class():
    """Better no substrates than substrates from a class you did not ask for."""
    cat = pd.DataFrame({"snr": np.full(30, 6.0), "n_beams": np.full(30, 50),
                        "row": np.arange(30)})
    assert I.select_substrates(cat, 10,
                               beams=I.SUBSTRATE_CLASSES["single"]).empty


def test_zero_snr_injection_is_a_no_op():
    """
    The control must traverse the same code path as the injected rows, or an
    asymmetry there contaminates the one comparison the experiment rests on.
    `run()` routes the control through inject(snr=0); this pins that it is free.
    """
    c, m = _cube(), _meta()
    assert np.array_equal(I.inject(c, m, snr=0.0, drift_hz_s=0.2), c)


def test_extraction_crop_matches_the_pipeline():
    """
    Two independent literals with a comment saying they must agree is not a
    guarantee. If bluse-features changes its crop, every stamp feature computed
    here goes through a different window than the model was trained on, and the
    offset would present as an injection effect.
    """
    import argparse

    from bluse import track_b_features as B

    p = argparse.ArgumentParser()
    B.build_parser(p) if hasattr(B, "build_parser") else None
    src = open(B.__file__).read()
    marker = '"--crop", type=int, default='
    assert marker in src, "the CLI's crop default moved; update this test"
    default = int(src.split(marker)[1].split(",")[0])
    assert I.EXTRACTION_CROP == default, (I.EXTRACTION_CROP, default)


def test_fluctuating_injection_stops_the_ridge_being_unnaturally_smooth():
    """
    The defect review found: a constant-in-time ridge raises the mean without
    raising the standard deviation, so f10_timeseries_std FALLS with injected
    brightness -- the stamp becomes smoother than pure noise, which no real
    detection does.

    Measured on real sband_short substrates: smooth 0.0347 -> 0.0288 across the
    grid, Gamma-fluctuating 0.0347 -> 0.0348. Pinned here on synthetic cubes so
    it runs anywhere.
    """
    c, m = _cube(n=60, seed=3), _meta(n=60)
    smooth = I.inject(c, m, snr=150, drift_hz_s=0.0, fluctuate=False)
    rough = I.inject(c, m, snr=150, drift_hz_s=0.0, fluctuate=True, seed=1)

    def f10(cube):
        return np.nanmedian(I.stamp_features(cube, m, crop_channels=60)
                            ["f10_timeseries_std"])

    base = f10(c)
    assert f10(smooth) < base, "the smooth ridge should flatten the timeseries"
    assert f10(rough) > f10(smooth), "fluctuation should restore the scatter"


def test_fluctuation_is_reproducible_and_preserves_the_pad():
    c, m = _cube(), _meta()
    a = I.inject(c, m, snr=50, drift_hz_s=0.1, fluctuate=True, seed=7)
    b = I.inject(c, m, snr=50, drift_hz_s=0.1, fluctuate=True, seed=7)
    assert np.array_equal(a, b)
    assert (a[:, :, :120 - 80] == F.PAD_VALUE).all()


def test_effective_dof_recovers_a_known_gamma_shape():
    rng = np.random.default_rng(0)
    for k in (5.0, 30.0):
        a = rng.gamma(shape=k, scale=1.0 / k, size=(4, 40, 120))
        got = np.median(I.effective_dof(a))
        assert abs(got - k) / k < 0.15, (k, got)


def test_threshold_sweep_reports_the_price_of_retention():
    """
    A retention curve alone cannot choose a threshold -- keeping everything
    keeps every false positive too. The first version of this experiment
    reported only the keep side, which is how "at 0.5 the best case is 65%"
    got published as if it were a trade-off.
    """
    import pandas as pd

    inj = pd.DataFrame({"rfi_score": np.linspace(0, 1, 1000)})
    real = np.linspace(0, 1, 500)
    sw = I.threshold_sweep(inj, real, thresholds=[0.1, 0.5, 0.9])
    assert list(sw.columns) == ["threshold", "retained", "n_shortlist",
                                "shortlist_frac", "retained_per_admitted"]
    assert sw.retained.is_monotonic_increasing
    assert sw.n_shortlist.is_monotonic_increasing, "cost must rise with recall"
    assert np.isclose(sw.retained.iloc[0], 0.1, atol=0.01)


@pytest.mark.parametrize("cls", ["single", "multibeam"])
def test_committed_artefact_still_says_what_the_writeup_says(cls):
    """
    docs/injections-2026-09.md quotes these. The artefact is COMMITTED, so
    unlike the Track E numbers this can be pinned without a workspace -- if one
    of them moves, the document changed.
    """
    import os

    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "aug_2026_workshop", "scores", "injections.parquet")
    if not os.path.exists(path):
        pytest.skip("injections.parquet not in this checkout")
    df = pd.read_parquet(path)
    d = df[df.substrate_class == cls]
    ctl = d[d.kind == "control"]
    inj = d[d.kind == "injected"]
    expect = {"single": (0.428, 0.335), "multibeam": (0.001, 0.930)}[cls]
    assert abs((ctl.rfi_score < 0.1).mean() - expect[0]) < 0.01
    if cls == "single":
        bright = inj[(inj.injected_snr == 100) & (inj.drift_hz_s == 0.0)]
        assert (bright.rfi_score > 0.9).mean() > 0.85, "the 91% prune rate"
        best = inj.groupby(["injected_snr", "drift_hz_s"]).rfi_score.apply(
            lambda x: (x < 0.1).mean()).max()
        assert 0.55 < best < 0.62, best
