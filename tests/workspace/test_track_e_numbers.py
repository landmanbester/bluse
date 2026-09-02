"""
Track E, measured on the real feature matrices, 2026-09-03.

These are the tests that stop the headline being a selection effect. A 0.99 AUC
for predicting beam multiplicity from stamp morphology has several boring
explanations, and each assertion below closes one. If one of them moves, the
claim in docs/track-e-2026-09.md changed -- re-measure and rewrite it rather
than widening the tolerance.

Gitignored data, so this cannot gate a commit. Every number states what it was
measured on.
"""

import os

import pandas as pd
import pytest

from bluse import paths
from bluse import track_e_score as E

pytestmark = pytest.mark.workspace

_CACHE = {}


def _report():
    if "r" in _CACHE:
        return _CACHE["r"]
    path = os.path.join(paths.features_dir(), "all_features.parquet")
    if not os.path.exists(path):
        pytest.skip("no all_features.parquet in this workspace")
    need = sorted(set(E.FEATURE_SETS["all"] + E.FLAG_COLUMNS + [
        "weak_label", "group_id", "file", "snr", "frequency", "driftRate",
        "coarseChannel", "n_beams"]))
    _CACHE["r"] = E.validate(pd.read_parquet(path, columns=need),
                             n_splits=5, seed=0, verbose=False)
    return _CACHE["r"]


def test_stamp_morphology_beats_track_a_s_entire_flag_set():
    """
    The headline. Twelve numbers computed from the stamp pixels against the six
    hand-built flags of the classical filter, on the same rows and the same
    folds. Measured 0.9887 against 0.9373.
    """
    a = _report()["ablation"]
    assert a["stamp"]["roc_auc"] >= 0.985
    assert a["stamp"]["roc_auc"] - a["flags"]["roc_auc"] > 0.04


def test_the_result_is_not_the_zero_drift_freebie():
    """
    x01_drift_residual is NaN exactly where drift is zero, and P(RFI | NaN) is
    0.996 -- a 29.6% slice of the data that is nearly free. Drop every
    zero-drift row and the result must survive. Measured 0.9903, i.e. it rises.
    """
    r = _report()["nonzero_drift"]
    assert r["stamp"] >= 0.985
    assert r["stamp"] > r["flags"]


def test_the_result_is_not_a_brightness_meter():
    """
    A signal must be bright to be DETECTED in 32 beams, so the label is partly
    a selection effect on SNR. Within an SNR decile it cannot be. Measured
    0.8956 n-weighted across deciles, against 0.9887 overall -- brightness
    contributes and does not carry it.
    """
    assert _report()["snr_stratified"]["weighted"] >= 0.85


def test_the_result_is_not_duplicate_inflation():
    """
    One emitter yields up to 64 near-duplicate rows, one per beam, which could
    inflate every row-wise metric. Collapse to one row per signal: 0.9890.
    Restrict to signals appearing exactly once: 0.9822.
    """
    r = _report()["signal_level"]
    assert r["median_per_signal"] >= 0.98
    assert r["singleton"] >= 0.97
    assert r["n_singletons"] > 100_000


def test_it_generalises_to_a_band_it_never_saw():
    """
    Hold out a whole band and train on the other two. This is the difference
    between general RFI morphology and a memorised catalogue of band-specific
    emitters. Measured 0.90-0.99 on all three.
    """
    for band, r in _report()["cross_band"].items():
        assert r["stamp"] >= 0.85, f"{band} failed to transfer"


def test_the_score_orders_beam_counts_it_was_never_trained_on():
    """
    The strongest single piece of evidence, and the one figure worth showing.

    Training sees only <=2 beams and >=32. Scored on the untrained 3-31 range,
    the model reproduces beam multiplicity monotonically across every bin. A
    fitted decision boundary has no reason to interpolate; something physical
    does.
    """
    r = _report()["n_beams_monotonicity"]
    assert r["monotone"], [b["mean_score"] for b in r["bins"]]
    untrained = [b for b in r["bins"] if not b["in_training"]]
    assert len(untrained) == 5
    assert sum(b["n"] for b in untrained) > 300_000


def test_the_prefiltered_file_is_flagged_as_out_of_distribution():
    """
    mk_sample_hits carries ~25 hits per observation against ~2,931 elsewhere, so
    its beam multiplicity tops out at 9 of up to 64 formed beams and it can
    contribute no positives at all. The model, trained on the other six files,
    scores 94.2% of its nominally-confined hits as RFI -- and is RIGHT to.

    8,116 hits appear in both mk_sample_hits and lband_long under the same id,
    frequency and beam; the same hit is counted in a mean of 1.87 beams there
    against 29.71 in lband_long, and of the 6,141 mk calls confined, lband_long
    puts 1,813 in >=32 beams. The labels are the artefact, not the score.

    Pinned so the number stays visible: it is the mechanism behind the caveat
    that matters operationally -- a hit list selected differently from the one
    this score was fitted on will not behave the same way, and neither will the
    spatial filter it learned from.
    """
    r = _report()["ood"]
    assert r["n_positive"] == 0
    assert r["frac_scored_rfi"] > 0.5
    assert r["auc_including_it"] < _report()["ablation"]["stamp"]["roc_auc"]
