import numpy as np

from bluse import diagnostics as D
from tests.unit import fixtures


def _audit():
    X, cols, kinds = fixtures.synthetic_matrix(n=500, seed=0)
    return {r["col"]: r for r in
            D.audit(X, cols, scaling="robust", kinds=kinds, min_samples=8)}


def test_recovers_the_planted_tie():
    a = _audit()
    assert a["tie_col"]["max_tie_fraction"] == 0.30
    assert a["tie_col"]["tie_value"] == 0.0


def test_recovers_the_planted_clip():
    assert _audit()["clip_col"]["clip_frac"] == 0.05


def test_recovers_the_planted_level_count():
    assert _audit()["ordinal_col"]["n_distinct"] == 8


def test_clean_columns_are_not_flagged():
    a = _audit()
    assert a["plain_a"]["flags"] == []
    assert a["plain_b"]["flags"] == []


def test_tie_and_clip_columns_are_flagged():
    a = _audit()
    assert "tie" in a["tie_col"]["flags"]
    assert "clip" in a["clip_col"]["flags"]


def test_tie_threshold_does_not_apply_to_flag_kind():
    X, cols, kinds = fixtures.synthetic_matrix(n=500, seed=0)
    kinds = dict(kinds, tie_col="flag")
    rows = D.audit(X, cols, scaling="robust", kinds=kinds, min_samples=8)
    tie = [r for r in rows if r["col"] == "tie_col"][0]
    assert "tie" not in tie["flags"]


def test_shares_sum_to_one():
    assert abs(sum(r["share_global"] for r in _audit().values()) - 1.0) < 1e-9


def test_robust_scaling_gives_unit_scaled_iqr():
    assert abs(_audit()["plain_a"]["iqr_scaled"] - 1.0) < 1e-9


def _globally_loud_locally_ordinary(n=6000, d=6, seed=0):
    """
    A matrix where share_global and share_knn give DIFFERENT verdicts on
    column 0 -- structurally the x03_channel_offset story.

    Column 0 is a tight clump with a 10% heavy tail. Robust scaling divides by
    the clump's small IQR, so the tail dominates random-pair distances (global
    share ~50%, i.e. share-high), while any point's near neighbours sit inside
    the clump and agree in that coordinate, leaving its local share ordinary.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    tail = rng.random(n) < 0.10
    c0 = rng.normal(0, 0.02, n)
    c0[tail] = rng.normal(0, 10.0, tail.sum())
    X[:, 0] = c0
    return X, [f"c{i}" for i in range(d)]


def test_share_flags_key_on_the_knn_statistic():
    """
    P0-3. share_knn is now PRIMARY and share_global secondary, per
    implementation review section 4: HDBSCAN responds to core distances and
    mutual reachability, both local, so the local statistic describes what the
    clusterer actually sees.

    This fixture is built so the two rules DISAGREE -- the old global rule
    calls column 0 share-high, the calibrated local rule calls it ordinary.
    The flag must follow knn. That is the x03_channel_offset case in
    miniature: 24.3% global against 7.4% local on sband_short, indicted by the
    original review on the strength of the statistic that overstates it.
    """
    from bluse import diagnostics as D

    X, cols = _globally_loud_locally_ordinary()
    rows = {r["col"]: r for r in D.audit(X, cols, scaling="robust")}
    a = rows["c0"]
    eq = a["equal_share"]
    assert a["share_global"] > D.SHARE_HIGH * eq     # old rule: share-high
    assert a["share_knn"] <= D.SHARE_HIGH * eq       # new rule: not high
    assert "share-high" not in a["flags"]


def test_share_disagree_flags_a_misleading_global_share():
    """
    The new flag, carrying the finding this audit produced: where the two
    shares disagree badly, the global number misleads. Measured across all
    seven per-file parquets a 2.5x ratio marks a mean of 2.3 columns of 15 --
    x03_channel_offset (3.3x on sband_short) and f02_abs_drift (9.8x on
    uhf_short) among them.
    """
    from bluse import diagnostics as D

    X, cols = _globally_loud_locally_ordinary()
    rows = {r["col"]: r for r in D.audit(X, cols, scaling="robust")}
    assert (rows["c0"]["share_global"] / rows["c0"]["share_knn"]
            >= D.SHARE_DISAGREE)
    assert "share-disagree" in rows["c0"]["flags"]
    assert "share-disagree" not in rows["c5"]["flags"]


def test_share_flag_thresholds_are_named_constants():
    """They are calibrated numbers, not magic -- see the P0-3 write-up."""
    from bluse import diagnostics as D

    assert D.SHARE_HIGH == 2.0
    assert D.SHARE_LOW == 0.5
    assert D.SHARE_DISAGREE == 2.5


def test_knn_sample_default_is_large_enough_to_stop_flag_flicker():
    """
    At the old default of 5,000 the worst per-column CV of share_knn is 0.064,
    and on sband_short f07_kurt_bw_corr sits at 13.0% against a 13.3%
    threshold -- close enough that it flipped verdict on roughly one audit
    seed in twelve. A diagnostic that changes with the audit seed misleads.
    At 20,000 the CV falls to 0.043 and the flicker is gone.
    """
    import inspect

    from bluse import diagnostics as D

    assert inspect.signature(D.audit).parameters["knn_sample"].default == 20_000
