import numpy as np
import pytest

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


# --- P1-5: contribution-equalising weights ---------------------------------

def test_closed_form_equalises_the_global_share():
    """w ∝ 1/sigma is the exact solution: share_global_j ∝ w_j^2 var_j."""
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4000, 4)) * np.array([1.0, 5.0, 0.2, 2.0])
    w, info = D.equalising_weights(Z, strategy="closed")
    s = D._shares(Z * w, np.random.default_rng(0))
    assert np.abs(s / 0.25 - 1).max() < 0.15
    assert info["strategy"] == "closed"


def test_boolean_and_flag_columns_keep_weight_one():
    """
    Spec section 2.2. A low-variance indicator draws a huge equalising weight --
    a zero-drift boolean measured 0.33% k-NN share, which would have drawn
    10.256, twice the constant measured to destroy eom.
    """
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.normal(0, 3, 3000), rng.normal(0, 1, 3000),
                         (rng.random(3000) < 0.27).astype(float)])
    cols = ["a", "b", "is_x"]
    w, info = D.equalising_weights(Z, columns=cols, kinds={"is_x": "boolean"})
    assert w[2] == 1.0
    assert info["skipped"] == ["is_x"]


def test_flag_columns_are_frozen_too():
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.normal(0, 3, 2000), rng.normal(0, 0.2, 2000)])
    w, info = D.equalising_weights(Z, columns=["a", "b_saturated"],
                                   kinds={"b_saturated": "flag"})
    assert w[1] == 1.0
    assert info["skipped"] == ["b_saturated"]


def test_weights_are_deterministic():
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3000, 5)) * np.array([1.0, 4.0, 0.3, 2.0, 1.0])
    a, _ = D.equalising_weights(Z, strategy="closed")
    b, _ = D.equalising_weights(Z, strategy="closed")
    assert np.array_equal(a, b)


def test_iterative_strategy_respects_its_cap():
    """
    The undamped k-NN fixed point does NOT converge -- the k-NN graph is itself
    a function of w, so the map is not a contraction. Measured: weights run
    from 0.500-3.047 after one iteration to 0.044-10.347 after eight while the
    share is still 0.33 off equal. The cap is what makes it usable at all.
    """
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3000, 5)) * np.array([1.0, 8.0, 0.2, 3.0, 1.0])
    w, info = D.equalising_weights(Z, strategy="knn", iters=4, cap=2.0)
    assert w.max() <= 2.0 + 1e-9
    assert w.min() >= 1 / 2.0 - 1e-9
    assert len(info["dev_trace"]) == 4


def test_iterative_strategy_records_its_trajectory():
    """
    Spec/plan Task 2 step 2b: the endpoint alone cannot separate "slow" from
    "the target is unattainable". A multiplicative update toward an unreachable
    target diverges monotonically; only the trajectory tells them apart.
    """
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(2000, 4)) * np.array([1.0, 6.0, 0.3, 2.0])
    _, info = D.equalising_weights(Z, strategy="knn", iters=3, knn_sample=1500)
    assert len(info["dev_trace"]) == 3
    assert all(np.isfinite(info["dev_trace"]))


def test_extreme_weight_spread_is_flagged():
    """
    The sd > 1e-12 floor only catches exactly-constant columns. A column with
    sigma = 0.001 draws a huge weight and passes silently, and the failure then
    presents as "clustering got worse after I added a feature", which is close
    to undiagnosable. 2.0 is the measured scale to reason from: it is where eom
    breaks.
    """
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(2000, 8))
    Z[:, 7] *= 0.001
    w, info = D.equalising_weights(Z, strategy="closed")
    assert info["spread_warning"]
    assert w.max() <= D.EQUALISE_MAX_WEIGHT + 1e-9


def test_degenerate_columns_do_not_produce_infinite_weights():
    from bluse import diagnostics as D

    Z = np.random.default_rng(0).normal(size=(500, 3))
    Z[:, 1] = 4.0                                    # zero variance
    w, _ = D.equalising_weights(Z, strategy="closed")
    assert np.isfinite(w).all()
    assert w[1] == 1.0


def test_with_info_false_skips_the_expensive_diagnostic():
    """
    _shares_knn costs 1,311 ms on sband_short at knn_sample=20,000 against
    1.7 ms for the weights. scale() is called per run, per rail render and once
    per stability seed, so the info block must be opt-in.
    """
    from bluse import diagnostics as D

    Z = np.random.default_rng(0).normal(size=(3000, 4)) * [1, 5, 0.3, 2]
    w, info = D.equalising_weights(Z, strategy="closed", with_info=False)
    assert "max_dev_knn" not in info
    assert info["weight_max"] == pytest.approx(w.max())
    full = D.equalising_weights(Z, strategy="closed", with_info=True)[1]
    assert "max_dev_knn" in full and "max_dev_global" in full


def test_equalising_default_is_the_measured_winner():
    """
    Decision pin. The default strategy was chosen by the experiment in
    docs/equalising-scaling-experiment-2026-09.md.

    The k-NN strategy scored 0.83 family ARI at 36 families against the closed
    form's 0.51, and was still rejected -- not on score, but because it fails a
    precondition. Its target is unattainable for a column with a large tie
    fraction, it concentrates 96% of the squared weight into three of fifteen
    columns (so it is not "equalising" anything), and a three-line hand-built
    f02+f01 weighting beats it. If this test fails, read that write-up rather
    than editing the constant.
    """
    from bluse import diagnostics as D

    assert D.EQUALISE_STRATEGY == "closed"
    assert D.EQUALISE_ITERS == 0
    assert D.EQUALISE_CAP is None


def test_the_shipped_strategy_actually_equalises():
    """
    The property the k-NN strategy failed: the CONTRIBUTIONS must end up
    spread across the columns, not concentrated.

    Measure it on the shares, not on the weights. The first version of this
    test used the participation ratio of w**2 and failed, correctly: to
    equalise columns whose spreads differ 25x you NEED weights that differ 25x,
    so a low participation ratio of the weights is the right behaviour, not a
    defect. The n_eff figures quoted in the write-up (10.77 for the closed
    form, 2.22 for the k-NN fit) are comparable to each other only because they
    are computed on the same real matrix, where the spread of sigma is modest.
    """
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4000, 6)) * np.array([1.0, 5.0, 0.2, 2.0, 0.5, 3.0])
    w, _ = D.equalising_weights(Z, strategy=D.EQUALISE_STRATEGY,
                                iters=D.EQUALISE_ITERS, cap=D.EQUALISE_CAP,
                                with_info=False)
    share = D._shares(Z * w, np.random.default_rng(0))
    n_eff = 1 / np.sum((share / share.sum()) ** 2)
    assert n_eff > 0.9 * len(w), n_eff


# --- P1-5 Task 3: the robust-equalised scaling mode -------------------------

def test_robust_equalised_is_robust_times_the_weights():
    from bluse import diagnostics as D

    X = np.random.default_rng(0).normal(size=(2000, 4)) * [1, 6, 0.3, 2]
    base = D.scale(X, "robust")
    w, _ = D.equalising_weights(base, strategy=D.EQUALISE_STRATEGY,
                                iters=D.EQUALISE_ITERS, cap=D.EQUALISE_CAP,
                                with_info=False)
    assert np.allclose(D.scale(X, "robust-equalised"), base * w)


def test_equalised_mode_still_reports_clipping():
    """
    audit()'s clip_frac keys on scaling == "robust". The equalised mode has a
    robust base and clips exactly as much, so omitting it would silently report
    0.0 -- the same false-negative shape as the --scaling none bug.

    A WIDENED GUARD IS NOT ENOUGH, and this is the trap. clip_frac is measured
    as (|Z| >= CLIP).mean() on whatever scale() returned, and the equalised
    mode returns base * w. A value sitting exactly on the clip at +/-5 in the
    base is no longer at +/-5 after weighting. Measured on this very fixture:
    clip_frac on the base is 0.050, the weight for column a is 0.661, so the
    clipped rows land at 3.31 and the threshold test misses every one of them.
    clip_frac must be computed on the ROBUST BASE, before weights.
    """
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X[:200, 0] = 500.0                                # forces the clip
    rows = {r["col"]: r for r in
            D.audit(X, list("abc"), scaling="robust-equalised")}
    assert rows["a"]["clip_frac"] > 0.01
    assert "clip" in rows["a"]["flags"]


def test_clip_frac_matches_the_unweighted_robust_mode():
    """The clip is a property of the base transform, so both modes must agree."""
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X[:200, 0] = 500.0
    a = {r["col"]: r for r in D.audit(X, list("abc"), scaling="robust")}
    b = {r["col"]: r for r in
         D.audit(X, list("abc"), scaling="robust-equalised")}
    for c in "abc":
        assert a[c]["clip_frac"] == pytest.approx(b[c]["clip_frac"])


def test_scale_honours_column_kinds():
    from bluse import diagnostics as D

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0, 3, 2000), rng.normal(0, 1, 2000),
                         (rng.random(2000) < 0.27).astype(float)])
    cols = ["a", "b", "is_x"]
    Z = D.scale(X, "robust-equalised", kinds={"is_x": "boolean"}, columns=cols)
    base = D.scale(X, "robust")
    assert np.allclose(Z[:, 2], base[:, 2])           # weight exactly 1.0


def test_unknown_scaling_raises_instead_of_silently_meaning_none():
    """A typo'd mode used to fall through to `return X`, i.e. silently 'none'."""
    from bluse import diagnostics as D

    with pytest.raises(ValueError, match="scaling"):
        D.scale(np.zeros((10, 2)), "robsut")
    assert D.scale(np.ones((10, 2)), "none").shape == (10, 2)
