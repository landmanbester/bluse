"""
Track E scorer.

The two properties worth a test are not "does it predict well" -- that is
measured on the real data in tests/workspace -- but the two ways this result
could be silently worthless: the model seeing the label, and a row being scored
by a model that saw its observation.
"""

import numpy as np
import pytest

from bluse import track_e_score as E
from tests.unit import fixtures


def test_no_feature_set_can_see_beam_multiplicity():
    """
    `n_beams` IS the label -- weak_label is a threshold on it. A feature set
    reaching it would score ~1.0 and mean nothing at all.

    Asserted rather than avoided by care, because the same class of defect has
    landed in this repo twice: a scaling mode that silently did nothing, and a
    metrics record that disagreed with the matrix actually clustered. Both were
    "we were careful" until someone measured.
    """
    for name in E.FEATURE_SETS:
        cols = set(E.feature_columns(name))
        assert not (cols & E.LABEL_COLUMNS), f"{name} leaks the label"
        assert not any("beam" in c for c in cols), f"{name} smells of the label"


def test_feature_sets_come_from_the_registry_not_a_literal():
    """A hard-coded column list goes stale the first time a feature is added."""
    from bluse import features as F

    stamp = {c + "_n" for c in F.all_columns(kind="stamp")
             if not c.endswith("_saturated")}
    assert set(E.FEATURE_SETS["stamp"]) == stamp
    assert set(E.FEATURE_SETS["all"]) == stamp | set(E.FEATURE_SETS["meta"])


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError, match="unknown feature set"):
        E.feature_columns("morphology")


def test_every_labelled_row_is_scored_out_of_fold():
    """
    Acceptance criterion 5, and the property the whole result rests on.

    Cross-validation here is the delivery mechanism, not the audit: the score
    that SHIPS for a labelled row is its out-of-fold score. If a group could
    appear in both the training and scoring side of one fold, every number in
    the write-up would be optimistic and nothing would flag it.
    """
    df = fixtures.synthetic_weak_labelled(n=4000, groups=12, seed=0)
    score, fold, info = E.fit_score(df, features="all", n_splits=4)

    lab = df.weak_label.to_numpy() >= 0
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
    assert (fold[lab] >= 0).all(), "a labelled row was left out of every fold"
    assert (fold[~lab] == -1).all(), "an unlabelled row was given a fold"

    g = df.group_id.to_numpy()
    for k in range(4):
        held = set(g[lab & (fold == k)])
        elsewhere = set(g[lab & (fold != k)])
        assert not (held & elsewhere), f"group split leaked in fold {k}"
    assert info["n_scored_by_fold_mean"] == int((~lab).sum())


def test_ambiguous_rows_are_scored_by_the_fold_mean():
    """
    The 410,691 real rows in 3-31 beams are in NO training fold, so their score
    is the mean of the fold models. That is what makes the score defined for
    exactly the population the spatial filter abstains on -- the reason Track E
    exists. A run that left them NaN would look fine on every labelled metric.
    """
    df = fixtures.synthetic_weak_labelled(n=3000, groups=10, seed=2)
    amb = df.weak_label.to_numpy() < 0
    score, fold, _ = E.fit_score(df, features="all", n_splits=3)
    assert amb.sum() > 0
    assert np.isfinite(score[amb]).all()
    # Planted halfway between the blobs, so they must land between the classes.
    lo = score[df.weak_label.to_numpy() == 0].mean()
    hi = score[df.weak_label.to_numpy() == 1].mean()
    assert lo < score[amb].mean() < hi


def test_score_is_deterministic_and_tolerates_a_partly_nan_column():
    """
    `x01_drift_residual` is NaN on 29.6% of real rows by construction -- the
    trajectory residual is undefined at zero drift, and 26-47% of hits are
    zero-drift. HistGradientBoosting handles NaN natively.

    Pinned because swapping in a model that does not would either crash or
    silently drop a third of the survey, and the AUC would barely move.
    """
    df = fixtures.synthetic_weak_labelled(n=3000, groups=10, seed=1)
    col = df["x01_drift_residual_n"].to_numpy().copy()
    col[::3] = np.nan                      # 33%, close to the real fraction
    df["x01_drift_residual_n"] = col
    a, _, ia = E.fit_score(df, features="all", n_splits=3, seed=7)
    b, _, _ = E.fit_score(df, features="all", n_splits=3, seed=7)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert ia["dropped_columns"] == []


def test_an_all_nan_column_is_dropped_by_name_not_crashed_on(capsys):
    """
    Regression for a real failure mode with a useless error message.

    A column that is non-finite everywhere gives the histogram binner zero bins,
    and sklearn surfaces that from inside joblib as "window shape cannot be
    larger than input array shape" -- naming neither the column nor the cause.
    Reachable in practice: f08_turning_bw_hz is unresolved for ~72% of hits, so
    a small enough subset can be all-NaN in it.
    """
    df = fixtures.synthetic_weak_labelled(n=3000, groups=10, seed=1)
    df["x01_drift_residual_n"] = np.nan
    score, _, info = E.fit_score(df, features="all", n_splits=3, seed=7)
    assert np.isfinite(score).all()
    assert info["dropped_columns"] == ["x01_drift_residual_n"]
    assert "x01_drift_residual_n" not in info["columns"]
    assert "x01_drift_residual_n" in capsys.readouterr().out


def test_prefiltered_files_are_excluded_from_training_but_still_scored():
    """
    mk_sample_hits carries zero weak_label == 1 rows, so it can only contribute
    a biased block of negatives. It is still part of the survey and must still
    receive a score.
    """
    df = fixtures.synthetic_weak_labelled(n=3000, groups=10, seed=3)
    df.loc[df.index[:500], "file"] = "mk_sample_hits"
    score, fold, info = E.fit_score(df, features="all", n_splits=3,
                                    exclude_mk=True)
    mk = (df.file == "mk_sample_hits").to_numpy()
    assert np.isfinite(score[mk]).all(), "excluded rows must still be scored"
    assert (fold[mk] == -1).all(), "excluded rows must not carry a training fold"
    assert info["n_train"] == int(((df.weak_label >= 0) & ~mk).sum())


def test_too_few_groups_raises_rather_than_silently_reducing_folds():
    df = fixtures.synthetic_weak_labelled(n=600, groups=3, seed=0)
    with pytest.raises(ValueError, match="fewer than n_splits"):
        E.fit_score(df, features="all", n_splits=5)


def test_the_feature_matrix_is_float64():
    """
    Every feature column in the parquet is float64, and every other science
    path in this package asks for float64 explicitly. This one used to narrow
    to float32 -- the only place in the repo that did.

    Narrowing is not catastrophic here (HistGradientBoosting bins to 256 levels
    and upcasts internally, so the AUC moved by 4e-4), but it perturbs which
    side of a bin edge a value falls on, and the per-hit verdicts move with it:
    the contrarian count went 2,972 -> 3,303 on the real data. Pinned so the
    matrix cannot quietly narrow again.
    """
    import numpy as np

    assert np.dtype(E.FEATURE_DTYPE) == np.float64
    df = fixtures.synthetic_weak_labelled(n=1200, groups=6, seed=0)
    X = df[E.feature_columns("all")].to_numpy(E.FEATURE_DTYPE)
    assert X.dtype == np.float64


def test_the_score_is_averaged_over_seeds_by_default():
    """
    Not a refinement. HistGradientBoosting draws its 256 bin edges from a random
    200,000-row subsample, so a single seed's per-hit verdict is substantially
    churn -- measured on the real data, two seeds share only 93% of the
    shortlist, while f32-vs-f64 at one seed shares 95%. The seed perturbs the
    model MORE than the dtype does.

    Averaging must not weaken the no-leak property: every seed uses the same
    GroupKFold split, so a row's score stays the mean of models that never saw
    its observation.
    """
    import numpy as np

    df = fixtures.synthetic_weak_labelled(n=3000, groups=9, seed=0)
    one, f1, i1 = E.fit_score(df, features="all", n_splits=3, n_seeds=1)
    three, f3, i3 = E.fit_score(df, features="all", n_splits=3, n_seeds=3)

    assert i1["n_seeds"] == 1 and i1["seeds"] == [0]
    assert i3["n_seeds"] == 3 and i3["seeds"] == [0, 1, 2]
    assert i3["dtype"] == "float64"
    assert np.array_equal(f1, f3), "averaging must not change the fold split"
    assert not np.array_equal(one, three), "averaging did nothing"
    assert np.isfinite(three).all() and ((three >= 0) & (three <= 1)).all()


def test_seed_averaging_is_the_mean_of_the_single_seed_runs():
    """The arithmetic, pinned: a three-seed score is the mean of seeds 0, 1, 2."""
    import numpy as np

    df = fixtures.synthetic_weak_labelled(n=2000, groups=8, seed=5)
    parts = [E.fit_score(df, features="all", n_splits=4, seed=s, n_seeds=1)[0]
             for s in (0, 1, 2)]
    avg, _, _ = E.fit_score(df, features="all", n_splits=4, seed=0, n_seeds=3)
    assert np.allclose(avg, np.mean(parts, axis=0))


def test_n_seeds_below_one_raises():
    df = fixtures.synthetic_weak_labelled(n=600, groups=6, seed=0)
    with pytest.raises(ValueError, match="at least 1"):
        E.fit_score(df, features="all", n_splits=3, n_seeds=0)


def test_predict_held_out_agrees_exactly_with_fit_score():
    """
    The no-leak property of the injection harness rests entirely on
    fold_models/predict_held_out, and PR #3's review pointed out that neither
    had a test.

    Agreement must be EXACT, not approximate: both average the same n_seeds
    models over the same GroupKFold split, so any difference means the two
    training loops have diverged -- which they can, because fold_models
    duplicates fit_score's loop without its all-NaN column drop.
    """
    import numpy as np

    df = fixtures.synthetic_weak_labelled(n=3000, groups=9, seed=4)
    score, fold, _ = E.fit_score(df, features="all", n_splits=3, n_seeds=2)
    models, fold_of_group = E.fold_models(df, features="all", n_splits=3,
                                          n_seeds=2)
    lab = df.weak_label.to_numpy() >= 0
    got, fold_used, n_fb = E.predict_held_out(
        models, fold_of_group, df.loc[lab, E.feature_columns("all")].to_numpy(),
        df.loc[lab, "group_id"].to_numpy())
    assert n_fb == 0
    assert np.array_equal(fold_used, fold[lab])
    assert np.allclose(got, score[lab], atol=0, rtol=0)


def test_no_group_is_scored_by_a_fold_that_trained_on_it():
    """
    The property the whole injection result depends on. An injected stamp sits
    on a real hit that is IN the training set, so scoring it with a model that
    saw its own substrate leaks the substrate's morphology in through a side
    door -- exactly what the group split exists to prevent.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    df = fixtures.synthetic_weak_labelled(n=3000, groups=9, seed=5)
    models, fold_of_group = E.fold_models(df, features="all", n_splits=3,
                                          n_seeds=1)
    y = df.weak_label.to_numpy()
    g = df.group_id.to_numpy()
    tr_idx = np.flatnonzero(y >= 0)
    trained_on = {k: set(g[tr_idx[tr]]) for k, (tr, _) in
                  enumerate(GroupKFold(3).split(tr_idx, y[tr_idx], g[tr_idx]))}
    for group, k in fold_of_group.items():
        assert group not in trained_on[k], (group, k)


def test_an_unmapped_group_is_flagged_rather_than_silently_scored():
    """
    A group absent from fold_of_group falls back to a model that MAY have seen
    it. That has to be attributable per row, not counted at run level -- a
    scalar count gives no way to exclude the affected rows from a headline.
    """
    import numpy as np

    df = fixtures.synthetic_weak_labelled(n=2000, groups=8, seed=6)
    models, fold_of_group = E.fold_models(df, features="all", n_splits=4,
                                          n_seeds=1)
    X = df[E.feature_columns("all")].to_numpy()[:5]
    groups = np.array(["obs000", "NOT_A_GROUP", "obs001", "NOT_A_GROUP", "obs002"])
    _, fold_used, n_fb = E.predict_held_out(models, fold_of_group, X, groups)
    assert n_fb == 2
    assert list(fold_used < 0) == [False, True, False, True, False]
