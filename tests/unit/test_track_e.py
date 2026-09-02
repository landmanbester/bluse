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
