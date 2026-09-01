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
