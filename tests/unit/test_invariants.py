"""
The five invariants. Three of the defects in AGENTS.md gotcha 9 presented
identically -- "the bench looks insensitive to every knob except
min_cluster_size" -- and these separate them.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from bluse import diagnostics as D
from bluse.bench import app
from tests.unit import fixtures


def _matrix(n=900, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0., 0.], [6., 0.], [0., 6.]])
    X = np.vstack([c + rng.normal(0, 0.6, (n // 3, 2)) for c in centres])
    # a wildly unequal third column, so scaling has something to equalise
    return np.column_stack([X, rng.normal(0, 40.0, len(X))])


class _DS:
    """Minimal stand-in for bench.app.Dataset -- cluster() uses .raw/.columns."""

    def __init__(self, X):
        self.raw = X
        self.columns = [f"c{i}_n" for i in range(X.shape[1])]


def test_invariant_1_cluster_ids_are_globally_unique():
    ds = _DS(_matrix())
    labels, _, origin, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                       4, 8, 3, 200, 0)
    ids = np.unique(labels[labels >= 0])
    assert len(ids) == len(origin)
    assert {int(i) for i in ids} == set(origin)


def test_invariant_2_changing_scaling_changes_the_labels():
    """
    THE SEED IS PINNED, and that is the whole test.

    At a free seed, two runs of an IDENTICAL configuration score ARI 0.024, so
    `ARI < 1.0` passes on shuffle noise even if scale() were stubbed to return
    its input -- the exact bug class this exists to catch. At a fixed seed a
    genuine no-op scores exactly 1.0 and the assertion bites.
    """
    ds = _DS(_matrix())
    a, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs", 4, 8, 3, 200, 0)
    b, _, _, _ = app.cluster(ds, ds.columns, "none", "epochs", 4, 8, 3, 200, 0)
    assert adjusted_rand_score(a, b) < 1.0


def test_invariant_3_reported_count_matches_the_labels():
    ds = _DS(_matrix())
    labels, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                  4, 8, 3, 200, 0)
    assert len(app.summarise_basic(labels)) == \
        len(np.unique(labels[labels >= 0]))


def test_invariant_4_no_continuous_column_ties_above_half():
    X, cols, kinds = fixtures.synthetic_matrix(n=500, seed=0)
    for r in D.audit(X, cols, scaling="robust", kinds=kinds):
        if r["kind"] in ("continuous", "ordinal"):
            assert r["max_tie_fraction"] <= 0.5, r["col"]


def test_invariant_5_restricted_ari_is_defined_and_in_range():
    """
    Smoke test, deliberately wide.

    ari_restricted has meaningful variance of its own at the values we see, so
    a band recorded from a single draw would be flaky in the way that trains
    people to ignore a suite. The real bands live in tests/workspace.
    """
    from bluse import metrics as M
    ds = _DS(_matrix())

    def run_fn(seed):
        lab, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                   4, 8, 3, 200, seed)
        return lab

    s = M.stability(run_fn, seeds=(0, 1, 2))
    assert 0.0 <= s["ari_restricted"] <= 1.0
    assert s["k_min"] >= 1


def test_leaf_produces_at_least_as_many_clusters_as_eom():
    ds = _DS(_matrix())
    e, _, _, _ = app.cluster(ds, ds.columns, "robust", "single",
                             4, 8, 1, 3000, 0, method="eom")
    lf, _, _, _ = app.cluster(ds, ds.columns, "robust", "single",
                              4, 8, 1, 3000, 0, method="leaf")
    assert len(np.unique(lf[lf >= 0])) >= len(np.unique(e[e >= 0]))


def test_epoch_trace_is_returned():
    ds = _DS(_matrix())
    _, _, _, trace = app.cluster(ds, ds.columns, "robust", "epochs",
                                 4, 8, 3, 200, 0)
    assert len(trace) >= 1
    assert all(isinstance(t, int) for t in trace)


def test_superseded_prefers_the_file_with_more_usable_rows():
    """
    The rule that replaced a name-based one, after the BLUSE team re-delivered
    lband_short.h5 and uhf_long.h5 without the corrupt regions (2026-09-02).

    The old rule preferred `<name>_clean` BY NAME. That was right while the
    original was corrupt and wrong the moment it was fixed: the repaired
    lband_short.h5 yields 866,002 usable rows, so a stale
    lband_short_clean_features.parquet left in the features directory would
    have silently discarded 402,377 good rows -- the exact failure the rule
    exists to prevent, with the sign flipped.

    Preferring whichever file carries more USABLE rows makes the same decision
    in the old world (clean 463,625 against the corrupt original's 462,002
    stamp-bearing rows) and the right one in the new.
    """
    import pandas as pd

    from bluse.track_b_features import drop_superseded

    # New world: the repaired original is complete, the _clean copy is stale.
    df = pd.DataFrame({
        "file": ["lband_short"] * 900 + ["lband_short_clean"] * 400,
        "feature_ok": [True] * 900 + [True] * 400,
    })
    kept = set(drop_superseded(df.copy())["file"].unique())
    assert kept == {"lband_short"}, kept

    # Old world: the original is mostly stamp-less, so the _clean copy wins.
    df = pd.DataFrame({
        "file": ["lband_short"] * 900 + ["lband_short_clean"] * 400,
        "feature_ok": [True] * 300 + [False] * 600 + [True] * 400,
    })
    kept = set(drop_superseded(df.copy())["file"].unique())
    assert kept == {"lband_short_clean"}, kept


def test_superseded_is_a_no_op_without_a_clean_twin():
    import pandas as pd

    from bluse.track_b_features import drop_superseded

    df = pd.DataFrame({"file": ["sband_short"] * 10, "feature_ok": [True] * 10})
    assert len(drop_superseded(df.copy())) == 10
