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
