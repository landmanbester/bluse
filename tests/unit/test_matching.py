import numpy as np

from bluse import matching
from tests.unit import fixtures


def test_gap_rule_recovers_the_planted_family_structure():
    """
    rule="gap" gets the right answer on genuinely separated families, which is
    what this fixture has. The DEFAULT is rule="pct", because real data is not
    separated: on sband_short the gap rule is dominated by the root merges and
    collapses 2,162 clusters into 4 families spanning the whole band.

    READ WITH test_gap_rule_is_not_evidence_of_structure_sensitivity BELOW.
    This test passing is WEAK evidence: the gap rule returns 2-4 families on
    almost any input, so landing on 3 here is partly luck. It is kept because
    getting the planted answer wrong would still be a regression.
    """
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, info = matching.match(labels, X, rule="gap")
    assert info["n_clusters"] == 9
    assert info["n_families"] == 3


def test_gap_rule_is_not_evidence_of_structure_sensitivity():
    """
    Pin the negative result from the P0-1 experiment, so nobody reads the test
    above as proof the gap rule detects structure. It does not: because the
    root merges of any dendrogram dominate the gap statistic, the rule returns
    a handful of families on STRUCTURELESS data of the same shape, and hits
    exactly 3 about a quarter of the time by coincidence.

    Restricting the gap search to merges below a percentile bound (the fix
    proposed in the implementation review) does not repair this -- see
    docs/matching-cut-experiment-2026-09.md. It was measured and rejected.
    """
    labels, _ = fixtures.synthetic_centroid_space(seed=0)
    got = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        centres = rng.normal(0, 5, (9, 3))          # no family structure
        X = np.vstack([c + rng.normal(0, 0.3, (40, 3)) for c in centres])
        got.append(matching.match(labels, X, rule="gap")[1]["n_families"])
    assert all(g <= 4 for g in got), got
    assert 3 in got, got


def test_a_distance_cut_is_exactly_a_choice_of_family_count():
    """
    The finding that decides what any cut rule can possibly do: every
    horizontal cut of a fixed Ward tree is uniquely determined by the number of
    families it leaves. Verified here on the fixture and, in the P0-1
    experiment, over 1,000 thresholds on the real sband_short trees (eom and
    leaf) with zero exceptions.

    So a cut rule cannot select a BETTER partition, only a point on a fixed
    nested chain. "Structure-sensitive cut" can therefore only mean "picks a
    count that moves with the data" -- which is the property the gap rule was
    measured against, and failed.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import adjusted_rand_score

    labels, X = fixtures.synthetic_centroid_space(seed=0)
    ids, C = matching.centroids(labels, X)
    Z = linkage(C, method="ward")
    for t in np.linspace(Z[:, 2].min(), Z[:, 2].max(), 50):
        by_distance = fcluster(Z, t=t, criterion="distance")
        n = len(np.unique(by_distance))
        by_count = fcluster(Z, t=n, criterion="maxclust")
        assert adjusted_rand_score(by_distance, by_count) == 1.0


def test_n_families_returns_exactly_that_many():
    """
    The count is the honest parameter, given the identity above. This is the
    interface derive_cut_pct's docstring has been telling readers to prefer
    since it was written -- it did not exist until the P0-1 experiment showed
    that a count is all a cut rule ever chooses.
    """
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    for n in (2, 3, 5, 9):
        fam, info = matching.match(labels, X, n_families=n)
        assert info["n_families"] == n, (n, info["n_families"])
        assert len(np.unique(fam[fam >= 0])) == n


def test_n_families_recovers_the_planted_families():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, _ = matching.match(labels, X, n_families=3)
    fam_of = {c: int(np.unique(fam[labels == c])[0]) for c in range(9)}
    assert fam_of[0] == fam_of[1] == fam_of[2]
    assert fam_of[3] == fam_of[4] == fam_of[5]
    assert fam_of[0] != fam_of[3] != fam_of[6]


def test_n_families_beats_pct_which_ignores_it():
    """n_families outranks the pct dial; an explicit cut still outranks both."""
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    assert matching.match(labels, X, n_families=3, pct=90)[1]["n_families"] == 3
    assert matching.match(labels, X, n_families=3,
                          cut=1000.0)[1]["n_families"] == 1


def test_n_families_is_clamped_to_what_the_tree_can_give():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    assert matching.match(labels, X, n_families=99)[1]["n_families"] == 9
    assert matching.match(labels, X, n_families=1)[1]["n_families"] == 2


def test_families_group_the_right_clusters():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, _ = matching.match(labels, X, rule="gap")
    fam_of = {c: int(np.unique(fam[labels == c])[0]) for c in range(9)}
    assert fam_of[0] == fam_of[1] == fam_of[2]
    assert fam_of[3] == fam_of[4] == fam_of[5]
    assert fam_of[0] != fam_of[3]
    assert fam_of[0] != fam_of[6]


def test_noise_stays_noise():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    labels = labels.copy()
    labels[:10] = -1
    fam, _ = matching.match(labels, X)
    assert (fam[:10] == -1).all()


def test_is_deterministic():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    a, _ = matching.match(labels, X)
    b, _ = matching.match(labels, X)
    assert np.array_equal(a, b)


def test_pct_rule_is_a_granularity_dial_not_a_structure_detector():
    """
    Pin the property that matters about the default rule: it returns about
    k(1 - pct/100) groups REGARDLESS of structure.

    The previous version of this test asserted only that the family count is
    monotone in pct, which holds by construction on any input including pure
    noise -- a test that cannot fail. This one asserts the actual arithmetic,
    on data with NO family structure, so it fails if the rule ever stops being
    a granularity dial (which would be an improvement, and should be noticed).
    """
    rng = np.random.default_rng(0)
    k, per = 60, 10
    labels = np.repeat(np.arange(k), per).astype(np.int32)
    X = rng.normal(size=(k * per, 6))          # structureless
    for pct in (25, 50, 75):
        n_fam = matching.match(labels, X, pct=pct)[1]["n_families"]
        assert abs(n_fam - round(k * (1 - pct / 100))) <= 2, (pct, n_fam)


def test_cut_source_is_reported():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    assert matching.match(labels, X)[1]["cut_source"] == "merge-height p50"
    assert matching.match(labels, X, rule="gap")[1]["cut_source"] == \
        "merge-height gap"
    assert matching.match(labels, X, cut=5.0)[1]["cut_source"] == "explicit"
    assert matching.match(labels, X, n_families=3)[1]["cut_source"] == \
        "n_families=3"


def test_explicit_cut_overrides_the_derived_one():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    _, info = matching.match(labels, X, cut=1000.0)
    assert info["cut"] == 1000.0
    assert info["n_families"] == 1


def test_single_cluster_is_a_single_family():
    labels = np.zeros(50, dtype=np.int32)
    X = np.random.default_rng(0).normal(size=(50, 3))
    fam, info = matching.match(labels, X)
    assert info["n_families"] == 1
    assert (fam == 0).all()


def test_info_always_carries_cut_source():
    """
    The CLI indexes info['cut_source'] unconditionally, so every return path
    must set it -- including the all-noise and single-cluster early returns,
    which previously omitted it and raised KeyError on a valid result.
    """
    X = np.random.default_rng(0).normal(size=(50, 3))
    for labels in (np.full(50, -1, dtype=np.int32),
                   np.zeros(50, dtype=np.int32)):
        _, info = matching.match(labels, X)
        assert "cut_source" in info


def test_refuses_a_centroid_count_that_would_exhaust_memory():
    """Exact Ward is O(k^2) in MEMORY; 78k centroids is a 24 GB matrix."""
    import pytest
    n = matching.MAX_CENTROIDS + 10
    labels = np.arange(n, dtype=np.int32)
    X = np.random.default_rng(0).normal(size=(n, 2))
    with pytest.raises(MemoryError, match="condensed"):
        matching.match(labels, X)
