import numpy as np

from bluse import matching
from tests.unit import fixtures


def test_gap_rule_recovers_the_planted_family_structure():
    """
    rule="gap" is the right rule for genuinely separated families, which is
    what this fixture has. The DEFAULT is rule="pct", because real data is not
    separated: on sband_short the gap rule is dominated by the root merges and
    collapses 2,162 clusters into 4 families spanning the whole band. Each rule
    is tested on the data it is for.
    """
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, info = matching.match(labels, X, rule="gap")
    assert info["n_clusters"] == 9
    assert info["n_families"] == 3


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
