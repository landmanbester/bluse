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


def test_default_pct_rule_merges_monotonically():
    """A higher percentile must never yield more families."""
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    counts = [matching.match(labels, X, pct=p)[1]["n_families"]
              for p in (10, 25, 50, 75, 90)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] >= 1


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
