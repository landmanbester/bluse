import numpy as np

from bluse import metrics as M
from tests.unit import fixtures


def test_narrow_frac_matches_the_planted_share():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    assert q["narrow_frac"] == 0.20
    assert q["narrow_clusters"] == 6


def test_narrow_frac_reported_at_both_thresholds():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    assert set(q["narrow_frac_at"]) == {0.1, 1.0}
    assert q["narrow_frac_at"][1.0] == 0.20
    assert q["narrow_frac_at"][0.1] == 0.0


def test_permuting_labels_destroys_the_narrow_share():
    """
    The null does its job: the planted narrow clusters are an arrangement, not
    an arithmetic artefact of the size distribution.

    Permuting preserves cluster sizes exactly and leaves NO narrow cluster, so
    both the observed share and the null are zero. Enrichment is then
    undefined, and must report nan rather than inf -- 0/0 is "no signal either
    way", not "infinitely enriched".
    """
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    shuffled = np.random.default_rng(1).permutation(labels)
    q = M.quality(shuffled, df, n_perm=9, seed=0)
    assert q["narrow_frac"] == 0.0
    assert q["narrow_frac_null"] == 0.0
    assert np.isnan(q["narrow_enrichment"])


def test_planted_narrow_clusters_beat_their_null():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df, n_perm=9, seed=0)
    assert q["narrow_frac"] == 0.20
    assert q["narrow_frac_null"] == 0.0
    assert q["narrow_enrichment"] == float("inf")


def test_enrichment_is_a_fraction_of_hits_not_clusters():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    assert M.quality(labels, df)["enrichment"] == 0.10


def test_basic_counts():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    assert q["n_clusters"] == 30
    assert q["clustered_pct"] == 100.0
    assert q["median_size"] == 20
    assert q["largest_pct"] == 20 / 600 * 100


def test_all_noise_does_not_raise():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(np.full(len(labels), -1, np.int32), df)
    assert q["n_clusters"] == 0
    assert q["clustered_pct"] == 0.0
    assert np.isnan(q["narrow_frac"])
