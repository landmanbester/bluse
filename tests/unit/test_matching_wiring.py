import numpy as np

from bluse import matching, metrics
from tests.unit import fixtures


def test_family_stability_is_measurable_with_the_existing_api():
    """
    Acceptance criterion 6 in miniature.

    A stable family COUNT is compatible with scrambled family MEMBERSHIP, so
    the criterion is ari_restricted on family ids. stability() needs only
    run_fn(seed) -> labels, so running it on families is a call-site change.
    """
    labels, X = fixtures.synthetic_centroid_space(seed=0)

    def run_fn(seed):
        rng = np.random.default_rng(seed)
        fam, _ = matching.match(labels, X + rng.normal(0, 0.05, X.shape),
                                rule="gap")
        return fam

    s = metrics.stability(run_fn, seeds=(0, 1, 2))
    assert s["ari_restricted"] > 0.9


def test_families_are_never_more_numerous_than_clusters():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    _, info = matching.match(labels, X)
    assert info["n_families"] <= info["n_clusters"]
