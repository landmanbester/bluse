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


def test_metrics_json_records_how_the_family_count_was_chosen():
    """
    The family count is a CHOICE, not a derived quantity -- every cut of the
    Ward tree is uniquely determined by the count it leaves, so no rule can
    discover it (docs/matching-cut-experiment-2026-09.md). A family result is
    therefore uninterpretable without the count and its source, and both used
    to reach stdout only.

    Asserts on the CLI's own serialisation path, since that is where the
    NaN/Infinity guard lives too.
    """
    import json

    from bluse.track_b_cluster import _json_safe

    labels, X = fixtures.synthetic_centroid_space(seed=0)
    _, info = matching.match(labels, X, n_families=3)
    q = {"matching": {k: v for k, v in info.items() if k != "nn_distances"}}
    round_tripped = json.loads(json.dumps(_json_safe(q), allow_nan=False))
    assert round_tripped["matching"]["n_families"] == 3
    assert round_tripped["matching"]["cut_source"] == "n_families=3"


def test_equalised_with_eom_warns(capsys):
    """
    Spec section 2.1: warn loudly, still run. Refusing would block reproducing
    the measurement; running silently would let someone quote a broken number.
    """
    from bluse.track_b_cluster import warn_if_eom_equalised

    warn_if_eom_equalised("robust-equalised", "eom")
    out = capsys.readouterr().out
    assert "WARNING" in out and "0.519" in out and "leaf" in out

    warn_if_eom_equalised("robust-equalised", "leaf")
    assert capsys.readouterr().out == ""
    warn_if_eom_equalised("robust", "eom")
    assert capsys.readouterr().out == ""
