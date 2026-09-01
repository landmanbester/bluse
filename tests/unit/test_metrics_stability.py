import numpy as np

from bluse import metrics as M


def test_deterministic_run_is_perfectly_stable():
    labels = np.repeat(np.arange(10), 20).astype(np.int32)
    s = M.stability(lambda seed: labels.copy(), seeds=(0, 1, 2))
    assert s["ari_composite"] == 1.0
    assert s["ari_restricted"] == 1.0
    assert s["noise_agreement"] == 1.0
    assert s["k_mean"] == 10.0


def test_high_noise_agreement_does_not_imply_stable_membership():
    """
    The regression test for the withdrawn '20x more reproducible' claim.

    A run that agrees perfectly about WHICH points are noise, while assigning
    the rest at random, must score high noise_agreement and low
    ari_restricted. Reporting a single composite ARI hides exactly this, and
    that is how leaf's composite 0.480 was mistaken for a stability advantage
    when its membership ARI was 0.032 against eom's 0.028.
    """
    n = 1000
    noise_mask = np.arange(n) % 2 == 0          # identical in every run

    def run_fn(seed):
        r = np.random.default_rng(seed)
        lab = r.integers(0, 50, n).astype(np.int32)
        lab[noise_mask] = -1
        return lab

    s = M.stability(run_fn, seeds=(0, 1, 2, 3))
    assert s["noise_agreement"] == 1.0
    assert s["ari_restricted"] < 0.05
    assert s["ari_composite"] > s["ari_restricted"]


def test_stability_returns_three_separate_numbers():
    labels = np.repeat(np.arange(10), 20).astype(np.int32)
    s = M.stability(lambda seed: labels.copy(), seeds=(0, 1))
    for key in ("ari_composite", "ari_restricted", "noise_agreement"):
        assert key in s


def test_epoch_trace_arithmetic():
    rows = M.epoch_trace([4240, 59, 17, 17], 34933)
    assert rows[0]["removed"] == 34933 - 4240
    assert rows[0]["pct_of_original"] == (34933 - 4240) / 34933 * 100
    assert rows[1]["removed"] == 4240 - 59
    assert rows[3]["removed"] == 0
    assert rows[3]["pct_of_original"] == 0.0


def test_coarsening_null_is_near_zero_for_arbitrary_grouping():
    """
    The control for the headline family-ARI result.

    Matching's default cut returns about k/2 groups regardless of structure, so
    a sceptic asks whether coarsening alone raises agreement. Permuting the
    cluster -> family map preserves the coarsening and destroys the
    correspondence; ARI's chance correction should leave it at zero.
    """
    rng = np.random.default_rng(0)
    n = 900
    cl_runs, fam_runs = [], []
    for seed in range(3):
        r = np.random.default_rng(seed)
        cl = r.integers(0, 60, n).astype(np.int32)
        fam = (cl // 2).astype(np.int32)      # a genuine 2:1 coarsening
        cl_runs.append(cl)
        fam_runs.append(fam)
    null = M.coarsening_null(cl_runs, fam_runs, seed=0)
    assert abs(null) < 0.05


def test_coarsening_null_leaves_a_real_correspondence_detectable():
    """A shared coarsening of a SHARED clustering must beat its own null."""
    import itertools

    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(1)
    cl = rng.integers(0, 60, 900).astype(np.int32)
    fam = (cl // 2).astype(np.int32)
    runs = [cl.copy() for _ in range(3)]
    fams = [fam.copy() for _ in range(3)]
    real = np.mean([adjusted_rand_score(a, b)
                    for a, b in itertools.combinations(fams, 2)])
    assert real == 1.0
    assert M.coarsening_null(runs, fams, seed=0) < real
