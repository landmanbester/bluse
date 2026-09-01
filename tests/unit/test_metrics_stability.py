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
