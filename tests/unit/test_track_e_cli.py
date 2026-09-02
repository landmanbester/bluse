"""
The CLI's own serialisation and catalogue paths.

The defect this file exists to prevent has landed in this repo twice: a metrics
record that disagreed with the matrix actually clustered, and a scaling mode
the CLI accepted and silently did not apply. Both were caught by review, not by
a test. So assert the written artefacts against the computation, not against
themselves.
"""

import json

import numpy as np
import pandas as pd

from bluse import track_e_score as E
from tests.unit import fixtures


def _survivor_frame(n=3000, seed=0):
    """A weak-labelled frame carrying the provenance columns catalogues() needs."""
    df = fixtures.synthetic_weak_labelled(n=n, groups=10, seed=seed)
    rng = np.random.default_rng(seed)
    df["row"] = np.arange(n)
    df["id"] = np.arange(n) + 1000
    df["obsid"] = df["group_id"]
    df["sourceName"] = [f"src{i % 37}" for i in range(n)]
    df["beam"] = rng.integers(0, 64, n)
    df["frequency"] = rng.uniform(856.0, 1712.0, n)
    df["driftRate"] = rng.normal(0, 0.2, n)
    df["snr"] = rng.lognormal(3, 1, n)
    df["beam_frac"] = rng.random(n)
    # n_beams consistent with weak_label, so the catalogue cuts mean something.
    nb = np.where(df.weak_label == 1, rng.integers(32, 65, n),
                  np.where(df.weak_label == 0, rng.integers(1, 3, n),
                           rng.integers(3, 32, n)))
    df["n_beams"] = nb
    df["pass_all"] = (nb <= 4) & (rng.random(n) < 0.5)
    return df


def test_catalogues_partition_survivors_by_the_stated_thresholds():
    df = _survivor_frame()
    score = np.linspace(0, 1, len(df))
    cats = E.catalogues(df, score, shortlist_below=0.1, pruned_above=0.9)

    c = cats["candidates"]
    assert len(c) == int(df.pass_all.sum())
    assert c.rfi_score.is_monotonic_increasing, "candidates must be ranked"
    assert (c.loc[c.verdict == "shortlist", "rfi_score"] < 0.1).all()
    assert (c.loc[c.verdict == "pruned", "rfi_score"] > 0.9).all()
    assert set(c.verdict) <= {"shortlist", "uncertain", "pruned"}


def test_contrarian_is_the_disagreement_not_the_agreement():
    """
    >=32 beams AND a clean score. The spatial filter and morphology disagree in
    the one direction that cannot be explained by the filter being
    conservative, so this set is either instrumental or the model's blind spot.
    An off-by-one in the comparison would silently return multi-beam RFI the
    model also calls RFI -- a large, useless, plausible-looking table.
    """
    df = _survivor_frame()
    score = np.linspace(0, 1, len(df))
    con = E.catalogues(df, score, shortlist_below=0.1)["contrarian"]
    assert len(con)
    assert (con.n_beams >= 32).all()
    assert (con.rfi_score < 0.1).all()


def test_ambiguous_is_exactly_the_range_the_spatial_filter_abstains_on():
    """3-31 beams: too many to call confined, too few to call RFI. This table
    is the reason Track E exists, so its bounds are worth pinning."""
    df = _survivor_frame()
    amb = E.catalogues(df, np.linspace(0, 1, len(df)))["ambiguous"]
    assert (amb.n_beams > 2).all() and (amb.n_beams < 32).all()


def test_catalogues_carry_what_bluse_explore_stamps_needs():
    """
    `bluse-explore stamps --rows FILE` selects on `row` and filters on `file`.
    Without both, a candidate list cannot be eyeballed without a manual join --
    which is the step people skip.
    """
    df = _survivor_frame()
    for tab in E.catalogues(df, np.linspace(0, 1, len(df))).values():
        assert {"row", "file", "rfi_score"} <= set(tab.columns)


def test_the_report_is_strict_json_with_no_nan():
    """
    validate() returns NaN wherever an AUC is undefined -- a subgroup with one
    class. json.dump writes bare NaN by default, which is invalid JSON that
    many parsers accept and some reject, so the failure surfaces on someone
    else's machine.
    """
    obj = {"a": float("nan"), "b": [1.0, float("inf")], "c": np.int64(3),
           "d": np.True_, "e": {"f": np.float32("nan")}, "g": True}
    s = json.dumps(E._json_safe(obj), allow_nan=False)
    assert json.loads(s) == {"a": None, "b": [1.0, None], "c": 3, "d": True,
                             "e": {"f": None}, "g": True}


def test_booleans_survive_as_booleans_not_as_ones():
    """
    bool is a subclass of int, so an int branch placed first catches True and
    writes 1. `1 == True` in Python, so an equality assertion does NOT catch it
    -- the first version of the test above passed while the report was writing
    integers.

    It matters downstream: np.array([1, 1, 0]) is an INT array, where `x[mask]`
    is fancy indexing rather than masking and `~mask` is bitwise NOT. That is
    how the monotonicity figure first drew the wrong five points, silently and
    plausibly.
    """
    out = E._json_safe({"t": True, "f": False, "n": np.True_, "i": 1})
    assert out["t"] is True and out["f"] is False and out["n"] is True
    assert isinstance(out["i"], int) and not isinstance(out["i"], bool)
    assert '"t": true' in json.dumps(out)


def test_the_written_score_reproduces_from_the_recorded_info():
    """
    Acceptance criterion 6, and the D-4 defect in miniature: a record that
    disagrees with what was computed is worse than no record, because it is
    believed. Refit from `info` alone and require the same numbers.
    """
    df = _survivor_frame(seed=3)
    score, fold, info = E.fit_score(df, features="stamp", n_splits=4, seed=11)
    again, fold2, _ = E.fit_score(
        df, features=info["features"], n_splits=info["n_splits"],
        seed=info["seed"], exclude_mk=info["exclude_mk"],
        max_iter=info["max_iter"])
    assert np.array_equal(score, again)
    assert np.array_equal(fold, fold2)
    assert info["n_scored_out_of_fold"] + info["n_scored_by_fold_mean"] == len(df)


def test_scores_dir_is_created_on_demand(tmp_path, monkeypatch):
    """A workspace built before Track E existed has no scores/ directory."""
    import os

    from bluse import paths

    (tmp_path / "data").mkdir()
    (tmp_path / "features").mkdir()
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    paths.set_workspace(str(tmp_path))
    d = paths.scores_dir()
    assert d.endswith("scores") and os.path.isdir(d)
    paths.set_workspace(None)
