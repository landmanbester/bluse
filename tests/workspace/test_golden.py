"""
Golden values measured on the real feature matrices, 2026-09-01.

These catch a regression in the science. They cannot gate a commit, because
aug_2026_workshop/features/ is gitignored and the data is not in the repo.
Every number below states the file it was measured on.
"""

import os

import numpy as np
import pandas as pd
import pytest

from bluse import diagnostics as D
from bluse import features as F
from bluse import paths

pytestmark = pytest.mark.workspace

_CACHE = {}


def _audit(name):
    if name in _CACHE:
        return _CACHE[name]
    path = os.path.join(paths.features_dir(), f"{name}_features.parquet")
    if not os.path.exists(path):
        pytest.skip(f"no {name} feature matrix in this workspace")
    df = pd.read_parquet(path)
    df = df[df.feature_ok].reset_index(drop=True)
    cols = [c + "_n" for c in F.all_columns()
            if c + "_n" in df.columns and not c.endswith("_saturated")
            and not c.startswith("f08_")]
    X = df[cols].to_numpy(dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    kinds = {c + "_n": k for c, k in F.column_kinds().items()}
    _CACHE[name] = {r["col"]: r for r in
                    D.audit(X, cols, scaling="robust", kinds=kinds,
                            min_samples=8)}
    return _CACHE[name]


def test_f02_is_a_42_level_ordinal_on_sband_short():
    a = _audit("sband_short")["f02_abs_drift_n"]
    assert a["n_distinct"] == 42
    assert a["max_tie_fraction"] == pytest.approx(0.266, abs=0.005)
    assert a["tie_value"] == pytest.approx(-5.199, abs=0.01)
    assert a["iqr_raw"] == pytest.approx(5.954, rel=0.01)
    assert a["kind"] == "ordinal"
    assert "tie" in a["flags"]


def test_x03_is_over_weighted_ONLY_GLOBALLY_on_sband_short():
    """
    The correction this audit exists to have made.

    x03_channel_offset was indicted by the original review as over-weighted at
    24.3% of the global distance share. It is 24.3% globally and only ~7.4%
    locally, against an equal share of 6.7% -- i.e. locally it is an ordinary
    column. Since HDBSCAN responds to core distances and mutual reachability,
    both local, the local number is the one that describes what the clusterer
    sees, and P0-3 moved the flag threshold onto it.

    So x03 must NOT carry share-high any more. It carries share-disagree
    instead, which says precisely the true thing: the global statistic
    misleads here.
    """
    a = _audit("sband_short")["x03_channel_offset_n"]
    assert a["share_global"] == pytest.approx(0.243, abs=0.02)
    assert a["share_knn"] == pytest.approx(0.059, abs=0.015)
    assert "share-high" not in a["flags"]
    assert "share-disagree" in a["flags"]


def test_f09_is_the_largest_local_contributor_on_sband_short():
    """
    The other half of the correction. f09_temporal_skew is 6.7% global -- dead
    on an equal share, invisible to the old rule -- and 15.5% local, the
    largest local contributor in the matrix. The calibrated rule flags it.
    """
    a = _audit("sband_short")["f09_temporal_skew_n"]
    assert a["share_knn"] == pytest.approx(0.153, abs=0.02)
    assert "share-high" in a["flags"]


def test_f02_is_under_weighted_on_sband_short():
    """
    f02 is under-weighted on BOTH statistics -- 1.7% global, 0.8% local -- so
    the flag survives the move onto share_knn. Suppressing it is nevertheless
    deliberate and measured; see docs/f02-rework-experiment-2026-09.md.
    """
    a = _audit("sband_short")["f02_abs_drift_n"]
    assert a["share_global"] == pytest.approx(0.017, abs=0.01)
    assert a["share_knn"] == pytest.approx(0.004, abs=0.004)
    assert "share-low" in a["flags"]


def test_f07_clips_on_sband_short():
    a = _audit("sband_short")["f07_kurt_bw_corr_n"]
    assert a["clip_frac"] == pytest.approx(0.010, abs=0.004)
    assert "clip" in a["flags"]
