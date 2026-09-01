from bluse import features as F


def test_registered_features_default_to_continuous():
    assert F.column_kinds()["f01_frequency"] == "continuous"


def test_declared_ordinal_columns_report_ordinal():
    kinds = F.column_kinds()
    assert kinds["f02_abs_drift"] == "ordinal"
    assert kinds["f12_bandwidth_hz"] == "ordinal"
    assert kinds["x02_time_occupancy"] == "ordinal"


def test_saturation_columns_are_flags():
    assert F.column_kinds()["f08_turning_bw_saturated"] == "flag"


def test_every_registered_column_has_a_kind():
    kinds = F.column_kinds()
    for col in F.all_columns():
        assert col in kinds
        assert kinds[col] in {"continuous", "ordinal", "boolean", "flag"}


def test_f02_keeps_its_rank_transform():
    """
    Decision pin, not a behaviour test.

    f02_abs_drift is a 42-level ordinal on an exact lattice whose 26.6%
    zero-drift tie the quantile-normal transform throws to the -5.199 bound,
    leaving it the least-weighted column in the matrix (0.8% of the k-NN
    distance share). That reads like a bug, and replacing it with the native
    linear grid -- with or without a zero-drift indicator -- was the planned
    repair.

    It was measured and rejected (docs/f02-rework-experiment-2026-09.md):
    worse under eom at every family count, better under leaf only in the
    degenerate 8-24 family corner where the narrow share is 0.000%, and it does
    not rescue the equalising scaling mode it was ordered ahead of.

    This test exists so the change cannot be made silently a second time. If
    you are here because it failed, re-run that experiment rather than
    updating the constant.
    """
    assert F.TRANSFORMS["f02_abs_drift"] == "quantile-normal"


def test_no_zero_drift_indicator_is_registered():
    """
    The other half of the rejected rework. A zero-drift boolean is the
    lowest-share column measured (0.33% k-NN), so contribution equalisation
    would weight it 10.256 -- twice the 5.216 on f02 that is measured to
    destroy eom. It is also constant on mk_sample_hits, which has no zero-drift
    hits at all.

    Track A's `flag_zero_drift` still exists on the dataframe and is not a
    registered feature, so it stays out of the clustering matrix. That is the
    intended arrangement.
    """
    assert not [c for c in F.all_columns() if "zero_drift" in c]
