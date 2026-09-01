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
