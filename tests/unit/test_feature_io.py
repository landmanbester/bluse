"""
The workshop's feature-delivery convention (features/format.md).

Several of these tests exist because the convention is easy to break silently.
On pandas 3, `set_index` on arithmetic ids gives a RangeIndex, which is stored
as metadata rather than as a column, so the file ends up with no `id` field --
invisible to a pandas reader, fatal to every other one. And naming `id` in
`columns=` returns it as the index rather than as a column, so `df["id"]`
raises even though you asked for it. Both are cheap to get wrong and expensive
to notice.
"""

import os

import numpy as np
import pandas as pd
import pytest

from bluse import feature_io as FIO


def _frame(ids, n_cols=2):
    ids = np.asarray(ids)
    data = {f"f{i:02d}": np.arange(len(ids), dtype=np.float64) + i
            for i in range(n_cols)}
    data["weak_label"] = np.ones(len(ids), dtype=np.int8)
    data["feature_ok"] = np.ones(len(ids), dtype=bool)
    data["stamp_ok"] = np.ones(len(ids), dtype=bool)
    return pd.DataFrame({"id": ids, **data})


# --- naming ---------------------------------------------------------------


def test_filename_carries_dataset_and_method():
    assert FIO.filename("lband_short") == "lband_short_globular_features.parquet"
    assert FIO.filename("mk_sample_hits", "resnet18") == \
        "mk_sample_hits_resnet18_features.parquet"


def test_dataset_is_recovered_from_a_name_with_underscores():
    """`mk_sample_hits_globular` splits four ways on `_` and one way correctly."""
    assert FIO.split_filename("mk_sample_hits_globular_features.parquet") == \
        "mk_sample_hits"
    assert FIO.split_filename("all_globular_features.parquet") == "all"


def test_a_foreign_method_is_not_mistaken_for_ours():
    assert FIO.split_filename("mk_sample_resnet18_penultimate_features.parquet") \
        is None
    assert FIO.split_filename("lband_short_features.parquet") is None
    assert FIO.split_filename("notes.md") is None


# --- the two traps --------------------------------------------------------


# Every one of these is an arithmetic sequence, which is what pandas 3 turns
# into a RangeIndex -- not just the 0..N-1 case. Ascending, offset, stepped and
# descending all trip it.
ARITHMETIC_IDS = [
    [0, 1, 2, 3, 4],
    [100, 101, 102, 103, 104],
    [7, 9, 11, 13, 15],
    [5, 4, 3, 2, 1],
]


@pytest.mark.parametrize("ids", ARITHMETIC_IDS)
def test_arithmetic_ids_are_still_written_as_a_column(ids, tmp_path):
    """
    THE TRAP. On pandas 3, `set_index("id")` on any arithmetic run of ids gives
    a RangeIndex, which pandas stores as start/stop/step in the metadata rather
    than as a column. The file then has no `id` field at all -- and because
    pandas reconstructs the values from those three numbers, a pandas-only
    round trip cannot see it. write() passes index=True.
    """
    import pyarrow.parquet as pq

    dest = FIO.write(_frame(np.array(ids, dtype=np.int32)),
                     str(tmp_path / FIO.filename("toy")))
    assert "id" in pq.read_schema(dest).names
    assert FIO.read(dest)["id"].tolist() == ids


@pytest.mark.parametrize("ids", ARITHMETIC_IDS)
def test_the_naive_write_really_does_drop_the_id_field(ids, tmp_path):
    """
    Why write() cannot just call set_index. Guards the reason, not only the
    behaviour -- if a future pandas stops inferring RangeIndex here, this test
    fails and the module docstring needs revisiting rather than the code.

    Note what is and is not lost: pandas reads the values back correctly, so
    the assertion has to be about the SCHEMA. Anything checking only the values
    would pass while the file was unusable to pyarrow, polars or R/arrow.
    """
    import pyarrow.parquet as pq

    naive = tmp_path / "naive.parquet"
    df = _frame(np.array(ids, dtype=np.int32)).set_index("id")
    df.to_parquet(naive)

    if type(df.index).__name__ != "RangeIndex":
        pytest.skip("this pandas does not infer a RangeIndex from set_index")

    assert "id" not in pq.read_schema(naive).names
    assert pd.read_parquet(naive).index.tolist() == ids, \
        "the values do round-trip through pandas -- that is what makes it silent"


def test_reading_a_column_subset_may_name_the_id(tmp_path):
    """
    THE OTHER TRAP. Once `id` is the index, asking for it by name in `columns=`
    succeeds and still does not give you a column -- it comes back as the index,
    so `df["id"]` raises KeyError. read() appends `id` to the request and resets
    the index afterwards, so callers keep the column lists they already had.
    """
    dest = FIO.write(_frame([7, 9, 11]), str(tmp_path / FIO.filename("toy")))

    naive = pd.read_parquet(dest, columns=["id", "f00"])
    assert "id" not in naive.columns          # asked for, not delivered
    with pytest.raises(KeyError):
        naive["id"]

    got = FIO.read(dest, columns=["id", "f00"])
    assert list(got.columns) == ["id", "f00"]
    assert got["id"].tolist() == [7, 9, 11]


def test_the_id_comes_back_even_when_the_caller_did_not_ask(tmp_path):
    """It is the index, so it is free -- and a row without it cross-matches
    against nothing. Callers that never mentioned `id` still get it."""
    dest = FIO.write(_frame([7, 9, 11]), str(tmp_path / FIO.filename("toy")))
    got = FIO.read(dest, columns=["f00"])
    assert got["id"].tolist() == [7, 9, 11]

    legacy = tmp_path / FIO.legacy_filename("toy2")
    _frame([7, 9, 11]).to_parquet(legacy, index=False)
    assert FIO.read(str(legacy), columns=["f00"])["id"].tolist() == [7, 9, 11]


def test_naming_the_id_against_a_naively_written_file_raises(tmp_path):
    """The other half of trap 2: with the ids gone, the same call is an error."""
    naive = tmp_path / "naive.parquet"
    _frame(np.arange(3, dtype=np.int32)).set_index("id").to_parquet(naive)
    with pytest.raises(Exception):
        pd.read_parquet(naive, columns=["id", "f00"])


# --- round trip -----------------------------------------------------------


def test_round_trip_preserves_ids_dtype_and_order(tmp_path):
    df = _frame(np.array([316974, 5, 900001], dtype=np.int32))
    dest = FIO.write(df, str(tmp_path / FIO.filename("toy")))

    raw = pd.read_parquet(dest)
    assert raw.index.name == "id"          # the convention, as delivered
    assert "id" not in raw.columns

    back = FIO.read(dest)
    assert back["id"].dtype == np.int32
    pd.testing.assert_frame_equal(back, df)


def test_write_refuses_a_frame_with_no_id(tmp_path):
    df = _frame([1, 2, 3]).drop(columns=["id"])
    with pytest.raises(ValueError, match="no `id` column"):
        FIO.write(df, str(tmp_path / FIO.filename("toy")))


def test_write_refuses_duplicate_ids(tmp_path):
    """An index that repeats cannot cross-match, which is the whole point."""
    with pytest.raises(ValueError, match="not unique"):
        FIO.write(_frame([1, 2, 2]), str(tmp_path / FIO.filename("toy")))


def test_read_rejects_a_parquet_that_is_not_a_feature_file(tmp_path):
    p = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(p)
    with pytest.raises(ValueError, match="no `id`"):
        FIO.read(str(p))


# --- discovery and migration ----------------------------------------------


def test_discover_ignores_the_combined_table_and_foreign_features(tmp_path):
    d = str(tmp_path)
    FIO.write(_frame([1, 2]), os.path.join(d, FIO.filename("sband_short")))
    FIO.write(_frame([3, 4]), os.path.join(d, FIO.filename("mk_sample_hits")))
    FIO.write(_frame([1, 2, 3, 4]), os.path.join(d, FIO.filename(FIO.COMBINED)))
    # Somebody else's features, sharing the directory. Concatenating these with
    # ours would silently mix two feature spaces.
    pd.DataFrame({"a": [1.0]}, index=pd.Index([1], name="id")).to_parquet(
        os.path.join(d, "mk_sample_resnet18_penultimate_features.parquet"))

    assert sorted(FIO.discover(d)) == ["mk_sample_hits", "sband_short"]
    assert sorted(FIO.discover(d, include_combined=True)) == \
        ["all", "mk_sample_hits", "sband_short"]


def test_a_legacy_file_is_found_read_and_migrated(tmp_path):
    d = str(tmp_path)
    df = _frame([10, 20, 30])
    legacy = os.path.join(d, FIO.legacy_filename("uhf_long"))
    df.to_parquet(legacy, index=False)          # the pre-convention layout

    assert FIO.find("uhf_long", d) == legacy
    assert FIO.is_legacy(legacy)
    assert sorted(FIO.discover(d)) == ["uhf_long"]
    pd.testing.assert_frame_equal(FIO.read(legacy), df)

    moved, already = FIO.migrate(d)
    assert moved == [(legacy, FIO.path("uhf_long", d))]
    assert already == []
    assert os.path.exists(legacy), "migration must not delete the original"
    assert FIO.find("uhf_long", d) == FIO.path("uhf_long", d)
    assert pd.read_parquet(moved[0][1]).index.name == "id"
    # And the directory no longer yields the same dataset twice.
    assert sorted(FIO.discover(d)) == ["uhf_long"]

    # Running it again is a no-op that reports what is now redundant, rather
    # than rewriting or deleting anything.
    assert FIO.migrate(d) == ([], [legacy])


def test_migration_leaves_foreign_feature_files_alone(tmp_path):
    d = str(tmp_path)
    foreign = os.path.join(d, "mk_sample_resnet18_penultimate_features.parquet")
    pd.DataFrame({"a": [1.0]}, index=pd.Index([1], name="id")).to_parquet(foreign)
    assert FIO.migrate(d) == ([], [])
    assert not FIO.is_legacy(foreign)
    assert FIO.legacy_files(d) == []
