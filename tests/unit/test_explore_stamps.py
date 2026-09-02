import argparse

from bluse.explore import stamp_tag


def _args(**kw):
    base = dict(rows=None, family=None, sort="random", each_family=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_tag_falls_back_to_the_sort_mode():
    assert stamp_tag("lband_short", _args(sort="snr")) == "snr"


def test_tag_names_the_selection_not_the_sort():
    """
    --sort is ignored when rows are given, so keying the filename on it put
    every family on <name>_stamps_random.png, each silently overwriting the
    last.
    """
    a = _args(rows="clusters/lband_short_candidates.csv", sort="random")
    assert stamp_tag("lband_short", a) == "candidates"


def test_tag_does_not_repeat_the_dataset_name():
    """Otherwise: lband_short_stamps_lband_short_candidates_fam19.png."""
    a = _args(rows="clusters/lband_short_candidates.csv", family=19)
    assert stamp_tag("lband_short", a) == "candidates_fam19"


def test_tag_keeps_an_unrelated_csv_name_intact():
    a = _args(rows="/tmp/my_picks.csv", family=3)
    assert stamp_tag("lband_short", a) == "my_picks_fam3"
