import numpy as np
import pandas as pd
import pytest

from bluse import taxonomy


def _frame(freqs, fams, **extra):
    n = len(freqs)
    d = {"frequency": np.asarray(freqs, dtype=float),
         "driftRate": np.zeros(n), "snr": np.full(n, 20.0),
         "n_beams": np.full(n, 2), "obsid": ["o1"] * n,
         "file": ["f"] * n, "row": np.arange(n)}
    d.update(extra)
    return pd.DataFrame(d), np.asarray(fams)


def test_band_of_names_a_documented_frequency():
    """930 MHz is GSM downlink in SARAO's table; 2500 MHz is in none of it."""
    assert taxonomy.band_of(930.0) == "GSM downlink"
    assert taxonomy.band_of(890.0) == "GSM uplink"
    # 2700 MHz sits in no SARAO or ITU allocation in our table. (2500 does --
    # it is ITU LTE band 7 uplink -- which the first version of this test got
    # wrong.)
    assert taxonomy.band_of(2700.0) is None


def test_band_of_is_vectorised():
    got = taxonomy.band_of(np.array([930.0, 2700.0, 890.0]))
    assert list(got) == ["GSM downlink", None, "GSM uplink"]


def test_named_family_reports_its_band_and_confidence():
    df, fam = _frame([930.0, 931.0, 932.0, 2700.0], [0, 0, 0, 0])
    t = taxonomy.name_families(df, fam)
    r = t.iloc[0]
    assert r["band"] == "GSM downlink"
    assert r["documented_frac"] == pytest.approx(0.75)
    assert bool(r["explained"])


def test_a_family_outside_every_documented_band_is_the_interesting_one():
    """
    The point of the taxonomy is the residue. A family that matches nothing
    documented is a technosignature candidate population, not a failure.
    """
    df, fam = _frame([2700.0, 2701.0, 2702.0], [0, 0, 0])
    t = taxonomy.name_families(df, fam)
    assert t.iloc[0]["band"] is None
    assert t.iloc[0]["documented_frac"] == 0.0
    assert not bool(t.iloc[0]["explained"])


def test_spread_is_reported_robustly_as_well_as_by_full_span():
    """
    Full span is max-min, an extreme-value statistic that one outlier member
    dominates. Measured on lband_short the median family spans 66.1 MHz by that
    ruler and 9.4 MHz by the interquartile range, so both are reported.
    """
    df, fam = _frame([930.0] * 20 + [1000.0], [0] * 21)
    r = taxonomy.name_families(df, fam).iloc[0]
    assert r["freq_span_mhz"] == pytest.approx(70.0)
    assert r["freq_iqr_mhz"] == pytest.approx(0.0)


def test_noise_is_excluded_not_treated_as_a_family():
    df, fam = _frame([930.0, 930.0, 2700.0], [0, 0, -1])
    t = taxonomy.name_families(df, fam)
    assert list(t["family"]) == [0]
    assert t.iloc[0]["n"] == 2


def test_all_noise_returns_an_empty_table_with_the_right_columns():
    df, fam = _frame([930.0, 2700.0], [-1, -1])
    t = taxonomy.name_families(df, fam)
    assert len(t) == 0
    for c in ("family", "n", "band", "explained", "freq_iqr_mhz"):
        assert c in t.columns


def test_candidate_rows_come_from_unexplained_families_only():
    df, fam = _frame([930.0, 930.5, 2700.0, 2701.0], [0, 0, 1, 1])
    t = taxonomy.name_families(df, fam)
    cand = taxonomy.candidate_hits(df, fam, t)
    assert set(cand["row"]) == {2, 3}
    assert set(cand["family"]) == {1}


def test_summary_does_not_label_an_unexplained_family_with_a_band():
    """
    `band` is the PLURALITY band, which a family can have while still being
    mostly outside every allocation. Measured on lband_short, family 16 has a
    median at 887.97 MHz -- inside GSM uplink -- and an IQR of 40 MHz, so under
    half its hits are documented. Printing "GSM uplink" beside it while
    counting it in the residue is contradictory.
    """
    df, fam = _frame([888.0, 889.0] + [1400.0] * 6, [0] * 8)
    t = taxonomy.name_families(df, fam)
    assert t.iloc[0]["band"] == "GSM uplink"      # plurality
    assert not bool(t.iloc[0]["explained"])       # but mostly undocumented
    text = "\n".join(taxonomy.summarise(t))
    assert "UNEXPLAINED" in text
    assert "nearest" in text


def test_summary_separates_the_beam_confined_residue():
    """
    A 64-beam family is RFI whatever band it sits in -- it is in every beam at
    once. The candidate set that matters is the residue that is ALSO spatially
    confined, so the report must not let 64-beam populations dominate the
    headline.
    """
    df, fam = _frame([1400.0] * 4 + [1401.0] * 4, [0] * 4 + [1] * 4,
                     n_beams=np.array([64, 64, 64, 64, 2, 2, 2, 2]))
    text = "\n".join(taxonomy.summarise(taxonomy.name_families(df, fam)))
    assert "beam-confined" in text
