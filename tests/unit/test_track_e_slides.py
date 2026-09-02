"""
The deck is generated from the report, not typed by hand.

Same rule the figures follow, for the same reason: a slide that disagrees with
the measurement is worse than no slide, because it is the version people
remember. These tests assert the numbers on the slides trace back to
report.json rather than to a literal in the template.
"""

import json
import os

import pytest

from bluse import track_e_slides as S


def _report(auc=0.9887, pruned=3003):
    """The shape validate() + the CLI produce, with the fields the deck reads."""
    return {
        "info": {"features": "stamp", "columns": ["a"] * 12, "n_splits": 5,
                 "n_seeds": 3, "seeds": [0, 1, 2], "dtype": "float64",
                 "n_train": 1_599_299, "n_groups": 444},
        "counts": {"candidates": 4565, "contrarian": 3303, "ambiguous": 411_898},
        "verdicts": {"pruned": pruned, "uncertain": 1038, "shortlist": 524},
        "survey": {"n_hits": 2_014_055, "n_survivors": 4565},
        "validation": {
            "ablation": {k: {"n_features": 12, "roc_auc": auc, "pr_auc": 0.99}
                         for k in ("flags", "meta", "stamp", "all")},
            "nonzero_drift": {"stamp": 0.9903, "all": 0.99, "flags": 0.88},
            "snr_stratified": {"weighted": 0.8956, "deciles": []},
            "signal_level": {"median_per_signal": 0.9890, "singleton": 0.9822,
                             "n_singletons": 234_809, "n_signals": 289_869},
            "cross_band": {"L": {"stamp": 0.9895}, "UHF": {"stamp": 0.8975},
                           "S": {"stamp": 0.8962}},
            "n_beams_monotonicity": {"monotone": True, "bins": [
                {"bin": "(0, 1]", "n": 200_959, "in_training": True},
                {"bin": "(2, 3]", "n": 53_031, "in_training": False},
                {"bin": "(3, 5]", "n": 64_765, "in_training": False}]},
            "ood": {"frac_scored_rfi": 0.942},
        },
    }


@pytest.fixture
def plots(tmp_path):
    """A 1x1 PNG under each name the deck embeds."""
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000100ffff0300000600"
        "05a3a2ee0d0000000049454e44ae426082")
    for n in ("track_e_ablation", "track_e_monotonicity", "track_e_funnel",
              "uhf_long_stamps_candidates"):
        (tmp_path / f"{n}.png").write_bytes(png)
    return str(tmp_path)


def test_every_headline_number_comes_from_the_report(plots):
    """Change the report, the deck changes. That is the whole contract."""
    a = S.build(_report(auc=0.9887, pruned=3003), plots)
    b = S.build(_report(auc=0.1234, pruned=42), plots)
    assert "0.9887" in a and "0.9887" not in b
    assert "3,003" in a and "3,003" not in b
    assert "0.1234" in b and "42" in b


def test_the_untrained_hit_count_is_summed_not_typed(plots):
    """
    422,480 is the sum of the bins the model never trained on. Typing it would
    survive a change to the bin edges and quietly become wrong.
    """
    r = _report()
    html = S.build(r, plots)
    expect = sum(b["n"] for b in r["validation"]["n_beams_monotonicity"]["bins"]
                 if not b["in_training"])
    assert f"{expect:,}" in html


def test_the_deck_is_one_self_contained_file(plots):
    """
    It gets moved to a laptop and presented offline. Every image must be inline;
    the only external request allowed is the font stylesheet.
    """
    html = S.build(_report(), plots)
    assert "data:image" in html
    assert 'src="track_e' not in html and "src=\"./" not in html
    external = [t for t in html.split('href="')[1:] if t.startswith("http")]
    assert all("fonts.g" in t.split('"')[0] for t in external), external


def test_the_positive_unlabelled_caveat_cannot_be_dropped_silently(plots):
    """
    The one thing on the deck that must never go missing: slide 4 says a low
    score is not evidence of a technosignature. Everything else is a number;
    this is the sentence that stops the numbers being misread.
    """
    html = S.build(_report(), plots)
    assert "positive&ndash;unlabelled" in html
    assert "not</strong>\n        evidence of a technosignature" in html.replace(
        "\r", "") or "evidence of a technosignature" in html


def test_it_is_a_complete_html_document(plots):
    """A standalone file, not an Artifact fragment -- it needs its own shell."""
    html = S.build(_report(), plots)
    for tag in ("<!doctype html>", '<html lang="en">', "<head>", "<body>",
                "<title>", 'charset="utf-8"', "viewport"):
        assert tag in html, tag
    assert html.rstrip().endswith("</html>")


def test_write_deck_round_trips(tmp_path, plots):
    out = S.write_deck(_report(), plots, str(tmp_path / "deck.html"))
    assert os.path.exists(out)
    assert "Track E" in open(out).read()
