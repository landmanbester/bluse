#!/usr/bin/env python3
"""
taxonomy.py -- turn clusters into NAMED RFI families, and isolate the residue.

The point of an RFI taxonomy is not the taxonomy. It is the leftovers: once
"these hits are GSM downlink, these are aircraft transponders", whatever
matches nothing documented is the technosignature candidate set. Until this
module existed the pipeline produced clusters, families and metrics but never
said which real-world emitters they were, so the residue had never been
isolated.

Measured on the repaired lband_short (856-1068 MHz), 35,000 hits, `leaf`, cut
to 40 families: 36 families fall in a SARAO-documented band and hold 96.8% of
the clustered hits. The named ones are exactly what a spectrum manager would
expect -- GSM downlink near 930 and 950 MHz, GSM uplink near 888 MHz, aircraft
transponders/DME near 984 and 1000 MHz -- and the clusterer separates the same
band into several families differing in drift rate.

TWO HONEST CAVEATS, because the table reads better than the science warrants.

Families are BROAD. The median family's interquartile frequency range is
9.4 MHz. That is a description of a service band, not of a transmitter: GSM
downlink is a 35 MHz allocation, and a family with a 6 MHz IQR inside it is
"the GSM downlink population" rather than "transmitter #4". Do not read a
family as one emitter.

Families FRAGMENT, and the stamps prove it. Five families sit at
950.42-950.46 MHz differing only in drift rate (0.09 to 0.26 Hz/s), and the
same happens in the residue: ALL FIVE unexplained lband_short families contain
one bright emitter at 874.9955 MHz, which alone accounts for 20.4% of the
residue (7,423 of 36,338 hits). Families 18 and 19 are nearly pure (98.8% and
84.8% of their members within 10 kHz of it) while 13, 14 and 16 are broad
mixtures that happen to include it -- family 13's median reads 924.99 MHz, yet
its twelve brightest members are all at 874.99.

So a family count is an upper bound on the number of populations, not an
estimate of it, and a family's median frequency is not a reliable description
of it when the IQR runs to tens of MHz. No metric surfaced this; plotting the
stamps did, in one command. See `bluse-explore stamps --rows ... --each-family`.

Both spreads are reported. Full span is max-minus-min, an extreme-value
statistic that a single outlier member dominates: on lband_short the median
family spans 66.1 MHz by that ruler against 9.4 MHz by the interquartile
range. The IQR is the honest summary of where a family's members actually sit;
the span is kept because it is what `metrics.quality` has always reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import rfi_masks

# A family counts as explained when at least this fraction of its hits fall in
# some documented band. A majority, deliberately: the alternative is to key on
# the family's median frequency alone, which calls a family explained when half
# its members sit outside every allocation we know about.
EXPLAINED_FRAC = 0.5

COLUMNS = ["family", "n", "band", "band_frac", "documented_frac", "explained",
           "median_freq_mhz", "freq_iqr_mhz", "freq_p5_p95_mhz",
           "freq_span_mhz", "median_abs_drift", "median_snr",
           "median_n_beams", "n_obs"]


def bands(include_itu=True):
    """(lo, hi, label) for every documented allocation."""
    return [(lo, hi, lab) for lo, hi, lab, src in rfi_masks.MASKS
            if include_itu or src == "SARAO"]


def band_of(freq_mhz, include_itu=True):
    """
    Documented band label for a frequency in MHz, or None.

    Scalar in, scalar out; array in, object array out. Where allocations
    overlap the first match in the table wins, which is the same precedence
    Track A's mask uses.
    """
    table = bands(include_itu)
    scalar = np.isscalar(freq_mhz)
    f = np.atleast_1d(np.asarray(freq_mhz, dtype=float))
    out = np.full(len(f), None, dtype=object)
    for lo, hi, lab in table:
        hit = (out == None) & (f >= lo) & (f <= hi)   # noqa: E711
        out[hit] = lab
    return out[0] if scalar else out


def name_families(df, fam, *, include_itu=True, explained_frac=EXPLAINED_FRAC):
    """
    One row per family, with the band it matches and how confidently.

    `band` is the allocation holding a PLURALITY of the family's hits, not the
    one containing its median frequency -- a family straddling two allocations
    has a median that may sit in neither. `documented_frac` is the fraction in
    any documented band at all, and is what `explained` thresholds.
    """
    fam = np.asarray(fam)
    freq = df["frequency"].to_numpy(dtype=float)
    labels = band_of(freq, include_itu)

    rows = []
    for f in np.unique(fam[fam >= 0]):
        m = fam == f
        fr, lb = freq[m], labels[m]
        named = np.array([x is not None for x in lb])
        best, best_n = None, 0
        if named.any():
            vals, counts = np.unique(lb[named].astype(str), return_counts=True)
            j = int(counts.argmax())
            best, best_n = str(vals[j]), int(counts[j])
        q25, q75 = np.percentile(fr, [25, 75])
        p5, p95 = np.percentile(fr, [5, 95])
        rows.append({
            "family": int(f), "n": int(m.sum()),
            "band": best, "band_frac": best_n / m.sum(),
            "documented_frac": float(named.mean()),
            "explained": bool(named.mean() >= explained_frac),
            "median_freq_mhz": float(np.median(fr)),
            "freq_iqr_mhz": float(q75 - q25),
            "freq_p5_p95_mhz": float(p95 - p5),
            "freq_span_mhz": float(fr.max() - fr.min()),
            "median_abs_drift": float(np.median(np.abs(
                df["driftRate"].to_numpy()[m]))),
            "median_snr": float(np.median(df["snr"].to_numpy()[m])),
            "median_n_beams": (float(np.median(df["n_beams"].to_numpy()[m]))
                               if "n_beams" in df else float("nan")),
            "n_obs": (int(pd.Series(df["obsid"].to_numpy()[m]).nunique())
                      if "obsid" in df else 0),
        })
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows)[COLUMNS].sort_values("n", ascending=False)


def candidate_hits(df, fam, table):
    """
    The residue: every hit whose family matches nothing documented.

    This is the output the whole pipeline exists to produce. Carries `file` and
    `row` so the stamps can be plotted straight from the HDF5 --
    `bluse-explore stamps <file>.h5 --rows <tag>_candidates.csv`.
    """
    fam = np.asarray(fam)
    if not len(table):
        return df.iloc[[]].assign(family=[])
    keep = set(table.loc[~table["explained"].astype(bool), "family"].tolist())
    m = np.array([f in keep for f in fam])
    out = df[m].copy()
    out["family"] = fam[m]
    return out


def summarise(table, n_show=15):
    """Printable report. Returns the lines so callers can log or write them."""
    if not len(table):
        return ["  no families to name (every hit is noise)"]
    ex = table["explained"].astype(bool)
    tot = int(table["n"].sum())
    out = [
        f"  {'fam':>4s} {'hits':>7s} {'median MHz':>11s} {'IQR':>8s} "
        f"{'|drift|':>8s} {'beams':>6s}  band",
    ]
    for _, r in table.head(n_show).iterrows():
        # Never print a band beside a family the threshold calls unexplained.
        # `band` is the PLURALITY band, which a family can have while most of
        # its hits sit outside every allocation -- lband_short family 16 has a
        # median inside GSM uplink and a 40 MHz IQR -- and printing the label
        # next to a family counted in the residue contradicts the totals below.
        if r.explained:
            what = r.band
        elif r.band:
            what = f"— UNEXPLAINED — (nearest: {r.band}, "\
                   f"{100 * r.documented_frac:.0f}% documented)"
        else:
            what = "— UNEXPLAINED —"
        out.append(f"  {r.family:4d} {r.n:7d} {r.median_freq_mhz:11.3f} "
                   f"{r.freq_iqr_mhz:8.3f} {r.median_abs_drift:8.4f} "
                   f"{r.median_n_beams:6.0f}  {what}")
    if len(table) > n_show:
        out.append(f"  ... {len(table) - n_show} more")
    out += [
        "",
        f"  EXPLAINED   {int(ex.sum()):3d} of {len(table)} families, "
        f"{int(table.loc[ex, 'n'].sum()):,} hits "
        f"({100 * table.loc[ex, 'n'].sum() / tot:.1f}%)",
        f"  RESIDUE     {int((~ex).sum()):3d} of {len(table)} families, "
        f"{int(table.loc[~ex, 'n'].sum()):,} hits "
        f"({100 * table.loc[~ex, 'n'].sum() / tot:.1f}%)",
    ]
    # A 64-beam family is RFI whatever band it sits in -- it is in every beam
    # at once. So the residue is not the candidate set by itself; the residue
    # that is ALSO spatially confined is. Measured on lband_short every
    # unexplained family has a 64-beam median, i.e. the residue there is
    # entirely undocumented RFI rather than anything to follow up.
    if "median_n_beams" in table and table["median_n_beams"].notna().any():
        conf = (~ex) & (table["median_n_beams"] <= 4)
        out.append(
            f"  of which beam-confined (<=4 beams): {int(conf.sum()):3d} "
            f"families, {int(table.loc[conf, 'n'].sum()):,} hits "
            f"({100 * table.loc[conf, 'n'].sum() / tot:.2f}%)"
            f"   <- the candidate set")
    out += [
        "",
        "  A family is a service band, not a transmitter: the median family's",
        "  interquartile frequency range is several MHz. Inspect the residue",
        "  with `bluse-explore stamps <file>.h5 --rows <tag>_candidates.csv`.",
    ]
    return out
