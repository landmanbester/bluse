#!/usr/bin/env python3
"""
Known-RFI frequency masks for MeerKAT.

Every entry carries a `source` tag so we can always tell documented fact from
inference:

  "SARAO"     -- verbatim from the SARAO External Service Desk Knowledge Base,
                 "Radio Frequency Interference (RFI)", Table 1. This is the
                 same reference Tremblay et al. (2026) used for the K2-18b
                 MeerKAT search.
  "ITU"       -- standard ITU/national spectrum allocations that are NOT in the
                 SARAO table but are physically expected in the band. Included
                 because our S-band data (1969-2825 MHz) is almost entirely
                 outside SARAO's tabulated coverage. Treat as a hypothesis.
  "empirical" -- derived from these data by track_a_filter.py --derive-mask.

Frequencies are MHz. Ranges are inclusive.

Reference: https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/305332225/
"""

from __future__ import annotations

# (f_lo_MHz, f_hi_MHz, label, source)
MASKS: list[tuple[float, float, str, str]] = [
    # ---- mobile network downlinks (UHF band) ----------------------------
    (768.0,   778.0,  "Vodacom downlink",            "SARAO"),
    (801.0,   811.0,  "MTN downlink",                "SARAO"),
    (811.0,   821.0,  "Telkom downlink",             "SARAO"),

    # ---- GSM -------------------------------------------------------------
    (880.0,   915.0,  "GSM uplink",                  "SARAO"),
    (925.0,   960.0,  "GSM downlink",                "SARAO"),

    # ---- aviation --------------------------------------------------------
    # Many narrow, intermittent transponder/DME signals. SARAO documents the
    # whole span rather than individual lines; masking all of it is blunt but
    # it is what the documentation supports.
    (962.0,  1213.0,  "aircraft transponders / DME", "SARAO"),

    # ---- GNSS ------------------------------------------------------------
    (1565.0, 1585.0,  "GPS L1",                      "SARAO"),
    (1217.0, 1237.0,  "GPS L2",                      "SARAO"),
    (1375.0, 1387.0,  "GPS L3",                      "SARAO"),
    (1166.0, 1186.0,  "GPS L5",                      "SARAO"),
    (1592.0, 1610.0,  "GLONASS L1",                  "SARAO"),
    (1242.0, 1249.0,  "GLONASS L2",                  "SARAO"),

    # ---- satellite communications ---------------------------------------
    (1616.0, 1626.0,  "Iridium",                     "SARAO"),
    (1526.0, 1554.0,  "Inmarsat",                    "SARAO"),
    (2483.5, 2495.0,  "Globalstar",                  "SARAO"),

    # ---- ISM -------------------------------------------------------------
    (2400.0, 2495.0,  "Wi-Fi",                       "SARAO"),
    (2400.0, 2483.5,  "Bluetooth",                   "SARAO"),

    # ---- S band: not covered by the SARAO table --------------------------
    # Our S-band files span 1968.8-2825.0 MHz. SARAO's table only reaches into
    # this range for the ISM/Globalstar entries above, so the following are
    # standard allocations added by us. Flag, inspect, then decide.
    #
    # 2200-2290 MHz is the space-operations / space-research downlink band.
    # This is where JWST's telemetry sits (2270.5 MHz), which BLUSE itself used
    # as its end-to-end test signal (Czech et al. 2026). Our own exploration
    # found a very strong 64-beam signal at 2242.500206 MHz, squarely inside
    # this band -- almost certainly satellite telemetry.
    (2200.0, 2290.0,  "space ops/research downlink", "ITU"),
    (1920.0, 1980.0,  "IMT-2000 uplink",             "ITU"),
    (2110.0, 2170.0,  "IMT-2000 downlink",           "ITU"),
    (2300.0, 2400.0,  "LTE band 40 (TD)",            "ITU"),
    (2500.0, 2570.0,  "LTE band 7 uplink",           "ITU"),
    (2620.0, 2690.0,  "LTE band 7 downlink",         "ITU"),
]

# Galileo carriers are given as single frequencies by SARAO. E1/E5a/E5b/E5-AltBOC
# all fall inside ranges already masked above (GPS L1, GPS L5, DME). Only E6
# needs its own entry; the half-width is a judgement call, not documentation.
POINT_MASKS: list[tuple[float, str, str]] = [
    (1278.75, "Galileo E6", "SARAO"),
]
POINT_HALFWIDTH_MHZ = 5.0

# South African digital terrestrial TV: 8 MHz channels centred on
# (306 + 8 * channel) MHz.
#
# WARNING, and the reason this is OFF by default: channels 21-68 tile
# contiguously from 466 to 858 MHz. Enabling the full comb therefore masks a
# solid 392 MHz block, which annihilates the entire uhf_short band (544-680 MHz)
# -- we measured 208,774 of 208,774 hits flagged. The SARAO formula describes
# where TV channels *may* sit, not which are actually transmitting near the
# Karoo. Masking all of them is not defensible.
#
# Use --derive-mask instead: it finds the channels actually radiating into
# these data. Enable this comb with --dtv only if you know which channels are
# live and narrow DTV_CHANNEL_RANGE accordingly.
DTV_CHANNEL_RANGE = (21, 68)
DTV_FORMULA = lambda ch: (306.0 + 8.0 * ch)   # noqa: E731
DTV_HALFWIDTH_MHZ = 4.0


def build_mask_table(include_itu: bool = True,
                     include_dtv: bool = False,
                     extra: list[tuple[float, float, str, str]] | None = None):
    """Return the full list of (f_lo, f_hi, label, source) tuples in MHz."""
    table = [m for m in MASKS if include_itu or m[3] != "ITU"]

    for f0, label, src in POINT_MASKS:
        table.append((f0 - POINT_HALFWIDTH_MHZ, f0 + POINT_HALFWIDTH_MHZ,
                      label, src))

    if include_dtv:
        for ch in range(DTV_CHANNEL_RANGE[0], DTV_CHANNEL_RANGE[1] + 1):
            fc = DTV_FORMULA(ch)
            table.append((fc - DTV_HALFWIDTH_MHZ, fc + DTV_HALFWIDTH_MHZ,
                          f"digital TV ch{ch}", "SARAO"))

    if extra:
        table.extend(extra)

    return sorted(table)


def load_empirical_mask(path):
    """Read a CSV written by track_a_filter.py --derive-mask."""
    import csv
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append((float(row["f_lo_mhz"]), float(row["f_hi_mhz"]),
                        row.get("label", "empirical"), "empirical"))
    return out


if __name__ == "__main__":
    tbl = build_mask_table()
    print(f"{len(tbl)} mask ranges\n")
    print(f"{'f_lo (MHz)':>12} {'f_hi (MHz)':>12}  {'source':<10} label")
    for lo, hi, label, src in tbl:
        print(f"{lo:12.3f} {hi:12.3f}  {src:<10} {label}")
