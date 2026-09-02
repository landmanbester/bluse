# P0-3: calibrating a threshold for `share_knn`

**Date:** 2026-09-02
**Question:** `share_knn` shipped reported-but-unthresholded, because no value
for it had been measured when the flag rules were written and inventing a bound
would have been the unverified-claim pattern this review cycle exists to
correct. The baselines now exist. Calibrate it, and make it primary.
**Answer:** Done — `share-high` >2× an equal share, `share-low` <½×, plus a new
`share-disagree` flag at ≥2.5×. Along the way the estimator itself turned out
to be biased, which mattered more than the thresholds.

Reproduce with `bluse-cluster --file <name> --report`.

---

## 1. The estimator was biased, and had to be fixed first

`_shares_knn` subsampled 5,000 rows and then built the neighbour index **on the
subsample**. That thins the data, so each point's k nearest neighbours sit
further away than they truly are and the statistic drifts toward the global
share. This is a systematic bias, not sampling noise, and it is an order of
magnitude larger than the noise:

| index sample | mean \|error\| vs exact | worst column |
|---:|---:|---|
| 2,000 | 1.01 pts | 3.14 pts (`f01_frequency`) |
| 5,000 (the old default) | 0.66 pts | 2.18 pts (`f01_frequency`) |
| 20,000 | 0.17 pts | 0.49 pts |

Seed-to-seed noise at the same settings is ~0.05 pts, so the bias dominated it
by 13×. `f01_frequency` read **7.35% against a true 5.28%**.

**The fix is to build the index on every row and sample only the query
points.** Neighbourhoods are then the true ones and the sample contributes
variance only:

| query sample | old (subsampled index) | new (full index) |
|---:|---:|---:|
| 2,000 | 1.01 pts | **0.059 pts** |
| 5,000 | 0.66 pts | **0.044 pts** |
| 20,000 | 0.17 pts | **0.014 pts** |

15–23× more accurate at equal sample size — the new estimator at 2,000 queries
beats the old one at 20,000. Cost is 5.3 s on the largest file (395,896 rows)
against 0.9 s, which is small beside the clustering it accompanies. The default
sample is now 20,000, where the estimate is effectively exact.

**Every `share_knn` figure previously quoted in this repo was biased.** The
corrections on sband_short: `x03_channel_offset` 7.4% → **5.9%**,
`f01_frequency` 7.2% → **5.2%**, `f09_temporal_skew` 15.5% → **15.3%**,
`f02_abs_drift` 0.8% → **0.4%**. Live tables have been updated.

This does **not** disturb any conclusion in P0-1, P0-2 or P1-4: those were
measured by clustering and scoring the result, not read off these shares. The
`x03` correction in fact strengthens the finding — `x03` is not merely ordinary
locally, it sits *below* an equal share.

## 2. The thresholds

Calibrated across all seven per-file parquets, on the corrected estimator.
Columns flagged per file, out of 15:

| rule | mean flags/file | range |
|---|---:|---|
| **`share_knn` >2× / <½×** | **5.0** | 2–8 |
| `share_knn` >2× / <⅓× | 3.0 | 0–5 |
| `share_global` >2× / <½× (the old rule) | 7.4 | 6–11 |

The local statistic is already the more selective of the two at identical
multipliers, so `share_knn` inherits them unchanged — no new tuning constant,
and the rule reads the same as the one it replaces.

**`share-disagree`** is new, and is where `share_global` earns its keep: a
column whose two shares differ by ≥2.5× either way is one where the global
number misleads.

| ratio bound | mean flags/file | distinct columns ever flagged |
|---:|---:|---:|
| 2.0× | 5.0 | 11 of 15 (too loose) |
| **2.5×** | **2.1** | **6** |
| 3.0× | 1.9 | 5 (drops `f13_redness`) |

## 3. What this changes on real data

sband_short, corrected estimator, sorted by k-NN share:

| column | global | k-NN | flags |
|---|---:|---:|---|
| `f09_temporal_skew` | 6.7% | **15.3%** | **share-high** |
| `f07_kurt_bw_corr` | 13.1% | 12.8% | clip |
| `f12_bandwidth_hz` | 9.9% | 12.0% | tie |
| `f13_redness` | 6.1% | 11.0% | — |
| `f10_timeseries_std` | 8.8% | 9.4% | — |
| `x02_time_occupancy` | 2.5% | 8.3% | tie, share-disagree |
| `x03_channel_offset` | **24.3%** | 5.9% | tie, **share-disagree** |
| `f06_bimodality` | 4.2% | 5.4% | — |
| `f01_frequency` | 3.2% | 5.2% | — |
| `x01_drift_residual` | 2.0% | 3.9% | — |
| `f03_snr` | 6.7% | 3.8% | — |
| `f11_spectrum_std` | 5.3% | 2.5% | share-low |
| `f05_spectral_kurtosis` | 2.6% | 2.1% | share-low |
| `f04_spectral_skew` | 2.8% | 2.0% | share-low |
| `f02_abs_drift` | 1.7% | 0.4% | tie, share-low, share-disagree |

Two flags move, and they are the two the whole audit was built to settle:

- **`x03_channel_offset` loses `share-high`.** It was indicted by the original
  review as over-weighted at 24.3%, on the strength of the statistic that
  overstates it. Locally it is 5.9% against a 6.7% equal share — *below*
  parity. It now carries `share-disagree`, which says the true thing.
- **`f09_temporal_skew` gains `share-high`.** At 6.7% global it sat exactly on
  an equal share and the old rule could never have flagged it; it is the
  largest local contributor in the matrix.

The golden tests were updated to assert this, including renaming
`test_x03_is_over_weighted_on_sband_short`, whose name encoded the claim the
measurement refutes.

## 4. Caveats worth carrying

- **A column's local share is a property of the file, not of the feature.**
  `f10_timeseries_std` runs 2.96% on lband_short_clean and 12.29% on uhf_long;
  `f02_abs_drift` runs 0.31% to 6.31%. A flag describes this run, not the
  feature in general.
- **The two share flags remain ONE observation.** Shares sum to 1, so a column
  at 24% mechanically depresses every other toward the lower bound.
- **`share_knn` is not sample-size invariant** even with the corrected
  estimator — it is exact only at full sample. Any figure quoted should state
  the sample it was measured at; the default is 20,000 queries.
