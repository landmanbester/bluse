# P1-4: the `f02` ordinal rework

**Date:** 2026-09-01
**Question:** `f02_abs_drift` is a 42-level ordinal carrying a 26.6% zero-drift
tie that `quantile-normal` throws to −5.199, leaving it the least-weighted
column (1.7% global, 0.8% k-NN distance share). P0-2 showed that contribution
equalisation, which therefore hands it a weight of 5.216, reproduces the entire
`eom` collapse. The planned repair: a `zero_drift` indicator plus the drift
magnitude on its native linear grid.
**Answer:** Rejected. Every variant is worse under `eom` at every family count,
the indicator is separately dangerous, and — the point of the exercise — the
rework does **not** unblock the equalising scaling mode. P1-5 was never blocked
by `f02`.

---

## 1. What the raw feature actually is

Measured across all eight feature parquets. `driftRate` is already present in
the feature parquet, and `flag_zero_drift` already exists as a Track A flag, so
**no `driftSteps` plumbing is needed** — the constraint recorded in the TODO
does not bind for this design.

| file | rows | zero-drift | levels | step (Hz/s) | max |
|---|---:|---:|---:|---:|---:|
| lband_long | 557,690 | 46.55% | 50 | 0.00504 | 0.3329 |
| lband_short | 866,002 | 46.60% | 42 | 0.01025 | 0.4203 |
| mk_sample_hits | 15,119 | **0.00%** | 42 | 0.00504 | 0.2472 |
| sband_long | 36,132 | 32.95% | 42 | 0.00527 | 0.2161 |
| sband_short | 38,576 | 33.55% | 42 | 0.01071 | 0.4391 |
| uhf_long | 299,878 | 41.81% | 66 | 0.00204 | 0.1508 |
| uhf_short | 208,774 | 22.74% | 42 | 0.00856 | 0.3508 |

The lattice is exact — values are precisely k × step for k = 0…41 — and the
step spans 5.25× across files, confirming `driftSteps` is a per-file index that
is not comparable across files.

> **Updated 2026-09-02** for the repaired `lband_short.h5` and `uhf_long.h5`.
> Only the `lband_short` row moved, and only in rows and zero-drift fraction
> (463,625 / 26.75% on the retired `_clean` subset → 866,002 / 46.60% on the
> full file — the stripped region was disproportionately zero-drift). Every
> lattice constant is unchanged, so the conclusions drawn from this table
> stand. All the clustering in this document is on `sband_short`, whose feature
> parquet is byte-identical before and after the repair. **`mk_sample_hits` has no zero-drift hits at
all**, so a zero-drift indicator is a constant column there.

On the 34,933 sband_short rows reaching the clusterer the tie is 26.62%,
matching the recorded figure exactly.

## 2. Variants measured

All at `mcs=4 ms=8 epochs=8 batch=3000`, robust scaling, seeds 0–2, family ARI
restricted to jointly clustered points. Because a family count is a granularity
dial (P0-1), **every variant is swept across the whole cut chain** rather than
compared at one count — the first pass of this experiment compared only at 36
families and got a materially different (and wrong) answer.

`linear` replaces `quantile-normal` with a linear scale. Note this is
equivalent to transform `"none"` for a single file: robust scaling divides by
the IQR, which absorbs any constant factor exactly.

**Best family ARI achieved anywhere on the chain:**

| variant | `eom` | at | `leaf` | at |
|---|---:|---:|---:|---:|
| **baseline** (quantile-normal) | **0.519** | 36 | 0.607 | 8 |
| `linear` | 0.372 | 24 | **0.742** | 12 |
| `linear` + indicator | 0.288 | 8 | 0.602 | 8 |
| indicator only | 0.290 | 20 | 0.622 | 16 |

Under `eom` the baseline dominates at **every** family count — this is not a
plateau artefact. Under `leaf` the linear variant looks like a large win.

## 3. Why the `leaf` win is not a win

The region where `linear` leads (8–24 families) has a **narrow-cluster share of
0.000% for every leaf configuration**, with families spanning 240–280 MHz
median. That is the degenerate corner P0-1 already flagged: family ARI is
trivially high when families are few and broad. Restricted to counts that
produce any narrow families at all:

| leaf configuration | famARI @36 | narrow % | median span |
|---|---:|---:|---:|
| baseline | 0.4888 | 0.194 | 265.0 |
| baseline **equalised** | **0.6659** | 0.168 | **78.6** |
| `linear` | 0.4679 | 0.228 | 161.1 |
| `linear` equalised | 0.6051 | 0.178 | 155.0 |

The best configuration measured anywhere in this work is **baseline +
equalisation under `leaf`** — famARI 0.7683 at 16 families, 0.6659 at 36 where
the narrow share becomes non-zero. `linear` does not reach it.

## 4. The indicator is separately dangerous

A `zero_drift` boolean is the **lowest-share column in the matrix**: 1.68%
global, **0.33% k-NN**. Contribution equalisation therefore assigns it a weight
of **10.256** — twice the 5.216 that P0-2 showed destroys `eom`. Adding the
indicator would rebuild the P0-2 failure with a worse constant.

If equalisation ever ships it must skip `boolean` and `flag` columns, which is
what the registry's `kind` field is for. That is a constraint on P1-5, recorded
here, and it holds whether or not any indicator is added.

## 5. The rework does not unblock P1-5

This was the whole purpose of ordering `f02` first.

| | `eom` plain | `eom` equalised | `leaf` plain | `leaf` equalised |
|---|---:|---:|---:|---:|
| baseline | 0.5190 | 0.0438 | 0.4888 | **0.6659** |
| `linear` | 0.3359 | 0.1251 | 0.4679 | 0.6051 |

Repairing `f02` leaves `eom` under equalisation still broken (0.125 against
0.519 plain) and makes `leaf` under equalisation **worse** (0.666 → 0.605).

**This corrects P0-2's forward inference, not its measurement.** P0-2 measured
that boosting `f02` alone reproduces the `eom` collapse to three figures, and
that stands. What does not stand is the conclusion drawn from it — that
repairing `f02` would prevent the collapse. With `linear`, `f02`'s equalising
weight falls to 1.949 and the largest weight in the matrix is 2.199 on `f04`,
yet `eom` still collapses. So `f02` was the **vehicle** of the collapse, not
its cause: `eom`'s single root-level stability comparison is fragile to
reweighting in general, and any column that equalisation happens to amplify
will trigger it.

## 6. Is drift information being lost?

No — and this is why suppressing `f02` is defensible rather than merely
convenient.

**Zero-drift is a genuine RFI marker.** Among labelled sband_short rows, 28.14%
of RFI hits have exactly zero drift against 11.70% of beam-confined hits —
Fisher exact odds ratio **2.96**, p = 6.8 × 10⁻³¹. Physically expected:
terrestrial interference is stationary in frequency.

**But drift magnitude is a weak feature, and a better one already carries the
signal.** Mutual information against `weak_label`, all 15 columns:

| rank | column | MI (nats) |
|---:|---|---:|
| 1 | **`x01_drift_residual`** | **0.0289** |
| 2 | `f03_snr` | 0.0283 |
| 3 | `f01_frequency` | 0.0219 |
| … | | |
| 10 | **`f02_abs_drift`** | **0.0059** |
| 15 | `f10_timeseries_std` | 0.0013 |

`f02` ranks 10th of 15. The top-ranked feature is `x01_drift_residual` — the
scatter of the stamp's peak channel about a straight-line fit, i.e. how well
the signal follows *any* linear drift — which carries 4.9× more label
information than the drift rate itself. Drift is informative; drift
**magnitude** is not, and the feature that captures drift **coherence** is
already the strongest column we have.

## 7. Conclusion

Keep `quantile-normal` on `f02_abs_drift`. The transform's effect — pushing the
zero-drift slab to the bound and leaving the column contributing under 1% of
the k-NN distance — is **load-bearing**, not a bug: it suppresses a 26.6%
coincident slab that carries little family information, and every attempt to
give the column more weight costs reproducibility.

Shipped: nothing but the measurement. `features.py` now records why the
transform is deliberate, and a decision-pin test fails if it is changed without
a new experiment.

**P1-5 is unblocked and reframed.** It should ship for `leaf` only, with the
existing `f02`, skipping `boolean` and `flag` columns, and it should be
reported at a stated family count in the non-degenerate region (≥36).
