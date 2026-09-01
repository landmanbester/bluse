# P0-2 — does contribution-equalising scaling help? Measured

**Date:** 2026-09-01. **Answer:** yes under `leaf`, catastrophically no under
`eom`, and the vehicle is `f02`.

> **SUPERSEDED IN PART, 2026-09-01.** This page originally concluded that the
> `f02` rework was a **prerequisite** for the scaling work. That inference was
> tested in P1-4 and is **false** — repairing `f02` leaves `eom` under
> equalisation broken (family ARI 0.125 against 0.519 plain) and makes `leaf`
> under equalisation *worse* (0.666 → 0.605). The measurements on this page
> stand; the ordering they implied does not. `f02` was the vehicle of the
> `eom` collapse, not its cause: `eom`'s single root-level stability
> comparison is fragile to reweighting in general. See
> [`f02-rework-experiment-2026-09.md`](f02-rework-experiment-2026-09.md).

Reproduce: `/tmp/.../p0_2.py` (throwaway); method and numbers below.

---

## Setup

`sband_short`, 34,933 rows, 15 features, robust-scaled as usual. A weight
vector is then applied per column, `Z * w`, and clustering runs on that.

Equalisation solves for `w` such that every column's **k-NN** distance share is
equal. Share ∝ variance ∝ *w*², so iterating `w ← w·√(target/share)` converges
in a few passes: shares go from **0.9%–15.4%** to **6.1%–7.2%**.

The resulting weights are informative on their own:

| column | weight | why |
|---|---:|---|
| `f02_abs_drift` | **5.216** | most under-weighted locally (0.8% k-NN share) |
| `f04_spectral_skew` | 1.623 | |
| `f05_spectral_kurtosis` | 1.464 | |
| … | | |
| `f07_kurt_bw_corr` | 0.169 | |
| `f09_temporal_skew` | **0.165** | largest local contributor (15.4%) |

**All comparisons below are at matched family count.** The matching cut is a
granularity dial, so comparing configurations at differing family counts is
confounded — and the configurations here move *k* a great deal (72 → 38 → 162).

---

## Result

### `leaf` — equalisation is a clear win

| config | k | narrow % | family ARI | median span |
|---|---:|---:|---:|---:|
| baseline | 2,162 | 0.194 | 0.4888 | 265.0 |
| **EQUALISE (k-NN)** | 2,779 | 0.168 | **0.6659** | **78.6** |

at 36 families; and at 500 families, family ARI **0.1552 → 0.3031** with narrow
**2.345% → 2.800%**. Reproducibility improves 36–95%, median family span drops
from 265 MHz to 79 MHz, and coherence is flat-to-better. Unambiguous.

### `eom` — equalisation is catastrophic

| config | k | narrow % | family ARI | median span |
|---|---:|---:|---:|---:|
| baseline | 72 | 0.670 | 0.5190 | 22.7 |
| EQUALISE (k-NN) | 38 | 0.037 | **0.0438** | 285.9 |
| **ONLY `f02` ×5.2** | 25 | 0.026 | **0.0347** | **285.9** |
| EQUALISE, `f02` held at 1 | 162 | **0.727** | 0.4459 | 40.9 |

**`f02` alone accounts for the whole collapse.** Boosting only `f02` reproduces
it — family ARI 0.0347 against 0.0438, and a median span of 285.9 MHz in *both*
cases, identical to three figures. The other fourteen weights contribute
essentially nothing to the damage. Hold `f02` at unity and `eom` recovers, with
a narrow share slightly *above* baseline (0.727% against 0.670%).

---

## Why

`f02_abs_drift` is a 42-level ordinal with a **26.6% tie at −5.199** — the
zero-drift slab, and the subject of the original review's Finding 1.
Equalisation sees it as the most under-weighted column and amplifies it 5.2×,
which turns a quarter of the dataset into a dense coincident slab dominating the
metric.

The two selection methods respond oppositely because of where they make their
decision:

- **`eom`** makes one stability comparison at the root of the condensed tree.
  An amplified slab is exactly the density spike that wins that comparison, so
  *k* collapses 72 → 25 and every family becomes a band-spanning blob.
- **`leaf`** takes the condensed tree's leaves and never makes the root
  comparison. The slab becomes a set of well-populated leaves instead, which
  *reproduce well* across seeds — hence family ARI *rising* to 0.586 under
  `f02`-only boosting.

This is the same bistability that set `min_samples=8` (AGENTS.md gotcha 9),
reached from the other direction.

---

## What this changes

1. **The scaling spec is worth writing** — a 36–95% reproducibility gain and a
   3.4× reduction in median family span under `leaf` is a real result.
2. ~~**`f02`'s ordinal rework is a PREREQUISITE, not a parallel item.**~~
   **Wrong, and measured wrong in P1-4.** Equalising a column with a 26.6% tie
   does amplify the tie — that part is measured here and stands — but repairing
   the tie does not prevent the collapse, because equalisation then amplifies
   whichever column is next-lowest. Ship equalisation for `leaf` only, with the
   existing `f02`, skipping `boolean` and `flag` columns (a zero-drift boolean
   would draw a weight of 10.256, twice `f02`'s 5.216).
3. **The benefit is method-dependent** and must be reported per method.
   Equalisation is not a universal improvement, and anyone quoting it should say
   which selection method it was measured under.
4. **Weight caps do not rescue it** — capping at 2.0 still leaves `eom` at family
   ARI 0.0742. The fix is structural, not a clamp.

## The earlier ablation, finally explained

The pre-review experiment dropped `x03` and `f07` and read the resulting
coherence loss as evidence against equalisation. The implementation review said
that removed a well-behaved column and proved nothing about equalisation. Both
are right, and now there is a positive result to replace it: equalisation does
help, under `leaf`. It does not need `f02` moved out of the way first — the
best configuration measured in this work is equalisation over the **existing**
feature set under `leaf`, family ARI 0.7683 at 16 families and 0.6659 at 36,
where the narrow share becomes non-zero.
