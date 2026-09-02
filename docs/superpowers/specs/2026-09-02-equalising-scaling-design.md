# Contribution-equalising scaling — design

**Date:** 2026-09-02 · **Item:** P1-5 · **Branch:** `feature/equalising-scaling`
**Status:** design agreed, one question deliberately left to measurement.

---

## 1. Goal

Robust scaling equalises the **interquartile range**, but HDBSCAN responds to
**variance**, and the IQR-to-variance ratio depends on distribution shape. On
sband_short the per-column contributions to the distance therefore run 1.7% to
24.3% globally against an equal share of 6.7%. Add a scaling mode that targets
the distance share directly rather than a spread proxy.

**The number to beat**, measured in P0-2 and re-confirmed in P1-4, on
sband_short at Bench defaults, `leaf`, seeds 0–2:

| configuration | family ARI @36 | @16 | narrow % @36 | median span @36 |
|---|---:|---:|---:|---:|
| baseline (`robust`) | 0.4888 | — | 0.194 | 265.0 MHz |
| **equalised** | **0.6659** | **0.7683** | 0.168 | **78.6 MHz** |

## 2. Constraints already established by measurement

Every one of these is a finding from earlier work, not a preference.

1. **`leaf` only.** Under `eom` equalisation collapses family reproducibility
   (0.519 → 0.044). P1-4 showed no `f02` treatment rescues it: `eom`'s single
   root-level stability comparison is fragile to reweighting in general, and
   equalisation amplifies whichever column is locally weakest. Weight caps do
   not help (capped at 2.0, `eom` still sits at 0.0742).
   *Decided:* warn loudly, still run. Refusing would block reproducing the
   measurement; running silently would let someone quote a broken number.
   *What was actually tested, since a reader will ask:* P0-2 measured
   equalisation with `f02`'s weight **pinned to 1 while every other column was
   equalised** — `eom` recovered to 0.4459 from 0.0438 — and P1-4 measured three
   *reworkings* of `f02`'s transform. **Dropping the `f02` column entirely was
   never tested.** Task 2 now does, because the closed form makes the question
   sharper (see §3).
2. **Skip `boolean` and `flag` columns.** A low-variance indicator draws a huge
   equalising weight — a zero-drift boolean measured 0.33% k-NN share and would
   have drawn 10.256, twice the constant that destroys `eom`. This is what the
   registry's `kind` field is for. Presently a guard rather than a live case:
   no boolean is registered, and `*_saturated` flags are already excluded from
   the matrix.
3. **Report at a stated family count ≥36.** Below ~30 families every `leaf`
   configuration has a 0.000% narrow share and famARI there is meaningless.
   Family count is a granularity dial, so all comparisons use `n_families=`.
4. **`f02` stays on `quantile-normal`.** P1-4 measured and rejected the rework.
5. **`share_knn` is the statistic HDBSCAN responds to** (P0-3), which is what
   makes §3 a real question rather than an obvious one.

## 3. The open question: which target, and how

P0-2's result was produced by an iterative fixed point on the **k-NN** share,
`w ← w·√(target/share)`, six iterations. Profiling that procedure for
production use exposed three defects:

| | iterative k-NN | closed form on the global share |
|---|---|---|
| converges | **no** — `max\|share/equal − 1\|` is still 0.33 after 8 iterations and falling slowly, while weights diverge from 0.500–3.047 to 0.044–10.347 | exact in one step |
| cost of the weights, lband_short (593k rows) | 9.3 s per iteration, ~56 s for a fit | **1.5 ms** |
| 35k-sample fit vs full population | **29.5% max** weight error, 14.9% median | **0.78% max**, 0.17% median |
| determinism | depends on seed and iteration count | fully determined |
| targets the primary statistic (P0-3) | **yes** | no — k-NN share still 1.25 off equal |

The iteration does not converge because the k-NN graph is itself a function of
`w`: reweighting changes which points are neighbours, so the map is not a
contraction and the answer depends on where you stop.

**The cost row above is the cost of computing the weights, not of calling
`equalising_weights()`.** The `info` block reports achieved deviation for both
statistics, and `_shares_knn` dominates that: measured on sband_short
(34,933 x 15), the closed-form weights take 1.7 ms, `_shares` 1 ms, and
`_shares_knn` **1,311 ms** at `knn_sample=20_000` (435 ms at 5,000). So the
mode advertised as free would cost ~1.3 s on every `scale()` call — in
`cluster()` per run, in `audit()` per rail render, and once per seed in
`/stability`. Hence `with_info=False` in the plan's Task 1, with `scale()`
using it. (A reviewer measured 10,953 ms for the same call; that is 8.4x our
figure and is consistent with the pre-P0-3 estimator, which built its
neighbour index on the subsample. `audit()`'s default is now 20,000, so
`equalising_weights` matches it rather than reverting to 5,000.)

Equalising the **global** share has a closed form. Since
`share_global_j ∝ w_j²·var_j`, setting them equal gives `w_j ∝ 1/σ_j` — one
pass, no iteration, no seed.

**Name it for what it is: winsorised standardisation.** Dividing by the IQR and
then by the standard deviation of the result is dividing by the standard
deviation of the original, so `robust` → clip ±5 → `1/σ` is exactly "winsorise
at ±5 IQR units, then z-score". The robust step contributes nothing except
*where the clip lands*, which makes the clip the visible design decision. This
is a perfectly good transform; it is just a much smaller idea than
"contribution-equalising" implies, and the write-up should say so.

**What it cannot do, measured on sband_short before Task 2 runs.** Rank
correlation between the closed-form weight and the k-NN share is −0.68, so it
pushes in broadly the right direction. But its two largest actions are both
wrong by the local measurement:

| column | k-NN share | closed-form weight | |
|---|---:|---:|---|
| `f02_abs_drift` | 0.41% (lowest) | **1.594 (highest)** | amplifies the 26.6% zero-drift tie |
| `f09_temporal_skew` | 15.26% (highest) | 0.794 | barely touches the column P0-3 indicts |
| `x03_channel_offset` | 5.85% (below equal) | **0.430 (lowest)** | suppresses a column that is already fine locally |

`f02` has the smallest post-robust σ (0.4405) precisely because the tie that
inflates its IQR has been divided out, so the closed form hands it the largest
weight in the matrix. That is the P0-2 pathology in miniature — 1.594x rather
than 5.216x, but the same column and the same mechanism. This is a **stated
prior, not a result**: it raises the chance that A underperforms, and it is why
Task 2 prices the k-NN target directly.

So the cheap, deterministic, reproducible method targets the statistic we
demoted, and the method targeting the statistic we promoted is ill-behaved.

**This is settled by measurement, not by argument.** Task 2 of the plan
compares three candidates against the objective that matters — family ARI,
narrow share and median family span at matched family count under `leaf` — and
the winner becomes the mode. The candidates:

- **A. closed form**, `w ∝ 1/σ` on the robust-scaled matrix.
- **B. iterative k-NN**, damped (`w ← w·(target/share)^α`, α ≤ 0.5) with a
  weight cap, iteration count fixed by measurement.
- **C. closed form seeded, then two damped k-NN steps** — start from A, refine
  toward the local statistic without letting the weights run away.
- **D. diagnostics, to price the k-NN target and close the `f02` question**:
  A with `f09` hand-set to its equal *local* share; A with `f02` pinned to
  weight 1; and `f02` dropped from the matrix entirely. These are not
  candidates to ship — they say whether the k-NN target buys anything real and
  whether `f02` is still the vehicle.

If A wins or ties, it ships: it is 350× cheaper per pass, deterministic, and
sample-stable, and those properties are worth more than a marginal score. If B
or C wins materially, it ships with its iteration count pinned by a test and
its sample-sensitivity documented as a known limitation of Bench-vs-CLI
agreement.

## 4. API

Agreed with the user: **a new value in the existing `scaling` enum**, not an
orthogonal flag. Only the robust base is measured, so this builds exactly what
the evidence covers and adds no signature to any call site.

```
scale(X, "robust-equalised", stats, *, kinds=None, columns=None)
--scaling {robust,quantile,none,robust-equalised}
Bench dropdown:  "robust + equalised — target equal contribution"
```

`kinds`/`columns` are new keyword-only optional arguments so the skip rule in
§2.2 can be applied; every existing call site keeps working unchanged.

Weights are computed inside `scale()` from the matrix it is already scaling,
so the Bench's D-4 fit/transform split carries over: `stats` still comes from
the full population. The weights themselves are fit on the rows passed in,
which for the Bench is its 35k sample — quantified in §3 and, for candidate A,
negligible.

## 5. Surfaces

- **`diagnostics.equalising_weights(Z, *, kinds, columns, ...)`** returns
  `(weights, info)`; `info` carries the strategy, iterations, achieved
  `max|share/equal − 1|` for both statistics, and which columns were skipped.
- **`audit()`** accepts the new scaling so the rail's shares describe the
  metric actually in use. Its `clip_frac` test currently keys on
  `scaling == "robust"`; the equalised mode also clips and must be included, or
  it will silently report 0.0.
- **CLI**: `--scaling robust-equalised`; the `eom` warning; weights printed in
  `--report`; `scaling` and the weights recorded in the metrics JSON.
- **Bench**: dropdown option, per-column weight in the feature rail, the `eom`
  warning on the results panel, weights cached per (dataset, columns).

## 6. Acceptance

> **OUTCOME (2026-09-02): criterion 1 FAILED and remains open.** The shipped
> closed form gives 0.513 / 183.4 MHz. Why the bar cannot be reached is *not*
> established — tracked as P0-4 in [`../../TODO.md`](../../TODO.md) with the
> experiment that would settle it. Criteria 4a and 6 passed; 4a was corrected
> during implementation, since bit-identity across two code paths is not an
> achievable bar.

1. `leaf` + equalised reproduces family ARI ≥0.60 at 36 families and a median
   family span ≤120 MHz on sband_short, against the 0.4888 / 265.0 MHz
   baseline. (The P0-2 figures are 0.6659 / 78.6 MHz; the margin allows for the
   corrected k-NN estimator and a possible change of strategy.) **Narrow share
   is deliberately not a criterion**: at 36 families it reads 0.194% against
   0.168%, which on 34,933 clustered hits is 68 hits against 59. A nine-hit
   difference cannot carry a decision. See §6.1 below.
2. `eom` + equalised emits the warning on both entry points and still runs.
3. Weights are exactly 1.0 for any `boolean` or `flag` column.
4. **(a, assertable)** `equalising_weights` returns bit-identical weights for
   the same row set, through either entry point — a unit test on a synthetic
   fixture, gating commits.
   **(b, measured and recorded)** Bench-sample weights deviate from
   full-population weights by ≤1% max for the shipped strategy, measured on
   `sband_short` and `lband_short` and written into the experiment record.
   Criterion 4 was originally one line saying the two entry points agree, which
   is trivially true by determinism while the property anyone cares about —
   that a Bench configuration transfers to the CLI — is **false by
   construction**, because §4 fits weights on the rows passed in and those are
   35k against the full population. Candidate A measures 0.78% max / 0.17%
   median and passes; the iterative candidates measure 29.5% / 14.9% and would
   ship with that in the acceptance record as a known limitation, not a
   footnote. D-4's lesson was exactly that a fix shipped under the wrong claim
   keeps the real cause hidden.
5. Chosen strategy and its parameters are pinned by a decision test, with the
   experiment written up in `docs/`.
6. No existing measured number moves: the full acceptance run under `robust`
   reproduces `eom` 72 clusters / 0.776% / famARI 0.5190 and `leaf` 2162 /
   6.820% / 0.1077.

### 6.1 Why `narrow_frac` is reported but not ranked on

`narrow_frac` became the headline metric because `leaf` scored **6.820%**
against a 0.020% permutation null — a 341x enrichment at natural granularity.
Merging 2,162 clusters down to 36 families collapses it to 0.194%, a 35x loss,
and constraint 3 records that every `leaf` configuration reads **0.000%** below
~30 families. So at matched family count the metric is at its floor for
everything and has stopped discriminating.

**Median family span is the coherence metric at matched granularity** — 265.0 →
78.6 MHz is a 3.4x improvement in physical coherence and is the real result
here. Rank on family ARI, then median span. Report narrow share; do not rank on
it, and do not add it back later.

Also note that **family ARI is granularity-relative**, not an absolute measure
of reproducibility: the same configuration reads 0.6659 at 36 families and
0.7683 at 16. It is only meaningful against another number at the same family
count, so the 0.6659 must never be quoted against the 0.5190 from the earlier
`eom` work, which was measured at a different granularity.

## 7. Explicitly out of scope

Equalising on top of `quantile` or `none` (unmeasured); per-file weight
presets; exposing the strategy as a user knob once measured — an unresolved
knob is what this repo avoids; and any change to `f02`, settled in P1-4.
