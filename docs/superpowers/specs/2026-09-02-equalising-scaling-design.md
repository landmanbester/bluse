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
| cost, lband_short (593k rows) | 9.3 s per iteration, ~56 s for a fit | **1.5 ms** |
| 35k-sample fit vs full population | **29.5% max** weight error, 14.9% median | **0.78% max**, 0.17% median |
| determinism | depends on seed and iteration count | fully determined |
| targets the primary statistic (P0-3) | **yes** | no — k-NN share still 1.25 off equal |

The iteration does not converge because the k-NN graph is itself a function of
`w`: reweighting changes which points are neighbours, so the map is not a
contraction and the answer depends on where you stop.

Equalising the **global** share has a closed form. Since
`share_global_j ∝ w_j²·var_j`, setting them equal gives `w_j ∝ 1/σ_j` — one
pass, no iteration, no seed. On the robust-scaled matrix (already clipped to
±5) the variance is well behaved, so the usual objection to standardisation
does not apply here.

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

1. `leaf` + equalised reproduces family ARI ≥0.60 at 36 families and a median
   family span ≤120 MHz on sband_short, against the 0.4888 / 265.0 MHz
   baseline. (The P0-2 figures are 0.6659 / 78.6 MHz; the margin allows for the
   corrected k-NN estimator and a possible change of strategy.)
2. `eom` + equalised emits the warning on both entry points and still runs.
3. Weights are exactly 1.0 for any `boolean` or `flag` column.
4. The Bench and the CLI produce identical weights for identical inputs.
5. Chosen strategy and its parameters are pinned by a decision test, with the
   experiment written up in `docs/`.
6. No existing measured number moves: the full acceptance run under `robust`
   reproduces `eom` 72 clusters / 0.776% / famARI 0.5190 and `leaf` 2162 /
   6.820% / 0.1077.

## 7. Explicitly out of scope

Equalising on top of `quantile` or `none` (unmeasured); per-file weight
presets; exposing the strategy as a user knob once measured — an unresolved
knob is what this repo avoids; and any change to `f02`, settled in P1-4.
