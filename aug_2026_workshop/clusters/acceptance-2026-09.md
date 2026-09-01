# Acceptance measurements — Cluster Bench review work, 2026-09

Reproduce with `uv run python aug_2026_workshop/acceptance.py`.

All figures on `sband_short_features.parquet`, 34,933 rows reaching the
clusterer, at Bench defaults: `mcs=4`, `ms=8`, `epochs=8`, `batch=3000`,
`scaling=robust`, f08 off, 15 features. Stability over seeds 0–2.

Spec: `docs/superpowers/specs/2026-09-01-cluster-bench-review-design.md` §12.

---

## The eight criteria

| # | criterion | expected | measured | |
|---|---|---|---|---|
| 1 | `f02` flags tie + share-low | 42 vals, tie 0.266 | 42, 0.266, share 1.7% | ✅ |
| 1 | `x03` flags share-high | share ≈24.3% | 24.3% | ✅ |
| 2 | `f07` clip fraction | ≈0.010 | 0.010 | ✅ |
| 3 | narrow share, eom / leaf | 0.78% / 6.8% | 0.776% / 6.820% | ✅ |
| 4 | epoch 1, then epochs 4–8 | 87.9%, then 0 | 87.9%, 5 dead epochs | ✅ |
| 5 | eom composite/restricted/noise | .024/.028/.999 | .0280/.0279/.9985 | ✅ |
| 5 | leaf composite/restricted/noise | .48/.032/.78 | .4799/.0316/.7820 | ✅ |
| 6 | family membership ARI | **no prediction** | **eom 0.5190, leaf 0.1077** | ✅ |
| 7 | eom-vs-leaf decision | recorded | made a user choice | ✅ |
| 8 | suite passes, 5 invariants | — | 48 tests | ✅ |

---

## Criterion 6 — the finding

This was the only criterion with no predicted value, and the plan committed in
advance to what each outcome would mean. The outcome is the strong one.

| | cluster ARI | family ARI | gain |
|---|---:|---:|---:|
| `eom` | 0.0279 | **0.5190** | **18.6×** |
| `leaf` | 0.0316 | 0.1077 | 3.4× |

**Cluster membership is not reproducible under either selection method.** Both
sit at the noise floor: two runs of an identical configuration, differing only
in shuffle seed, agree at ARI ≈0.03. That is because HDBSCAN mints local ids
`0..k-1` in every batch and the epoch loop runs many batches, so one physical
population becomes a fresh id in every batch and epoch it appears in.

**Grouping into families recovers it, decisively for `eom`** — 0.0279 → 0.5190,
with the narrow-cluster share essentially intact (0.776% → 0.670%) and a stable
family count across seeds (33–40 from 66–80 clusters). Matching is therefore a
correctness fix, not an enhancement, and this is the strongest single result in
the work programme.

The gain is much smaller for `leaf` (3.4×), which is part of the trade-off
below.

---

## `eom` vs `leaf` — why this is a user choice

They win on different axes, so neither is a default waiting to be tuned.

| | `eom` | `leaf` |
|---|---:|---:|
| clusters | 72 | 2,162 |
| clustered | 100.0% | 50.3% |
| median size | 11 | 6 |
| **narrow share, <1 MHz** | 0.776% | **6.820%** |
| narrow share, <0.1 MHz | 0.587% | **4.400%** |
| permutation null | 0.000% | 0.020% |
| **family membership ARI** | **0.5190** | 0.1077 |
| epochs doing work | 3 of 8 | **8 of 8** |
| minority-class enrichment | **12.33%** | 1.19% |
| AMI | 0.0048 | 0.0026 |

`leaf` wins coherence 8.8×; `eom` plus matching wins reproducibility 4.8× —
**but both of those comparisons are at mismatched granularity**, see below.
`leaf` also restores a working epoch loop — 12.9, 7.3, 6.5, 5.7, 5.1, 4.9, 4.2,
3.8 per cent removed per epoch, against `eom` spending 87.9% in a single pass
and doing nothing at all in epochs 4–8.

Pick `leaf` to build a taxonomy; pick `eom` to rank candidates. Both entry
points carry these numbers next to the control. `eom` remains the default
because every committed result in this workspace was produced with it.

---

## Matched granularity — added after the implementation review

The comparison above puts `eom` at 36 families against `leaf` at 1,081, because
the default cut returns about *k*/2 groups and *k* differs by 30×. Finer
partitions are intrinsically harder to reproduce, so the reviewer asked whether
the reproducibility gap was method or granularity. It was largely granularity —
**and so was the coherence gap, in the opposite direction.**

Cut both to the same family count with `criterion="maxclust"`, 3 seeds:

| target families | method | family ARI | narrow % | median span |
|---:|---|---:|---:|---:|
| 20 | `eom` | 0.5189 | 0.567 | 40.3 |
| 20 | `leaf` | 0.4524 | 0.000 | 273.8 |
| **36** | **`eom`** | **0.5190** | **0.670** | 22.7 |
| **36** | **`leaf`** | **0.4888** | **0.194** | 265.0 |
| 54 | `eom` | 0.1994 | 0.776 | 20.8 |
| 54 | `leaf` | 0.4475 | 0.194 | 77.1 |
| 72 | `eom` | 0.0332 | 0.776 | 25.1 |
| 72 | `leaf` | 0.4170 | 0.774 | 76.3 |

**The 4.8× reproducibility gap shrinks to 6%** at 36 families (0.519 vs 0.489).
**And `leaf`'s 8.8× coherence advantage inverts**: coarsened to 36 families its
narrow share falls from 6.820% to 0.194%, *below* `eom`'s 0.670%, with a median
family span of 265 MHz against 22.7.

So `leaf`'s coherence is a property of its **fine granularity**, not of the
extraction method. Its 2,162 small groups are frequency-coherent; merged to 36
they are not, because the merging is driven by centroid proximity in a feature
space where frequency is one column among fifteen.

Two things follow.

**The user-choice framing survives, but the axes are different from what the PR
said.** It is not "coherent vs reproducible method". It is a granularity choice:
~2,000 small coherent groups that individually do not reproduce across seeds
(`leaf`), or ~36 large groups that reproduce well and stay reasonably narrow
(`eom` + matching). Both are legitimate; they answer different questions.

**`eom`'s default is now positively justified**, not merely precedent. At every
matched family count from 20 to 36 it is at least as reproducible and 3–∞×
more coherent. Note `eom` is non-monotone — its ARI collapses from 0.519 at 36
families to 0.033 at 72 (i.e. unmatched) — so *matching is doing the work*, and
`eom` without `--match` is the worst of the options measured.

One genuinely interesting cell: `leaf` cut to 72 families scores ARI 0.4170 at
narrow 0.774%, against `eom`'s raw 72 clusters at 0.0332 / 0.776%. At equal
group count and equal coherence, `leaf`-plus-matching is **12.6× more
reproducible** than unmatched `eom`. Anyone comparing to the committed results,
which are all unmatched, should know that.

### The headline survives its own null

`metrics.coarsening_null()` permutes the cluster→family assignment while
preserving family sizes, so it measures what family ARI would be if matching
grouped clusters arbitrarily. Measured in the Bench: **family ARI 0.519 against
a null of 0.020.** Coarsening does not inflate ARI; ARI's chance correction
handles it. Independently confirmed by the reviewer on structureless data
(cluster ARI 0.00005 → family ARI −0.00003).

---

## New findings, not anticipated by the spec

### 1. The k-NN share overturns "x03 dominates"

`share_knn` shipped reported-but-unthresholded because no value for it had been
measured. Now measured:

| column | global share | k-NN share |
|---|---:|---:|
| `x03_channel_offset` | 24.3% | **7.4%** |
| `f09_temporal_skew` | 6.7% | **15.5%** |
| `f07_kurt_bw_corr` | 13.1% | 13.0% |
| `f02_abs_drift` | 1.7% | **0.8%** |

`x03` collapses to roughly its equal share (6.7%) locally, and
`f09_temporal_skew` — benign on every global statistic — becomes the largest
local contributor. **HDBSCAN responds to local density, so the k-NN column is
the more relevant one**, and the "`x03` is over-weighted 3.6×" reading that the
review, the addendum and our own response all rested on does not survive it.

`f02` behaves exactly as predicted: its 26.6% tie *halves* its local share,
because a tied column contributes zero to every tie–tie pair and tied points are
disproportionately near neighbours. The gap between the two columns is the tie
diagnostic, as designed.

**These are the baseline values for thresholding `share_knn` in a follow-up.**

### 2. Enrichment discriminates where AMI does not

`eom` concentrates the minority class in **12.33%** of clustered hits against
`leaf`'s **1.19%** — a 10× separation. AMI sees the same two runs as 0.0048 and
0.0026, a 1.8× separation in the third decimal. The hypergeometric enrichment
metric that replaced AMI as the labelled proxy earns its place.

### 3. The narrow share is not an artefact of the threshold

Reported at two thresholds, the ratio holds: 8.8× at 1 MHz (0.776 / 6.820) and
7.5× at 0.1 MHz (0.587 / 4.400). And against a size-preserving permutation null,
`leaf`'s 6.820% is **341×** its null of 0.020%. The confound the null was built
to control for is not driving the result.

### 4. The provisional-taxonomy hedge points the other way

The reviewer's addendum §7 predicted families would be "grouped substantially by
channel offset and by a clipped correlation coefficient", and that the taxonomy
must be re-derived once contribution-equalising scaling reduces those columns'
weight. Dropping `x03` and `f07` outright — the crudest possible version of that
fix — makes families **less** coherent, not more:

| | clusters | families | narrow share | median span |
|---|---:|---:|---:|---:|
| `leaf`, all 15 features | 2,162 | 1,081 | **6.820%** | 34.0 MHz |
| `leaf`, without x03+f07 | 2,308 | 1,154 | **2.969%** | 33.9 MHz |

The narrow share more than halves. This is not a controlled comparison — the
clustering itself changes with the feature set, not just the matching — but it
is evidence *against* the assumption that down-weighting the two dominant
columns will improve the taxonomy. Combined with finding 1, the case for the
deferred scaling work is weaker than either review document assumed, and it
should be evaluated against `narrow_frac` before being built.

---

## Caveats

- **The first family taxonomy is provisional**, for the reason in the spec §7 —
  though finding 4 suggests the direction of the correction is not the one
  predicted.
- **Enrichment must not be compared across different `min_cluster_size`.** At
  the measured global rate of 3.13%, only a *fully* confined cluster of 4 clears
  Benjamini–Hochberg; 3-of-4 gives p ≈ 1.2 × 10⁻⁴ against a threshold near
  2.4 × 10⁻⁵ and fails.
- **`noise_agreement` is degenerate for `eom`** (0.9985), which clusters 99.9%
  of points. Recorded, not to be gated on.
- **The two share flags are one observation.** Shares sum to 1, so a column at
  24% mechanically depresses every other toward the lower bound.
- AMI and enrichment are proxies for "is this rediscovering beam multiplicity",
  not detection metrics. `weak_label == 0` means *spatially confined*, not
  *verified clean*.
