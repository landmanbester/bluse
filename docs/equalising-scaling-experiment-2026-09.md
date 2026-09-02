# P1-5 Task 2: which equalising strategy, measured

**Date:** 2026-09-02 · **Branch:** `feature/equalising-scaling`
**Question:** the spec left one thing open — the cheap deterministic strategy
targets the *global* share (the statistic P0-3 demoted) and the strategy
targeting the *k-NN* share (the promoted one) does not converge. Which ships?
**Answer:** the closed form. The k-NN target is **ill-posed**, and its higher
score is not equalisation at all — it is feature selection down to two columns,
which a three-line hand-built control beats.

**This overrides the spec's stated tie-break**, and §5 explains why that is a
disqualification rather than a moved goalpost.

All on sband_short, `leaf`, `mcs=4 ms=8 epochs=8 batch=3000`, family ARI
restricted to jointly clustered points, compared at matched family count.

---

## 1. The candidate table

Seeds 0–2, the selection set.

| candidate | k | ARI@36 | span@36 | ARI@16 | fit s |
|---|---:|---:|---:|---:|---:|
| `robust` (baseline) | 2162 | 0.4888 | 265.0 | 0.4599 | — |
| **closed** | 2275 | 0.4949 | 206.6 | 0.6265 | 1.6 |
| knn i=2 | 2748 | 0.5881 | 236.9 | 0.6812 | 3.0 |
| knn i=4 | 2878 | 0.5921 | 148.3 | 0.6690 | 3.9 |
| knn i=6 | 2823 | 0.6993 | 143.7 | 0.7189 | 4.2 |
| knn i=8 | — | 0.7333 | 50.7 | 0.6908 | — |
| knn i=10 | — | 0.8271 | 30.1 | 0.8317 | — |
| knn i=12 | — | 0.8259 | 23.4 | 0.8006 | — |
| hybrid i=1 cap=2 | 2531 | 0.6361 | 156.0 | 0.6696 | 2.4 |
| hybrid i=2 cap=2 | 2538 | 0.5930 | 74.8 | 0.6250 | 3.7 |

On score alone the k-NN family wins by a mile and keeps improving with every
iteration. Three checks say not to believe it.

## 2. The k-NN target is unattainable, and we can say exactly why

`info["dev_trace"]` — `max|share/equal − 1|` after each step:

| iters | trace |
|---|---|
| uncapped | 1.289 → 1.064 → 0.750 → 0.593 → 0.600 → 0.611 → 0.606 → 0.598 → 0.586 → 0.564 → 0.543 → 0.520 |
| cap 2.0 | 1.289 → 1.066 → 1.199 → 1.207 → 1.212 → 1.218 (rises, then flat) |

The uncapped run falls to ~0.59 by iteration 4 and then **stops improving**,
drifting sideways for eight more steps. This is the plateau case: equal local
share is not reachable.

**The mechanism was already in the codebase.** `_shares_knn`'s docstring
records that a tied column contributes exactly zero to every tie–tie pair, and
that tie–tie pairs are disproportionately likely to be mutual near neighbours
*because they already agree in that coordinate*. So for a column with a large
tie fraction, no weight can lift its k-NN share to parity — the tied pairs
contribute zero however hard you push. `f02_abs_drift` carries a **26.6% tie**,
and the iteration duly drives its weight to the 5.0 ceiling by iteration 2 and
holds it there forever.

The k-NN equalisation target is therefore **ill-posed in the presence of ties**.
That is a stronger and more durable statement than "candidate B lost".

## 3. What the iteration actually does: n_eff 15 → 2.2

| | effective dimensionality | weight range |
|---|---:|---|
| `robust` (uniform) | 15.00 of 15 | 1.000–1.000 |
| **closed form** | **10.77** of 15 | 0.430–1.594 |
| k-NN i=12 | **2.22** of 15 | 0.065–5.000 |

(Participation ratio of the squared weights, on this one real matrix, where
the spread of per-column sigma is modest — 0.44 to 1.63.) The top three columns
hold 96% of the squared weight at i=12.

*Read that statistic only as a comparison between rows of this table.* A low
participation ratio of the weights is not by itself a defect: equalising
columns whose spreads differ 25x requires weights that differ 25x. The property
that actually distinguishes the two strategies is where the **contributions**
end up, and that is what `test_the_shipped_strategy_actually_equalises`
asserts — on the shares, not the weights. The first version of that test got
this wrong and failed, correctly. A mode named *contribution-equalising* that
concentrates 96% of the contribution into three of fifteen columns is not
equalising anything — it is doing the opposite, and shipping it under that name
would be false labelling.

The two survivors are `f02_abs_drift` (pinned at the ceiling from iteration 2)
and `f01_frequency`, which climbs 0.79 → 4.41 and reaches rank 2. Everything
GLOBULAR contributes — `f03`–`f13` — is driven to 0.06–0.6.

**`f01` rising is partly circular**, and it inflates the coherence metric:
`median_span_mhz` measures how tightly a family sits *in frequency*, and the
procedure is up-weighting frequency. Clusters become frequency-local by
construction and the metric rewards it.

## 4. The control that settles it

Give arbitrary column pairs the same weight-profile *shape* (two columns high,
the rest at 0.065) so `n_eff` matches, and score them identically:

| case | n_eff | ARI@36 | span@36 |
|---|---:|---:|---:|
| k-NN i=12 (the candidate) | 2.22 | 0.8259 | 23.4 |
| **control: `f02`+`f01`, hand-built** | 1.98 | **0.8501** | **12.1** |
| control: `f03`+`f07` | 1.98 | 0.5009 | 849.0 |
| control: `f10`+`f13` | 1.98 | 0.4255 | 694.6 |
| control: `x02`+`f06` | 1.98 | 0.4605 | 801.0 |

Two things follow, and they point in opposite directions for the two questions
in play:

- **It is not generic dimensionality reduction.** Arbitrary pairs at the same
  `n_eff` score 0.43–0.50, no better than the 15-column baseline. Collapsing to
  two dimensions buys nothing by itself.
- **It is entirely the specific pair.** A three-line hand-built weighting that
  keeps `f02` and `f01` and suppresses everything else **beats** the
  twelve-iteration fit on every metric — ARI 0.850 against 0.826, span 12.1
  against 23.4 MHz.

So the k-NN procedure is an expensive, non-convergent way of rediscovering
"cluster on drift rate and frequency". Whatever that is, it is not a scaling
mode, and it should not be shipped as one.

## 5. Why this overrides the stated tie-break

The spec fixed the rule in advance — rank on famARI, closed form wins only if
within 0.03 — precisely so the decision could not be reverse-engineered from
the numbers. By that rule the k-NN candidates win by 0.31.

The rule is not being bent. It presupposed that every candidate *was an
equalisation strategy*; §3 shows candidate B is not one, and §4 shows a trivial
control beats it, so it fails a precondition rather than losing on score. The
honest summary is that the spec asked the wrong question — "which strategy
equalises best" — when the answer turned out to be "one of them does not
equalise at all".

Recording this rather than quietly reporting a 0.83 is the point of having
written the rule down.

## 6. What ships

**`strategy="closed"`** — `w ∝ 1/σ` on the robust-scaled matrix, i.e.
**winsorised standardisation**: winsorise at ±5 IQR units, then z-score.

Replicated across **three independent seed triples** (0–2, 3–5, 6–8), so the
result is not selection noise:

| | ARI@36 | span@36 |
|---|---:|---:|
| `robust` baseline | 0.479 | 265.5 MHz |
| **closed form** | **0.513** | **183.4 MHz** |

A **+0.034 gain in family reproducibility and a 31% reduction in median family
span**, consistent in sign on all three seed triples. Deterministic, 1.7 ms,
and its 35k-sample weights match the full population to 0.78% (criterion 4b).

### It does not meet acceptance criterion 1

Criterion 1 asked for **ARI ≥0.60 and median span ≤120 MHz**. The closed form
delivers 0.513 and 183.4 MHz. **It fails, and the criterion should be
retired rather than the result massaged**, because the criterion was written
from P0-2's 0.6659 / 78.6 MHz — figures now understood to come from the
ill-posed k-NN procedure running on the pre-P0-3 biased estimator. They were
never a target a well-posed method could hit.

That is a decision for the reader of this document, not for its author. The
options are to ship the modest but real and well-posed gain, or to ship nothing
and keep `robust`. **Resolved 2026-09-02: shipped, with the criterion left
standing as failed rather than retired.**

**The remaining question is why the bar cannot be reached**, which this document
does not answer. The leading hypothesis — that 0.6659 was reachable only
because the configuration was degenerate (n_eff ~2, with `f01_frequency`
up-weighted and `median_span_mhz` a frequency statistic) — is inference from
the 12-iteration fit in §3, not a measurement of the 6-iteration configuration
that actually produced it. Tracked as P0-4 in [`TODO.md`](TODO.md), with the
experiment that settles it either way.

## 7. Side findings, recorded not acted on

- **`f02`+`f01` alone gives ARI@36 0.850 and a 12.1 MHz median span**, far
  beyond anything the full 15-column feature set achieves. That is a
  **feature-selection** result, not a scaling one, and part of it is the
  frequency circularity in §3. It deserves its own investigation — it suggests
  the other thirteen hand-crafted features may be contributing mostly noise at
  this granularity — and it belongs in the TODO, not smuggled in as a scaling
  mode. Note the minority-class enrichment does *not* follow the same ordering
  (11.7% for k-NN i=12, 16.4% for the hand-built pair, but 29.1% for the
  arbitrary `f03`+`f07` control), so reproducibility and enrichment are
  measuring different things and the pair is not simply "the best two columns".
- **Dropping `f02` entirely**, never tested before this: ARI@36 0.5057 against
  the closed form's 0.4949, median span 75.2 against 206.6. Better span, ARI
  within noise. The P1-4 conclusion that `f02` should keep its rank transform
  stands, but exclusion is now measured rather than assumed.
- **`f02` pinned to weight 1.0** under the closed form: 0.5032, indistinguishable
  from leaving it free. Under the closed form `f02` draws only 1.594, so there
  is nothing to rescue — unlike the 5.216 of P0-2.
- **`f09` hand-corrected toward equal local share**: 0.5339 against 0.4949, a
  real if small gain. The k-NN target *is* buying something on `f09`
  specifically; it is the `f02` tie that makes the general procedure ill-posed.
- **The boolean guard was exercised on the real code path** with a synthetic
  zero-drift indicator appended: weight exactly 1.0000, `skipped` reported.
