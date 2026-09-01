# P0-1: is there a structure-sensitive cut for the family matching?

**Date:** 2026-09-01
**Question:** `derive_cut_pct` is a granularity dial. The implementation
review (§1, option 2) proposed restricting the gap rule's search to merges
below ~the 90th percentile, so the root merges cannot dominate it, in the hope
of getting a cut that responds to structure. Does it?
**Answer:** No — and the framing was wrong. A cut rule cannot respond to
structure in the way the review intended, because a cut only ever chooses a
family count. Rejected the restricted gap rule; shipped `n_families=` instead.

Reproduce: `aug_2026_workshop/acceptance.py` for the baselines; the experiment
itself is pinned by `tests/unit/test_matching.py`.

---

## 1. The result that reframes the question

**Every horizontal cut of a fixed Ward tree is uniquely determined by the
number of families it leaves.** Cutting at distance `t` and asking for
`maxclust=k` produce the *same partition* whenever they produce the same
count.

Measured over 1,000 thresholds spanning the full height range of the real
sband_short trees (`eom`, k=72 and `leaf`, k=2,162), plus 400 on synthetic
noise and planted blobs:

| tree | thresholds tested | partitions differing from `maxclust` |
|---|---:|---:|
| sband_short `eom` | 500 | 0 |
| sband_short `leaf` | 500 | 0 |
| synthetic noise | 200 | 0 |
| synthetic blobs | 200 | 0 |

This is obvious in hindsight — the achievable partitions are a nested chain of
k−1 of them, and a horizontal cut just indexes into it — but it decides what
the experiment can possibly show. A cut rule **cannot select a better
partition**, only a point on a fixed chain. So "structure-sensitive cut" can
only mean *picks a count that moves with the data*. That is the property the
restricted gap rule was then measured against.

Pinned by `test_a_distance_cut_is_exactly_a_choice_of_family_count`.

## 2. The restricted gap rule fails on its own terms

**The bound is itself a granularity dial, and a far worse-behaved one than
`pct`.** Family count picked on sband_short, seed 0:

| rule | `eom` (k=72) | `leaf` (k=2,162) |
|---|---:|---:|
| `pct` p25 | 54 | 1,621 |
| `pct` p50 | 36 | 1,081 |
| `pct` p75 | 19 | 541 |
| `pct` p90 | 8 | 217 |
| gap, unbounded | 4 | 4 |
| gap, `max_pct=99` | 4 | 31 |
| gap, `max_pct=95` | 9 | 113 |
| **gap, `max_pct=90`** | **9** | **2,157** |
| gap, `max_pct=75` | 23 | 2,157 |

`pct` tracks its parameter smoothly and predictably (every value within
rounding of the k(1−p/100) dial). The gap rule's bound moves the `leaf` answer
from 113 families to 2,157 — no merging at all — on a five-point change in an
arbitrary parameter. It replaces one arbitrary knob with a more sensitive one.

**And it does not respond to structure.** The decisive test: 60 clusters in
15-D, identical in every respect except geometry — structureless, 3 planted
families, or 6 planted families — so a structure-sensitive rule must return
different counts and ideally the planted ones.

| case | `pct` p50 | gap | gap p95 | gap p90 | gap p75 |
|---|---:|---:|---:|---:|---:|
| structureless | 30 | 3 | 7 | 14 | 59 |
| 3 planted families | 30 | 3 | 6 | **30** | 30 |
| 6 planted families | 30 | 2 | 6 | 13 | 57 |

`pct` returns 30 everywhere, as documented — an honest dial. No gap variant
recovers a planted count, none distinguishes 3 families from 6, and every
variant returns values on structureless noise that are indistinguishable from
(or larger than) its values on strongly structured data. At `max_pct=90` it
returns *more* families on noise (14) than on 6 planted families (13).

**The fixture test was weaker evidence than it looked.** `rule="gap"` returns
3 on the synthetic fixture, which reads like structure recovery. Run against
20 structureless datasets of the same shape, it returns 2–4 every time and
exactly 3 in 5 of 20. It lands on the right answer partly by coincidence,
because root merges force it to a handful of families on any input. The test
is kept — getting the planted answer wrong would still be a regression — but
it is now labelled, and paired with
`test_gap_rule_is_not_evidence_of_structure_sensitivity`.

## 3. What the cut chain actually looks like

Since a rule only picks a count, the honest thing is to measure quality along
the whole chain. sband_short, seeds 0–2, family ARI restricted to jointly
clustered points:

**`eom`** — a broad plateau, then a cliff:

| families | 4 | 8 | 19 | 23 | 30 | 36 | 39 | 45 | 54 | 72 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| famARI | 0.477 | 0.403 | 0.519 | 0.519 | 0.519 | **0.519** | 0.519 | 0.261 | 0.199 | 0.033 |
| narrow % | 0.000 | 0.553 | 0.553 | 0.567 | 0.621 | **0.670** | 0.670 | 0.710 | 0.776 | 0.776 |
| med span (MHz) | 399.0 | 45.9 | 43.0 | 37.5 | 27.7 | 22.7 | 23.8 | 21.5 | 20.8 | 25.1 |

The default lands at 36: inside the plateau and at its best narrow share, with
about 10% headroom before reproducibility collapses between 39 and 45. That is
luck rather than measurement, but it is why the default has held up, and it is
now documented so a future change to `pct` is checked against the cliff.

**`leaf`** — no plateau, no knee, a pure monotone trade-off:

| families | 4 | 31 | 113 | 217 | 541 | 1,081 | 2,157 |
|---|---:|---:|---:|---:|---:|---:|---:|
| famARI | 0.762 | 0.508 | 0.343 | 0.238 | 0.151 | 0.106 | 0.032 |
| narrow % | 0.000 | 0.194 | 0.968 | 1.144 | 2.453 | 4.156 | 6.791 |
| med span (MHz) | 736.1 | 265.0 | 75.6 | 54.3 | 43.0 | 34.0 | 22.7 |

Reproducibility falls and coherence rises monotonically, so **there is nothing
for any rule to find** — the high ARI at 4 families is the degenerate answer
(zero narrow clusters, 736 MHz median span). For `leaf` the family count is a
trade-off the analyst has to state, not a quantity to be derived.

## 4. What shipped

**`n_families=`** on `matching.match()`, and `--match-families` on the CLI.
Given §1 this is the honest interface: it says which point on the chain you
want, without a derived threshold implying something was measured. It uses
`criterion="maxclust"`, which is exact under tied merge heights where a
height-based cut can overshoot, and it is clamped to `[2, k]`.

This parameter was already promised. `derive_cut_pct`'s docstring has told
readers to "prefer `n_families=`" since it was written, and no such parameter
existed anywhere in the codebase — a documented interface that was never
built. Found while checking what the rules had in common.

**Not shipped:** the `max_pct` bound. Adding it would put a third rule in the
codebase that is wrong everywhere, where the two existing failed rules are
each kept because they are right on some data. The measurement is recorded in
`derive_cut_gap`'s docstring with an explicit "DO NOT ADD A max_pct BOUND", so
the next reader finds the result rather than repeating the experiment.

## 5. Consequences

- **The provisional-taxonomy caveat stands, and is now sharper.** The family
  count is a choice, and §1 says no cut rule can turn it into a discovery. Any
  write-up of the first taxonomy should state the count it chose and why.
- **Quote `eom` family results at 19–39 families.** Outside that plateau the
  reproducibility that justifies matching is not there.
- **`leaf` family counts must be argued, not derived.** State the trade-off.
- **Finding a real family count needs a different instrument** — cluster
  persistence or DBCV from the `hdbscan` backend (P2), or the synthetic
  injections, which are the only true objective function available. Not
  another threshold rule on the same tree.
