# Implementation review — `cluster-bench-metrics`

**Status:** external reviewer's review of the branch, against
`docs/superpowers/specs/2026-09-01-cluster-bench-review-design.md` and the
spec-feedback document.
**Read from:** the bundle at `origin/cluster-bench-metrics` (16 feature commits,
2,810 insertions), the three new modules in full, the wiring in both entry points, the
test suite, `acceptance-2026-09.md`, and the PR body.

**Tags:** **[repo]** read from the branch · **[measured]** computed by the implementor ·
**[verified]** re-derived independently by this reviewer during the review ·
**[hypothesis]** untested, with a stated test.

---

## 0. Verdict

Ship it, with one caption fix (§4) before merge and two items (§1, §2) resolved before
anyone quotes the `eom`-vs-`leaf` numbers in a talk.

The work is good and several parts of it are better than what was asked for. All four
spec-feedback items were taken, and taken properly rather than nominally: the test suite
is split by what it can honestly gate, invariant 2's seed is pinned with the reasoning in
the docstring, family ARI became criterion 6, and stability seeds are excluded from
`HISTORY` with a comment explaining why. Keeping the two *failed* cut rules in
`matching.py` with their failure modes documented, each tested on the data it is correct
for, is unusually disciplined and I would not want it tidied away later.

The bugs the new machinery caught in its own work — `--report` double-scaling (the tell
being `iqr_raw` of exactly 1.000 for every column), `narrow_enrichment` returning `inf`
for 0/0, the CLI's hardcoded `cluster_selection_method` producing a duplicate keyword —
are the sign of instrumentation that actually works. So is the CLI/Bench divergence:
`ari_composite` 0.1604 against 0.028 while `ari_restricted` is 0.0279 in both, traced to
3,643 rows marked `-2`. That is the three-number decomposition catching, on a second code
path, exactly the artefact it was built for. Worth keeping in the write-up.

Two findings below are substantive enough to change conclusions. Neither is a coding
error; both are about what the numbers mean.

---

## 1. `derive_cut_pct` is a granularity dial, not a data-derived cut

**This is the main finding of the review and it bears on questions 1 and 3.**

`match()`'s docstring says a hardcoded constant is not an option because the per-file
drift lattice spans 5.26×, so the cut is derived from the merge-height distribution
instead. The reasoning is right and the conclusion does not follow from the
implementation.

Cutting a Ward dendrogram at the *p*-th percentile of its merge heights performs the
lowest *p*% of the *n*−1 merges, so it returns approximately *k*(1−*p*/100) clusters —
**independent of whether the data has any structure at all**. On pure Gaussian noise in
15 dimensions: **[verified]**

| k | p25 | p50 | p90 | k(1−p/100) |
|---:|---:|---:|---:|---|
| 72 | 54 | 36 | 8 | 54 / 36 / 7 |
| 500 | 375 | 250 | 51 | 375 / 250 / 50 |
| 2,162 | 1,621 | 1,081 | 217 | 1,622 / 1,081 / 216 |

The branch's own measurements sit exactly on that line: `leaf` 2,162 → **1,081** families,
`eom` 66–80 clusters → **33–40** families **[repo]**. Every one is *k*/2 to within
rounding.

So `pct=50` is arithmetically "halve the number of clusters". That is a defensible default
and it clearly works, but three things follow:

**It answers question 3, and dissolves it.** A percentile of merge heights is
scale-invariant by construction, so `uhf_long`'s 5.26×-different lattice cannot move it.
The rule will transfer. But it transfers for a worse reason than robustness: it is
invariant to structure as well as to scale. Whatever the dendrogram looks like on
`uhf_long`, p50 will halve the cluster count there too. The risk was never
mis-calibration across files; it is that the rule never responds to the data in the first
place.

**The docstring should say so.** As written it argues against a hardcoded constant and
then ships one in percentile units. Recommend rewording to state plainly that the cut sets
family *granularity* as a fraction of the cluster count, that this is deliberate, and that
p50 was chosen because reproducibility peaks there.

**Two better options, in order of effort:**

1. **Expose the granularity directly.** `n_families=` or `frac=` is what the parameter
   actually controls, and it is easier to reason about than a percentile of a quantity
   with no natural units. Keep `pct` as an alias.
2. **Rescue the gap rule.** It failed because the final merges of any dendrogram dominate
   the gap statistic — measured, and correct. Restricting the gap search to merges below
   roughly the 90th percentile removes the root merges and makes it usable, and unlike the
   percentile rule it *does* respond to structure. Worth one experiment before accepting
   that no structure-sensitive rule works here. **[hypothesis]**

---

## 2. `eom` vs `leaf` is compared at mismatched granularity — question 1

Because the family count is *k*/2 by construction, the headline comparison is **`eom` at
36 families against `leaf` at 1,081 families**. Finer partitions are harder to reproduce,
so a large part of the 4.8× reproducibility gap could be granularity rather than method.

The branch's own sweep is the strongest evidence for this, and it points the same way:

| | p25 | p50 | p90 |
|---|---:|---:|---:|
| `eom` family ARI | 0.2137 | **0.5190** | 0.4031 |
| `leaf` family ARI | 0.0686 | 0.1077 | **0.2383** |

`eom` peaks at p50 and falls. `leaf` is **monotone increasing across the whole range
tested** — it never reaches a peak. So `eom` was evaluated at its optimum and `leaf` was
not, and the comparison stops at the point where `leaf` is still improving.

**The missing run:** cut `leaf` to ≈36 families, which is about p98 (2,162 × 0.017), and
compare there. If `leaf`-at-36-families reaches ARI ≥ 0.52 while keeping a narrow share
above `eom`'s 0.670%, then `leaf` dominates on both axes and the default question answers
itself. If its narrow share collapses on the way — and the p25→p90 column suggests it
might, 5.140 → 4.156 → 1.144 — then the trade-off is real and the user-choice framing is
right. Either way it is one sweep of an existing harness.

**On the default itself:** hold `eom`, but change the reason. The PR says nothing in the
measurements favours it and it wins only on precedent. That undersells it in one respect
and oversells the case in another. It genuinely wins enrichment 10× and family ARI 4.8×,
so something does favour it; but the family-ARI half of that is confounded as above, so
the honest statement is *the comparison is not yet decisive at matched granularity, and
`eom` is what produced every committed result*. Precedent is a perfectly good tie-breaker
when the measurement is not yet conclusive. It is a bad one when the measurement is
conclusive and pointing the other way, which is why it is worth doing the p98 run before
this framing hardens.

Presenting them as a user choice is right regardless, and carrying the numbers next to the
control is the right call. Flipping a toggle and seeing 2,162 clusters instead of 72 would
otherwise read as breakage.

---

## 3. The headline result survives the obvious null — I checked

Criterion 6 is the only number without a null, and it is the one the PR leads with. Since
§1 shows the family count is mechanically *k*/2, the natural worry is that coarsening
inflates ARI on its own. It does not. **[verified]**

**Pure-coarsening null**, structureless 15-D data, two independent random partitions into
72 clusters, each coarsened by Ward on its own centroids at p50:

```
cluster ARI  0.00005   ->   family ARI  -0.00003
```

Coarsening contributes nothing. ARI's chance correction handles it.

**Positive control**, simulating the batch-minting pathology directly — 20 true
populations, each run splitting every population into four arbitrary pieces with different
splits per run, 80 clusters per run:

```
cluster ARI  0.2394   ->   family ARI  0.5874   (family vs ground truth: 0.7569)
```

The mechanism reproduces in simulation and recovers the true populations. The branch's
0.0279 → 0.5190 is a real result and the interpretation in the PR is sound.

**Still worth adding the null to the code**, for the same reason `narrow_frac` got one:
the headline metric should not be the only one without a control, and someone will ask.
Ten lines — permute the cluster→family assignment preserving family sizes, recompute ARI
across seeds — and it belongs next to `narrow_frac_null` in `quality()` or beside
`stability()`.

---

## 4. The rail now contradicts the acceptance document — fix before merge

`audit()` sets `share-high` / `share-low` from `share_global` only **[repo,
`diagnostics.py:139`]**. On `sband_short` the rail therefore renders:

```
x03_channel_offset   share 24.3% / knn 7.4%   ... [share-high]
```

A user reads that as "x03 is over-weighted". `acceptance-2026-09.md` finding 1 says that
reading "does not survive" the k-NN measurement. The rail caption explains why `knn` is
unthresholded but does not warn that the flag itself may now be the misleading number.

This is the one thing I would fix before merge, and it is a caption, not a refactor:

> Flags use the **global** share. Where `knn` differs sharply from it, prefer `knn` —
> HDBSCAN responds to local density. Measured: `x03` 24.3% global against 7.4% local.

Longer term the primary should be `share_knn` with global demoted to a secondary column,
which is the conclusion your own finding 1 reaches. Holding the threshold until it is
calibrated is right (question 4 — agreed, and the discipline is the right one); shipping a
flag that the same repository documents as misleading is a different matter.

---

## 5. Question 2 — you are over-reading finding 4, and the reason is sharper than "uncontrolled"

Dropping `x03` and `f07` halves `leaf`'s narrow share (6.820% → 2.969%), and the PR reads
this as evidence against the deferred contribution-equalising scaling. Two problems beyond
the confound you already name.

**Dropping is not down-weighting.** Setting a weight to zero is the endpoint of the
intervention, not a scaled version of it. Equalisation would move `x03` from 24.3% to
6.7%, a 3.6× reduction. Your own ARI-versus-cut curve is non-monotone, so this system
demonstrably does not respond monotonically to its knobs, and reading the endpoint as the
direction of the interior is not safe.

**More decisively: finding 1 already tells you `x03` is fine, so this experiment removed a
well-behaved column.** `x03`'s k-NN share is 7.4% against an equal share of 6.7%. It is
*already* contributing about what it should, locally. Contribution-equalising scaling
targeting the local share would barely touch it. So finding 4 is not evidence about
equalisation — it is a second demonstration of finding 1, which is that `x03` was never
the problem.

**The column finding 1 actually indicts is `f09_temporal_skew`**, at 15.5% local against
6.7% equal, and benign on every global statistic. If you want a fast read on whether
equalisation would help or hurt, that is the column to down-weight, not `x03`.

**What this means for my §7 hedge:** withdraw it as stated. I predicted families would be
grouped substantially by channel offset and a clipped correlation coefficient, and finding
1 shows the first half of that was based on the wrong statistic. Your measurement beat my
inference — which is the second time this cycle, and the reason for the tag discipline.

The general caution survives for a different reason: the first family taxonomy is
provisional because the *cut rule* is a granularity dial (§1), not because the metric
over-weights `x03`. The `matching.py` caveat should be rewritten accordingly — it
currently names `x03` at 24.3% and `f07` at 13.1% as the concern, and finding 1 supersedes
half of that.

**Net effect on the deferred scaling work:** weaker case than either review document
assumed, agreed — but for finding 1's reason, not finding 4's. And the target list has
changed: `f09` in, `x03` out.

---

## 6. Two performance items, both `all`-scale only

**6.1 `O(k·n)` grouping will not scale to `all` + `leaf`.** The mask-per-cluster pattern
appears in `matching.centroids`, `metrics._narrow` (called 7× per `quality()` — two
thresholds plus five permutations), `metrics._enrichment`, and the `spans` comprehension.
Timed at 1,281,878 rows × 15 columns: **[verified]**

| k | mask-per-cluster | sort + `reduceat` |
|---:|---:|---:|
| 1,491 (`eom` on `all`) | ~5.9 s | 1.2 s |
| 20,000 | ~80 s | 1.4 s |

At the ~80,000 clusters `leaf` would produce on `all`, `centroids()` alone is several
minutes and `quality()` is worse because of the 7× multiplier. One `argsort` plus
`np.add.reduceat` / `np.minimum.reduceat` / `np.maximum.reduceat` fixes all four call
sites and is flat in *k*. Not urgent — the Bench samples 35k — but `bluse-cluster --match`
on `all_features.parquet` under `leaf` is exactly the run someone will start before lunch.

**6.2 `/stability` clusters twice per seed.** `run_fn` and `run_fam` each call `cluster()`
for the same seed, and `cluster()` is not memoised **[repo, `app.py:596–613`]**. That is
2*N* clustering runs where *N* would do, on the route the docstring correctly identifies
as the slowest thing in the tool. Compute `labels` once per seed, keep them, and derive
both statistics — `stability()` takes a `run_fn`, so either pass a closure over a
per-seed cache or add a variant taking a list of label vectors.

---

## 7. Smaller points

**`test_default_pct_rule_merges_monotonically` cannot fail.** Given §1, family count is
*k*(1−*p*/100), so monotonicity in `pct` holds by construction on any input including pure
noise. It is not wrong, but it does not test the rule — the informative version is whether
the rule recovers the *planted* family count on the fixture, which is what the gap-rule
test does and why that one is the better test.

**`_enrichment` mixes two denominators.** The hypergeometric test runs on
`(labels == c) & known` while `sizes` records the full `(labels == c).sum()`. Defensible —
you want the fraction of *clustered hits*, and the docstring says so — but a cluster that
is 90% unlabelled will contribute its whole size on the strength of a test over a tenth of
it. Worth one line in the docstring, or restrict `sizes` to `known` and rename.

**`quality()`'s `narrow_frac_at` is keyed by float.** `at[t]` with `t` a float key works,
but it makes the JSON in `<tag>_metrics.json` awkward and comparisons brittle. String keys
(`"0.1"`, `"1.0"`) would survive a round-trip.

**The `eom`/`leaf` epoch profiles are a good result and are undersold.** `leaf` giving
12.9 / 7.3 / 6.5 / 5.7 / 5.1 / 4.9 / 4.2 / 3.8 against GLOBULAR's 47.6 then a flat 22–30
is the closest this pipeline has come to the paper's regime, and it is buried under the
reproducibility discussion. If the p98 run in §2 goes well, that plus the epoch profile is
the argument for `leaf` as default.

---

## 8. Answers to the four questions, condensed

**1. Is `eom` the right default?** Hold it, but not on the stated grounds. Something does
favour it (enrichment 10×), and the family-ARI half of the case is confounded by
granularity (§2). Do the p98 run first; if `leaf` at 36 families matches `eom`'s ARI while
keeping a higher narrow share, it should become the default. Until then "the comparison is
not decisive and `eom` produced every committed result" is the honest framing.

**2. Does finding 4 contradict the reviewer's §7?** It contradicts my §7, which I withdraw
— but it is not evidence about equalisation (§5). Finding 1 already established that `x03`
is well-behaved locally; dropping it tested the wrong column. `f09_temporal_skew` is the
target the measurement actually implies. The provisional-taxonomy caveat should be kept
and re-grounded in §1.

**3. Is p50 tuned on one file?** No, and it could not be — the rule is scale-invariant, so
the 5.26× lattice spread cannot reach it. But it is also structure-invariant (§1), which is
the concern you should have instead. Re-document, and consider exposing family count
directly.

**4. Is leaving `share_knn` unthresholded right?** Yes, and the discipline is right. But
the interim state is contradictory (§4): the rail flags on the statistic your own
acceptance document says is the misleading one. One caption line closes it now; when the
thresholds land, `share_knn` should be primary.

---

## 9. Suggested order

**Before merge**

1. §4 rail caption. One sentence.
2. §1 docstring correction in `derive_cut_pct` and `match` — the cut is a granularity
   dial, stated plainly.
3. §5 rewrite of the `matching.py` provisional-taxonomy caveat to cite §1 rather than
   `x03`'s global share.

**Before the numbers are presented anywhere**

4. §2's p98 run for `leaf`, and the `eom`-vs-`leaf` table restated at matched family
   count.
5. §3's coarsening null alongside the family-ARI result.

**Follow-ups**

6. §6.1 sort+`reduceat` in the four grouping sites, before anyone runs `--match` on `all`
   under `leaf`.
7. §6.2 halve `/stability`'s cost.
8. §1 option 2 — the gap rule restricted below the 90th percentile — as one experiment,
   since a structure-sensitive cut would be better than a granularity dial if one exists.
9. §5's `f09_temporal_skew` down-weighting experiment, which is the real precursor to the
   deferred scaling spec.
