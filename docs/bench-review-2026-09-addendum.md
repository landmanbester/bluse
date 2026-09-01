# Addendum to the Cluster Bench review (2026-09)

**Status:** reply to `bench-review-2026-09-response.md`. Second and final pass from the
external reviewer. Supersedes the original review wherever the two conflict.

**Tags** as before: **[repo]** read from committed code/CSVs, **[measured]** computed by
whoever is speaking, **[analytic]** closed-form, **[hypothesis]** untested with a stated
test. Numbers attributed to the response are theirs, measured on the real feature matrices;
treat them as authoritative over anything in the original review.

The response is a better document than the review it answers. Six of its rebuttals stand,
one of my own headline experiments was the wrong instrument, and its promotion of
seed-stability to P0 is correct. What follows is: what I withdraw, three places its own
measurements support a stronger conclusion than it drew, two places the reasoning has a
gap, and the resulting amendments to its plan.

---

## 1. Withdrawn

**R-3 — Finding 2 is wrong.** The `f06 = (f04² + 1) / f05` identity is exact on raw values
and does not survive `normalise()`, which applies `unit` to f04, `log-unit` to f05 and
`none` to f06. Distance is taken after those transforms, so the algebra never reaches the
metric space. I asserted a ~1.5× directional weighting from an identity without checking
that it propagated. The VIF table is the correct diagnostic and f04 (53.1), f05 (48.5) and
f11 (31.0) are the correct targets. §2 of the review should be rewritten around those
three, and the P1-5 acceptance test changed to f04/f05 as the response says.

One thing to carry forward anyway: the raw identity should be documented in `features.py`
next to `f06_bimodality`. It does not bite today because of the transform mismatch, which
means it is latent — a future change to the `f06` or `f05` transform reactivates it
silently.

**R-4 — D-4 overclaimed.** A median and an IQR from n=35,000 are precise estimates; saying
"a 2% draw" implied a sampling problem that does not exist, and 14 of 15 columns agreeing
to 1.1% settles it. The claim is weaker still than the response allows: `normalise()`
already fits its transforms globally at extraction time **[repo]**, so GLOBULAR's
global-scaling-before-batching requirement (§3.1) is satisfied upstream, and the Bench's
robust/quantile refit is only the second stage. Fitting it on the full column set is
correct and cheap; it is not a reproducibility fix. Their framing — real, mis-diagnosed,
oversold — is accurate.

**R-6(a), R-6(b) — two factual corrections accepted.** 26.6% of rows reaching the
clusterer, not 33.5% of the file; and the severity is file-dependent (`f02_abs_drift_n`
IQR 1.239 on `all` against 5.954 on `sband_short`). §1.1 of the review presents the
`sband_short` figure as universal and must be corrected in place.

**My §1.3 step 3 was the wrong instrument, and R-1 is right that it cannot resolve
anything.** ARI compares partition *identity*. My hypothesis was about density
*structure*. Comparing labellings across runs on different row sets, under a labelling the
response then demonstrates is a batch artefact, was a category error. The right statistics
were my own steps 1 and 2 — largest-cluster fraction, cluster count, noise fraction,
epoch-1 removal — with seed replicates supplying the error band. Delete step 3 from §1.3
and replace it with "report each statistic as mean and range over N seeds".

**Their promotion of P1-1 to P0-1 is correct** and I withdraw my ordering. See §7 below
for one refinement to how that metric is computed.

---

## 2. Where the response's own numbers support a stronger conclusion

### 2.1 `x03_channel_offset` may explain the 852 MHz cluster spans

R-5(a)'s distance-share table is the most important measurement in the response, and it
has a consequence the response does not draw.

`x03_channel_offset` is the position of a hit *within its coarse channel*, 0–1 **[repo]**.
It is therefore approximately independent of absolute frequency by construction. A cluster
whose membership is driven largely by x03 will contain hits from coarse channels spread
across the whole band — which is precisely the signature in the committed summaries: 14
clusters on `sband_short` each spanning ~852 MHz, and on `all` a 2,156 MHz span on the
largest clusters **[repo]**.

**[hypothesis]** The one-blob-per-batch clusters are substantially *channel-offset*
clusters, not drift-rate clusters. If so, the primary diagnosis in the review is aimed at
the wrong feature.

**Test, cheap:** re-run the harness with x03 off, at fixed seed and otherwise default
settings, and report median cluster `freq_span_mhz`, largest-cluster fraction, and
epoch-1 removal, each over N seeds. If median span collapses toward the coarse-channel
width, x03 is the driver.

This test also happens to be the discriminating version of R-1's claim — see §3.2.

### 2.2 The ±5 clip is probably manufacturing a tie, in x03 specifically

I raised the clip for `f02` in conversation and correctly dropped it: the slab lands at
−0.884, nowhere near ±5. But x03 is the opposite case and I did not check it.

x03 is bounded in [0,1] and has the **smallest raw IQR in the matrix, 0.036** **[repo]**.
`(X - med) / 0.036` sends anything more than **0.18** in native units away from the median
past ±5 **[analytic]**, where `np.clip` deposits it on an exact value. On a variable whose
full range is 1.0, that is a narrow window, and x03 already carries a 26.6% tie and only
83 distinct values by the response's own count.

**[hypothesis]** A large fraction of x03 sits at exactly +5 or exactly −5 after robust
scaling. That is a manufactured coordinate tie of exactly the kind GLOBULAR removed
268,084 duplicate hits to avoid (§3 of the technical reference), created by our own
scaling rather than present in the data.

**Test:** add `fraction at exactly ±5` per column to the P0-2 diagnostics. One line. It
may be the largest tie in the matrix, and unlike every other tie discussed so far it is
entirely our own doing.

### 2.3 Robust scaling fails in both directions — the unifying statement

The review argued that the IQR is not robust to a 33% tie and therefore *under*-corrects
`f02`. R-5(a) shows x03 *over*-weighted by 3.6×. These are the same defect with opposite
signs:

| | tie position | effect on IQR | effect on weight |
|---|---|---|---|
| `f02_abs_drift` | at the extreme (−5.199) | inflated to 5.954 | under-weighted, 1.7% share |
| `x03_channel_offset` | near the centre | deflated to 0.036 | over-weighted, 24.3% share |

The IQR is unrepresentative of the distribution in both cases, and dividing by it
propagates the misrepresentation into the metric. This is a much better argument for the
response's ordering — general fix before instance fix — than the one I gave, and it is how
the scaling work should be motivated in the spec. It also means neither `robust` nor
`quantile` is the right default long-term: a contribution-equalising mode is, and it should
target the distance share directly rather than a spread proxy.

---

## 3. Two gaps in the response's reasoning

### 3.1 The "20× more reproducible" figure needs decomposing before it is quoted

C-3 reports seed-only pairwise ARI of 0.024 for `eom` against 0.483 for `leaf`, and R-1
makes that the strongest case for `leaf`. The comparison is not measuring the same quantity
in the two arms.

`eom` clusters 99.9% of points; `leaf` clusters 50.5%. `sklearn.metrics.adjusted_rand_score`
treats −1 as an ordinary label, so under `leaf` roughly a quarter of all pairs are
within-noise pairs and count as agreement whenever both runs call both points noise. ARI
does correct for chance given the marginals, so this is not pure artefact — but a single
scalar over a partition where half the mass carries one label conflates two different
claims: *agreement about which points are unclusterable*, and *agreement about cluster
membership*.

**Report both:**

- ARI restricted to the points clustered (label ≥ 0) in **both** runs.
- Agreement rate on the noise indicator, i.e. ARI or simple accuracy on the binary
  `label >= 0` vector.

`leaf` may well still win decisively on the restricted statistic. The point is that the
number will then mean something, and P0-1's acceptance criterion ("reproduces ARI ≈0.02 for
`eom` and ≈0.48 for `leaf`") will be checking a well-defined quantity rather than freezing
in a composite.

### 3.2 R-1's strong claim rests on the weakest available test

R-1 concludes: "Batch membership, not feature geometry, is what determines which cluster a
hit lands in", from ARI 0.892 when `f02` is dropped at fixed seed. It then uses that to
promote cross-batch matching from P2 to P1 as a *correctness* issue.

But R-5(a) measures `f02` at **1.7% of the squared-distance share — the least-weighted of
fifteen columns**. Removing the least influential feature and observing little change is
equally consistent with two readings:

- geometry does not matter (their conclusion), or
- `f02` does not matter (which R-5(a) independently establishes).

The experiment cannot separate them. The discriminating test is dropping **x03 at 24.3%**,
and it is the same run as §2.1's test:

- If ARI stays ≈0.89 with the most influential column removed, geometry genuinely does not
  drive membership, the claim is established, and the P2-1 → P1 promotion is justified on
  correctness grounds.
- If ARI collapses, geometry matters a great deal, and cross-batch matching should be
  promoted on the *original* grounds — that it is the repo's own top-graded gap — rather
  than on this argument.

Either way the promotion probably survives. But it should survive on a test that could
have failed.

---

## 4. Two refinements to the P0-2 diagnostics

**Compute the distance share on k-NN pairs, not random pairs.** HDBSCAN responds to core
distances and mutual reachability, both local. Global mean pairwise squared distance is a
proxy for a local quantity, and the approximation is worst exactly where ties are: a
tied column contributes **zero** to every tie–tie pair, and tie–tie pairs are
disproportionately likely to *be* mutual near neighbours, because they already agree in
that coordinate.

So `f02`'s local contribution is plausibly *below* even its 1.7% global share, while the
slab's effect on local density is correspondingly stronger than the random-pair table
suggests. Same computation, restricted to each point's k nearest neighbours at the run's
`min_samples`. Report both columns — global and local — because the gap between them is
itself the tie diagnostic.

**Add a nonlinear dependence measure to the redundancy panel (P1-5).** VIF and linear R²
are the right notions for a Euclidean metric, so R-3's ranking stands for its stated
purpose and f04/f05 is the right acceptance test. But f04/f05/f06 is precisely a case where
a *deterministic* relationship registers as R² = 0.710 because it passes through a ratio
and a log. A panel built on VIF alone will systematically miss that class of dependency,
and this codebase has at least one instance of it. Add one of: mutual information, or the
out-of-fold R² of a small gradient-boosted regressor per column against the rest. Keep
f04/f05 as the linear acceptance test and add f06 as the *nonlinear* one — which turns my
wrong claim into a useful test case rather than discarding it.

---

## 5. The `f02` fix changes shape, not just its position in the queue

R-6's ordinal finding is the best thing in the response and it invalidates **both** of my
proposed fixes, not merely their ordering.

If `|driftRate|` takes 42 distinct values on a constant 0.010711 Hz/s grid — the seticore
Taylor-tree drift step — then `f02_abs_drift` is a 42-level ordinal on a uniform lattice.
The quantile-normal transform is then mis-specified for a reason that has nothing to do
with the zero bin: **a rank transform re-spaces the levels by population**, so adjacent
drift steps end up far apart where hits are dense and close together where hits are sparse.
For a physically meaningful, uniformly quantised quantity, that is a distortion, not a
correction. My fix 1 removes the zero tie and leaves a 41-level ordinal still being
rank-transformed as though it were a continuum — which is R-6's point, and it applies to
the fix as much as to the status quo.

**Revised fix:** an `is_zero_drift` indicator plus the non-zero drift kept on its **native
linear grid**. `abs(driftSteps)` is already that integer ordinal **[repo]** and would
reproduce the 42 levels exactly, so the natural implementation is to use it directly rather
than re-deriving the lattice from `driftRate`. Confirm that `abs(driftSteps)` and the
measured 0.010711 Hz/s spacing agree before building on it. GLOBULAR flagged the drift
discretisation gaps as unmitigated future work (§11 of the technical reference), so this is
territory the paper left open rather than a departure from it.

**On R-5(b), agreed with one addition.** The boolean split is a weighting decision
presented as a cleanup, and it should follow the scaling work. But it has a property no
other column in the matrix has: its contribution is analytic. Mean pairwise squared
distance for a two-valued column is $2p(1-p)$, which at $p = 0.266$ is **0.390**, against
$1/6 = 0.167$ for a uniform column — so it lands at **2.34×** the uniform baseline
**[analytic]**, and can be scaled to any target share exactly, with no estimation. Far from
being the hardest weight to justify, it is the only one that can be set by calculation
rather than by measurement.

**Consequence for the diagnostics:** the tie statistics need a declared column type. A
boolean has `max_tie_fraction ≥ 0.5` by definition, so the review's proposed `> 0.1` flag
and D-5's proposed `> 0.5` assertion both misfire on it by construction. Add a `kind` field
(`continuous` / `ordinal` / `boolean` / `flag`) to the feature registry, and apply the tie
thresholds only to `continuous` and `ordinal`. For `ordinal`, the informative statistic is
levels-per-decade of range, not tie fraction.

---

## 6. The objective function

**R-2 is accepted.** AMI against `weak_label` is not a tuning signal at 31:1, my acceptance
criterion was too weak to have caught that, and minority-class enrichment against a
hypergeometric null is the better construction. Demote as proposed. Two additions.

**AMI could not have arbitrated `eom` vs `leaf` even in principle.** 79 clusters at 99.9%
clustered and 2,127 at 50.5% are not comparable partitions — the same objection as §3.1.
"Tuning on AMI says `eom` beats `leaf`" is therefore not a finding about `eom`; it is a
second symptom of comparing two configurations on a statistic that is not measuring the
same thing in each. The conclusion (demote AMI) is right; the argument for it is stronger
if framed that way, because it does not require AMI to be *wrong*, only inapplicable.

**There is a label-free objective already available with real dynamic range, and it encodes
the actual deliverable.** The stated Track B deliverable is an RFI taxonomy **[repo]**, and
a taxonomy is good when its classes are physically coherent: narrow in frequency, recurring
across observations, consistent in drift. `freq_span_mhz` and `n_obs` are already in the
summary.

Proposed headline metric: **fraction of clustered hits sitting in clusters that span
< 1 MHz.** Baseline from the committed runs **[repo]**:

| run | clusters | narrow clusters | hits in narrow clusters |
|---|---:|---:|---:|
| `sband_short`, eom | 72 | 25 | 271 of 34,916 = **0.78%** |
| `all`, eom | 1,491 | 107 | 1,240 of 1,281,791 = **0.097%** |

*(Correction to something I said in conversation: I quoted ~1.8% for `sband_short`, having
conflated the 629 hits in clusters of n<50 with the hits in narrow clusters. The correct
figure is 0.78%. The direction of the argument is unaffected and slightly strengthened.)*

Under `leaf` this should rise by orders of magnitude, and it discriminates precisely where
AMI does not: it needs no labels, has a dynamic range of 0.1% to plausibly tens of percent,
and rewards the thing the workshop is actually for. Ship it alongside enrichment at P1, and
make it the metric the scaling work in their P2 item 11 is evaluated against.

**Promote the synthetic-injection recovery rate to a scheduled item.** It is the only true
objective function available — GLOBULAR's own (§3.1, graded "High and cheap" in §12.5) —
and it is currently unscheduled in both documents. A drifting narrowband Gaussian line plus
noise, matched to the stamp geometry, is on the order of 30 lines and unblocks both the
recovery metric and GLOBULAR's cluster-seeding trick.

---

## 7. `leaf`: control now, default gated on matching

`leaf` gives 2,127 clusters at 50.5% clustered. GLOBULAR reached ~59 families at 93.1%
reduction. Those 2,127 are not a result — they are the **input** to matching, and
GLOBULAR's route to families was exactly this: many microclusters from a low `n_pts`, then
merge (ε_m within batch, centroid matching across batches; §8 tuning step 3 and §3.2).

So: ship `leaf` as the **control** at P0, as the response has it. But do not make it the
**default** before cross-batch matching exists — 2,127 unmatched discovery-order ids is a
less readable cluster table than 79, and the repo's own guidance is already that the table
should be read as "these hits grouped with something" rather than as a taxonomy **[repo]**.
State the gate explicitly in the spec: **the default flips to `leaf` when P1-6
(cross-batch matching) lands, not before.** Otherwise the first consequence of the best
recommendation in the review is a worse-looking tool, which is how good changes get
reverted.

---

## 8. Amendments to the revised plan

The response's re-prioritisation is accepted as the baseline. Deltas only:

| item | amendment |
|---|---|
| **P0-1** seed-stability | Report **two** ARI statistics: restricted to jointly-clustered points, and the noise-indicator agreement. Acceptance criterion checks the restricted one. §3.1 |
| **P0-2** per-feature diagnostics | Add: distance share on **k-NN pairs** at the run's `min_samples`, alongside the global share (§4); **fraction at exactly ±5** per column (§2.2); a `kind` field per feature so tie thresholds apply only to continuous/ordinal columns (§5) |
| **P0-3** `leaf` control | Add an explicit gate: default stays `eom` until cross-batch matching lands. §7 |
| **new P0-5** the x03 test | Re-run with `x03_channel_offset` off, fixed seed, N replicates. Report median cluster `freq_span_mhz`, largest-cluster fraction, epoch-1 removal, and ARI vs baseline. Settles §2.1 *and* supplies the discriminating version of R-1's claim (§3.2). Two runs of an existing harness |
| **P1-5** redundancy panel | Add a nonlinear dependence measure. Linear acceptance test f04/f05; nonlinear acceptance test f06. §4 |
| **P1** objectives | Add **fraction of clustered hits in clusters spanning < 1 MHz** as the label-free headline metric, baselined at 0.78% / 0.097% (§6). Keep AMI + enrichment, captioned as weak proxies |
| **new P1** injections | Schedule the synthetic stamp generator and injected-signal recovery rate. §6 |
| **P2 item 11** scaling | Motivate as *contribution equalisation in both directions* (§2.3), and evaluate against the narrow-cluster metric and P0-1 stability, not by eye. The `f02` decision follows it, per R-5(b) |
| **P2 item 11a** the `f02` fix | Revised: `is_zero_drift` indicator **plus non-zero drift on its native linear grid** (probably `abs(driftSteps)`), not a rank transform of the remainder. §5 |
| **D-5** tests | The `max_tie_fraction > 0.5` assertion needs the `kind` field or it misfires on any boolean feature. §5 |

Not contested: everything else in the response's plan, including the D-4 disposition, the
`f06` demotion, and the out-of-scope list.

---

## 9. Edits owed to the original review document

These are corrections to `bench-review-2026-09.md` itself, so the two documents do not
disagree in the repo:

1. **§1.1** — 26.6% at the clusterer, not 33.5%; and add that severity is file-dependent
   (IQR 1.239 on `all`). R-6.
2. **§1.1** — add the ordinal finding: 42 levels on a 0.010711 Hz/s lattice. This is the
   deeper defect and the tie is a symptom of it. R-6.
3. **§1.3** — delete step 3 (pairwise ARI across the three configurations). Replace with
   "report steps 1–2's statistics as mean and range over N seeds".
4. **§1.4** — replace fixes 1 and 2 per §5 above; promote fix 3 to first and restate it as
   contribution equalisation per §2.3.
5. **§2** — rewrite around f04/f05/f11 per R-3. Keep the raw `f06` identity as a documented
   latent hazard and as the nonlinear acceptance test, not as the exemplar.
6. **§3.1** — upgrade the epoch-1 collapse from `[hypothesis]` to `[measured]`, citing
   C-2's table: 87.9% removed in epoch 1, epochs 4–8 dead.
7. **§3.3** — add the `leaf` default gate. §7.
8. **§4** — rescope per R-2; add the 31:1 imbalance, the narrow-cluster metric, and the
   injection item.
9. **§6, D-4** — restate: real, but the cause is 35k vs 1.28M row counts and seed
   instability, not the scaler fit. R-4.
10. **§0** — add a line noting that the response document supersedes this one wherever they
    conflict, and that `[measured]` figures in the original were simulation-calibrated and
    have since been confirmed or corrected on real data.

---

## 10. Revised top three

Their P0 ordering, with the amendments above:

1. **Seed-stability with the noise decomposition** (P0-1 + §3.1). Nothing else is
   interpretable, and the composite ARI needs splitting before it becomes an acceptance
   criterion.
2. **Extended per-feature diagnostics** (P0-2 + §4, §2.2, §5). This subsumes my zero-drift
   experiment as one instance of a general audit, which is R-5(a)'s point and it is right.
3. **The x03 test** (new P0-5). Two harness runs. If it lands it relocates the primary
   diagnosis from the drift feature to the channel-offset feature, promotes the scaling work
   to P0, and simultaneously supplies the test R-1's central claim is currently missing.

The zero-drift work, which was my number one, drops out of the top three entirely. R-5(a)
is the reason: it is a 1.7% column, and I was arguing about the least influential feature
in the matrix.
