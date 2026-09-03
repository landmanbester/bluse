# The arc — three days

**31 August to 3 September 2026. 2,014,055 narrowband hits from seven MeerKAT
files.**

The shape of it, before the detail: **two days making the measurements
trustworthy, one evening getting a result, and one day discovering how much of
that result was wrong.** That ratio is the story.

---

## Day 0 — Sunday 31 August: a pipeline by evening

The starting position: 21 GB of HDF5, ~2M hits, each one "something at
874.9955 MHz drifting at −0.11 Hz/s in beam 44". Almost all of it is
interference. The question is which of it isn't.

**14:48 — first commit.** Track A, the classical chain from Tremblay et al.
2026: known-RFI frequency masks, drop zero-drift, multi-beam coincidence, an
SNR window, cross-epoch persistence. Applied in that order, with a cut-flow
recording what each step removed.

The first number that mattered arrived immediately, and it was humbling: **the
known-RFI frequency mask alone removes 91.3% of `lband_short`.** Everything
clever we might build afterwards would be operating on the 8.7% a lookup table
could not explain.

**16:04 — Track B.** A feature registry, then GLOBULAR's thirteen hand-crafted
features (Jacobson-Bell et al. 2025) plus three BLUSE-specific extras, computed
over the stamp cubes. HDBSCAN on top, in batches of 3,000 over 8 epochs,
following the paper.

**21:45 — Cluster Bench.** An interactive explorer: change a hyperparameter in
a browser, re-cluster, see the scatter move.

A pipeline, end to end, in a day. It looked like the hard part was over.

---

## Day 1 — Monday 1 September: the instruments were broken

The Bench was the first thing that could be *played with*, and playing with it
immediately produced a bad smell: **nothing mattered except
`min_cluster_size`.** Every other knob — `min_samples`, epochs, batch size,
epsilon — moved the picture barely or not at all.

The comfortable reading is "the method is robust." The correct reading was that
we were not measuring what we thought.

**Three defects, all found in one sitting** (`358a282`, "Fix three defects that
made every knob but `min_cluster_size` inert"):

1. **Cluster ids collided across batches.** HDBSCAN mints local ids `0..k−1` in
   *every* batch. The code offset by epoch only, so batch 0's cluster 3 and
   batch 7's cluster 3 became the same cluster. On `sband_short`, **731 real
   groups were being reported as 205.** The "cluster count" was a maximum over
   batches rather than a total — which is precisely why only `min_cluster_size`
   appeared to do anything.
2. **`min_samples` defaulted to 2**, which in sklearn is the same call as 1: no
   core-distance smoothing, i.e. pure single linkage. The runs were *bistable*
   — ten identical 3,000-point draws returned either k=2 holding 99.7% of
   points, or ~200 microclusters with 40% noise. **A 112× swing on the shuffle
   alone.**
3. **`cluster_selection_epsilon` was inert, then fatal.** sklearn tests epsilon
   against `1/d`, the *reciprocal* of a leaf's split distance. Our splits are
   ≳1.7, so below ≈0.55 every value is bit-identical to zero and above it the
   code raises `TypeError`. The no-op region and the crash region tile the
   domain. There is no working value.

Only after that did the day's real work start: three new modules
(`diagnostics`, `metrics`, `matching`), cross-batch cluster matching by Ward
linkage, and **the repository's first test suite**.

**21:22 — review.** A human review, then Copilot. Copilot found what the human
review had missed, and the commit message is the honest one:

> `fix: address the Copilot review, including a fix I had claimed but not made`

A defect had been reported fixed in the PR and **had never been applied**. The
code computed full-population scaler statistics into a field that nothing ever
read. Separately, a memory claim taken from a review without checking turned
out to be wrong by three orders of magnitude at real scale — Ward is O(k²) in
memory, not O(k), which at ~78,000 centroids means 24.3 GB.

**21:42 — PR #1 merged.** By 22:17 the first experiment had returned a verdict,
and the verdict was *no*.

---

## Day 2 — Tuesday 2 September: the day of rejections

Four improvements were queued. Three were rejected by measuring them.

**P0-1, the restricted gap rule** (`84e42ff`, 22:17 the previous night). The
proposal was a smarter rule for where to cut the Ward tree into families. The
measurement reframed the question entirely: **every horizontal cut of a fixed
Ward tree is uniquely determined by the family count it leaves.** 1,400
thresholds tested across real trees and synthetics, zero exceptions. A cut rule
never selects a better partition — only a point on a fixed nested chain. What
shipped instead was `n_families=`, an interface the docstrings had been telling
readers to prefer since before it existed.

**P1-4, the `f02` rework.** `f02_abs_drift` is a 42-level ordinal whose 26.6%
zero-drift tie the quantile transform throws to a bound, leaving it the
least-weighted column in the matrix. That reads like a bug. The planned repair —
native linear grid plus a zero-drift indicator — is **worse under `eom` at every
family count**. Kept the transform, and pinned it with a test whose docstring
says: *"This test exists so the change cannot be made silently a second time."*

**P0-3, thresholding the distance share.** Before the threshold could be set,
the estimator feeding it had to be fixed: `_shares_knn` built its neighbour
index on its own subsample, drifting the statistic toward the global share by
**up to 2.18 points — 13× the seed noise.** Every live figure had to be
re-measured.

**Mid-morning: the data changed underneath us.** The BLUSE team re-delivered two
repaired files. Usable rows went from 1,605,678 to **2,014,055 (+25.4%)**. Every
conclusion had to be re-checked against them. None moved — but the retired
interim file turned out to have been a *biased* subset of its band (26.75%
zero-drift against the full file's 46.60%), so per-band results computed on it
were wrong in a way nobody would have noticed.

**P1-5, contribution-equalising scaling.** Specced, planned, externally
reviewed, built across six tasks. The iterative k-NN strategy scored 0.83
against the closed form's 0.51 and was **rejected anyway**: its target is
mathematically unattainable for a column with a large tie fraction, it collapses
the metric from 15 effective dimensions to 2.22, and a three-line hand-built
weighting beats it. What shipped is the closed form — which turns out to be
plain winsorised standardisation, and **does not meet its own acceptance
criterion.** Why it cannot is still an open question (`P0-4`).

Then, at 19:04, Copilot again:

> `Fix: the CLI accepted robust-equalised and clustered the raw matrix`

The new mode had been wired into one code path and missed by the other. The CLI
accepted the flag, reported the weights in its metrics file, and clustered the
**unweighted** matrix. Verified elementwise identical to `scaling="none"`. My
own end-to-end check had compared against the wrong baseline.

By evening: a named RFI taxonomy, an undocumented 874.9955 MHz emitter, and
`NOMENCLATURE.md`, written because the word "family" appeared a dozen times
across three documents and was defined in none of them.

---

## Day 2, the last night — Track E

**19:47 to 23:32.** Track E had been in the backlog since day 0, described in
the original brainstorm as *"the cheapest genuinely novel contribution"* and
listed under P3, "still queued", for three days.

The idea in one sentence: **the multi-beam spatial filter is the survey's
strongest discriminant and it is blind exactly where it matters** — it needs
many beams' worth of evidence, so it cannot judge a hit seen in one beam, and a
technosignature is a one-beam hit. Take the filter's own verdicts as free
labels; learn them from the stamp pixels; get the same judgement where the
filter has nothing to say.

It worked on the first try, which was itself suspicious. Stamp morphology
predicts the spatial filter's verdict at **0.9899 ROC-AUC**. So most of the
evening went into trying to break it:

- drop every zero-drift row (the 29.6% where one feature is NaN and 99.6% are
  RFI) → **0.9927**, it goes *up*
- measure within each SNR decile, so brightness cannot be the label → **0.8934**
- collapse the up-to-64 near-duplicate rows per emitter → **0.9835** on
  singletons alone
- hold out an entire band → **0.90–0.99**

And the one that convinced us. The model trains only on ≤2 beams and ≥32 beams.
Scored on the untrained 3–31 range — 422,480 hits in five bins it never saw —
it reproduces beam multiplicity **monotonically**: 0.18, 0.27, 0.44, 0.71, 0.90.
A fitted decision boundary has no reason to interpolate.

The night still produced six more wrong turns, three of them in the last two
hours. They are in [`02-wrong-turns.md`](02-wrong-turns.md), and the last two
are the ones worth reading.

---

## Day 3 — Wednesday 3 September: the day the result was audited

Track E's own write-up had named the missing piece: *"every number measures
agreement with the spatial filter, which is a good instrument and not truth."*
So the last day went to **synthetic injections** — putting signals we built
ourselves into real stamps, which is the only ground truth this archive can
supply.

Building the harness turned up something about the pipeline nobody had noticed.
Chasing a 0.070 measurement floor: re-extraction was exact, so the floor had to
be normalisation; interpolating on the stored curve cut it to 0.127, still
wrong; the residual was one raw value mapping to two normalised values, which
should be impossible. **It is possible because `normalise()` is called per
file** — `bluse-features` runs it inside `extract()` and `summarise()`
concatenates without re-normalising, so every `_n` column is a rank within its
own file rather than within the survey. Per-file interpolation is exact.

Then the first result, and it looked decisive: *a cut at the shipped threshold
keeps at most 43% of injected real signals.* Written up, pushed, PR opened.

**Then the review, and both headline numbers fell.**

Copilot found the SNR formula was not the matched filter — a unit-weight
numerator with a matched-filter denominator, delivering 14.19 when asked for 20.
The test that should have caught it asserted the algebraic inverse of the
function under test, so it could not fail.

The human reviewer found the substrate set was **42% confirmed multi-beam
RFI** — `select_substrates` filtered on SNR alone, and low SNR does not imply
few beams. The 43% was a composition effect. Stratified, on the population the
score is actually deployed on, retention is 0.72.

And the claim listed under *safe as stated* in my own summary — "pruning is
unaffected" — turned out never to have been measured. Measured: at catalogue
SNR ≈51 with no drift, **91% of injected real signals are pushed above the
pruning threshold.** A bright non-drifting sky signal would be actively
discarded.

Re-running it properly reversed two more conclusions. The high-SNR collapse is
**not** an artefact of a noiseless injection — a Gamma-fluctuating version
reproduces it to within a few points. And the shipped `shortlist_below=0.1`,
which the first write-up called "not a defensible operating point", turns out
to be **the best of every threshold swept**, by injected signal kept per
survivor admitted.

The day's last finding was a data-handling one, and it is still open:
`AGENTS.md` said *"the repository is private for that reason"*. It is not.
