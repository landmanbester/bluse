# Seventeen wrong turns

The useful part of the record. Each entry: what we believed, what was true, how
it surfaced, and what it cost.

Read the **How it surfaced** line first. Counted honestly:

| how it surfaced | n |
|---|---:|
| ran the planned experiment and believed the answer | **4** |
| investigated a result that felt implausible | 3 |
| adversarial review (Copilot) | 2 |
| reconciled two numbers that disagreed | 2 |
| was asked to justify a confident claim | 2 |
| a measurement taken for another purpose | 1 |
| checked a claim that had been taken on trust | 1 |
| looked at a rendered output | 1 |
| noticed a test count change | 1 |

The largest single category is **running the experiment you planned and
believing what it says** — which is worth sitting with, because three of those
four were experiments expected to confirm an improvement and instead killed it.

> **This table was itself wrong on first writing.** It originally read "six
> found by an adversarial reviewer, five by writing a test, four by a
> measurement taken for another purpose" — invented figures, in a document
> about overclaiming, stated with no hedge. Three of the four were wrong: the
> real counts are 2, 0 and 1. Caught by counting the entries instead of
> recalling them, which is [lesson 8](03-what-it-taught-us.md). It is left
> here because it demonstrates the failure mode better than the prose does.

---

## A. The instrument was broken before the science was

### 1 — "The method is robust to its hyperparameters"

**Believed:** the Bench barely responded to anything except `min_cluster_size`,
which we read as stability.
**True:** cluster ids collided across batches. HDBSCAN mints local ids `0..k−1`
in every batch; the code offset by epoch only. On `sband_short` **731 real
groups were reported as 205**, and the reported count was a *maximum over
batches* rather than a total — so of course only `min_cluster_size` moved it.
**Surfaced:** playing with the tool and disbelieving the result.
**Cost:** every cluster number produced before `358a282` is wrong.

### 2 — "`min_samples=2` is a reasonable default"

**True:** sklearn counts the point itself, so `ms=1` and `ms=2` are the same
call — no core-distance smoothing, pure single linkage. The runs were
*bistable*: ten identical 3,000-point draws returned either k=2 holding 99.7%
of points or ~200 microclusters with 40% noise. **A 112× swing on the shuffle
alone.**
**Cost:** any run at the old default was a coin flip we were reading as a
measurement.

### 3 — "`cluster_selection_epsilon` needs tuning"

**True:** sklearn tests epsilon against `1/d`, the *reciprocal* of a leaf's
split distance. Our splits are ≳1.7, so `1/d ≈ 0.55`: below that every value is
bit-identical to zero (ARI 1.000 for 0.0, 0.05, 0.18, 0.5); above it the code
raises `TypeError`. **The no-op region and the crash region tile the domain.**
**Surfaced:** reading `_tree.pyx:606` after the values did nothing.
**Outcome:** removed rather than re-ranged. There is no working value.

---

## B. Three of four planned improvements were wrong

### 4 — The restricted gap rule

**Believed:** a smarter rule would find a better place to cut the Ward tree into
families.
**True:** **every horizontal cut of a fixed Ward tree is uniquely determined by
the family count it leaves.** 1,400 thresholds across real trees and synthetics,
zero exceptions. A cut rule cannot select a better partition — only a point on a
fixed nested chain.
**Cost of not knowing:** we would have built and tuned a rule that could not
possibly do what it was for. Shipped `n_families=` instead — the interface the
docstrings had been recommending since before anyone built it.

### 5 — The `f02` ordinal rework

**Believed:** `f02_abs_drift` being the least-weighted column, with 26.6% of its
mass thrown to a transform bound, is a bug worth fixing.
**True:** the repair is **worse under `eom` at every family count** (best 0.372
against 0.519), and the proposed zero-drift indicator would have been weighted
**10.256** by the equalisation it was meant to serve — twice the value already
measured to destroy the clustering.
**Outcome:** kept, and pinned by a test whose docstring reads *"This test exists
so the change cannot be made silently a second time. If you are here because it
failed, re-run that experiment rather than updating the constant."*

### 6 — The k-NN equalising strategy

**Believed:** iteratively reweighting until every feature contributes equally to
local distances is the principled version of the fix.
**True:** the target is **mathematically unattainable for a tied column** — a
tied column contributes exactly zero to every tie–tie pair, and those pairs are
disproportionately near neighbours, so no weight lifts it to parity. The
iteration pins it at the ceiling from step 2 forever. It also **does not
equalise**: it collapses 15 effective dimensions to 2.22, top three columns
holding 96% of the squared weight — the opposite of the mode's name.
**Rejected despite scoring 0.83 against the shipped form's 0.51**, because a
three-line hand-built weighting beats it (0.850) and arbitrary column pairs at
the same effective dimensionality score 0.43–0.50. It was an expensive way of
rediscovering "cluster on drift and frequency".

### 7 — The one that shipped anyway

`--scaling robust-equalised` was built, reviewed, merged, and **does not meet
its own acceptance criterion** (0.513 against ≥0.60). Why it cannot be met is
**still open** (`P0-4`). It was merged on the explicit understanding that this
is answered later, not dropped, and the TODO says so in a banner.

---

## C. The measurements measuring the measurements

### 8 — The biased estimator underneath a threshold

We were about to set flag thresholds on a per-feature distance share.
`_shares_knn` built its neighbour index on **its own subsample**, thinning the
data and drifting the statistic toward the global share — a systematic error of
**up to 2.18 points, 13× the seed noise**. Fixed to index on all rows and sample
only the query points: 15–23× more accurate at the same cost. Every live figure
re-measured; two flags changed on `sband_short`, both of them the point of the
exercise.

### 9 — A fix reported but never applied

The PR said defect D-4 was fixed. It was not. `load_dataset()` computed
full-population median and IQR into a field **nothing ever read**; every path
still refit on the 35k sample. Found by Copilot, not by the human review, not by
me. Commit message: `including a fix I had claimed but not made`.

### 10 — A performance claim taken on trust

The module docstring claimed scipy's nn-chain Ward is O(n) in memory. Taken from
a review without checking. It is **O(k²)** — peak allocation tracks the
condensed distance matrix exactly, so the ~78,000 centroids `leaf` produces
would need **24.3 GB**. `match()` now refuses above 40,000 with a message naming
the size.

### 11 — A new mode that silently did nothing

`--scaling robust-equalised` was wired into one code path and missed by the
other. The CLI **accepted the flag, reported the weights in its metrics file,
and clustered the unweighted matrix** — verified elementwise identical to
`scaling="none"`. Found by Copilot. My own end-to-end verification had compared
against `robust` rather than `none`, so it could not have caught it.

**This is the same defect as #9, twice.** A record that disagrees with what was
computed. It happened a third time on the last night (#12).

---

## D. Track E, one evening, six more

### 12 — Accuracy mixed with distribution shift

`fit_score` excluded a pre-filtered file from training. `validate` still
**evaluated on it** — so the headline silently averaged real accuracy with
performance on data the model had never been allowed to learn. Reported 0.9795
where the trainable population gives 0.9899. Found by porting the numbers from a
scratch script and noticing they disagreed.

### 13 — `bool` is a subclass of `int`

The report serialiser wrote `True` as `1`. It survives `json.loads`, stays
truthy, and looks harmless — until a consumer does `np.array(flags)` and gets an
**int array**, where `x[mask]` is fancy indexing rather than masking and `~mask`
is bitwise NOT. **The money plot highlighted the wrong five points**, plausibly
and without error.

The existing test did not catch it **because `1 == True` in Python.** The
assertion compared dictionaries and passed while the value was an integer. It
now asserts identity, and that the serialised text says `true`.

### 14 — A test that silently disabled twelve others

A new test pinned the workspace to a temp directory and tried to restore it with
`set_workspace(None)` — which is a **no-op**, its body guarded by `if path:`.
Every later test in the session then resolved against the temp directory, so the
twelve real-data tests reported as **SKIPPED**. That reads as "no data on this
machine", not as a failure. **One commit was made with them silently not
running.**

### 15 — "This cannot be resolved from these data"

A pre-filtered file's labels disagreed with the model on 94.2% of its
"confined" hits. I looked for the same signals in a full file, found a 0.6%
match rate, and wrote that the question was unresolvable.

It was resolvable, and I had looked in the wrong table. The combined feature
table **deduplicates on `id`**, so the overlap was structurally invisible in it.
The per-file catalogues keep it: **8,116 hits appear in both files under the
same id, frequency and beam**, and the same hit is counted in a mean of **1.87
beams** in one and **29.71** in the other. The labels were the artefact. The
model had been right.

**Surfaced by reconciling two survivor counts that disagreed** — 5,413 against
4,565 — which I had noticed and nearly let pass as a rounding difference.

### 16 — A headline that misattributed 90% of its own gap

"Twelve numbers from the stamp pixels beat the entire classical filter by
0.053 AUC." Arithmetically true. Decomposed:

| step | ΔAUC | share |
|---|---:|---:|
| stop binarising (`flags` → `meta`) | **+0.0471** | **90%** |
| use the pixels (`meta` → `stamp`) | +0.0054 | 10% |

Track A's flags are *thresholds* on quantities the metadata holds continuously.
Adding the flags on top of those quantities buys **+0.0012** — the continuous
values essentially subsume their own thresholds. The claim was not measuring
morphology; it was measuring binarisation.

**Surfaced by asking, of a chart about to go in front of the people who built
that filter, "would this survive someone sharp?"** It would not have.

The correction is the better result: the *actionable* finding is that
**thresholding costs Track A 0.047 AUC** — nine times what morphology adds.

### 17 — An extrapolation the data does not support

"~81,000 survivors survey-wide", from: our 2M hits span ids 1–35.9M, so we hold
5.6%, so scale the 4,565 survivors accordingly.

The ids are distributed across that range with a **107× ratio between the
densest and sparsest decile** — 56% fall in the first tenth. Not a uniform draw.
The survivor *rate* varies eightfold between files anyway. And the source note
offers 5.6% as *a caution about representativeness*, which I turned into a
multiplier.

**Surfaced by being asked to justify confidence.** It had already been pushed.

---

## The tally

| | |
|---|---|
| planned improvements | 4 |
| rejected by measuring them | **3** |
| shipped without meeting its acceptance criterion | 1 |
| unambiguous wins | 1 — from the backlog, on the last night |
| defects where the record disagreed with the computation | **3** |
| defects found by adversarial review rather than by us | 6 |
| tests that could not have failed | 2 |
