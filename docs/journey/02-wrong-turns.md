# Twenty-five wrong turns

The useful part of the record. Each entry: what we believed, what was true, how
it surfaced, and what it cost.

Read the **How it surfaced** line first. Counted honestly:

| how it surfaced | n |
|---|---:|
| **adversarial review** (Copilot ×3, human reviewer ×4) | **7** |
| ran the planned experiment and believed the answer | 4 |
| investigated a result that felt implausible | 3 |
| reconciled two numbers that disagreed | 2 |
| was asked to justify a confident claim | 2 |
| ran a control that showed the test was confounded | 2 |
| a measurement taken for another purpose | 1 |
| checked a claim that had been taken on trust | 1 |
| looked at a rendered output | 1 |
| noticed a test count change | 1 |
| the collaborators handed over a document we had not thought to ask for | 1 |

After day 3 the largest category is **adversarial review** — but that is
because day 3 was the first time anything was reviewed by someone who had not
written it. On the six days' work reviewed only by its author, the largest
category was *running the planned experiment and believing what it says*, and
three of those four killed an improvement they were expected to confirm.

Both numbers point the same way: **nothing here was caught by reasoning
harder.** It was caught by measuring, or by someone else looking.

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

## E. Day 3 — the result gets audited, and most of it moves

### 18 — The SNR formula was not the matched filter

**Believed:** `A·Σg/(σ·√Σg²)` was a matched-filter SNR, and the docstring said
so at length.
**True:** it takes its numerator from a **unit-weight** filter and its
denominator from a **matched** one — not a consistent filter at all. Monte
Carlo against the real statistic: asking for 20 delivered **14.19**, a factor
of 1/√2. Every published x-axis label was wrong.
**Surfaced:** Copilot, on the PR.
**The test could not have caught it.** It asserted the algebraic inverse of the
function under test, with `sigma=1`. It would only ever fail if someone edited
the function and the test together. The replacement measures the statistic by
simulation.

### 19 — The substrate set was 42% confirmed multi-beam RFI

**Believed:** filtering substrates on `snr <= 8` selects the emptiest cubes.
**True:** low SNR does not imply few beams. The draw was **42.4% multibeam,
44.4% ambiguous, 13.1% single-beam** — so the headline pooled the population
the score is deployed on with the one it was trained to reject, and **the two
respond in opposite directions.** Stratified, retention on single-beam
substrates is 0.72 where the pooled number said 0.43.
**Surfaced:** review, with one `groupby` on the artefact I had committed.

### 20 — "Pruning is unaffected", published as safe, never measured

The claim was that Track E's 2,988-hit pruned set was untouched by any of this.
I put it under *safe as stated* in my own summary of what to trust.

It had never been measured. Copilot's comment said exactly that: the experiment
reported retention below 0.1 and said nothing about crossings above 0.9.
Measured: at catalogue SNR ≈51 with no drift, **91.4% of injected real signals
land above the pruning threshold**, against a 33% control. **A bright
non-drifting sky signal would be actively discarded, not merely ranked low.**

Of everything in this document, this is the one that would have done real
damage, because it was the reassurance attached to the deliverable.

### 21 — The diagnosis of my own measurement floor was misattributed

**Believed and documented at length:** refitting the transforms shifted scores
because `QuantileTransformer` redraws its 200,000-row subsample.
**True:** **none of the 12 stamp columns uses a quantile transform.** They are
`unit`, `log-unit`, `none` and `unit-max`; only `f01`/`f02` are quantile and
both are *meta* features the model never sees. The real mechanism is per-file
**min–max** — an injected value beyond a file's observed max moves `lo`/`hi`
and rescales every row in that file.
**Surfaced:** review, by checking `features.TRANSFORMS` against the column list.
The fix was right for the wrong reason, which is the hardest kind to catch.

### 22 — The "lower bound" ran in two directions at once

The write-up said the numbers were a lower bound because the substrates were
mostly RFI. True for **retention** — a cleaner substrate can only raise it. The
opposite for the **paired shift**, which is largest on RFI substrates and near
zero on the cleanest, so the pooled shift is an *upper* bound. One bound claim
was carried over two quantities that move oppositely.

### 23 — The test I ran to check validity was itself confounded

Review proposed a good check: train a classifier to separate injected stamps
from real hits of the same brightness. It returned **AUC 0.997–1.000**, which
reads as "the model is extrapolating on every injected stamp".

Two controls say it cannot mean that. Real hits against *other* real hits at
matched brightness give **0.479–0.499**, so the method works. But real **faint**
hits against real **bright** hits give **1.000** — with no injection at all. A
faint stamp is already perfectly separable from a bright one on these twelve
features, and every injection sits on a faint substrate.

So the check cannot attribute the separability to the signal model, and the
validity ceiling is **unresolved rather than established**. Reported as such.
The lesson is narrow and useful: a control that returns the expected floor does
not make a comparison unconfounded.

### 24 — A file described a protection that did not exist

**Believed:** `AGENTS.md` said *"The repository is **private** for that
reason"*, and that sentence was the project's whole record of how the
unpublished-survey data was being protected.
**True:** the repository was public. The tracked derived tables —
`candidates.csv` with 3,028 distinct `sourceName`, `injections.parquet` with
2,160 real hits over 362 observations — were world-readable.
**Surfaced:** review, checking a licensing claim against `gh repo view`.
**Resolution:** the owner had in fact cleared publication with the BLUSE team.
So no harm, and the failure is entirely documentary — **nothing in the
repository recorded the clearance, and the file that should have was asserting
the opposite.**

This is [lesson 3](03-what-it-taught-us.md) — *a record that disagrees with the
computation is worse than no record* — applied to a fact rather than a number,
and it is the worst case of the four, because a protection described in a file
and absent in fact is the one people check instead of checking reality.

### 25 — We never asked whether there was a format to deliver in

**Believed:** the shape of a feature file was ours to choose.
**True:** the workshop had a written convention the whole time —
`<dataset>_<method>_features.parquet`, hit `id` as the index — so that any
group's features can be cross-matched against any other's. We had written
`<dataset>_features.parquet` with `id` as an ordinary column and a
`RangeIndex`, for four days, across seven files and 830 MB.
**Surfaced:** the repository owner obtained `format.md` from the BLUSE side and
handed it over on the last morning. Not by review, not by measurement — by
someone asking a question of the collaborators that we had not thought to ask.
**Cost:** an afternoon, and nothing scientific. The files were already correct;
only the name and the location of the id were wrong, so
`bluse-features --migrate` rewrote them without re-extracting and every golden
number reproduced to the digit.

The interesting part is what adopting it exposed. The convention's own example
snippet is

    pd.DataFrame(index=data_file['id'][:], data=features).to_parquet(path)

and if a dataset's ids happen to run 0, 1, 2 … N−1, pandas turns that into a
`RangeIndex` and stores it as **three numbers in the file metadata instead of
writing an id column at all**. The ids are then not in the file, and a reader
gets 0…N−1 back — which is indistinguishable from having read them correctly.
None of our seven deliveries starts at id 0. That is luck, not protection, and
it is the same shape as [wrong turn 24](#24--a-file-described-a-protection-that-did-not-exist):
a safeguard everyone believes is in place, absent in fact, and silent about it.
The note went back into the shared document for the groups whose ids might.

There is no clever lesson here. Four days of building on an interface we
assumed rather than asked about, resolved in one conversation. **Ask what the
people you are delivering to expect, before you build the thing you deliver.**

---

## The tally

| | |
|---|---|
| planned improvements | 4 |
| rejected by measuring them | **3** |
| shipped without meeting its acceptance criterion | 1 |
| unambiguous wins | 1 — from the backlog, on the last night |
| **of that win's published conclusions, later withdrawn** | **4 of 8** |
| defects where the record disagreed with reality | **4** |
| interfaces we assumed instead of asking about | **1** |
| defects found by adversarial review rather than by us | **7** |
| tests that could not have failed | **3** |
| claims I listed as *safe as stated* that were not | **1** |
