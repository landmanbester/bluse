# What it taught us

Ten things, each anchored to an incident in
[`02-wrong-turns.md`](02-wrong-turns.md). None of them are general advice; all
of them cost us something specific.

---

## 1. "Nothing changes when I change things" is an alarm, not a reassurance

We read an unresponsive Bench as a robust method. It was three bugs
([#1–3](02-wrong-turns.md#a-the-instrument-was-broken-before-the-science-was)),
and one of them was reporting **731 groups as 205**.

Insensitivity has exactly two explanations: the parameter genuinely does not
matter, or you are not measuring what you think. The second is far more common
and is the one that costs a day.

> **Slide version:** if a knob does nothing, prove it does nothing before you
> call the method stable.

---

## 2. Measure the instrument before the science

Roughly two of the three days went into making measurements trustworthy rather
than into making discoveries — three new modules, a test suite, cross-batch
matching, and a per-feature diagnostic before any threshold could be set on it.

Every single time we looked at an instrument closely, something was wrong with
it. The estimator feeding our flag thresholds was **biased by 13× the seed
noise** ([#8](02-wrong-turns.md)). The one time we took a performance claim on
trust rather than measuring it, it was wrong by three orders of magnitude at
real scale ([#10](02-wrong-turns.md)).

That ratio is not a failure of planning. It is what the work is.

---

## 3. A record that disagrees with the computation is worse than no record

This happened **three separate times**, in three different subsystems, over
three days:

- a fix reported in a PR that had never been applied ([#9](02-wrong-turns.md));
- a CLI flag that was accepted, recorded in the metrics file, and **did nothing**
  ([#11](02-wrong-turns.md));
- a headline AUC that silently mixed accuracy with distribution shift
  ([#12](02-wrong-turns.md)).

A fourth instance is not about code at all: `AGENTS.md` stated *"the repository
is **private** for that reason"* as the project's entire record of how
unpublished survey data was protected, while the repository was public
([#24](02-wrong-turns.md)). Publication had in fact been cleared, so no harm —
but nothing recorded the clearance and the file that should have was asserting
its opposite.

No record is a known gap. A wrong record is believed. For code the pattern is
identical each time — two paths, one of them updated. For facts about the world
it is worse, because a protection described in a file is the thing people check
*instead of* checking reality.

The defence that worked was making tests assert **the written artefact against
the computation**, not against itself — refit from the recorded parameters and
require the same numbers.

---

## 4. Write the rejections down, and pin them with a test

Three of four planned improvements were rejected by measurement. Each rejection
is now a document explaining *why*, because every one of them is an idea a
competent person would propose again — they were proposed by competent people
the first time.

One is pinned in code:

> *"This test exists so the change cannot be made silently a second time. If you
> are here because it failed, re-run that experiment rather than updating the
> constant."*

A rejected idea with the reasoning attached is an asset. A rejected idea with no
record is a trap you will walk into twice.

---

## 5. A test that cannot fail is worse than no test

Two of them, both invisible:

- an equality assertion that passed while the value was wrong, **because
  `1 == True` in Python** ([#13](02-wrong-turns.md));
- a test named `test_amplitude_delivers_the_snr_it_was_asked_for` that asserted
  the **algebraic inverse of the function under test**, so it could only fail if
  someone edited both together — while the formula it guarded was wrong by
  1/√2 ([#18](02-wrong-turns.md));
- a test that pinned global state and turned **twelve real-data tests into
  silent skips** — which reads as "no data on this machine", not as a failure.
  A commit shipped with them not running ([#14](02-wrong-turns.md)).

Both passed. Both had the shape of coverage. Ask of any test you rely on: *what
change would make this fail?* If you cannot answer concretely, it is decoration.

---

## 6. When two numbers disagree, that disagreement is the finding

Two survivor counts differed — **5,413 against 4,565** — and it looked like a
deduplication detail. Chasing it produced the most immediately actionable result
of the three days: one delivered file is sampled at ~1% of the others' hit
density, so its beam-coincidence counts are meaningless. The same hit is counted
in **1.87 beams** there and **29.71** in the full file. Everything derived from
that file's beam counts is an artefact ([#15](02-wrong-turns.md)).

It had been written off an hour earlier as "cannot be resolved from these data."
It could; I had looked in the deduplicated table, where the evidence was
structurally invisible.

> **Slide version:** the reconciliation you are tempted to skip is the
> experiment.

---

## 7. Decompose your own headline before someone else does

The claim about to go on the opening slide was *"twelve numbers from the stamp
pixels beat the entire classical filter by 0.053 AUC"*, in front of the people
who built that filter.

Decomposed, **90% of the gap was the cost of binarising** and 10% was the
pixels ([#16](02-wrong-turns.md)). The claim was arithmetically true and
attributed almost all of its own effect to the wrong cause.

The correction is a better result than the original. What we can now hand them
is *"your thresholds are discarding nine tenths of the information in their own
inputs, and here is the 0.047 AUC you get back"* — a finding about their
pipeline rather than a competition with it.

---

## 8. Confidence is not calibration

Five times now, a claim in the highest-confidence class has failed on contact
with a check:

- the headline attribution (**wrong by 9×**);
- the survey-scale extrapolation (**not supported — 107× non-uniformity in the
  quantity it assumed was uniform**);
- the attribution table in a document *about* overclaiming (**invented**);
- the injection SNR axis (**wrong by 1/√2**, and the guarding test could not
  fail);
- **"pruning is unaffected" — which I had listed under *safe as stated*** in a
  summary whose whole purpose was to separate the safe claims from the unsafe
  ones. It had never been measured. Measured, it is 91% wrong
  ([#20](02-wrong-turns.md)).

The last one is the instructive failure. Sorting claims into *safe* and *not
safe* is exactly the right instinct, and doing it by introspection rather than
by checking which ones had been measured put the single most damaging claim on
the wrong side of the line.

Both had already been pushed. Neither was flagged as uncertain. Both were
produced by exactly the same process as the claims that turned out to be
correct, which is the uncomfortable part: **the internal feeling was identical.**

The only thing that separated them was someone asking.

**It happened a third time while writing these files.** The attribution table at
the top of [`02-wrong-turns.md`](02-wrong-turns.md) — in a document *about*
overclaiming — was invented rather than counted, and three of its four figures
were wrong. It is left in place, with the correction, because a worked example
beats the paragraph you are currently reading.

---

## 9. A control that returns the expected floor does not make a test unconfounded

Review proposed a good validity check and it returned a clean-looking **AUC
0.997–1.000**. The obvious control — real hits against other real hits at
matched brightness — returned **0.479–0.499**, exactly the floor. Method
validated, result apparently solid.

The second control killed it. Real *faint* hits against real *bright* hits
return **1.000**, with no injection at all — and every injection in the
experiment sits on a faint substrate. So the test could not attribute anything
to the thing it was built to test ([#23](02-wrong-turns.md)).

One control tells you the instrument works. It takes a different control to
tell you the comparison is clean.

## 10. The backlog item won

Track E was listed at P3 — *"still queued"* — for the whole three days, while
the effort went into the clustering track that produced three rejections and one
mode that misses its own acceptance criterion.

It was described in the original brainstorm as *"the cheapest genuinely novel
contribution"* and it took one evening.

This is not an argument for skipping the clustering work; without the labels,
the test discipline and the review reflexes built on days 0–1, the last night
would have produced a 0.99 AUC and no idea whether to believe it. But it is an
argument for **asking, at the start of every day, whether the queued item is
still the highest-value one** — and for noticing that a task described as cheap
and novel had been sitting untouched because it was neither urgent nor the thing
already in hand.

---

## A note on how this was done

The work was done with an AI coding agent, at ~78 commits in three days. That
pace is the reason the journey looks like this, in both directions.

**What it made cheap:** running an experiment to reject an idea rather than
arguing about it. Three of the four rejections in
[`02-wrong-turns.md`](02-wrong-turns.md) exist because measuring was faster than
debating. So does the entire de-confounding battery behind Track E — six
independent attempts to break a result, in one evening.

**What it made dangerous:** every failure in section D has the same shape. Output
that is *plausible*, *confident*, and *wrong in a way that does not raise an
error* — a chart highlighting the wrong five points, a test passing because
`1 == True`, a headline attributing its effect to the wrong cause, an
extrapolation stated without a hedge. None of these announce themselves.

The practices that actually caught things, in order of how much they caught:

1. **adversarial review** — 6 of the 23 entries, plus 13 more comments on PR #1
   alone. On day 3, the first time anything was read by someone who had not
   written it, review found **both** headline errors within hours;
2. **running the planned experiment and believing the answer** — the largest
   category across the self-reviewed days, and three of those four killed an
   improvement they were expected to confirm;
3. **reconciling numbers that disagree**, however small the discrepancy looks —
   a 848-row difference in a survivor count produced the single most actionable
   finding of the three days;
4. **being asked to justify confidence** — twice, in the last two hours, with a
   **100% hit rate**. Both claims had already been pushed.

The last one is the cheapest and the least automatable. It also has the worst
ratio of effort to yield in our favour: two questions, two retractions.
