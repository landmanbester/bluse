# Synthetic injections — the first ground truth

**Measured 2026-09-03.** 900 substrates (150 each from six files), 21
injection settings, 19,800 stamps through the real feature path. Reproduce with
`bluse-score` for the model and the harness in `src/bluse/injections.py`.

Spec context: [`track-e-2026-09.md`](track-e-2026-09.md) §8, falsification
risks 1 and 2 — this answers both, and not in Track E's favour.

---

## 1. What this buys that nothing else could

Every Track E number measures agreement with the multi-beam spatial filter. The
filter is a good instrument and it is **not truth**: no hit in this survey is
confirmed clean, so "0.9899 ROC-AUC" says two instruments agree.

An injected signal is one we built, so we know the answer. That makes three
previously unanswerable questions answerable, and the first of them is the one
the whole score is blocked on: **what threshold keeps a real signal?**

## 2. The three checks that had to pass first

A harness that measures itself is worse than no harness. Each of these was
measured, and the third found a defect in our understanding of the pipeline.

| check | why it matters | result |
|---|---|---|
| re-extraction is exact | if control features differ from the stored ones, everything after is measuring the harness | **all 12 raw columns bit-identical** |
| scoring is held out | the substrate is a real hit and is IN the training set | fold ensemble that excluded its `obsid` |
| normalisation is exact | any error here is a floor under the effect | **0.000e+00** across 3,000 rows |

The third took three attempts. Refitting `features.normalise` on the union
moved scores by **0.070 mean absolute** — four times the seed noise, enough to
swamp the effect — because `QuantileTransformer` redraws its 200,000-row
subsample when the array length changes. Interpolating on the stored
`(raw, raw_n)` curve cut it to **0.127**, still wrong. The residual was one raw
value mapping to two normalised values, which should be impossible.

**It is possible because the normalisation is fitted per file.**
`bluse-features` calls `normalise()` inside `extract()`, once per file, and
`summarise()` concatenates without re-normalising. A `_n` value is a rank
*within its own file*, not within the survey: `x01_drift_residual = 0.2790005`
is **0.1523** in `lband_long` and **0.0253** in `mk_sample_hits`. Interpolating
per file gives exactly zero residual. This was not documented anywhere and it
is not what "normalised feature" implies.

## 3. The result

`plots/track_e_injection_retention.png`. Fraction of injected signals a cut at
the shipped `shortlist_below=0.1` would **keep**:

| injected SNR | 5 | 8 | 12 | 20 | **35** | 60 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| drift 0.3 Hz/s | 0.376 | 0.372 | 0.381 | 0.386 | **0.430** | 0.399 | 0.204 |
| drift 0.1 Hz/s | 0.280 | 0.286 | 0.296 | 0.287 | 0.283 | 0.263 | 0.148 |
| drift 0.0 Hz/s | 0.230 | 0.234 | 0.228 | 0.202 | 0.157 | 0.093 | **0.006** |

Control — the same substrates with nothing injected — is **0.108**.

Three things, and two of them are limitations.

**The score does respond to a real signal.** Retention rises from 11% to 43%,
and the paired shift is negative at every setting below SNR 100: adding a real
drifting carrier makes a hit look less like multi-beam RFI, by −0.15 to −0.44
in score.

**But it keeps at most 43% of them.** At the shipped threshold, **more than
half of injected real signals would be cut.** Whatever else this experiment
established, it establishes that 0.1 is not a defensible operating point for
anyone who cares about not discarding signals. At 0.5 the best case is 65%.

**And it gets worse as the signal gets brighter.** Retention peaks near SNR 35
and collapses: at SNR 100 with no drift it is **0.006**, and the paired shift
turns **positive (+0.066)** — a bright carrier is scored as *more* RFI-like
than the noise it replaced. This is falsification risk 2 confirmed.

## 4. Why the reversal happens

Not extrapolation: at SNR 100 **no injected feature falls outside the range the
model was trained on** (checked per column, per file), so the model is
interpolating throughout.

The mechanism is visible in the features. Median `f04_spectral_skew` against
injected SNR, `sband_short`:

| injected SNR | control | 5 | 20 | 60 | 100 | 300 |
|---|---:|---:|---:|---:|---:|---:|
| median `f04` | 1.87 | 0.22 | 1.51 | 2.66 | 2.88 | 3.04 |

The file's own median is **3.99**. So as the injected signal brightens, the
stamp's morphology converges toward *what a typical hit in this file looks
like* — and a typical hit is RFI.

That is the honest statement of the limit: **a bright narrowband carrier is a
bright narrowband carrier.** Morphology cannot separate origin at high SNR,
because there is nothing morphological to separate. What separates them is beam
multiplicity — the thing the score deliberately refuses to look at, which is
exactly what made it useful on single-beam hits and exactly what makes it blind
here.

## 5. What it means for the shipped score

- **Pruning survives.** "This looks like RFI, vet it last" is unaffected: the
  score's high end is where it was trained and where it is reliable. The
  2,988-hit pruned set is not undermined by anything here.
- **The shortlist is weaker than it looked.** It keeps ~40% of real signals in
  the best case, so it is a *ranking with a known miss rate*, not a filter. It
  should not be used to discard.
- **Do not trust the score on very bright hits.** Above SNR ~60 it inverts.
  Any operational use should either exclude the bright tail or carry the
  caveat.
- **Drift is doing the work.** A zero-drift injection is scored as RFI at every
  brightness. Consistent with `f02`'s odds ratio of 2.96, and it means the
  score inherits the classical filter's blindness to a non-drifting sky signal.

## 6. The caveat this design carries

**Every injection sits on top of a real hit.** There is no empty cube in this
archive; the substrates are hits near the detection floor, and their controls
score 0.82 on average — they are mostly RFI themselves. So these retention
numbers are a **lower bound**: they measure "does adding a real signal to an
RFI-contaminated stamp make it look clean", which is harder than the question
we would like to answer.

The alternative — fully synthetic cubes with noise matched to real stamp
statistics — was considered and not taken, on the grounds that a wrong noise
model makes injections trivially separable for the wrong reason. **Running both
and comparing is the obvious next experiment**, and the difference between them
would itself measure how much the substrate matters.

## 7. What would change these conclusions

1. **A clean-substrate run.** If synthetic cubes give 90% retention where these
   give 43%, the limit is substrate contamination rather than the score, and
   the operating point is recoverable.
2. **A wider drift grid.** 0.3 Hz/s is the best of three values tested and the
   curve is still rising at the edge. The real distribution runs further.
3. **Injecting into more than one beam.** A terrestrial emitter appears in
   many; ours appears in one, by construction. Injecting the same signal across
   a beam set would test the filter and the score against each other directly.
