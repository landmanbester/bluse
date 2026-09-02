# Track E — a morphology-only RFI score for every BLUSE hit

> **Superseded numbers, kept as the record of what was specified.** Every
> figure below was measured with a float32 feature matrix and a single seed.
> The shipped configuration is float64 averaged over three seeds, which moves
> most of them and reverses one conclusion — the 16-feature model now scores
> *higher* than the 12-feature default, so the D1 rationale in §5 is restated
> in [`../../track-e-2026-09.md`](../../track-e-2026-09.md) §4. Current numbers
> and the precision measurement itself are in §9 of that document. Annotated
> rather than rewritten, so the design decisions stay legible as they were
> taken.

**Date:** 2026-09-03 · **Status:** spec, measured · **Owner:** last day of the
August 2026 workshop

## 1. What this is, in one paragraph

The multi-beam spatial filter is BLUSE's strongest discriminant: a signal seen
in ≥32 of 64 coherent beams is local interference, a signal confined to one or
two beams might be on the sky. It is also **blind exactly where it matters** —
it needs many beams' worth of evidence, so it cannot judge a hit that appears
in one beam. A genuine technosignature appears in one beam. So does a weak
terrestrial emitter that only clears the detection threshold at boresight. The
filter cannot tell them apart.

Track E closes that gap. We take the spatial filter's own verdicts as free
training labels, train a classifier on **stamp morphology alone**, and get a
per-hit score that reproduces the spatial filter's judgement without using beam
counts at all — and therefore keeps working on the single-beam hits the filter
must abstain on.

## 2. Why this is worth the last day — it is measured, not hoped

Every number below is from a group-5-fold cross-validation on `obsid` over
1,603,364 labelled hits from seven files, 444 observations, measured on
2026-09-02 in `scratchpad/spike_e{1..5}.py`. Nothing here is projected.

| | ROC-AUC |
|---|---:|
| **stamp morphology, 12 features** | **0.9891** |
| all 16 features | 0.9895 |
| metadata features only (4) | 0.9844 |
| **Track A's entire flag set** | **0.9373** |
| `f01_frequency` alone | 0.8757 |
| raw `snr` alone | 0.7398 |

**Twelve numbers computed from the stamp pixels beat Track A's whole hand-built
flag set by 0.052 AUC.** That is the headline, and it is a statement about the
data, not about our code.

### It survives every stress test we could think of

| test | why it could have killed the result | measured |
|---|---|---:|
| drop all zero-drift rows | `x01` is NaN exactly there and P(RFI\|NaN)=0.996 — a 29.6% freebie | **0.9909** (up) |
| SNR-stratified, weighted mean over deciles | detection in 32 beams needs brightness; the score could be a brightness meter | **0.9243** |
| collapse to one row per signal | one emitter yields up to 64 near-duplicate rows | **0.9901** |
| singleton signals only (236,970) | removes duplication entirely | **0.9836** |
| hold out a whole band, train on the other two | could be memorising band-specific emitters | **0.90–0.99** |
| per file, within-file CV | could be carried by one file | **0.90–0.99**, all seven |

### The validation figure that makes the case

The model never sees `n_beams`. It is trained **only** on hits with ≤2 beams
and hits with ≥32. Scored on the untrained middle it reproduces beam
multiplicity monotonically across every bin:

| `n_beams` | 1 | 2 | 3 | 4–5 | 6–9 | 10–17 | 18–32 | 33–49 | 50–65 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean score | 0.044 | 0.090 | 0.140 | 0.232 | 0.411 | 0.682 | 0.881 | 0.949 | 0.984 |
| in training? | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

Five bins covering 329,000 hits were never labelled, and the score orders them
correctly and smoothly. That is out-of-class generalisation, and it is the
single strongest piece of evidence that the model has learned something
physical rather than a decision boundary.

## 3. The deliverable to the BLUSE team

Not a model — an **actioned candidate list plus the evidence for it**.

1. **A score on all 2,014,055 hits.** Honest everywhere: a labelled hit is
   scored only by a fold model that never saw its observation; an ambiguous hit
   is scored by the mean of the five.
2. **The candidate list, pruned and ranked.** Track A leaves 4,565 survivors.
   The model calls **2,708 of them (59.3%) morphologically indistinguishable
   from multi-beam RFI**, and 618 (13.5%) clean. That is a 59% reduction in
   vetting load at zero observational cost, with a per-hit reason attached.
3. **A verdict on the 410,691 hits the spatial filter abstains on** (3–31
   beams). 44.3% score >0.9, 41.1% score <0.5.
4. **Which morphology means RFI** — permutation importance over the 12 stamp
   features. `f04_spectral_skew` alone reaches 0.9636 AUC; `f06_bimodality`
   0.9531. These are directly reusable by any narrowband search.
5. **Stamps for the top candidates**, through the existing `bluse-explore
   stamps --rows`.

## 4. What this score is NOT — and this must travel with it

**Positive-unlabelled, not positive-negative.** Label 0 means *seen in ≤2
beams*. It does **not** mean verified clean. Most single-beam hits are still
RFI — the spatial filter is a sieve, not a proof. So:

- A **high** score is strong evidence of RFI: the hit is morphologically
  indistinguishable from signals independently known to be terrestrial.
- A **low** score is **not** evidence of a technosignature. It says only that
  the hit does not look like the multi-beam RFI in this survey. Reading it
  otherwise is the single most likely way this gets misused.

**It is a proxy.** The model predicts P(n_beams ≥ 32 | morphology). We use that
as an RFI proxy because the spatial filter is trusted, and the proxy inherits
every one of the filter's own blind spots.

**Brightness is partly confounded with the label.** A signal must be bright
enough to be detected in 32 beams. Within-SNR-decile AUC is 0.9243 against
0.9895 overall, so brightness contributes but does not carry the result.

**`mk_sample_hits` is pre-filtered** and contributes zero positive examples.
Excluded from training by default; still scored, and flagged in the output.
Measured cost of excluding it: 0.0012 AUC.

## 5. Design decisions, with the measurement behind each

**D1 — the default model is stamp-morphology-only (12 features).** It costs
0.0004 AUC against all 16 (0.9891 vs 0.9895) and buys three things: it cannot
relearn the RFI frequency mask, so it is not circular with Track A's own band
cut; it is the only variant that transfers to an unseen band without
retraining (held-out L: 0.9895 stamp vs 0.9834 all-16); and "twelve numbers
from the pixels" is a claim we can defend. `--features all` ships alongside and
both are reported everywhere.

**D2 — cross-validation is the delivery mechanism, not just the audit.** Five
fold models are kept. Labelled rows carry their out-of-fold score; unlabelled
rows carry the five-model mean. No row is ever scored by a model that saw its
observation.

**D3 — group split on `obsid`, always.** Hits in one observation share a
pointing, an RFI environment and a calibration. 444 groups.

**D4 — no new dependencies.** `sklearn.ensemble.HistGradientBoostingClassifier`
is already in the dependency set. A deliverable another team has to `pip
install lightgbm` to run is a deliverable they will not run. Full fit + score
of 2M hits takes **73 seconds** on 22 cores.

**D5 — the score column is `rfi_score`, and `NOMENCLATURE.md` defines it** in
the same terms as §4. The name is the one people will use; the definition has
to be one click away.

## 6. Scope boundary

**In:** the score, its validation, the pruned candidate list, the ambiguous
verdict, feature importance, stamps for the shortlist, the write-up.

**Out, deliberately:** SHAP (permutation importance answers the question with
no new dependency); calibration curves (the score is used as a *ranking*, and
the PU framing means a calibrated probability of "RFI" is not identifiable);
any deep model; retraining per band; and P0-4, which stays open.

## 7. Acceptance criteria

Written to be falsifiable, and each one reproduces a number already measured
in the spike — if the shipped code disagrees with the spike, the shipped code
is wrong.

1. `bluse-score` fits and scores all 2,014,055 hits in under 5 minutes with no
   dependency outside the current lock file.
2. Group-5-fold OOF ROC-AUC on the labelled set is ≥0.985 for the stamp model.
3. The `n_beams` monotonicity table is reproduced: mean score increases across
   all nine bins, with the five untrained bins interpolating.
4. SNR-decile-weighted AUC ≥0.90, signal-level AUC ≥0.98, singleton-only
   AUC ≥0.97 — the three de-confounding tests, pinned as regression tests.
5. Every labelled row's score comes from a fold that excluded its `obsid`,
   asserted in a test rather than argued in a comment.
6. The candidate list carries, per row, the score, the model, the fold, the
   Track A flags and enough provenance (`file`, `id`, `obsid`, `sourceName`,
   `frequency`, `snr`, `n_beams`) to look the hit up in the HDF5.
