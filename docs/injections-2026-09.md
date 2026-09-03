# Synthetic injections — the first ground truth

**Measured 2026-09-03, re-run twice after review.** 2,160 substrates (120 per
beam class per file, six files), 21 injection settings, 47,520 stamps through
the real feature path, **Gamma-fluctuating injection**. One command:

```bash
bluse-inject --fluctuate          # ~85 s; writes scores/injections{,_report}.parquet
```

> **This document replaces a first version whose headline was wrong twice
> over.** Both errors were found by review of PR #3 and both are recorded in §7,
> because the corrections are more instructive than the original. In short: the
> SNR axis was mis-derived by a factor of √2, and the headline retention number
> was a composition effect from a substrate set that was 42% confirmed
> multi-beam RFI.

Context: [`track-e-2026-09.md`](track-e-2026-09.md) §8, falsification risks 1
and 2.

---

## 1. What this buys

Every Track E number measures agreement with the multi-beam spatial filter — a
good instrument, not truth. No hit in this survey is confirmed clean. An
injected signal is one we built, so we know the answer.

## 2. The units, and how to convert them

The injected strength is a **matched-filter SNR**: for a template `g` the filter
weights the data by `g`, giving `SNR = A·√(Σg²)/σ`. That is independent of how
many channels and timesteps the signal occupies, so it means the same thing in
every band.

It is **not** seticore's SNR and must not be compared against the catalogue's
`snr` column. The harness records the single-channel dedoppler equivalent per
stamp, which is:

| harness SNR | 5 | 8 | 12 | 20 | 35 | 60 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **≈ catalogue-style SNR** | 2.5 | 4.1 | 6.1 | 10.2 | 17.8 | 30.6 | 50.9 |

So the grid spans roughly catalogue SNR 2.5–51 — ordinary survey brightness, not
an exotic tail. Every conclusion below is quoted in both units.

## 3. The three pre-checks

A harness that measures itself is worse than none.

| check | result |
|---|---|
| re-extraction reproduces the stored features | all 12 raw columns **bit-identical** |
| new data scored by a model that never saw its observation | `fold_models` / `predict_held_out`, pinned by an **exact** equality test against `fit_score` |
| normalisation floor | **0.000e+00**; `stored` and `control` scores agree exactly |
| values outside the trained range | recorded per row; **max 2 of 12 columns** |
| the injection is physically shaped | Gamma(k, 1/k) per sample, k from each stamp's own mean²/variance |

The normalisation check took three attempts and found something undocumented:
**the `_n` columns are fitted per file.** `bluse-features` calls `normalise()`
inside `extract()` once per file and `summarise()` concatenates without
re-normalising, so `x01_drift_residual = 0.2790005` is 0.1523 in `lband_long`
and 0.0253 in `mk_sample_hits`. The mechanism for a refit shift is per-file
**min–max** rescaling — `unit` and `log-unit` cover 8 of the 12 stamp columns
and **none of the 12 is a quantile transform**, contrary to what the first
version of this document claimed.

## 4. The result, stratified — because pooling it is what broke it

Substrate class is the largest single effect in the experiment. Reported
separately, always.

### The deployment population: single-beam substrates (n=720)

These are the hits the score is actually applied to. Control retention at the
shipped `shortlist_below=0.1` is **0.428**.

Fraction kept by a cut at 0.1:

| drift \ harness SNR | 5 | 12 | 20 | 35 | 60 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 Hz/s | 0.544 | 0.540 | **0.581** | 0.564 | 0.346 | 0.188 |
| 0.1 Hz/s | 0.496 | 0.489 | 0.418 | 0.343 | 0.161 | 0.068 |
| 0.0 Hz/s | 0.499 | 0.362 | 0.174 | 0.097 | 0.019 | 0.003 |

Paired shift, same substrate, injected − control:

| drift \ harness SNR | 5 | 12 | 20 | 35 | 60 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 Hz/s | −0.192 | −0.193 | −0.209 | −0.201 | **+0.011** | **+0.178** |
| 0.0 Hz/s | −0.129 | −0.085 | **+0.056** | **+0.211** | **+0.358** | **+0.469** |

**The response is modest and it reverses.** Injecting a real drifting signal
lifts retention from 0.428 to at best **0.581** — a 15-point gain, not the
transformation the first draft implied. Above catalogue SNR ≈ 10 (zero drift) or
≈ 30 (0.3 Hz/s) the shift turns **positive**: the injection makes the hit look
*more* like RFI than the noise it replaced.

### Pruning is NOT unaffected — this is the correction that matters most

The first version claimed the 2,988-hit pruned set was untouched by any of
this. Measured directly, fraction of injected signals pushed **above the
pruning threshold of 0.9**, single-beam substrates:

| drift \ harness SNR | control | 5 | 20 | 35 | 60 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 Hz/s | 0.335 | 0.199 | 0.228 | 0.504 | 0.740 | **0.914** |
| 0.3 Hz/s | 0.335 | 0.139 | 0.114 | 0.126 | 0.321 | 0.490 |

**At catalogue SNR ≈ 51 with no drift, 91% of injected real signals land in the
pruned bin.** A bright non-drifting sky signal would be actively discarded, not
merely ranked low. The claim that pruning survives was wrong and is withdrawn.

### Injecting into known RFI clears it (n=720 multibeam)

Control retention **0.001** — the score is confident and correct about these.
Superimposing a drifting carrier moves them into the shortlist up to **46%** of
the time (harness 35, drift 0.3), with paired shifts to −0.58.

That is a demonstrated false-**negative** pathway: a real signal coincident with
a known interferer drags the interferer into the candidate list. It is also the
strongest available argument that a low score is not evidence of anything.

## 4b. Two things the review asked for, and what they returned

### The injection model is not the explanation

The first version added a **noiseless, unmodulated ridge** — a morphology no
real detection has, because real integrated power adds a cross-term whose
variance grows with signal power. Measured, that showed up exactly where review
predicted: median `f10_timeseries_std` *fell* with injected brightness
(0.0347 → 0.0288), i.e. the stamp got smoother in time than pure noise.

Multiplying the added power by Gamma(k, 1/k), with k = mean²/variance per stamp
(median 13.7), fixes it: `f10` now runs 0.0347 → 0.0348, flat.

**The reversal survives.** Smooth against fluctuating, single-beam substrates:

| | keep @0.1, SNR 100 drift 0 | prune @0.9, SNR 100 drift 0 |
|---|---:|---:|
| smooth ridge | 0.003 | 0.914 |
| Gamma-fluctuating | 0.007 | **0.925** |

Within a few points everywhere. So the high-SNR reversal and the false-pruning
are **not** artefacts of a smooth ridge. Everything below is from the
fluctuating run.

### The injected-vs-real discriminator is confounded, and cannot settle validity

The proposed test — train a classifier to separate injected stamps from real
hits of matched brightness, on the same 12 features — returns **AUC 0.997–1.000**
at every setting. Read naively that says the model is extrapolating on every
injected stamp and its verdict there means nothing.

Two controls say the test cannot support that reading:

| comparison | AUC |
|---|---:|
| real hits vs *other* real hits at matched brightness | **0.479–0.499** |
| real **faint** hits (catalogue SNR 6–8) vs real hits at SNR ≈32 | **1.000** |
| our injections vs real hits at matched brightness | 0.997–1.000 |

The first validates the method — it returns chance when there is nothing to
separate. The second is fatal to the inference: **a faint real stamp is already
perfectly separable from a bright real stamp on these 12 features, with no
injection involved at all.** Every injection here sits on a faint substrate, so
"separable from real bright hits" is what a faint substrate does on its own, and
this design cannot attribute it to the signal model.

**The validity ceiling is therefore unresolved, not established.** A version
that could settle it has to hold the substrate fixed — two signal models on the
same substrates, or substrates drawn from the target brightness class rather
than the detection floor.

Worth noting on its own: that the 12 stamp features separate faint from bright
real hits at AUC 1.000 is a large fact about this feature space, and it is
consistent with everything else here about the score being brightness-driven.

## 4c. The operating point

The thing this experiment set out to produce and the first version left out. A
retention curve alone cannot choose a threshold — keeping everything keeps every
false positive too. Injected signals at drift 0.3 Hz/s and catalogue SNR ≈6–18,
against the 2,305 single-beam Track A survivors:

| threshold | injected kept | survivors admitted | | kept per admitted |
|---|---:|---:|---:|---:|
| 0.02 | 0.322 | 251 | 10.9% | 2.96 |
| 0.05 | 0.440 | 300 | 13.0% | 3.38 |
| **0.10 — shipped** | **0.545** | **346** | **15.0%** | **3.63** |
| 0.20 | 0.618 | 425 | 18.4% | 3.35 |
| 0.50 | 0.716 | 540 | 23.4% | 3.06 |
| 0.90 | 0.875 | 879 | 38.1% | 2.30 |

**The shipped 0.1 is the best of these**, by kept-per-admitted. That was a
convention when it was chosen and it survives being measured — which is a
better outcome than the first version's "0.1 is not a defensible operating
point", and the opposite conclusion.

What it costs is now stated rather than guessed: at 0.1 you keep **55%** of
injected signals in the favourable regime and admit 15% of survivors. Roughly
half of real signals in this band are lost to gain a 6.7× reduction in vetting.

## 5. What is safe to say

**Safe.**
- The score's discriminating power degrades at high brightness. Corroborated
  three ways: the injection reversal here; `report.json`'s `snr_stratified`
  deciles, where within-decile AUC falls from 0.996 at catalogue SNR 14–30 to
  0.61 at 498–777; and `candidates.csv`, where real single-beam survivors go
  from 26% shortlist retention at catalogue SNR 10–15 to 1.5% above 300.
- A low score is not evidence of a sky signal — §4's RFI-clearing result makes
  this concrete rather than cautionary.
- The shortlist is a ranking with a **measured** miss rate: on the deployment
  population it keeps 43% of nothing-injected and at best 57% with a signal
  present.
- **The shipped 0.1 threshold is the best of those swept**, by injected-signal
  kept per survivor admitted (§4c). It was a convention; it now has a number.
- The high-SNR reversal is **not** an artefact of a noiseless injection (§4b).

**Not safe, and withdrawn.**
- ~~"Pruning is unaffected."~~ False; 91% false-pruning at the bright end.
- ~~"A cut at 0.1 keeps at most 43% of injected real signals."~~ A composition
  effect. On single-beam substrates it is 43% *with nothing injected*.
- ~~"Morphology cannot separate origin at high SNR."~~ A causal claim resting on
  a synthetic morphology no real signal has (§6).
- ~~"Above SNR ~60 it inverts."~~ Right direction, wrong units, and the
  reversal point depends on drift: catalogue ≈10 at zero drift, ≈30 at
  0.3 Hz/s.

## 6. What this design still cannot tell us

**The Gamma multiply under-represents the cross-term.** It gives the added
component variance S²/k; the exact term is (S²+2SN)/k. So the injection is still
*less* noisy than a real detection of the same brightness — the conservative
direction for the smoothness question, but not exact.

**Bandwidth is fixed at 3.0 Hz** and never swept, though it drives
`f12_bandwidth_hz` and `f11_spectrum_std` directly. A real carrier is
unresolved at ~1 Hz channelisation; ours is not.

**The drift arms are not matched in feature space.** Seven of the twelve stamp
features come from the time-integrated spectrum, and at fixed matched-filter SNR
a drifting signal is several times less prominent there than a stationary one.
"Drift is doing the work" may partly reduce to "drifting signals perturb the
spectral features less".

**The substrate is never empty.** There is no empty cube in this archive.
Note the bound runs in opposite directions for the two quantities: a cleaner
substrate can only *raise* retention, but the paired shift is largest on RFI
substrates and near zero on the cleanest, so the pooled shift is an **upper**
bound on the response. The first version claimed a single lower bound over both.

## 7. What the first version got wrong

Kept because the corrections are the useful part.

**The SNR formula was not the matched filter.** It used
`A·Σg/(σ·√Σg²)` — a unit-weight numerator with a matched-filter denominator,
not a consistent filter at all. Monte Carlo: asking for 20 delivered **14.19**,
a factor of 1/√2. For the shipped Gaussian grid the error was a near-constant
√2 rescale (1.4142 across bands and drifts, worst deviation 0.4%), so no
qualitative conclusion moved — but it is genuinely profile-dependent once
bandwidth is swept, and an unresolved 0.8 Hz carrier gives 1.678 at zero drift
against 1.389 at 0.3 Hz/s. Found by Copilot. The test that should have caught it
asserted the algebraic inverse of the function under test, so it could not fail;
it now measures the real statistic by simulation.

**The substrate set was 42% confirmed multi-beam RFI.**
`select_substrates` filtered on `snr <= 8` alone, and low SNR does not imply few
beams: the draw was 42.4% multibeam, 44.4% ambiguous, **13.1% single-beam**. The
pooled "43% retention" mixed a population the score is deployed on with one it
was trained to reject, and the two respond in opposite directions. Substrates
are now sampled per class and reported per class.

**The refit-shift diagnosis was misattributed.** Blamed on
`QuantileTransformer` redrawing its subsample; none of the 12 stamp columns uses
a quantile transform. The mechanism is per-file min–max.

**`stored_score` was described as the shipped pipeline's score.** It is a
harness self-consistency check computed from one fold's ensemble, and the
"mean 0.017" attached to it was stale — it is exactly zero.

## 8. Next, in value order

Done since the first version: Gamma-fluctuating injection (§4b), the
injected-vs-real discriminator (§4b — confounded, reported as such), the
operating-point sweep (§4c), `bluse-inject`, provenance in
`scores/injections_report.json`.

1. **A discriminator that holds the substrate fixed** — the only way to bound
   the bright arm's validity, since the brightness-matched version cannot
   (§4b).
2. **Bandwidth in the grid**, including an unresolved carrier. It drives
   `f12_bandwidth_hz` and `f11_spectrum_std` directly and is the only harness
   parameter never swept. Now also the parameter that makes the SNR conversion
   profile-dependent (§7).
3. **Crop-retained Σg² per setting.** The amplitude is set on the full window
   but features are computed after cropping to 60 channels; at UHF, 0.3 Hz/s
   puts the track ±25.5 channels against a half-width of 30, so the realised
   in-window SNR is below the stated value and the drift arm is biased per band.
4. **A drift-matched column**, normalising on integrated-spectrum prominence
   instead of matched-filter SNR. Seven of the twelve features come from the
   integrated spectrum, where a drifting signal is several times less prominent
   at fixed matched-filter SNR — so "drift is doing the work" may partly be a
   statement about the normalisation.
5. **Cluster-bootstrap CIs over `obsid`.** 2,160 substrates over ~280
   observations are not independent; plain binomial intervals are optimistic.
6. **Real empty substrates**, selected on `x01_drift_residual` near its
   pure-noise value (≈ 60/√12 = 17.3 channels) rather than on catalogue SNR.
