# Finding Interesting Things in BLUSE Data — Techniques Survey & Plan

**Workshop:** August 2026 · **Data:** `aug_2026_workshop/data/*.h5` · **Companion docs:** `papers/BLUSE-*.md`, `papers/Astronomaly-*.md`

This document surveys candidate techniques from the recent literature and proposes a plan. Everything in §1 was measured directly from the delivered files with `explore.py`; §2 onward is the technique survey and prioritisation.

---

## 1. What we actually have (verified, not assumed)

Seven HDF5 files, 21 GB, **2,022,171 hits total**. All share one columnar schema: 26 scalar-per-hit metadata columns plus a `data` cube of time-frequency cutouts. This is seticore "stamp" output flattened into a table — **the ideal input shape for everything discussed below**, because each row is simultaneously a feature vector *and* an image.

| File | Hits | Stamp cube | Duration | Δf | Band | Obs | Sources |
|---|---:|---|---:|---:|---|---:|---:|
| `lband_long.h5` | 557,690 | (n,1,57,120) | 286.0 s | 1.59 Hz | 855.7–1702.8 MHz | 100 | 3998 |
| `lband_short.h5` | 866,002 | (n,1,24,120) | 120.4 s | 1.59 Hz | 856.0–1068.0 MHz | 143 | 3420 |
| `uhf_long.h5` | 299,878 | (n,1,36,120) | 284.2 s | 1.01 Hz | 543.9–1080.0 MHz | 63 | 3850 |
| `uhf_short.h5` | 208,774 | (n,1,15,120) | 118.4 s | 1.01 Hz | 544.0–679.8 MHz | 76 | 2242 |
| `sband_long.h5` | 36,132 | (n,1,59,120) | 289.6 s | 1.63 Hz | 1968.8–2825.0 MHz | 32 | 866 |
| `sband_short.h5` | 38,576 | (n,1,24,120) | 117.8 s | 1.63 Hz | 1968.8–2825.0 MHz | 30 | 770 |
| `mk_sample_hits.h5` | 15,119 | (n,1,57,120) | 286.0 s | 1.59 Hz | 856.0–1702.8 MHz | 93 | 2695 |

Channel widths match the published values exactly (UHF 1.01, L 1.59, S 1.62–1.63 Hz). `_long` ≈ the paper's 290 s standard pointing; `_short` ≈ 118–120 s, i.e. **below the 150 s viability cut** quoted in Czech et al. — worth asking why they are in the sample.

### Columns

```
id index beam coarseChannel startChannel numChannels numTimesteps
frequency driftRate driftSteps snr power incoherentPower
ra dec fch1 foff tsamp tstart tstartts fileoffset telescopeId
sourceName obsid filename
data                       # (n, 1, numTimesteps, 120) float32
```

### Five findings that should shape the whole workshop

**1. `incoherentPower` is identically zero in every file.** The single most diagnostic classical test — `SNR_coherent ≤ √N_ant · SNR_incoherent` (Tremblay et al. 2026) — is **not available**. *(Confirmed by the BLUSE team: it was never measured for this data, so this is permanent, not a delivery we are waiting on.)* Multi-beam coincidence has to carry that load alone.

**2. Multi-beam coincidence alone kills ~96% of the data.** Measured on `sband_short` (1 Hz tolerance, per observation):

| Cut | Surviving hits |
|---|---|
| all | 38,576 (100%) |
| in ≤ 32 beams | 10,275 (26.6%) |
| in ≤ 8 beams | 2,684 (7.0%) |
| in ≤ 4 beams | 1,532 (4.0%) |
| **≤ 4 beams AND non-zero drift** | **1,300 (3.4%)** |

A real sky signal appears in one or a few coherent beams; local RFI lights up the whole 64-beam field. This is free, comes from metadata alone, and is the correct **first** step.

**3. 22–47% of hits have drift rate exactly 0.** Standard practice is to reject these as local RFI. Note `mk_sample_hits.h5` has 0% zero-drift, so it has evidently already been filtered — do not mix it with the others without accounting for that.

**4. There is a detached ultra-high-SNR population at SNR ~10⁷–10⁸,** completely separated from the main distribution (which runs 6 to ~10⁴). It sits at drift ≈ 0 and vanishes entirely under the coincidence cut — the max surviving SNR drops from 9.9×10⁷ to 2059. This matches Tremblay et al.'s finding that >90% of very-high-SNR detections are instrumental artefacts, and their consequent 100σ *upper* cut. **Do not feed raw SNR to any distance-based algorithm without a log transform and probably an upper clip.**

**5. Hits-per-beam is flat for beams 0–48, then steps down at ~49 and again at ~55.** *(Resolved after this document was first written.)* Not an instrumental systematic — BLUSE forms one coherent beam per catalogue target in the primary field of view, filling beams contiguously from 0, so a sparse field forms fewer than 64. Beam↔`sourceName` is 1:1 in 30/30 `sband_short` observations and beams formed falls monotonically with galactic latitude. The real consequence is that **the multi-beam coincidence denominator varies per observation**; `n_beams_formed` and `beam_frac` are recorded for that reason. `python explore.py beams <file>` visualises it.

### Practical gotchas

- **Padding:** stamps are right-aligned in a 120-channel buffer; unused leading columns are filled with exactly `-1`. `numChannels` (79–120) gives the true width. Mask `== -1` before doing anything numerical — feeding `-1` into a normaliser or CNN will poison it.
- **Compression:** the six band files are gzip-chunked at one stamp per chunk, so random access to individual stamps is efficient but bulk reads are CPU-bound on decompression. `mk_sample_hits.h5` is uncompressed and contiguous.
- **Dynamic range:** `power` spans 4.9×10¹⁵ to 1.9×10²². Always work in log space.
- **Drift rates are small** (±0.15 to ±0.44 Hz/s) and heavily quantised by `driftSteps` (integer). This is the detected drift, not the searched range.

---

## 2. Technique survey

### 2.1 Classical filtering — the essential baseline

Before any ML, reproduce the standard post-processing chain. If we skip this, every downstream algorithm spends its capacity rediscovering RFI.

From **Tremblay et al. 2026** (K2-18b, VLA + MeerKAT — the closest published precedent to our exact data):

| Step | Criterion |
|---|---|
| RFI frequency masks | Observatory-published bands; SARAO documentation for MeerKAT UHF/L/S |
| Zero-drift rejection | Remove drift rate exactly 0 |
| Drift-rate window | Physically motivated; e.g. ±0.4 Hz/s at 544–1500 MHz |
| Multi-beam spatial filter | Frequency tolerance **±1 Hz (MeerKAT)**, drift tolerance **±1 drift step** |
| Coherent/incoherent ratio | `SNR_coh ≤ √N_ant · SNR_incoh` — **unavailable to us (see finding 1)** |
| SNR window | 10σ–100σ; below is mostly false positives, above is mostly instrumental |
| Multi-epoch | Same frequency + drift on different days ⇒ reject |
| Visual inspection | Surviving candidates plotted as dynamic spectra |

Their result on real data: **zero surviving candidates**. That is the honest expectation for us too, and it is fine — the workshop goal is to find *interesting* structure, not to detect ETI.

Also relevant: **COSMIC's VLASS survey description** (2025) for a commensal-survey post-processing strategy at scale.

### 2.2 Feature engineering + clustering — GLOBULAR

**Brzycki et al. 2025**, *"Anomaly Detection and RFI Classification with Unsupervised Learning in Narrowband Radio Technosignature Searches"* (AJ). The most directly transferable published method.

- **13 hand-crafted features** per hit: observation frequency (quantile-transformed to uniform), |drift rate| (quantile-transformed to Gaussian), SNR, spectral-window skewness, spectral-window kurtosis, spectral bimodality (Sarle's coefficient), correlation between kurtosis and log bandwidth, turning-point bandwidth, temporal-window skewness, time-series standard deviation, power-spectrum standard deviation, signal bandwidth at 1% of max power, and "redness" of the spectral periodogram (detects comb structures).
- **HDBSCAN** clustering: `n_pts=4`, `ρ_pts=2`, `ε_m=0.18`. Run in **batches of ~3000 hits over 8 iterative epochs**, with cross-batch cluster matching.
- Supporting: PCA to 6 components, t-SNE (perplexity 40, early exaggeration 4) for visualisation, random forests + SHAP for feature-importance interpretation.
- **Result: 93.1% reduction in false-positive hits** (1.9M → 133K); 99.3% reduction in events; 69 of 98 injected synthetic signals recovered; ~59 distinct RFI source clusters identified.

**Why this fits us perfectly:** every one of those 13 features is computable from our `data` cube plus metadata. Their dataset (1.9M GBT hits) is almost exactly our scale (2.0M). Their key insight is that clustering *names the RFI families* rather than just scoring them — which turns "here are some outliers" into "here are 59 kinds of interference, and these 200 hits belong to none of them."

### 2.3 Self-supervised deep representations

**Ma, Croft et al. 2023**, *"A deep neural network based reverse radio spectrogram search algorithm"* (RASTI):
- **β-Variational Autoencoder** trained on energy-detection hits, with **a positional-embedding layer borrowed from the Transformer architecture to inject metadata (e.g. frequency) into the latent space** — an elegant trick directly applicable to us, where `frequency`, `driftRate` and `beam` all carry information the pixels don't.
- Operates on ~715 Hz windows at 2.79 Hz resolution — our stamps are 120 × 1.0–1.6 Hz ≈ 120–195 Hz, same regime.
- Enables **similarity search**: "find me more things that look like this one." Extremely useful for interactive vetting.

**Pardo, Poznanski et al. 2025**, *"Using anomaly detection to search for technosignatures in Breakthrough Listen observations"*:
- β-VAE plus ranking by **two orthogonal axes: rarity in frequency, and persistence in time** across consecutive spectrograms of the same target.
- Scale: ~10¹¹ spectrograms, ~20,000 candidates manually reviewed, none survived scrutiny.
- The ranking idea transfers directly: our `sourceName` + `obsid` + `tstart` columns let us build both axes.

**Mesarcik et al. 2023, ROAD** (*Radio Observatory Anomaly Detector*):
- Self-supervised on LOFAR spectrograms using **combined context-prediction and reconstruction losses**; 7050 spectrograms, 10 anomaly classes.
- F-2 = 0.92 at ~2% false-positive rate, <1 ms per spectrogram; beat autoencoder baselines by a large margin.
- Framed as **telescope health monitoring** — a legitimate and under-served use of our data. "Which observations behaved oddly?" is a real question with real value.

**BYOL / SimCLR for radio astronomy** (Slijepcevic et al. 2023; Mohale & Lochner 2024; Riggi et al. 2024): self-supervised features for source classification and peculiar-object discovery, plus efforts toward a radio-astronomy foundation model. This is the feature extractor behind Astronomaly's Protégé.

> ⚠️ **Augmentation warning, restated because it matters here.** All the above image-based methods rely on augmentations that encode "what transformations are physically meaningless." For galaxy images that includes rotation and flips. **For a dynamic spectrum it does not.** Time and frequency are not interchangeable, and rotating a stamp destroys the drift rate — the single most diagnostic quantity we have. Any BYOL/SimCLR run on these stamps must drop `RandomRotation`, and think hard about flips (a frequency flip negates drift sign). Legitimate augmentations: small time/frequency translations, noise injection, amplitude scaling, time-reversal-free crops.

### 2.4 Human-in-the-loop ranking — Astronomaly / Protégé

Covered in depth in `papers/Astronomaly-technical-reference.md`. Summary of fit:

- **Protégé** (GP active regression, Expected Improvement acquisition, no anomaly detector) is the better match for deep features; **classic Astronomaly** (iForest/LOF + `NeighbourScore`) is the better match for the ~13 hand-crafted features of §2.2.
- Our data maps cleanly onto `RawFeatures` (tabular) or `ImageThumbnailsDataset` (rendered stamps).
- Value here is less "find the alien" and more **"teach the machine our RFI taste in 100 clicks, then have it rank 2M hits by it."** That is genuinely useful and demonstrable within a week.

### 2.5 Supervised RFI morphology classification

CNN-based direction-of-origin filters and convolutional-autoencoder + random-forest RFI rejection are well established (e.g. Pinchuk & Margot 2022; Vafaei Sadr et al. 2020; Yang et al. 2025 for FAST-SETI). These need labels — but note that **§1 finding 2 gives us free weak labels**: anything appearing in >32 beams is RFI with near-certainty. That is a training set of ~28,000 confident RFI examples in `sband_short` alone, at zero labelling cost. A classifier trained on that could then score the *low*-multiplicity hits, which is exactly where the interesting things must live.

This "weak supervision from the spatial filter" idea is, I think, the highest-value original angle available to us.

### 2.6 Generic outlier detection worth trying cheaply

For the tabular feature matrix, these are one-liners and cost minutes: Isolation Forest, LOF, **HDBSCAN** (per GLOBULAR), ECOD/COPOD (parameter-free, in `PyOD`), and UMAP + HDBSCAN as a two-stage. Worth running several and comparing — [L&B21] and [L&R24] both stress that no single algorithm is complete.

---

## 3. Proposed plan

### Track A — Classical baseline *(essential; half a day)*
1. Apply SARAO RFI frequency masks per band.
2. Drop zero-drift hits.
3. Multi-beam coincidence filter, ±1 Hz and ±1 drift step, per `obsid`.
4. Log-transform SNR/power; flag the detached ≥10⁶ population.
5. Cross-observation repeat check via `sourceName` / `tstart`.

**Deliverable:** a filtered catalogue, plus the numbers showing what each step removed. Every later track starts from this. *`explore.py coincidence` already does step 3.*

### Track B — Features + clustering *(the workhorse; 1–2 days)*
Implement the GLOBULAR feature set (§2.2) over the stamp cubes, then HDBSCAN. Produce a **named taxonomy of RFI families** in our data, and isolate the hits belonging to no cluster.

**Deliverable:** an RFI atlas for BLUSE — genuinely publishable, and useful to the BLUSE team regardless of what else we find.

### Track C — Human-in-the-loop *(1 day, runs alongside B)*
Astronomaly on the Track B feature matrix. Classic path first (matched to hand-crafted features); Protégé if we get deep features working.

**Deliverable:** a ranked shortlist that a human actually reviewed, plus a recall curve versus random ordering.

### Track D — Self-supervised *(stretch; 1–2 days, needs a GPU)*
β-VAE or BYOL on the stamps with **drift-preserving augmentations**. Use it for similarity search ("more like this") rather than for raw anomaly scoring — that is where §2.3 shows the clearest wins.

**Deliverable:** a similarity-search tool over 2M stamps.

### Track E — Weak-supervision classifier *(the original angle; 1 day)*
Train on free labels from the spatial filter (high multiplicity = RFI, §2.5), then score the low-multiplicity hits. Effectively a learned, morphology-aware version of the spatial filter that works even where the spatial filter cannot.

**Deliverable:** a per-hit "RFI-likeness" score independent of beam multiplicity — and therefore able to *disagree* with it, which is precisely where interesting things hide.

### Also worth considering
- **Observation-level anomaly detection (ROAD-style).** Aggregate per `obsid` and ask which *observations* were weird. Different question, cheap to ask, plausibly the most immediately actionable output for the BLUSE team.
- **Cross-band coincidence.** Harmonics and intermodulation products of the same terrestrial transmitter should appear across UHF/L/S. We have all three.
- **Beam 49/55 step (finding 5).** Someone should just work out what that is.

---

## 4. Recommended priority

1. **Track A** — non-negotiable, half a day, removes 96% of the problem.
2. **Track B** — the workhorse; produces something valuable even in the null case.
3. **Track E** — cheapest genuinely novel contribution.
4. **Track C** — good demo value, low risk, and it is the workshop's stated theme.
5. **Track D** — highest ceiling, highest risk, GPU-dependent. Only if A–C are done.

## 5. Questions for the BLUSE team

1. Can `incoherentPower` be populated? It is zero everywhere and it is the strongest single discriminant. (Answer - the incoherentPower was not measured for this data.)
2. Why are the `_short` files ~118 s, below the stated 150 s viability threshold?
   (The 150 s is ours, from Czech et al. 2026 §6: of ~1.5M coherent beams
   "approximately 1.2 million were viable for technosignature searching (the
   remainder were too short in duration, less than 150s)". That describes survey
   triage, not a hard rule — but it does mean the `_short` files should be
   reported separately from the `_long` ones.)
3. What causes the hits-per-beam step at beams ~49 and ~55? (Answered by us:
   one coherent beam per catalogue target, filled contiguously from 0, so sparse
   sky forms fewer beams — 64 at |b|≈11°, 20 at |b|≈65°. Benign. See
   `explore.py beams`.)
4. Has `mk_sample_hits.h5` been pre-filtered? It has 0% zero-drift hits while the others have 22–47%. (Answer - yes, this one has been pre-filtered.)
5. Are per-antenna stamp data available? That would unlock true coherence testing rather than beam-multiplicity proxying. (Answer - The stamp files are huge, not in scope for this workshop.)

---

## References

- Czech, D. J. et al. 2026, *Breakthrough Listen's Automated Commensal Technosignature Survey with MeerKAT*, [arXiv:2607.23651](https://arxiv.org/abs/2607.23651)
- Tremblay, C. D. et al. 2026, *A Narrowband Technosignature Search toward the Hycean Candidate K2-18b Using the VLA and MeerKAT*, [arXiv:2602.09553](https://arxiv.org/html/2602.09553) · [IOP](https://iopscience.iop.org/article/10.3847/1538-3881/ae448e)
- Brzycki, B. et al. 2025, *Anomaly Detection and Radio-frequency Interference Classification with Unsupervised Learning in Narrowband Radio Technosignature Searches*, AJ, [IOP](https://iopscience.iop.org/article/10.3847/1538-3881/adb8e7)
- Tremblay, C. D. et al. 2025, *COSMIC's Large-scale Search for Technosignatures during the VLA Sky Survey*, AJ, [IOP](https://iopscience.iop.org/article/10.3847/1538-3881/ad9ea5)
- Ma, P. X., Croft, S. et al. 2023, *A deep neural network based reverse radio spectrogram search algorithm*, RASTI, [arXiv:2302.13854](https://arxiv.org/abs/2302.13854)
- Pardo, S., Poznanski, D. et al. 2025, *Using anomaly detection to search for technosignatures in Breakthrough Listen observations*, [arXiv:2505.03927](https://arxiv.org/abs/2505.03927)
- Mesarcik, M. et al. 2023, *The ROAD to discovery: machine learning-driven anomaly detection in radio astronomy spectrograms*, [arXiv:2307.01054](https://arxiv.org/abs/2307.01054)
- Lochner, M. & Bassett, B. A. 2021, *Astronomaly*, Astronomy & Computing, [arXiv:2010.11202](https://arxiv.org/abs/2010.11202)
- Lochner, M. & Rudnick, L. 2024, *Astronomaly: Protégé*, ApJ, [arXiv:2411.04188](https://arxiv.org/abs/2411.04188)
- Slijepcevic, I. V. et al. 2023, *Radio Galaxy Zoo: Towards building the first multi-purpose foundation model for radio astronomy with self-supervised learning*, [arXiv:2305.16127](https://arxiv.org/abs/2305.16127)
- *A Novel Technosignature Search in the Breakthrough Listen Green Bank Telescope Archive* 2025, AJ, [arXiv:2412.05786](https://arxiv.org/abs/2412.05786)
