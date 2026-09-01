# GLOBULAR Clustering: Finding the Unusual Signal by Naming All the Usual Ones

**A high-level overview for human readers**

Source: Jacobson-Bell, B., Croft, S., Choza, C., Andersson, A., Bautista, D., Gajjar, V., Lebofsky, M., MacMahon, D. H. E., Painter, C., Siemion, A. P. V. — *"Anomaly Detection and Radio-frequency Interference Classification with Unsupervised Learning in Narrowband Radio Technosignature Searches"*, **The Astronomical Journal 169:206** (17 pp.), 2025 April. [doi:10.3847/1538-3881/adb8e7](https://doi.org/10.3847/1538-3881/adb8e7), arXiv:2411.16556. Open access (CC BY 4.0). Code: [github.com/bjacobell/gbt-hdbscan](https://github.com/bjacobell/gbt-hdbscan).

> **Citation note.** This work was cited throughout our repository as "Brzycki et al. 2025" until 2026-09. That was wrong: Bryan Brzycki is not an author. He is cited *within* the paper as the author of two tools it depends on — `setigen` (Brzycki et al. 2022) for synthetic signal injection and `blscint` (Brzycki et al. 2023) for the bandwidth measurement in Feature 12 — which is the likely source of the confusion. The correct short form is **Jacobson-Bell et al. 2025**.

---

## The one-paragraph version

A technosignature search is an anomaly-detection problem with a terrible ratio: millions of detections, essentially all of them radio-frequency interference (RFI), and a handful — maybe zero — worth a human's attention. The usual response is a chain of filters that throw away anything looking like interference. GLOBULAR clustering inverts that. Instead of asking "does this signal look like RFI?", it measures 13 statistical properties of every detection and lets a density-based clustering algorithm discover, unsupervised, which *families* of interference are actually present in the data. Anything that joins no family is the output. On a real Breakthrough Listen survey of 97 nearby galaxies this removed **93.1% of false-positive hits and 99.3% of false-positive events** — 288 events down to 2 — while still recovering 69 of the 86 synthetic technosignatures that the conventional pipeline found.

---

## Why this matters for the BLUSE workshop

This is the most directly transferable published method we have, and it is the basis of our Track B.

The conventional chain (our Track A) is a sequence of hand-written rules: mask known RFI bands, reject zero drift, apply an SNR window, demand spatial localisation. Every rule encodes a human's prior belief about what interference looks like. That works, and it is interpretable, but it can only reject the interference someone thought to describe.

GLOBULAR clustering is **agnostic by design**. It has no library of RFI types. It finds the common morphologies purely from their prevalence in the data, which matters because — as the paper puts it — RFI sources are diverse and growing more so. A rule-based filter ages badly; a prevalence-based one does not.

The output is also a different *kind* of thing. Track A hands you "7,143 hits survived". GLOBULAR hands you "here are ~59 named families of interference, and here are the hits belonging to none of them". The second is a far better starting point for a human, because it converts an unexplained pile into a short list plus an explanation of everything else.

---

## The problem it solves

Two facts make naive clustering fail on this data, and the paper's contribution is mostly in handling them.

**HDBSCAN does not scale.** Its complexity is O(N²), accelerable to O(N log N) only for some data sets. On ~2 million hits their unbatched run took ~13 hours. Worse than slow, it was *wrong*: at high volume HDBSCAN lost sensitivity and dumped over 90% of hits into a single cluster.

**Density is not the same as interest.** Some frequency bands are crammed with signals and their neighbours are empty. Left alone, the clusterer finds one enormous cluster where the spectrum is busy and calls everything in the quiet bands an outlier — which is precisely backwards.

The answers are, respectively, **iterative batching** and **careful per-feature preprocessing**. Both are load-bearing; neither is a detail.

---

## How it works, in plain terms

```
turboSETI hit list                        2,186,151 hits
        │  remove duplicates + band edges
        ▼                                 1,917,903 unique hits
   13 features per hit                    a many-pixel waterfall → 13 numbers
        │  preprocess: quantile / log / rescale
        ▼
   ┌──────────── epoch (× 8) ─────────────┐
   │  shuffle, split into batches of ~3000│  639 batches
   │  HDBSCAN each batch independently    │
   │  KEEP the outliers, DISCARD the rest │  ← the reduction happens here
   └──────────────────────────────────────┘
        │                                  93.1% of hits removed
        ▼
  anomalies ──→ conventional spatial filter ──→ 2 events (from 288)
        │
  non-anomalies ──→ PCA → t-SNE → HDBSCAN ──→ ~59 named RFI families
```

The batching loop is the heart of it and the part most likely to be misread. Each epoch clusters ~3,000 hits at a time and then **throws away everything that clustered**, keeping only the points HDBSCAN could not group. Those survivors are pooled, reshuffled, and put through again. Each pass strips off another layer of recognisable interference; after eight passes 6.9% of the original data remains.

Batching is not a memory workaround. Small batches cluster *better* — and they are faster, since splitting an O(N²) problem into `k` batches costs `AN²/k`, a factor-`k` speedup before any parallelism. Their 13 hours became a few minutes.

One clever detail: 100 synthetic drifting narrowband signals are injected into **every** batch, identical each time. They seed clusters with the morphology you want to *keep*, so real technosignature-like signals join them and can be tracked through the epochs rather than quietly eroded. The paper notes this can be generalised — seed with whatever signal type you care about, and the method is tuned toward it without being told anything about RFI.

---

## The 13 features, in plain language

Three come free from the detection pipeline; ten are computed from the waterfall.

| # | Feature | What it captures |
|---|---|---|
| 1 | Observation frequency | Similar RFI clusters in frequency, because regulators allocate bands |
| 2 | Drift rate (magnitude) | How fast the signal slides in frequency — the classic technosignature discriminant |
| 3 | Signal-to-noise ratio | Brightness |
| 4 | Spectral window skewness | Is the spectrum lopsided? |
| 5 | Spectral window kurtosis | How peaked — high values mean broad, drifted, or comb-like |
| 6 | Spectral bimodality | Sarle's coefficient: sharpness of the peak; two lines or one? |
| 7 | Kurtosis vs log-bandwidth correlation | How the spectral neighbourhood behaves as you widen the window |
| 8 | Turning-point bandwidth | The window width at which kurtosis peaks — a proxy for spacing to the nearest neighbour |
| 9 | Temporal window skewness | Duty cycle: is the signal pulsed, or on the whole time? |
| 10 | Time-series standard deviation | Temporal variability |
| 11 | Power-spectrum standard deviation | Spectral variability |
| 12 | Signal bandwidth | Width at 1% of peak power. Most signals are a few hertz wide |
| 13 | Periodogram "redness" | Comb structure — many evenly spaced lines, as GPS and DTV produce |

The authors are explicit that this list is **representative, not exhaustive or optimal**, that it was developed by hand on a 3,068-hit toy data set, and that the right feature space probably differs by telescope and band. That is an invitation to modify it, which is exactly what our `features.py` registry is built for.

They checked, twice, that no single feature dominates — with a random forest trained on the cluster labels, and with SHAP values. Both said the features contribute comparably. This matters more than it sounds: if one feature dominated, HDBSCAN would effectively be clustering on that feature alone. (On *our* data, that is exactly what went wrong before we fixed the scaling — see below.)

---

## Does it actually work?

Benchmarked against the Choza et al. 2024 turboSETI search of the same 97 galaxies at L band:

| | False-positive hits | False-positive events | True-positive hits | True-positive events |
|---|---:|---:|---:|---:|
| turboSETI alone | 1,917,903 | 288 | 296 | 86 |
| **+ GLOBULAR** | **132,885** | **2** | **264** | **69** |
| Reduction | 93.1% | 99.3% | 10.8% | 19.8% |

Read the right-hand columns as the price. Roughly a fifth of genuine injected signals are lost, almost always because a hit was stripped from one "on" pointing, which disqualifies the whole event under the pipeline's duty-cycle assumption. Whether trading 20% of your sensitivity for a 99% cut in manual follow-up is a good deal depends entirely on how much follow-up capacity you have — and for a survey the size of BLUSE's, it is not a close call.

Both surviving events were inspected by eye and neither is a compelling candidate. One had been found before; one was new, near the Galactic 21 cm line, and appears to be noise-driven. That new find is the genuinely interesting result: removing dense RFI let the event-finder see something it had previously been blind to. **Cleaning up interference can reveal signals, not just remove them.**

---

## What the paper is honest about

- **t-SNE embeddings are treacherous.** Apparent groupings can be artefacts of the projection. The authors say plainly that visual spot-checks are necessary, and that some cross-batch clusters are visibly composites of two or three real families.
- **Their time resolution is poor.** 16 samples across a five-minute observation, which makes Features 9 and 10 noisy — visible as the messiest columns in the SHAP plot. (BLUSE is *better* here; see below.)
- **The turning-point bandwidth often runs off the end.** A significant number of hits have turning points beyond the 100 kHz search limit and are simply pinned to the maximum.
- **Anomaly ranking did not work.** They tried ordering the survivors by distance to the nearest non-anomalous neighbour, in 6–11 principal components. It gave no improvement in recovering injected signals. A useful negative result, and worth knowing before spending a day on the idea.
- **The features are hand-picked and admittedly unrefined.** Future work "may refine the list by introducing new features or pruning relatively unimportant ones".

---

## What we took, and what is different for BLUSE

Our Track B implements this method. Four differences matter, and the first is not a shortcoming:

**Our stamps are much narrower, and much finer in time.** GLOBULAR computes spectral statistics over a 2.7 kHz window and sweeps bandwidths from 200 Hz to 100 kHz. A BLUSE stamp is 120 channels at 1–1.6 Hz — about 150 Hz total, *below their minimum sweep bandwidth*. So our Features 7 and 8 probe the shape of the line itself rather than its spectral neighbourhood, and Feature 12 saturates for anything wider than the window. Conversely our time axis is 24–57 samples at 5.017 s against their 16, so the temporal features they call noisiest are the ones we have most reason to trust.

**We have beams, not an ABACAD cadence.** GBT rejects RFI by revisiting a target between off-source pointings. BLUSE forms 64 simultaneous beams, so our spatial filter is multi-beam coincidence. The clustering transfers unchanged; the filter it feeds does not.

**Scaling turned out to dominate everything.** GLOBULAR's published transforms alone left our features with interquartile ranges spanning 0.036 to 5.88, so Euclidean distance became drift rate and almost nothing else — the exact failure their random-forest check exists to catch. Our default adds a robust rescaling on top. This is a property of our data, not an error in theirs.

**We have not implemented cross-batch matching.** This is the gap between our "1,187 clusters" and their "~59 families". Their recipe is in §7 of the technical reference and is the highest-value piece of unimplemented work in Track B.

---

## Glossary

- **Hit** — a single narrowband detection in one beam or pointing.
- **Event** — a hit that survives the spatial filter, i.e. appears where a real sky signal should and not where it shouldn't.
- **HDBSCAN** — hierarchical density-based clustering. Finds clusters of varying density without being told how many to look for, and labels what fits none of them as noise. That noise label is the whole point here.
- **Epoch** — one full pass of batch-and-discard over the surviving hits. Named by analogy with supervised training; nothing is learned between them.
- **Anomaly / anomalous class** — hits that no batch could cluster. The output.
- **Seed signal** — a synthetic signal injected identically into every batch so that real signals resembling it have something to cluster with.
- **turboSETI** — Breakthrough Listen's narrowband detection pipeline. `FindDoppler` produces hits; `FindEvent` applies the spatial filter.
- **Redness** — power concentrated at low "frequency" in the Fourier transform *of the spectrum*, i.e. evenly spaced spectral lines. Not a time-domain frequency.
