# GLOBULAR Clustering Technical Reference

**Purpose:** dense, agent-oriented context on the GLOBULAR clustering method — its feature space, preprocessing, batching scheme, hyperparameters, results and known limits — plus an explicit map onto our Track B implementation, recording where we match the paper and where we deliberately or accidentally diverge.

**Primary source:** Jacobson-Bell, B., Croft, S., Choza, C., Andersson, A., Bautista, D., Gajjar, V., Lebofsky, M., MacMahon, D. H. E., Painter, C., Siemion, A. P. V. 2025, *"Anomaly Detection and Radio-frequency Interference Classification with Unsupervised Learning in Narrowband Radio Technosignature Searches"*, AJ 169:206 (17 pp.), doi:10.3847/1538-3881/adb8e7, arXiv:2411.16556. Received 2024-12-02, accepted 2025-02-19, published 2025-03-13. CC BY 4.0. Local copy: `papers/globular_2025.pdf`. Reference implementation: https://github.com/bjacobell/gbt-hdbscan

All section, figure and table numbers below refer to that paper.

---

## 1. Citation, naming, and two internal inconsistencies

**Cite as Jacobson-Bell et al. 2025.** Until 2026-09 this repository cited it as "Brzycki et al. 2025", which is wrong — **Bryan Brzycki is not an author**. He appears only in the reference list, as author of tools the paper uses:

- `setigen` (Brzycki et al. 2022, AJ 163, 222) — synthetic signal injection.
- `blscint` (Brzycki et al. 2023, ApJ 952, 46) — the bandwidth measurement behind Feature 12.
- Brzycki et al. 2024 (AJ 168, 284) — a Galactic-centre scintillation filter, mentioned as *not* used here.

If you see "Brzycki 2025" anywhere, it is this error propagating. Fix it.

**Two inconsistencies inside the paper itself.** Do not treat either as an error on our side:

1. **The acronym expands two ways.** The abstract has "Grouping Low-frequency Observations By Unsupervised Learning After **Reduction**"; §1 has "...After **Rejection**". Prefer quoting the acronym alone.
2. **The family count is 59 or 58.** Figure 4's caption says "The 59 clusters shown here are an estimate"; the Appendix says "there are 58 clusters in total, not counting outliers". Our docs say "~59"; the tilde is doing real work. Neither number is a hard identification — the caption explicitly calls it an estimate of common RFI sources, not a 1-to-1 mapping.

---

## 2. Scope and premises

- GLOBULAR clustering **does not detect signals.** It consumes an existing hit list — turboSETI's here, ours from seticore — and any list of sufficiently precise detected frequencies will do.
- It is inserted **before** the spatial filter, not after. That ordering is the point: it de-densifies the RFI environment so the spatial filter can work in regions it previously could not.
- Because detection already happened, **clusters are RFI and outliers are the product**. This inverts earlier unsupervised work (e.g. Mesarcik et al. 2022) where noise forms the clusters and RFI are the outliers.
- Narrowband, continuous-wave signals only. Broadband/pulsed/dispersed searches need different methods and are out of scope.

---

## 3. Data sample and hit accounting (§1.2, §3)

Benchmarked against Choza et al. 2024: 97 of the 123 targets in the BL nearby-galaxy sample (Isaacson et al. 2017), dec > −20°, GBT L band, 100 cadences, 1.1–1.9 GHz. Search range ±4 Hz s⁻¹, SNR threshold 10.

| Stage | Hits |
|---|---:|
| turboSETI raw | 2,186,151 |
| − duplicates | −268,084 |
| − band-edge (<50 kHz from edge; breaks Features 7/8) | −164 |
| **unique, feature-calculable** | **1,917,903** |
| + recovered synthetic injections | +296 |
| **pre-HDBSCAN** | **1,918,199** |

Two details worth carrying:

- **Duplicates are harmless to turboSETI and fatal to density clustering.** 12.3% of the raw hits were duplicates. They do not change turboSETI's conclusions but they inflate local density, which is exactly what HDBSCAN keys on. We hit the same issue: 8,116 rows shared between `mk_sample_hits.h5` and `lband_long.h5`, deduplicated on `id` in `all_globular_features.parquet`.
- **turboSETI underestimates SNR by ~3.3×** (per Choza et al. 2024), so a nominal SNR-10 search is really an SNR-33 search. Figure 1 plots the raw uncorrected values. Relevant if comparing SNR distributions to ours.

**Injections, two distinct sets — do not conflate:**

| Purpose | Count | Where |
|---|---|---|
| **Recovery testing** | 100 events / 300 hits (296 recovered by `FindDoppler`) | injected into the M31 cadence once |
| **Seeding** | 100 signals × 13 features | injected into *every* batch, identical each time |

Both from `setigen`, drift rates −3 to 3 Hz s⁻¹, SNR 1000–10000, modelled on spacecraft carriers (e.g. Voyager 1). Ceiling for recovery is 98 events, not 100, because `FindDoppler` missed 4 hits in 2 pairs.

---

## 4. The 13-feature space (§2)

Features 1–3 come from turboSETI; 4–13 are computed afterwards.

| # | Name | Definition | Notes |
|---|---|---|---|
| 1 | Observation frequency | MHz, from turboSETI | Useful because FCC allocations co-locate similar RFI. Risk: spectral occupancy varies enormously between bands, producing dense frequency clusters with few outliers between |
| 2 | Drift rate | Hz s⁻¹, **magnitude only, sign discarded** | Discretised at Δf = Δν/T_obs = 0.010204 Hz s⁻¹ for BL fine products. The paper flags the artificial gaps this creates near zero drift as unresolved |
| 3 | SNR | turboSETI value | See the 3.3× underestimate above |
| 4 | Spectral window skewness (SWS) | γ = m₃/m₂^(3/2), biased central moments, over a time-integrated spectrum of **2.7 kHz** | Window width from Painter et al. 2025 (in prep.): wide enough to capture the environment without drowning the signal |
| 5 | Spectral window kurtosis (SWK) | κ = m₄/m₂², **Pearson definition, not excess** | Pearson is required for Feature 6 to stay in unit range. Can reach ~10³ when the kurtic bandwidth is small |
| 6 | Spectral bimodality | Sarle's b = (γ²+1)/κ (Tarbă et al. 2022) | Bounded [0,1]; uniform distribution gives 5/9. Paper notes its PDF is unknown so no significance test applies; suggests Hartigan's dip test as an alternative |
| 7 | corr(SWK, log bandwidth) | Pearson r, over **50 log-spaced kurtic bandwidths from 200 Hz to 100 kHz**, but computed **only on the 200 Hz–2.7 kHz sub-range** | Proxy for linearity of the kurtosis–log(bw) relation. Departures indicate nearby signals |
| 8 | Turning-point bandwidth | Bandwidth maximising SWK, from the same 50-point sweep. **Parabola fitted through the maximum and its two neighbours**, vertex taken | A significant number of hits turn over beyond 100 kHz and are **pinned to 100 kHz** with no flag |
| 9 | Temporal window skewness (TWS) | As Feature 4, on the full ~5 min frequency-averaged time series | Indicates duty cycle. **Only 16 time samples (~18 s each)** — the paper calls this a known weakness |
| 10 | Time-series std | σ of the frequency-averaged time series | Same coarse-time caveat |
| 11 | Power-spectrum std | σ of the time-averaged spectrum | |
| 12 | Signal bandwidth | Width at **1% of maximum power**, via `blscint` | Searched to 5 kHz; nothing above ~3.1 kHz found; the vast majority are a few Hz |
| 13 | Periodogram redness | **Lomb–Scargle** over a **10 kHz** window; take the first fifth of the periodogram; ratio of mean power in its first half to its second half | Detects comb structure. "Frequency" here is cycles per hertz — the FT of the *frequency* axis, not of time |

**Mechanism behind Features 7 and 8 (§2, Feature 7).** Kurtosis over a window centred on a signal rises roughly linearly with log(bandwidth) at low bandwidths (~10²–10³ Hz). If a second signal sits Δν₀ away, kurtosis drops sharply once the window reaches 2Δν₀, then resumes rising. Eventually it peaks (the turning point) and falls toward the noise limit (3, Pearson). So Feature 8 is effectively "how far to the nearest neighbouring signal" and Feature 7 is "how cleanly linear is this relationship".

---

## 5. Preprocessing (§2) — authoritative table

Applied to the **full data set before batching**, so that scaling is global, not per-batch. This is explicit in §3.1 and matters: per-batch scaling would make clusters incomparable across batches.

| # | Feature | Transform |
|---|---|---|
| 1 | Frequency | quantile → **uniform**, then unit range |
| 2 | Drift rate | quantile → **normal**, then **unit variance** |
| 3 | SNR | log, then unit range |
| 4 | SWS | unit range |
| 5 | SWK | log, then unit range |
| 6 | Bimodality | **none** (already ~unit) |
| 7 | corr(SWK, log bw) | **none** (already ~unit) |
| 8 | Turning-point bw | log, then unit range |
| 9 | TWS | **unit maximum, negatives permitted** |
| 10 | Time-series std | log, then unit range |
| 11 | Spectrum std | log, then unit range |
| 12 | Bandwidth | unit range — **no log** |
| 13 | Redness | log, then unit range |

Derived from the sentence-form spec in §2: log for {3,5,8,10,11,13}; unit range for {1,3,4,5,8,10,11,12,13}; F2 unit variance post-quantile; F9 unit maximum with negatives kept; F6/F7 untouched.

**Why quantile transforms, not scaling.** Feature 1 gets a *uniform* quantile transform to preserve relative position in frequency while destroying relative density — otherwise a crowded band becomes one giant cluster and a quiet band becomes all outliers. Feature 2 gets a *normal* quantile transform because the zero-drift spike would otherwise be read as a cluster and the wings as outliers, which is "generally an incorrect interpretation". The trade is explicit: better low-drift anomaly sensitivity at the cost of some high-drift sensitivity, justified because high-drift emitters (Iridium) span a wider drift range where precision matters less.

**Feature ranges are a deliberate importance knob.** §2: "The range of any feature can be adjusted to induce a corresponding effect on the feature's relative importance." Rescaling to change a feature's influence is sanctioned by the paper, not a hack.

---

## 6. Batching and epochal iteration (§3.1)

**The failure being fixed:** at high volume HDBSCAN "lost sensitivity", grouping >90% of hits into one cluster.

Loop:

1. Preprocess the whole set (global scaling).
2. Draw batches of **~3,000** hits uniformly at random. Their set gave **639 batches**.
3. Inject the 100 identical seed signals into each batch.
4. Run HDBSCAN per batch, same hyperparameters everywhere.
5. **Keep only the outliers.** Discard clustered hits. Hits landing in a *seeded* cluster are retained as though outliers.
6. Pool survivors, reshuffle, repeat.

**Epoch-by-epoch (Table 1), 1,918,199 start:**

| Epoch | Anomalies | Reduction vs previous | % of original |
|---|---:|---:|---:|
| 0 | 1,918,199 | — | 100.0% |
| 1 | 1,005,473 | 47.6% | 52.4% |
| 2 | 749,904 | 25.4% | 39.1% |
| 3 | 585,257 | 22.0% | 30.5% |
| 4 | 452,430 | 22.7% | 23.6% |
| 5 | 330,067 | 27.0% | 17.2% |
| 6 | 250,142 | 24.2% | 13.0% |
| 7 | 190,122 | 24.0% | 9.9% |
| 8 | 133,149 | 30.0% | 6.9% |

Per-epoch reduction is roughly flat at 22–30% after the first — it does not asymptote within 8 epochs. Epoch count is a free parameter; the paper warns only that false negatives grow if it is set too high.

**Compute.** Unbatched O(N²) at cost AN²; `k` batches of N/k cost k·A(N/k)² = AN²/k — a factor-`k` speedup serially, up to `k²` with parallelism. Measured: **~13 hours unbatched → a few minutes for epoch 1 across 639 batches, serially.**

**Figure 3 — the bimodal batch population.** Cluster count per batch splits in two: a left population with very few clusters (insensitive: ≥90% of hits in one cluster, contributing no reduction) and a right population with ~10² clusters. **80.9% of batches land in the right population.** A batch in the left population is not an error; it simply passes its hits to the next epoch.

> This bimodality is the same phenomenon as the bistability we measured on our own data with `min_samples=2` — identical 3,000-point draws returning either k=2 holding 99.7% of points, or ~200 microclusters. See §8 and gotcha 9 in `AGENTS.md`. The paper's tuning heuristic exists precisely to keep runs out of the left population.

---

## 7. Cross-batch cluster matching (§3.2) — the piece we have not built

Cluster ids are local to a batch. To characterise the whole RFI environment they must be matched. **This is the recipe for our largest open Track B gap.**

1. **Key each hit to its cluster's centroid** in the 13-D feature space. A 100-hit cluster contributes its centroid 100 times, so the sample is weighted by cluster size.
2. **Sample 10,000 centroids** at random, with repetition (arising naturally from step 1).
3. **PCA → 6 components.** Chosen as the number capturing >95% of variance in the centroid set. Empirically better than embedding the raw 13-D.
4. **t-SNE → 2-D.** Perplexity 15–40 (Figure 4 used **40**); early exaggeration order 1–10 (used **4**). Higher perplexity favours high-level separation over low-level detail. The right perplexity depends on sample size.
5. **HDBSCAN on the 2-D embedding.** Its clusters are the RFI families.

**Health warnings, all from the paper:**

- t-SNE distances come from a nonlinear map and carry no intuitive meaning; axes are unitless (Wattenberg et al. 2016 cited on misreading embeddings).
- HDBSCAN on the embedding may split one family or merge several. There is a real homogeneity/redundancy trade-off; Figure 4 was tuned toward homogeneity, accepting redundancy.
- **Visual spot-checking is described as necessary, not optional.**
- Manual inspection found many cross-batch clusters as homogeneous as single-batch ones, but not all. Most inhomogeneous ones are clean composites of 2–3 homogeneous clusters.
- Suggested improvements, both deferred: hierarchical re-embedding of inhomogeneous clusters, and active learning.

**Identified families (Figure 9), by eye:** low-drift narrowband consistent with aeronautical radionavigation; sparse comb consistent with GPS L3 near 1381 MHz; dense comb consistent with GPS L5 near 1176 MHz; broad pulsed consistent with Iridium, 1610–1626.5 MHz.

---

## 8. Hyperparameters and the tuning heuristic (§3.1, §4)

Three parameters, in the paper's notation, with the scikit-learn names:

| Paper | scikit-learn | Large-scale run | Toy run (3,387 hits) |
|---|---|---|---|
| n_pts | `min_cluster_size` | **4** | 7 |
| ρ_pts | `min_samples` | **2** | 2 |
| ε_m | `cluster_selection_epsilon` | **0.18** | 0.24 |
| — | batch size | ~3000 | n/a |
| — | epochs | 8 | n/a |

**ε_m is the Malzer & Baum (2019) merging threshold** — clusters closer together in feature space than ε_m are merged. The paper footnotes explicitly that it is **not** DBSCAN's ε. It corresponds to sklearn's `cluster_selection_epsilon`.

**Tuning order (§4), reproduced because it is directly useful:**

1. **n_pts first**, with ρ_pts = 2 and ε_m = 0. Sweep integers 3→~10. There is a **stability threshold** above which a clear majority of hits (≥60%, sometimes >90%) collapse into one cluster. Set n_pts as high as possible without breaching it.
2. **ρ_pts = 2.** Their feature space is diffuse enough that raising it rarely helps once n_pts is below threshold. Raising it makes clusters more conservative and mostly just peels hits into the anomalous class. Their strategy is many small subclusters via low n_pts/ρ_pts, then merge with ε_m.
3. **ε_m**, sweep 0.15–0.25 in steps of 0.01, again aiming just below a stability threshold. Finer steps only pay off on small data sets.

For broader sweeps, they point at CORE-SG (Neto et al. 2022) and MustaCHE (Neto et al. 2018).

**Consecutive differently-tuned HDBSCANs help.** Doubling their toy set capped a single HDBSCAN at ~70% reduction; isolating the anomalies and running a second, differently tuned HDBSCAN raised it to ~95%. This is the small-scale justification for epochal iteration.

**Implementation note:** the main analysis uses **scikit-learn's** HDBSCAN. The McInnes `hdbscan` package is used only for SHAP, because `approximate_predict` had no sklearn equivalent as of 1.5.2.

---

## 9. Validation of the feature space (§4.1–4.3)

All three checks answer one question: *does any single feature dominate?* If one did, HDBSCAN would be clustering on it alone.

- **Random forest (§4.1).** 128 trees, all 13 features, labels from HDBSCAN, mean decrease in Gini impurity, normalised to unit sum, error bars spanning 68% of trees. Result: no feature significantly more important than the others. The paper is candid that training a forest on ~3k samples is not good practice and that error bars are wide.
- **SHAP (§4.2).** `KernelExplainer`, background 100 hits, explaining 400. Mean |SHAP| significantly above zero for every feature and comparable across features to within an order of magnitude. Feature values map sensibly onto Shapley values. **TWS and time-series std are visibly the noisiest** — consistent with 16 time bins.
- **PCA (§4.3).** >99% of variance in 10 of 13 components, but **no PCA is applied to the features**. Rationale: after feature extraction the data is already reduced, so "even 1% of the variance is significant for anomaly detection", and no correlations were strong enough to need decorrelating. PCA is used *only* on centroids before t-SNE (§7 above).

**If you take one thing from §4:** the remedy for a dominant feature is to **downscale its range in preprocessing**. That is the paper's own recommendation and it is what our `--scaling robust` does wholesale.

---

## 10. Results (§5, Table 2)

| | FP hits | FP events | TP hits | TP events |
|---|---:|---:|---:|---:|
| turboSETI alone | 1,917,903 | 288 | 296 | 86 |
| + GLOBULAR | 132,885 | **2** | 264 | 69 |
| Reduction | **93.1%** | **99.3%** | 10.8% | 19.8% |

- Of the 2 surviving events: one matches a Choza et al. 2024 event; one is new. Neither is a compelling candidate on visual inspection.
- The new event sits near the Galactic 21 cm line (~1420 MHz) in NGC 628 and appears to be stochastic intensity variation, not a narrowband signal. Speculated cause: the 21 cm line biases turboSETI's rms estimate over the 3 MHz coarse channel, lifting noise spikes above threshold. **The significant part is that de-densifying RFI let `FindEvent` find something it previously could not.**
- Of 296 recovered injected hits, GLOBULAR erroneously removed 32 (→264). Event recovery 86 → 69.
- Every event GLOBULAR recovered was also recovered by turboSETI alone; GLOBULAR adds no true positives.
- Typical loss mode: one hit removed from a single "on" scan, disqualifying the event under the high-duty-cycle assumption. **Refinements to the spatial filter, not to the clustering, are where those losses would be recovered.**

**Asymmetric trimming (§3).** RFI is trimmed from the "on" pointings only; "off" pointings are left untouched. Trimming both produced ~50 new false positives, by removing a continuous signal from an "off" without removing its counterpart in an "on" — manufacturing apparent sky localisation. The paper notes a counter-argument (a dense RFI environment can make a real signal *appear* in an "off"), and leaves it open.

---

## 11. Negative and null results worth not repeating

- **Anomaly ranking by nearest non-anomalous neighbour failed.** Euclidean distance to the nearest non-anomalous point, computed in 6–11 principal components (lower dimension chosen because Euclidean distance loses discrimination with dimensionality). Trials across that range gave **no significant improvement** in recovering injected signals.
- **PCA on the feature space was rejected**, deliberately — no clean elbow, and low-variance components matter for anomaly detection.
- **The drift-rate discretisation gaps are unmitigated.** Explicitly flagged as future work.
- **Turning-point bandwidths beyond 100 kHz are pinned, unflagged.** Extending the sweep is noted as potentially discovering new outliers, at prohibitive compute cost.

---

## 12. Map onto our implementation

Ours: `src/bluse/features.py` (registry), `src/bluse/track_b_features.py` (extraction), `src/bluse/track_b_cluster.py` (clustering), `src/bluse/bench/` (Cluster Bench).

### 12.1 Instrument differences that change the features

| | GLOBULAR (GBT) | BLUSE (MeerKAT) | Consequence |
|---|---|---|---|
| Spectral window | 2.7 kHz | **~121–196 Hz** (120 ch × 1.01–1.63 Hz) | Below their *minimum* sweep bandwidth |
| Kurtic bw sweep | 200 Hz – 100 kHz, 50 steps | ~5–196 Hz, 12 steps | F7/F8 probe the line, not its neighbourhood |
| Redness window | 10 kHz, Lomb–Scargle | ~150 Hz, **FFT periodogram** | Fine comb across the line, not wide combs. FFT is valid: our spectrum is uniformly sampled, which is the case Lomb–Scargle exists to avoid needing |
| Time samples | 16 × ~18 s | **24–57 × 5.017 s** | **We are better here.** Their noisiest features (F9, F10) are ones we have more reason to trust |
| Spatial filter | ABACAD on/off cadence | 64 simultaneous coherent beams | Clustering transfers unchanged; the filter it feeds does not |
| Bandwidth (F12) | `blscint`, to 5 kHz | our own, capped by the window | Saturates for anything wider than the stamp — check `f12_bandwidth_saturated` |

Net: **our feature values are not numerically comparable to published GLOBULAR values.** Same constructions, different regime.

### 12.2 Divergences from the paper's spec

Checked against §2 and the table in §5 above.

1. **F9 was normalised to unit range, not unit maximum.** Paper: *unit maximum, negatives permitted* — temporal skew is signed and the paper keeps that sign deliberately. We had `"unit"`, a min–max rescale to [0,1], which discards it. **Fixed 2026-09**: a `unit-max` transform divides by the largest magnitude, so the scale is unity and the sign survives.
2. **F12 was given a log the paper does not apply.** The paper's log list is {3,5,8,10,11,13}; F12 is unit-range only. **Fixed 2026-09**: `f12_bandwidth_hz` is now `"unit"`.

Both were re-run end to end; the effect on cluster counts is recorded in `aug_2026_workshop/README.md`.

**A third suspected divergence turned out not to be one, and the correction is worth recording.** An earlier version of this document claimed F7's correlation ran over the wrong sub-range: the paper correlates over 200 Hz–2.7 kHz within a 200 Hz–100 kHz sweep, whereas we correlate over our entire sweep. The paper's own cross-reference settles it — "using the bandwidths between 200 Hz and 2.7 kHz **(see Feature 4)**" — and Feature 4's 2.7 kHz *is* their spectral window. So the rule is "correlate from the bottom of the sweep up to the spectral-window width", not "up to 2.7 kHz". Our spectral window is the whole stamp, which is also the whole sweep, so correlating over the full sweep is the faithful transfer. No change needed.

### 12.3 Where we match

- Batching scheme, defaults `--batch 3000`, `--epochs 8`, `--min-cluster-size 4`. Keep-the-outliers loop is theirs.
- Global preprocessing before batching, not per-batch.
- F2 magnitude-only, quantile-normal; F1 quantile-uniform; Pearson (not excess) kurtosis for F5 so F6 stays bounded.
- Parabola-vertex refinement of the turning point (F8), with a saturation flag they do not have.
- Raw values retained alongside `_n` columns, so a downstream model can pick its own scaling.

### 12.4 Where we deliberately depart

- **`--scaling robust` is our default; theirs is "none" (their literal spec).** Their transforms alone left our features with IQRs spanning 0.036 (`x03_channel_offset`) to 5.88 (`f02_abs_drift`, whose quantile-normal transform throws the 33.5% zero-drift spike out to −5.2). Euclidean distance became drift rate and nothing else: two clusters holding 90% of the data. `--scaling none` reproduces their spec and clusters poorly *on our data*. This is sanctioned by §2 ("the range of any feature can be adjusted...") and is the same remedy §4 prescribes for a dominant feature.
- **`--min-samples 8`, not their ρ_pts = 2.** sklearn counts the point itself, so `min_samples` 1 and 2 are the same call — no core-distance smoothing, i.e. pure single linkage. That put us squarely in their Figure 3 left-hand population: identical 3,000-point draws returned k=2 with 99.7% of points, or ~200 microclusters with 40% noise. 8 collapses that to k ∈ [2,12].
- **No `cluster_selection_epsilon`.** Their ε_m = 0.18 was tuned for their scaling, not ours, and there is no working range for it here: sklearn's `_hdbscan/_tree.pyx:606` compares against `1/d`, so on our distance scale the parameter is either inert or annihilates every cluster. The control is removed from both the CLI and Cluster Bench. See gotcha 9 in `AGENTS.md`.
- **Three extra features** (`x01_drift_residual`, `x02_time_occupancy`, `x03_channel_offset`), not in the paper. The paper explicitly invites this.
- **F8 excluded from clustering by default** (`drop_saturated`): unresolved for ~72% of our hits, so including it mostly clusters on "hit the window edge".

### 12.5 Not implemented

| Gap | Where the recipe is | Value |
|---|---|---|
| **Cross-batch cluster matching** | §7 above | **Highest.** The difference between our 1,187 clusters and their ~59 families. Without it our cluster table is "these hits grouped with something", not a taxonomy |
| **Seed signals** | §3/§6 above | High and cheap. Inject identical synthetic drifting narrowband signals into every batch; hits joining a seeded cluster are retained as outliers. Makes the method sensitive to a signal type by example rather than by rule. We have `setigen`-free stamps, so this needs a synthetic-stamp generator first |
| Random-forest / SHAP importance audit | §9 above | Moderate. Would have caught our scaling problem directly rather than via IQR inspection |
| Anomaly ranking | §11 | **Low — they tried it and it did not work** |

---

## 13. Quick-reference constants

```
Citation          Jacobson-Bell et al. 2025, AJ 169:206
                  doi:10.3847/1538-3881/adb8e7, arXiv:2411.16556
Code              github.com/bjacobell/gbt-hdbscan

Features          13 (3 from turboSETI, 10 computed)
Spectral window   2.7 kHz            (SWS, SWK, bimodality)
Kurtic sweep      200 Hz - 100 kHz, 50 log-spaced points
  F7 sub-range    200 Hz - 2.7 kHz
Redness window    10 kHz, Lomb-Scargle, first 1/5 of periodogram
Bandwidth         width at 1% of peak power, searched to 5 kHz
Time series       16 samples over ~5 min (~18 s each)
Drift resolution  0.010204 Hz/s      (BL fine products)

Batch size        ~3000 hits         -> 639 batches
Epochs            8
n_pts             4                  (sklearn min_cluster_size)
rho_pts           2                  (sklearn min_samples)
eps_m             0.18               (sklearn cluster_selection_epsilon;
                                      Malzer & Baum merging threshold,
                                      NOT DBSCAN's epsilon)
Toy-set optimum   n_pts 7, rho_pts 2, eps_m 0.24  (3,387 hits)
Sensible ranges   n_pts 3-10, eps_m 0.15-0.25 in 0.01 steps

t-SNE             perplexity 15-40 (used 40), early exaggeration 1-10 (used 4)
PCA before t-SNE  6 components (>95% variance of the centroid set)
Centroid sample   10,000, with repetition

Hits              2,186,151 raw -> 1,917,903 unique -> 1,918,199 with injections
Reduction         93.1% hits, 99.3% events (288 -> 2)
TP cost           296 -> 264 hits (10.8%), 86 -> 69 events (19.8%)
RFI families      ~59 (Fig. 4 caption) / 58 (Appendix)
Compute           ~13 hr unbatched -> minutes batched (639 batches, serial)
```

---

## 14. Caveats for downstream reasoning

1. **Do not cite this as Brzycki et al.** §1.
2. **"~59 RFI families" is an estimate, not an identification.** The paper says so in Figure 4's caption, and the Appendix says 58.
3. **The 93.1% figure is a *hit* reduction on a *known-RFI* set.** It is not a detection efficiency and not transferable to our survivor counts without care.
4. **Their reduction rate depends on epoch count, which is arbitrary.** 8 epochs is a choice, not a convergence. Table 1 shows 22–30% per epoch with no plateau.
5. **Batching is integral, not an optimisation.** A single large pass clusters *worse*, not just slower — >90% of hits into one cluster for them; 2 clusters versus 71 on our `sband_short`. Anyone "simplifying" the loop away has broken the method.
6. **Clusters are RFI; outliers are the product.** The opposite of most unsupervised RFI work. Reading the cluster table as "the interesting things" is backwards.
7. **Cluster ids are meaningless across batches without §7's matching.** Ours are globally unique but still unmatched — distinct is not the same as matched. One RFI family still appears under many ids.
8. **Their preprocessing is not sufficient on our data.** Reproducing their spec literally (`--scaling none`) gives two clusters holding 90% of the data. Ours is not a deviation for its own sake.
9. **t-SNE structure must be spot-checked by eye.** The paper insists on this. If we implement §7, budget for the inspection, not just the code.
10. **The feature list is not sacred.** The authors call it representative rather than exhaustive, developed by hand on 3,068 hits, and probably instrument-dependent. Adding to it is expected practice, not a departure.
