# Cluster Bench review — findings and recommended work

**Status:** external review, 2026-09. Input to a spec, not a spec.
**Reviewed:** `src/bluse/bench/`, `src/bluse/track_b_cluster.py`, `src/bluse/features.py`,
`papers/GLOBULAR-technical-reference.md`, `AGENTS.md`, `aug_2026_workshop/README.md`,
and the committed `aug_2026_workshop/clusters/*_summary.csv`.
**Not reviewed:** the HDF5 data itself, `track_a_filter.py`, `explore.py`, `rfi_masks.py`.

---

## 0. How to read this

Every claim below is tagged, following the repo's own "verify before claiming" convention:

- **[repo]** — read directly out of the committed code, docs, or CSVs. Take as given.
- **[measured]** — computed by this reviewer in a standalone simulation calibrated to
  match a number the repo already reports. The *mechanism* is established; the exact
  figures are indicative and must be re-measured on the real feature matrix before
  being quoted anywhere.
- **[hypothesis]** — a causal claim that follows from the above but has not been tested.
  Each one comes with a cheap test. Do the test before acting at scale.

The single most important structural point, which shapes everything else: **the Bench is
currently a very good instrument with no objective function.** Every knob is exposed and
well explained, but nothing on screen says whether a configuration is *better*. Section 5
fixes that with data already sitting in the parquet, and it should be built before any of
the exploratory additions in section 7, because without it those additions produce more
things to look at rather than more things to conclude.

---

## 1. Corrections to earlier advice — do not implement these

Four suggestions were made to Landman before this repo was read. All four are wrong, and
they are recorded here so they do not leak into the spec.

1. **"Move the embedding downstream of the clusterer."** Already correct in the code.
   `cluster()` in `bench/app.py` runs HDBSCAN on the full-dimensional scaled matrix;
   `embed()` projects the same matrix for display only. The docstring records that an
   earlier version froze the plot geometry and that this was fixed. No action. **[repo]**

2. **"Add `cluster_selection_epsilon`."** Deliberately removed, for a good reason:
   sklearn's `epsilon_search` (`_hdbscan/_tree.pyx:606`) compares epsilon against `1/d`,
   so on this distance scale every value is either a bit-identical no-op or raises
   `TypeError`. Measured ARI 1.000 across 0.0/0.05/0.18/0.5. Do not re-add it to the
   sklearn path. Section 4.3 proposes reaching the same capability by another route.
   **[repo]**

3. **"Add unit-range / min-max scaling to reproduce the paper."** `--scaling none`
   already *is* the paper's spec: the GLOBULAR log / quantile / unit-range transforms
   live in `features.normalise()` and are applied upstream, so "none" means "the paper's
   preprocessing and nothing further". No action. **[repo]**

4. **"Rank anomalies by distance to the nearest non-anomalous point."** GLOBULAR tried
   this over 6–11 principal components and got no improvement in injected-signal
   recovery (§11 of the technical reference). `papers/GLOBULAR-technical-reference.md`
   §12.5 already grades it "Low — they tried it and it did not work". GLOSH is a
   different construction and is worth one cheap experiment, but it is a P3, not a P1.
   **[repo]**

---

## 2. Finding 1 — the zero-drift tie defeats robust scaling (highest priority)

This is the most consequential thing in the review. It is a *feature-space* defect, not a
UI defect, but the Bench is where it will be diagnosed and where the fix has to be
steerable.

### 2.1 The chain

- 22–47% of hits per file have `driftRate` **exactly** 0.0; on `sband_short` it is 33.5%.
  **[repo]**
- `f02_abs_drift` takes `abs(driftRate)`, so all of those become exactly 0.0. **[repo]**
- `normalise()` applies `quantile-normal` to it. `QuantileTransformer` maps ties to a
  single output value, and its normal output is bounded at ±5.199. So a third of the rows
  land on **one exact coordinate**, −5.199. **[measured]**
- That tie sits below the 25th percentile, so it *inflates the IQR of the column to
  ≈5.87*. This reproduces the 5.88 the repo reports for `f02_abs_drift`, which is the
  calibration check for everything below. **[measured, matches repo]**
- `--scaling robust` then divides the column by that inflated IQR. Result:

  | | after robust scaling |
  |---|---|
  | zero-drift slab | 33.5% of rows at one exact value (≈ −0.88) |
  | non-zero drift | compressed into an interval of IQR **≈0.16** |
  | a well-behaved feature, for comparison | IQR 1.00 by construction |
  | slab-to-continuum gap | ≈0.96, i.e. about one unit of a normal feature |

  **[measured]**

### 2.2 What this means

Two separate problems, both bad, and neither is what `robust` was added to fix.

**The IQR is not robust to a 33% tie.** The tie is *inside* the interquartile window, so
dividing by the IQR does not equalise this feature — it over-shrinks it. The informative
part of drift rate, the actual non-zero drift values, ends up contributing roughly **6×
less** to the Euclidean distance than any other feature. The Bench's own feature rail
cannot show this, because the bars are raw pre-scaling IQRs and the whole point is that
the post-scaling behaviour is anomalous.

**A third of every batch is exactly collinear in one dimension.** Those rows are confined
to a 14-dimensional slice of the 15-dimensional space. Their mutual distances are
therefore systematically smaller than for points spread over all 15 dimensions, so local
density in that slice is elevated by construction. GLOBULAR removed 268,084 duplicate hits
specifically because duplicates inflate local density and that is what HDBSCAN keys on
(§3 of the technical reference); a 33% coordinate tie is a weaker version of the same
pathology, and it is present in every batch by construction rather than by accident.

**[hypothesis]** This is a substantial part of the reason `allow_single_cluster=True`
returns k=1 on every batch and why every batch reads as "one connected blob". It is not
offered as the whole explanation.

### 2.3 Test it before acting

Cheap, on `sband_short_features.parquet`, no new code paths:

1. For every `*_n` column, report `n_distinct`, `max_tie_fraction`, and the post-scaling
   IQR under each of the three scaling modes. Confirm or refute the 0.16 figure for
   `f02_abs_drift_n` on real data.
2. Cluster three ways at otherwise identical settings — all hits; `f02` turned off; only
   `driftRate != 0` hits — and compare cluster-count, largest-cluster fraction, and noise
   fraction. If the tie is load-bearing, dropping the zero-drift rows should move the
   largest-cluster fraction substantially.
3. Report pairwise ARI across the three. Do not eyeball the scatter.

### 2.4 Fixes, in preference order

1. **Treat zero drift as a discrete state, not a small number.** Split `f02` into a
   boolean `f02_is_zero_drift` and a `f02_abs_drift` that is quantile-transformed over
   the non-zero rows only. This is honest — a zero-drift hit is categorically local RFI,
   not a hit that happens to drift slowly — and it removes the tie from the continuous
   feature entirely. **[measured]** confirms the non-zero rows then span the full
   transform range with ~25k distinct values.
2. **Add a pre-filter control to the Bench** (see item P1-3): cluster all hits / non-zero
   drift only / Track A survivors only / exclude RFI-masked. Track A already flags
   zero-drift as local RFI, so excluding it is the better-posed question in any case, and
   this makes it a one-click experiment rather than a code change.
3. **Add a `scaling` mode that equalises post-transform, tie-aware.** Divide by a spread
   statistic computed on the *distinct* values, or by the IQR of the non-modal part of
   the distribution. Keep `robust` as-is for comparability.

Whatever is chosen, `papers/GLOBULAR-technical-reference.md` §12.4 needs updating: it
currently presents `robust` as the remedy for the drift-rate dominance problem, and the
measurement above says it under-corrects rather than over-corrects.

---

## 3. Finding 2 — `f06_bimodality` carries no information and adds weight

`f04 = skew(_norm(spectrum))`, `f05 = kurtosis(_norm(spectrum), fisher=False)`, and
`f06 = (skew² + 1) / kurtosis` on **the same** `_norm(b.spectrum)`. So

$$\texttt{f06} = \frac{\texttt{f04}^2 + 1}{\texttt{f05}}$$

exactly, row by row, with no independent measurement anywhere in it. **[repo]**

This is not a criticism of the paper — Sarle's coefficient is defined that way, and
GLOBULAR includes all three. But Euclidean distance treats the 15 columns as an
orthogonal, equally informative basis, and here one column is a deterministic function of
two others. The (skewness, kurtosis) direction of the space is therefore weighted roughly
1.5× relative to everything else, for free, invisibly.

Note that the paper's own random-forest and SHAP audits (§9) would not have caught this:
they measure whether any *single* feature dominates, not whether a *direction* is
double-counted.

**Recommended:** do not remove it (reproducibility matters, and the nonlinearity means it
is not perfectly redundant for a *tree* model even though it is for a *metric*). Instead:

- add a **feature redundancy panel** to the Bench: correlation matrix over the active
  `*_n` columns, plus an explicit check for near-exact functional dependence, plus the
  variance-inflation-style question "how much of column *j* is predictable from the
  others". `f06` should light up immediately, and it is a good acceptance test for the
  panel.
- offer a `whiten` scaling mode (PCA with variances equalised, i.e. Mahalanobis rather
  than Euclidean distance) as the principled answer to double-counting. Mark it clearly as
  a departure from GLOBULAR, which deliberately rejected PCA on the features (§4.3) —
  their reasoning was about *dimensionality reduction*, not about decorrelation, so this
  is not in conflict with the paper, but it should not be the default.

Worth checking the same way while in there: `f07`/`f08` come from one shared sweep, and
`f10`/`f11` are both "std over mean" of the two projections of the same stamp.

---

## 4. Finding 3 — the clustering is stabilised on GLOBULAR's *insensitive* branch

### 4.1 What the committed results actually show

From `aug_2026_workshop/clusters/*_summary.csv`, computed directly: **[repo]**

| | `sband_short` | `all` |
|---|---:|---:|
| clusters | 72 | 1,491 |
| hits clustered | 34,916 | 1,281,791 |
| largest cluster | 2,987 | 2,721 |
| **median cluster size** | **11** | **518** |
| clusters with n > 1000 | 14, holding **98.2%** of clustered hits | 433, holding 80.8% |
| clusters with n < 50 | 58, holding 1.8% | 620, holding 0.6% |
| freq span < 1 MHz | 25 | 107 |

On `sband_short`: 12 batches of 3000 in epoch 1, and 14 clusters holding 98.2% of the
data. That is one blob per batch, and it means **epoch 1 consumes essentially the whole
dataset**; epochs 2–8 operate on the ~600 hits that survived. The epoch budget is being
spent in a single pass. **[hypothesis, but the arithmetic is tight]**

Compare GLOBULAR Table 1: 47.6% reduction in epoch 1, then a flat 22–30% per epoch, no
plateau at epoch 8, ending at 6.9% of the original. Ours ends at ~0.1%. That is not the
same regime, and the difference is not a tuning detail.

### 4.2 The branch problem

`AGENTS.md` gotcha 9 records that with `min_samples ≤ 2` the run is bistable: ten
identical 3000-point draws returned either k=2 holding 99.7% of points, or ~200
microclusters with 40% noise. `min_samples=8` was chosen to collapse that to k ∈ [2,12].
**[repo]**

The observation to add: **k ≈ 200 with 40% noise is GLOBULAR's right-hand population, and
k ∈ [2,12] with 99.9% clustered is their left-hand population.** Their Figure 3 says
80.9% of batches land on the ~10² side, and explicitly describes the few-cluster side as
*insensitive* and as contributing no reduction. The bistability was real and worth
fixing — but the fix landed on the branch the paper identifies as the failure mode.

The cause of the bistability is structural, not statistical. With
`allow_single_cluster=False`, EOM makes a single stability comparison at the root of the
condensed tree: either the root's two children win (k=2) or EOM descends all the way to
the leaves (k≈200). There is nothing in between *by construction*. Raising `min_samples`
does not remove the knife edge; it just biases which side of it you land on.

### 4.3 Proposal

**Add `cluster_selection_method` as a control, defaulting to `eom`, with `leaf` available.**
`leaf` takes the leaves of the condensed tree directly. It never makes the root
comparison, so it is not bistable, and it yields many small homogeneous clusters — which
is the regime GLOBULAR operated in and the regime that makes the epoch loop meaningful.
It is a one-word change to `HDBSCAN(...)` in two files. This is the highest
value-per-line item in the review. **[hypothesis: test with the section 5 metrics]**

`leaf` will over-split. GLOBULAR's answer to over-splitting was ε_m — "discover many
prospective subclusters with a low n_pts and ρ_pts, and then merge them with a
well-chosen ε_m" (§8, tuning step 3). That is exactly the capability sklearn cannot
provide here.

**So: add an optional `hdbscan` (McInnes) backend, selectable per run.** Justifications,
in order:

1. It is the reference implementation of Malzer & Baum's ε_m, so whether the merge
   threshold is usable outside sklearn becomes a measurable question rather than a
   blocked one. **Treat this as a hypothesis to test, not a fact** — the repo's finding
   about the `1/d` comparison may or may not carry over, and if it does not carry over,
   the leaf + merge strategy becomes available and GLOBULAR's ε_m ≈ 0.18–0.24 range
   becomes meaningful again.
2. `condensed_tree_.plot(select_clusters=True)` — a faithful picture of what the
   algorithm did, rather than a projection of its input. On the current configuration it
   would render one enormous high-mass branch per batch and make the collapse visible in
   a way no scatter plot will.
3. `cluster_persistence_`, for sorting the cluster table by something other than size.
4. `hdbscan.validity.validity_index` (DBCV), the density-based analogue of silhouette.
   Silhouette assumes convex clusters and will actively mislead here.
5. `approximate_predict`, which GLOBULAR needed for their SHAP audit (§12.5 grades that
   audit "Moderate — would have caught our scaling problem directly").

It is an optional extra in `pyproject.toml`, exactly like `umap-learn` already is, with
the sklearn path as the fallback. Do not make it the default until item P0-2 shows it is
better.

---

## 5. Finding 4 — there is no objective function, and one is already in the parquet

`_results.html` shows clusters, clustered %, noise, noise ≤4 beams, features, time. Not
one of those says whether a run is *good*. Tuning is therefore by eye, on a scatter plot
whose geometry the repo's own docs (correctly) warn against over-reading.

**The fix needs no synthetic data and no new science.** `weak_label` is already in every
feature parquet: 1 for hits in ≥ `--rfi-beams` beams, 0 for hits in ≤ `--clean-beams`
beams, −1 ambiguous. It derives from beam multiplicity, which is **not** in the feature
matrix — the 16 feature columns are all `f*`/`x*` and none is beam-based **[repo]** — so
it is a genuinely external label, not a leak.

Add, as headline stats:

- **AMI(cluster_label, weak_label)** over the rows with `weak_label != -1`. Adjusted
  mutual information, because it corrects for chance and for differing cluster counts,
  which raw MI and purity do not. A clustering that separates confident RFI from
  spatially confined hits scores well; one that finds 14 blobs of everything scores ~0.
- **Cluster purity distribution** w.r.t. `weak_label`, not just the per-cluster
  `rfi_pct` already in the table. The interesting statistic is what fraction of clustered
  hits sit in clusters that are >90% one class.
- **Largest-cluster fraction** and **median cluster size**. Two numbers that would have
  made section 4.1 visible on the first run.

Be honest about what this measures, in the UI copy as well as the code: `weak_label` 0
means *spatially confined*, not *verified clean*, and the repo is already careful about
this (positive-unlabelled framing, `AGENTS.md`). AMI against a weak label is a proxy for
"is this clustering picking up the thing we already know", not a detection metric. It is
still infinitely better than nothing, and its failure mode — rewarding a clustering that
merely rediscovers beam multiplicity — is detectable by checking whether AMI improves
when features known to be uncorrelated with beam count are added.

The synthetic-injection route (`setigen`-equivalent seed signals, GLOBULAR §3.1, graded
"High and cheap" in §12.5) is the right long-term objective function and gives a true
recovery rate. It needs a synthetic stamp generator first — a drifting narrowband
Gaussian line plus noise, matched to the stamp geometry, is on the order of 30 lines.
Sequence it after the AMI metric, because AMI is available today and unblocks the tuning
loop immediately.

---

## 6. Prioritised work items

Effort estimates are relative, not hours. Each item lists where it lands and how to know
it worked.

### P0 — build these first; nothing else is measurable without them

**P0-1. Objective-function stats in the results strip.**
AMI vs `weak_label`, largest-cluster fraction, median cluster size, purity summary.
*Where:* `bench/app.py::summarise` and the `Run.stats` dict; `templates/_results.html`.
*Done when:* switching `scaling` from `robust` to `none` produces a visibly different AMI,
and the number is reproducible across identical runs.

**P0-2. Per-feature tie and distinct-value diagnostics in the feature rail.**
Add `n_distinct`, `max_tie_fraction`, and post-scaling IQR (under the currently selected
scaling) alongside the existing raw-IQR bar. Flag any column with `max_tie_fraction > 0.1`.
*Where:* `bench/app.py::pick_dataset`, `templates/_controls.html`.
*Done when:* `f02_abs_drift` flags at ~0.34 on `sband_short`, and the post-scaling IQR
column shows the ~0.16 anomaly from section 2.1 (or refutes it).

**P0-3. `cluster_selection_method` control, `eom` default, `leaf` available.**
*Where:* `run_hdbscan` in both `bench/app.py` and `track_b_cluster.py`; one new select in
`_controls.html`; one new argparse flag.
*Done when:* `leaf` on `sband_short` at otherwise-default settings is reported with its
cluster count, largest-cluster fraction, noise fraction and AMI, and the comparison
against `eom` is written into `aug_2026_workshop/README.md` with numbers.

**P0-4. Per-epoch reduction table.**
GLOBULAR Table 1 for our runs: hits remaining after each epoch, reduction vs previous,
percent of original. `cluster()` already computes `alive` per epoch and records `origin`;
this is bookkeeping, not new logic.
*Where:* `bench/app.py::cluster` returns an epoch trace; new block in `_results.html`.
*Done when:* the trace on default settings shows the epoch-1 collapse described in
section 4.1, or shows that it does not happen.

### P1 — high value, unblocked by P0

**P1-1. Seed-run stability: N seeds, report ARI.**
Run the same configuration over *k* seeds (default 5), report mean and spread of pairwise
adjusted Rand index between the label vectors, plus the spread in cluster count. This is
the guard against exactly the bistability gotcha 9 documents, and for a benchmarking tool
it is arguably the most important single number on the page. Cache per (config, seed) so
the marginal cost is one run.
*Done when:* `min_samples=2` reproduces the documented 112× cluster-count swing as a
visible ARI collapse, and `min_samples=8` does not.

**P1-2. Cluster stamp grid.**
Click a table row → a tiled panel of 20–50 member stamps, sorted by distance to the
cluster centroid, nearest first, with the farthest few also shown. This is GLOBULAR
Figure 9, and the paper describes visual spot-checking as necessary rather than optional
(§7); §12.5 says to budget for the inspection, not just the code. The single-point
inspector is the right primitive at the wrong granularity for validating hundreds of
clusters.
*Where:* new `/cluster_stamps` route reusing `stamp()`; consider a small thumbnail cache,
since `lband_short_clean.h5` is uncompressed and random single-stamp reads are ~0.4 ms
(`AGENTS.md` gotcha 8) — 50 stamps is ~20 ms of I/O plus 50 matplotlib renders, which is
the real cost.
*Done when:* a >1000-hit cluster and a <20-hit cluster can be compared side by side by eye.

**P1-3. Pre-filter control.**
A select: all hits / `driftRate != 0` / Track A survivors (`pass_all`) / exclude
RFI-masked. Every one of those columns is already in the feature parquet. This makes
section 2.3's experiment a click, and it connects Tracks A and B in the tool where they
are currently connected only in the pipeline.
*Done when:* the non-zero-drift option changes largest-cluster fraction on `sband_short`,
and the delta is recorded.

**P1-4. Colour-by-feature-value on the scatter.**
Switchable between colour-by-cluster and colour-by-any-active-feature, with a diverging
ramp. Colouring by `f02_abs_drift_n` should render the zero-drift slab immediately; if
colouring by `f01_frequency_n` reproduces the cluster structure, that is a one-click
finding.
*Where:* `scatter.js` (`buildRGB`, plus a values endpoint alongside `labels.bin`).

**P1-5. Feature redundancy panel.**
Correlation matrix over active `*_n` columns plus a functional-dependence check.
*Done when:* `f06_bimodality` is flagged as predictable from `f04`/`f05`.

### P2 — worth doing, sequence after P1

**P2-1. Cross-batch cluster matching.** The repo's own largest Track B gap
(§12.5: "Highest"). Recipe in §7 of the technical reference. Two routes, and I would
build the second as default and the first as a reproduction path:
- *GLOBULAR's:* centroid keyed per hit → 10k sample with repetition → PCA to 6 → t-SNE
  (perplexity 40, early exaggeration 4) → HDBSCAN on the 2-D embedding.
- *Deterministic alternative:* Ward linkage on cluster centroids in the original
  15-dimensional space with an explicit distance cut. No perplexity, no seed, and the
  dendrogram can be cut interactively in the UI. The paper's own health warnings about
  t-SNE (§7) are the argument for having this option.
*Done when:* the results header reads "1,491 raw clusters → N matched families" and the
matched families survive the P1-2 stamp-grid inspection.

**P2-2. `hdbscan` backend as an optional extra.** Section 4.3. Ships
condensed-tree plots, `cluster_persistence_`, DBCV, `approximate_predict`, and answers
the ε_m question. Add as `pyproject.toml` optional-dependency `hdbscan`, fold into `all`,
fall back to sklearn when absent — mirroring how `umap` is handled today.

**P2-3. Run pinning and diff.** Pin a run, then show two runs side by side with a delta
column over the P0-1 metrics and a config diff. `HISTORY` already caches 12 runs by
parameter hash, so the state is there; only the comparison view is missing. For a tool
whose stated purpose is benchmarking strategies, this is the missing primitive.

**P2-4. Export the run config as a CLI invocation.** A copy-paste
`bluse-cluster --file … --scaling … --min-samples …` for the current Bench state. See
defect D-4 below: the two paths do not currently produce the same numbers, and this is
where that gets surfaced and fixed.

**P2-5. Marginal histograms per feature.** One small sparkline per feature, raw and
post-scaling, in the rail or on click. HDBSCAN clusters density modes, so any visible
spike is a cluster the algorithm will find whether or not it is wanted. Lower priority
than P0-2 because the tie statistics catch the specific pathology we know about; the
histograms catch the ones we do not.

### P3 — cheap experiments, low expected value, do them once and record the result

- **densMAP** (`umap.UMAP(densmap=True)`) as a third embedding. It adds a
  density-preservation term, which is the property you actually want when inspecting a
  density-based clustering. UMAP already costs ~60 s on 35k and blocks the request; if a
  third slow option goes in, move embedding computation to a background thread with a
  polling endpoint first.
- **PaCMAP** — much faster than UMAP, better global structure, fewer hyperparameters.
  Probably the better default than UMAP if it holds up.
- **PHATE** — designed for continuous filamentary manifolds. Worth one look precisely
  because the committed `*_space.png` plots look filamentary: if the feature space is
  continuous rather than blobby, that is a finding about whether density clustering is
  the right tool at all, and PHATE will show it where UMAP will manufacture islands.
- **GLOSH outlier scores** to rank the noise class. Downgraded per section 1, item 4 —
  GLOBULAR's analogous attempt failed. One experiment, record the result either way.
- **DBCV** as an internal validity index, if the `hdbscan` backend lands. Do not add
  silhouette; it assumes convex clusters and will mislead.

---

## 7. Small defects found while reading

**D-1.** `templates/_results.html` renders `{{ h.params.eps }}` in the run-history
fragment. `eps` was removed from the params dict, so Jinja renders it as empty and the
history line reads `… mcs 4 · eps  · 15f`. Cosmetic, one line.

**D-2.** The empty-result error in the same template advises "try a smaller
min_cluster_size, **a larger epsilon**, or turn more features on". Stale — the epsilon
control is gone and cannot come back on the sklearn path. Replace with the levers that
exist.

**D-3.** `scatter.js::buildGrid()` builds a 90×90 spatial index into `grid`, and
`nearest()` never uses it — it does a full O(n) scan with `toScreen` per point on every
`mousemove`, striding by 2 only above 120k points. At 35k that is 35,000 coordinate
transforms per mouse move. Either wire `nearest()` to the grid or delete `buildGrid()`.

**D-4.** The Bench and the CLI fit their scalers on different row sets. `load_dataset`
samples 35k *then* `scale()` fits on the sample; `track_b_cluster.feature_matrix` fits on
all `good` rows. GLOBULAR §3.1 is explicit that scaling must be global and pre-batching,
"because per-batch scaling would make clusters incomparable" — and a 35k sample of 1.6M
is a 2% draw. So a configuration tuned in the Bench does not reproduce in
`bluse-cluster`, which undercuts the tool's purpose. Fix by fitting the scaler on the
full column set and applying it to the sample, and note the choice in the UI. Pairs with
P2-4.

**D-5.** No tests anywhere in the repository, and no CI. Given that gotcha 9 records
three defects that all presented identically ("the bench looks insensitive to every knob
except `min_cluster_size`"), a small regression suite would pay for itself. Three
assertions would have caught two of the three: cluster ids are globally unique across
batches and epochs; changing `scaling` changes the label vector (ARI < 1.0); the reported
cluster count equals `len(np.unique(labels[labels >= 0]))`. Add a fourth for the
invariant this review is about: no `*_n` column has `max_tie_fraction > 0.5`.

---

## 8. Conventions the implementation must respect

Read from `AGENTS.md`; restated here because a spec that violates these will be rejected
on review rather than on merit.

- **No data paths from `__file__`.** Everything user-supplied resolves through
  `bluse.paths`. Only `bench/static` and `bench/templates` are located relative to the
  module, because they ship in the wheel.
- **New CLI flags go through `paths.add_workspace_arg(p)`**, and any `--outdir` default is
  `None`, filled in after `parse_args`.
- **Filtering is non-destructive.** Cuts add `flag_*` columns; nothing drops rows in a
  pipeline stage. The P1-3 pre-filter should follow this: select on existing flags, do not
  invent a new destructive path.
- **Tag provenance on anything not measured**, the way `rfi_masks.py` labels entries
  SARAO / ITU / empirical.
- **Report two ways** — what a change does alone, and what it adds given what came before.
  Large gaps mean redundancy. This applies to the new metrics as much as to Track A cuts.
- **Verify before claiming.** Quote the number or say you did not measure it. Several
  claims in this document are explicitly tagged `[hypothesis]` for that reason and must not
  be written into the repo's docs as findings until they are tested.
- New features go in the `features.py` registry via the decorators, batch-vectorised, and
  return **raw** values — `normalise()` owns the transforms.
- Don't oversell a result. The stated deliverables are an RFI taxonomy, systematics
  discovery, and a demonstrated ranking workflow.

---

## 9. Explicitly out of scope / do not do

- Do not re-add `cluster_selection_epsilon` to the sklearn path. Section 1.
- Do not remove the batching loop or "simplify" it into a single pass. It is integral,
  measured at 71 clusters batched vs 2 unbatched on `sband_short`.
- Do not make the embedding an input to the clusterer. It is display-only and that is
  correct.
- Do not add silhouette score. Use AMI against `weak_label` now, DBCV later.
- Do not remove `f06_bimodality` on the strength of section 3. Diagnose it, weight it,
  document it.
- Do not change the three unresolved Track A judgement calls (ITU masks, the DTV comb,
  `--tol-steps`) as a side effect of anything here.
- Do not add rotation or frequency-flip augmentations anywhere near this data if the
  self-supervised track is touched. Gotcha 6.

---

## 10. If only three things get built

1. **P0-1** (AMI + largest-cluster fraction + median cluster size). Without an objective
   function every other change is a matter of taste.
2. **P0-3** (`cluster_selection_method=leaf`). One word, and it is the most plausible
   route from the insensitive branch to the regime GLOBULAR actually operated in.
3. **Section 2.3's experiment** (zero-drift tie). Three clustering runs and a table. If it
   confirms, the feature-space fix in 2.4 matters more than anything in the UI.
