# Cluster Bench review — design

**Date:** 2026-09-01
**Inputs:** `docs/bench-review-2026-09.md` (external review),
`docs/bench-review-2026-09-response.md` (our reply, with measurements),
`docs/bench-review-2026-09-addendum.md` (reviewer's second pass).
**Scope agreed:** P0 items + the objective metrics + cross-batch matching.
**Out of scope, deferred to a follow-up spec:** the contribution-equalising
scaling mode and the `f02` ordinal rework (see §9).

---

## 1. Why

Cluster Bench exposes every knob and explains each one, but nothing on screen
says whether a configuration is *better*, and nothing says whether a result is
*reproducible*. Three rounds of review converged on that being the defect
underneath all the others.

The measurements that motivate the work, all on
`aug_2026_workshop/features/sband_short_features.parquet` (34,933 rows reaching
the clusterer) at Bench defaults — `mcs=4`, `ms=8`, `epochs=8`, `batch=3000`,
`scaling=robust`, f08 off, 15 features — with `all_features.parquet` (1,281,878
rows) where stated:

| observation | measured |
|---|---|
| Two runs differing **only** in shuffle seed | pairwise ARI **0.024** |
| Epoch 1 removes | **87.9%**; epochs 4–8 remove **0** |
| Distance-share spread across 15 columns | **1.7%** (`f02`) to **24.3%** (`x03`) — 14× |
| `f07_kurt_bw_corr` clipped at ±5 on `all` | **4.3%** of rows, at 13.1% distance share |
| `weak_label` class balance | **26,956 : 872** — 31:1 |
| AMI dynamic range across all configurations tried | **0.0017 – 0.0048** |
| Narrow-cluster share (<1 MHz), `eom` → `leaf` | **0.776% → 6.820%** |
| `leaf` vs `eom`, **restricted** ARI (membership only) | **0.0316 vs 0.0279** |

Two of these overturn claims made earlier in the review cycle and are the reason
the design looks the way it does:

- **`leaf`'s stability advantage was an artefact.** The headline "20× more
  reproducible" (composite ARI 0.480 vs 0.024) is almost entirely agreement
  about *which points are noise* — `leaf` leaves 49.5% unclustered and
  `adjusted_rand_score` scores −1 as an ordinary label. On cluster membership
  alone it is 0.0316 vs 0.0279, a 13% improvement. `leaf` is still the right
  recommendation, but on the narrow-cluster metric (8.8×), not on stability.
  **Consequence:** stability is never reported as one number (§4.2).
- **AMI cannot arbitrate.** At 31:1 imbalance its whole observed range sits in
  the third decimal, and it ranks `eom` above `leaf` — the opposite of every
  other signal. **Consequence:** AMI ships, captioned, but the headline
  label-free metric is narrow-cluster share (§4.1).

---

## 2. Principles

Carried from `AGENTS.md` and §7 of the review; a design violating these is
rejected on review rather than on merit.

- No data paths from `__file__`. Everything user-supplied resolves through
  `bluse.paths`; only `bench/static` and `bench/templates` are module-relative.
- New CLI flags go through `paths.add_workspace_arg(p)`; any `--outdir` default
  is `None`, filled after `parse_args`.
- **Filtering is non-destructive.** Cuts add `flag_*` columns; no pipeline stage
  drops rows.
- New features go in the `features.py` registry via the decorators,
  batch-vectorised, returning **raw** values. `normalise()` owns transforms.
- Verify before claiming. Quote the number or say you did not measure it.
- Report two ways — what a change does alone, and what it adds given what came
  before.

One principle added by this work:

- **A statistic computed for both the Bench and the CLI is written once, in a
  module both import.** The two paths have already drifted (D-4); every metric
  in this spec is a shared function, not a reimplementation on each side.

---

## 3. Architecture

Three new modules, each a pure library with no FastAPI and no argparse, plus
wiring in the two existing entry points.

```
src/bluse/
  diagnostics.py   NEW  per-feature column audit
  metrics.py       NEW  cluster-quality and run-stability statistics
  matching.py      NEW  cross-batch cluster matching
  features.py      MOD  `kind` field on the registry
  track_b_cluster.py MOD  consumes all three; new flags
  bench/app.py     MOD  consumes all three; new routes
  bench/templates/ MOD  rail, results strip, epoch trace, families
  bench/static/    MOD  colour-by-value, nearest() fix
tests/             NEW  the repo's first test suite
```

Why three modules rather than one: they have different inputs and different
change rates. `diagnostics` takes a matrix and column names; `metrics` takes
labels plus a provenance frame; `matching` takes labels plus a matrix. Each is
independently testable, and none needs the others.

---

## 4. Module: `metrics.py`

### 4.1 Cluster quality

```python
def quality(labels, df, *, narrow_mhz=1.0) -> dict
```

`df` supplies `frequency`, `obsid`, `weak_label`. Returns:

| key | definition |
|---|---|
| `n_clusters` | `len(unique(labels[labels >= 0]))` |
| `clustered_pct` | |
| `largest_pct` | largest cluster as % of all rows |
| `median_size` | median cluster size |
| `narrow_frac` | **fraction (0–1) of clustered hits in clusters spanning < `narrow_mhz`**; rendered as a percentage |
| `narrow_frac_null` | the same under a size-preserving label permutation |
| `narrow_enrichment` | `narrow_frac / narrow_frac_null` |
| `narrow_clusters` | count of such clusters |
| `median_span_mhz` | median cluster frequency span |
| `ami` | AMI vs `weak_label` over rows where it is not −1 |
| `enrichment` | **fraction of clustered hits sitting in clusters significantly enriched** in `weak_label == 0` |

`narrow_frac` is reported at **two thresholds, 0.1 and 1.0 MHz**, so a reader can
see it is not an artefact of where the line was drawn. `narrow_mhz` is already a
keyword; this is a loop.

`narrow_frac` is the headline. It needs no labels, has a dynamic range of ~0.1%
to tens of percent, and rewards physical coherence — which is what an RFI
taxonomy, the stated Track B deliverable, actually requires.

**It gets a null, because every other metric here has one.** Small clusters are
narrow by chance more often than large ones, and `leaf`'s median size is 6
against `eom`'s 11, so part of the 8.8× could be arithmetic. `narrow_frac_null`
comes from permuting labels while **preserving the cluster size distribution**;
`narrow_enrichment` is the ratio. The confound is expected to be small — the
analytic floor for a size-6 cluster drawn from a single emitter holding 30% of
all hits is `0.30^6 = 0.073%`, two orders of magnitude below both observations —
but the headline metric being the only one without a null is exactly the
asymmetry that let AMI through unchallenged for a round.

`enrichment` uses a one-sided hypergeometric test per cluster against the global
`weak_label == 0` rate, Benjamini–Hochberg corrected at q = 0.05, and is
expressed **in hits rather than in clusters** so that it shares a scale with
`narrow_frac`. A per-cluster percentage would compare 79 clusters against 2,127
on a statistic whose denominator is the cluster count — the same
non-comparability that sank AMI in R-2. This is the statistic AMI cannot provide
at 31:1: it asks whether *any* cluster concentrates the minority class, which is
not swamped by the majority.

**Detection floor, and why it constrains use.** At the global `weak_label == 0`
rate of 872/27,828 = 3.13%, a *fully* confined cluster of 6 gives
p ≈ 9.4 × 10⁻¹⁰ and clears BH at 2,127 tests comfortably; a fully confined
cluster of 4 gives p ≈ 9.6 × 10⁻⁷ and still clears; a cluster of 3 gives
p ≈ 3.1 × 10⁻⁵ against a BH threshold near 2.4 × 10⁻⁵ and is marginal. Note
further that at `min_cluster_size = 4` only a *fully* confined cluster clears —
3-of-4 confined gives p ≈ 1.2 × 10⁻⁴ and fails. So the metric is close to its
floor at our default, and **enrichment must not be compared across
configurations with different `min_cluster_size`.** That belongs in the UI
caption next to the 31:1 note.

**UI copy is part of this deliverable.** `weak_label == 0` means *spatially
confined*, not *verified clean*. AMI and enrichment are proxies for "is this
clustering rediscovering beam multiplicity", not detection metrics, and the rail
must say so next to the numbers.

### 4.2 Run stability

```python
def stability(run_fn, seeds=(0,1,2,3,4)) -> dict
```

`run_fn(seed) -> labels`. Returns **three separate numbers**, never one:

| key | definition |
|---|---|
| `ari_composite` | mean pairwise ARI over full label vectors |
| `ari_restricted` | mean pairwise ARI over points clustered in **both** runs |
| `noise_agreement` | mean agreement on the binary `labels >= 0` vector |

plus `k_mean`, `k_range`, and per-seed `quality()` as mean and range.

**`ari_restricted` is the acceptance statistic.** Reporting only the composite
is the exact error that produced the withdrawn "20×" claim, and the API makes it
impossible to repeat by never returning a single scalar.

**`stability()` runs at either level.** It needs only `run_fn(seed) -> labels`,
so passing family ids instead of cluster ids is a call-site change, not an API
change. Running it on families is the point of §7 — see acceptance criterion 6.

Cost: N seeds is N clustering runs. Cache by `(config, seed)` so the marginal
cost of the N-th is one run; default N=5, configurable, and it reuses the run
already computed for the current configuration.

### 4.3 Epoch trace

```python
def epoch_trace(trace, n_total) -> list[dict]
```

`cluster()` already computes `alive` per epoch; this is bookkeeping. One row per
epoch: alive after, removed, % of original. Renders GLOBULAR Table 1 for our
runs and makes the 87.9%/0/0/0/0 collapse visible on the first look.

---

## 5. Module: `diagnostics.py`

```python
def audit(raw, columns, *, scaling, kinds, min_samples, knn_sample=5000) -> list[dict]
```

Per column:

| key | why |
|---|---|
| `n_distinct` | catches `f02`'s 42 levels — an ordinal handled as a continuum |
| `max_tie_fraction` | the review's original finding |
| `iqr_raw` | what the rail already shows |
| `iqr_scaled` | 1.000 under `robust` by construction; non-trivial otherwise |
| `clip_frac` | **fraction landing on exactly ±5** |
| `share_global` | mean pairwise squared-distance share, random pairs |
| `share_knn` | the same, restricted to k-NN pairs at `min_samples` |

`share_knn` is not decoration. HDBSCAN responds to core distances and mutual
reachability, both local; the global share is a proxy for a local quantity and
the approximation is worst exactly where ties are, because a tied column
contributes zero to every tie–tie pair and tie–tie pairs are disproportionately
likely to be mutual near neighbours. **The gap between the two columns is itself
the tie diagnostic**, so both are reported.

`share_knn` is computed on a `knn_sample`-row subsample; at 5,000 rows a
`NearestNeighbors` query in 15-D is well under a second, and the rail must not
add perceptible latency to dataset load.

**Flagging** is `kind`-aware (§6): tie thresholds apply only to `continuous` and
`ordinal`. A column is flagged when

- `kind` in (`continuous`, `ordinal`) and `max_tie_fraction > 0.1`, or
- `clip_frac > 0.01`, or
- `share_global` above 2× or below ½ the equal share (`1/n_features`).

`share_global` carries the threshold because it is the statistic actually
measured (§1): on `sband_short` the bounds are 13.3% and 3.3% against an equal
share of 6.7%, and they flag `x03` (24.3%, over) and `f02` (1.7%, under) in one
pass. `share_knn` is **reported but not thresholded on first delivery** — no
value for it has been measured yet, and inventing a bound for it now would be
exactly the unverified-claim pattern this review cycle exists to correct.
Calibrate it on the first run, record the numbers in
`aug_2026_workshop/README.md`, and add the threshold in a follow-up.

Flagging both ends is the point: robust scaling misrepresents the distribution
in **both** directions, and fixing only the under-weighted instance leaves the
larger half in place.

**The two share flags are not independent tests, and the rail copy must say so.**
Shares sum to 1 by construction, so a column at 24.3% mechanically depresses
every other column toward the lower bound. `x03` firing "over" and `f02` firing
"under" is one observation about the distribution of shares, not two findings.
The flags remain individually useful for pointing at a column; they must not be
counted as independent evidence.

---

## 6. `features.py`: the `kind` field

```python
@meta_feature("my_col", kind="ordinal", description="...")
```

`kind` ∈ {`continuous`, `ordinal`, `boolean`, `flag`}, **defaulting to
`continuous`** so every existing feature and every feature a workshop
participant has already registered keeps working with no edit. `FeatureSpec`
gains the field; `all_columns()` gains an optional `kind=` filter.

Without this, tie thresholds are unusable: a boolean has
`max_tie_fraction >= 0.5` by definition, so both the review's `> 0.1` flag and
D-5's `> 0.5` assertion misfire on any boolean feature by construction — which
matters immediately, because the deferred `f02` fix (§9) introduces one.

Declared kinds for existing columns: `f02_abs_drift` → `ordinal` (42 levels on a
0.010711 Hz/s lattice on `sband_short`); `x02_time_occupancy`,
`f12_bandwidth_hz` → `ordinal`; `*_saturated` → `flag`; everything else
`continuous`.

Also in this module: document the raw `f06 = (f04² + 1) / f05` identity next to
`f06_bimodality`. It is exact on raw values and does **not** reach the metric
space, because `normalise()` applies `unit` to f04, `log-unit` to f05 and `none`
to f06. That makes it latent, not harmless — a future change to the f05 or f06
transform reactivates it silently, and the comment is the only thing that would
catch it.

---

## 7. Module: `matching.py`

```python
def match(labels, X, *, cut=None, method="ward") -> tuple[np.ndarray, dict]
```

Cluster ids at current settings are **batch artefacts**: dropping a feature at
fixed seed leaves ARI 0.75–0.89, while re-shuffling at identical settings leaves
0.024. The same physical population is minted as a fresh id in every batch and
every epoch. Matching is therefore a correctness fix, not an enhancement.

**Ward linkage on cluster centroids in the scaled feature space**, cut at an
explicit distance, returning a family id per hit. Deterministic: no perplexity,
no seed, no embedding. Measured cost, D=15:

| centroids | time |
|---:|---:|
| 2,000 (Bench, `leaf`) | 0.04 s |
| 20,000 | 6.78 s |
| ~78,000 (full `all`, extrapolated O(n²)) | ~100 s |

scipy's nn-chain is O(n) memory, so one exact implementation serves the Bench
interactively and the CLI offline. A k-NN-graph approximation was measured at
75.8 s for 80,000 — slower than exact Ward and strictly worse — and is not used.

**The cut is derived, not hardcoded.** Default: the q-th percentile of the
nearest-neighbour distance distribution between centroids (q configurable,
default 50). A fixed constant tuned on `sband_short` would be wrong on
`uhf_long`, whose drift lattice alone differs by 5.26×. The Bench exposes the
cut as a slider with a dendrogram beside it, so it is inspected rather than
guessed; `bluse-cluster` takes `--match-cut` and `--match-quantile`.

GLOBULAR's own route — per-hit centroid keying → 10k sample with repetition →
PCA to 6 → t-SNE (perplexity 40, early exaggeration 4) → HDBSCAN on the 2-D
embedding — is implemented behind `method="tsne"` as a reproduction path, not
the default. The paper's own health warnings about t-SNE (§7 of the technical
reference) are the argument for having a deterministic default.

Returns `(family_ids, info)` where `info` carries the cut used, the family count,
and the centroid NN-distance distribution for the dendrogram.

**The first family taxonomy is provisional and must be re-derived after the
contribution-equalising scaling of §9 lands.** Ward runs on centroids in the
scaled space, and that space carries the 14× share spread this spec explicitly
postpones fixing — with `x03_channel_offset` at 24.3% and `f07_kurt_bw_corr` at
13.1%, families will be grouped substantially by channel offset and by a clipped
correlation coefficient. The ordering is still right, because matching is what
makes the scaling work evaluable. But an RFI taxonomy is the headline
deliverable, and without this sentence someone will present the first run of
families as settled.

As a cheap bound on how much the deferred work will move the taxonomy: run
matching once with `x03` and `f07` excluded from the centroid space and compare
family count and median family frequency span against the full-space run. Two
runs of an existing harness.

---

## 8. Wiring

### 8.1 `cluster_selection_method`

New control in both entry points, `eom` default, `leaf` available. One `select`
in `_controls.html`, one argparse flag, one keyword in each `run_hdbscan`.

**The default flip is decided by measurement, not by this spec.** Once matching
lands, run both methods across N seeds and compare `narrow_frac`,
`ari_restricted`, and family count after matching. The addendum's reasoning —
2,127 unmatched discovery-order ids read worse than 79, so shipping `leaf` as
default before matching makes the best recommendation look like a regression —
is sound, and matching is precisely what removes the objection. Record the
outcome in `aug_2026_workshop/README.md` with numbers either way.

### 8.2 Bench

- Rail (`_controls.html`) renders the §5 audit: `n_distinct`, tie fraction,
  clip fraction, and both distance shares, with flagged columns marked. The
  existing raw-IQR bar stays — it answers a different question and its caption
  is already correct.
- Results strip (`_results.html`) gains `narrow_frac`, `largest_pct`,
  `median_size`, the epoch trace, and, once computed, the family count as
  "N raw clusters → M families".
- New `POST /stability` route, run on demand rather than on every cluster — it
  is N× the cost and belongs behind a button. Reuses the current run as one of
  its seeds. **Its seeds are kept out of `HISTORY`** — the cache key includes
  `seed` and `HISTORY` is capped at 12 with `del HISTORY[12:]`, so one N=5 sweep
  would otherwise insert five near-identical entries and evict most of the
  comparison history the user was building. Either exclude the sweep's seeds or
  insert it as one grouped entry showing the three ARI numbers.
- New `GET /values.bin?col=` for colour-by-feature-value on the scatter
  (P1-4), which is what makes the §5 shares visible rather than tabular.
- **D-4:** `load_dataset` fits the `robust`/`quantile` scaler on the full column
  set and applies it to the 35k sample, rather than fitting on the sample. See
  §9 for what this does and does not fix.

Both new endpoints are covered by the existing counted `.busy` indicator; the
stability run is the slowest thing in the tool and must not repeat the
two-phase spinner defect.

### 8.3 CLI

`bluse-cluster` gains `--cluster-selection-method`, `--seeds`, `--match-cut`,
`--match-quantile`, and `--report` (diagnostics table, no clustering). Metrics
land in `<tag>_summary.csv` as new columns plus a `<tag>_metrics.json`;
`family` becomes a column in the per-hit output alongside `cluster`.

### 8.4 Defects

- **D-1** `{{ h.params.eps }}` renders empty — `_results.html:71`. Remove.
- **D-2** "try … a larger epsilon" is stale — `_results.html:63`. Replace with
  the levers that exist (`min_cluster_size`, `min_samples`, feature set,
  `cluster_selection_method`).
- **D-3** `buildGrid()` populates `grid`; `nearest()` never reads it and does a
  full O(n) scan with a `toScreen` per point on every `mousemove`. Wire
  `nearest()` to the grid.

---

## 9. Deferred, with reasons

Recorded here so they are not lost, and because the spec's own metrics are the
precondition for deciding them.

**Contribution-equalising scaling.** The 14× share spread (1.7%–24.3%) is the
general form of the review's Finding 1; `robust` divides by a statistic that
misrepresents the distribution in both directions — inflated to 5.954 by a tie
at the extreme for `f02`, deflated to 0.042 by a tie near the centre for `x03`.
The fix targets the distance share directly rather than a spread proxy. It is
deferred because it must be *evaluated* against `narrow_frac` and
`ari_restricted`, which this spec builds.

**The `f02` ordinal rework.** `|driftRate|` takes 42 values on an exact
0.010711 Hz/s lattice on `sband_short` — the seticore Taylor-tree drift step —
so a rank transform re-spaces physically uniform levels by population. The fix
is an `is_zero_drift` indicator plus non-zero drift on its native linear grid.

Two constraints for whoever writes that spec, both measured here:

1. `driftSteps` exists in the HDF5 and in `catalogues/*_cat.parquet` but **not**
   in the feature parquet, so it needs plumbing through extraction first.
2. **The lattice is per-file.** Six distinct constants across the eight files,
   spanning 5.26× (`uhf_long` 0.00204, `sband_short` 0.01071 Hz/s). `driftSteps`
   is a per-file index, not a physical quantity, so using it directly would make
   `all_features.parquet` compare non-comparable values. Keep physical
   `abs(driftRate)` on a linear scale.

Also deferred: cluster stamp grid (P1-2), pre-filter control (P1-3), redundancy
panel (P1-5), `hdbscan` backend (P2-2), run pinning/diff (P2-3), CLI export
(P2-4), synthetic injections.

**Not doing:** re-adding `cluster_selection_epsilon` to the sklearn path; a
unit-range scaling mode; silhouette; removing the batching loop; making the
embedding an input to the clusterer. All per §8 of the review, uncontested.

D-4 is fixed — the Bench will fit its scaler on the full column set, because it
is correct and cheap — but **not** under the claim that it fixes bench↔CLI
divergence. Measured: a 35k-row IQR matches the 1,281,878-row population to
better than 1.1% on 14 of 15 columns, worst 6.4%. The paths differ because one
clusters 35k rows and the other 1.28M, and because of the seed instability in
§1.

---

## 10. Testing

The repository has no tests and no CI. Three of the defects recorded in
`AGENTS.md` gotcha 9 presented identically — "the bench looks insensitive to
every knob except `min_cluster_size`" — and a small suite would have separated
them.

### 10.1 Two suites, because they answer different questions

**`tests/unit/` — synthetic fixtures, committed, CI-able.** These gate a commit.
A ~500-row feature matrix built by a fixture module with a fixed seed, carrying
a **planted tie**, a **planted clip**, a **known narrow-cluster share** and a
**known family structure**, covers every unit test below and all five
invariants. Commit the generator, not the matrix.

**`tests/workspace/` — golden values against real data**, marked
`@pytest.mark.workspace` and skipped when `paths.workspace()` has no
`features/`. These catch a real regression in the science; they cannot gate a
commit. Each test states which file its number was measured on.

The split is not stylistic. The original draft ran everything against
`mk_sample_hits`, which fails on first execution for three independent reasons,
all measured:

| `mk_sample_hits` | value | consequence |
|---|---:|---|
| zero-drift fraction | **0.0000** | pre-filtered; `f02` tie is 0.4525, not the asserted 0.266 |
| overlap with `lband_long` by `id` | **53.7%** | worst available fixture for anything density-related; `AGENTS.md` already excludes it from pooled statistics |
| `weak_label` counts | **{0: 10206, −1: 4913}** | **no `weak_label == 1` rows at all**, so every AMI and enrichment test is degenerate |

And no real data is committable in any case: `aug_2026_workshop/features/` and
`data/` are both gitignored, 0 files tracked. A suite depending on either runs
on one machine and nowhere else — which, with no CI, means it silently stops
running the first time it breaks.

### 10.2 Invariants

1. Cluster ids are globally unique across batches and epochs.
2. **At a pinned seed**, changing `scaling` changes the label vector
   (ARI < 1.0). Compare `robust` against `none` at seed 0.
3. Reported cluster count equals `len(np.unique(labels[labels >= 0]))`.
4. No column with `kind` in (`continuous`, `ordinal`) has
   `max_tie_fraction > 0.5`.
5. `ari_restricted` for a fixed configuration stays within a recorded band.

**Invariant 2's seed pin is the whole test.** At a free seed, two runs of an
*identical* configuration score ARI 0.024, so `< 1.0` passes on shuffle noise
even if `scale()` were stubbed to return its input — which is the exact bug
class the invariant exists to catch, and what `AGENTS.md` gotcha 9 describes.
At a fixed seed a genuine no-op gives exactly 1.0 and the assertion bites.

**Invariant 5's band is derived from repeated 5-seed draws, not one.** At
`ari_restricted ≈ 0.028` the statistic has meaningful variance of its own; a
band recorded from a single draw is flaky in the way that trains people to
ignore a suite. Record it from repeated draws, label it a smoke test, and keep
it wide.

### 10.3 Unit tests

- `diagnostics.audit` on the synthetic fixture: the planted tie and planted clip
  are recovered at their known values.
- `metrics.quality` on a synthetic labelling with known narrow-cluster share;
  `narrow_enrichment ≈ 1.0` when labels are permuted.
- `metrics.stability` on a deterministic `run_fn` → `ari_restricted == 1.0`,
  `noise_agreement == 1.0`.
- `matching.match` on synthetic centroids with a known family structure.
- `features` registry: a feature registered without `kind` reports `continuous`.

### 10.4 Workspace tests (golden, skipped without a workspace)

Measured on `sband_short` unless stated: `f02_abs_drift_n` → `n_distinct = 42`,
`max_tie_fraction = 0.266`, `iqr_raw = 5.954`; `x03_channel_offset_n` →
`share_global ≈ 0.243`; `f07_kurt_bw_corr_n` → `clip_frac ≈ 0.010`, and
≈ 0.043 on `all`.

### 10.5 Regression test for the withdrawn claim

A labelling with 50% noise and randomised membership must show high
`noise_agreement` and low `ari_restricted`. This is the `leaf` artefact in
miniature, and it fails if anyone collapses the three stability numbers back
into one. Belongs in `tests/unit/` — it needs no real data.

---

## 11. Review-document housekeeping

`bench-review-2026-09.md` and its addendum disagree with each other and with the
response in ~10 places (addendum §9). Rather than rewrite the reviewer's prose,
add a header note to the original pointing at the response and the addendum and
stating that both supersede it where they conflict. Their document, their voice,
and the supersession chain stays legible.

---

## 12. Acceptance

Done when, on `sband_short` at Bench defaults:

1. The rail flags `f02_abs_drift_n` at `max_tie_fraction ≈ 0.266` and
   `n_distinct = 42`, and flags `x03_channel_offset_n` on distance share
   (≈24.3% global, equal share 6.7%).
2. The rail reports `f07_kurt_bw_corr_n` clip fraction ≈1.0% on `sband_short`
   and ≈4.3% on `all`.
3. The results strip reports `narrow_frac ≈ 0.78%` for `eom` and ≈6.8% for
   `leaf`.
4. The epoch trace shows 87.9% removed in epoch 1 and zero in epochs 4–8.
5. `POST /stability` at N=5 reports `ari_composite ≈ 0.024`,
   `ari_restricted ≈ 0.028`, `noise_agreement ≈ 0.999` for `eom`; and
   ≈0.48 / ≈0.032 / ≈0.78 for `leaf`. **`noise_agreement` is degenerate in the
   `eom` arm** — it clusters 99.9% of points, so the statistic has almost no
   variance there. Record it; do not read it as evidence, and gate nothing on
   it.
6. **`ari_restricted` computed on family ids** across N seeds, reported beside
   the cluster-level figure, with family count and its range as secondary.

   A stable family *count* is compatible with scrambled family *membership* —
   40 families every run, different hits in each — so the count alone is the
   easy half. This criterion matters more than a normal one because of what the
   §1 table already shows: `ari_restricted` is 0.0279 for `eom` and 0.0316 for
   `leaf`, meaning **cluster membership is currently not reproducible under
   either selection method**; both sit at the noise floor. Matching is the most
   plausible fix, because the mechanism destroying ARI is that one physical
   population is minted as a fresh id in every batch and every epoch (§7), and
   families are the level at which that should cancel.

   **Record the outcome either way.** Family ARI ≈ 0.05 means matching did not
   solve the problem, and that must be known before it ships. Family ARI ≈ 0.7
   against a cluster-level 0.03 is the strongest single result available from
   this work programme and should be written up as one.
7. The `eom`-vs-`leaf` default decision is recorded in
   `aug_2026_workshop/README.md` with the numbers behind it.
8. The test suite passes and covers all five invariants.
