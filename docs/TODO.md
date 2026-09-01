# Outstanding work — BLUSE

**Updated:** 2026-09-01, immediately after PR #1 merged (`1a1bae0`).
**Purpose:** enough context to resume any item without re-deriving it.

State: on `main`, clean tree, 52 tests passing (48 + 4 skipped outside a
workspace). PR #1 merged the Cluster Bench measurement work — three new
modules (`diagnostics`, `metrics`, `matching`), wiring into both entry points,
and the repository's first test suite.

---

## Reference numbers you will otherwise re-derive

All on `aug_2026_workshop/features/sband_short_features.parquet`, 34,933 rows
reaching the clusterer, Bench defaults (`mcs=4 ms=8 epochs=8 batch=3000
scaling=robust`, f08 off, 15 features), stability over seeds 0–2.
Reproduce everything with `uv run python aug_2026_workshop/acceptance.py`.

| | `eom` | `leaf` |
|---|---:|---:|
| clusters | 72 | 2,162 |
| clustered | 100.0% | 50.3% |
| narrow share <1 MHz | 0.776% | 6.820% |
| narrow share <0.1 MHz | 0.587% | 4.400% |
| cluster membership ARI | 0.0279 | 0.0316 |
| **family membership ARI** (p50 cut) | **0.5190** | 0.1077 |
| family coarsening null | 0.020 | — |
| epochs doing any work | 3 of 8 | 8 of 8 |
| minority-class enrichment | 12.33% | 1.19% |
| AMI | 0.0048 | 0.0026 |

**Matched family count** (this is the honest comparison; the table above is at
mismatched granularity):

| target families | `eom` ARI / narrow % | `leaf` ARI / narrow % |
|---:|---:|---:|
| 20 | 0.5189 / 0.567 | 0.4524 / 0.000 |
| 36 | 0.5190 / 0.670 | 0.4888 / 0.194 |
| 54 | 0.1994 / 0.776 | 0.4475 / 0.194 |
| 72 | 0.0332 / 0.776 | 0.4170 / 0.774 |

**Per-feature distance shares** (equal share 6.7%):

| column | global | k-NN | flags |
|---|---:|---:|---|
| `x03_channel_offset` | 24.3% | 7.4% | tie, share-high |
| `f07_kurt_bw_corr` | 13.1% | 13.0% | clip |
| `f12_bandwidth_hz` | 9.9% | 11.5% | tie |
| `f10_timeseries_std` | 8.8% | 8.4% | — |
| **`f09_temporal_skew`** | **6.7%** | **15.5%** | — |
| `f13_redness` | 6.1% | 10.5% | — |
| `f03_snr` | 6.7% | 4.2% | — |
| `f11_spectrum_std` | 5.3% | 2.6% | — |
| `f06_bimodality` | 4.2% | 4.4% | — |
| `f01_frequency` | 3.2% | 7.2% | share-low |
| `f04_spectral_skew` | 2.8% | 1.9% | share-low |
| `f05_spectral_kurtosis` | 2.6% | 1.8% | share-low |
| `x02_time_occupancy` | 2.5% | 7.2% | tie, share-low |
| `x01_drift_residual` | 2.0% | 3.6% | share-low |
| `f02_abs_drift` | 1.7% | 0.8% | tie, share-low |

`f02` is a **42-level ordinal** on an exact 0.010711 Hz/s lattice (the seticore
Taylor-tree drift step). The lattice constant is **per file** — six distinct
values across the eight files, spanning 5.26× (uhf_long 0.00204, sband_short
0.01071) — so `driftSteps` is a per-file index and is NOT interchangeable with
a physical drift rate.

`weak_label` among labelled rows: 26,956 : 872, i.e. **31:1**.

---

## P0 — do these before quoting numbers anywhere

### 1. Restricted gap rule for the matching cut
*Implementation review §1 option 2. One experiment.*

`derive_cut_pct` is a **granularity dial**, not a structure-sensitive rule:
cutting at the p-th percentile of k−1 merge heights performs the lowest p% of
merges, returning ≈k(1−p/100) groups regardless of structure. Verified on pure
noise. The `gap` rule fails on real data because the final merges of any
dendrogram dominate the gap statistic (it collapsed 2,162 clusters into 4
whole-band families).

**Test:** restrict the gap search to merges below ~the 90th percentile, which
removes the root merges. If it works it responds to structure, which a
granularity dial cannot. Compare family count, family ARI and narrow share
against `pct` at matched counts.

*Where:* `matching.derive_cut_gap` — add a `max_pct=90` bound and a new rule
name. `tests/unit/test_matching.py` already exercises `gap` on the separated
fixture, which is the case it should still pass.

### 2. `f09_temporal_skew` down-weighting — **DONE 2026-09-01**

Result in [`scaling-experiment-2026-09.md`](scaling-experiment-2026-09.md).
Short version: equalisation **helps under `leaf`** (family ARI 0.4888 → 0.6659
at 36 families, median family span 265 → 79 MHz, +95% at 500 families) and is
**catastrophic under `eom`** (0.5190 → 0.0438). `f02` alone accounts for the
whole collapse — boosting only `f02` reproduces it to three figures — because
equalisation amplifies its 26.6% tie 5.2×.

**Consequence: P1 items 4 and 5 are coupled and reordered — `f02` first.**

*Original task, for context:*

`f09` is 6.7% of the global distance share and **15.5% of the k-NN share** —
the largest local contributor — and carries no flag because flags key on the
global statistic. The earlier ablation dropped `x03` and `f07`, which the k-NN
measurement had already shown to be well behaved locally (7.4% against 6.7%
equal), so it removed the wrong column and its result (narrow share
6.820% → 2.969%) is a second demonstration of the k-NN finding, **not**
evidence about equalisation.

**Test:** down-weight `f09` (not drop it) and measure `narrow_frac` and family
ARI. This is the fast read on whether contribution-equalising scaling would
help or hurt, and it decides whether the deferred scaling spec is worth
writing.

### 3. Threshold `share_knn`
Currently reported but deliberately unthresholded — no value had been measured
when the flag rules were written. The baselines are now in the table above and
in `acceptance-2026-09.md`. When thresholds land, **`share_knn` should become
the primary statistic and `share_global` the secondary**, per implementation
review §4. Until then the rail caption warns readers to prefer `knn` where the
two disagree.

*Where:* `diagnostics.audit`, the flag block around the `share-high`/`share-low`
logic; and `bench/templates/_controls.html` for the caption.

---

## P1 — the deferred scaling work, gated on P0-2

### 4. `f02` ordinal rework — **now blocks item 5**
`is_zero_drift` indicator plus non-zero drift on its **native linear grid**.
Two constraints, both measured:
- `driftSteps` exists in the HDF5 and in `catalogues/*_cat.parquet` but **not**
  in the feature parquet, so it needs plumbing through extraction first.
- The lattice is **per file** (above), so keep physical `abs(driftRate)` on a
  linear scale rather than the step index, or `all_features.parquet` compares
  non-comparable values.

Note the boolean introduces a 0.734 tie by construction, which is why the
feature registry's `kind` field exists — tie thresholds skip `boolean`.

### 5. Contribution-equalising scaling mode
Robust scaling equalises the **IQR**, but HDBSCAN responds to **variance**, and
the IQR-to-variance ratio depends on distribution shape — so contributions run
1.7%–24.3% globally. Target the distance share directly rather than a spread
proxy. Evaluate against `narrow_frac` and `ari_restricted`, not by eye.

**Measured 2026-09-01 (P0-2): it works, but only after `f02` is fixed.** Under
`leaf` it buys family ARI 0.4888 → 0.6659 and median span 265 → 79 MHz. Under
`eom` it collapses the run, entirely because it amplifies `f02`'s 26.6% tie
5.2×. Weight caps do not rescue it. Report per method; it is not a universal
improvement.

---

## P2 — Bench features from the original review, not yet built

| item | review ref | note |
|---|---|---|
| Cluster stamp grid | P1-2 | click a row → 20–50 member stamps by distance to centroid. GLOBULAR Fig 9; the paper calls visual spot-checking necessary |
| Pre-filter control | P1-3 | all hits / non-zero drift / Track A survivors / exclude RFI-masked. Select on existing flags — **non-destructive**, do not add a dropping path |
| Feature redundancy panel | P1-5 | correlation + VIF. Acceptance test is **f04/f05** (VIF 53.1, 48.5), not f06 (7.0). Add a nonlinear measure too — f06 is the nonlinear acceptance case |
| `hdbscan` backend | P2-2 | optional extra like `umap`. Ships condensed-tree plots, `cluster_persistence_`, DBCV, `approximate_predict`, and answers the ε_m question |
| Run pinning / diff | P2-3 | `HISTORY` already caches 12 runs by hash; only the comparison view is missing |
| CLI export | P2-4 | copy-pasteable `bluse-cluster …` for the current Bench state |
| Synthetic injections | §6 | drifting narrowband Gaussian + noise, ~30 lines. The only **true** objective function; unblocks recovery rate and GLOBULAR's cluster-seeding trick |

P3 experiments, one look each, record the result either way: densMAP, PaCMAP,
PHATE (the `*_space.png` plots look filamentary — if the space is continuous
rather than blobby that is a finding about whether density clustering is the
right tool), GLOSH outlier ranking (GLOBULAR tried the analogous thing and got
no improvement).

---

## P3 — longer-standing project backlog, predates this work

- **Track E** — weak-supervision classifier. Still queued.
- **Myburgh filter 6** — continuity along the predicted drift trajectory. Not
  implemented.
- **Coherent/incoherent test applies an SNR relation to a power ratio.** An
  unverified substitution, flagged in the code and docs. Worth resolving with
  the BLUSE team.
- **`uhf_long.h5` has ~6,000 corrupt rows** and no `_clean` replacement. Ask.
- **Per-antenna voltages** would enable imaging follow-up. Ask.
- Three unresolved Track A judgement calls: ITU masks, the DTV comb,
  `--tol-steps`. Do not change as a side effect of anything else.

---

## Explicitly not doing

From the original review §8, uncontested: re-adding
`cluster_selection_epsilon` to the sklearn path (its no-op region and its crash
region tile the domain); a unit-range scaling mode; silhouette (assumes convex
clusters — use DBCV if the `hdbscan` backend lands); removing the batching
loop; making the embedding an input to the clusterer; rotation or
frequency-flip augmentations anywhere near this data.

Also: **do not make the repository public** or redistribute catalogue data
without clearing it with the BLUSE team. Tracked survivor CSVs contain sky
coordinates, Gaia/exotica source names and obsids from unpublished data
destined for forthcoming publications. The MIT licence covers code and
documentation only.

---

## Open threads

- **PR #1 has 13 top-level Copilot comments**, all addressed in code but the
  threads are not resolved on GitHub. Nothing outstanding in them.
- **`_enrichment` mixes denominators** — the hypergeometric test runs over
  labelled rows, the returned fraction counts full cluster size. Deliberate and
  now documented; revisit only if it misleads someone.
- **Enrichment must not be compared across different `min_cluster_size`.** At
  the 3.13% global rate, only a *fully* confined cluster of 4 clears BH;
  3-of-4 gives p ≈ 1.2 × 10⁻⁴ against a threshold near 2.4 × 10⁻⁵.
