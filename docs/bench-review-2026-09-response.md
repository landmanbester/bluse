# Response to the Cluster Bench review (2026-09)

**Status:** reply to `docs/bench-review-2026-09.md`. Every number below is
`[measured]` on `aug_2026_workshop/features/sband_short_features.parquet`
(34,933 rows reaching the clusterer) or `all_features.parquet` (1,281,878),
with the Bench's own defaults reproduced in a standalone harness: `mcs=4`,
`ms=8`, `epochs=8`, `batch=3000`, `scaling=robust`, f08 off, 15 features.

The harness reproduces the committed run exactly — k=72, largest cluster 8.6%,
median size 11 — so the measurements below are comparable to
`aug_2026_workshop/clusters/sband_short_summary.csv`.

**Overall:** this is a good review. Finding 1's mechanism is confirmed to three
significant figures from a simulation the reviewer never ran on our data, and
the `leaf` recommendation is the best single suggestion anyone has made about
this tool. Six points are wrong or mis-prioritised, and one omission is serious
enough that it invalidates the review's own primary diagnostic.

---

## Confirmed

### C-1. Finding 1's mechanism, exactly as predicted

| quantity | reviewer | measured |
|---|---:|---:|
| tie value after quantile-normal | −5.199 | **−5.199** |
| raw IQR of `f02_abs_drift_n` | 5.87 | **5.954** |
| slab position after robust scaling | ≈ −0.88 | **−0.884** |
| IQR of the *non-zero* rows after scaling | ≈0.16 | **0.152** |
| slab-to-continuum gap | ≈0.96 | **0.950** |

The reviewer's calibrated simulation was accurate. Note for the record that the
in-house prediction going in was that sklearn's forward/backward interpolation
would map the tie to ≈ −0.96 rather than −5.199; that was wrong and the
measurement settled it.

### C-2. The epoch-1 collapse — upgrade from `[hypothesis]` to `[measured]`

The reviewer tagged §3.1 "hypothesis, but the arithmetic is tight". It is worse
than the arithmetic suggested. Per-epoch survivors, seed 0:

| epoch | alive after | removed | % of original |
|---:|---:|---:|---:|
| 1 | 4,240 | 30,693 | **87.9%** |
| 2 | 59 | 4,181 | 12.0% |
| 3 | 17 | 42 | 0.1% |
| 4–8 | 17 | 0 | **0.0%** |

Epochs 4 through 8 are dead code at current settings. Against GLOBULAR Table 1
(47.6%, then a flat 22–30% per epoch, no plateau at 8) this is a different
regime, confirmed.

### C-3. `cluster_selection_method=leaf` — confirmed, and better than argued

| | `eom` | `leaf` |
|---|---:|---:|
| clusters (mean of 4 seeds) | 79 | 2,127 |
| clustered % | 99.9 | 50.5 |
| largest cluster % | 8.6 | 0.3 |
| median cluster size | 11 | 6 |
| epoch-1 removal | 87.9% | **12.9%** |
| **seed-only pairwise ARI** | **0.024** | **0.483** |
| runtime | ~2 s | ~8 s |

`leaf` restores a working epoch loop, and — the argument the review does not
make — it is **20× more reproducible**. See R-1: that is the strongest case for
it, not the cluster count.

### C-4. Small defects

D-1 (`h.params.eps` renders empty, `_results.html:71`), D-2 ("a larger epsilon",
`:63`), D-3 (`buildGrid()` populates `grid`; `nearest()` never reads it and does
a full O(n) scan) and D-5 (no tests, no CI) are all confirmed as described.

### C-5. `f06 = (f04² + 1) / f05` on raw values

Max absolute deviation across 38,576 rows: **0.0**. Exact. But see R-3 for what
this does and does not imply.

---

## Rebuttals

### R-1. No seed-stability null — this invalidates the review's own primary test

**The review never measures run-to-run variation, and §1.3 step 3 depends on it.**

Four identical configurations differing only in shuffle seed:

```
seed 0: k=72   seed 1: k=80   seed 2: k=66   seed 3: k=99
pairwise ARI: mean 0.0239, range [0.0188, 0.0341]
```

**ARI 0.024 is the noise floor.** Two runs of the *same configuration* agree
essentially not at all.

Now re-read §1.3: "cluster three ways … report pairwise ARI across the three."
Measured:

- baseline vs `f02` turned off (same seed): **ARI 0.892**
- baseline vs non-zero-drift-only rows: **ARI 0.090**

The second is *above the seed-only floor of 0.024*. Removing a quarter of the
data perturbs the labelling **less than re-shuffling does**. So the review's
headline experiment — its number 3 in "if only three things get built" — cannot
distinguish its hypothesis from noise as specified.

The first number is the more interesting one. Dropping `f02` entirely leaves
ARI 0.892, because the seed is held fixed and therefore the *batch composition*
is fixed. Batch membership, not feature geometry, is what determines which
cluster a hit lands in. That reframes the whole review: cluster identities at
current settings are batch artefacts, which is also why cross-batch matching
(their P2-1) deserves promotion rather than P2.

**Recommendation:** their P1-1 becomes P0-1. Nothing else in the document is
interpretable until it exists.

### R-2. The proposed objective function would have rejected the review's own best recommendation

§4 proposes AMI against `weak_label` as *the* objective function and §9 ranks it
first of three. Measured:

| configuration | AMI vs `weak_label` |
|---|---:|
| `eom`, robust (default) | 0.0048 |
| `eom`, scaling none | 0.0017 |
| **`leaf`, robust** | **0.0026** |

Tuning on AMI says `eom` beats `leaf` by ~2×. Reproducibility (C-3) says `leaf`
is 20× better. **The review's P0-1 and its P0-3 point in opposite directions,
and P0-1 loses.**

The cause is a class balance the review never checked. Among rows with
`weak_label != -1`: **26,956 ones to 872 zeros**, 96.9% one class, 31:1. AMI
does correct for chance, as the review says — but against a 31:1 split with 72+
clusters the achievable range is compressed into the third decimal, and the
whole dynamic range observed is 0.0017–0.0048. That is not a tuning signal.

The review's own "done when" ("switching `scaling` from `robust` to `none`
produces a visibly different AMI") is technically satisfied — 0.0048 vs 0.0017
is a 2.8× change — which is exactly why that acceptance criterion is too weak to
have caught the problem.

**Recommendation:** demote to P1; report it alongside minority-class enrichment
(what fraction of clusters are significantly enriched in `weak_label == 0`
against a hypergeometric null), which is not swamped by the majority class; and
label it in the UI as a weak proxy, not an objective.

### R-3. Finding 2 names the wrong column, and the "1.5× weighting" is not supported

The identity is exact on **raw** values (C-5). But HDBSCAN never sees raw
values. `normalise()` applies `unit` to f04, **`log-unit`** to f05, and `none`
to f06 — different transforms, so the algebra does not survive into the space
where Euclidean distance is taken. Measured on the normalised columns:

- `f06_n` from `f04_n, f05_n` alone: **R² = 0.710** (not 1.0)
- `corr(f06_n, f04_n) = +0.825`, `corr(f06_n, f05_n) = +0.776`

Full VIF-style audit, each column predicted from all others:

| column | R² | VIF |
|---|---:|---:|
| **f04_spectral_skew_n** | 0.981 | **53.1** |
| **f05_spectral_kurtosis_n** | 0.979 | **48.5** |
| **f11_spectrum_std_n** | 0.968 | **31.0** |
| f03_snr_n | 0.934 | 15.1 |
| f10_timeseries_std_n | 0.892 | 9.2 |
| *f06_bimodality_n* | *0.857* | *7.0* |
| x01_drift_residual_n | 0.726 | 3.7 |
| … | | |
| x03_channel_offset_n | 0.125 | 1.1 |

**`f06` is the least redundant member of the trio it was accused of
duplicating.** The multicollinearity is concentrated in f04/f05, which are
almost perfectly predictable from the rest of the matrix, and in f11 — which the
review guessed at in a throwaway line ("worth checking … f10/f11") without
following it up.

The redundancy panel (P1-5) is a good idea and we will build it. But its
proposed acceptance test — "`f06` should light up immediately, and it is a good
acceptance test for the panel" — is wrong: a *correct* panel ranks f06 sixth,
and using that criterion we would have concluded a working panel was broken.
The acceptance test should be f04/f05.

### R-4. D-4 is real, mis-diagnosed, and oversold

The two paths do fit scalers on different row sets, as described. The claimed
consequence does not follow. Comparing a 35k-sample IQR against the full
1,281,878-row population IQR, per column, three draws:

- 14 of 15 columns agree to **better than 1.1%**
- worst column (`x02_time_occupancy_n`) **6.4%**

A median and an IQR from n=35,000 are precise estimates; that is what sampling
does. So the scaler fit is **not** why a Bench configuration fails to reproduce
in `bluse-cluster`. It fails to reproduce because the Bench clusters 35k rows
and the CLI clusters 1.28M, and because of R-1.

"Undercuts the tool's purpose" is the right conclusion attached to the wrong
cause. We will fit on the full column set anyway — it is cheap and principled —
but fixing it will not make the two paths agree, and shipping it as though it
would is how the real cause stays hidden.

### R-5. §1.4's fix ordering is inverted, and fix 1 reintroduces the pathology

Two problems.

**(a) The review found one instance of a general failure and fixed the
instance.** Robust scaling equalises the IQR, but HDBSCAN's distance responds to
*variance*, and IQR-to-variance depends on distribution shape. Mean pairwise
squared-distance contribution per column, post-scaling, 4,000 random pairs:

| column | share |
|---|---:|
| **x03_channel_offset_n** | **24.3%** |
| f07_kurt_bw_corr_n | 13.1% |
| f12_bandwidth_hz_n | 9.9% |
| f10_timeseries_std_n | 8.8% |
| *(equal share would be 6.7%)* | |
| … | |
| **f02_abs_drift_n** | **1.7%** |

The spread is **14×**, from 1.7% to 24.3%. `f02` is the *least*-weighted column
and the review is right about that — but `x03_channel_offset_n` is
over-weighted by 3.6× and is never mentioned, despite carrying a 26.6% tie of
its own and only 83 distinct values. Fixing `f02` alone leaves the larger half
of the problem in place. The review's fix 3 (a tie-aware spread statistic),
ranked last, is the one that addresses the general case.

**(b) Fix 1 introduces a maximal tie while removing one.** Splitting `f02` into
a boolean `f02_is_zero_drift` creates a two-valued column with a 26.6/73.4
split — a tie fraction of 0.734, far worse than the 0.266 being removed, and it
would trip the review's own proposed `max_tie_fraction > 0.1` flag and D-5's
proposed `> 0.5` assertion. Under any contribution-equalising scaling that
boolean carries weight comparable to a full continuous feature. This may still
be the right call — a zero-drift hit *is* categorically different — but it is a
weighting decision presented as a cleanup, and it needs the scaling question
settled first.

### R-6. Two factual corrections, and the thing underneath Finding 1

- The zero-drift fraction is **33.5% of the file** but **26.6% of the rows that
  reach the clusterer**, after `feature_ok` and the finite-row filter. §1.1
  quotes 33.5% throughout as though it were the clustered fraction.
- Severity is **file-dependent** and §1 presents it as universal. On `all`, the
  `f02_abs_drift_n` IQR is **1.239**, not 5.954, and the tie is 0.201. Same
  pathology, roughly a fifth of the magnitude.

More importantly, the tie is a symptom of something the review does not
mention. `driftRate` on `sband_short` takes **83 distinct values**; `|driftRate|`
takes **42**; the spacing is a constant **0.010711 Hz/s** — the seticore
Taylor-tree drift step. So `f02_abs_drift_n` is not a continuous feature with an
unfortunate spike at zero. **It is a 42-level ordinal**, and the whole column
has 42 distinct values after transformation. Applying a quantile-normal
transform to a 42-level ordinal is questionable independently of the zero tie,
and any fix that treats only the zero bin leaves a 41-level ordinal being
handled as a continuum.

For comparison, `f08_turning_bw_hz_n` carries a **69.1%** tie at 1.000 — far
worse than `f02` — which is exactly why it is already off by default. That
default was set for a documented reason and the review's tie audit would have
rediscovered it; worth noting that the mechanism it proposes has already earned
its keep once.

---

## Revised plan

Re-prioritised against the measurements above. Items keep their original IDs.

### P0 — nothing is interpretable without these

1. **P1-1 → P0. Seed-stability: N seeds, pairwise ARI.** Promoted from P1 per
   R-1. Precondition for every other comparison in the review. Report mean and
   range of pairwise ARI plus cluster-count spread. Acceptance: reproduces
   ARI ≈0.02 for `eom` and ≈0.48 for `leaf`.
2. **P0-3. `cluster_selection_method` control, `eom` default, `leaf`
   available.** Unchanged, confirmed (C-3). Both `bench/app.py` and
   `track_b_cluster.py`, plus the argparse flag.
3. **P0-2. Per-feature diagnostics, extended.** `n_distinct`, `max_tie_fraction`
   and — added per R-5(a) — **mean pairwise squared-distance share**, which is
   what actually drives HDBSCAN and what the raw-IQR bars do not show. Flag
   `max_tie_fraction > 0.1` and any share above 2× or below ½ the equal share.
4. **P0-4. Per-epoch reduction table.** Unchanged. Confirmed to show something
   (C-2): it makes epochs 4–8 visibly dead.

### P1

5. **P0-1 → P1. Objective-function stats, rescoped per R-2.** Largest-cluster
   fraction and median cluster size are unambiguously good and go in now. AMI
   ships alongside minority-class enrichment and a UI caption stating the 31:1
   imbalance. Not billed as an objective function.
6. **P2-1 → P1. Cross-batch cluster matching.** Promoted per R-1: cluster ids
   are currently batch artefacts, which makes this a correctness issue rather
   than an enhancement. Build the deterministic Ward-linkage route as default,
   GLOBULAR's t-SNE route as the reproduction path.
7. **P1-3. Pre-filter control.** Non-destructive, selecting on existing flags.
8. **P1-5. Feature redundancy panel.** Acceptance test changed to f04/f05 per
   R-3.
9. **P1-4. Colour-by-feature-value.** Cheap, and makes R-5(a) visible.
10. **P1-2. Cluster stamp grid.**

### P2

11. **Scaling work per R-5.** A contribution-equalising mode, evaluated against
    the P0-1 seed-stability metric rather than by eye. The `f02` split (§1.4
    fix 1) is decided *after* this, not before, per R-5(b).
12. **P2-2 `hdbscan` backend**, **P2-3 run pinning/diff**, **P2-4 CLI export**.
13. **D-1, D-2, D-3** — cosmetic, batched into whichever change touches those
    files first.
14. **D-5 regression tests**, with the review's four assertions plus a fifth:
    seed-only ARI for a fixed config stays within a recorded band.

### Not doing

- **D-4 as specified.** We will fit the scaler on the full column set because it
  is correct, but not under the claim that it fixes bench↔CLI divergence (R-4).
- **`f06` treated as the redundancy exemplar** (R-3).
- Everything in the review's §8 stays out of scope; no disagreement there.
