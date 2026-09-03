# BLUSE workshop — August 2026

Finding interesting things in ~2M BLUSE narrowband hits.

**This directory is the working record — results, numbers, and what to
distrust.** For installing the software and an overview of what the pipeline
does, start at the [top-level README](../README.md). The code now lives in
`src/bluse/` and is installed as commands (`bluse-track-a`, `bluse-features`,
`bluse-cluster`, `bluse-bench`, `bluse-explore`); this directory is the
*workspace* those commands read and write.

| Path | What it is |
|---|---|
| `brainstorming.md` | Technique survey from the literature + proposed tracks A–E |
| `data/` | The HDF5 stamp files (21 GB) — see "Which data files to use" |
| `catalogues/` | Track A output: filtered catalogues + cut-flows |
| `features/` | Track B feature matrices |
| `clusters/` | Track B cluster tables and plots |
| `plots/` | PNGs from `bluse-explore` |
| `masks/` | Empirically derived RFI masks |

Run the commands from this directory (or any directory under it) and the
workspace resolves here automatically.

## Which data files to use

**All seven, as delivered.** The corrupt stamp cubes were repaired by the BLUSE
team on 2026-09-02; there is no longer a `*_clean.h5` file and one should not be
re-introduced.

| file | use it? | why |
|---|---|---|
| `lband_short.h5` | **yes** | 866,002 rows, 0 bad blocks, 100% usable |
| `uhf_long.h5` | **yes** | 299,878 rows, 0 bad blocks, 100% usable |
| the other five | yes | never had corruption |

Verify any future delivery the same way — `bluse-explore info` names any file
with bad blocks, and extraction reports usable rows per file.

### What the repair changed

Two of the original files carried a contiguous block of corrupt stamp cubes
raising `OSError: wrong B-tree signature` — a bad transfer, not truncation:
`lband_short.h5` rows 338,000–742,000 and `uhf_long.h5` rows 264,000–270,000.
Only stamps were ever affected; every metadata column always read fine, which
is why the Track A catalogues were always trustworthy. An interim
`lband_short_clean.h5` with the 402,377 unreadable rows stripped stood in for
the original until the repair.

| | before | after |
|---|---:|---:|
| usable rows, whole survey | 1,605,678 | **2,014,055** (+25.4%) |
| `lband_short` | 463,625 (as `_clean`) | **866,002** (+86.8%) |
| `uhf_long` | 293,878 | **299,878** (+6,000) |
| rows failing `feature_ok` | 6,000 | **0** |

The new total is exactly the row count of the team's own `filtered_hits.csv`
(2,014,055), which is a useful independent check that nothing is missing.

**The retired clean file was a biased subset of its band.** Its zero-drift
fraction was 26.75% against the full file's 46.60%, because the stripped region
was disproportionately zero-drift. Any per-band result computed on
`lband_short_clean` is drawn from a skewed sample. The Track A results table
below has already been re-derived on the repaired file.

**Don't let a stale copy into a run.** `bluse-features` with no arguments globs
`data/*.h5`, and `--combine-only` globs `features/*_globular_features.parquet`,
so a leftover `lband_short_clean_globular_features.parquet` is still picked up.
`drop_superseded()` keeps whichever of the pair carries more *usable* rows and
prints what it dropped — it used to prefer `_clean` by name, which after the
repair would have discarded 402,377 good rows. Delete superseded per-file
parquets anyway rather than relying on the guard.

To re-run the pipeline after a re-delivery:

```bash
bluse-track-a data/lband_short.h5 data/uhf_long.h5   # catalogues first
bluse-features data/lband_short.h5 data/uhf_long.h5  # then features
bluse-features --combine-only                        # rebuild the combined table
```

That last step matters: `bluse-features <files>` rebuilds
`all_globular_features.parquet` from **only the files it just extracted**, so
without it the combined table silently shrinks to those files alone.

## Setup

See the [top-level README](../README.md#install). In short, from the repository
root:

```bash
uv sync --extra all
source .venv/bin/activate
cd aug_2026_workshop
```

## Track A: the classical baseline

```bash
# everything, default parameters, ~20 s for all 2M hits
bluse-track-a

# one file, look at more survivors
bluse-track-a data/sband_short.h5 --show 40

# documented SARAO masks only -- drop our ITU-allocation guesses
bluse-track-a data/sband_short.h5 --no-itu

# derive an RFI mask from the data instead of trusting published tables
bluse-track-a data/uhf_short.h5 --derive-mask masks/uhf.csv

# when the real incoherent-beam powers turn up
bluse-track-a data/sband_short.h5 --incoherent-power incoh.csv
```

### The cuts

Applied in order, following Tremblay et al. 2026 (K2-18b, VLA + MeerKAT):

1. **Known-RFI frequency masks** [§3.1] — `rfi_masks.py`
2. **Zero drift rate** [§3.2] — local RFI
3. **Maximum drift rate** [§3.2] — faster than a bound companion plausibly
   drifts. **Off by default**; `--max-drift-coeff 4.18e-4` enables it
4. **SNR window** [§3.3] — below is mostly false positives, above is instrumental
5. **Multi-beam coincidence** [§3.4] — ±1 *fine channel*, ±1 drift step, per observation
6. **Coherent/incoherent ratio** [§3.7] — `SNR_coh ≤ √N·SNR_incoh` *(wired up, inert until data arrives)*
7. **Cross-epoch persistence** [§3.6] — same frequency **and drift** across many observations

Their §3.5 transit filter has no analogue here: it needs a transit-timed planet.
Full spec, our divergences and the paper's own internal inconsistencies are in
[`papers/Tremblay-technical-reference.md`](../papers/Tremblay-technical-reference.md).

**Nothing is deleted.** Each cut writes a boolean `flag_*` column; `pass_all` is
the AND of their negations. Change your mind about any cut by re-filtering the
parquet — no need to re-run.

### Results with default parameters

| File | Hits | Survivors | % | was |
|---|---:|---:|---:|---:|
| `lband_long` | 557,690 | 568 | 0.102 | 853 |
| `lband_short` | 866,002 | 798 | 0.092 | 740 |
| `uhf_long` | 299,878 | 2,281 | 0.761 | 2,406 |
| `uhf_short` | 208,774 | 786 | 0.376 | 1,870 |
| `sband_long` | 36,132 | 43 | 0.119 | 47 |
| `sband_short` | 38,576 | 43 | 0.111 | 46 |
| `mk_sample_hits` | 15,119 | 894 | 5.913 | 906 |
| **total** | **2,022,171** | **5,413** | **0.268** | **5,355** |

**`mk_sample_hits`' 894 survivors are almost entirely spurious, and the honest
survey total is 4,565.** That file is sampled at ~1% of the others' hit
density, and beam multiplicity is counted *within* a file, so its coincidence
counts are meaningless. 8,116 of its hits also appear in `lband_long` under the
same `id`, frequency and beam: the same hit is counted in a mean of **1.87
beams** there against **29.71** in `lband_long`. Of its 894 survivors, 848 are
such duplicates — and only **15** of those survive Track A when the surrounding
hits are present. `all_globular_features.parquet` deduplicates on `id`, so the combined
table carries 46 mk survivors and 4,565 in total. See
[`../docs/track-e-2026-09.md`](../docs/track-e-2026-09.md) §7.

Re-run on the repaired 2026-09-02 delivery. Only the `lband_short` row moved:
`uhf_long` returned 2,281 survivors both before and after, which confirms the
repair left metadata untouched (Track A is metadata-only). The `was` column is
the pre-repair set, where `lband_short_clean` stood in for `lband_short`.

**`lband_short` gained 58 survivors but its survival *rate* halved** (0.160% →
0.092%): the 402,377 recovered rows are dominated by hits the chain removes —
91.3% fall in the known-RFI mask against 89.9% before, and the zero-drift cut
flags 46.6% against 26.8%. The clean subset was not representative of its band.

### What changed in 2026-09, and why

Re-reading Tremblay et al. 2026 against our code found three divergences. All
three are now fixed; together they cost 19.3% of the survivors. See
[`papers/Tremblay-technical-reference.md`](../papers/Tremblay-technical-reference.md)
§6 for the full map.

**1. The multi-beam tolerance was ±1 Hz; it should be ±1 fine channel.** This is
the big one — it moves every file. §3.4 of the paper specifies one fine channel;
the "1 Hz" they quote is a consequence of *their* 1 Hz channels. Ours are 1.013
(UHF), 1.594 (L) and 1.630 (S) Hz, so in L and S band we were matching about 37%
too tightly, under-counting how many beams carried each hit and letting
multi-beam RFI through as survivors.

**2. There was no maximum drift-rate cut.** We implemented §3.2's zero-drift
rule but never its upper bound. It is not inert: it flags 14,306 hits in
`lband_short` (1.65%) and 1,134 in `uhf_short` (0.54%), and nothing anywhere
else. (Re-derived on the repaired file; it was 12,756 / 2.75% on the retired
`lband_short_clean` subset.) The limit scales with observing frequency at 4.18e-4 Hz/s per
MHz, which reproduces all three anchor values the paper quotes.

*Caveat, recorded in the code:* that coefficient bounds Earth's rotation
(~1.1e-4 Hz/s per MHz, universal) **plus K2-18b's own orbital acceleration**.
Our targets are arbitrary Gaia sources, so treat it as a generous envelope
rather than a per-target limit. This caveat is why the cut ended up **off by
default** a day later — see the Myburgh section below. `--max-drift-coeff
4.18e-4` switches it on.

**3. Cross-epoch persistence ignored drift rate.** §3.6 requires a signal to
recur at the same frequency *and* the same drift to count as interference. Ours
binned on frequency alone. The fix makes the cut strictly more conservative —
on `uhf_long` it now removes 8 hits where it removed 136, and on `uhf_short` 6
where it removed 121. `n_obs_at_freq` is unchanged and still frequency-only,
because Track B's provenance columns are built on it; the drift-aware count is
the new `n_obs_at_freq_drift`.

**A fourth divergence is deliberate and stays.** Their SNR ceiling is 100; ours
defaults to 1e6, because our data has a detached population at 1e7–1e8 that we
want to see rather than silently drop. `--snr-max 100` gives the literal recipe,
and `flag_snr_high` is recorded either way.

### Then Myburgh et al. 2026 changed two of those decisions

Reading the VLA high-frequency paper
([`papers/Myburgh-overview.md`](../papers/Myburgh-overview.md)) — same group,
but **blind Gaia targets like ours rather than one known planet** — moved two
more things.

**The maximum drift cut is now OFF by default.** Myburgh et al. search
±50 Hz/s deliberately, "as many of our targets are toward unknown planetary
systems". So are ours. Worse, our K2-18-derived limit was biting *inside* the
range `seticore` actually searched: on `lband_short` it lands at 0.358–0.402
Hz/s against an observed maximum of 0.4203 — exactly where a fast-drifting real
signal would sit. In a blind survey a false negative costs more than one more
waterfall to inspect. `--max-drift-coeff 4.18e-4` restores it, and on the
repaired file it costs 70 survivors (798 → 728).

**A duration-conditioned SNR floor is now ON.** Their filter 3 requires SNR > 15
for hits with fewer than 16 time samples, because `seticore`'s noise estimate
needs samples to average. Three papers say this independently — Tremblay §3.3
measured ~80% of 8σ detections with few samples to be false positives, and Czech
et al. §6 call beams under 150 s unviable. **`uhf_short` is 14–15 timesteps for
every one of its 208,774 hits**, and 456 of its 1,193 survivors had SNR ≤ 15.
No other file is affected. (`uhf_short` 1,193 → 786.) `--snr-min-short 0`
disables it.

**The clean file's survivors are not a subset of the original's.** Comparing by
hit `id` under the corrected parameters: 662 survive in both, 66 survive only in
the original — every one of those 66 is a row the clean file deletes, so they
never had a readable stamp and could not have reached Track B — and **8 survive
only in the clean file**. Those 8 are new.
The multi-beam coincidence cut counts how many beams a hit appears in, so
deleting 46% of the rows changes the denominator and a handful of hits that
previously looked like multi-beam RFI now look confined. Expect small
population-dependent shifts like this whenever the input row set changes; it is
the cut working as intended, not an inconsistency.

Read the per-file `_cutflow.csv` before trusting any of it. The `flagged_alone`
column shows what each cut catches on its own; `removed_here` shows what it adds
given the cuts before it. Large gaps between the two mean redundancy.

## Four things to decide before believing the numbers

**1. The ITU masks are ours, not SARAO's.** SARAO's published table barely
touches S band: on `sband_short` it flags **3 hits out of 38,576**. Our added
ITU allocations flag 33,449 — including 2200–2290 MHz (space-operations
downlink), which is where the strong 64-beam 2242.5 MHz signal lives. Survivors
go 533 → 46 depending on this one choice. Run `--no-itu` and compare before
committing.

**2. The digital-TV comb is off by default.** SARAO gives a formula for 8 MHz
DTV channels; channels 21–68 tile contiguously over 466–858 MHz, so enabling
the full comb masks *all* of `uhf_short` (208,774 of 208,774 hits). The formula
says where TV channels may sit, not which are transmitting near the Karoo. Use
`--derive-mask` instead. `--dtv` enables it if you narrow the channel range.

**3. `--tol-steps` leaks.** The strict ±1 drift-step match lets a strong emitter
through when its fitted drift varies between beams. The top `lband_short`
survivors are all 870.2323 MHz with drifts of −0.277, −0.318, −0.338 Hz/s —
plainly one transmitter, split into "distinct" signals by the tolerance.
`--tol-steps 999` matches on frequency alone: survivors 1,015 → 832.

**4. `mk_sample_hits.h5` is pre-filtered and half-duplicated** — 0% zero-drift
versus 22–47% elsewhere (confirmed by the BLUSE team), and 8,116 of its 15,119
rows are byte-identical duplicates of rows in `lband_long.h5`. It is a curated
sample carved out of that file, not an independent one. Its 5.99% survival rate
is not comparable to the others. Track B deduplicates the combined feature table
on `id`; anything else that pools files must do the same.

## Still open

- **`incoherentPower` was never measured for this data** (confirmed by the BLUSE
  team). Cut 5 is implemented and tested but permanently inert here. The
  strongest classical discriminant is simply not available, which puts the whole
  weight on multi-beam coincidence.
- **The `_short` files are ~118 s.** Czech et al. 2026 §6 describes beams shorter
  than 150 s as not "viable for technosignature searching" when triaging the
  survey. Ours are a deliberate short-integration subset; report them separately
  from the `_long` files rather than pooling.
- **`uhf_long.h5` still has no clean replacement.** Rows 264,000–270,000 (2.00%)
  remain unreadable, so ~6,000 stamps are still lost there. Worth asking for a
  re-copy of that one too, on the same terms as `lband_short.h5`. **Answered
  2026-09-02** — both files were re-delivered repaired.
- **Hits-per-beam steps at beams ~49/~55 — explained, benign.** One beam per
  target, filled from 0, and sparse sky has fewer targets. `bluse-explore
  beams <file>` shows it. Catalogues now carry `n_beams_formed` and `beam_frac`
  because the coincidence denominator varies per observation.

## Cluster Bench — interactive hyperparameter explorer

> Vocabulary — *cluster*, *family*, *residue*, *candidate* — is defined in
> [`../NOMENCLATURE.md`](../NOMENCLATURE.md).

```bash
bluse-bench                       # then open http://127.0.0.1:8000
bluse-bench --port 8080 --host 0.0.0.0   # share on the LAN
```

FastAPI + htmx + a canvas scatter. Pick a file, toggle features, adjust
HDBSCAN, press **Cluster**. A re-cluster on a 35k sample takes ~1.2 s.

Changing the file, sample or seed loads the new dataset immediately — the
selector never shows a file that is not the one on screen. (It used to: the
select held a pending choice while the cluster form kept the previously loaded
key, so switching file and pressing Cluster re-ran the *old* dataset, hit the
run cache and returned identical numbers. It looked like the app had frozen.)

**The feature rail.** Each feature shows its interquartile range as a bar —
the *raw* spread, before scaling. On `sband_short`, `f02_abs_drift` measures
5.954 against `f03_snr` at 0.092, a 65× imbalance. Since HDBSCAN's Euclidean
metric is a sum of those spreads, under `scaling: none` the widest bar simply
is the clustering. `robust` divides them out so every feature counts equally,
which is why it is the default.

(The bars used to be captioned "after scaling", which was false in the worst
way: robust scaling divides each column *by* its IQR, so every scaled IQR is
exactly 1.000 and the bars would all be equal. They show what the scaling
control exists to fix, not what HDBSCAN sees.)

Also exposed, in rough order of how much they matter: **scaling** (robust /
quantile / GLOBULAR-literal), **mode** (epochs / single), then `batch`,
`min_samples`, `min_cluster_size`, `epochs`, `seed`. There is no `epsilon`
control — see gotcha 9 in `AGENTS.md` for why it cannot work.

**Reading the colours.** A hue means *size rank*, not identity: the 12 biggest
clusters in a run get their own colour, everything smaller is one grey, noise is
darker still. So a colour answers "Nth biggest in this run" and does not carry
over between runs. The cluster *number* is a discovery-order serial, **not** a
batch number — HDBSCAN restarts its ids at 0 in every batch and we add a running
offset to keep them distinct, so one batch mints many consecutive ids. The `ep`
and `batch` columns give the real provenance. The legend above the table says
all of this in place.

While a run is in flight the time reading is replaced by a spinner, the stats
strip and the plot dim, and `working` appears in the header — the panel
otherwise keeps showing the previous numbers, which reads as nothing having
happened.

**The indicator covers both halves of a re-cluster, which is the whole point.**
A run is a server phase (POST `/cluster`) and then a client phase (refetch the
embedding, reload the labels, redraw). The server phase is the fast one —
measured at 0.6 ms on a cached run — while changing the feature set invalidates
the embedding and forces a fresh projection: 11 s for UMAP on a 35k sample of
`all`, up to a minute cold. htmx's `hx-indicator` only knows about the request,
so the spinner used to stop after ~0.8 s and the plot would then change on its
own many seconds later. `scatter.js` now holds a counted `busy` class across
both phases, and the HUD names the slow step (`projecting (umap)…`) rather than
leaving a silent gap.

Note that the `time` stat is the *clustering* time and nothing else. A run
reporting `0.69s` can still take half a minute of wall clock if it had to
re-project — that is the projection, not the clusterer.

How it works: the embedding projects exactly the matrix HDBSCAN sees, so it is
cached per (method, scaling, feature set) and refetched only when one of those
changes. A re-cluster at the same geometry ships an Int32Array of labels and
recolours in place — points crossfade over 260 ms so you can see which ones
changed cluster.
Every run is cached by parameter hash, so the run history is free and revisiting
a configuration is instant. Click a point for its waterfall; click a table row
to zoom to that cluster.

UMAP gives better visual separation than PCA but takes ~60 s on 35k and blocks
the request while it runs. It is cached per dataset, so you pay once.

## Measuring a clustering, not eyeballing it

Added 2026-09 after an external review of Cluster Bench. The review's one
structural point was that the Bench exposed every knob and explained each one,
but nothing on screen said whether a configuration was *better* — so tuning was
by eye, on a scatter plot whose geometry these notes already warn against
over-reading. Three modules fix that: `bluse.diagnostics` (what each feature
contributes), `bluse.metrics` (is this clustering any good, and is it
reproducible) and `bluse.matching` (do cluster ids mean anything across
batches).

Full numbers in `clusters/acceptance-2026-09.md`. The headlines:

### Cluster ids were batch artefacts, and matching fixes it

Two runs of an *identical* configuration, differing only in shuffle seed, agree
at **ARI 0.028**. That is the noise floor — HDBSCAN mints local ids `0..k-1` in
every batch, and the epoch loop runs many batches, so one physical population
becomes a fresh id in every batch it appears in.

Grouping clusters into families by Ward linkage on their centroids recovers it:

| | cluster ARI | family ARI |
|---|---:|---:|
| `eom` | 0.0279 | **0.5190** |
| `leaf` | 0.0316 | 0.1077 |

An 18.6× gain for `eom`, with the narrow-cluster share essentially intact.
`bluse-cluster --match`, or the Families control in the Bench. It survives its
own null: family ARI 0.519 against 0.020 when the cluster&rarr;family map is
permuted at fixed family sizes.

**Matching is doing the work, not the selection method.** Unmatched `eom` at 72
clusters scores ARI 0.033; `leaf` matched down to 72 families scores 0.417 at
the same narrow share. Every committed result in this workspace predates
matching and is therefore in the 0.03 regime.

### `eom` vs `leaf` is a user choice

They win on different axes, so the tool asks instead of deciding:

| | `eom` | `leaf` |
|---|---:|---:|
| clusters | 72 | 2,162 |
| narrow share <1 MHz | 0.776% | **6.820%** |
| family membership ARI | **0.5190** | 0.1077 |
| epochs doing any work | 3 of 8 | **8 of 8** |

**Both of those numbers are granularity effects, and they point opposite
ways.** The default cut returns about *k*/2 groups, so that table compares
`eom` at 36 families against `leaf` at 1,081. Matched at equal family count
(3 seeds):

| target families | `eom` ARI / narrow % | `leaf` ARI / narrow % |
|---:|---:|---:|
| 20 | 0.5189 / 0.567 | 0.4524 / 0.000 |
| 36 | 0.5190 / 0.670 | 0.4888 / 0.194 |
| 72 | 0.0332 / 0.776 | 0.4170 / 0.774 |

The reproducibility gap shrinks to 6%, and `leaf`'s coherence advantage
*inverts* — coarsened to 36 families its narrow share falls to 0.194%, below
`eom`'s 0.670%. `leaf`'s coherence is a property of its fine granularity, not
of the extraction method.

So the choice is really about granularity: ~2,000 small frequency-coherent
groups that individually do not reproduce (`leaf`), or ~36 large groups that
reproduce well and stay reasonably narrow (`eom` + matching). `eom` stays the
default, now on measurement rather than precedent — at every matched count from
20 to 36 it is at least as reproducible and more coherent.

**The epoch budget was being spent in one pass.** Under `eom`, epoch 1 removes
87.9%, epoch 2 removes 12.0%, epoch 3 removes 0.1%, and epochs 4–8 remove
*nothing at all*. GLOBULAR reached 47.6% in epoch 1 then a flat 22–30% per
epoch with no plateau at 8. The Bench now shows this as a table with dead
epochs dimmed.

### The objective function

`narrow_frac` — the fraction of clustered hits in clusters spanning under
1 MHz — is the headline. It needs no labels, has real dynamic range (0.776% to
6.820%), and rewards the physical coherence an RFI taxonomy actually requires.
It is reported at two thresholds so it cannot be an artefact of where the line
was drawn (the ratio holds: 8.8× at 1 MHz, 7.5× at 0.1 MHz) and against a
size-preserving permutation null (`leaf`'s 6.820% is 341× its null).

AMI against `weak_label` is **not** the objective function, and shipping it as
one would have been a mistake: among labelled rows the class balance is 31:1,
its whole observed range is 0.0017–0.0048, and it ranks `eom` above `leaf` —
the opposite of every other signal. It ships captioned. Hypergeometric
minority-class enrichment replaced it and discriminates 10× (12.33% vs 1.19%)
where AMI manages 1.8×.

### What each feature actually contributes

`bluse-cluster --file sband_short --report`, or the Bench rail. Robust scaling
equalises the *interquartile range*, but HDBSCAN responds to *variance*, and
the ratio between them depends on distribution shape — so contributions to the
distance are not equal at all:

| column | global share | k-NN share | flags |
|---|---:|---:|---|
| `x03_channel_offset` | 24.3% | 5.9% | tie, share-disagree |
| `f07_kurt_bw_corr` | 13.1% | 12.8% | clip |
| `f09_temporal_skew` | 6.7% | **15.3%** | **share-high** |
| `f02_abs_drift` | 1.7% | 0.4% | tie, share-low, share-disagree |

Equal share is 6.7%. **The two columns disagree, and that matters.** HDBSCAN
keys on local density, so the k-NN column is the relevant one — and by it
`x03` is unremarkable while `f09_temporal_skew` is the largest contributor.
The earlier reading that `x03` dominates came from the global share and does
not survive the local one.

`f02_abs_drift` is a **42-level ordinal**, not a continuous feature:
`|driftRate|` sits on an exact 0.010711 Hz/s lattice, the seticore Taylor-tree
drift step. The lattice constant is per-file — six distinct values across the
eight files, spanning 5.26× — so `driftSteps` is a per-file index and is *not*
interchangeable with a physical drift rate. Its 26.6% tie at −5.199 halves its
local share, exactly the tie–tie mechanism the audit predicts.

`share_knn` **now carries the flag threshold** — `share-high` above 2× an
equal share, `share-low` below ½×. It shipped unthresholded because no value
for it had been measured when the flag rules were written, and inventing a
bound then would have been the unverified-claim pattern this review cycle
exists to correct. It was calibrated in P0-3 across all seven per-file
parquets, where those bounds mark a mean of 4.9 columns of 15 per file against
7.4 for the same rule on the global share.

`share_global` is now secondary and earns its place where it **disagrees**: the
`share-disagree` flag marks a column whose two shares differ by ≥2.5× either
way, i.e. one where the global number misleads. On sband_short that is `x03`
(24.3% global, 5.9% local — so it is *not* flagged over-weighted any more) and
`f02` (9.8× on uhf_short). See
[`../docs/share-knn-threshold-2026-09.md`](../docs/share-knn-threshold-2026-09.md).

### `robust-equalised`: what it does and does not do

`--scaling robust-equalised`, or the Bench's *robust + equalised* option,
divides each column by its post-clip standard deviation. That is **winsorised
standardisation** — winsorise at ±5 IQR units, then z-score — and it equalises
each column's share of the *global* distance exactly.

It does **not** equalise the k-NN share, which P0-3 established is the
statistic HDBSCAN actually responds to. The strategy that targets that one was
measured and rejected: its target is unattainable for a tied column, so it
drives `f02` to the weight ceiling forever, and it collapses the metric to ~2
effective dimensions. See
[`../docs/equalising-scaling-experiment-2026-09.md`](../docs/equalising-scaling-experiment-2026-09.md).

**Use it with `leaf` only.** Under `eom` equalisation collapses family
reproducibility (family ARI 0.519 → 0.044); both entry points warn and still
run, so the measurement stays reproducible.

Measured on sband_short at 36 families, averaged over three independent seed
triples:

| | family ARI | median family span |
|---|---:|---:|
| `robust` | 0.479 | 265.5 MHz |
| `robust-equalised` | **0.513** | **183.4 MHz** |

A real but modest gain — +0.034 reproducibility and 31% tighter families,
consistent in sign on all three seed triples. It does **not** reach the ≥0.60 /
≤120 MHz bar the spec set. That bar came from a measurement made with a
strategy since shown to be ill-posed — but **why it cannot be reached is still
an open question**, tracked as P0-4 in
[`../docs/TODO.md`](../docs/TODO.md). Do not treat the bar as retired.

### Reproducibility is three numbers, never one

The Bench's stability check reports membership ARI, composite ARI and noise
agreement separately, because collapsing them is a measured error rather than a
hypothetical one. `sklearn`'s `adjusted_rand_score` scores `-1` as an ordinary
label, so a method leaving half its points unclustered is credited for every
within-noise pair. `leaf` scores composite 0.480 against `eom`'s 0.024 — an
apparent 20× advantage — while on membership the two are 0.032 and 0.028. A
regression test fails if anyone recombines them.

### Still deferred

A contribution-equalising scaling mode, and reworking `f02` onto its native
linear grid. The case is weaker than expected, but for a sharper reason than the
one first recorded here.

The original argument was that dropping `x03` and `f07` takes `leaf`'s narrow
share from 6.820% to 2.969%, so the dominant columns help rather than hurt. That
over-read the experiment, and the review said so: **dropping is not
down-weighting** (equalisation would move `x03` from 24.3% to 6.7%, not to
zero), and more decisively, **the k-NN measurement already says `x03` is fine** —
5.9% local against a 6.7% equal share. The experiment removed a well-behaved
column, so it is a second demonstration of the k-NN finding rather than evidence
about equalisation.

The column the local measurement actually indicts is **`f09_temporal_skew`**, at
15.3% local against 6.7% equal and benign on every global statistic. That is the
column to down-weight for a fast read on whether equalisation would help, and it
is the real precursor to the deferred scaling spec.


## Track E — a morphology-only RFI score

```bash
bluse-score                  # ~4 min end to end; --no-report scores in ~40 s
```

Full method, every de-confounding check and what would falsify it:
[`../docs/track-e-2026-09.md`](../docs/track-e-2026-09.md).

The multi-beam filter is blind on single-beam hits — it needs many beams'
worth of evidence, and a technosignature is a one-beam hit. Track E takes the
filter's own verdicts as free labels and learns them from **stamp morphology
alone**, so the same judgement survives where the filter has nothing to say.

Group 5-fold on `obsid`, 1,599,299 labelled hits, 444 observations:

| feature set | ROC-AUC |
|---|---:|
| all 16 features | 0.9911 |
| **12 stamp-morphology columns** — the default | **0.9899** |
| metadata only (4) | 0.9845 |
| Track A's entire flag set | 0.9373 |

The default is the 12-column model even though the 16-column one scores higher:
it cannot relearn the RFI frequency mask, so the claim it supports is not
partly circular with Track A's own first cut.

**The flag row is not the headline it looks like.** Track A's six booleans are
thresholds on quantities `meta` holds continuously, and 0.047 of the 0.053 gap
is the cost of binarising rather than the value of the pixels — see
[`../docs/track-e-2026-09.md`](../docs/track-e-2026-09.md) §10, which is the
most directly actionable thing in this work and is about the classical filter,
not the model.

Trained only on ≤2 and ≥32 beams, the score orders the untrained 3–31 range
monotonically across every bin — 422,480 hits, none of them in any training
fold. It survives dropping the zero-drift rows (0.9927), SNR stratification
(0.8934), collapsing beam duplicates (0.9835 on singletons) and holding out a
whole band (0.90–0.99).

float64 throughout, and the score averaged over three seeds — the bin edges
come from a random 200,000-row subsample, so a single seed's per-hit verdict is
substantially churn. See `../docs/track-e-2026-09.md` §9.

### What it produced

| | |
|---|---|
| Track A survivors | 4,565 |
| **called RFI by morphology** (>0.9) | **2,988 — 65.5%** |
| uncertain | 1,053 — 23.1% |
| shortlist (<0.1) | 524 — 11.5% |

Two thirds of the vetting list removed with a per-hit reason.

### Four things to know before quoting any of it

0. **It inverts at high brightness.** Synthetic injections (2026-09-03,
   `bluse-inject --fluctuate`, written up in
   [`../docs/injections-2026-09.md`](../docs/injections-2026-09.md)) put a known
   real signal into real stamps: above catalogue SNR ≈10 the score moves the
   wrong way, and at ≈51 with no drift **91%** of injected real signals are
   pushed above the *pruning* threshold. The shipped 0.1 shortlist threshold is
   nonetheless the best of those swept, by signal kept per survivor admitted.

1. **A low score is not a candidate.** The labels are positive-unlabelled:
   `weak_label == 0` means ≤2 beams, not verified clean.
2. **The 524-hit shortlist is about 10 emitters.** It collapses to 48
   distinct (file, 0.1 MHz) groups, the top ten holding 73%. Their stamps show
   two morphological populations, neither astrophysical: `lband_short` around
   867.8 MHz with bright first/last time rows in every hit, and `uhf_long` at
   599–678 MHz with blocky intermittent structure. That is an outlier ranking
   working, not a detection list.
3. **`x01_drift_residual` dominates** permutation importance (0.121; next is
   0.059) despite being *weaker alone* than `f04` (0.938 against 0.967), so it
   carries something the other eleven do not. Independently confirms the
   mutual-information ranking in `../docs/TODO.md` P1-4.

## Next

Track A output feeds everything else. `<name>_cat.parquet` carries all original
metadata plus `n_beams`, `n_obs_at_freq`, `log_snr`, `log_power`, `abs_drift`
and the flags — the starting feature matrix for Tracks B (feature engineering +
HDBSCAN), C (Astronomaly), D (self-supervised) and E (weak-supervision
classifier). `row` indexes back into the HDF5 stamp cube.

See `brainstorming.md` for what those tracks are.
