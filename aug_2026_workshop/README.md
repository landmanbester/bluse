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

**Prefer a `*_clean.h5` file over the same-named original, always.**

Two of the original files contain a contiguous block of corrupt stamp cubes
that raise `OSError: wrong B-tree signature` — a bad transfer, not truncation.
Replacements are arriving with those rows stripped out.

| file | use it? | why |
|---|---|---|
| `lband_short_clean.h5` | **yes** | 463,625 rows, reads clean end to end |
| `lband_short.h5` | no — superseded | 866,002 rows, but 46.65% of the stamps are unreadable |
| `uhf_long.h5` | yes, for now | still has 6,000 bad rows (264,000–270,000); no clean version yet |
| the other five | yes | never had corruption |

The clean file's *rows* are a strict subset of the original's by hit `id` —
same 26 columns, same schema, nothing altered, 402,377 unreadable rows removed.
Only the stamp cubes were ever affected; every metadata column reads fine in
both, which is why the Track A catalogues were always trustworthy. (Its Track A
*survivors* are a different matter — see the results table below.)

Two practical notes:

- **It reads 4.4× faster in bulk** (103,662 vs 23,717 rows/s) because it is
  stored uncompressed. Individual scattered stamp reads are a little slower
  (~0.4 ms vs ~0.0 ms), so `explore.py stamps --sort snr` will feel marginally
  less snappy while feature extraction gets much quicker.
- **Don't let both versions into one run.** `track_b_features.py` with no
  arguments globs `data/*.h5` and would ingest the old and the clean file as
  two datasets covering the same hits — and the *corrupt* one would win the
  tie-break, because it has more rows (866,002, of which 404,000 have no
  readable stamp) than the clean file's 463,625. `drop_superseded()` now
  removes any `X` whenever `X_clean` is present and prints what it dropped, so
  this is safe by default. Still cleanest to move the original aside.

Track A has already been re-run on the clean file, so
`catalogues/lband_short_clean_cat.parquet` exists and feature extraction will
not stop for it (670 survivors, 0.145% — see the Track A results table for why
that is not simply a subset of the original's 1,015).

To re-run the pipeline on the corrected set:

```bash
mkdir -p data/superseded && mv data/lband_short.h5 data/superseded/
bluse-features                             # now picks up *_clean.h5 only
bluse-cluster                              # all_features.parquet
```

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
3. **Maximum drift rate** [§3.2] — faster than a bound companion plausibly drifts
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
| `lband_short_clean` | 463,625 | 670 | 0.145 | 928 |
| `uhf_long` | 299,878 | 2,281 | 0.761 | 2,406 |
| `uhf_short` | 208,774 | 1,193 | 0.571 | 1,870 |
| `sband_long` | 36,132 | 43 | 0.119 | 47 |
| `sband_short` | 38,576 | 43 | 0.111 | 46 |
| `mk_sample_hits` | 15,119 | 894 | 5.913 | 906 |
| **total** | **1,619,794** | **5,692** | **0.351** | **7,056** |

The total is over the corrected set (`lband_short_clean`, not `lband_short`).
The `was` column is the same set before the 2026-09 corrections below, so it is
comparable row by row.

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
rule but never its upper bound. It is not inert: it flags 12,756 hits in
`lband_short_clean` (2.75%) and 1,134 in `uhf_short` (0.54%), and nothing
anywhere else. The limit scales with observing frequency at 4.18e-4 Hz/s per
MHz, which reproduces all three anchor values the paper quotes.

*Caveat, recorded in the code:* that coefficient bounds Earth's rotation
(~1.1e-4 Hz/s per MHz, universal) **plus K2-18b's own orbital acceleration**.
Our targets are arbitrary Gaia sources, so treat it as a generous envelope
rather than a per-target limit. `--no-max-drift` turns it off.

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
  re-copy of that one too, on the same terms as `lband_short_clean.h5`.
- **Hits-per-beam steps at beams ~49/~55 — explained, benign.** One beam per
  target, filled from 0, and sparse sky has fewer targets. `bluse-explore
  beams <file>` shows it. Catalogues now carry `n_beams_formed` and `beam_frac`
  because the coincidence denominator varies per observation.

## Cluster Bench — interactive hyperparameter explorer

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
strip and the plot dim, and `working` appears in the header — a re-cluster takes
1–2 s and the panel otherwise keeps showing the previous numbers, which reads as
nothing having happened.

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

## Next

Track A output feeds everything else. `<name>_cat.parquet` carries all original
metadata plus `n_beams`, `n_obs_at_freq`, `log_snr`, `log_power`, `abs_drift`
and the flags — the starting feature matrix for Tracks B (feature engineering +
HDBSCAN), C (Astronomaly), D (self-supervised) and E (weak-supervision
classifier). `row` indexes back into the HDF5 stamp cube.

See `brainstorming.md` for what those tracks are.
