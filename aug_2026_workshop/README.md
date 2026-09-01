# BLUSE workshop — August 2026

Finding interesting things in ~2M BLUSE narrowband hits.

| File | What it is |
|---|---|
| `brainstorming.md` | Technique survey from the literature + proposed tracks A–E |
| `explore.py` | Visual exploration of the raw HDF5 (stamps, metadata, coincidence) |
| `rfi_masks.py` | Known-RFI frequency masks for MeerKAT, with provenance labels |
| `track_a_filter.py` | **Track A** — the classical post-processing baseline |
| `data/` | The 7 HDF5 stamp files (21 GB) |
| `catalogues/` | Track A output: filtered catalogues + cut-flows |
| `plots/` | PNGs from `explore.py` |
| `masks/` | Empirically derived RFI masks |

## Setup

Either use `uv` (scripts carry inline dependency metadata):

```bash
uv run explore.py info
uv run track_a_filter.py
```

...or the venv already built here:

```bash
.venv/bin/python explore.py info
```

## Track A: the classical baseline

```bash
# everything, default parameters, ~20 s for all 2M hits
python track_a_filter.py

# one file, look at more survivors
python track_a_filter.py data/sband_short.h5 --show 40

# documented SARAO masks only -- drop our ITU-allocation guesses
python track_a_filter.py data/sband_short.h5 --no-itu

# derive an RFI mask from the data instead of trusting published tables
python track_a_filter.py data/uhf_short.h5 --derive-mask masks/uhf.csv

# when the real incoherent-beam powers turn up
python track_a_filter.py data/sband_short.h5 --incoherent-power incoh.csv
```

### The cuts

Applied in order, following Tremblay et al. 2026 (K2-18b, VLA + MeerKAT):

1. **Known-RFI frequency masks** — `rfi_masks.py`
2. **Zero drift rate** — local RFI
3. **SNR window** — below is mostly false positives, above is instrumental
4. **Multi-beam coincidence** — ±1 Hz, ±1 drift step, per observation
5. **Coherent/incoherent ratio** — `SNR_coh ≤ √N·SNR_incoh` *(wired up, inert until data arrives)*
6. **Cross-epoch persistence** — same frequency across many observations

**Nothing is deleted.** Each cut writes a boolean `flag_*` column; `pass_all` is
the AND of their negations. Change your mind about any cut by re-filtering the
parquet — no need to re-run.

### Results with default parameters

| File | Hits | Survivors | % |
|---|---:|---:|---:|
| `lband_long` | 557,690 | 853 | 0.153 |
| `lband_short` | 866,002 | 1,015 | 0.117 |
| `uhf_long` | 299,878 | 2,406 | 0.802 |
| `uhf_short` | 208,774 | 1,870 | 0.896 |
| `sband_long` | 36,132 | 47 | 0.130 |
| `sband_short` | 38,576 | 46 | 0.119 |
| `mk_sample_hits` | 15,119 | 906 | 5.992 |
| **total** | **2,022,171** | **7,143** | **0.353** |

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
- **Hits-per-beam steps at beams ~49/~55 — explained, benign.** One beam per
  target, filled from 0, and sparse sky has fewer targets. `python explore.py
  beams <file>` shows it. Catalogues now carry `n_beams_formed` and `beam_frac`
  because the coincidence denominator varies per observation.

## Cluster Bench — interactive hyperparameter explorer

```bash
python explorer/app.py            # then open http://127.0.0.1:8000
python explorer/app.py --port 8080 --host 0.0.0.0   # share on the LAN
```

FastAPI + htmx + a canvas scatter. Pick a file, toggle features, adjust
HDBSCAN, press **Cluster**. A re-cluster on a 35k sample takes ~1.2 s.

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
