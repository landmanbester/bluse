# AGENTS.md

Context for agents working in this repository.

## What this is

Analysis of data from **BLUSE** (Breakthrough Listen User Supplied Equipment), the
commensal technosignature survey running alongside the MeerKAT radio telescope.
The immediate goal is an **August 2026 workshop**: find interesting things in a
~2M-hit subset using anomaly detection and related techniques.

Owner: `lbester@sarao.ac.za` (SARAO).

Realistic framing, and please hold onto it: **this will almost certainly not find
a technosignature.** The published precedent (Tremblay et al. 2026) found zero
surviving candidates. What it can plausibly produce is an RFI taxonomy for BLUSE,
instrumental-systematics discovery, and a demonstrated human-in-the-loop ranking
workflow. Those are the real deliverables. Do not oversell a result.

## Layout

```
papers/                          reference papers + our summaries
  BLUSE-overview.md              how BLUSE works, for humans
  BLUSE-technical-reference.md   how BLUSE works, dense, for agents
  Astronomaly-overview.md        Astronomaly/Protege, for humans
  Astronomaly-technical-reference.md   ... dense, plus BLUSE application guide
  *.pdf                          source papers (untracked -- see .gitignore)

aug_2026_workshop/
  README.md                      workflow, Track A results, open decisions
  brainstorming.md               technique survey + Tracks A-E plan
  explore.py                     visual exploration of the raw HDF5
  rfi_masks.py                   MeerKAT RFI masks, provenance-tagged
  track_a_filter.py              Track A: classical filtering baseline
  features.py                    Track B: extensible feature registry
  track_b_features.py            Track B: feature extraction driver
  track_b_cluster.py             Track B: HDBSCAN clustering
  data/                          7 HDF5 files, 21 GB (untracked)
  catalogues/                    Track A output (.csv tracked, .parquet not)
  features/                      Track B feature matrices (untracked, ~350 MB)
  clusters/                      Track B clustering output
  masks/                         empirically derived RFI masks
  plots/                         PNGs from explore.py (untracked)
  .venv/                         python 3.11 env (untracked)
```

**Read `papers/BLUSE-technical-reference.md` before touching the data.** It
explains what a "hit", a "stamp", a coherent beam and a drift rate are, and the
instrument that produced them.

## Environment

No project-wide Python. Use the workshop venv or `uv` (scripts carry PEP-723
inline dependency metadata):

```bash
cd aug_2026_workshop
.venv/bin/python explore.py info        # or:  uv run explore.py info
```

System Python is 3.14 and has no scientific stack. Do not use it. The venv has
h5py, numpy, pandas, pyarrow, matplotlib on Python 3.11.

Astronomaly is cloned at `~/software/astronomaly` (v2.0) but is **not installed**
and needs its own environment — its pinned stack will not build on 3.14. BYOL
additionally needs torch/torchvision/byol-pytorch/kornia, which are absent from
its `requirements.txt`.

## The data

Seven HDF5 files, **2,022,171 hits**, 21 GB. Seticore "stamp" output flattened to
a columnar table: one row per narrowband detection, ~26 scalar metadata columns
plus a `data` cube of time-frequency cutouts. Each row is therefore both a
feature vector and an image.

| File | Hits | Cube | Duration | Δf | Band |
|---|---:|---|---:|---:|---|
| `lband_long` | 557,690 | (n,1,57,120) | 286.0 s | 1.59 Hz | 855.7–1702.8 MHz |
| `lband_short` | 866,002 | (n,1,24,120) | 120.4 s | 1.59 Hz | 856.0–1068.0 MHz |
| `uhf_long` | 299,878 | (n,1,36,120) | 284.2 s | 1.01 Hz | 543.9–1080.0 MHz |
| `uhf_short` | 208,774 | (n,1,15,120) | 118.4 s | 1.01 Hz | 544.0–679.8 MHz |
| `sband_long` | 36,132 | (n,1,59,120) | 289.6 s | 1.63 Hz | 1968.8–2825.0 MHz |
| `sband_short` | 38,576 | (n,1,24,120) | 117.8 s | 1.63 Hz | 1968.8–2825.0 MHz |
| `mk_sample_hits` | 15,119 | (n,1,57,120) | 286.0 s | 1.59 Hz | 856.0–1702.8 MHz |

Columns: `id index beam coarseChannel startChannel numChannels numTimesteps
frequency driftRate driftSteps snr power incoherentPower ra dec fch1 foff tsamp
tstart tstartts fileoffset telescopeId sourceName obsid filename data`

## Gotchas — read before writing analysis code

1. **Stamps are padded with `-1`.** Cutouts are right-aligned in a 120-channel
   buffer; unused leading columns are exactly `-1`. `numChannels` (79–120) gives
   the true width. Mask `== -1` before any arithmetic — feeding it to a
   normaliser or a CNN will poison the result. `explore.py:clean_stamp()`
   detects pad columns rather than trusting the stored width.

2. **`incoherentPower` is identically zero, and always will be.** The BLUSE
   team have confirmed it was **never measured** for this dataset — it is not
   pending delivery. The strongest classical discriminant,
   `SNR_coh ≤ √N·SNR_incoh`, is therefore permanently unavailable here, and
   multi-beam coincidence has to carry that load alone. Cut 5 of
   `track_a_filter.py` stays wired to `--incoherent-power FILE` so the same code
   works on a future dataset that does carry the incoherent beam.

3. **There is a detached SNR ~10⁷–10⁸ population**, separate from the main
   distribution (6 to ~10⁴). It sits at drift ≈ 0 and dies entirely under the
   multi-beam cut. Instrumental. Always work in log space; consider an upper
   clip. Never feed raw `snr` or `power` to a distance-based algorithm.

4. **Hits-per-beam steps down at beams ~49 and ~55 — this is astronomy, not a
   defect.** BLUSE assigns one coherent beam per catalogue target inside the
   primary field of view and fills beams contiguously from 0, so a pointing with
   only 49 targets populates beams 0–48 and leaves 49–63 empty. Verified on
   `sband_short`: beam↔`sourceName` is 1:1 in 30/30 observations, indices are
   contiguous from 0 in 28/30, and beams formed falls monotonically with
   galactic latitude (64 beams at |b|≈11°, 20 at |b|≈65°) — sparse sky simply
   has fewer targets. Run `python explore.py beams <file>` to see it.

   The consequence is that **the multi-beam coincidence denominator varies per
   observation**, so `n_beams` is not comparable across pointings. Catalogues
   carry `n_beams_formed` and `beam_frac` for that reason. We checked whether
   the cut should use a fraction instead of an absolute count: it moves 7 hits
   out of 7,143, all in the pre-filtered `mk_sample_hits`, so the absolute
   threshold stands.

   `beam` is still a poor feature for an anomaly detector — beam index encodes
   target rank, not anything physical — but it is not the landmine it looked
   like.

5. **`mk_sample_hits.h5` is pre-filtered** (0% zero-drift vs 22–47% elsewhere).
   Its survival rates are not comparable. Do not pool it with the others.

6. **Time and frequency are not interchangeable axes.** Any self-supervised or
   augmentation-based method (BYOL, SimCLR, contrastive) must drop
   `RandomRotation` — rotating a dynamic spectrum destroys the drift rate, the
   single most diagnostic quantity available. Frequency flips negate drift sign.
   The default augmentation lists in the literature are written for galaxy images
   and are actively wrong here.

7. **Two files have corrupt stamp cubes.** Verified by block-scanning `data`:
   `lband_short.h5` rows **338,000–742,000** (404,000 rows, 46.65%) and
   `uhf_long.h5` rows **264,000–270,000** (6,000 rows, 2.00%) raise
   `OSError: wrong B-tree signature`. Each is a single contiguous mid-file
   region with a readable tail — the signature of a bad transfer, not
   truncation. **Metadata columns are unaffected**, so Track A (metadata-only)
   is complete and correct; Track B loses those stamps and marks them
   `stamp_ok=False`. Re-copying those two files would recover 410,000 hits.
   `track_b_features.py:scan_bad_regions()` probes cheaply and skips them.

8. **Compression.** The six band files are gzip-chunked at one stamp per chunk:
   random access to individual stamps is cheap, bulk reads are CPU-bound on
   decompression. `mk_sample_hits.h5` is uncompressed and contiguous.

## Conventions

- **Filtering is non-destructive.** Cuts add boolean `flag_*` columns; `pass_all`
  is the AND of their negations. Never drop rows in a pipeline stage — someone
  will want to disagree with a cut without re-running.
- **Tag provenance on anything not measured.** `rfi_masks.py` labels every entry
  `"SARAO"` (documented), `"ITU"` (our inference from spectrum allocations), or
  `"empirical"` (derived from these data). This matters: on `sband_short` the
  SARAO masks flag 3 hits, ours flag 33,449.
- **Report cut-flows two ways** — what a cut catches alone, and what it adds
  given the cuts before it. Big gaps mean redundancy.
- **Verify before claiming.** Measure it and quote the number, or say you didn't.
  Several early recommendations in this repo were wrong until checked against the
  actual files (the `incoherentPower` assumption most notably).
- Scripts are CLI-driven with `argparse` subcommands, write outputs to a
  directory, and print the path.

## Where things stand

**Done — Track A** (`track_a_filter.py`): classical baseline following Tremblay
et al. 2026. Six cuts: RFI frequency masks, zero drift, SNR window, multi-beam
coincidence (±1 Hz, ±1 drift step, per observation), coherent/incoherent ratio
(inert), cross-epoch persistence. **2,022,171 → 7,143 survivors (0.353%)**, ~20 s
for the whole dataset. Output in `catalogues/*_cat.parquet`, which carries all
original metadata plus `n_beams`, `n_obs_at_freq`, `log_snr`, `log_power`,
`abs_drift`, the flags, and `row` (index back into the HDF5 stamp cube).

**Done — Track B** (`features.py`, `track_b_features.py`, `track_b_cluster.py`):

- `features.py` is an extensible **registry**. Decorate a function with
  `@meta_feature(...)` (vectorised over the catalogue) or `@stamp_feature(...)`
  (vectorised over a batch of stamps) and it is picked up automatically; one
  function may emit several columns when they share expensive work. Features
  return **raw** values; the GLOBULAR log/quantile/unit transforms are applied
  separately by `normalise()`, so both forms reach Track E.
- All 13 GLOBULAR features (Brzycki et al. 2025) plus 3 BLUSE-specific extras.
  **They are not numerically comparable to published GLOBULAR values**: our
  spectral window is ~121–196 Hz against their 2.7 kHz, which is narrower than
  their *minimum* sweep bandwidth. `f08_turning_bw_hz` is consequently
  unresolved for ~72% of hits (companion `f08_turning_bw_saturated` flag; the
  clusterer excludes it by default).
- Extraction over all 2,022,171 hits takes ~7 min; 1,612,171 have usable
  features (the shortfall is the corruption in gotcha 7).
- Clustering follows GLOBULAR's iterative batching. **The batching is integral,
  not a memory workaround** — their hyperparameters are tuned for ~3000-point
  batches and give 10 clusters instead of 205 on one large pass.

**Two known gaps in Track B**, both documented in `track_b_cluster.py`:
cross-batch cluster matching is **not implemented**, so cluster ids are not
comparable between batches and the same RFI family appears several times; and
scaling had to be added beyond GLOBULAR's spec (`--scaling robust`, default)
because their transforms alone leave feature IQRs spanning 0.036 to 5.88 on our
data, so Euclidean distance became drift rate and nothing else.

**Next — Track E**: weak-supervision classifier. `features/*_features.parquet`
already carries `weak_label` (1 RFI / 0 spatially-confined / −1 ambiguous),
`weak_label_reason` and `group_id`. **Split on `group_id`** — hits from one
observation share a pointing, an RFI environment and a calibration, so a random
row split leaks. Label 0 means *spatially confined*, not *verified clean*:
positive-unlabelled learning is the honest framing. Then Tracks C (Astronomaly)
and D (self-supervised, GPU, stretch). See `brainstorming.md`.

**Three Track A parameters are unresolved judgement calls** — the ITU masks, the
digital-TV comb (off by default; enabling it masks all of UHF), and `--tol-steps`
(the strict ±1 match leaks a known 870.2323 MHz emitter). All documented in
`aug_2026_workshop/README.md`. Do not silently change these defaults.

## Questions to the BLUSE team — answered

1. **Can `incoherentPower` be populated?** No — it was never measured for this
   data. Treat the coherent/incoherent test as permanently unavailable.
2. **Why are the `_short` files ~118 s?** Still open. The 150 s figure is ours,
   read off Czech et al. 2026 §6: of ~1.5M coherent beams "approximately 1.2
   million were viable for technosignature searching (the remainder were too
   short in duration, less than 150s)". That sentence describes how the *survey*
   was triaged; it is not necessarily a rule about what may be analysed. The
   `_short` files are a deliberate short-integration subset, and the honest
   reading is that they sit below the survey's own viability line and should be
   analysed and reported separately from the `_long` files, not that they are
   unusable. Worth confirming the intent.
3. **What causes the hits-per-beam step?** Answered by us, not the team — see
   gotcha 4. Sparse sky, fewer targets, fewer beams formed. Benign.
4. **Has `mk_sample_hits.h5` been pre-filtered?** Yes. Keep it out of pooled
   statistics.
5. **Are per-antenna stamp data available?** Out of scope for this workshop —
   the stamp files are too large.

## Licensing

The **code** in this repository is MIT licensed (`LICENSE`), copyright Landman
Bester.

That licence covers our code and documentation only. It does **not** cover the
BLUSE observational data, nor the derived catalogues in
`aug_2026_workshop/catalogues/`, which contain sky coordinates, Gaia and exotica
source identifiers, and observation IDs belonging to SARAO / Breakthrough Listen.
Czech et al. 2026 states those survey results are destined for forthcoming
publications. The repository is **private** for that reason. Clear it with the
BLUSE team before redistributing any of it or making the repository public.

## Key references

- Czech et al. 2026, BLUSE, arXiv:2607.23651 — the instrument
- Tremblay et al. 2026, K2-18b VLA+MeerKAT, arXiv:2602.09553 — the filtering recipe
- Brzycki et al. 2025, GLOBULAR clustering, AJ, doi:10.3847/1538-3881/adb8e7 — Track B
- Lochner & Bassett 2021, Astronomaly, arXiv:2010.11202
- Lochner & Rudnick 2024, Protégé, arXiv:2411.04188
