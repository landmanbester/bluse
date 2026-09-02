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
  GLOBULAR-overview.md           the Track B method, for humans
  GLOBULAR-technical-reference.md      ... dense, plus a map onto our code
  Tremblay-overview.md           where Track A comes from, for humans
  Tremblay-technical-reference.md      ... dense, plus a map onto our code
  Myburgh-overview.md            the same chain on blind targets, for humans
  Myburgh-technical-reference.md       ... dense, plus a map onto our code
  *.pdf                          source papers (untracked -- see .gitignore)

pyproject.toml                   installable package, uv_build backend
uv.lock                          pinned resolution -- commit changes to it
.venv/                           project env, `uv sync --extra all` (untracked)

src/bluse/                       THE CODE. Installed; importable as `bluse`.
  paths.py                       workspace resolution (see below)
  explore.py                     visual exploration of the raw HDF5
  rfi_masks.py                   MeerKAT RFI masks, provenance-tagged
  track_a_filter.py              Track A: classical filtering baseline
  features.py                    Track B: extensible feature registry
  track_b_features.py            Track B: feature extraction driver
  track_b_cluster.py             Track B: HDBSCAN clustering
  bench/                         Cluster Bench: FastAPI + htmx tuning UI
                                 (static/ and templates/ ship in the wheel)

aug_2026_workshop/               THE WORKSPACE. Data in, results out.
  README.md                      workflow, Track A results, open decisions
  brainstorming.md               technique survey + Tracks A-E plan
  data/                          HDF5 files + filtered_hits.csv (untracked).
  catalogues/                    Track A output (.csv tracked, .parquet not)
  features/                      Track B feature matrices (untracked, ~350 MB)
  clusters/                      Track B clustering output
  masks/                         empirically derived RFI masks
  plots/                         PNGs from bluse-explore (untracked)
```

**Code and workspace are separate, and that separation is load-bearing.**
Nothing under `src/bluse/` may resolve a data path from `__file__` -- the
package is installed, so `__file__` points into `.venv` or site-packages, not at
anyone's data. Use `bluse.paths`:

```python
from . import paths
paths.data_dir()            # <workspace>/data
paths.catalogues_dir()      # etc: features_dir, clusters_dir, plots_dir
paths.subdir("x", create=True)
paths.resolve_files(argv)   # bare names resolve against data/
```

The workspace is `--workspace DIR`, else `$BLUSE_ROOT`, else the nearest
directory at or above the cwd holding a `data/` **or** `features/` -- bounded by
the enclosing project (`.git` / `pyproject.toml`) and by `$HOME`, because an
unbounded search climbs out of the checkout and silently adopts an unrelated
`~/data`. If that finds nothing, one level *below* the cwd is checked and a
single unambiguous match is adopted, so running from the repository root works.
Every CLI prints the workspace it resolved **and how** (`found`, `$BLUSE_ROOT`,
`auto-detected below the cwd`, ...); if a run writes somewhere surprising, that
line is the first thing to read.

`features/` counts as a marker because Cluster Bench reads feature matrices and
only needs `data/` for stamp thumbnails.

**A command that cannot find what it needs must exit, not carry on.** Use
`paths.require_data_dir()` or `paths.missing_workspace_message(...)`; both print
the same actionable text, including any workspace-looking directories below the
cwd. `bluse-bench` used to print one warning line and start anyway, which
presents as an empty file selector and reads like a bug in the app rather than a
wrong directory.

New CLI flags go through `paths.add_workspace_arg(p)`, and any `--outdir`
default must be `None` and be filled in *after* `parse_args` (argparse defaults
are evaluated at import, before `--workspace` has been seen).

**Read `papers/BLUSE-technical-reference.md` before touching the data.** It
explains what a "hit", a "stamp", a coherent beam and a drift rate are, and the
instrument that produced them.

## Environment

The project installs. From the repository root:

```bash
uv sync --extra all          # .venv/ with bench + umap extras
source .venv/bin/activate
cd aug_2026_workshop         # the workspace
bluse-explore info
```

Five console scripts: `bluse-explore`, `bluse-track-a`, `bluse-features`,
`bluse-cluster`, `bluse-bench`. There are no runnable scripts any more -- the
PEP-723 inline metadata blocks are gone, `uv run explore.py` will not work, and
neither will `python track_a_filter.py`, because the modules use relative
imports.

For a one-off without touching the env, `uv run bluse-track-a ...` works from
the repository root.

System Python is 3.14 and has no scientific stack. Do not use it. The project
env is Python 3.11.

Astronomaly is cloned at `~/software/astronomaly` (v2.0) but is **not installed**
and needs its own environment — its pinned stack will not build on 3.14. BYOL
additionally needs torch/torchvision/byol-pytorch/kornia, which are absent from
its `requirements.txt`.

## The data

Seven HDF5 files, **2,022,171 hits**, 21 GB (2,014,055 after de-duplicating
`mk_sample_hits` on `id` -- gotcha 5). Seticore "stamp" output flattened to
a columnar table: one row per narrowband detection, ~26 scalar metadata columns
plus a `data` cube of time-frequency cutouts. Each row is therefore both a
feature vector and an image.

| File | Hits | Cube | Duration | Δf | Band |
|---|---:|---|---:|---:|---|
| `lband_long` | 557,690 | (n,1,57,120) | 286.0 s | 1.59 Hz | 855.7–1702.8 MHz |
| `lband_short` ✅ | 866,002 | (n,1,24,120) | 120.4 s | 1.59 Hz | 856.0–1068.0 MHz |
| `uhf_long` | 299,878 | (n,1,36,120) | 284.2 s | 1.01 Hz | 543.9–1080.0 MHz |
| `uhf_short` | 208,774 | (n,1,15,120) | 118.4 s | 1.01 Hz | 544.0–679.8 MHz |
| `sband_long` | 36,132 | (n,1,59,120) | 289.6 s | 1.63 Hz | 1968.8–2825.0 MHz |
| `sband_short` | 38,576 | (n,1,24,120) | 117.8 s | 1.63 Hz | 1968.8–2825.0 MHz |
| `mk_sample_hits` | 15,119 | (n,1,57,120) | 286.0 s | 1.59 Hz | 856.0–1702.8 MHz |

**`numTimesteps` is per-row, not per-file**, and it varies within two of them:
`lband_long` holds both 56 and 57, `uhf_short` both 14 and 15. The Duration
column above is `numTimesteps[0] × tsamp` and is therefore indicative, not
exact. The cube's third axis is the file's *maximum*, so a shorter hit is
padded — never assume every row fills it. Feature code should read the array
shape (as `prepare_batch` does), not the column.

Columns: `id index beam coarseChannel startChannel numChannels numTimesteps
frequency driftRate driftSteps snr power incoherentPower ra dec fch1 foff tsamp
tstart tstartts fileoffset telescopeId sourceName obsid filename data`

**`id` is the canonical hit key** — a global identifier assigned upstream, unique
across the whole delivery and stable between the HDF5 files and
`filtered_hits.csv`. Prefer it to `(file, row)` for joins.

## Nomenclature

`hit`, `stamp`, `beam`, `cluster`, `family`, `residue`, `candidate` are defined
once in [`NOMENCLATURE.md`](NOMENCLATURE.md), with what each is *not*. Do not
restate those definitions here or anywhere else -- link to that file.

The three that cause the most trouble: a **cluster** id is a batch artefact and
means nothing on its own; a **family** is a band of spectrum, not a
transmitter, and its count is a parameter rather than a measurement; and the
**candidate** set is the residue that is ALSO beam-confined, not the residue
itself.

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
   has fewer targets. Run `bluse-explore beams <file>` to see it.

   The consequence is that **the multi-beam coincidence denominator varies per
   observation**, so `n_beams` is not comparable across pointings. Catalogues
   carry `n_beams_formed` and `beam_frac` for that reason. We checked whether
   the cut should use a fraction instead of an absolute count: it moved 7 hits
   out of 7,143, all in the pre-filtered `mk_sample_hits`, so the absolute
   threshold stands. (Measured before the 2026-09 tolerance fix; the conclusion
   is about redundancy between two formulations of the same cut, which that fix
   does not touch.)

   `beam` is still a poor feature for an anomaly detector — beam index encodes
   target rank, not anything physical — but it is not the landmine it looked
   like.

5. **`mk_sample_hits.h5` is pre-filtered AND 53.7% duplicated.** It has 0%
   zero-drift where the others have 22–47% (confirmed by the BLUSE team), and
   **8,116 of its 15,119 rows are byte-identical duplicates of rows in
   `lband_long.h5`** — same `id`, frequency, snr, drift and obsid. It is a
   curated sample carved out of `lband_long`, not an independent file. Our seven
   files hold 1,619,794 rows but only **1,611,678 unique `id`s**.

   `track_b_features.py` deduplicates the combined `all_features.parquet` on
   `id`, keeping the copy from the larger file; per-file outputs are left alone.
   Anything else that pools files must do the same or it silently
   double-weights those hits. It is not a leakage risk — every duplicate pair
   shares an `obsid`, so `group_id` splitting keeps both copies together — but
   the weighting is wrong regardless. Its survival and label rates are not
   comparable to the other files either way.

6. **Time and frequency are not interchangeable axes.** Any self-supervised or
   augmentation-based method (BYOL, SimCLR, contrastive) must drop
   `RandomRotation` — rotating a dynamic spectrum destroys the drift rate, the
   single most diagnostic quantity available. Frequency flips negate drift sign.
   The default augmentation lists in the literature are written for galaxy images
   and are actively wrong here.

7. **Corrupt stamp cubes — FIXED 2026-09-02, and the old advice is now
   INVERTED.** The BLUSE team re-delivered `lband_short.h5` and `uhf_long.h5`
   with the corrupt regions repaired. Both now block-scan with **zero** bad
   blocks and extract at **100% usable rows**. Use the plain files. There is no
   longer a `*_clean.h5` anywhere, and one should not be re-introduced.

   | file | status |
   |---|---|
   | `lband_short.h5` | **use this**; 866,002 rows, 0 bad blocks, 100% usable |
   | `uhf_long.h5` | **use this**; 299,878 rows, 0 bad blocks, 100% usable |
   | `lband_short_clean.h5` | retired; was 463,625 rows, now deleted |
   | the other five | never had corruption |

   *History, because the repaired files must not be confused with the old
   ones.* Block-scanning `data` used to fail on `lband_short.h5` rows
   **338,000–742,000** (46.65%) and `uhf_long.h5` rows **264,000–270,000**
   (2.00%) with `OSError: wrong B-tree signature` — a single contiguous
   mid-file region with a readable tail, the signature of a bad transfer rather
   than truncation. Metadata was never affected, so Track A was always complete
   and correct; only Track B lost those stamps, marking them `stamp_ok=False`.
   `lband_short_clean.h5` was a strict subset by `id` with the 402,377
   unreadable rows stripped.

   **What the repair changed.** Usable rows across the survey went
   1,605,678 → **2,014,055 (+25.4%)**, which is now exactly the row count of
   the team's own `filtered_hits.csv`. `lband_short` gained 402,377 rows
   (463,625 → 866,002, +86.8%) and `uhf_long` the 6,000 it was missing. The
   retired clean file was also a **biased** subset of its band: 26.75%
   zero-drift against the full file's 46.60%, because the stripped region was
   disproportionately zero-drift. Treat any per-band result computed on
   `lband_short_clean` as drawn from a skewed sample.

   **The de-duplication hazard survived the fix, with its sign reversed.** A
   bare `bluse-features` globs `data/*.h5`, and `--combine-only` globs
   `features/*_features.parquet`, so a stale `lband_short_clean_features.parquet`
   left on disk is still picked up. `drop_superseded()` used to prefer `_clean`
   **by name**, which after the repair would have silently discarded 402,377
   good rows. It now keeps whichever file carries more **usable** rows, which
   gives the same answer on the old data and the right one on the new. Delete
   superseded per-file parquets anyway rather than relying on the guard.

   Keep `scan_bad_regions()` regardless: `uhf_long.h5` still needs it.

   **Track A survivors are not stable under a row-set change.** Re-running
   Track A on the clean file gives 928 survivors against 1,015: 922 in both,
   93 original-only (all inside the corrupt block, so they never had a stamp),
   and **6 clean-only**. The multi-beam cut counts beams per hit, so removing
   46% of the rows changes the denominator and a few hits stop looking like
   multi-beam RFI. This is the cut behaving correctly — but it means survivor
   sets from different row sets must not be compared hit-for-hit.

8. **All six band files are gzip-chunked at one stamp per chunk**
   (`chunks=(1,1,T,120)`): random single-stamp access is cheap, bulk reads are
   CPU-bound on decompression. `mk_sample_hits.h5` is uncompressed and
   contiguous. Verified on the 2026-09-02 re-delivery — the repaired
   `lband_short.h5` and `uhf_long.h5` kept the original chunking.

   *No longer applicable:* the retired `lband_short_clean.h5` was
   **uncompressed** with `chunks=(128,1,3,30)`, which inverted the access
   pattern — sequential batched reads 103,662 rows/s against 23,717 (4.4×
   faster), single random-row reads ~0.4 ms against ~0.0 ms. Nothing on disk
   has that layout now, so extraction is uniformly decompression-bound again;
   the repaired 866,002-row `lband_short.h5` extracts in 90 s.

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
et al. 2026. Seven cuts: RFI frequency masks, zero drift, **max drift**, SNR
window, multi-beam coincidence (**±1 fine channel**, ±1 drift step, per
observation), coherent/incoherent ratio (inert, now **two-sided**), cross-epoch
persistence (frequency **and** drift). The SNR floor is **raised to 15 for hits
under 16 timesteps**, and the max-drift cut is implemented but **off by
default**. **1,619,794 → 5,355 survivors (0.331%)**, ~20 s for the whole
dataset. Output in `catalogues/*_cat.parquet`, which carries all
original metadata plus `n_beams`, `n_obs_at_freq`, `n_obs_at_freq_drift`,
`log_snr`, `log_power`, `abs_drift`, `max_drift_hz_s`, the flags, and `row`
(index back into the HDF5 stamp cube).

The bolded items are 2026-09 corrections found by reading Tremblay et al. 2026
and Myburgh et al. 2026 against the code; net 7,056 → 5,355 on the same files.
Three things to carry:

- **The multi-beam tolerance is ±1 fine CHANNEL, not ±1 Hz.** Ours are
  1.013–1.630 Hz, so a hardcoded 1.0 matched ~37% too tightly in L and S band.
  Biggest single effect. `papers/Tremblay-technical-reference.md` §6.2.
- **`uhf_short` is 14–15 timesteps for all 208,774 hits** — the regime three
  separate papers warn produces mostly false positives. Its SNR floor is 15.
  `papers/Myburgh-technical-reference.md` §3.
- **The two source papers contradict each other on the coherent/incoherent
  test**, and ours implemented the less useful half. Now two-sided; still inert.
  `papers/Myburgh-technical-reference.md` §5.

**Done — Track B** (`features.py`, `track_b_features.py`, `track_b_cluster.py`):

- `features.py` is an extensible **registry**. Decorate a function with
  `@meta_feature(...)` (vectorised over the catalogue) or `@stamp_feature(...)`
  (vectorised over a batch of stamps) and it is picked up automatically; one
  function may emit several columns when they share expensive work. Features
  return **raw** values; the GLOBULAR log/quantile/unit transforms are applied
  separately by `normalise()`, so both forms reach Track E.
- All 13 GLOBULAR features (Jacobson-Bell et al. 2025) plus 3 BLUSE-specific extras.
  **They are not numerically comparable to published GLOBULAR values**: our
  spectral window is ~121–196 Hz against their 2.7 kHz, which is narrower than
  their *minimum* sweep bandwidth. `f08_turning_bw_hz` is consequently
  unresolved for ~72% of hits (companion `f08_turning_bw_saturated` flag; the
  clusterer excludes it by default).
- Extraction over all 1,619,794 hits takes ~7 min; `all_features.parquet` holds
  1,611,678 rows after deduplication on `id`. Clustering the combined table
  gives **1,491 clusters over 1,281,791 hits**, 87 noise, and **44 hits that are
  unclustered and confined to ≤4 beams** — the actual output. Re-run 2026-09
  after the Track A corrections above and the F9/F12 normalisation fixes.
- Clustering follows GLOBULAR's iterative batching. **The batching is integral,
  not a memory workaround** — one large pass collapses to 2 clusters against 71
  batched on sband_short.

**Gotcha 9 — three defects fixed 2026-09; do not reintroduce them.** All three
made the bench look insensitive to every knob except `min_cluster_size`:

1. **Cluster ids collided across batches.** HDBSCAN mints local ids `0..k-1` in
   *every* batch, but the code offset by epoch only (`lab + ep * 100_000`), so
   batch 0's cluster 3 and batch 7's cluster 3 became the same id and were
   fused. On sband_short 731 real groups collapsed into 205 reported ones. The
   reported count was a *max over batches*, not a total — which is exactly why
   only `min_cluster_size` appeared to do anything, and why the biggest ids
   carried the global medians. Ids now run globally via a running offset.
2. **`min_samples` default 2 → 8.** sklearn counts the point itself, so `ms=1`
   and `ms=2` are the same call: no core-distance smoothing, i.e. pure single
   linkage. With `allow_single_cluster=False` EOM then makes one stability
   comparison at the root and the run is **bistable** — ten identical
   3000-point draws returned either k=2 holding 99.7% of points or ~200
   microclusters with 40% noise, a 112× swing on the shuffle alone. `ms=8`
   collapses that to k∈[2,12]. Every batch really is one connected blob:
   `allow_single_cluster=True` returns k=1 on all of them.
3. **`cluster_selection_epsilon` removed, not re-ranged.** sklearn's
   `epsilon_search` (`_tree.pyx:606`) tests epsilon against `1/d`, the
   *reciprocal* of a leaf's split distance. Leaf splits here are ≳1.7, so
   `1/d ≈ 0.55`: below that every epsilon is bit-identical to 0 (ARI 1.000 for
   0.0/0.05/0.18/0.5), above it `traverse_upwards` raises `TypeError`. No-op
   region and crash region tile the domain — there is no working value.

**How to read the cluster table.** The ~14 clusters holding >1000 hits are
one-blob-per-batch and carry the *global* medians; they are an artefact of the
batching, not families. The information is in the small clusters — 16 of them
span <1 MHz in frequency, which is what a single emitter looks like — and in
the noise set.

**Two known gaps in Track B**, both documented in `track_b_cluster.py`:
cross-batch cluster matching is **not implemented**, so cluster ids are still
not comparable between batches and the same RFI family appears several times
(they are merely distinct now, not matched); and scaling had to be added beyond
GLOBULAR's spec (`--scaling robust`, default) because their transforms alone
leave feature IQRs spanning 0.036 to 5.88 on our data, so Euclidean distance
became drift rate and nothing else.

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

## `data/filtered_hits.csv`

892 MB, 2,014,055 rows, 25 columns — the same schema as the HDF5 metadata minus
the `data` cube. It is **exactly the deduplicated union of the metadata already
in our seven HDF5 files**: `id` sets match with zero on either side only, and a
full join over 14 columns shows no value differing anywhere. It contains no new
hits, no new columns, no stamps, and `incoherentPower` is all zero in it too.

Its value is threefold: it validated our HDF5 reader against an independent
export, it exposed the `mk_sample_hits` duplication in gotcha 5, and its `id`
range (1–35,904,223) suggests this workshop set is roughly a 5.6% draw from a
~36M-hit parent population — worth remembering when judging how representative
anything we find is. Despite the name it is not a further filter on top of our
data; "filtered" appears to describe how the subset was drawn from that parent.

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
- Tremblay et al. 2026, K2-18b VLA+MeerKAT, arXiv:2602.09553 — the Track A
  recipe. Summaries in `papers/Tremblay-overview.md` and
  `papers/Tremblay-technical-reference.md`. **Its section 4 does not apply the
  drift limits its section 3.2 prescribes** — we follow 3.2; see §7.1 of the
  technical reference before comparing cut-flows to its Table 3.
- Jacobson-Bell et al. 2025, GLOBULAR clustering, AJ 169:206,
  doi:10.3847/1538-3881/adb8e7, arXiv:2411.16556 — Track B. Summaries in
  `papers/GLOBULAR-overview.md` and `papers/GLOBULAR-technical-reference.md`.
  **Not** "Brzycki et al." — that was our error until 2026-09; Brzycki authored
  `setigen`/`blscint`, which the paper merely uses.
- Lochner & Bassett 2021, Astronomaly, arXiv:2010.11202
- Lochner & Rudnick 2024, Protégé, arXiv:2411.04188
