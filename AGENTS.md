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
  data/                          7 HDF5 files, 21 GB (untracked)
  catalogues/                    Track A output (.csv tracked, .parquet not)
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

2. **`incoherentPower` is identically zero in all seven files.** The strongest
   classical discriminant, `SNR_coh ≤ √N·SNR_incoh`, is unavailable. Cut 5 of
   `track_a_filter.py` is implemented and wired to `--incoherent-power FILE` but
   inert. The user is trying to source the real values.

3. **There is a detached SNR ~10⁷–10⁸ population**, separate from the main
   distribution (6 to ~10⁴). It sits at drift ≈ 0 and dies entirely under the
   multi-beam cut. Instrumental. Always work in log space; consider an upper
   clip. Never feed raw `snr` or `power` to a distance-based algorithm.

4. **Hits-per-beam steps down at beams ~49 and ~55.** Unexplained instrumental
   systematic. Any anomaly detector given `beam` as a feature will "discover" it.
   Exclude `beam` or understand it first.

5. **`mk_sample_hits.h5` is pre-filtered** (0% zero-drift vs 22–47% elsewhere).
   Its survival rates are not comparable. Do not pool it with the others.

6. **Time and frequency are not interchangeable axes.** Any self-supervised or
   augmentation-based method (BYOL, SimCLR, contrastive) must drop
   `RandomRotation` — rotating a dynamic spectrum destroys the drift rate, the
   single most diagnostic quantity available. Frequency flips negate drift sign.
   The default augmentation lists in the literature are written for galaxy images
   and are actively wrong here.

7. **Compression.** The six band files are gzip-chunked at one stamp per chunk:
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

**Next — Track B**: the GLOBULAR feature set (13 hand-crafted features per hit,
Brzycki et al. 2025) computed over the stamp cubes, then HDBSCAN, to produce a
named RFI taxonomy. Then Tracks C (Astronomaly human-in-the-loop), E
(weak-supervision classifier trained on free labels from the spatial filter), and
D (self-supervised, GPU, stretch). See `brainstorming.md`.

**Three Track A parameters are unresolved judgement calls** — the ITU masks, the
digital-TV comb (off by default; enabling it masks all of UHF), and `--tol-steps`
(the strict ±1 match leaks a known 870.2323 MHz emitter). All documented in
`aug_2026_workshop/README.md`. Do not silently change these defaults.

## Open questions for the BLUSE team

1. Can `incoherentPower` be populated?
2. Why are the `_short` files ~118 s, below the 150 s viability cut in Czech et al.?
3. What causes the hits-per-beam step at beams ~49 and ~55?
4. Has `mk_sample_hits.h5` been pre-filtered?
5. Are per-antenna stamp data available? That would allow true coherence testing.

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
