# bluse

Post-processing for BLUSE narrowband hits: take the ~2 million candidate
detections that MeerKAT's commensal technosignature survey produced, and get
down to the handful that nothing explains.

Two complementary approaches, both installed by this package:

- **Track A** — the classical filtering chain from the technosignature
  literature. Known-RFI masks, drift, SNR, multi-beam coincidence. Cheap,
  interpretable, and it throws away 99.6% of the data.
- **Track B** — unsupervised. Measure 16 morphological features per hit, then
  let HDBSCAN discover the *families* of interference present, so the output is
  "here are the named RFI populations, and here is what belongs to none of
  them" rather than "here are some outliers".

Plus **Cluster Bench**, a browser tool for trying clustering strategies
interactively instead of waiting on batch runs.

> **New here?** [`NOMENCLATURE.md`](NOMENCLATURE.md) defines the vocabulary —
> hit, stamp, beam, cluster, **family**, residue, candidate — and says what each
> one is *not*. Worth five minutes before reading any results: a "family" is a
> band of spectrum with characteristic behaviour, not a transmitter, and the
> number of families is a choice rather than a discovery.

> **Data note.** The code and documentation here are MIT-licensed. The BLUSE
> observational data and everything derived from it are **not** — they belong to
> SARAO / Breakthrough Listen and are destined for forthcoming publications. Do
> not redistribute catalogues, survivor lists or stamp files without clearing it
> with the BLUSE team first.

---

## Install

Needs Python ≥ 3.10. [`uv`](https://docs.astral.sh/uv/) is the quickest route
and the one the project is set up for.

```bash
git clone <this repo> bluse
cd bluse
uv sync --extra all          # creates .venv and installs everything
source .venv/bin/activate
```

`--extra all` pulls in Cluster Bench (FastAPI/uvicorn) and UMAP. If you only
want the batch pipeline, plain `uv sync` is a much smaller install; if you want
the bench but not UMAP's numba dependency, use `--extra bench`.

With pip instead:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
```

Either way you get five commands:

| Command | What it does |
|---|---|
| `bluse-explore` | Look at the raw HDF5 — schema, metadata distributions, stamp waterfalls |
| `bluse-track-a` | Track A: the classical filtering chain |
| `bluse-features` | Track B step 1: extract the feature matrix |
| `bluse-cluster` | Track B step 2: HDBSCAN over those features |
| `bluse-bench` | Cluster Bench: interactive clustering in the browser |
| `bluse-score` | Track E: a morphology-only RFI score for every hit |
| `bluse-score-plots` | redraw the Track E figures from an existing report |

All of them take `--help`.

## Getting the data

The HDF5 stamp files are ~21 GB and are not in the repo. Ask the BLUSE team for
access; they are seticore "stamp" files flattened to a columnar layout, one row
per hit, with a `data` cube holding each detection's time–frequency cutout.

The code is installed but the data is not, so every command resolves its paths
against a **workspace** — an ordinary directory you choose:

```
<workspace>/
    data/           the .h5 files you downloaded          ← the only input
    catalogues/     Track A output                        ← generated
    features/       Track B feature matrices              ← generated
    clusters/       Track B cluster tables and plots      ← generated
    plots/          bluse-explore PNGs                    ← generated
    masks/          empirically derived RFI masks         ← generated
```

Make `data/`, put the files in it, and work from the directory above it:

```bash
mkdir -p myrun/data && cd myrun
# ...copy or symlink the .h5 files into ./data...
bluse-explore info
```

Every command prints the workspace it resolved on startup, and how. It is found
by, in order: `--workspace DIR`, then `$BLUSE_ROOT`, then the nearest directory
at or above the cwd that has a `data/` or `features/`, then — if that finds
nothing — a single unambiguous match one level *below* the cwd. So `cd`-ing
anywhere inside a workspace works, and so does running from the repository root
with the workspace in a subdirectory:

```
workspace: /home/you/bluse/aug_2026_workshop  (auto-detected below the cwd)
```

That line is the first thing to read if a run writes somewhere surprising. And

```bash
export BLUSE_ROOT=/scratch/bluse
```

pins it from anywhere.

**Prefer a `*_clean.h5` file over the same-named original, always.** Two
originals contain a contiguous block of corrupt stamp cubes that raise
`OSError: wrong B-tree signature`. `lband_short_clean.h5` is the fixed
replacement; `uhf_long.h5` still has ~6,000 bad rows and no replacement yet. The
details, and why the clean file's Track A survivors are *not* a subset of the
original's, are in
[`aug_2026_workshop/README.md`](aug_2026_workshop/README.md#which-data-files-to-use).

## Running the pipeline from scratch

Four commands, in order. Each reads what the previous one wrote, so the order
matters. Run them from inside the workspace (or from anywhere with
`--workspace DIR`).

Timings below are for the full ~1.6M-hit workshop set on a normal laptop; a
single file is proportionally quicker.

```bash
cd myrun                       # the directory containing data/
```

**1. Look at the raw data first** *(optional, ~10 s)*

```bash
bluse-explore info             # schema, row counts, which files are usable
```

Worth doing once. It tells you whether a file is readable before you spend
seven minutes discovering it is not, and it names the corrupt-block files.

**2. Track A — the classical filtering chain** *(~20 s)*

```bash
bluse-track-a                  # all files, default parameters
bluse-track-a data/sband_short.h5      # or just one
```

Writes per file into `catalogues/`:

| File | What it is |
|---|---|
| `<name>_cat.parquet` | every hit, plus a `flag_*` column per cut and `pass_all` |
| `<name>_cutflow.csv` | how many hits each cut removed, alone and cumulatively |
| `<name>_survivors.csv` | the hits that passed everything |

Nothing is deleted — cuts add columns. `pass_all` is the survivor mask, so you
can always re-derive a different chain from the same catalogue without re-running.

**3. Track B step 1 — extract the features** *(~7 min, the slow step)*

```bash
bluse-features                 # all files
bluse-features --sample 50000  # subsample per file, for a quick look
bluse-features --list          # what features are registered, without running
```

Streams the stamp cubes, runs every feature in the registry, and writes
`features/<name>_features.parquet` plus a combined `features/all_features.parquet`.
This is the step that reads the 21 GB, and the only one that takes real time.

It needs `catalogues/` from step 2 — it joins the Track A flags onto each hit
for provenance.

**4. Track B step 2 — cluster** *(~1–3 min)*

```bash
bluse-cluster --file sband_short          # one file
bluse-cluster                             # the combined matrix
bluse-cluster --file sband_short --match --seeds 5   # + families + stability
```

Writes into `clusters/`: `<tag>_clusters.parquet` (every row plus its
`cluster_label`, and `family` if `--match`), `<tag>_summary.csv` (one row per
cluster), `<tag>_metrics.json` (the quality and stability numbers),
`<tag>_interesting.csv` (unclustered and spatially confined — the candidate
list), and diagnostic PNGs.

Before tuning anything, get the lay of the land without clustering at all:

```bash
bluse-cluster --file sband_short --report
```

That prints what each feature actually contributes to the distance HDBSCAN
takes, which is usually more informative than the first clustering run.

**5. Explore interactively**

```bash
bluse-bench                    # then open http://127.0.0.1:8000
```

Needs only `features/`, not `data/` — though clicking a point for its waterfall
does read the HDF5.

### Checking it worked

```bash
bluse-cluster --file sband_short --report | head -20
```

On the workshop's `sband_short` you should see `f02_abs_drift` with 42 distinct
values and a 0.266 tie, and `x03_channel_offset` at about 24% of the global
distance share. If those match, the chain is intact. The full set of reference
numbers is in
[`aug_2026_workshop/clusters/acceptance-2026-09.md`](aug_2026_workshop/clusters/acceptance-2026-09.md),
reproducible with `uv run python aug_2026_workshop/acceptance.py`.

### The short version

```bash
cd myrun && bluse-track-a && bluse-features && bluse-cluster && bluse-bench
```

## Resetting after a code change

**The install is editable** (`uv sync` puts a `.pth` file pointing at `src/`),
so editing the code takes effect on the next command with no reinstall. What
does *not* update automatically is anything already written to disk. Derived
files are not invalidated for you, and a stale feature matrix is silent — it
will cluster perfectly happily and give you last week's answer.

Everything below `data/` is regenerable, so when in doubt, delete and re-run.

| What you changed | What is now stale | Do this |
|---|---|---|
| Nothing; just pulled | possibly dependencies | `uv sync --extra all` |
| `pyproject.toml` deps | the venv | `uv sync --extra all` |
| `track_a_filter.py`, `rfi_masks.py` | `catalogues/`, and everything after | `rm -rf catalogues features clusters` then steps 2–4 |
| `features.py` (any feature, or `TRANSFORMS`) | `features/`, `clusters/` | `rm -rf features clusters` then steps 3–4 |
| `track_b_cluster.py`, `metrics.py`, `matching.py`, `diagnostics.py` | `clusters/` only | `rm -rf clusters` then step 4 |
| `bench/` — Python | nothing on disk | restart `bluse-bench` |
| `bench/` — CSS or JS | nothing on disk | reload the browser |

**Full reset, keeping the data:**

```bash
cd myrun
rm -rf catalogues features clusters plots masks
bluse-track-a && bluse-features && bluse-cluster
```

That is ~8 minutes and rebuilds everything from the HDF5. The `.h5` files are
never written to, so a reset cannot cost you anything you would have to
re-download.

**Rebuild the environment too:**

```bash
cd /path/to/bluse
rm -rf .venv
uv sync --extra all
```

`uv.lock` is committed deliberately, so this reproduces the exact resolution
everyone else installed rather than picking up whatever is newest.

### Things that are stale in ways you will not see

- **Cluster Bench caches per process.** Datasets, embeddings and runs are held
  in memory and keyed by parameter hash. Editing `features.py` while the bench
  is running changes nothing on screen. Restart it.
- **The browser caches CSS and JS.** There is a cache-buster keyed on file
  mtime, so this is normally handled — but if a style change appears not to
  land, hard-reload before believing it.
- **`--reload` re-imports in a fresh process** that inherits none of the
  argparse state, so pass `--workspace` via `$BLUSE_ROOT` if you use it.
- **A feature matrix does not record which code wrote it.** If you are unsure
  whether `features/` predates a change to `features.py`, compare timestamps
  (`ls -l features/ src/bluse/features.py`) or just re-extract; seven minutes
  is cheaper than a wrong conclusion.

### Verifying the code itself

```bash
uv run pytest                  # from the repo root
```

52 tests. `tests/unit/` is synthetic and runs anywhere; `tests/workspace/`
checks golden values against real feature matrices and skips cleanly when no
workspace is present. So `52 passed` inside a workspace and
`48 passed, 4 skipped` outside one are both correct — the skips are the
golden-value tests declining to run without data, not a failure.

## What the pipeline does

```
data/*.h5                      1,619,794 hits, 21 GB of stamp cubes
    │
    │  bluse-track-a           metadata only — never touches the cubes, ~20 s
    ▼
catalogues/*_cat.parquet       every hit + 8 boolean flag_* columns + pass_all
    │                          5,355 survive all cuts (0.331%)
    │
    │  bluse-features          streams the cubes, ~7 min
    ▼
features/*_features.parquet    1,611,678 rows × 16 features + weak labels
    │
    │  bluse-cluster           HDBSCAN, iterative batching
    ▼
clusters/all_summary.csv       1,491 clusters over 1,281,791 hits
clusters/all_interesting.csv   44 hits: clustered by nothing, ≤4 beams
```

### Track A — the classical chain

Reproduces the standard filtering sequence ([Tremblay et al.
2026](papers/Tremblay-overview.md), K2-18b with VLA + MeerKAT), reading metadata
only:

1. **Known-RFI frequency masks** — SARAO's published table, plus ITU allocations
   we added and flagged as inference, plus optionally a mask derived from the
   data itself.
2. **Zero drift rate** — a signal that does not Doppler-drift is not moving
   relative to the telescope, so it is local.
3. **Maximum drift rate** — nothing bound to a star drifts arbitrarily fast.
   Implemented, **off by default**: the published bound is derived for one known
   planetary system and ours is a blind survey. `--max-drift-coeff` enables it.
4. **SNR window** — below is mostly false positives, above is instrumental. The
   floor is raised for short integrations, where the pipeline's noise estimate
   has too few samples to be reliable.
5. **Multi-beam coincidence** — the strongest discriminant available here.
   A signal appearing in most of a 64-beam field is not on the sky.
6. **Coherent/incoherent power ratio** — implemented, but inert: BLUSE never
   measured `incoherentPower` for this data, which the paper's own Table 3
   independently confirms.
7. **Cross-epoch persistence** — the same frequency *and drift* on many days.

**Nothing is deleted.** Each cut adds a boolean `flag_*` column and `pass_all`
is the AND of their negations, so you can inspect what each cut caught on its
own, disagree with any of it, and re-decide by re-filtering the parquet instead
of re-running. The per-file `_cutflow.csv` records exactly that.

```bash
bluse-track-a                                  # every file in data/
bluse-track-a data/sband_short.h5 --show 40
bluse-track-a data/sband_short.h5 --no-itu     # SARAO's masks only
bluse-track-a data/uhf_short.h5 --derive-mask masks/uhf.csv
```

Four parameter choices move the survivor count by more than an order of
magnitude. They are written up under
["Four things to decide before believing the numbers"](aug_2026_workshop/README.md#four-things-to-decide-before-believing-the-numbers)
— read that before quoting any of these figures.

### Track B step 1 — features

Implements the 13 features of GLOBULAR clustering ([Jacobson-Bell et al.
2025](papers/GLOBULAR-overview.md), AJ 169:206) — spectral skew and kurtosis,
bimodality, the
kurtosis-versus-bandwidth turning point, temporal structure, "redness" —
plus three BLUSE-specific extras (drift residual, time occupancy, channel
offset). Raw values and normalised `*_n` versions are both kept, so a model
that wants its own scaling is not stuck with GLOBULAR's transforms.

```bash
bluse-features                       # all files, ~7 min
bluse-features --list                # print the feature registry
bluse-features --sample 50000        # subsample each file while iterating
```

The output also carries the free supervision that a classifier could exploit:
`weak_label` is 1 for hits seen in ≥32 beams (confidently RFI), 0 for hits
confined to ≤2 beams (**not** confidently astrophysical — treat these as
unlabelled, and if you train a plain binary classifier on 1-vs-0, say so), and
−1 for everything between. `group_id` is the observation; split on it, because
hits from one pointing share an RFI environment and a random row split will
flatter any model badly.

Our stamps are 120 channels at 1–1.6 Hz, a ~150 Hz window — narrower than
GLOBULAR's *minimum* sweep bandwidth. The features are the same construction in
a different regime, and are not numerically comparable to the published values.
`features.py` says exactly which ones this affects and how.

### Track B step 2 — clustering

The point is not to score hits as anomalous. It is to *name the RFI*, so that
what remains is a short list of things nothing accounted for.

```bash
bluse-cluster                              # all_features.parquet
bluse-cluster --file sband_short
bluse-cluster --mode single --sample 60000
```

Default mode is GLOBULAR's iterative batching: cluster ~3,000 hits, keep only
the noise points, reshuffle, repeat. Each epoch strips another layer of
recognisable RFI. **The batching is integral, not a memory workaround** — one
large pass collapses to 2 clusters on `sband_short` where batching finds 71, at
the same 99.9% clustered.

Read the result in two parts. The largest clusters are one-blob-per-batch
artefacts of the scheme and carry global medians, not a family's. The
information is in the small clusters — the ones spanning under a megahertz,
which is what a single emitter looks like.

**Known gap: cluster ids are distinct but not *matched* across batches.** The
same RFI family recurs in every batch and gets a fresh id each time. GLOBULAR
closed this with cross-batch matching to reach ~59 named families; we have not
implemented it. Their recipe — centroid keying, PCA to 6 components, t-SNE, then
HDBSCAN on the embedding — is written up in
[`papers/GLOBULAR-technical-reference.md`](papers/GLOBULAR-technical-reference.md)
§7 if you want to pick it up. Until then, read the cluster table as "these hits grouped with
something" rather than as a taxonomy, and treat the **unclustered** set — which
is well defined either way — as the actual output.

### Cluster Bench

```bash
bluse-bench                                  # http://127.0.0.1:8000
bluse-bench --port 8080 --host 0.0.0.0       # share on the LAN
```

FastAPI + htmx + a canvas scatter. Pick a file, toggle features, adjust
HDBSCAN, press **Cluster**; a re-cluster on a 35k sample takes ~1.2 s and the
points crossfade so you can see which ones changed. Click a point for its
waterfall, a table row to isolate that cluster.

It exists because the answer turned out not to be a hyperparameter. Feature
*scaling* dominates everything: GLOBULAR's transforms leave interquartile ranges
spanning 0.036 to 5.88 on our data, so Euclidean distance became drift rate and
almost nothing else. The feature rail shows each feature's raw spread as a bar
so that imbalance is visible before you cluster on it.

Full notes — what the colours and cluster numbers mean, why there is no
`epsilon` control — are in
[`aug_2026_workshop/README.md`](aug_2026_workshop/README.md#cluster-bench--interactive-hyperparameter-explorer).

## Adding your own features

The feature registry is the extension point, and it is deliberately small.
Write a function, decorate it, and it appears everywhere — in the extractor, in
`bluse-features --list`, and as a toggle in Cluster Bench. No driver code to
touch.

```python
# src/bluse/features.py

@meta_feature("snr_per_channel", description="SNR divided by hit width")
def snr_density(df):
    # df: the whole catalogue as a DataFrame
    return {"snr_per_channel": df.snr.to_numpy() / df.numChannels.to_numpy()}


@stamp_feature("peak_time_frac", description="where in the scan the peak sits")
def peak_time(b):
    # b: a StampBatch with .img (B,T,C), .spectrum (B,C), .timeseries (B,T),
    #    .df_hz, .dt_s, .meta
    return {"peak_time_frac": b.timeseries.argmax(axis=1) / b.img.shape[1]}
```

Two rules. **Batch-vectorised, never a per-hit Python loop** — these run over
1.6M hits. And **return raw values**: the log/quantile/unit-range transforms are
applied afterwards by `normalise()`, which keeps the raw column available for
anything that wants its own scaling. One function may emit several columns; do
that when they share expensive intermediate work.

Then re-extract and look:

```bash
bluse-features --list                    # confirm it registered
bluse-features data/sband_short.h5       # one file is enough to iterate
bluse-bench                              # toggle it on and off against the rest
```

To try a different *clustering* strategy rather than different features, the
seam is `feature_matrix()` and `cluster_epochs()` in
`src/bluse/track_b_cluster.py` — both take the feature DataFrame and return
labels, so a swap for a different algorithm is local.

## Repository layout

```
src/bluse/               the installed package
    paths.py             workspace resolution
    rfi_masks.py         MeerKAT RFI masks, each tagged SARAO / ITU / empirical
    explore.py           bluse-explore
    track_a_filter.py    bluse-track-a
    features.py          the feature registry
    track_b_features.py  bluse-features
    track_b_cluster.py   bluse-cluster
    track_e_score.py     bluse-score
    track_e_plots.py     bluse-score-plots
    bench/               bluse-bench — Cluster Bench
aug_2026_workshop/       the working record: results, findings, decisions
    README.md            per-file numbers, caveats, what to distrust
    brainstorming.md     technique survey and the proposed tracks A–E
    catalogues/          tracked cut-flows and survivor lists
    clusters/            tracked cluster summaries
    data/               ← put the .h5 files here (gitignored, 21 GB)
papers/                  reference literature + our summaries
    Tremblay-*.md        where Track A comes from: overview (human) and
                         technical reference (dense, maps onto our code)
    Myburgh-*.md         the same chain on blind targets — three extra
                         filters, and a contradiction worth knowing
    GLOBULAR-*.md        the Track B method, same pair
AGENTS.md                notes for agents working in this repo — including a
                         list of gotchas that cost real debugging time
```

The generated parquet files are gitignored; the CSV summaries are tracked,
because they are the record of what we found.

## Where this is going

Track A output feeds everything. `<name>_cat.parquet` carries all original
metadata plus `n_beams`, `n_obs_at_freq`, `log_snr`, `log_power`, `abs_drift`
and the flags, and `row` indexes back into the HDF5 stamp cube.

Still open, in rough priority order:

- **Cross-batch cluster matching** — the gap that stands between "1,187
  clusters" and "~59 named RFI families". Recipe in
  `papers/GLOBULAR-technical-reference.md` §7.
- **Seed signals** — inject identical synthetic drifting narrowband signals into
  every batch, so real signals resembling them have something to cluster with
  and can be tracked through the epochs. Cheap, and the paper's own suggestion
  for tuning the method toward a signal type by example.
- **Tracks C and D** — Astronomaly-style active learning, and self-supervised
  representations. See `brainstorming.md`.
- **Synthetic injections.** The only true objective function available: every
  Track E number measures agreement with the spatial filter, which is a good
  instrument and not ground truth.

*(Track E landed 2026-09-03 — see below. `uhf_long.h5` and `lband_short.h5`
were re-delivered repaired on 2026-09-02, so the "clean file" item is closed.)*

## Track E — the RFI score

```bash
bluse-score                      # ~9 min: fit, score 2M hits, catalogues, report, figures
bluse-score --no-report          # ~80 s: scores only
bluse-score --seeds 1            # single seed, ~30 s (see the caveat below)
bluse-score --features all       # 16 features instead of the 12 stamp columns
bluse-score-plots                # redraw the figures without refitting
```

The multi-beam spatial filter is the survey's strongest discriminant and it is
**blind on single-beam hits** — it needs many beams' worth of evidence, and a
technosignature is a one-beam hit. Track E takes the filter's own verdicts as
free labels and learns them from the **stamp morphology alone**, so the same
judgement is available where the filter has nothing to say.

Group 5-fold on `obsid`, 1,599,299 labelled hits, 444 observations:

| | ROC-AUC |
|---|---:|
| metadata + stamp (16) | 0.9911 |
| **12 stamp-morphology features** — the default | **0.9899** |
| Track A's entire flag set | 0.9373 |

Trained only on ≤2 and ≥32 beams, the score orders the untrained 3–31 range
monotonically across every bin — 422,480 hits it never saw. It survives
dropping the zero-drift rows, SNR stratification, collapsing beam duplicates,
and holding out a whole band. Full method, every de-confounding check, and what
would falsify it: [`docs/track-e-2026-09.md`](docs/track-e-2026-09.md).

The matrix is float64 and the score is averaged over three seeds. Neither is
decoration — `HistGradientBoosting` draws its 256 bin edges from a random
200,000-row subsample, so a single seed's per-hit verdict is substantially
churn: the `contrarian` count alone swings 2,972–3,368 on the seed. §9 of that
document has the measurement, and the control that pins the mechanism.

**Output** in `scores/`, with figures in `plots/`:

| file | what it is |
|---|---|
| `candidates.csv` | every Track A survivor, ranked. **2,988 of 4,565 (65.5%) look exactly like multi-beam RFI** |
| `contrarian.csv` | 1,515 hits in ≥32 beams that score clean — filter and morphology disagreeing |
| `ambiguous.csv` | 411,898 hits in 3–31 beams, where the filter abstains |
| `all_scores.parquet` | `rfi_score` on all 2,014,055 hits |

**A high score is strong evidence of RFI. A low score is not evidence of a
technosignature** — the labels are positive-unlabelled, and the 524-hit
shortlist collapses to 48 frequency groups whose stamps show two *instrumental*
populations. See
[`NOMENCLATURE.md`](NOMENCLATURE.md#rfi-score) before quoting any of it.

## Licence

MIT — see [`LICENSE`](LICENSE). **Code and documentation only.** It does not
extend to the BLUSE observational data or to catalogues derived from it.
