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

All five take `--help`.

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

Every command prints the workspace it resolved on startup. It is found by, in
order: `--workspace DIR`, then `$BLUSE_ROOT`, then the nearest directory at or
above the cwd containing a `data/`, then the cwd. So `cd`-ing anywhere inside a
workspace just works, and

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

## What the pipeline does

```
data/*.h5                      2,022,171 hits, 21 GB of stamp cubes
    │
    │  bluse-track-a           metadata only — never touches the cubes, ~20 s
    ▼
catalogues/*_cat.parquet       every hit + 7 boolean flag_* columns + pass_all
    │                          7,143 survive all cuts (0.353%)
    │
    │  bluse-features          streams the cubes, ~7 min
    ▼
features/*_features.parquet    1,611,678 rows × 16 features + weak labels
    │
    │  bluse-cluster           HDBSCAN, iterative batching
    ▼
clusters/all_summary.csv       1,187 clusters over 1,281,826 hits
clusters/all_interesting.csv   27 hits: clustered by nothing, ≤4 beams
```

### Track A — the classical chain

Reproduces the standard filtering sequence (Tremblay et al. 2026, K2-18b with
VLA + MeerKAT), reading metadata only:

1. **Known-RFI frequency masks** — SARAO's published table, plus ITU allocations
   we added and flagged as inference, plus optionally a mask derived from the
   data itself.
2. **Zero drift rate** — a signal that does not Doppler-drift is not moving
   relative to the telescope, so it is local.
3. **SNR window** — below is mostly false positives, above is instrumental.
4. **Multi-beam coincidence** — the strongest discriminant available here.
   A signal appearing in most of a 64-beam field is not on the sky.
5. **Coherent/incoherent power ratio** — implemented, but inert: BLUSE never
   measured `incoherentPower` for this data.
6. **Cross-epoch persistence** — the same frequency turning up on many days.

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

Implements the 13 features of GLOBULAR clustering (Brzycki et al. 2025,
arXiv:2411.16556) — spectral skew and kurtosis, bimodality, the
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
implemented it. Until then, read the cluster table as "these hits grouped with
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
    bench/               bluse-bench — Cluster Bench
aug_2026_workshop/       the working record: results, findings, decisions
    README.md            per-file numbers, caveats, what to distrust
    brainstorming.md     technique survey and the proposed tracks A–E
    catalogues/          tracked cut-flows and survivor lists
    clusters/            tracked cluster summaries
    data/               ← put the .h5 files here (gitignored, 21 GB)
papers/                  reference literature
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
  clusters" and "~59 named RFI families".
- **Track E: a weak-supervision classifier** on `weak_label` / `group_id`.
  The columns are already in every feature parquet.
- **Tracks C and D** — Astronomaly-style active learning, and self-supervised
  representations. See `brainstorming.md`.
- **A clean `uhf_long.h5`**, on the same terms as `lband_short_clean.h5`.

## Licence

MIT — see [`LICENSE`](LICENSE). **Code and documentation only.** It does not
extend to the BLUSE observational data or to catalogues derived from it.
