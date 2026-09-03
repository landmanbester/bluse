# The feature-delivery format

The August 2026 workshop fixed a convention for sharing feature sets so that
anyone's features can be cross-matched against anyone else's, and against the
hit metadata. It is specified by the BLUSE side in
`aug_2026_workshop/features/format.md`. That file is **not tracked here** — it
is the workshop's shared document, and it describes several collaborators'
unpublished feature sets by name. This page records our half of it: what the
convention is, how we implement it, and the description we contributed back.

## The convention

    <dataset>_<method>_features.parquet        hit `id` as the index, named `id`

- `dataset` is the delivery the hits came from: `lband_long`, `lband_short`,
  `mk_sample_hits`, `sband_long`, `sband_short`, `uhf_long`, `uhf_short`.
- `method` is a short name for the extraction. Ours is **`globular`**.
- The index is the point of the whole thing. Without it a feature matrix is a
  block of numbers that cannot be joined to anything.

Adopted 2026-09-03. The files are otherwise unchanged: same rows, same columns,
same values. `bluse-features --migrate` converts a pre-convention workspace
without re-extracting — it reads and rewrites, and leaves the originals alone.

## Where it lives in the code

`src/bluse/feature_io.py`, and nowhere else. Every feature file this project
writes goes through `FIO.write`; every one it reads comes back through
`FIO.read`. In memory `id` stays an ordinary column, exactly as before — only
the on-disk layout changed, which is why adopting the convention touched one
line in each of six modules and nothing in the science.

Do not call `to_parquet` or `read_parquet` on a feature matrix directly. Two
ways of breaking the convention are completely silent.

### Trap 1 — a RangeIndex is stored as metadata, not as a column

The workshop's own snippet is

```python
pd.DataFrame(index=data_file['id'][:], data=features).to_parquet(path)
```

which is right for every file we hold. But if a dataset's ids happen to run
0, 1, 2 … N−1, `set_index` gives back a **RangeIndex**, and pandas stores a
RangeIndex as three numbers in the file metadata rather than as a column. The
ids are then not in the file at all — and a reader gets 0…N−1 back, which is
indistinguishable from having read them. `to_parquet(path, index=True)` forces
materialisation. `FIO.write` passes it, and builds the index with `pd.Index` so
the dtype survives the round trip as well.

None of our seven deliveries starts at id 0. That is luck, not protection, and
it is the reason the note went back into `format.md` for everyone else.

### Trap 2 — `columns=` does not give you back an `id` column

Once `id` is the index:

```python
pd.read_parquet(path, columns=["id", "snr"])    # succeeds
df["id"]                                        # KeyError
```

The id came back as the *index*. Against a naively-written file — trap 1, ids
never materialised — the same call instead raises `ArrowInvalid: No match for
FieldRef.Name(id)`. Two different failures from one mistake. `FIO.read` strips
`id` from the request, resets the index afterwards, and **always** returns the
id as a column whether or not the caller asked for it.

## Why `discover()` does not glob `*_features.parquet`

The shared features directory holds other people's work:
`mk_sample_resnet18_penultimate_features.parquet`,
`Full_DataSet_Spatial_Centre_Padded_VAE_features.parquet`, and so on. Globbing
the suffix and concatenating would silently mix two feature spaces. So
`discover()` globs `*_globular_features.parquet`, and the fallback that still
picks up pre-convention `<dataset>_features.parquet` files checks the schema for
columns no other feature set in the workshop has (`weak_label`, `feature_ok`,
`stamp_ok`) rather than trusting the name.

Note also that `<dataset>_<method>` cannot be split on `_`:
`mk_sample_hits_globular` is a four-token dataset and a one-token method. The
known method is peeled off the right.

## What we contributed back

The convention asks for a short description in the shared document. This is the
text added to `aug_2026_workshop/features/format.md` under
**GLOBULAR morphology features**; keep the two in step if the feature set
changes.

> One file per delivery — `lband_long`, `lband_short`, `mk_sample_hits`,
> `sband_long`, `sband_short`, `uhf_long`, `uhf_short` — plus a combined
> `all_globular_features.parquet`, which is **deduplicated on `id`** because
> 8,116 hits appear in both `mk_sample_hits.h5` and `lband_long.h5` and would
> otherwise be double-weighted.
>
> These are **not** deep features. They are the 13 hand-built features of
> GLOBULAR clustering (Jacobson-Bell et al. 2025, AJ 169:206) plus three extras
> of our own, computed directly from the stamp pixels — spectral skew,
> kurtosis, bimodality, the kurtosis-vs-bandwidth turning point, temporal skew,
> timeseries and spectrum spread, bandwidth at 1% of peak, and comb structure
> ("redness"), alongside frequency, |drift rate| and SNR from the catalogue.
>
> Two things to know before using them:
>
> - **Each raw column has a normalised twin `<col>_n`**, fitted **per file**, so
>   `f03_snr_n` is a within-file rank and is *not* comparable across files. The
>   raw columns are.
> - **They are not numerically comparable to published GLOBULAR values.** Our
>   spectral window is ~121–196 Hz against their 2.7 kHz — narrower than their
>   *minimum* sweep bandwidth — so `f07`/`f08` probe the shape of the line
>   rather than its neighbourhood, and `f08_turning_bw_hz` is unresolved for
>   ~72% of hits.
>
> Each row also carries provenance, the Track A boolean `flag_*` cuts,
> `stamp_ok` / `feature_ok`, and a weak label from the spatial filter:
> `weak_label` is 1 for hits seen in ≥32 beams, 0 for ≤2, −1 in between, with
> `group_id = obsid` for group-aware cross-validation. **`weak_label == 0` means
> "seen in ≤2 beams", not "verified clean"** — treat it as unlabelled, not as a
> negative.

## A caution about sharing the files

The convention governs the format, not permission. The per-hit tables carry sky
coordinates, `sourceName`, `obsid` and row indices, which remain SARAO /
Breakthrough Listen property — see the Licensing section of `AGENTS.md`.
Publishing the repository does not publish the data, and this document does not
change that.
