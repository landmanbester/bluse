#!/usr/bin/env python3
"""
feature_io.py -- the workshop's feature-file convention, in one place.

`aug_2026_workshop/features/format.md` fixes how a feature set is delivered so
that anyone in the workshop can cross-match it against anyone else's:

    <dataset>_<method>_features.parquet     `id` as the index, named `id`

Every feature file this project writes goes through `write()`, and every one it
reads comes back through `read()`. Nothing else should call `to_parquet` on a
feature matrix, because two details of the convention are easy to get wrong and
silent when you do.

THE TWO TRAPS
-------------
**1. On pandas 3, `set_index("id")` can stop the ids being written at all.**
pandas 3 infers a *RangeIndex* from any index whose values form an arithmetic
sequence -- 0,1,2,3,4 but equally 7,9,11,13,15 or 104,103,102 -- and it stores a
RangeIndex as `{"kind": "range", start, stop, step}` in the pandas metadata
instead of writing a column. The file's schema then has no `id` field:

    pandas 2.2.3   set_index -> Index        `id` column written
    pandas 3.0.5   set_index -> RangeIndex   `id` column NOT written

The values are not lost *to pandas* -- start/stop/step reproduce them exactly,
so a `pd.read_parquet` round trip looks perfect and this will never show up in
a pandas-only test. What is lost is the id **column**, and with it:

  - every non-pandas reader. `pq.read_table(path).column_names` is `['x']`;
    the same is true for polars, DuckDB, R/arrow, Spark. For a convention whose
    entire purpose is cross-matching between groups, that is the failure.
  - `read_parquet(path, columns=["id", ...])`, which raises `ArrowInvalid: No
    match for FieldRef.Name(id)`.
  - the dtype, which comes back int64 whatever went in.

`write()` passes `index=True`, which materialises the column whatever the index
type, and builds the index with `pd.Index` so the dtype survives too.

`format.md`'s own snippet -- `pd.DataFrame(index=data_file['id'][:],
data=features)` -- is **not** affected: constructing with `index=` gives a plain
`Index`, and the column is written on both pandas 2 and 3. `set_index` is the
one to avoid, and it is the obvious thing to write when the ids arrive as a
column, which is how ours do.

**2. `columns=` does not give you back an `id` column.** Once `id` is the
index, `pd.read_parquet(path, columns=["id", "snr"])` succeeds and returns a
frame with one column, `snr` -- the id came back as the *index*, so `df["id"]`
raises `KeyError` even though you asked for it by name. (Against a file written
the naive way, with the ids never materialised, the same call instead raises
`ArrowInvalid: No match for FieldRef.Name(id)`. Two different failures from one
mistake.) `read()` appends `id` to whatever column list it is given and resets
the index afterwards, so callers keep the lists they already had, the id always
comes back as a column, and the rest of the codebase did not have to change.

DATASET AND METHOD
------------------
`dataset` is the HDF5 delivery the hits came from (`lband_short`,
`mk_sample_hits`, ...). `method` names the extraction, and ours is
`globular` -- the 13 features of GLOBULAR clustering (Jacobson-Bell et al.
2025) plus three BLUSE-specific extras. See `features.py`.

The two are joined by an underscore and cannot be told apart by splitting on
one: `mk_sample_hits_globular` is a four-token dataset and a one-token method.
So `split_filename()` peels the *known* method off the right rather than
guessing. This is also why `discover()` globs `*_globular_features.parquet` and
not `*_features.parquet`: a colleague's `mk_sample_resnet18_penultimate_
features.parquet` dropped into the same directory is a different feature space
entirely, and concatenating it with ours would be nonsense.
"""

from __future__ import annotations

import os
from glob import glob

import pandas as pd
import pyarrow.parquet as pq

# The index every feature file carries, per format.md. This is the whole point
# of the convention: it is what lets a feature set be cross-matched against the
# catalogue, the scores, and anyone else's features.
INDEX = "id"

SUFFIX = "_features.parquet"

# Our extraction method's short name, as it appears in the filename.
METHOD = "globular"

# The dataset name of the combined table -- every delivery concatenated and
# deduplicated on `id`. Not an HDF5 file, but it is what Track E trains on and
# it obeys the same convention.
COMBINED = "all"


def filename(dataset: str, method: str = METHOD) -> str:
    """`lband_short` -> `lband_short_globular_features.parquet`."""
    return f"{dataset}_{method}{SUFFIX}"


def path(dataset: str, outdir: str, method: str = METHOD) -> str:
    return os.path.join(outdir, filename(dataset, method))


def split_filename(name: str, method: str = METHOD) -> str | None:
    """
    Dataset name out of a conforming filename, or None if it does not conform.

    Peels the known method off the right. Splitting on `_` cannot work --
    `mk_sample_hits` has three underscores of its own.
    """
    base = os.path.basename(name)
    if not base.endswith(SUFFIX):
        return None
    stem = base[: -len(SUFFIX)]
    tail = f"_{method}"
    if not stem.endswith(tail) or len(stem) <= len(tail):
        return None
    return stem[: -len(tail)]


def legacy_filename(dataset: str) -> str:
    """The pre-convention name, `<dataset>_features.parquet`.

    Kept only so an existing workspace can be migrated in place instead of
    re-extracted -- extraction over all 2,022,171 hits is about seven minutes
    and 830 MB of parquet, and none of it needs redoing.
    """
    return f"{dataset}{SUFFIX}"


def write(df: pd.DataFrame, dest: str) -> str:
    """
    Write one feature set in the delivery format: `id` as a materialised index.

    Takes a frame with `id` as an ordinary column, which is how the rest of the
    codebase carries it.
    """
    if INDEX not in df.columns:
        raise ValueError(
            f"cannot write {dest}: no `{INDEX}` column. Every feature file is "
            f"indexed by hit id so it can be cross-matched -- see format.md.")
    if df[INDEX].duplicated().any():
        n = int(df[INDEX].duplicated().sum())
        raise ValueError(
            f"cannot write {dest}: `{INDEX}` is not unique ({n:,} repeats). "
            f"An index that repeats cannot cross-match. The combined table is "
            f"deduplicated on `id` for exactly this reason; a per-file table "
            f"with duplicate ids means the delivery itself repeats a hit.")

    # pd.Index rather than set_index: on pandas 3, set_index infers a
    # RangeIndex from any arithmetic sequence of ids, and a RangeIndex is
    # stored as metadata rather than as a column -- so the file ends up with no
    # `id` field for any non-pandas reader. index=True materialises whatever we
    # hand it, so the two together are belt and braces. See the module
    # docstring, and https://github.com/landmanbester/bluse/issues/5.
    idx = pd.Index(df[INDEX].to_numpy(), name=INDEX)
    out = df.drop(columns=[INDEX]).set_axis(idx, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    out.to_parquet(dest, index=True)
    return dest


def read(src: str, columns=None) -> pd.DataFrame:
    """
    Read a feature set back with `id` as an ordinary column.

    `id` ALWAYS comes back, whether or not `columns` asked for it: it is the
    index, so it costs nothing, and a feature row without its id cannot be
    cross-matched with anything. Callers keep the column lists they already had.

    Also reads a pre-convention file, where `id` is an ordinary column. Naming
    `id` in the request happens to be right for both layouts -- pandas routes it
    to the index for a conforming file and to a column for a legacy one -- so
    one call covers the two without sniffing the schema first.
    """
    if columns is not None:
        columns = list(dict.fromkeys(list(columns) + [INDEX]))
    try:
        df = pd.read_parquet(src, columns=columns)
    except Exception as exc:
        if INDEX in str(exc):
            raise ValueError(
                f"{src} has no `{INDEX}`. It is not a BLUSE feature file -- "
                f"see format.md.") from exc
        raise
    if df.index.name == INDEX:
        df = df.reset_index()
    elif INDEX not in df.columns:
        raise ValueError(
            f"{src} has no `{INDEX}`, neither as an index nor as a column. It "
            f"is not a BLUSE feature file -- see format.md.")
    return df


def find(dataset: str, outdir: str, method: str = METHOD) -> str | None:
    """Locate one dataset's feature file, conforming name first, then legacy."""
    p = path(dataset, outdir, method)
    if os.path.exists(p):
        return p
    old = os.path.join(outdir, legacy_filename(dataset))
    return old if os.path.exists(old) else None


def discover(outdir: str, method: str = METHOD,
             include_combined: bool = False) -> dict[str, str]:
    """
    Every per-dataset feature file in `outdir`, as {dataset: path}.

    Conforming files win over legacy ones for the same dataset, so a
    half-migrated directory does not yield the same hits twice.
    """
    found: dict[str, str] = {}
    for p in sorted(glob(os.path.join(outdir, f"*_{method}{SUFFIX}"))):
        ds = split_filename(p, method)
        if ds and (include_combined or ds != COMBINED):
            found[ds] = p
    for p in sorted(glob(os.path.join(outdir, f"*{SUFFIX}"))):
        if not is_legacy(p, method):
            continue                       # conforming, or not ours at all
        ds = os.path.basename(p)[: -len(SUFFIX)]
        if ds in found:
            continue                       # conforming copy wins
        if include_combined or ds != COMBINED:
            found[ds] = p
    return found


def is_legacy(p: str, method: str = METHOD) -> bool:
    """A pre-convention file of OURS, as opposed to somebody else's features.

    The name alone cannot decide this. `mk_sample_resnet18_penultimate_
    features.parquet` -- a real file in the workshop's shared directory -- is
    indistinguishable by name from a legacy BLUSE table for a dataset called
    `mk_sample_resnet18_penultimate`. So we look at the schema instead, which
    costs a footer read and nothing else.
    """
    if not p.endswith(SUFFIX) or split_filename(p, method) is not None:
        return False
    return _is_ours(p)


# Columns no other feature set in this workshop has. Cheaper and steadier than
# matching on the f01..f13 block, which a future extraction could rename.
_MARKERS = ("weak_label", "feature_ok", "stamp_ok")


def _is_ours(p: str) -> bool:
    try:
        names = set(pq.read_schema(p).names)
    except Exception:
        return False
    return all(m in names for m in _MARKERS)


def legacy_files(outdir: str, method: str = METHOD) -> list[str]:
    """Our pre-convention feature files in `outdir`, whether or not migrated."""
    return [p for p in sorted(glob(os.path.join(outdir, f"*{SUFFIX}")))
            if is_legacy(p, method)]


def migrate(outdir: str, method: str = METHOD) -> tuple[list[tuple[str, str]],
                                                        list[str]]:
    """
    Rewrite pre-convention feature files under the conforming name and index.

    Reads and rewrites rather than renames, because the name is only half the
    convention -- the id index is the other half. Leaves the original in place;
    deleting 830 MB of someone's extraction is not this function's call, and
    that makes the operation safe to repeat.

    Returns the (source, destination) pairs it wrote, and the pre-convention
    files it left alone because a conforming twin already existed -- those are
    the ones the caller can offer to delete.
    """
    done, already = [], []
    for p in legacy_files(outdir, method):
        ds = os.path.basename(p)[: -len(SUFFIX)]
        dest = path(ds, outdir, method)
        if os.path.exists(dest):
            already.append(p)
            continue
        write(read(p), dest)
        done.append((p, dest))
    return done, already
