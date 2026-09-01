# Cluster Bench Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Cluster Bench an objective function and a reproducibility measure, and make cluster identities survive the batching loop, so that configuration choices become measurable instead of a matter of taste.

**Architecture:** Three new pure-library modules (`diagnostics.py`, `metrics.py`, `matching.py`) with no FastAPI and no argparse, imported by both existing entry points (`bench/app.py` and `track_b_cluster.py`) so a statistic is written once. Plus a `kind` field on the feature registry and the repository's first test suite, split into synthetic tests that gate commits and workspace-marked golden tests that do not.

**Tech Stack:** Python ≥3.10, numpy, pandas, scipy (`cluster.hierarchy`), scikit-learn (`HDBSCAN`, `NearestNeighbors`, `metrics`), FastAPI + Jinja2 + htmx 1.9.12 for the Bench, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-cluster-bench-review-design.md`

## Global Constraints

- **No data paths from `__file__`.** Everything user-supplied resolves through `bluse.paths`. Only `bench/static` and `bench/templates` are module-relative, because they ship in the wheel.
- **New CLI flags go through `paths.add_workspace_arg(p)`**, and any `--outdir` default is `None`, filled in after `parse_args`.
- **Filtering is non-destructive.** Cuts add `flag_*` columns; no pipeline stage drops rows.
- **New features return raw values.** `normalise()` owns the log / quantile / unit transforms.
- **Verify before claiming.** Quote the number or say you did not measure it.
- **A statistic used by both the Bench and the CLI is written once**, in a module both import.
- Python floor is `>=3.10`; scikit-learn floor is `>=1.3` (first release shipping `sklearn.cluster.HDBSCAN`). Do not raise either.
- Run everything with the project venv: `/home/bester/projects/bluse/.venv/bin/python`, or `uv run`.
- The workspace for manual checks is `aug_2026_workshop/`. `bluse-bench` auto-detects it from the repo root.

### Reference values (measured 2026-09-01, `sband_short`, 34,933 rows, `mcs=4 ms=8 epochs=8 batch=3000 scaling=robust`, f08 off, 15 features)

| quantity | value |
|---|---|
| `f02_abs_drift_n` | `n_distinct=42`, `max_tie_fraction=0.266`, `iqr_raw=5.954`, tie at −5.199 |
| `x03_channel_offset_n` | `share_global≈0.243`, `iqr_raw=0.042` |
| `f07_kurt_bw_corr_n` | `clip_frac≈0.010` (`sband_short`), `≈0.043` (`all`) |
| seed-only ARI, `eom` | composite 0.024, restricted 0.028, noise agreement 0.999 |
| seed-only ARI, `leaf` | composite 0.480, restricted 0.032, noise agreement 0.782 |
| `narrow_frac` (<1 MHz) | 0.776% (`eom`), 6.820% (`leaf`) |
| epoch trace, `eom` | 87.9% removed epoch 1, 12.0% epoch 2, 0.1% epoch 3, 0 thereafter |
| `weak_label` | −1: 7105, 0: 872, 1: 26956 → 31:1 among labelled |
| `mk_sample_hits` | 0% zero-drift, 53.7% overlap with `lband_long`, **no `weak_label==1` rows** |

---

## File Structure

| file | responsibility |
|---|---|
| `src/bluse/diagnostics.py` | NEW. Per-column audit of a feature matrix: ties, distinct values, IQRs, clip fraction, distance shares, flags. |
| `src/bluse/metrics.py` | NEW. Cluster-quality statistics, run-stability statistics, epoch trace. |
| `src/bluse/matching.py` | NEW. Cross-batch cluster matching by Ward linkage on centroids. |
| `src/bluse/features.py` | MOD. `kind` field on the registry; `f06` identity documented. |
| `src/bluse/track_b_cluster.py` | MOD. Consumes all three modules; new flags. |
| `src/bluse/bench/app.py` | MOD. Consumes all three; new routes; D-4 scaler fix. |
| `src/bluse/bench/templates/_controls.html` | MOD. Diagnostics rail, `cluster_selection_method` select. |
| `src/bluse/bench/templates/_results.html` | MOD. Metrics strip, epoch trace, families; D-1, D-2. |
| `src/bluse/bench/static/scatter.js` | MOD. Colour-by-value; D-3 grid fix. |
| `tests/conftest.py` | NEW. `workspace` marker and skip logic. |
| `tests/unit/fixtures.py` | NEW. Synthetic feature matrix with planted defects. |
| `tests/unit/test_*.py` | NEW. Commit-gating tests, no real data. |
| `tests/workspace/test_golden.py` | NEW. Golden values on real files, skipped without a workspace. |

---

## Task 1: Test scaffolding and the synthetic fixture

Everything downstream asserts against this fixture, so it comes first and its own properties are tested.

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`, `tests/unit/fixtures.py`
- Test: `tests/unit/test_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fixtures.synthetic_matrix(n=500, seed=0) -> (X, columns, kinds)` where `X` is `(n, 5) float64`, `columns` is `list[str]`, `kinds` is `dict[str, str]`; `fixtures.synthetic_labelled(n=600, seed=0) -> (labels, df)` where `labels` is `int32` and `df` is a `DataFrame` with `frequency`, `obsid`, `weak_label`; `fixtures.synthetic_centroid_space(seed=0) -> (labels, X)` with three well-separated families.

- [ ] **Step 1: Add the dev extra**

In `pyproject.toml`, after the `umap` extra and before `all`:

```toml
# The test suite. tests/unit is synthetic and runs anywhere; tests/workspace
# needs a features/ directory and is skipped without one.
dev = ["pytest>=7.4"]
```

and change the `all` line to:

```toml
all = ["bluse[bench,umap,dev]"]
```

Then add, at the end of the file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "workspace: needs a BLUSE workspace with features/; skipped without one",
]
```

- [ ] **Step 2: Write the fixture module**

Create `tests/unit/__init__.py` (empty) and `tests/unit/fixtures.py`:

```python
"""
Synthetic fixtures with PLANTED defects, so every diagnostic has a known answer.

Nothing here reads the workspace. The real feature matrices are gitignored
(aug_2026_workshop/features/ and data/ both are, 0 files tracked), so a suite
built on them runs on one machine and nowhere else -- and with no CI, it would
silently stop running the first time it broke.

The earlier draft of this suite used mk_sample_hits as its fixture. That file
cannot serve: it has 0% zero-drift (pre-filtered), so f02's tie fraction there
is 0.4525 rather than the 0.266 measured on sband_short; it is 53.7% duplicated
against lband_long, which makes it the worst available choice for anything
density-related; and it contains no weak_label == 1 rows at all, so every
label-based metric is degenerate on it.
"""

import numpy as np
import pandas as pd

# What each planted column is for. Keep these in sync with test assertions.
TIE_FRACTION = 0.30      # of rows sit at exactly 0.0 in "tie_col"
CLIP_FRACTION = 0.05     # of rows are pushed past +/-5 IQRs in "clip_col"
ORDINAL_LEVELS = 8       # distinct values in "ordinal_col"


def synthetic_matrix(n=500, seed=0):
    """
    A feature matrix whose defects are known by construction.

    Returns (X, columns, kinds).

      tie_col     30% of rows at exactly 0.0        -> max_tie_fraction == 0.30
      clip_col    5% of rows 20 sigma out           -> clip_frac == 0.05
      ordinal_col 8 evenly spaced levels            -> n_distinct == 8
      plain_a     clean normal                      -> no flag
      plain_b     clean normal                      -> no flag
    """
    rng = np.random.default_rng(seed)

    n_tie = int(round(n * TIE_FRACTION))
    tie = rng.normal(5.0, 1.0, n)
    tie[:n_tie] = 0.0

    n_clip = int(round(n * CLIP_FRACTION))
    clip = rng.normal(0.0, 1.0, n)
    clip[:n_clip] = 20.0

    ordinal = rng.integers(0, ORDINAL_LEVELS, n).astype(float)

    X = np.column_stack([tie, clip, ordinal,
                         rng.normal(0, 1, n), rng.normal(0, 1, n)])
    columns = ["tie_col", "clip_col", "ordinal_col", "plain_a", "plain_b"]
    kinds = {"tie_col": "continuous", "clip_col": "continuous",
             "ordinal_col": "ordinal", "plain_a": "continuous",
             "plain_b": "continuous"}
    return X, columns, kinds


def synthetic_labelled(n=600, seed=0):
    """
    A labelling with a KNOWN narrow-cluster share and a known enrichment.

    30 clusters of 20 hits each. Clusters 0-5 are "narrow": all their hits sit
    within 0.2 MHz. The rest are spread over 500 MHz. So 6/30 clusters and
    120/600 hits are narrow -> narrow_frac == 0.20 at narrow_mhz=1.0.

    weak_label is 0 for every hit in clusters 0-2 and 1 elsewhere, so those
    three clusters are perfectly enriched in the minority class.
    """
    rng = np.random.default_rng(seed)
    k, per = 30, n // 30
    labels = np.repeat(np.arange(k), per).astype(np.int32)

    freq = np.empty(n)
    for c in range(k):
        m = labels == c
        if c < 6:
            freq[m] = 1000.0 + c + rng.uniform(0, 0.2, m.sum())
        else:
            freq[m] = rng.uniform(1000.0, 1500.0, m.sum())

    weak = np.where(labels < 3, 0, 1).astype(np.int64)
    df = pd.DataFrame({
        "frequency": freq,
        "obsid": rng.integers(0, 4, n),
        "weak_label": weak,
    })
    return labels, df


def synthetic_centroid_space(seed=0):
    """
    Labels plus a feature matrix whose clusters fall into THREE families.

    9 clusters of 40 points. Clusters 0-2 sit near (0,0,0), 3-5 near (10,0,0),
    6-8 near (0,10,0). Any sane matching cuts this into exactly 3 families.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0., 0., 0.]] * 3 + [[10., 0., 0.]] * 3
                       + [[0., 10., 0.]] * 3)
    per = 40
    labels = np.repeat(np.arange(9), per).astype(np.int32)
    X = np.vstack([c + rng.normal(0, 0.3, (per, 3)) for c in centres])
    return labels, X
```

- [ ] **Step 3: Write conftest**

Create `tests/conftest.py`:

```python
"""
Two suites with different jobs.

tests/unit/       synthetic, deterministic, runs anywhere, gates commits.
tests/workspace/  golden values measured on real feature matrices. Catches a
                  regression in the science, but cannot gate a commit because
                  the data is not in the repository.
"""

import os

import pytest

from bluse import paths


def pytest_collection_modifyitems(config, items):
    """Skip workspace-marked tests when there is no features/ to read."""
    try:
        feat = paths.features_dir()
        have = os.path.isdir(feat) and any(
            f.endswith("_features.parquet") for f in os.listdir(feat))
    except Exception:
        have = False
    if have:
        return
    skip = pytest.mark.skip(reason="no BLUSE workspace with features/ found")
    for item in items:
        if "workspace" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 4: Write the fixture's own test**

Create `tests/unit/test_fixtures.py`:

```python
import numpy as np

from tests.unit import fixtures


def test_tie_column_has_the_planted_tie():
    X, cols, _ = fixtures.synthetic_matrix(n=500, seed=0)
    tie = X[:, cols.index("tie_col")]
    v, c = np.unique(tie, return_counts=True)
    assert c.max() / len(tie) == 0.30
    assert v[c.argmax()] == 0.0


def test_ordinal_column_has_the_planted_level_count():
    X, cols, _ = fixtures.synthetic_matrix(n=500, seed=0)
    assert len(np.unique(X[:, cols.index("ordinal_col")])) == 8


def test_labelled_fixture_has_the_planted_narrow_share():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    narrow = 0
    for c in np.unique(labels):
        f = df.frequency[labels == c]
        if f.max() - f.min() < 1.0:
            narrow += int((labels == c).sum())
    assert narrow / len(labels) == 0.20


def test_centroid_fixture_has_three_separated_families():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    assert len(np.unique(labels)) == 9
    cent = np.array([X[labels == c].mean(0) for c in np.unique(labels)])
    # within-family spread must be far smaller than between-family spread
    assert np.linalg.norm(cent[0] - cent[1]) < 2.0
    assert np.linalg.norm(cent[0] - cent[3]) > 8.0
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_fixtures.py -v`
Expected: 4 passed. If `bluse` fails to import, run `uv sync --extra all` first.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/
git commit -m "test: add pytest scaffolding and synthetic fixtures

The repository had no tests. tests/unit is synthetic and gates commits;
tests/workspace is golden-value and skipped without a workspace, because
features/ and data/ are both gitignored and no real-data suite is CI-able."
```

---

## Task 2: `kind` on the feature registry

**Files:**
- Modify: `src/bluse/features.py`
- Test: `tests/unit/test_features_kind.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FeatureSpec.kind: str`; `meta_feature(*columns, kind="continuous", description="")` and `stamp_feature(...)` accept `kind`; `all_columns(kind=None, feature_kind=None)`; `column_kinds() -> dict[str, str]` mapping every registered column to its kind.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_features_kind.py`:

```python
from bluse import features as F


def test_registered_features_default_to_continuous():
    kinds = F.column_kinds()
    assert kinds["f01_frequency"] == "continuous"


def test_declared_ordinal_columns_report_ordinal():
    kinds = F.column_kinds()
    assert kinds["f02_abs_drift"] == "ordinal"
    assert kinds["f12_bandwidth_hz"] == "ordinal"
    assert kinds["x02_time_occupancy"] == "ordinal"


def test_saturation_columns_are_flags():
    kinds = F.column_kinds()
    assert kinds["f08_turning_bw_saturated"] == "flag"


def test_every_registered_column_has_a_kind():
    kinds = F.column_kinds()
    for col in F.all_columns():
        assert col in kinds
        assert kinds[col] in {"continuous", "ordinal", "boolean", "flag"}
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_features_kind.py -v`
Expected: FAIL with `AttributeError: module 'bluse.features' has no attribute 'column_kinds'`

- [ ] **Step 3: Add the field**

In `src/bluse/features.py`, replace the `FeatureSpec` dataclass and the three registry helpers with:

```python
# What a column's values MEAN, which decides which diagnostics apply to it.
# A boolean has max_tie_fraction >= 0.5 by definition, so a tie threshold that
# does not know the kind misfires on it by construction. Defaults to
# "continuous" so every feature registered before this field existed -- and any
# a workshop participant has already written -- keeps working unchanged.
KINDS = ("continuous", "ordinal", "boolean", "flag")

# Columns whose kind is not "continuous". Everything absent is continuous.
# f02 is ORDINAL, not continuous: |driftRate| takes 42 values on an exact
# 0.010711 Hz/s lattice on sband_short -- the seticore Taylor-tree drift step --
# so it is a 42-level ordinal that normalise() currently rank-transforms as
# though it were a continuum. The lattice constant is per-file (6 distinct
# values across the eight files, spanning 5.26x), so driftSteps is a per-file
# index and NOT interchangeable with a physical drift rate.
COLUMN_KINDS = {
    "f02_abs_drift": "ordinal",
    "f12_bandwidth_hz": "ordinal",
    "x02_time_occupancy": "ordinal",
}


@dataclass
class FeatureSpec:
    name: str
    func: Callable
    columns: tuple[str, ...]
    kind: str            # "meta" or "stamp"
    description: str


def _register(kind, columns, description):
    def deco(fn):
        spec = FeatureSpec(fn.__name__, fn, tuple(columns), kind,
                           description or (fn.__doc__ or "").strip())
        if spec.name in REGISTRY:
            raise ValueError(f"duplicate feature name {spec.name}")
        REGISTRY[spec.name] = spec
        return fn
    return deco


def meta_feature(*columns, description=""):
    """Register a feature computed from catalogue metadata alone."""
    return _register("meta", columns, description)


def stamp_feature(*columns, description=""):
    """Register a feature computed from the stamp cube."""
    return _register("stamp", columns, description)


def all_columns(kind=None):
    return [c for s in REGISTRY.values()
            if kind is None or s.kind == kind for c in s.columns]


def column_kinds():
    """
    Map every registered column to its value kind.

    Saturation flags are "flag"; anything named in COLUMN_KINDS takes the kind
    declared there; everything else is "continuous".
    """
    out = {}
    for col in all_columns():
        if col.endswith("_saturated"):
            out[col] = "flag"
        else:
            out[col] = COLUMN_KINDS.get(col, "continuous")
    return out
```

Note that `FeatureSpec.kind` keeps its existing meaning (`"meta"` / `"stamp"`); the value kind is a per-*column* property and lives in `COLUMN_KINDS`, because one feature function may emit several columns of different kinds (`f07`/`f08` already do).

- [ ] **Step 4: Document the f06 identity**

In `src/bluse/features.py`, extend the `f06_bimodality` decorator's description string so it ends with:

```
                           "LATENT IDENTITY: f06 == (f04**2 + 1) / f05 "
                           "EXACTLY on raw values (max deviation 0.0 over "
                           "38,576 rows). It does NOT reach the metric space "
                           "today, because normalise() applies 'unit' to f04, "
                           "'log-unit' to f05 and 'none' to f06, and the "
                           "algebra does not survive those. Measured "
                           "consequence: f06_n is only R^2=0.710 predictable "
                           "from f04_n/f05_n, and its VIF is 7.0 against 53.1 "
                           "for f04 and 48.5 for f05. Changing the f05 or f06 "
                           "transform reactivates the identity silently.")
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_features_kind.py -v`
Expected: 4 passed.

- [ ] **Step 6: Check nothing else broke**

Run: `uv run python -c "from bluse import features as F; print(len(F.all_columns()), 'columns'); print(F.column_kinds()['f02_abs_drift'])"`
Expected: prints the column count and `ordinal`.

- [ ] **Step 7: Commit**

```bash
git add src/bluse/features.py tests/unit/test_features_kind.py
git commit -m "feat: declare a value kind per feature column

Tie diagnostics are meaningless without it: a boolean has
max_tie_fraction >= 0.5 by definition. Defaults to continuous so no
existing or participant-written feature needs an edit.

f02 is declared ordinal -- 42 levels on an exact 0.010711 Hz/s lattice.
Also documents the latent f06 == (f04^2+1)/f05 identity, which is exact
on raw values and currently does not reach the metric space."
```

---

## Task 3: `diagnostics.py`

**Files:**
- Create: `src/bluse/diagnostics.py`
- Test: `tests/unit/test_diagnostics.py`, `tests/workspace/test_golden.py`

**Interfaces:**
- Consumes: `features.column_kinds()` from Task 2.
- Produces: `diagnostics.audit(raw, columns, *, scaling="robust", kinds=None, min_samples=8, knn_sample=5000, seed=0) -> list[dict]`, one dict per column with keys `col`, `label`, `kind`, `n_distinct`, `max_tie_fraction`, `tie_value`, `iqr_raw`, `iqr_scaled`, `clip_frac`, `share_global`, `share_knn`, `flags` (a `list[str]`). Also `diagnostics.scale(X, how)` — the single shared scaler, moved here.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_diagnostics.py`:

```python
import numpy as np

from bluse import diagnostics as D
from tests.unit import fixtures


def _audit():
    X, cols, kinds = fixtures.synthetic_matrix(n=500, seed=0)
    rows = D.audit(X, cols, scaling="robust", kinds=kinds, min_samples=8)
    return {r["col"]: r for r in rows}


def test_recovers_the_planted_tie():
    a = _audit()
    assert a["tie_col"]["max_tie_fraction"] == 0.30
    assert a["tie_col"]["tie_value"] == 0.0


def test_recovers_the_planted_clip():
    a = _audit()
    assert a["clip_col"]["clip_frac"] == 0.05


def test_recovers_the_planted_level_count():
    a = _audit()
    assert a["ordinal_col"]["n_distinct"] == 8


def test_clean_columns_are_not_flagged():
    a = _audit()
    assert a["plain_a"]["flags"] == []
    assert a["plain_b"]["flags"] == []


def test_tie_and_clip_columns_are_flagged():
    a = _audit()
    assert "tie" in a["tie_col"]["flags"]
    assert "clip" in a["clip_col"]["flags"]


def test_tie_threshold_does_not_apply_to_flag_kind():
    X, cols, kinds = fixtures.synthetic_matrix(n=500, seed=0)
    kinds = dict(kinds, tie_col="flag")
    rows = D.audit(X, cols, scaling="robust", kinds=kinds, min_samples=8)
    tie = [r for r in rows if r["col"] == "tie_col"][0]
    assert "tie" not in tie["flags"]


def test_shares_sum_to_one():
    a = _audit()
    total = sum(r["share_global"] for r in a.values())
    assert abs(total - 1.0) < 1e-9


def test_robust_scaling_gives_unit_scaled_iqr():
    a = _audit()
    # robust divides each column by its own IQR, so the scaled IQR is 1.0
    # by construction for any column the clip does not reach.
    assert abs(a["plain_a"]["iqr_scaled"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bluse.diagnostics'`

- [ ] **Step 3: Write the module**

Create `src/bluse/diagnostics.py`:

```python
#!/usr/bin/env python3
"""
diagnostics.py -- per-column audit of a feature matrix.

Why this exists: the Bench's feature rail shows raw interquartile ranges, which
answer "how unequal are the columns before scaling". They cannot answer "what is
each column actually contributing to the distance HDBSCAN takes", and that is
where the defects are.

Measured on sband_short at Bench defaults, post-scaling distance shares run from
1.7% (f02_abs_drift) to 24.3% (x03_channel_offset) -- a 14x spread against an
equal share of 6.7%. `robust` scaling equalises the IQR, but HDBSCAN responds to
variance, and the IQR-to-variance ratio depends on distribution shape. So robust
misrepresents the distribution in BOTH directions: f02's IQR is inflated to
5.954 by a 26.6% tie sitting at the extreme (-5.199), and x03's is deflated to
0.042 by a tie near the centre.

Everything here is a pure function of (matrix, columns). No FastAPI, no
argparse, no workspace access -- both entry points import it.
"""

from __future__ import annotations

import numpy as np

CLIP = 5.0          # scale() clips robust z-scores here
TIE_FLAG = 0.10     # flag a continuous/ordinal column tying above this
CLIP_FLAG = 0.01    # flag a column landing on the clip more often than this


def scale(X, how):
    """
    Equalise how much each feature contributes to the Euclidean distance.

    The single shared implementation. bench/app.py and track_b_cluster.py both
    call this one, so the two paths cannot drift.

      "robust"    centre on the median, divide by the IQR, clip to +/-CLIP.
      "quantile"  rank-transform every feature to a uniform distribution.
      "none"      GLOBULAR's literal spec -- their transforms are applied
                  upstream in features.normalise(), so "none" means "the
                  paper's preprocessing and nothing further".
    """
    X = np.array(X, copy=True)
    if how == "robust":
        med = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
        return np.clip((X - med) / iqr, -CLIP, CLIP)
    if how == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, len(X)),
                                 subsample=200_000, random_state=0)
        return qt.fit_transform(X)
    return X


def _shares(Z, rng, n_pairs=4000):
    """Mean per-column share of the squared Euclidean distance, random pairs."""
    n = len(Z)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    d2 = (Z[a] - Z[b]) ** 2
    per = d2.mean(axis=0)
    tot = per.sum()
    return per / tot if tot > 0 else np.full(len(per), np.nan)


def _shares_knn(Z, k, rng, sample=5000):
    """
    The same, restricted to each point's k nearest neighbours.

    Not decoration. HDBSCAN responds to core distances and mutual reachability,
    both LOCAL, and the global random-pair share is a proxy for a local
    quantity. The approximation is worst exactly where ties are: a tied column
    contributes zero to every tie-tie pair, and tie-tie pairs are
    disproportionately likely to be mutual near neighbours because they already
    agree in that coordinate. The gap between share_global and share_knn is
    therefore itself the tie diagnostic, which is why both are reported.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(Z)
    if n > sample:
        idx = rng.choice(n, sample, replace=False)
        Zs = Z[idx]
    else:
        Zs = Z
    k = int(max(2, min(k, len(Zs) - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zs)
    _, ind = nn.kneighbors(Zs)
    a = np.repeat(np.arange(len(Zs)), k)
    b = ind[:, 1:].ravel()
    d2 = (Zs[a] - Zs[b]) ** 2
    per = d2.mean(axis=0)
    tot = per.sum()
    return per / tot if tot > 0 else np.full(len(per), np.nan)


def audit(raw, columns, *, scaling="robust", kinds=None, min_samples=8,
          knn_sample=5000, seed=0):
    """
    Audit every column of `raw` (n, n_features), pre-scaling.

    Returns one dict per column. `kinds` maps column name -> value kind; any
    column absent is treated as "continuous".
    """
    raw = np.asarray(raw, dtype=np.float64)
    kinds = kinds or {}
    rng = np.random.default_rng(seed)

    Z = scale(raw, scaling)
    q75r, q25r = np.percentile(raw, [75, 25], axis=0)
    iqr_raw = q75r - q25r
    q75s, q25s = np.percentile(Z, [75, 25], axis=0)
    iqr_scaled = q75s - q25s

    sg = _shares(Z, rng)
    sk = _shares_knn(Z, min_samples, rng, knn_sample)
    equal = 1.0 / max(len(columns), 1)

    out = []
    for i, col in enumerate(columns):
        x = raw[:, i]
        vals, counts = np.unique(x, return_counts=True)
        j = int(counts.argmax())
        tie = float(counts[j] / len(x))
        kind = kinds.get(col, "continuous")
        clip_frac = float((np.abs(Z[:, i]) >= CLIP - 1e-12).mean())

        flags = []
        if kind in ("continuous", "ordinal") and tie > TIE_FLAG:
            flags.append("tie")
        if clip_frac > CLIP_FLAG:
            flags.append("clip")
        # share_global carries the threshold because it is the statistic
        # actually measured. share_knn is reported but NOT thresholded: no
        # value for it has been measured yet, and inventing a bound now would
        # be exactly the unverified-claim pattern this work exists to correct.
        if np.isfinite(sg[i]):
            if sg[i] > 2 * equal:
                flags.append("share-high")
            elif sg[i] < 0.5 * equal:
                flags.append("share-low")

        out.append({
            "col": col,
            "label": col[:-2] if col.endswith("_n") else col,
            "kind": kind,
            "n_distinct": int(len(vals)),
            "max_tie_fraction": tie,
            "tie_value": float(vals[j]),
            "iqr_raw": float(iqr_raw[i]),
            "iqr_scaled": float(iqr_scaled[i]),
            "clip_frac": clip_frac,
            "share_global": float(sg[i]),
            "share_knn": float(sk[i]),
            "equal_share": float(equal),
            "flags": flags,
        })
    return out
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_diagnostics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Write the golden workspace test**

Create `tests/workspace/__init__.py` (empty) and `tests/workspace/test_golden.py`:

```python
"""
Golden values measured on the real feature matrices, 2026-09-01.

These catch a regression in the science. They cannot gate a commit, because
aug_2026_workshop/features/ is gitignored and the data is not in the repo.
Every number below states the file it was measured on.
"""

import os

import numpy as np
import pandas as pd
import pytest

from bluse import diagnostics as D
from bluse import features as F
from bluse import paths

pytestmark = pytest.mark.workspace


def _matrix(name):
    path = os.path.join(paths.features_dir(), f"{name}_features.parquet")
    if not os.path.exists(path):
        pytest.skip(f"no {name} feature matrix in this workspace")
    df = pd.read_parquet(path)
    df = df[df.feature_ok].reset_index(drop=True)
    cols = [c + "_n" for c in F.all_columns()
            if c + "_n" in df.columns and not c.endswith("_saturated")
            and not c.startswith("f08_")]
    X = df[cols].to_numpy(dtype=np.float64)
    good = np.isfinite(X).all(axis=1)
    return X[good], cols, df[good].reset_index(drop=True)


def _audit(name):
    X, cols, _ = _matrix(name)
    kinds = {c + "_n": k for c, k in F.column_kinds().items()}
    return {r["col"]: r for r in D.audit(X, cols, scaling="robust",
                                         kinds=kinds, min_samples=8)}


def test_f02_is_a_42_level_ordinal_on_sband_short():
    a = _audit("sband_short")["f02_abs_drift_n"]
    assert a["n_distinct"] == 42
    assert a["max_tie_fraction"] == pytest.approx(0.266, abs=0.005)
    assert a["tie_value"] == pytest.approx(-5.199, abs=0.01)
    assert a["iqr_raw"] == pytest.approx(5.954, rel=0.01)
    assert a["kind"] == "ordinal"
    assert "tie" in a["flags"]


def test_x03_is_over_weighted_on_sband_short():
    a = _audit("sband_short")["x03_channel_offset_n"]
    assert a["share_global"] == pytest.approx(0.243, abs=0.02)
    assert "share-high" in a["flags"]


def test_f02_is_under_weighted_on_sband_short():
    a = _audit("sband_short")["f02_abs_drift_n"]
    assert a["share_global"] == pytest.approx(0.017, abs=0.01)
    assert "share-low" in a["flags"]


def test_f07_clips_on_sband_short():
    a = _audit("sband_short")["f07_kurt_bw_corr_n"]
    assert a["clip_frac"] == pytest.approx(0.010, abs=0.004)
    assert "clip" in a["flags"]
```

- [ ] **Step 6: Run the golden tests**

Run: `uv run pytest tests/workspace -v` from the repo root (so the workspace auto-detects).
Expected: 4 passed. If they are skipped, the workspace was not found — check `uv run python -c "from bluse import paths; print(paths.banner())"`.

- [ ] **Step 7: Commit**

```bash
git add src/bluse/diagnostics.py tests/unit/test_diagnostics.py tests/workspace/
git commit -m "feat: add per-column feature diagnostics

Reports n_distinct, tie fraction, raw and scaled IQR, clip fraction, and
distance share both globally and on k-NN pairs. Flags are kind-aware.

share_global carries the flag threshold because it is the statistic
actually measured; share_knn is reported but deliberately not thresholded
until a value for it has been measured.

Also moves scale() here as the single shared implementation, so the Bench
and the CLI cannot drift apart on it."
```

---

## Task 4: `metrics.py` — cluster quality

**Files:**
- Create: `src/bluse/metrics.py`
- Test: `tests/unit/test_metrics_quality.py`

**Interfaces:**
- Consumes: `fixtures.synthetic_labelled` from Task 1.
- Produces: `metrics.quality(labels, df, *, narrow_mhz=(0.1, 1.0), n_perm=5, seed=0) -> dict` with keys `n`, `n_clusters`, `clustered_pct`, `largest_pct`, `median_size`, `narrow_frac`, `narrow_frac_null`, `narrow_enrichment`, `narrow_clusters`, `median_span_mhz`, `ami`, `enrichment`, and `narrow_frac_at` (a dict keyed by threshold).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_metrics_quality.py`:

```python
import numpy as np

from bluse import metrics as M
from tests.unit import fixtures


def test_narrow_frac_matches_the_planted_share():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    assert q["narrow_frac"] == 0.20
    assert q["narrow_clusters"] == 6


def test_narrow_frac_reported_at_both_thresholds():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    assert set(q["narrow_frac_at"]) == {0.1, 1.0}
    # the planted narrow clusters span 0.2 MHz, so they clear 1.0 but not 0.1
    assert q["narrow_frac_at"][1.0] == 0.20
    assert q["narrow_frac_at"][0.1] == 0.0


def test_permuted_labels_have_narrow_enrichment_near_one():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(labels)
    q = M.quality(shuffled, df, n_perm=9, seed=0)
    assert 0.2 < q["narrow_enrichment"] < 5.0


def test_enrichment_is_a_fraction_of_hits_not_clusters():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    # clusters 0,1,2 are perfectly confined: 3 of 30 clusters, 60 of 600 hits
    assert q["enrichment"] == 0.10


def test_basic_counts():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(labels, df)
    assert q["n_clusters"] == 30
    assert q["clustered_pct"] == 100.0
    assert q["median_size"] == 20
    assert q["largest_pct"] == 20 / 600 * 100


def test_all_noise_does_not_raise():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    q = M.quality(np.full(len(labels), -1, np.int32), df)
    assert q["n_clusters"] == 0
    assert q["clustered_pct"] == 0.0
    assert np.isnan(q["narrow_frac"])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_metrics_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bluse.metrics'`

- [ ] **Step 3: Write the quality half of the module**

Create `src/bluse/metrics.py`:

```python
#!/usr/bin/env python3
"""
metrics.py -- does this clustering configuration do anything worth having?

Cluster Bench exposed every knob and explained each one, but nothing on screen
said whether a configuration was BETTER, so tuning was by eye on a scatter plot
whose geometry the docs correctly warn against over-reading.

Two things this module deliberately does NOT do, both learned by measurement:

  - It never reports stability as one number. See stability().
  - It does not treat AMI against weak_label as an objective function. Among
    labelled rows the class balance is 26,956 : 872, i.e. 31:1, and AMI's whole
    observed range across every configuration tried is 0.0017-0.0048. It also
    ranks eom above leaf, the opposite of every other signal. It ships
    captioned, alongside a metric that works.

The headline is narrow_frac: the fraction of clustered hits sitting in clusters
that span less than a threshold in frequency. It needs no labels, has a dynamic
range of 0.1% to tens of percent (measured: 0.776% for eom, 6.820% for leaf on
sband_short), and rewards physical coherence -- which is what an RFI taxonomy,
the stated Track B deliverable, actually requires.
"""

from __future__ import annotations

import numpy as np

BH_Q = 0.05                 # Benjamini-Hochberg false-discovery rate
NARROW_MHZ = (0.1, 1.0)     # report the headline at two thresholds


def _sizes(labels):
    lab = labels[labels >= 0]
    if not len(lab):
        return np.array([], dtype=int)
    counts = np.bincount(lab)
    return counts[counts > 0]


def _narrow(labels, freq, thresh):
    """(hits in clusters narrower than `thresh`, count of such clusters)."""
    hits = 0
    n_cl = 0
    for c in np.unique(labels[labels >= 0]):
        m = labels == c
        f = freq[m]
        if f.max() - f.min() < thresh:
            hits += int(m.sum())
            n_cl += 1
    return hits, n_cl


def _enrichment(labels, weak):
    """
    Fraction of clustered hits in clusters significantly enriched in
    weak_label == 0, one-sided hypergeometric, Benjamini-Hochberg at BH_Q.

    Expressed in HITS, not in clusters, so it shares a scale with narrow_frac.
    A per-cluster percentage would compare 79 clusters against 2,127 on a
    statistic whose denominator is the cluster count -- the same
    non-comparability that made AMI useless here.

    Detection floor, at the measured global rate of 872/27,828 = 3.13%: a fully
    confined cluster of 6 gives p ~ 9.4e-10 and a fully confined cluster of 4
    gives p ~ 9.6e-7, both clearing BH at ~2,000 tests; a cluster of 3 gives
    p ~ 3.1e-5 against a threshold near 2.4e-5 and is marginal. At
    min_cluster_size = 4 only a FULLY confined cluster clears -- 3-of-4 gives
    p ~ 1.2e-4 and fails. So enrichment must not be compared across
    configurations with different min_cluster_size.
    """
    from scipy.stats import hypergeom

    known = weak != -1
    if known.sum() == 0:
        return float("nan")
    M_pop = int(known.sum())
    n_pos = int((weak == 0).sum())
    if n_pos == 0 or n_pos == M_pop:
        return float("nan")

    ids, ps, sizes = [], [], []
    for c in np.unique(labels[labels >= 0]):
        m = (labels == c) & known
        N = int(m.sum())
        if N == 0:
            continue
        k = int((weak[m] == 0).sum())
        ids.append(c)
        sizes.append(int((labels == c).sum()))
        ps.append(float(hypergeom.sf(k - 1, M_pop, n_pos, N)))
    if not ps:
        return float("nan")

    ps = np.asarray(ps)
    sizes = np.asarray(sizes)
    order = np.argsort(ps)
    m_tests = len(ps)
    thresh = (np.arange(1, m_tests + 1) / m_tests) * BH_Q
    passing = ps[order] <= thresh
    n_sig = int(np.max(np.nonzero(passing)[0]) + 1) if passing.any() else 0
    sig_hits = int(sizes[order][:n_sig].sum())
    return sig_hits / max(int((labels >= 0).sum()), 1)


def quality(labels, df, *, narrow_mhz=NARROW_MHZ, n_perm=5, seed=0):
    """
    Cluster-quality statistics for one labelling.

    `df` supplies `frequency` and, when present, `weak_label`. Rows of `df`
    correspond one-to-one with `labels`.
    """
    labels = np.asarray(labels)
    n = len(labels)
    freq = df["frequency"].to_numpy(dtype=np.float64)
    sizes = _sizes(labels)
    n_clustered = int((labels >= 0).sum())

    out = {
        "n": n,
        "n_clusters": int(len(sizes)),
        "clustered_pct": 100.0 * n_clustered / n if n else 0.0,
        "largest_pct": 100.0 * sizes.max() / n if len(sizes) else 0.0,
        "median_size": int(np.median(sizes)) if len(sizes) else 0,
    }

    if not len(sizes):
        out.update(narrow_frac=float("nan"), narrow_frac_null=float("nan"),
                   narrow_enrichment=float("nan"), narrow_clusters=0,
                   median_span_mhz=float("nan"), ami=float("nan"),
                   enrichment=float("nan"),
                   narrow_frac_at={t: float("nan") for t in narrow_mhz})
        return out

    thresholds = tuple(narrow_mhz)
    at = {}
    for t in thresholds:
        hits, n_cl = _narrow(labels, freq, t)
        at[t] = hits / n_clustered
        if t == thresholds[-1]:
            out["narrow_clusters"] = n_cl
    out["narrow_frac_at"] = at
    out["narrow_frac"] = at[thresholds[-1]]

    # Null for the headline, because every other metric here has one. Small
    # clusters are narrow by chance more often than large ones, and leaf's
    # median size is 6 against eom's 11, so part of the 8.8x could be
    # arithmetic. Permuting the label VECTOR preserves the cluster size
    # distribution exactly, which is the confound being controlled for.
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(max(1, n_perm)):
        hits, _ = _narrow(rng.permutation(labels), freq, thresholds[-1])
        nulls.append(hits / n_clustered)
    out["narrow_frac_null"] = float(np.mean(nulls))
    out["narrow_enrichment"] = (out["narrow_frac"] / out["narrow_frac_null"]
                                if out["narrow_frac_null"] > 0 else float("inf"))

    spans = [float(freq[labels == c].max() - freq[labels == c].min())
             for c in np.unique(labels[labels >= 0])]
    out["median_span_mhz"] = float(np.median(spans))

    if "weak_label" in df.columns:
        from sklearn.metrics import adjusted_mutual_info_score
        weak = df["weak_label"].to_numpy()
        known = weak != -1
        out["ami"] = (float(adjusted_mutual_info_score(weak[known],
                                                       labels[known]))
                      if known.sum() > 10 else float("nan"))
        out["enrichment"] = _enrichment(labels, weak)
    else:
        out["ami"] = float("nan")
        out["enrichment"] = float("nan")
    return out
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_metrics_quality.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bluse/metrics.py tests/unit/test_metrics_quality.py
git commit -m "feat: add cluster-quality metrics

narrow_frac -- fraction of clustered hits in clusters spanning under a
threshold -- is the headline: label-free, real dynamic range (0.776% for
eom against 6.820% for leaf on sband_short), and it rewards the physical
coherence an RFI taxonomy needs.

It gets a size-preserving permutation null, because the headline being the
only metric without one is the asymmetry that let AMI through unchallenged.

Enrichment is expressed in hits rather than clusters so it shares a scale
with narrow_frac, and its detection floor near min_cluster_size=4 is
documented."
```

---

## Task 5: `metrics.py` — stability and the epoch trace

**Files:**
- Modify: `src/bluse/metrics.py`
- Test: `tests/unit/test_metrics_stability.py`

**Interfaces:**
- Consumes: `metrics.quality` from Task 4.
- Produces: `metrics.stability(run_fn, seeds=(0,1,2,3,4)) -> dict` with keys `ari_composite`, `ari_restricted`, `noise_agreement`, `k_mean`, `k_min`, `k_max`, `n_seeds`; `metrics.epoch_trace(alive_after, n_total) -> list[dict]` with keys `epoch`, `alive`, `removed`, `pct_of_original`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_metrics_stability.py`:

```python
import numpy as np

from bluse import metrics as M


def test_deterministic_run_is_perfectly_stable():
    labels = np.repeat(np.arange(10), 20).astype(np.int32)
    s = M.stability(lambda seed: labels.copy(), seeds=(0, 1, 2))
    assert s["ari_composite"] == 1.0
    assert s["ari_restricted"] == 1.0
    assert s["noise_agreement"] == 1.0
    assert s["k_mean"] == 10.0


def test_high_noise_agreement_does_not_imply_stable_membership():
    """
    The regression test for the withdrawn '20x more reproducible' claim.

    A run that agrees perfectly about WHICH points are noise, while assigning
    the rest at random, must score high noise_agreement and low
    ari_restricted. Reporting a single composite ARI hides exactly this, and
    that is how leaf's composite 0.480 was mistaken for a stability advantage
    when its membership ARI was 0.032 against eom's 0.028.
    """
    n = 1000
    rng = np.random.default_rng(0)
    noise_mask = np.arange(n) % 2 == 0          # identical in every run

    def run_fn(seed):
        r = np.random.default_rng(seed)
        lab = r.integers(0, 50, n).astype(np.int32)
        lab[noise_mask] = -1
        return lab

    s = M.stability(run_fn, seeds=(0, 1, 2, 3))
    assert s["noise_agreement"] == 1.0
    assert s["ari_restricted"] < 0.05
    assert s["ari_composite"] > s["ari_restricted"]


def test_stability_returns_three_separate_numbers():
    labels = np.repeat(np.arange(10), 20).astype(np.int32)
    s = M.stability(lambda seed: labels.copy(), seeds=(0, 1))
    for key in ("ari_composite", "ari_restricted", "noise_agreement"):
        assert key in s


def test_epoch_trace_arithmetic():
    rows = M.epoch_trace([4240, 59, 17, 17], 34933)
    assert rows[0]["removed"] == 34933 - 4240
    assert rows[0]["pct_of_original"] == (34933 - 4240) / 34933 * 100
    assert rows[1]["removed"] == 4240 - 59
    assert rows[3]["removed"] == 0
    assert rows[3]["pct_of_original"] == 0.0
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_metrics_stability.py -v`
Expected: FAIL with `AttributeError: module 'bluse.metrics' has no attribute 'stability'`

- [ ] **Step 3: Append the implementation**

Append to `src/bluse/metrics.py`:

```python
def stability(run_fn, seeds=(0, 1, 2, 3, 4)):
    """
    How reproducible is this configuration across shuffle seeds?

    `run_fn(seed) -> labels`. Returns THREE separate numbers and never one
    scalar, because collapsing them is a measured error rather than a
    hypothetical one:

      ari_composite    pairwise ARI over the full label vectors
      ari_restricted   pairwise ARI over points clustered in BOTH runs
      noise_agreement  agreement on the binary (labels >= 0) vector

    sklearn's adjusted_rand_score treats -1 as an ordinary label, so a method
    that leaves half its points unclustered scores agreement for every
    within-noise pair. Measured: leaf scores composite 0.480 against eom's
    0.024 -- a 20x apparent advantage -- but restricted to cluster membership
    the two are 0.0316 and 0.0279, a 13% difference. The composite was
    measuring agreement about what is noise, not agreement about what belongs
    together.

    ari_restricted is the acceptance statistic. The larger reading of those
    numbers is that cluster membership is currently not reproducible under
    EITHER selection method -- both sit at the noise floor -- which is what
    matching (bluse.matching) exists to fix.
    """
    import itertools

    from sklearn.metrics import adjusted_rand_score

    seeds = tuple(seeds)
    runs = [np.asarray(run_fn(s)) for s in seeds]
    ks = [int(len(np.unique(r[r >= 0]))) for r in runs]

    comp, rest, noise = [], [], []
    for a, b in itertools.combinations(runs, 2):
        comp.append(adjusted_rand_score(a, b))
        both = (a >= 0) & (b >= 0)
        rest.append(adjusted_rand_score(a[both], b[both])
                    if both.sum() > 1 else float("nan"))
        noise.append(float(((a >= 0) == (b >= 0)).mean()))

    return {
        "n_seeds": len(seeds),
        "ari_composite": float(np.mean(comp)) if comp else float("nan"),
        "ari_restricted": float(np.nanmean(rest)) if rest else float("nan"),
        "noise_agreement": float(np.mean(noise)) if noise else float("nan"),
        "k_mean": float(np.mean(ks)),
        "k_min": int(min(ks)),
        "k_max": int(max(ks)),
    }


def epoch_trace(alive_after, n_total):
    """
    GLOBULAR Table 1 for our runs: how much each epoch actually removed.

    `alive_after[i]` is the number of hits still unclustered after epoch i+1.
    Measured on sband_short at defaults: epoch 1 removes 87.9%, epoch 2 12.0%,
    epoch 3 0.1%, and epochs 4-8 remove nothing at all -- so the epoch budget
    is spent in a single pass and five of the eight epochs are dead.
    """
    rows = []
    prev = n_total
    for i, alive in enumerate(alive_after, start=1):
        removed = prev - alive
        rows.append({
            "epoch": i,
            "alive": int(alive),
            "removed": int(removed),
            "pct_of_original": 100.0 * removed / n_total if n_total else 0.0,
        })
        prev = alive
    return rows
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_metrics_stability.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bluse/metrics.py tests/unit/test_metrics_stability.py
git commit -m "feat: add run-stability metrics and the epoch trace

stability() returns three numbers and never one. sklearn's ARI scores -1
as an ordinary label, so a method leaving half its points unclustered is
credited for every within-noise pair -- which is how leaf's composite
0.480 against eom's 0.024 was mistaken for a 20x stability advantage when
the membership figures are 0.0316 and 0.0279.

Includes a regression test that fails if the three are ever collapsed."
```

---

## Task 6: `matching.py`

**Files:**
- Create: `src/bluse/matching.py`
- Test: `tests/unit/test_matching.py`

**Interfaces:**
- Consumes: `fixtures.synthetic_centroid_space` from Task 1.
- Produces: `matching.match(labels, X, *, cut=None, quantile=50, method="ward") -> (family_ids, info)` where `family_ids` is `int32` aligned with `labels` (−1 wherever `labels` is −1) and `info` is a dict with keys `cut`, `n_clusters`, `n_families`, `nn_distances`, `method`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_matching.py`:

```python
import numpy as np

from bluse import matching
from tests.unit import fixtures


def test_recovers_the_planted_family_structure():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, info = matching.match(labels, X)
    assert info["n_clusters"] == 9
    assert info["n_families"] == 3


def test_families_group_the_right_clusters():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, _ = matching.match(labels, X)
    fam_of = {c: int(np.unique(fam[labels == c])[0]) for c in range(9)}
    assert fam_of[0] == fam_of[1] == fam_of[2]
    assert fam_of[3] == fam_of[4] == fam_of[5]
    assert fam_of[0] != fam_of[3]
    assert fam_of[0] != fam_of[6]


def test_noise_stays_noise():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    labels = labels.copy()
    labels[:10] = -1
    fam, _ = matching.match(labels, X)
    assert (fam[:10] == -1).all()


def test_is_deterministic():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    a, _ = matching.match(labels, X)
    b, _ = matching.match(labels, X)
    assert np.array_equal(a, b)


def test_explicit_cut_overrides_the_derived_one():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    _, info = matching.match(labels, X, cut=1000.0)
    assert info["cut"] == 1000.0
    assert info["n_families"] == 1


def test_single_cluster_is_a_single_family():
    labels = np.zeros(50, dtype=np.int32)
    X = np.random.default_rng(0).normal(size=(50, 3))
    fam, info = matching.match(labels, X)
    assert info["n_families"] == 1
    assert (fam == 0).all()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_matching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bluse.matching'`

- [ ] **Step 3: Write the module**

Create `src/bluse/matching.py`:

```python
#!/usr/bin/env python3
"""
matching.py -- group the clusters the batching loop mints separately.

HDBSCAN mints local ids 0..k-1 in EVERY batch, and the epoch loop runs many
batches, so one physical population becomes a fresh cluster id in every batch
and every epoch it appears in. That is not cosmetic. Measured on sband_short:
two runs of an identical configuration differing only in shuffle seed agree at
ARI 0.024, while dropping a whole feature at FIXED seed leaves ARI 0.75-0.89.
Batch membership, not feature geometry, is what decides which cluster a hit
lands in -- so cluster ids as they stand are batch artefacts, and matching is a
correctness fix rather than an enhancement.

Ward linkage on cluster centroids, cut at an explicit distance. Deterministic:
no perplexity, no seed, no embedding. Measured cost at D=15, scipy nn-chain
(O(n) memory):

    2,000 centroids (Bench, leaf)      0.04 s
   20,000 centroids                    6.78 s
  ~78,000 centroids (full `all`)      ~100 s   (extrapolated O(n^2))

so one exact implementation serves the Bench interactively and the CLI offline.
A k-NN-graph approximation was measured at 75.8 s for 80,000 -- slower than
exact Ward and strictly worse -- and is not used.

CAVEAT, and it belongs in any write-up of the first family taxonomy: Ward runs
on centroids in the SCALED feature space, and that space still carries the 14x
distance-share spread that the contribution-equalising scaling work has not yet
fixed. With x03_channel_offset at 24.3% and f07_kurt_bw_corr at 13.1%, families
are grouped substantially by channel offset and by a clipped correlation
coefficient. The first taxonomy is provisional and must be re-derived once that
scaling lands.
"""

from __future__ import annotations

import numpy as np


def centroids(labels, X):
    """(cluster ids, their centroids in the columns of X)."""
    ids = np.unique(labels[labels >= 0])
    C = np.vstack([X[labels == c].mean(axis=0) for c in ids]) if len(ids) \
        else np.zeros((0, X.shape[1]))
    return ids, C


def derive_cut(C, quantile=50):
    """
    A distance cut read off the centroid nearest-neighbour distribution.

    Not a hardcoded constant: the per-file drift lattice alone spans 5.26x
    across our eight files (uhf_long 0.00204 Hz/s, sband_short 0.01071), so a
    value tuned on one file is wrong on another.
    """
    from sklearn.neighbors import NearestNeighbors
    if len(C) < 2:
        return 0.0
    nn = NearestNeighbors(n_neighbors=2).fit(C)
    d, _ = nn.kneighbors(C)
    return float(np.percentile(d[:, 1], quantile))


def match(labels, X, *, cut=None, quantile=50, method="ward"):
    """
    Group clusters into families.

    Returns (family_ids, info). `family_ids` is aligned with `labels` and is
    -1 wherever `labels` is -1.

    method="ward"  exact Ward linkage on centroids. The default.
    method="tsne"  GLOBULAR's own route -- PCA to 6, t-SNE (perplexity 40,
                   early exaggeration 4), HDBSCAN on the 2-D embedding. Kept
                   as a reproduction path, not a default: the paper's own
                   health warnings about t-SNE are the argument for having a
                   deterministic alternative.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    labels = np.asarray(labels)
    X = np.asarray(X, dtype=np.float64)
    ids, C = centroids(labels, X)

    fam = np.full(len(labels), -1, dtype=np.int32)
    info = {"cut": float(cut) if cut is not None else 0.0,
            "n_clusters": int(len(ids)), "n_families": 0,
            "nn_distances": [], "method": method}
    if len(ids) == 0:
        return fam, info
    if len(ids) == 1:
        fam[labels == ids[0]] = 0
        info["n_families"] = 1
        return fam, info

    info["nn_distances"] = _nn_distances(C)

    if method == "tsne":
        assign = _match_tsne(C)
    else:
        if cut is None:
            cut = derive_cut(C, quantile)
        info["cut"] = float(cut)
        Z = linkage(C, method="ward")
        # criterion="distance" with t<=0 would put every cluster in its own
        # family; guard so a degenerate cut does not look like a real answer.
        assign = fcluster(Z, t=max(float(cut), 1e-12), criterion="distance")

    # Map cluster id -> family id, vectorised. ids is sorted, so searchsorted
    # gives each hit's position in `ids` in one pass.
    assign = np.asarray(assign, dtype=np.int32)
    clustered = labels >= 0
    pos = np.searchsorted(ids, labels[clustered])
    fam[clustered] = assign[pos] - assign.min()
    info["n_families"] = int(len(np.unique(assign)))
    return fam, info


def _nn_distances(C):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(C)
    d, _ = nn.kneighbors(C)
    return [float(v) for v in d[:, 1]]


def _match_tsne(C):
    """GLOBULAR's route, for reproduction. Not the default; not deterministic."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_pc = min(6, C.shape[1], len(C) - 1)
    P = PCA(n_components=n_pc, random_state=0).fit_transform(C)
    perp = float(min(40, max(5, len(C) // 4)))
    E = TSNE(n_components=2, perplexity=perp, early_exaggeration=4,
             random_state=0, init="pca").fit_transform(P)
    lab = HDBSCAN(min_cluster_size=2, n_jobs=-1).fit_predict(E)
    # t-SNE noise gets its own family each, so the count stays honest.
    out = lab.copy()
    nxt = int(lab.max()) + 1 if (lab >= 0).any() else 0
    for i in np.nonzero(lab < 0)[0]:
        out[i] = nxt
        nxt += 1
    return out
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_matching.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bluse/matching.py tests/unit/test_matching.py
git commit -m "feat: add cross-batch cluster matching by Ward linkage

Cluster ids are batch artefacts: reshuffling at identical settings gives
ARI 0.024, while dropping a whole feature at fixed seed gives 0.75-0.89.
Families are the level at which per-batch id minting should cancel.

Exact Ward on centroids, deterministic, with the cut derived from the
centroid nearest-neighbour distribution rather than hardcoded -- the
per-file drift lattice alone spans 5.26x. Measured 0.04s at 2k centroids
and ~100s extrapolated at 78k, so one implementation serves both paths.
GLOBULAR's t-SNE route is kept behind method='tsne' for reproduction."
```

---

## Task 7: `cluster_selection_method`, the epoch trace, and the invariants

The first task that touches both entry points. `cluster()` starts returning its epoch trace, and `run_hdbscan` grows a parameter.

**Files:**
- Modify: `src/bluse/bench/app.py` (`run_hdbscan`, `cluster`, `do_cluster`), `src/bluse/track_b_cluster.py` (`run_hdbscan`, `cluster_epochs`, `main`)
- Modify: `src/bluse/bench/templates/_controls.html`
- Test: `tests/unit/test_invariants.py`

**Interfaces:**
- Consumes: `metrics.epoch_trace` from Task 5, `diagnostics.scale` from Task 3.
- Produces: `bench.app.run_hdbscan(X, mcs, ms, method="eom")`; `bench.app.cluster(...) -> (labels, X, origin, trace)` where `trace` is `list[int]` of survivors after each epoch; the form field `csm`.

- [ ] **Step 1: Write the failing invariant tests**

Create `tests/unit/test_invariants.py`:

```python
"""
The five invariants. Three of the defects in AGENTS.md gotcha 9 presented
identically -- "the bench looks insensitive to every knob except
min_cluster_size" -- and these separate them.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from bluse import diagnostics as D
from bluse.bench import app


def _matrix(n=900, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0., 0.], [6., 0.], [0., 6.]])
    X = np.vstack([c + rng.normal(0, 0.6, (n // 3, 2)) for c in centres])
    # a wildly unequal second block, so scaling has something to equalise
    return np.column_stack([X, rng.normal(0, 40.0, len(X))])


class _DS:
    """Minimal stand-in for bench.app.Dataset -- cluster() uses .raw/.columns."""
    def __init__(self, X):
        self.raw = X
        self.columns = [f"c{i}_n" for i in range(X.shape[1])]


def test_invariant_1_cluster_ids_are_globally_unique():
    ds = _DS(_matrix())
    labels, _, origin, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                       4, 8, 3, 200, 0)
    ids = np.unique(labels[labels >= 0])
    assert len(ids) == len(origin)
    assert set(int(i) for i in ids) == set(origin)


def test_invariant_2_changing_scaling_changes_the_labels():
    """
    THE SEED IS PINNED, and that is the whole test.

    At a free seed, two runs of an IDENTICAL configuration score ARI 0.024, so
    `ARI < 1.0` passes on shuffle noise even if scale() were stubbed to return
    its input -- the exact bug class this exists to catch. At a fixed seed a
    genuine no-op scores exactly 1.0 and the assertion bites.
    """
    ds = _DS(_matrix())
    a, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs", 4, 8, 3, 200, 0)
    b, _, _, _ = app.cluster(ds, ds.columns, "none", "epochs", 4, 8, 3, 200, 0)
    assert adjusted_rand_score(a, b) < 1.0


def test_invariant_3_reported_count_matches_the_labels():
    ds = _DS(_matrix())
    labels, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                  4, 8, 3, 200, 0)
    summary = app.summarise_basic(labels)
    assert len(summary) == len(np.unique(labels[labels >= 0]))


def test_invariant_4_no_continuous_column_ties_above_half():
    from tests.unit import fixtures
    X, cols, kinds = fixtures.synthetic_matrix(n=500, seed=0)
    for r in D.audit(X, cols, scaling="robust", kinds=kinds):
        if r["kind"] in ("continuous", "ordinal"):
            assert r["max_tie_fraction"] <= 0.5, r["col"]


def test_invariant_5_restricted_ari_band_is_recorded():
    """
    Smoke test, deliberately wide.

    ari_restricted has meaningful variance of its own at the values we see, so
    a band recorded from a single draw would be flaky in the way that trains
    people to ignore a suite. This asserts only that the statistic is defined
    and in range; the real bands live in tests/workspace.
    """
    from bluse import metrics as M
    ds = _DS(_matrix())

    def run_fn(seed):
        lab, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                   4, 8, 3, 200, seed)
        return lab

    s = M.stability(run_fn, seeds=(0, 1, 2))
    assert 0.0 <= s["ari_restricted"] <= 1.0
    assert s["k_min"] >= 1


def test_leaf_produces_more_clusters_than_eom():
    ds = _DS(_matrix())
    e, _, _, _ = app.cluster(ds, ds.columns, "robust", "single",
                             4, 8, 1, 3000, 0, method="eom")
    l, _, _, _ = app.cluster(ds, ds.columns, "robust", "single",
                             4, 8, 1, 3000, 0, method="leaf")
    assert len(np.unique(l[l >= 0])) >= len(np.unique(e[e >= 0]))


def test_epoch_trace_is_returned():
    ds = _DS(_matrix())
    _, _, _, trace = app.cluster(ds, ds.columns, "robust", "epochs",
                                 4, 8, 3, 200, 0)
    assert len(trace) >= 1
    assert all(isinstance(t, int) for t in trace)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/unit/test_invariants.py -v`
Expected: FAIL — `cluster()` returns 3 values, not 4, and `summarise_basic` does not exist.

- [ ] **Step 3: Update `bench/app.py`**

Replace `run_hdbscan` with:

```python
def run_hdbscan(X, mcs, ms, method="eom"):
    """
    One HDBSCAN pass.

    cluster_selection_epsilon is deliberately absent. sklearn's epsilon_search
    (_tree.pyx:606) compares epsilon against the RECIPROCAL of a leaf's split
    distance, not the distance. Leaf splits in this feature space are >~1.7, so
    1/d is ~0.55: every epsilon below that leaves the leaf set bit-identical to
    epsilon=0, and every epsilon above it reaches traverse_upwards, which
    assigns length-1 arrays into scalar cdefs and raises TypeError. The no-op
    region and the crash region tile the domain, so the knob is gone rather
    than re-ranged. Measured: eps 0.0/0.05/0.18/0.5 give ARI 1.000 against each
    other; 5.0 raises on 14/14 batches.

    `method` selects EOM or leaf extraction. With allow_single_cluster=False,
    EOM makes one stability comparison at the root of the condensed tree:
    either the root's two children win (k=2) or it descends to the leaves
    (k~200), with nothing in between BY CONSTRUCTION. Raising min_samples
    biases which side of that knife edge you land on but does not remove it.
    leaf takes the condensed tree's leaves directly, never makes the root
    comparison, and is not bistable. Measured on sband_short: eom gives 72
    clusters at 99.9% clustered with 87.9% removed in epoch 1; leaf gives 2,162
    at 50.3% clustered with 12.9% removed in epoch 1 -- i.e. leaf restores a
    working epoch loop -- and raises the narrow-cluster share from 0.776% to
    6.820%.
    """
    # Built OUTSIDE the try: an unsupported keyword raises TypeError at
    # construction, and the except below would otherwise swallow it and hand
    # back an all-noise result that looks like a legitimate clustering.
    est = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                  cluster_selection_method=method,
                  copy=True,
                  n_jobs=-1)
    try:
        return est.fit_predict(X)
    except ValueError:
        return np.full(len(X), -1, dtype=np.int32)
```

In `cluster()`, change the signature and thread `method` plus the trace:

```python
def cluster(ds, cols, scaling, mode, mcs, ms, epochs, batch, seed,
            method="eom"):
    use = [ds.columns.index(c) for c in cols]
    X = scale(ds.raw[:, use], scaling)
    n = len(X)
    labels = np.full(n, -1, dtype=np.int32)
    origin = {}
    trace = []

    if mode == "single":
        labels = run_hdbscan(X, mcs, ms, method).astype(np.int32)
        for c in np.unique(labels[labels >= 0]):
            origin[int(c)] = (1, 0)
        trace = [int((labels < 0).sum())]
    else:
        alive = np.arange(n)
        rng = np.random.default_rng(seed)
        next_id = 0
        for ep in range(1, epochs + 1):
            rng.shuffle(alive)
            survivors = []
            for s in range(0, len(alive), batch):
                b = alive[s:s + batch]
                if len(b) < mcs * 2:
                    survivors.append(b)
                    continue
                lab = run_hdbscan(X[b], mcs, ms, method)
                hit = lab >= 0
                if hit.any():
                    # HDBSCAN mints local ids 0..k-1 in EVERY batch. A running
                    # offset keeps every batch's clusters distinct; offsetting
                    # by epoch alone fused batch 0's cluster 3 with batch 7's.
                    labels[b[hit]] = lab[hit] + next_id
                    for c in range(next_id, next_id + int(lab[hit].max()) + 1):
                        origin[c] = (ep, s // batch)
                    next_id += int(lab[hit].max()) + 1
                survivors.append(b[~hit])
            alive = np.concatenate(survivors) if survivors else np.array([], int)
            trace.append(int(len(alive)))
            if len(alive) < mcs * 2:
                break
        labels[alive] = -1
    return labels, X, origin, trace
```

Add, next to `summarise`:

```python
def summarise_basic(labels):
    """Cluster ids and sizes only. Used where no provenance frame is at hand."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    return pd.DataFrame({"cluster": ids.astype(int), "n": counts.astype(int)})
```

Replace the module-level `scale` with a re-export so there is one implementation:

```python
from ..diagnostics import scale          # single shared implementation
```

and delete the old `def scale(...)` body from `app.py`.

In `do_cluster`, add `csm` to the params dict, after `mode`:

```python
             csm=form.get("csm", "eom"),
```

and pass it through, changing the `cluster(...)` call to:

```python
        labels, _, origin, trace = cluster(ds, cols, p["scaling"], p["mode"],
                                           p["mcs"], p["ms"], p["epochs"],
                                           p["batch"], p["seed"], p["csm"])
```

Store the trace on the `Run`: add `trace: list = field(default_factory=list)` to the `Run` dataclass, and pass `trace` when constructing it.

- [ ] **Step 4: Update `track_b_cluster.py`**

In `run_hdbscan`, add the method parameter and pass it:

```python
    est = HDBSCAN(min_cluster_size=args.min_cluster_size,
                  min_samples=args.min_samples,
                  cluster_selection_method=getattr(
                      args, "cluster_selection_method", "eom"),
                  copy=True,
                  n_jobs=-1)
```

Make `cluster_epochs` return its trace as well. Its current signature is
`cluster_epochs(df, args) -> (labels, cols, epoch_of)` (line ~192, returning at
line ~236). Add a `trace` list, append `len(alive)` at the end of each epoch
loop beside the existing `pct` print, and change the return to
`(labels, cols, epoch_of, trace)`. Update the single call site in `main`
(line ~391) from

```python
        labels, cols, _ = cluster_epochs(df, args)
```

to

```python
        labels, cols, _, trace = cluster_epochs(df, args)
```

and give the `single` branch a matching shape:

```python
        labels, cols = cluster_single(df, args)
        trace = [int((labels < 0).sum())]
```

In `main`, after the `--scaling` argument:

```python
    p.add_argument("--cluster-selection-method", choices=["eom", "leaf"],
                   default="eom",
                   help="EOM (default) or leaf extraction from the condensed "
                        "tree. leaf is not bistable and restores a working "
                        "epoch loop; measured on sband_short it raises the "
                        "narrow-cluster share from 0.78%% to 6.82%%. The "
                        "default stays eom until cross-batch matching has "
                        "been shown to make leaf's ~2000 clusters readable")
```

- [ ] **Step 5: Add the Bench control**

In `_controls.html`, immediately after the `mode` select's closing `</label>`:

```html
  <label class="field"><span>cluster_selection_method</span>
    <select name="csm">
      <option value="eom" selected>eom — stability, few clusters</option>
      <option value="leaf">leaf — condensed-tree leaves, many</option>
    </select>
  </label>
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `uv run pytest tests/unit -v`
Expected: all pass, including the 7 in `test_invariants.py`.

- [ ] **Step 7: Check the Bench still starts and clusters**

```bash
uv run bluse-bench --port 8765 &
sleep 4
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/
kill %1
```
Expected: `200`.

- [ ] **Step 8: Commit**

```bash
git add src/bluse/bench/app.py src/bluse/track_b_cluster.py \
        src/bluse/bench/templates/_controls.html tests/unit/test_invariants.py
git commit -m "feat: add cluster_selection_method and return the epoch trace

leaf is available in both entry points; eom stays the default until
matching shows leaf's ~2000 clusters are readable.

Adds the five invariants. Invariant 2 pins the seed -- at a free seed it
passed on shuffle noise even against a stubbed scale(), which is a test
credited as coverage that could not fail.

scale() is now imported from bluse.diagnostics in both paths."
```

---

## Task 8: the Bench diagnostics rail, and the D-4 scaler fix

**Files:**
- Modify: `src/bluse/bench/app.py` (`load_dataset`, `pick_dataset`)
- Modify: `src/bluse/bench/templates/_controls.html`
- Modify: `src/bluse/bench/static/app.css`

**Interfaces:**
- Consumes: `diagnostics.audit` from Task 3, `features.column_kinds` from Task 2.
- Produces: `Dataset.scaler_stats: dict` carrying the full-population median and IQR; the `features` list handed to `_controls.html` now carries every `audit()` key.

- [ ] **Step 1: Fix the scaler fit (D-4)**

In `load_dataset`, compute the scaling statistics on the **full** column set before sampling. Replace the sampling block with:

```python
    # D-4: fit the scaler on the FULL population, then sample. GLOBULAR's
    # requirement that scaling be global and pre-batching is already satisfied
    # upstream -- features.normalise() fits its transforms globally at
    # extraction time -- so this is the second stage only.
    #
    # Be honest about what this fixes: measured, a 35k-row IQR matches the
    # 1,281,878-row population to better than 1.1% on 14 of 15 columns, worst
    # 6.4%. It is NOT why a Bench configuration fails to reproduce in
    # bluse-cluster. That is because the Bench clusters 35k rows and the CLI
    # clusters 1.28M, and because two runs of one configuration at different
    # seeds already agree at only ARI 0.024.
    med_full = np.median(X, axis=0)
    q75f, q25f = np.percentile(X, [75, 25], axis=0)
    scaler_stats = {"median": med_full, "iqr": q75f - q25f}

    if sample and len(df) > sample:
        idx = np.sort(np.random.default_rng(seed).choice(len(df), sample,
                                                         replace=False))
        df, X = df.iloc[idx].reset_index(drop=True), X[idx]
```

Add `scaler_stats: dict = field(default_factory=dict)` to the `Dataset` dataclass and pass it in the constructor.

- [ ] **Step 2: Replace the rail computation in `pick_dataset`**

Replace the `q75r, q25r = ...` block and the `feats` loop with:

```python
    kinds = {c + "_n": k for c, k in F.column_kinds().items()}
    rows = diagnostics.audit(ds.raw, ds.columns, scaling="robust",
                             kinds=kinds, min_samples=8)

    sat = {}
    if "f08_turning_bw_saturated" in ds.df.columns:
        sat["f08_turning_bw_hz_n"] = float(
            ds.df.f08_turning_bw_saturated.mean() * 100)

    iqr_max = max((r["iqr_raw"] for r in rows), default=1.0) or 1.0
    feats = []
    for r in rows:
        feats.append({
            **r,
            "iqr_rel": r["iqr_raw"] / iqr_max,
            "saturated": sat.get(r["col"]),
            "on": not r["col"].startswith("f08_"),
            "share_pct": 100.0 * r["share_global"],
            "share_knn_pct": 100.0 * r["share_knn"],
            "equal_pct": 100.0 * r["equal_share"],
            "tie_pct": 100.0 * r["max_tie_fraction"],
            "clip_pct": 100.0 * r["clip_frac"],
        })
```

and add `from .. import diagnostics` to the imports.

- [ ] **Step 3: Render the diagnostics in the rail**

In `_controls.html`, replace the explanatory paragraph and the feature loop with:

```html
  <p style="font-family:var(--mono);font-size:10px;color:var(--dimmer);margin:0 0 8px">
    bar = raw spread (IQR) before scaling. <b>share</b> = what the column
    actually contributes to the distance HDBSCAN takes, against an equal share
    of {{ '%.1f'|format(features[0].equal_pct if features else 0) }}%.
    <b>tie</b> = largest repeated value. <b>clip</b> = fraction landing on
    &plusmn;5.
    <br><span style="color:var(--dimmer)">The two share flags are one
    observation, not two: shares sum to 1, so a column at 24% mechanically
    depresses every other toward the lower bound.</span>
  </p>
  {% for f in features %}
  <label class="feat {% if f.flags %}flagged{% endif %}">
    <input type="checkbox" name="feat" value="{{ f.col }}" {% if f.on %}checked{% endif %}>
    <span class="name">{{ f.label }}</span>
    <span class="iqr">{{ '%.3f'|format(f.iqr_raw) }}{% if f.saturated %} <span class="flag">SAT {{ f.saturated|round|int }}%</span>{% endif %}</span>
    <span class="bar"><i style="width: {{ (f.iqr_rel * 100)|round(1) }}%"></i></span>
    <span class="diag">
      share {{ '%.1f'|format(f.share_pct) }}%
      <span class="dim">/ knn {{ '%.1f'|format(f.share_knn_pct) }}%</span>
      · {{ f.n_distinct }} vals
      {% if f.max_tie_fraction > 0.01 %}· tie {{ '%.0f'|format(f.tie_pct) }}%{% endif %}
      {% if f.clip_frac > 0.0001 %}· clip {{ '%.1f'|format(f.clip_pct) }}%{% endif %}
      {% for fl in f.flags %}<span class="flag">{{ fl }}</span>{% endfor %}
    </span>
  </label>
  {% endfor %}
```

- [ ] **Step 4: Style the new row**

Append to `src/bluse/bench/static/app.css`:

```css
/* Diagnostics line under each feature in the rail. */
.feat .diag {
  grid-column: 1 / -1;
  font-family: var(--mono);
  font-size: 9.5px;
  color: var(--dimmer);
  padding-left: 20px;
  line-height: 1.5;
}
.feat .diag .dim { opacity: 0.6; }
.feat.flagged { background: rgba(232, 115, 74, 0.06); }
.feat.flagged .name { color: #e8a04a; }
```

- [ ] **Step 5: Verify by eye**

```bash
uv run bluse-bench --port 8765 &
sleep 4
curl -s -X POST http://127.0.0.1:8765/dataset \
     -d "file=sband_short&sample=35000&seed=0" | grep -o 'share [0-9.]*%' | head -20
kill %1
```
Expected: a `share ...%` per feature. `x03_channel_offset` should read ≈24%, `f02_abs_drift` ≈1.7%.

- [ ] **Step 6: Commit**

```bash
git add src/bluse/bench/app.py src/bluse/bench/templates/_controls.html \
        src/bluse/bench/static/app.css
git commit -m "feat: show per-feature diagnostics in the Bench rail

Distance share (global and k-NN), distinct values, tie fraction and clip
fraction alongside the existing raw-IQR bar, with kind-aware flags. The
raw bar answers 'how unequal are the columns before scaling'; it cannot
answer what each contributes to the distance, which is where the defects
are.

Also fits the Bench scaler on the full population rather than the 35k
sample (D-4), with a comment recording that this is correct but is NOT
why the Bench and the CLI disagree."
```

---

## Task 9: the results strip — metrics, epoch trace, D-1 and D-2

**Files:**
- Modify: `src/bluse/bench/app.py` (`do_cluster`)
- Modify: `src/bluse/bench/templates/_results.html`
- Modify: `src/bluse/bench/static/app.css`

**Interfaces:**
- Consumes: `metrics.quality`, `metrics.epoch_trace` from Tasks 4–5.
- Produces: `Run.stats` gains every key from `metrics.quality` plus `seconds` and `n_features`; `Run.epochs` is the rendered trace.

- [ ] **Step 1: Compute the metrics in `do_cluster`**

Replace the `RUNS[sig] = Run(...)` block with:

```python
        summary = summarise(ds.df, labels, origin)
        noise = labels == -1
        conf = noise & (ds.df.n_beams.to_numpy() <= 4)
        rank = {int(c): i for i, c in enumerate(summary["cluster"])}
        q = metrics.quality(labels, ds.df)
        stats = dict(q)
        stats.update({
            "noise": int(noise.sum()),
            "confined": int(conf.sum()),
            "seconds": elapsed,
            "n_features": len(cols),
        })
        RUNS[sig] = Run(sig, labels, summary, stats, p, rank,
                        metrics.epoch_trace(trace, len(labels)))
```

Add `from .. import metrics` to the imports, and `epochs: list = field(default_factory=list)` as the last field of `Run`.

- [ ] **Step 2: Render the metrics and the trace**

In `_results.html`, extend the stats strip (the block of `<span class="stat">` elements) with:

```html
  <span class="stat"><b>{{ '%.2f'|format(run.stats.narrow_frac * 100) }}%</b>
    narrow <span class="dim">&lt;1 MHz</span></span>
  <span class="stat"><b>{{ '%.1f'|format(run.stats.narrow_enrichment) }}&times;</b>
    vs null</span>
  <span class="stat"><b>{{ '%.1f'|format(run.stats.largest_pct) }}%</b>
    largest</span>
  <span class="stat"><b>{{ run.stats.median_size }}</b> median</span>
```

and add, immediately before the cluster table:

```html
{% if run.epochs %}
<div class="eyebrow">Epochs</div>
<table class="epochs">
  <thead><tr><th>epoch</th><th>alive after</th><th>removed</th>
    <th>% of original</th></tr></thead>
  <tbody>
  {% for e in run.epochs %}
    <tr {% if e.removed == 0 %}class="dead"{% endif %}>
      <td>{{ e.epoch }}</td>
      <td>{{ '{:,}'.format(e.alive) }}</td>
      <td>{{ '{:,}'.format(e.removed) }}</td>
      <td class="pct">{{ '%.1f'|format(e.pct_of_original) }}%</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
<p class="note">GLOBULAR reached 47.6% in epoch 1 and a flat 22–30% per epoch
  after, with no plateau at 8. An epoch removing 0 did no work.</p>
{% endif %}
```

Add the weak-proxy caption next to the label-based numbers:

```html
<p class="note">AMI {{ '%.4f'|format(run.stats.ami) }} ·
  enrichment {{ '%.2f'|format(run.stats.enrichment * 100) }}% of clustered hits.
  Both are <b>weak proxies, not detection metrics</b>:
  <code>weak_label == 0</code> means <i>spatially confined</i>, not <i>verified
  clean</i>, and the class balance among labelled rows is 31:1, which compresses
  AMI's whole usable range into the third decimal. Enrichment is near its
  detection floor at <code>min_cluster_size</code> 4 — only a fully confined
  cluster clears — so do not compare it across different
  <code>min_cluster_size</code>.</p>
```

- [ ] **Step 3: Fix D-1 and D-2**

In `_results.html` line ~63, replace the empty-result message:

```html
<p class="error">Nothing clustered. Every point is noise — try a smaller
  min_cluster_size, a smaller min_samples, <code>cluster_selection_method =
  leaf</code>, or turn more features on.</p>
```

and in the history fragment, drop the dead `eps` field:

```html
    <span class="meta">{{ h.params.scaling }} · {{ h.params.mode }} ·
      {{ h.params.csm }} · mcs {{ h.params.mcs }} ·
      {{ h.stats.n_features }}f</span>
```

- [ ] **Step 4: Style the epoch table**

Append to `app.css`:

```css
table.epochs { width: 100%; border-collapse: collapse; font-family: var(--mono);
  font-size: 10.5px; margin-bottom: 10px; }
table.epochs th { text-align: left; color: var(--dimmer); font-weight: 400;
  padding: 2px 6px; }
table.epochs td { padding: 2px 6px; }
table.epochs tr.dead { opacity: 0.4; }
p.note { font-family: var(--mono); font-size: 10px; color: var(--dimmer);
  line-height: 1.55; margin: 4px 0 12px; }
```

- [ ] **Step 5: Verify end to end**

```bash
uv run bluse-bench --port 8765 &
sleep 4
curl -s -X POST http://127.0.0.1:8765/dataset -d "file=sband_short&sample=35000&seed=0" >/dev/null
curl -s -X POST http://127.0.0.1:8765/cluster \
  -d "key=sband_short:35000:0&file=sband_short&scaling=robust&mode=epochs&csm=eom&mcs=4&ms=8&epochs=8&batch=3000&seed=0" \
  $(for f in f01_frequency f02_abs_drift f03_snr f04_spectral_skew f05_spectral_kurtosis f06_bimodality f07_kurt_bw_corr f09_temporal_skew f10_timeseries_std f11_spectrum_std f12_bandwidth_hz f13_redness x01_drift_residual x02_time_occupancy x03_channel_offset; do echo -n " -d feat=${f}_n"; done) \
  | grep -oE '(narrow|[0-9.]+% of original)' | head
kill %1
```
Expected: the epoch table shows ≈87.9% removed in epoch 1 and 0.0% for epochs 4–8, and the strip shows a narrow figure near 0.78%.

- [ ] **Step 6: Commit**

```bash
git add src/bluse/bench/app.py src/bluse/bench/templates/_results.html \
        src/bluse/bench/static/app.css
git commit -m "feat: report cluster quality and the epoch trace in the Bench

Adds narrow-cluster share with its null, largest-cluster fraction and
median size to the results strip, and a per-epoch reduction table that
makes the single-pass collapse visible -- 87.9% removed in epoch 1 and
nothing at all in epochs 4-8.

AMI and enrichment are captioned as weak proxies with the 31:1 imbalance
and the enrichment detection floor stated.

Fixes D-1 (h.params.eps rendered empty) and D-2 (stale 'larger epsilon'
advice for a control that no longer exists)."
```

---

## Task 10: the `/stability` route

**Files:**
- Modify: `src/bluse/bench/app.py`
- Modify: `src/bluse/bench/templates/_results.html`
- Create: `src/bluse/bench/templates/_stability.html`

**Interfaces:**
- Consumes: `metrics.stability` from Task 5.
- Produces: `POST /stability` returning the `_stability.html` fragment.

- [ ] **Step 1: Add the route**

In `bench/app.py`, after `do_cluster`:

```python
@app.post("/stability", response_class=HTMLResponse)
async def do_stability(request: Request):
    """
    Re-run one configuration across N seeds and report how much of it survives.

    Deliberately behind a button rather than on every cluster: it is N times
    the cost and it is the slowest thing in the tool.

    Its seeds are kept OUT of HISTORY. The run cache key includes the seed and
    HISTORY is capped at 12, so one N=5 sweep would otherwise insert five
    near-identical entries and evict most of the comparison history the user
    was building.
    """
    form = await request.form()
    key = form["key"]
    ds = DATASETS.get(key)
    if ds is None:
        return HTMLResponse('<p class="error">Dataset expired. Load it again.</p>')

    cols = [c for c in form.getlist("feat") if c in ds.columns]
    if len(cols) < 2:
        return HTMLResponse('<p class="error">Keep at least two features on.</p>')

    n_seeds = max(2, min(int(form.get("n_seeds", 5)), 10))
    p = dict(scaling=form.get("scaling", "robust"),
             mode=form.get("mode", "epochs"),
             csm=form.get("csm", "eom"),
             mcs=int(form.get("mcs", 4)),
             ms=int(form.get("ms", 8)),
             epochs=int(form.get("epochs", 8)),
             batch=int(form.get("batch", 3000)))

    def run_fn(seed):
        labels, _, _, _ = cluster(ds, cols, p["scaling"], p["mode"], p["mcs"],
                                  p["ms"], p["epochs"], p["batch"], seed,
                                  p["csm"])
        return labels

    t0 = time.time()
    s = metrics.stability(run_fn, seeds=tuple(range(n_seeds)))
    s["seconds"] = time.time() - t0
    return templates.TemplateResponse(request, "_stability.html",
                                      {"s": s, "params": p})
```

- [ ] **Step 2: Write the fragment**

Create `src/bluse/bench/templates/_stability.html`:

```html
<div class="eyebrow">Stability across {{ s.n_seeds }} seeds</div>
<div class="stats">
  <span class="stat"><b>{{ '%.3f'|format(s.ari_restricted) }}</b>
    ARI <span class="dim">membership</span></span>
  <span class="stat"><b>{{ '%.3f'|format(s.ari_composite) }}</b>
    ARI <span class="dim">composite</span></span>
  <span class="stat"><b>{{ '%.3f'|format(s.noise_agreement) }}</b>
    noise agree</span>
  <span class="stat"><b>{{ '%.0f'|format(s.k_mean) }}</b>
    clusters <span class="dim">{{ s.k_min }}–{{ s.k_max }}</span></span>
  <span class="stat"><b>{{ '%.1f'|format(s.seconds) }}s</b></span>
</div>
<p class="note"><b>Read the membership figure.</b> The composite scores
  <code>-1</code> as an ordinary label, so a configuration that leaves half its
  points unclustered is credited for every within-noise pair. Measured:
  <code>leaf</code> scores 0.480 composite against <code>eom</code>'s 0.024 — a
  20&times; apparent advantage — while on membership the two are 0.032 and
  0.028. Noise agreement is near-degenerate for <code>eom</code>, which
  clusters 99.9% of points; record it, do not gate on it.</p>
```

- [ ] **Step 3: Add the button**

In `_results.html`, immediately after the stats strip:

```html
<form hx-post="/stability" hx-target="#stability" hx-swap="innerHTML"
      hx-include="#cluster-form" hx-indicator="#results, #spin">
  <input type="hidden" name="key" value="{{ key }}">
  <input type="hidden" name="n_seeds" value="5">
  <button type="submit" class="ghost">Check stability (5 seeds)</button>
</form>
<div id="stability"></div>
```

- [ ] **Step 4: Verify**

```bash
uv run bluse-bench --port 8765 &
sleep 4
curl -s -X POST http://127.0.0.1:8765/dataset -d "file=sband_short&sample=35000&seed=0" >/dev/null
curl -s -X POST http://127.0.0.1:8765/stability \
  -d "key=sband_short:35000:0&scaling=robust&mode=epochs&csm=eom&mcs=4&ms=8&epochs=8&batch=3000&n_seeds=3" \
  $(for f in f01_frequency f02_abs_drift f03_snr f04_spectral_skew f05_spectral_kurtosis f06_bimodality f07_kurt_bw_corr f09_temporal_skew f10_timeseries_std f11_spectrum_std f12_bandwidth_hz f13_redness x01_drift_residual x02_time_occupancy x03_channel_offset; do echo -n " -d feat=${f}_n"; done) \
  | grep -oE '<b>[0-9.]+</b>' | head -4
kill %1
```
Expected: four numbers. The membership ARI should be ≈0.028 and the composite ≈0.024 for `eom`.

- [ ] **Step 5: Commit**

```bash
git add src/bluse/bench/app.py src/bluse/bench/templates/
git commit -m "feat: add the stability check to the Bench

Runs one configuration across N seeds and reports membership ARI,
composite ARI and noise agreement as three separate numbers, with copy
explaining which to read and why.

Kept behind a button because it is N times the cost, and its seeds are
excluded from HISTORY so one sweep does not evict the comparison history."
```

---

## Task 11: wire matching into both entry points

**Files:**
- Modify: `src/bluse/bench/app.py`, `src/bluse/track_b_cluster.py`
- Modify: `src/bluse/bench/templates/_controls.html`, `_results.html`
- Test: `tests/unit/test_matching_wiring.py`

**Interfaces:**
- Consumes: `matching.match` from Task 6.
- Produces: `Run.families: np.ndarray` and `Run.match_info: dict`; the form fields `match` (on/off) and `match_q`; the `family` column in the CLI's per-hit output.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_matching_wiring.py`:

```python
import numpy as np

from bluse import matching, metrics
from tests.unit import fixtures


def test_family_stability_is_measurable_with_the_existing_api():
    """
    Acceptance criterion 6 in miniature.

    A stable family COUNT is compatible with scrambled family MEMBERSHIP, so
    the criterion is ari_restricted on family ids. stability() needs only
    run_fn(seed) -> labels, so running it on families is a call-site change.
    """
    labels, X = fixtures.synthetic_centroid_space(seed=0)

    def run_fn(seed):
        # jitter the clusters a little per seed; the FAMILIES must survive it
        rng = np.random.default_rng(seed)
        fam, _ = matching.match(labels, X + rng.normal(0, 0.05, X.shape))
        return fam

    s = metrics.stability(run_fn, seeds=(0, 1, 2))
    assert s["ari_restricted"] > 0.9


def test_families_are_never_more_numerous_than_clusters():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    fam, info = matching.match(labels, X)
    assert info["n_families"] <= info["n_clusters"]
```

- [ ] **Step 2: Run it to make sure it fails, then passes**

Run: `uv run pytest tests/unit/test_matching_wiring.py -v`
Expected: PASS immediately — this test exercises Task 6's module through the Task 5 API and exists to lock the call-site contract. If it fails, the bug is in `matching.match` or `metrics.stability`, not here.

- [ ] **Step 3: Compute families in `do_cluster`**

In `bench/app.py`, inside `do_cluster` where the run is built, after `q = metrics.quality(...)`:

```python
        # Cluster ids are batch artefacts -- one physical population is minted
        # afresh in every batch and every epoch. Families are the level at
        # which that should cancel.
        families = np.full(len(labels), -1, dtype=np.int32)
        match_info = {}
        if form.get("match") == "on":
            Xs = scale(ds.raw[:, [ds.columns.index(c) for c in cols]],
                       p["scaling"])
            families, match_info = matching.match(
                labels, Xs, quantile=int(form.get("match_q", 50)))
            stats_extra = metrics.quality(families, ds.df)
            match_info["family_narrow_frac"] = stats_extra["narrow_frac"]
            match_info["family_median_span_mhz"] = stats_extra["median_span_mhz"]
```

then pass `families` and `match_info` into the `Run`, adding to the dataclass:

```python
    families: np.ndarray = None
    match_info: dict = field(default_factory=dict)
```

Add `from .. import matching` to the imports.

- [ ] **Step 4: Add the Bench controls**

In `_controls.html`, after the `seed` row:

```html
  <div class="row2">
    <label class="field"><span>match families</span>
      <select name="match">
        <option value="off" selected>off</option>
        <option value="on">on — Ward on centroids</option>
      </select></label>
    <label class="field"><span>cut percentile</span>
      <input type="number" name="match_q" value="50" min="1" max="99"></label>
  </div>
```

- [ ] **Step 5: Render the family line**

In `_results.html`, after the stats strip:

```html
{% if run.match_info %}
<p class="note"><b>{{ '{:,}'.format(run.match_info.n_clusters) }} raw clusters
  &rarr; {{ '{:,}'.format(run.match_info.n_families) }} families</b>
  at cut {{ '%.3f'|format(run.match_info.cut) }};
  narrow share {{ '%.2f'|format(run.match_info.family_narrow_frac * 100) }}%,
  median family span
  {{ '%.1f'|format(run.match_info.family_median_span_mhz) }} MHz.
  <br><b>Provisional.</b> Ward runs on centroids in the scaled space, which
  still carries a 14&times; distance-share spread —
  <code>x03_channel_offset</code> at 24.3% and <code>f07_kurt_bw_corr</code> at
  13.1% — so these families are grouped substantially by channel offset and by
  a clipped correlation coefficient. Re-derive after the contribution-
  equalising scaling lands.</p>
{% endif %}
```

- [ ] **Step 6: Add the CLI flags**

In `track_b_cluster.py::main`, after `--cluster-selection-method`:

```python
    p.add_argument("--match", action="store_true",
                   help="group clusters into families by Ward linkage on "
                        "centroids. Cluster ids are batch artefacts -- one "
                        "population is minted afresh per batch and epoch -- "
                        "so families are the level at which that cancels")
    p.add_argument("--match-cut", type=float, default=None,
                   help="explicit linkage distance cut. Default: derived from "
                        "the centroid nearest-neighbour distribution")
    p.add_argument("--match-quantile", type=int, default=50,
                   help="percentile of the centroid NN distance used as the "
                        "cut when --match-cut is not given")
```

and in `main`, after the clustering branch and **before** the
`summarise(df, labels, args.outdir, tag)` call (line ~396). Note that `main`
does not hold the scaled matrix -- `feature_matrix` is called inside
`cluster_epochs` -- so recompute it on the columns actually used:

```python
    if args.match:
        X, _, _ = feature_matrix(df, columns=cols, scaling=args.scaling)
        fam, minfo = matching.match(labels, X, cut=args.match_cut,
                                    quantile=args.match_quantile)
        df["family"] = fam
        print(f"  matched {minfo['n_clusters']:,} clusters -> "
              f"{minfo['n_families']:,} families at cut {minfo['cut']:.3f}")
```

`summarise()` writes `df` to `<tag>_clusters.parquet`, so the `family` column
travels with the per-hit output for free. Add `from . import matching` to the
imports.

- [ ] **Step 7: Run all the tests**

Run: `uv run pytest tests/unit -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/bluse/bench/app.py src/bluse/track_b_cluster.py \
        src/bluse/bench/templates/ tests/unit/test_matching_wiring.py
git commit -m "feat: wire cross-batch matching into the Bench and the CLI

Reports 'N raw clusters -> M families' with the family-level narrow share
and median span, and marks the first taxonomy provisional pending the
deferred scaling work."
```

---

## Task 12: colour-by-feature-value, and D-3

**Files:**
- Modify: `src/bluse/bench/app.py`, `src/bluse/bench/static/scatter.js`, `_controls.html`

**Interfaces:**
- Consumes: `Dataset.raw`.
- Produces: `GET /values.bin?key=&col=` returning a `Float32Array` normalised to [0,1].

- [ ] **Step 1: Add the endpoint**

In `bench/app.py`:

```python
@app.get("/values.bin")
def values_bin(key: str, col: str):
    """
    One feature column, normalised to [0,1], for colour-by-value.

    This is what makes the rail's distance shares visible rather than tabular:
    colouring by f02_abs_drift_n renders the zero-drift slab immediately, and
    if colouring by f01_frequency_n reproduces the cluster structure, that is a
    one-click finding.
    """
    ds = DATASETS.get(key)
    if ds is None or col not in ds.columns:
        return Response(status_code=404)
    v = ds.raw[:, ds.columns.index(col)].astype(np.float32)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = hi - lo if hi - lo > 1e-12 else 1.0
    return Response(((v - lo) / span).astype(np.float32).tobytes(),
                    media_type="application/octet-stream")
```

- [ ] **Step 2: Fix D-3 — use the grid**

In `scatter.js`, replace `nearest()` with:

```js
/* Was a full O(n) scan with a toScreen() per point on every mousemove -- 35,000
 * coordinate transforms per mouse move -- while buildGrid() populated a 90x90
 * index that nothing read. Now it reads it: convert the cursor to data space,
 * walk only the cells within the hit radius. */
function nearest(mx, my) {
  if (!emb || !grid) return -1;
  const cells = 90;
  const r = 14 * dpr;
  // Invert toScreen for the two corners of the hit box to get a data-space box.
  const [x0, y0] = toData(mx - r, my - r);
  const [x1, y1] = toData(mx + r, my + r);
  const cx0 = Math.max(0, Math.floor(Math.min(x0, x1) * cells));
  const cx1 = Math.min(cells - 1, Math.floor(Math.max(x0, x1) * cells));
  const cy0 = Math.max(0, Math.floor(Math.min(y0, y1) * cells));
  const cy1 = Math.min(cells - 1, Math.floor(Math.max(y0, y1) * cells));
  let best = -1, bd = r * r;
  for (let cy = cy0; cy <= cy1; cy++) {
    for (let cx = cx0; cx <= cx1; cx++) {
      for (const i of grid[cy * cells + cx]) {
        const [sx, sy] = toScreen(i);
        const d = (sx - mx) ** 2 + (sy - my) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
    }
  }
  return best;
}
```

and add the inverse transform next to `toScreen`:

```js
/* Inverse of toScreen, so nearest() can turn a pixel box into a cell range. */
function toData(sx, sy) {
  return [(sx / dpr - pan.x) / zoom / W, (sy / dpr - pan.y) / zoom / H];
}
```

**Note for the implementer:** `toScreen` is at `scatter.js:48`. Read it first and write `toData` as its exact algebraic inverse — the variable names above (`pan`, `zoom`, `W`, `H`) are indicative. If `toScreen` does not decompose cleanly, keep the O(n) scan and delete `buildGrid()` instead; a dead index and a linear scan is better than a wrong one. Either resolution closes D-3.

- [ ] **Step 3: Add the colour-by control and the fetch**

In `_controls.html`, after the embedding select:

```html
  <label class="field"><span>colour by</span>
    <select id="colour-by" onchange="loadValues()">
      <option value="" selected>cluster</option>
      {% for f in features %}
      <option value="{{ f.col }}">{{ f.label }}</option>
      {% endfor %}
    </select>
  </label>
```

In `scatter.js`, add:

```js
let values = null;   // Float32Array in [0,1], or null to colour by cluster

async function loadValues() {
  const col = (document.getElementById('colour-by') || {}).value || '';
  if (!col) { values = null; buildRGB(); draw(); return; }
  setBusy(true);
  try {
    const res = await fetch(`/values.bin?key=${encodeURIComponent(window.DSKEY)}`
                            + `&col=${encodeURIComponent(col)}`);
    values = res.ok ? new Float32Array(await res.arrayBuffer()) : null;
    buildRGB();
    draw();
  } finally { setBusy(false); }
}

/* Diverging ramp, blue -> grey -> orange, for colour-by-value. */
function rampRGB(t) {
  const c = Math.max(0, Math.min(1, t));
  return c < 0.5
    ? [74 + (140 - 74) * c * 2, 163 + (150 - 163) * c * 2, 232 + (150 - 232) * c * 2]
    : [140 + (232 - 140) * (c - 0.5) * 2, 150 + (115 - 150) * (c - 0.5) * 2,
       150 + (74 - 150) * (c - 0.5) * 2];
}
```

and in `buildRGB()`, before the existing per-point colour lookup:

```js
    if (values && values.length === n) {
      const [r, g, b] = rampRGB(values[i]);
      curRGB[i * 3] = r; curRGB[i * 3 + 1] = g; curRGB[i * 3 + 2] = b;
      continue;
    }
```

- [ ] **Step 4: Verify**

```bash
uv run bluse-bench --port 8765 &
sleep 4
curl -s -X POST http://127.0.0.1:8765/dataset -d "file=sband_short&sample=35000&seed=0" >/dev/null
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" \
  "http://127.0.0.1:8765/values.bin?key=sband_short:35000:0&col=f02_abs_drift_n"
kill %1
```
Expected: `200 140000` (35,000 float32s).

- [ ] **Step 5: Commit**

```bash
git add src/bluse/bench/app.py src/bluse/bench/static/scatter.js \
        src/bluse/bench/templates/_controls.html
git commit -m "feat: colour the scatter by any feature value

Makes the rail's distance shares visible rather than tabular -- colouring
by f02_abs_drift_n renders the zero-drift slab at once.

Fixes D-3: nearest() did a full O(n) scan with a coordinate transform per
point on every mousemove while buildGrid() maintained an index nothing
read."
```

---

## Task 13: CLI flags, the metrics file, and `--report`

**Files:**
- Modify: `src/bluse/track_b_cluster.py`

**Interfaces:**
- Consumes: `diagnostics.audit`, `metrics.quality`, `metrics.stability`.
- Produces: `<tag>_metrics.json` in the output directory; `--report`, `--seeds`.

- [ ] **Step 1: Add the flags**

In `main`, after the matching flags:

```python
    p.add_argument("--seeds", type=int, default=0,
                   help="if >1, re-run across this many seeds and report "
                        "membership ARI, composite ARI and noise agreement. "
                        "Costs one clustering run per seed")
    p.add_argument("--report", action="store_true",
                   help="print the per-feature diagnostics table and exit "
                        "without clustering")
```

- [ ] **Step 2: Implement `--report`**

In `main`, immediately after `df` is loaded and `tag` is set (line ~388), and
**before** the `if args.mode == "epochs":` branch:

```python
    if args.report:
        X, columns, good = feature_matrix(df, scaling=args.scaling)
        kinds = {c + "_n": k for c, k in F.column_kinds().items()}
        rows = diagnostics.audit(X[good], columns, scaling=args.scaling,
                                 kinds=kinds, min_samples=args.min_samples)
        eq = 100.0 / max(len(columns), 1)
        print(f"\n  {len(X[good]):,} rows, {len(columns)} features, "
              f"scaling={args.scaling}, equal share {eq:.1f}%\n")
        head = (f"  {'column':30s} {'vals':>7s} {'tie':>6s} {'clip':>6s} "
                f"{'IQR':>9s} {'share':>7s} {'knn':>7s}  flags")
        print(head)
        for r in rows:
            print(f"  {r['label']:30s} {r['n_distinct']:7d} "
                  f"{r['max_tie_fraction']:6.3f} {r['clip_frac']:6.3f} "
                  f"{r['iqr_raw']:9.3f} {100*r['share_global']:6.1f}% "
                  f"{100*r['share_knn']:6.1f}%  {','.join(r['flags'])}")
        print()
        return
```

Add `from . import diagnostics, matching, metrics` and `from . import features as F` to the imports as needed.

- [ ] **Step 3: Write the metrics file**

In `main`, immediately after the `summarise(df, labels, args.outdir, tag)`
call. `main` holds `df`, `labels`, `cols`, `trace`, `tag` and `args`, which is
everything needed:

```python
    q = metrics.quality(labels, df)
    q["epochs"] = metrics.epoch_trace(trace, len(labels))
    if args.seeds and args.seeds > 1:
        def run_fn(seed):
            a = argparse.Namespace(**vars(args))
            a.seed = seed
            lab, _, _, _ = cluster_epochs(df, a)
            return lab
        q["stability"] = metrics.stability(run_fn,
                                           seeds=tuple(range(args.seeds)))
    # narrow_frac_at is keyed by float, which json cannot serialise as a key.
    q["narrow_frac_at"] = {str(k): v for k, v in q["narrow_frac_at"].items()}
    path = os.path.join(args.outdir, f"{tag}_metrics.json")
    with open(path, "w") as fh:
        json.dump(q, fh, indent=2, default=float)
    print(f"  wrote {path}")
```

`--seeds` re-clusters the **whole** file per seed, which on `all` is 1.28M rows
and minutes per seed. Say so in the flag's help text.

Add `import json` to the imports (`argparse` is already there).

- [ ] **Step 4: Verify**

```bash
cd aug_2026_workshop
uv run bluse-cluster --file sband_short --report | head -25
```
Expected: a table in which `f02_abs_drift` shows 42 values, tie 0.266, share ≈1.7%, flags `tie,share-low`; and `x03_channel_offset` shows share ≈24.3%, flag `share-high`.

- [ ] **Step 5: Commit**

```bash
git add src/bluse/track_b_cluster.py
git commit -m "feat: add --report, --seeds and a metrics.json to bluse-cluster

--report prints the per-feature diagnostics without clustering, so the
distance-share audit is available from the CLI and not only in the Bench."
```

---

## Task 14: run the acceptance measurements and record them

The last task produces no new code. It runs the tool against the eight acceptance criteria, decides the `eom`/`leaf` default from what comes back, and writes the numbers down. Nothing in this plan is finished until this task's numbers exist, because the whole point of the work is that decisions stop being a matter of taste.

**Files:**
- Create: `aug_2026_workshop/clusters/acceptance-2026-09.md`
- Modify: `aug_2026_workshop/README.md`
- Modify: `docs/bench-review-2026-09.md` (header note only)
- Modify: `src/bluse/bench/templates/_controls.html` or `track_b_cluster.py` (only if the default flips)

**Interfaces:**
- Consumes: everything.
- Produces: a recorded decision.

- [ ] **Step 1: Write the measurement script**

Create `aug_2026_workshop/acceptance.py`:

```python
#!/usr/bin/env python3
"""
Run the spec's acceptance criteria and print a table. Throwaway-adjacent: it
exists so the numbers in acceptance-2026-09.md are reproducible rather than
transcribed by hand.

    uv run python aug_2026_workshop/acceptance.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bluse import diagnostics as D          # noqa: E402
from bluse import features as F             # noqa: E402
from bluse import matching, metrics, paths  # noqa: E402
from bluse.bench import app                 # noqa: E402


class DS:
    def __init__(self, X, cols, df):
        self.raw, self.columns, self.df = X, cols, df


def load(name):
    path = os.path.join(paths.features_dir(), f"{name}_features.parquet")
    df = pd.read_parquet(path)
    df = df[df.feature_ok].reset_index(drop=True)
    cols = [c + "_n" for c in F.all_columns()
            if c + "_n" in df.columns and not c.endswith("_saturated")
            and not c.startswith("f08_")]
    X = df[cols].to_numpy(dtype=np.float64)
    good = np.isfinite(X).all(axis=1)
    return DS(X[good], cols, df[good].reset_index(drop=True))


def main():
    out = {}
    ds = load("sband_short")
    kinds = {c + "_n": k for c, k in F.column_kinds().items()}

    # criteria 1 and 2 -- diagnostics
    rows = {r["col"]: r for r in D.audit(ds.raw, ds.columns, scaling="robust",
                                         kinds=kinds, min_samples=8)}
    out["diagnostics"] = {
        c: {k: rows[c][k] for k in ("n_distinct", "max_tie_fraction",
                                    "clip_frac", "share_global", "share_knn",
                                    "flags")}
        for c in ("f02_abs_drift_n", "x03_channel_offset_n",
                  "f07_kurt_bw_corr_n")
    }

    # criteria 3, 4, 5, 6 -- per selection method
    for method in ("eom", "leaf"):
        labels, X, _, trace = app.cluster(ds, ds.columns, "robust", "epochs",
                                          4, 8, 8, 3000, 0, method)
        q = metrics.quality(labels, ds.df)
        fam, minfo = matching.match(labels, X)
        fq = metrics.quality(fam, ds.df)

        def run_cl(seed, m=method):
            lab, _, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                       4, 8, 8, 3000, seed, m)
            return lab

        def run_fam(seed, m=method):
            lab, Xs, _, _ = app.cluster(ds, ds.columns, "robust", "epochs",
                                        4, 8, 8, 3000, seed, m)
            f, _ = matching.match(lab, Xs)
            return f

        out[method] = {
            "quality": {k: v for k, v in q.items() if k != "narrow_frac_at"},
            "narrow_frac_at": {str(k): v for k, v in q["narrow_frac_at"].items()},
            "epochs": metrics.epoch_trace(trace, len(labels)),
            "stability_clusters": metrics.stability(run_cl, seeds=(0, 1, 2, 3, 4)),
            "stability_families": metrics.stability(run_fam, seeds=(0, 1, 2, 3, 4)),
            "matching": {k: v for k, v in minfo.items() if k != "nn_distances"},
            "family_quality": {"narrow_frac": fq["narrow_frac"],
                               "median_span_mhz": fq["median_span_mhz"]},
        }

    # the section 7 hedge: how much does the deferred scaling work move things?
    drop = [c for c in ds.columns
            if not c.startswith(("x03_", "f07_"))]
    labels, X, _, _ = app.cluster(ds, drop, "robust", "epochs",
                                  4, 8, 8, 3000, 0, "leaf")
    fam, minfo = matching.match(labels, X)
    fq = metrics.quality(fam, ds.df)
    out["leaf_without_x03_f07"] = {
        "n_families": minfo["n_families"],
        "narrow_frac": fq["narrow_frac"],
        "median_span_mhz": fq["median_span_mhz"],
    }

    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and capture the output**

```bash
cd /home/bester/projects/bluse
uv run python aug_2026_workshop/acceptance.py \
  > /tmp/claude-1000/-home-bester-projects-bluse/716eaba4-3e20-4c61-bf4a-bd5b364434bd/scratchpad/acceptance.json
head -60 /tmp/claude-1000/-home-bester-projects-bluse/716eaba4-3e20-4c61-bf4a-bd5b364434bd/scratchpad/acceptance.json
```

Expected, against the spec's §12:

| criterion | expected |
|---|---|
| 1 | `f02` `n_distinct=42`, `max_tie_fraction≈0.266`, flags include `tie` and `share-low`; `x03` `share_global≈0.243`, flag `share-high` |
| 2 | `f07` `clip_frac≈0.010` |
| 3 | `eom` `narrow_frac≈0.0078`; `leaf` `≈0.068` |
| 4 | `eom` epoch 1 ≈87.9%, epochs 4–8 exactly 0.0 |
| 5 | `eom` composite ≈0.024, restricted ≈0.028, noise ≈0.999; `leaf` ≈0.48 / ≈0.032 / ≈0.78 |
| 6 | `stability_families` restricted ARI — **no expected value; this has never been measured** |

- [ ] **Step 3: Judge criterion 6 and write the finding**

This is the one criterion with no predicted answer, and it decides how the work is written up. Read `stability_families.ari_restricted` for both methods against `stability_clusters.ari_restricted` (≈0.028 and ≈0.032):

- **Family ARI ≳ 0.5** — matching solved the reproducibility problem. This is the strongest result available from the whole programme: cluster membership is irreproducible under both selection methods, and grouping into families recovers it. Write it up as the headline finding.
- **Family ARI ≈ 0.05–0.3** — partial. Report the number plainly, and record that families are more reproducible than clusters but still not stable.
- **Family ARI ≲ 0.05** — matching did **not** solve it. Say so explicitly. Do not ship the family view as a fix, and record that the mechanism (per-batch id minting) is not the whole cause of the ARI collapse, which reopens the question.

Write `aug_2026_workshop/clusters/acceptance-2026-09.md` containing: the eight criteria as a pass/fail table with measured values; the criterion-6 finding in prose; and the `leaf_without_x03_f07` comparison as a bound on how much the deferred scaling work is likely to move the family taxonomy.

- [ ] **Step 4: Decide the default**

Compare `eom` and `leaf` on `narrow_frac`, `stability_families.ari_restricted`, and family count.

Flip the default to `leaf` **only if** its family-level ARI is at least as good as `eom`'s and its `narrow_frac` advantage survives (expected ≈8.8×). If it flips, change the `selected` attribute in `_controls.html` and the `default=` in `track_b_cluster.py`, and say so in the commit message. If it does not flip, leave both at `eom` and record why — the addendum's reasoning was that ~2,127 unmatched discovery-order ids read worse than 79, and matching is what removes that objection, so this decision is exactly the thing matching was built to inform.

- [ ] **Step 5: Update the workshop README**

Add a section to `aug_2026_workshop/README.md` covering: the diagnostics table for `sband_short`; the epoch trace showing epochs 4–8 doing no work; the stability numbers at both levels with a sentence on why the membership figure is the one to read; the `eom`/`leaf` decision and its basis; and a pointer to `acceptance-2026-09.md`.

State plainly that the first family taxonomy is provisional pending the contribution-equalising scaling work, and that `share_knn` is reported but not yet thresholded because no value for it had been measured when the threshold rules were written. Record the `share_knn` values now measured, so the follow-up has a baseline.

- [ ] **Step 6: Add the review-document header note**

At the top of `docs/bench-review-2026-09.md`, immediately after the `**Status:**` line:

```markdown
> **Superseded in places.** `bench-review-2026-09-response.md` measured this
> document's claims on the real feature matrices and corrected several;
> `bench-review-2026-09-addendum.md` is the reviewer's second pass, which
> withdraws its §2 (the `f06` redundancy claim), its §1.3 step 3 (pairwise ARI
> across configurations), and the D-4 framing. Where they conflict with this
> document, they win. The `[measured]` figures below were
> simulation-calibrated; treat the response's as authoritative.
> `docs/superpowers/specs/2026-09-01-cluster-bench-review-design.md` is the
> design that came out of all three.
```

Leave the reviewer's prose otherwise untouched — their document, their voice, and the supersession chain stays legible.

- [ ] **Step 7: Run the whole suite one last time**

```bash
uv run pytest tests/ -v
```
Expected: `tests/unit` all pass; `tests/workspace` all pass from the repo root and skip cleanly from a directory with no workspace. Verify the skip actually works:

```bash
cd /tmp && uv run --project /home/bester/projects/bluse pytest \
  /home/bester/projects/bluse/tests/ -v 2>&1 | tail -5
```
Expected: the workspace tests report as skipped, not failed.

- [ ] **Step 8: Commit**

```bash
cd /home/bester/projects/bluse
git add aug_2026_workshop/ docs/bench-review-2026-09.md \
        src/bluse/bench/templates/_controls.html src/bluse/track_b_cluster.py
git commit -m "docs: record the acceptance measurements

Runs the spec's eight acceptance criteria and writes the numbers to
aug_2026_workshop/clusters/acceptance-2026-09.md.

Criterion 6 -- membership ARI on family ids -- had no predicted value and
is the finding this work programme turns on: cluster membership is
irreproducible under both selection methods (0.0279 eom, 0.0316 leaf), and
this is the measurement of whether grouping into families recovers it.

Also adds a supersession header to the original review so the three review
documents do not silently disagree in the repo."
```

---

## Self-review notes

Checked against the spec, 2026-09-01:

- **Spec coverage.** §3 architecture → Tasks 3–6; §4.1 quality → Task 4; §4.2 stability → Task 5; §4.3 epoch trace → Task 5; §5 diagnostics → Task 3; §6 `kind` → Task 2; §7 matching → Tasks 6, 11; §8.1 `cluster_selection_method` → Task 7; §8.2 Bench → Tasks 8–12; §8.3 CLI → Tasks 7, 11, 13; §8.4 D-1/D-2 → Task 9, D-3 → Task 12, D-4 → Task 8; §10 tests → Tasks 1–7, 11; §11 housekeeping → Task 14; §12 acceptance → Task 14.
- **Deliberately not covered**, per the spec's §9: the contribution-equalising scaling mode and the `f02` ordinal rework. Task 14 measures the `leaf_without_x03_f07` bound so the follow-up spec starts with a number.
- **Type consistency.** `cluster()` returns a 4-tuple from Task 7 onward and every later call site uses four values. `run_hdbscan(X, mcs, ms, method="eom")` is the signature from Task 7. `diagnostics.audit` returns the key set used by Tasks 8, 13, 14. `matching.match` returns `(family_ids, info)` throughout. `metrics.quality` returns `narrow_frac_at` keyed by float, which Task 14 stringifies for JSON.
- **One known soft spot**, flagged inline rather than hidden: Task 12's `toData` assumes `toScreen` inverts cleanly. The step says to read `scatter.js:48` first and, if it does not, to delete `buildGrid()` instead. Either resolution closes D-3.
- **Signatures verified against the real code, not assumed.** The first draft of
  Tasks 11 and 13 targeted a `process()` function that does not exist; the CLI's
  summary writer is `summarise(df, labels, outdir, tag)` (line ~239) and the
  orchestration lives in `main()` (line ~340). `cluster_epochs` currently
  returns `(labels, cols, epoch_of)` and `cluster_single` returns
  `(labels, cols)`, so Task 7 now makes both return a trace explicitly rather
  than leaving Task 13 to assume one. `main()` does not hold the scaled matrix —
  `feature_matrix` is called inside the clustering functions — so Task 11
  recomputes it rather than referencing a variable that is not in scope.
