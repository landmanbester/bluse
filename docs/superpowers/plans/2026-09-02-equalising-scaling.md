# Contribution-equalising scaling — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans`
> (or `subagent-driven-development`) to implement this task-by-task. Steps use
> checkbox syntax for tracking.

**Goal:** add a `robust-equalised` scaling mode that targets each feature's
contribution to the distance directly, instead of the IQR proxy `robust` uses.

**Architecture:** weights are computed in `diagnostics` from the already-scaled
matrix and applied inside `scale()`, so both entry points get them from the one
shared implementation and cannot drift. Which weighting strategy becomes the
default is decided by Task 2's measurement, not by this plan.

**Tech stack:** numpy, scikit-learn (`NearestNeighbors`), FastAPI + htmx for the
Bench, pytest.

**Spec:** [`../specs/2026-09-02-equalising-scaling-design.md`](../specs/2026-09-02-equalising-scaling-design.md) — read it first; the
constraints in its §2 are measured findings, not preferences.

## Global constraints

- **Branch `feature/equalising-scaling`.** Do not merge to `main` without review.
- **`leaf` only for any quoted result.** Under `eom` equalisation collapses
  family ARI 0.519 → 0.044. Warn, do not refuse.
- **Every family comparison at matched count via `n_families=`**, at ≥36
  families. Family count is a granularity dial; below ~30 the narrow share is
  0.000% for every leaf configuration and famARI is meaningless.
- **`boolean` and `flag` columns always get weight exactly 1.0.**
- **Seeds 0–2** for every stability number, family ARI restricted to jointly
  clustered points (`metrics.stability(...)["ari_restricted"]`).
- **Baseline to beat:** `leaf`/`robust` on sband_short — famARI 0.4888 at 36
  families, narrow 0.194%, median span 265.0 MHz.
- Run everything with `uv run`. Experiment scripts are throwaway and live in the
  scratchpad; only their *conclusions* are committed, to `docs/`.

---

### Task 1: `equalising_weights()` with both strategies

**Files:**
- Modify: `src/bluse/diagnostics.py`
- Test: `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: `_shares`, `_shares_knn`, `scale` (existing).
- Produces: `equalising_weights(Z, *, kinds=None, columns=None,
  strategy="closed", iters=0, damping=0.5, cap=None, min_samples=8,
  knn_sample=20_000, seed=0) -> (np.ndarray, dict)`. `info` keys:
  `strategy`, `iters`, `skipped` (list of column names), `max_dev_global`,
  `max_dev_knn`, `weight_min`, `weight_max`.

- [ ] **Step 1: Write the failing tests**

```python
def test_closed_form_equalises_the_global_share():
    """w ∝ 1/sigma is the exact solution: share_global_j ∝ w_j^2 var_j."""
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4000, 4)) * np.array([1.0, 5.0, 0.2, 2.0])
    w, info = D.equalising_weights(Z, strategy="closed")
    s = D._shares(Z * w, np.random.default_rng(0))
    assert np.abs(s / 0.25 - 1).max() < 0.15
    assert info["strategy"] == "closed"


def test_boolean_and_flag_columns_keep_weight_one():
    """
    Spec section 2.2. A low-variance indicator draws a huge equalising weight
    -- a zero-drift boolean measured 0.33% k-NN share, which would have drawn
    10.256, twice the constant measured to destroy eom.
    """
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.normal(0, 3, 3000), rng.normal(0, 1, 3000),
                         (rng.random(3000) < 0.27).astype(float)])
    cols = ["a", "b", "is_x"]
    w, info = D.equalising_weights(Z, columns=cols,
                                   kinds={"is_x": "boolean"})
    assert w[2] == 1.0
    assert info["skipped"] == ["is_x"]


def test_weights_are_deterministic():
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3000, 5)) * np.array([1.0, 4.0, 0.3, 2.0, 1.0])
    a, _ = D.equalising_weights(Z, strategy="closed")
    b, _ = D.equalising_weights(Z, strategy="closed")
    assert np.array_equal(a, b)


def test_iterative_strategy_respects_its_cap():
    """
    The undamped k-NN fixed point does NOT converge -- the k-NN graph is itself
    a function of w, so the map is not a contraction. Measured: weights run
    from 0.500-3.047 after one iteration to 0.044-10.347 after eight while the
    share is still 0.33 off equal. The cap is what makes it usable at all.
    """
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3000, 5)) * np.array([1.0, 8.0, 0.2, 3.0, 1.0])
    w, _ = D.equalising_weights(Z, strategy="knn", iters=4, cap=2.0)
    assert w.max() <= 2.0 + 1e-9
    assert w.min() >= 1 / 2.0 - 1e-9


def test_degenerate_columns_do_not_produce_infinite_weights():
    from bluse import diagnostics as D
    Z = np.random.default_rng(0).normal(size=(500, 3))
    Z[:, 1] = 4.0                                    # zero variance
    w, _ = D.equalising_weights(Z, strategy="closed")
    assert np.isfinite(w).all()
    assert w[1] == 1.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/unit/test_diagnostics.py -q`
Expected: five failures, `AttributeError: module 'bluse.diagnostics' has no
attribute 'equalising_weights'`.

- [ ] **Step 3: Implement**

```python
def equalising_weights(Z, *, kinds=None, columns=None, strategy="closed",
                       iters=0, damping=0.5, cap=None, min_samples=8,
                       knn_sample=20_000, seed=0):
    """
    Per-column weights that equalise each column's contribution to the
    distance. See docs/superpowers/specs/2026-09-02-equalising-scaling-design.md.

    strategy="closed"  w ∝ 1/sigma, the exact solution for the GLOBAL share
                       (share_global_j ∝ w_j^2 var_j). One pass, no seed.
    strategy="knn"     damped fixed point on the k-NN share, which is the
                       statistic HDBSCAN responds to but which does NOT
                       converge: the neighbour graph is a function of w. Only
                       usable with `iters` and `cap` fixed by measurement.
    strategy="hybrid"  closed form, then `iters` damped k-NN steps.

    boolean and flag columns keep weight exactly 1.0 -- their low variance
    draws a huge weight, and a measured example (a zero-drift indicator at
    0.33% k-NN share) would have drawn 10.256.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n_cols = Z.shape[1]
    kinds = kinds or {}
    columns = list(columns) if columns is not None else [None] * n_cols
    frozen = np.array([kinds.get(c) in ("boolean", "flag") for c in columns])

    def _closed(w):
        sd = (Z * w).std(axis=0)
        out = w / np.where(sd > 1e-12, sd, 1.0)
        return out

    def _step(w):
        s = _shares_knn(Z * w, min_samples, np.random.default_rng(seed),
                        knn_sample)
        s = np.where(np.isfinite(s) & (s > 1e-9), s, 1.0 / n_cols)
        return w * ((1.0 / n_cols) / s) ** damping

    w = np.ones(n_cols)
    if strategy in ("closed", "hybrid"):
        w = _closed(w)
    if strategy in ("knn", "hybrid"):
        for _ in range(int(iters)):
            w = _step(w)
            w = _normalise(w, frozen, cap)
    w = _normalise(w, frozen, cap)

    rng = np.random.default_rng(seed)
    sg = _shares(Z * w, rng)
    sk = _shares_knn(Z * w, min_samples, rng, knn_sample)
    eq = 1.0 / n_cols
    info = {
        "strategy": strategy, "iters": int(iters),
        "skipped": [c for c, f in zip(columns, frozen) if f and c is not None],
        "max_dev_global": float(np.abs(sg / eq - 1).max()),
        "max_dev_knn": float(np.abs(sk / eq - 1).max()),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
    }
    return w, info


def _normalise(w, frozen, cap):
    """Mean-1 over the free columns, cap, and pin the frozen ones at 1.0."""
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    w[frozen] = 1.0
    free = ~frozen
    if free.any() and w[free].mean() > 0:
        w[free] = w[free] / w[free].mean()
    if cap:
        w[free] = np.clip(w[free], 1.0 / cap, cap)
    return w
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_diagnostics.py -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bluse/diagnostics.py tests/unit/test_diagnostics.py
git commit -m "P1-5: equalising_weights with closed-form and k-NN strategies"
```

---

### Task 2: choose the strategy by measurement

This is the task the spec defers to. **Do not skip it and do not guess the
answer** — the whole point is that the cheap method targets the demoted
statistic and the method targeting the promoted one is ill-behaved.

**Files:**
- Create: `docs/equalising-scaling-experiment-2026-09.md`
- Test: `tests/unit/test_diagnostics.py` (decision pin)

- [ ] **Step 1: Write the experiment script** in the scratchpad. Load
  `sband_short_features.parquet`, keep `feature_ok` rows with all-finite
  features, build `Z = scale(X, "robust")`. For each candidate compute weights,
  cluster with `bench.app.cluster(..., "none", "epochs", 4, 8, 8, 3000, seed,
  "leaf")` on `Z * w` for seeds 0, 1, 2, and score at `n_families=` 16, 24, 36,
  44 using `matching.match(lab, Zs, n_families=n)` and `metrics.quality`.

  Candidates: `robust` baseline; `closed`; `knn` with `iters` ∈ {2, 4, 6},
  `damping` 0.5, `cap` ∈ {None, 2.0}; `hybrid` with `iters` ∈ {1, 2}.

- [ ] **Step 2: Also record, for each candidate**: fit cost on sband_short and
  on lband_short, and the 35k-sample-vs-population weight deviation. The
  measured figures for the undamped k-NN chase are 9.3 s/iteration and 29.5%
  max deviation; for the closed form 1.5 ms and 0.78%. Reproducibility of the
  *weights* is part of the score, not a footnote.

- [ ] **Step 3: Pick the winner.** Rank on family ARI at 36 families first,
  then median family span, then narrow share. **If the closed form is within
  0.03 famARI of the best, it wins** — determinism, 350× lower cost and
  sample-stability are worth that margin, and the spec says so in advance so
  the rule cannot be chosen after seeing the numbers.

- [ ] **Step 4: Write it up** in `docs/equalising-scaling-experiment-2026-09.md`
  with the full table, including the candidates that lost. Record the cost and
  sample-stability columns.

- [ ] **Step 5: Pin the decision**

```python
def test_equalising_default_is_the_measured_winner():
    """
    Decision pin. The default strategy and its parameters were chosen by the
    experiment in docs/equalising-scaling-experiment-2026-09.md, ranked on
    family ARI at 36 families under leaf. If this fails, re-run that
    experiment rather than editing the constant.
    """
    from bluse import diagnostics as D
    assert D.EQUALISE_STRATEGY == "<winner>"
    assert D.EQUALISE_ITERS == <n>
    assert D.EQUALISE_CAP == <cap>
```

- [ ] **Step 6: Commit** the write-up, the constants and the pin.

---

### Task 3: wire it into `scale()`

**Files:**
- Modify: `src/bluse/diagnostics.py`
- Test: `tests/unit/test_diagnostics.py`

**Interfaces:**
- Produces: `scale(X, "robust-equalised", stats=None, *, kinds=None,
  columns=None)`; `audit(..., scaling="robust-equalised")` works.

- [ ] **Step 1: Write the failing tests**

```python
def test_robust_equalised_is_robust_times_the_weights():
    from bluse import diagnostics as D
    X = np.random.default_rng(0).normal(size=(2000, 4)) * [1, 6, 0.3, 2]
    base = D.scale(X, "robust")
    w, _ = D.equalising_weights(base, strategy=D.EQUALISE_STRATEGY,
                                iters=D.EQUALISE_ITERS, cap=D.EQUALISE_CAP)
    assert np.allclose(D.scale(X, "robust-equalised"), base * w)


def test_equalised_mode_still_reports_clipping():
    """
    audit()'s clip_frac keys on scaling == "robust". The equalised mode has a
    robust base and clips exactly as much, so omitting it would silently
    report 0.0 -- the same false-negative shape as the --scaling none bug.
    """
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X[:200, 0] = 500.0                                # forces the clip
    rows = {r["col"]: r for r in
            D.audit(X, list("abc"), scaling="robust-equalised")}
    assert rows["a"]["clip_frac"] > 0.01
    assert "clip" in rows["a"]["flags"]


def test_unknown_scaling_still_raises():
    from bluse import diagnostics as D
    with pytest.raises(ValueError):
        D.scale(np.zeros((10, 2)), "not-a-mode")
```

- [ ] **Step 2: Run them and watch them fail.**

- [ ] **Step 3: Implement.** In `scale()`, add the branch before the `robust`
  return; in `audit()`, change the `clip_frac` guard from
  `scaling == "robust"` to `scaling.startswith("robust")` and thread
  `kinds`/`columns` through to `scale`.

- [ ] **Step 4: Run the full suite** — `uv run pytest -q`, 70+ passing.

- [ ] **Step 5: Commit.**

---

### Task 4: CLI wiring, the `eom` warning, and the metrics JSON

**Files:**
- Modify: `src/bluse/track_b_cluster.py`
- Test: `tests/unit/test_matching_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
def test_equalised_with_eom_warns(capsys):
    """
    Spec section 2.1: warn loudly, still run. Refusing would block reproducing
    the measurement; running silently would let someone quote a broken number.
    """
    from bluse.track_b_cluster import warn_if_eom_equalised
    warn_if_eom_equalised("robust-equalised", "eom")
    out = capsys.readouterr().out
    assert "WARNING" in out and "0.519" in out and "leaf" in out
    warn_if_eom_equalised("robust-equalised", "leaf")
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run it and watch it fail.**

- [ ] **Step 3: Implement**
  - add `robust-equalised` to the `--scaling` choices and to
    `feature_matrix`'s docstring table;
  - `warn_if_eom_equalised(scaling, method)` printing the agreed text —
    name the measurement (0.519 → 0.044), say why (equalisation amplifies the
    locally weakest column; `eom`'s root-level stability comparison is fragile
    to that), and name the fix (`leaf`, or `--scaling robust`);
  - call it once in `main()` after argument parsing;
  - print the per-column weights in the `--report` table as a `weight` column;
  - record `scaling`, the weights and `equalising_weights`' `info` in the
    metrics JSON next to `matching` (Task from P0-1: the JSON is the
    machine-readable record, stdout is not).

- [ ] **Step 4: Verify end to end**

```bash
uv run bluse-cluster --file sband_short --scaling robust-equalised \
  --cluster-selection-method leaf --match --match-families 36
uv run bluse-cluster --file sband_short --scaling robust-equalised \
  --cluster-selection-method eom --report      # must print the WARNING
```

- [ ] **Step 5: Commit.**

---

### Task 5: Bench wiring

**Files:**
- Modify: `src/bluse/bench/app.py`,
  `src/bluse/bench/templates/_controls.html`,
  `src/bluse/bench/templates/_results.html`

- [ ] **Step 1: Add the dropdown option** `robust + equalised — target equal
  contribution` to the scaling select in `_controls.html`.

- [ ] **Step 2: Cache the weights** per `(dataset key, tuple(columns))` on the
  dataset object, beside `ds.embedding`. The closed form is 1.5 ms and needs no
  cache; a k-NN strategy is 9.3 s per iteration on the larger files and does.
  Implement the cache regardless of which strategy won — it costs four lines
  and removes a cliff if the strategy is ever revisited.

- [ ] **Step 3: Show the weight per column in the feature rail**, beside the
  existing `share / knn` figures, and only when the equalised mode is selected.

- [ ] **Step 4: Show the `eom` warning** on the results panel when the mode and
  method are combined, using the same wording as the CLI.

- [ ] **Step 5: Verify by hand**

```bash
uv run bluse-bench &
curl -s -X POST localhost:8000/dataset \
  --data "file=sband_short&sample=35000&seed=0" | grep -o "equalised"
```
Then cluster once from the UI with `leaf` + equalised and confirm the rail
shows weights and the numbers match the CLI for the same configuration.
**Acceptance criterion 4 is that the two agree exactly** — check it here, not
by assertion.

- [ ] **Step 6: Commit.**

---

### Task 6: acceptance, documentation, and the review branch

**Files:**
- Modify: `aug_2026_workshop/acceptance.py`, `aug_2026_workshop/README.md`,
  `docs/TODO.md`, `docs/scaling-experiment-2026-09.md`

- [ ] **Step 1: Extend `acceptance.py`** with a `leaf` + `robust-equalised`
  block reporting famARI at 16/24/36/44 families, narrow share and median span.

- [ ] **Step 2: Run it and check acceptance criteria 1 and 6**

```bash
uv run python aug_2026_workshop/acceptance.py > /tmp/acc.json
```
Criterion 1: famARI ≥0.60 at 36 families, median span ≤120 MHz.
Criterion 6: the `robust` numbers are unmoved — `eom` 72 / 0.776% / 0.5190 and
`leaf` 2162 / 6.820% / 0.1077.

- [ ] **Step 3: Update the docs.** `scaling-experiment-2026-09.md` gets a
  pointer to the shipped mode; `TODO.md` closes P1-5 and records what shipped;
  the workshop README documents the mode and its `leaf`-only constraint.

- [ ] **Step 4: Full suite plus a clean-tree check**

```bash
uv run pytest -q
git status --short
```

- [ ] **Step 5: Open the PR for review**

```bash
git push -u origin feature/equalising-scaling
gh pr create --base main --title "P1-5: contribution-equalising scaling" \
  --body "<the spec's §3 table, the Task 2 result, and the acceptance numbers>"
```

The PR body must state which strategy won and why, including the candidates
that lost, and must carry the `leaf`-only constraint prominently.
