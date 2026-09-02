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
  families, median span 265.0 MHz.
- **Rank on family ARI, then median family span. Do NOT rank on
  `narrow_frac`.** At 36 families it reads 0.194% against 0.168%, which on
  34,933 clustered hits is 68 hits against 59 — a nine-hit difference that
  cannot carry a decision. Report it; never rank on it. Spec §6.1 says why.
- **`matching.match(..., n_families=)` exists** — it landed in P0-1 (`84e42ff`)
  along with `--match-families`. Confirmed on this branch; no need to add it.
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
  knn_sample=20_000, seed=0, with_info=True) -> (np.ndarray, dict)`. `info`
  keys: `strategy`, `iters`, `skipped` (list of column names),
  `max_dev_global`, `max_dev_knn`, `dev_trace` (per-iteration k-NN deviation),
  `weight_min`, `weight_max`, `spread_warning`.

**`with_info=False` is not optional polish.** The `info` block calls
`_shares_knn`, which costs **1,311 ms** on sband_short at `knn_sample=20_000`
(measured; 435 ms at 5,000) against 1.7 ms for the weights themselves.
`scale()` is called in `cluster()` per run, in `audit()` per rail render, and
once per seed in `/stability`, so leaving it on would make the mode advertised
as free cost ~1.3 s everywhere — and the Task 5 cache would hide it in the
Bench only. `scale()` passes `with_info=False`; only `--report`, the rail and
the Task 2 experiment ask for `info`.

Keep `knn_sample=20_000`, which is `audit()`'s default since P0-3. (A review
suggested 5,000 "matching `audit()`"; that was `audit()`'s pre-P0-3 default.)

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


def test_extreme_weight_spread_is_flagged():
    """
    The sd > 1e-12 floor only catches exactly-constant columns. A column with
    sigma = 0.01 draws weight ~100 and passes silently, and the failure then
    presents as "clustering got worse after I added a feature", which is close
    to undiagnosable. 2.0 is the measured scale to reason from: it is where eom
    breaks.
    """
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.normal(0, 1, 2000), rng.normal(0, 1, 2000),
                         rng.normal(0, 0.01, 2000)])
    w, info = D.equalising_weights(Z, strategy="closed")
    assert info["spread_warning"]
    assert w.max() <= D.EQUALISE_MAX_WEIGHT


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
    """
    Mean-1 over the free columns, cap, and pin the frozen ones at 1.0.

    NOTE the cap is applied AFTER mean-normalisation and the result is not
    re-normalised, so a capped weight vector has mean != 1. That is harmless
    for the percentile matching cut, which is scale-invariant, but
    derive_cut_quantile and any explicit `cut=` are absolute distances and
    would shift. Re-normalising would push weights back over the cap, so this
    is a deliberate choice, not an oversight.

    EQUALISE_MAX_WEIGHT is a hard ceiling applied even when `cap` is None: a
    column with sigma = 0.01 would otherwise draw weight ~100 unnoticed.
    """
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

  **Plus four diagnostics that are not candidates to ship**, because they price
  the k-NN target and close the `f02` question the spec's §2.1 leaves open:
  - `closed` with `f09_temporal_skew` hand-set to its equal *local* share. If
    this beats plain `closed`, the k-NN target is buying something real and C
    is the right answer; if not, A wins on the merits rather than on the
    tie-break.
  - `closed` with `f02_abs_drift` pinned to weight 1.0. The closed form gives
    `f02` the **largest weight in the matrix (1.594)** — smallest post-robust
    σ, because the tie that inflates its IQR is divided out — which is the
    P0-2 pathology in miniature.
  - `f02` **dropped from the matrix entirely**. P0-2 tested `f02` *pinned*
    and P1-4 tested *reworkings*; exclusion has never been tested, and it is
    the first thing a reader will ask.
  - one run with a **synthetic boolean column appended**, so the
    `boolean`/`flag` guard is exercised on the real code path and not only in a
    unit test. No registered feature is boolean today.

- [ ] **Step 2: Also record, for each candidate**: fit cost on sband_short and
  on lband_short, and the 35k-sample-vs-population weight deviation. The
  measured figures for the undamped k-NN chase are 9.3 s/iteration and 29.5%
  max deviation; for the closed form 1.5 ms and 0.78%. Reproducibility of the
  *weights* is part of the score, not a footnote — it is acceptance criterion
  4b.

- [ ] **Step 2b: Record `max_dev_knn` PER ITERATION, not just its final
  value.** This distinguishes two different failures that the endpoint alone
  cannot separate:
  - if it **falls and plateaus** near 0.33, equal local share is *unattainable*
    for some column and the target is mis-specified. Candidate B is then
    ill-posed and should be dropped outright rather than beaten on score, and
    candidate C — bounded refinement toward an unreachable target — is the
    honest form of the idea.
  - if it is **still falling** at eight iterations, the iteration is merely
    slow and a damped version may reach the target.

  A multiplicative update toward an unreachable target diverges monotonically,
  which is what weights spreading 0.044–10.347 in *opposite* directions looks
  like. Four extra numbers, and it turns "B lost" into "B is ill-posed", which
  is the more durable finding.

- [ ] **Step 3: Pick the winner.** Rank on family ARI at 36 families first,
  then median family span. **Narrow share is reported but NOT ranked on** — at
  36 families the baseline and equalised readings are 68 hits against 59, and a
  nine-hit difference is indistinguishable from a coin toss (spec §6.1).
  **If the closed form is within 0.03 famARI of the best, it wins** — determinism, 350× lower cost and
  sample-stability are worth that margin, and the spec says so in advance so
  the rule cannot be chosen after seeing the numbers.

  *One difference to state in the write-up:* the experiment builds
  `Z = scale(X, "robust")` over the full matrix, computes `w`, and clusters
  `Z * w` with `scaling="none"`, whereas production computes `w` inside
  `scale()` from the rows handed to it. Same answer on the CLI path, different
  on the Bench path, so the experiment's numbers are the CLI-path numbers.

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

    A WIDENED GUARD IS NOT ENOUGH, and this is the trap. clip_frac is measured
    as (|Z| >= CLIP).mean() on whatever scale() returned, and the equalised
    mode returns base * w. A value sitting exactly on the clip at +/-5 in the
    base is no longer at +/-5 after weighting. On this very fixture, measured:
    clip_frac on the base is 0.050, the closed-form weight for column a is
    0.661, so the clipped rows land at 3.31 and the threshold test misses every
    one of them -- clip_frac on base * w is 0.000.

    That is WORSE than the bug it replaces: the original reported 0.0 for a
    mode that does not clip; this would report 0.0 for a mode that does.
    clip_frac must therefore be computed on the ROBUST BASE, before weights.
    """
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X[:200, 0] = 500.0                                # forces the clip
    rows = {r["col"]: r for r in
            D.audit(X, list("abc"), scaling="robust-equalised")}
    assert rows["a"]["clip_frac"] > 0.01
    assert "clip" in rows["a"]["flags"]


def test_clip_frac_matches_the_unweighted_robust_mode():
    """The clip is a property of the base transform, so the two modes must
    report the same fraction on the same data."""
    from bluse import diagnostics as D
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    X[:200, 0] = 500.0
    a = {r["col"]: r for r in D.audit(X, list("abc"), scaling="robust")}
    b = {r["col"]: r for r in
         D.audit(X, list("abc"), scaling="robust-equalised")}
    for c in "abc":
        assert a[c]["clip_frac"] == pytest.approx(b[c]["clip_frac"])


def test_unknown_scaling_still_raises():
    from bluse import diagnostics as D
    with pytest.raises(ValueError):
        D.scale(np.zeros((10, 2)), "not-a-mode")
```

- [ ] **Step 2: Run them and watch them fail.**

- [ ] **Step 3: Implement.**
  - In `scale()`, add the `robust-equalised` branch: build the robust base,
    call `equalising_weights(base, ..., with_info=False)`, return `base * w`.
  - In `audit()`, **compute `clip_frac` from the robust base, not from the
    returned matrix.** Widening the guard to `scaling.startswith("robust")` is
    necessary but NOT sufficient — see the test above. Either compute
    `base = scale(raw, "robust")` when `scaling.startswith("robust")` and take
    `clip_frac` from that, or have `scale()` optionally hand back the clip mask
    so nothing is recomputed. Prefer the second if the extra pass shows up in
    the rail render; measure before choosing.
  - Thread `kinds`/`columns` through `audit()` to `scale()`.
  - **`iqr_scaled` changes meaning** under the equalised mode: it becomes the
    weighted IQR. That is arguably the more useful number — it is the spread
    HDBSCAN sees — but the rail must say which it is showing rather than let
    the column silently change meaning between modes.

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
  Label the `iqr_scaled` column so it is clear it is the *weighted* IQR in this
  mode (Task 3 step 3), rather than letting it change meaning silently.

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
Criterion 1: famARI ≥0.60 at 36 families, median span ≤120 MHz. Narrow share
is reported but is **not** a criterion — spec §6.1.
Criterion 4b: record the Bench-sample vs full-population weight deviation for
the shipped strategy on `sband_short` and `lband_short`. ≤1% passes; if an
iterative strategy won at ~29.5%, that number goes in the acceptance record as
a stated limitation, not a footnote.
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
that lost, and must carry the `leaf`-only constraint prominently. It must also
say that family ARI is **granularity-relative** — the same configuration reads
0.6659 at 36 families and 0.7683 at 16 — so the new number is never to be
quoted against the 0.5190 from the earlier `eom` work, which was measured at a
different family count.
