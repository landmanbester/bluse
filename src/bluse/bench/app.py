#!/usr/bin/env python3
"""
Cluster Bench -- interactive HDBSCAN exploration for the BLUSE feature matrix.

    bluse-bench                       # then open http://127.0.0.1:8000
    bluse-bench --port 8080 --host 0.0.0.0

Reads <workspace>/features/*_features.parquet, so run `bluse-features` first.
Stamp thumbnails come from <workspace>/data/<file>.h5.

Why this exists: the Track B defaults cluster badly, and the reason is not
obvious from a static run. Feature scaling turned out to matter more than any
HDBSCAN parameter -- GLOBULAR's transforms leave interquartile ranges spanning
0.036 to 5.88, so Euclidean distance became drift rate and almost nothing else.
This tool makes that visible and adjustable.

Design notes:
  - The embedding is computed ONCE per dataset and never recomputed. Only
    cluster labels change with hyperparameters, so a re-cluster ships an
    Int32Array of labels, not a new plot.
  - Everything is cached by parameter hash, so revisiting a configuration is
    instant and the run history is free.
  - Binary endpoints (.bin) rather than JSON: 35k points is 280 KB as
    Float32Array and about 1.5 MB as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Query, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from .. import features as F
from .. import paths
# scale() lives in diagnostics so the Bench and the CLI cannot drift
# apart on the one thing that matters most to the metric.
from .. import diagnostics
from .. import matching
from .. import metrics
from ..diagnostics import scale

# Static assets and templates ship inside the wheel, so they -- and only they --
# are still located relative to this file. Everything the user supplies (feature
# matrices, stamp cubes) resolves against the workspace instead.
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Cluster Bench")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))

# A run routinely produces 200+ clusters. Cycling a palette over all of them
# turns the scatter into confetti in which the 15,000-point RFI family and a
# 4-point singleton look equally important. So colour is assigned by RANK, not
# by label: the TOP_N largest clusters get distinct hues, everything smaller is
# drawn in one muted grey. The eye then reads the structure that matters and
# the long tail stays visible as texture. Must match static/scatter.js.
COLOURS = [
    "#e8734a", "#4aa3e8", "#7fc96b", "#c77dd6", "#e8c34a", "#5fd0c0",
    "#e06b8b", "#8f9ce8", "#b8d24a", "#4ac2e8", "#e89a4a", "#9ad67f",
]
TOP_N = len(COLOURS)
NOISE_COLOUR = "#39424f"
MINOR_COLOUR = "#5c6779"


def colour_for(lab, rank=None):
    """Colour of a cluster given the rank map {label: position by size}."""
    if lab is None or int(lab) < 0:
        return NOISE_COLOUR
    if rank is None:
        return COLOURS[abs(int(lab)) % len(COLOURS)]
    i = rank.get(int(lab))
    return COLOURS[i] if i is not None and i < TOP_N else MINOR_COLOUR


templates.env.globals["colour"] = colour_for

# Cache-buster for the static assets: without it a browser happily serves a
# stale app.css against a freshly restarted server, which looks exactly like a
# CSS change that "did not work".
templates.env.globals["assetv"] = lambda: str(int(os.path.getmtime(
    os.path.join(HERE, "static", "app.css")) +
    os.path.getmtime(os.path.join(HERE, "static", "scatter.js"))))


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    key: str
    name: str
    df: pd.DataFrame                 # provenance columns only
    raw: np.ndarray                  # (n, n_feat) normalised, pre-scaling
    columns: list[str]
    embedding: dict = field(default_factory=dict)   # method -> (n,2) float32
    scaler_stats: dict = field(default_factory=dict)  # full-population med/IQR


@dataclass
class Run:
    run_id: str
    labels: np.ndarray
    summary: pd.DataFrame
    stats: dict
    params: dict
    rank: dict = field(default_factory=dict)   # cluster label -> size rank
    epochs: list = field(default_factory=list)   # rendered epoch trace
    families: np.ndarray = None
    match_info: dict = field(default_factory=dict)


DATASETS: dict[str, Dataset] = {}
RUNS: dict[str, Run] = {}
HISTORY: list[str] = []


def available_files():
    feat = paths.features_dir()
    if not os.path.isdir(feat):
        return []
    out = []
    for f in sorted(os.listdir(feat)):
        if f.endswith("_features.parquet") and not f.startswith("all_"):
            out.append(f.replace("_features.parquet", ""))
    if os.path.exists(os.path.join(feat, "all_features.parquet")):
        out.append("all")
    return out


def feature_columns(df):
    """Normalised feature columns, excluding the boolean saturation flags."""
    return [c + "_n" for c in F.all_columns()
            if c + "_n" in df.columns and not c.endswith("_saturated")]


PROV = ["id", "row", "file", "obsid", "sourceName", "beam", "frequency",
        "driftRate", "snr", "n_beams", "beam_frac", "weak_label",
        "f12_bandwidth_hz", "f08_turning_bw_saturated"]


def load_dataset(name, sample, seed):
    key = f"{name}:{sample}:{seed}"
    if key in DATASETS:
        return DATASETS[key]

    path = os.path.join(paths.features_dir(), f"{name}_features.parquet")
    df = pd.read_parquet(path)
    df = df[df.feature_ok].reset_index(drop=True)
    cols = feature_columns(df)
    X = np.array(df[cols].to_numpy(dtype=np.float64), copy=True)
    good = np.isfinite(X).all(axis=1)
    df, X = df[good].reset_index(drop=True), X[good]

    # D-4: fit the scaling statistics on the FULL population, then sample.
    # GLOBULAR's requirement that scaling be global and pre-batching is already
    # satisfied upstream -- features.normalise() fits its transforms globally at
    # extraction time -- so this is the second stage only.
    #
    # Be honest about what this fixes. Measured, a 35k-row IQR matches the
    # 1,281,878-row population to better than 1.1% on 14 of 15 columns, worst
    # 6.4%. It is NOT why a Bench configuration fails to reproduce in
    # bluse-cluster. That is because the Bench clusters 35k rows and the CLI
    # clusters 1.28M, and because two runs of one configuration at different
    # seeds already agree at only ARI 0.024.
    q75f, q25f = np.percentile(X, [75, 25], axis=0)
    scaler_stats = {"median": np.median(X, axis=0), "iqr": q75f - q25f}

    if sample and len(df) > sample:
        idx = np.sort(np.random.default_rng(seed).choice(len(df), sample,
                                                         replace=False))
        df, X = df.iloc[idx].reset_index(drop=True), X[idx]

    keep = [c for c in PROV if c in df.columns]
    ds = Dataset(key=key, name=name, df=df[keep].copy(), raw=X, columns=cols,
                 scaler_stats=scaler_stats)
    DATASETS[key] = ds
    return ds


def embed(ds, method, cols=None, scaling="robust"):
    """
    2-D projection for the scatter.

    This used to hardcode scale(ds.raw, "robust") over ALL columns, so the plot
    geometry was frozen: turning features off or switching the scaling could not
    move a single point, which made every run look alike no matter what the
    clusterer actually did. It now projects exactly the matrix HDBSCAN sees.
    """
    cols = list(ds.columns) if cols is None else list(cols)
    ck = (method, scaling, tuple(cols))
    if ck in ds.embedding:
        return ds.embedding[ck]
    use = [ds.columns.index(c) for c in cols]
    Z = scale(ds.raw[:, use], scaling)
    if method == "umap":
        try:
            import umap
            emb = umap.UMAP(n_neighbors=15, min_dist=0.01, n_components=2,
                            random_state=0).fit_transform(Z)
        except Exception:
            emb = PCA(n_components=2, random_state=0).fit_transform(Z)
    else:
        emb = PCA(n_components=2, random_state=0).fit_transform(Z)
    emb = np.asarray(emb, dtype=np.float32)
    lo, hi = emb.min(axis=0), emb.max(axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    ds.embedding[ck] = ((emb - lo) / span).astype(np.float32)
    return ds.embedding[ck]


def run_hdbscan(X, mcs, ms, method="eom"):
    """
    One HDBSCAN pass.

    cluster_selection_epsilon is deliberately absent. sklearn's epsilon_search
    (_tree.pyx:606) compares epsilon against the RECIPROCAL of a leaf's split
    distance, not the distance. Leaf splits in this feature space are >~1.7, so
    1/d is ~0.55: every epsilon below that leaves the leaf set bit-identical to
    epsilon=0, and every epsilon above it reaches traverse_upwards, which
    assigns length-1 arrays into scalar cdefs and raises

        TypeError: only 0-dimensional arrays can be converted to Python scalars

    The no-op region and the crash region tile the domain -- there is no value
    that both works and does not crash -- so the knob is gone rather than
    re-ranged. Measured: eps 0.0/0.05/0.18/0.5 give ARI 1.000 against each
    other; 5.0 raises on 14/14 batches.

    `method` selects EOM or leaf extraction, and it is a genuine USER CHOICE
    rather than a default waiting to be tuned, because the two win on different
    axes. With allow_single_cluster=False, EOM makes one stability comparison at
    the root of the condensed tree: either the root's two children win (k=2) or
    it descends to the leaves (k~200), with nothing in between BY CONSTRUCTION.
    Raising min_samples biases which side of that knife edge you land on but
    does not remove it. leaf takes the condensed tree's leaves directly, never
    makes the root comparison, and is not bistable.

    Measured on sband_short, 3 seeds -- narrow-cluster share, and membership
    ARI after matching at the p50 cut:

                    clusters  narrow %   family ARI
        eom               72     0.776        0.519
        leaf           2,162     6.820        0.108

    leaf wins coherence 8.8x; eom-plus-matching wins reproducibility 4.8x.
    Which you want depends on whether you are building a taxonomy or ranking
    candidates, so the tool asks rather than deciding.
    """
    # Built OUTSIDE the try: an unsupported keyword raises TypeError at
    # construction, and the except below would otherwise swallow it and hand
    # back an all-noise result that looks like a legitimate clustering.
    # copy=True matches sklearn's future default (1.10) and silences its
    # FutureWarning; it is inert for us because it only applies when
    # metric="precomputed", and ours is euclidean.
    est = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                  cluster_selection_method=method,
                  copy=True,
                  n_jobs=-1)
    try:
        return est.fit_predict(X)
    except ValueError:
        return np.full(len(X), -1, dtype=np.int32)


def cluster(ds, cols, scaling, mode, mcs, ms, epochs, batch, seed,
            method="eom"):
    use = [ds.columns.index(c) for c in cols]
    X = scale(ds.raw[:, use], scaling)
    n = len(X)
    labels = np.full(n, -1, dtype=np.int32)
    trace = []

    # origin[cluster_id] = (epoch, batch) it was discovered in. Cluster ids are
    # a discovery-order serial, NOT a batch number -- one batch usually mints
    # several -- so the mapping has to be recorded rather than inferred.
    origin = {}

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
                    # HDBSCAN mints local ids 0..k-1 in EVERY batch. The old
                    # code offset by epoch only, so batch 0's cluster 3 and
                    # batch 7's cluster 3 both became 100003 and were fused.
                    # On sband_short that collapsed 731 real groups into 205
                    # reported ones -- 526 unrelated clusters merged, which is
                    # why the top ids carried the GLOBAL medians and why
                    # n_clusters tracked max-over-batches instead of the total.
                    # A running offset keeps every batch's clusters distinct.
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


def summarise_basic(labels):
    """Cluster ids and sizes only. Used where no provenance frame is at hand."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    return pd.DataFrame({"cluster": ids.astype(int), "n": counts.astype(int)})


def summarise(df, labels, origin=None):
    origin = origin or {}
    rows = []
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        d = df[m]
        rows.append({
            "cluster": int(lab), "n": int(m.sum()),
            "epoch": origin.get(int(lab), (0, 0))[0],
            "batch": origin.get(int(lab), (0, 0))[1],
            "rfi_pct": float((d.weak_label == 1).mean() * 100),
            "freq": float(d.frequency.median()),
            "span": float(d.frequency.max() - d.frequency.min()),
            "snr": float(d.snr.median()),
            "beams": float(d.n_beams.median()),
            "obs": int(d.obsid.nunique()),
        })
    cols = ["cluster", "n", "epoch", "batch", "rfi_pct", "freq", "span",
            "snr", "beams", "obs"]
    if not rows:
        # Every point is noise. Without the explicit columns the empty frame has
        # none, and sort_values("n") raises KeyError before the caller's
        # all-noise message can ever be rendered.
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "files": available_files(),
        "has_features": bool(available_files()),
    })


@app.post("/dataset", response_class=HTMLResponse)
def pick_dataset(request: Request, file: str = Form(...),
                 sample: int = Form(35000), seed: int = Form(0)):
    t0 = time.time()
    try:
        ds = load_dataset(file, sample, seed)
    except FileNotFoundError:
        return HTMLResponse(
            f'<p class="error">No feature file for {file}. '
            f'Run <code>bluse-features</code> first.</p>')

    # Raw spread only. The rail used to compute a scaled IQR here, never use
    # it, and caption the raw bars "after scaling" -- which is false in the
    # worst way: robust scaling divides each column BY its IQR, so every scaled
    # IQR is exactly 1.000 and the bars would all be equal. What the bars
    # actually show is how unequal the columns are BEFORE scaling, i.e. what
    # the scaling control is there to fix.
    # The raw IQR bar stays -- it answers "how unequal are the columns before
    # scaling", which is what the scaling control exists to fix. It cannot
    # answer what each column CONTRIBUTES to the distance HDBSCAN takes, and
    # that is where the defects are, so the audit supplies the rest.
    kinds = {c + "_n": k for c, k in F.column_kinds().items()}
    rows = diagnostics.audit(ds.raw, ds.columns, scaling="robust",
                             kinds=kinds, min_samples=8)

    sat = {}
    if "f08_turning_bw_saturated" in ds.df.columns:
        sat["f08_turning_bw_hz_n"] = float(ds.df.f08_turning_bw_saturated.mean() * 100)

    iqr_max = max((r["iqr_raw"] for r in rows), default=1.0) or 1.0
    feats = []
    for r in rows:
        feats.append({
            **r,
            "iqr_rel": float(r["iqr_raw"] / iqr_max),
            "saturated": sat.get(r["col"]),
            "on": not r["col"].startswith("f08_"),
            "share_pct": 100.0 * r["share_global"],
            "share_knn_pct": 100.0 * r["share_knn"],
            "equal_pct": 100.0 * r["equal_share"],
            "tie_pct": 100.0 * r["max_tie_fraction"],
            "clip_pct": 100.0 * r["clip_frac"],
        })
    return templates.TemplateResponse(request, "_controls.html", {
        "ds": ds, "features": feats,
        "n": len(ds.df), "load_ms": int((time.time() - t0) * 1000),
    })


def emb_sig(key, method, scaling, cols):
    """
    Identity of an embedding; the client refetches when this changes.

    The dataset key belongs in here. Without it two files with the same feature
    names and scaling hashed identically, so a run on a newly loaded dataset
    could tell the client its geometry was current when it was still the old
    file's. Nothing exercised that while every load refetched anyway, but it
    was one reordering away from a plot that silently showed the wrong points.
    """
    return hashlib.sha1(json.dumps([key, method, scaling, sorted(cols)]).encode()
                        ).hexdigest()[:12]


@app.get("/embedding.bin")
def embedding_bin(request: Request, key: str, method: str = "pca",
                  scaling: str = "robust"):
    ds = DATASETS.get(key)
    if ds is None:
        return Response(status_code=404)
    cols = [c for c in request.query_params.getlist("feat") if c in ds.columns]
    if len(cols) < 2:
        cols = list(ds.columns)
    return Response(embed(ds, method, cols, scaling).tobytes(),
                    media_type="application/octet-stream",
                    headers={"X-Emb-Sig": emb_sig(key, method, scaling, cols)})


@app.post("/cluster", response_class=HTMLResponse)
async def do_cluster(request: Request):
    form = await request.form()
    key = form["key"]
    ds = DATASETS.get(key)
    if ds is None:
        return HTMLResponse('<p class="error">Dataset expired. Load it again.</p>')

    # The file select lives outside this form and is pulled in via hx-include.
    # If it has moved on from the loaded dataset, clustering the stale key would
    # return a cached run for the PREVIOUS file and look like nothing happened.
    chosen = form.get("file")
    if chosen and chosen != ds.name:
        return HTMLResponse(
            f'<p class="error">Showing <b>{ds.name}</b>, but the file selector '
            f'says <b>{chosen}</b>. Press Load to switch datasets, then '
            f'cluster.</p>')

    cols = [c for c in form.getlist("feat") if c in ds.columns]
    if len(cols) < 2:
        return HTMLResponse('<p class="error">Keep at least two features on. '
                            'Distance needs something to compare.</p>')

    p = dict(scaling=form.get("scaling", "robust"),
             mode=form.get("mode", "epochs"),
             # eom or leaf. A USER CHOICE, not a default awaiting tuning:
             # measured on sband_short, leaf wins coherence 8.8x (narrow share
             # 6.820% against 0.776%) while eom-plus-matching wins
             # reproducibility 4.8x (family ARI 0.519 against 0.108).
             csm=form.get("csm", "eom"),
             match=form.get("match", "off"),
             match_pct=int(form.get("match_pct", 50)),
             mcs=int(form.get("mcs", 4)),
             # min_samples defaults to 8, not 2. sklearn counts the point
             # itself, so ms=1 and ms=2 are the same call: no core-distance
             # smoothing at all, i.e. pure single linkage. That put EOM on a
             # knife edge -- identical 3000-point draws returned either k=2
             # holding 99.7% of points or ~200 microclusters with 40% noise,
             # a 112x swing on the shuffle alone. ms=8 collapses that to
             # k in [2,12]. See AGENTS.md.
             ms=int(form.get("ms", 8)),
             epochs=int(form.get("epochs", 8)),
             batch=int(form.get("batch", 3000)),
             seed=int(form.get("seed", 0)))

    sig = hashlib.sha1(json.dumps([key, sorted(cols), p],
                                  sort_keys=True).encode()).hexdigest()[:12]
    cached = sig in RUNS
    if not cached:
        t0 = time.time()
        labels, _, origin, trace = cluster(ds, cols, p["scaling"], p["mode"],
                                           p["mcs"], p["ms"], p["epochs"],
                                           p["batch"], p["seed"], p["csm"])
        elapsed = time.time() - t0
        summary = summarise(ds.df, labels, origin)
        noise = labels == -1
        conf = noise & (ds.df.n_beams.to_numpy() <= 4)
        rank = {int(c): i for i, c in enumerate(summary["cluster"])}
        stats = dict(metrics.quality(labels, ds.df))
        stats.update({
            "noise": int(noise.sum()),
            "confined": int(conf.sum()),
            "seconds": elapsed,
            "n_features": len(cols),
        })
        # Cluster ids are batch artefacts -- one physical population is minted
        # afresh in every batch and every epoch. Families are the level at
        # which that cancels, measured: eom membership ARI 0.028 -> 0.519.
        families, match_info = None, {}
        if p["match"] == "on":
            Xs = scale(ds.raw[:, [ds.columns.index(c) for c in cols]],
                       p["scaling"])
            families, match_info = matching.match(labels, Xs,
                                                  pct=p["match_pct"])
            fq = metrics.quality(families, ds.df)
            match_info["family_narrow_frac"] = fq["narrow_frac"]
            match_info["family_median_span_mhz"] = fq["median_span_mhz"]

        RUNS[sig] = Run(sig, labels, summary, stats, p, rank,
                        metrics.epoch_trace(trace, len(labels)),
                        families, match_info)
        if sig in HISTORY:
            HISTORY.remove(sig)
        HISTORY.insert(0, sig)
        del HISTORY[12:]

    run = RUNS[sig]
    resp = templates.TemplateResponse(request, "_results.html", {
        "run": run, "cached": cached, "rank": run.rank,
        "colours": COLOURS, "minor": MINOR_COLOUR, "noise": NOISE_COLOUR,
        "summary": run.summary.head(40).to_dict("records"),
        "history": [RUNS[h] for h in HISTORY if h in RUNS],
        "key": key,
    })
    # The client needs the size order to colour by rank; 12 ints is cheap.
    top = [int(c) for c in run.summary["cluster"].head(TOP_N)]
    resp.headers["HX-Trigger"] = json.dumps({"clusterDone": {
        "run": run.run_id, "top": top,
        # The embedding now depends on the feature set and the scaling, so the
        # client has to know when the geometry it is showing went stale.
        "emb": emb_sig(key, form.get("embed", "pca"), p["scaling"], cols),
    }})
    return resp


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
    s_cl = metrics.stability(run_fn, seeds=tuple(range(n_seeds)))

    # The same question one level up. Cluster ids are batch artefacts, so the
    # interesting number is whether FAMILIES reproduce where clusters do not.
    def run_fam(seed):
        labels, X, _, _ = cluster(ds, cols, p["scaling"], p["mode"], p["mcs"],
                                  p["ms"], p["epochs"], p["batch"], seed,
                                  p["csm"])
        fam, _ = matching.match(labels, X)
        return fam

    s_fam = metrics.stability(run_fam, seeds=tuple(range(n_seeds)))
    elapsed = time.time() - t0
    return templates.TemplateResponse(request, "_stability.html", {
        "s": s_cl, "f": s_fam, "params": p, "seconds": elapsed,
    })


@app.get("/labels.bin")
def labels_bin(run: str):
    r = RUNS.get(run)
    if r is None:
        return Response(status_code=404)
    return Response(r.labels.astype(np.int32).tobytes(),
                    media_type="application/octet-stream")


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


@app.get("/hit", response_class=HTMLResponse)
def hit(request: Request, key: str, i: int, run: str = ""):
    ds = DATASETS.get(key)
    if ds is None or i < 0 or i >= len(ds.df):
        return HTMLResponse("")
    row = ds.df.iloc[i]
    lab = None
    if run in RUNS:
        lab = int(RUNS[run].labels[i])
    return templates.TemplateResponse(request, "_hit.html", {
        "r": row, "i": i, "key": key, "label": lab,
    })


@app.get("/stamp.png")
def stamp(file: str, row: int):
    """Render one hit's waterfall. The stamp is why this data is worth having."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    path = os.path.join(paths.data_dir(), f"{file}.h5")
    try:
        with h5py.File(path, "r") as h:
            cube = h["data"][row]
            nch = int(h["numChannels"][row])
    except Exception:
        return Response(status_code=404)

    img = np.asarray(cube)
    while img.ndim > 2:
        img = img[0]
    valid = ~np.all(img == -1, axis=0)
    if valid.any():
        img = img[:, valid]
    med = np.median(img)
    img = np.clip(img / (med if med > 0 else 1), 1e-3, None)

    fig, ax = plt.subplots(figsize=(3.0, 2.6), facecolor="#161b22")
    ax.imshow(img, aspect="auto", origin="upper", cmap="magma",
              norm=LogNorm(), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2a323e")
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#161b22")
    plt.close(fig)
    return Response(buf.getvalue(), media_type="image/png")


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="Cluster Bench")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true",
                    help="restart on source edits (development only)")
    ap.add_argument("--allow-empty", action="store_true",
                    help="start even with no feature matrices in the "
                         "workspace. The file list is rebuilt per request, so "
                         "a page refresh picks them up once they exist")
    paths.add_workspace_arg(ap)
    a = ap.parse_args()
    paths.set_workspace(a.workspace)

    found = available_files()
    say = lambda m: print(m, flush=True)
    say(f"\n  {paths.banner()}")

    # Every other command exits here via paths.require_data_dir(). The bench
    # used to print one warning line and start anyway, which is how you end up
    # staring at an empty file selector wondering where your features went --
    # the banner scrolls past and the page looks like a bug in the app. If we
    # cannot see any feature matrices, say so the same way the rest of the CLI
    # does and stop, unless the caller explicitly asked for an empty start.
    if not found and not a.allow_empty:
        sys.exit(paths.missing_workspace_message(
            "feature matrices (*_features.parquet)", paths.features_dir())
            + "\n\nIf the workspace is right and you have not extracted "
              "features yet, run bluse-features -- or start anyway with "
              "--allow-empty.")

    if found:
        say(f"  datasets: {', '.join(found)}")
    else:
        say(f"  NO feature matrices in {paths.features_dir()} -- "
            f"the file selector will be empty until you run bluse-features")
    say(f"\n  Cluster Bench -> http://{a.host}:{a.port}\n")

    # --reload needs an import string, and uvicorn re-imports it in a fresh
    # process that inherits none of our argparse state -- so the workspace has
    # to travel via the environment or the child resolves it from its own cwd.
    if a.reload:
        os.environ[paths.ENV_VAR] = paths.workspace()
    uvicorn.run("bluse.bench.app:app" if a.reload else app,
                host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
