#!/usr/bin/env python3
"""
Cluster Bench -- interactive HDBSCAN exploration for the BLUSE feature matrix.

    uv run explorer/app.py            # then open http://127.0.0.1:8000
    .venv/bin/python explorer/app.py --port 8080

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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import features as F  # noqa: E402

FEAT_DIR = os.path.join(ROOT, "features")
DATA_DIR = os.path.join(ROOT, "data")

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


@dataclass
class Run:
    run_id: str
    labels: np.ndarray
    summary: pd.DataFrame
    stats: dict
    params: dict
    rank: dict = field(default_factory=dict)   # cluster label -> size rank


DATASETS: dict[str, Dataset] = {}
RUNS: dict[str, Run] = {}
HISTORY: list[str] = []


def available_files():
    if not os.path.isdir(FEAT_DIR):
        return []
    out = []
    for f in sorted(os.listdir(FEAT_DIR)):
        if f.endswith("_features.parquet") and not f.startswith("all_"):
            out.append(f.replace("_features.parquet", ""))
    if os.path.exists(os.path.join(FEAT_DIR, "all_features.parquet")):
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

    path = os.path.join(FEAT_DIR, f"{name}_features.parquet")
    df = pd.read_parquet(path)
    df = df[df.feature_ok].reset_index(drop=True)
    cols = feature_columns(df)
    X = np.array(df[cols].to_numpy(dtype=np.float64), copy=True)
    good = np.isfinite(X).all(axis=1)
    df, X = df[good].reset_index(drop=True), X[good]

    if sample and len(df) > sample:
        idx = np.sort(np.random.default_rng(seed).choice(len(df), sample,
                                                         replace=False))
        df, X = df.iloc[idx].reset_index(drop=True), X[idx]

    keep = [c for c in PROV if c in df.columns]
    ds = Dataset(key=key, name=name, df=df[keep].copy(), raw=X, columns=cols)
    DATASETS[key] = ds
    return ds


def scale(X, how):
    """
    Equalise how much each feature contributes to the Euclidean distance.

    This is the control that matters most. See feature_matrix() in
    track_b_cluster.py for the measurements behind that claim.
    """
    X = np.array(X, copy=True)
    if how == "robust":
        med = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = np.where((q75 - q25) > 1e-12, q75 - q25, 1.0)
        return np.clip((X - med) / iqr, -5, 5)
    if how == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, len(X)),
                                 subsample=200_000, random_state=0)
        return qt.fit_transform(X)
    return X                                  # "none" = GLOBULAR's literal spec


def embed(ds, method):
    if method in ds.embedding:
        return ds.embedding[method]
    Z = scale(ds.raw, "robust")
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
    ds.embedding[method] = ((emb - lo) / span).astype(np.float32)
    return ds.embedding[method]


def run_hdbscan(X, mcs, ms, eps):
    """Retry with epsilon=0 on the sklearn epsilon-search bug (see AGENTS.md)."""
    kw = dict(min_cluster_size=mcs, min_samples=ms,
              cluster_selection_method="eom", n_jobs=-1)
    try:
        return HDBSCAN(cluster_selection_epsilon=eps, **kw).fit_predict(X)
    except (TypeError, ValueError):
        try:
            return HDBSCAN(cluster_selection_epsilon=0.0, **kw).fit_predict(X)
        except (TypeError, ValueError):
            return np.full(len(X), -1, dtype=np.int32)


def cluster(ds, cols, scaling, mode, mcs, ms, eps, epochs, batch, seed):
    use = [ds.columns.index(c) for c in cols]
    X = scale(ds.raw[:, use], scaling)
    n = len(X)
    labels = np.full(n, -1, dtype=np.int32)

    if mode == "single":
        labels = run_hdbscan(X, mcs, ms, eps).astype(np.int32)
    else:
        alive = np.arange(n)
        rng = np.random.default_rng(seed)
        for ep in range(1, epochs + 1):
            rng.shuffle(alive)
            survivors = []
            for s in range(0, len(alive), batch):
                b = alive[s:s + batch]
                if len(b) < mcs * 2:
                    survivors.append(b)
                    continue
                lab = run_hdbscan(X[b], mcs, ms, eps)
                hit = lab >= 0
                labels[b[hit]] = lab[hit] + ep * 100_000
                survivors.append(b[~hit])
            alive = np.concatenate(survivors) if survivors else np.array([], int)
            if len(alive) < mcs * 2:
                break
        labels[alive] = -1
    return labels, X


def summarise(df, labels):
    rows = []
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        d = df[m]
        rows.append({
            "cluster": int(lab), "n": int(m.sum()),
            "rfi_pct": float((d.weak_label == 1).mean() * 100),
            "freq": float(d.frequency.median()),
            "span": float(d.frequency.max() - d.frequency.min()),
            "snr": float(d.snr.median()),
            "beams": float(d.n_beams.median()),
            "obs": int(d.obsid.nunique()),
        })
    cols = ["cluster", "n", "rfi_pct", "freq", "span", "snr", "beams", "obs"]
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
            f'Run <code>python track_b_features.py</code> first.</p>')

    Z = scale(ds.raw, "robust")
    q75, q25 = np.percentile(Z, [75, 25], axis=0)
    iqr_scaled = q75 - q25
    q75r, q25r = np.percentile(ds.raw, [75, 25], axis=0)
    iqr_raw = q75r - q25r

    sat = {}
    if "f08_turning_bw_saturated" in ds.df.columns:
        sat["f08_turning_bw_hz_n"] = float(ds.df.f08_turning_bw_saturated.mean() * 100)

    feats = []
    for i, c in enumerate(ds.columns):
        feats.append({
            "col": c,
            "label": c[:-2],
            "iqr_raw": float(iqr_raw[i]),
            "iqr_rel": float(iqr_raw[i] / max(iqr_raw.max(), 1e-12)),
            "saturated": sat.get(c),
            "on": not c.startswith("f08_"),
        })
    return templates.TemplateResponse(request, "_controls.html", {
        "ds": ds, "features": feats,
        "n": len(ds.df), "load_ms": int((time.time() - t0) * 1000),
    })


@app.get("/embedding.bin")
def embedding_bin(key: str, method: str = "pca"):
    ds = DATASETS.get(key)
    if ds is None:
        return Response(status_code=404)
    return Response(embed(ds, method).tobytes(),
                    media_type="application/octet-stream")


@app.post("/cluster", response_class=HTMLResponse)
async def do_cluster(request: Request):
    form = await request.form()
    key = form["key"]
    ds = DATASETS.get(key)
    if ds is None:
        return HTMLResponse('<p class="error">Dataset expired. Load it again.</p>')

    cols = [c for c in form.getlist("feat") if c in ds.columns]
    if len(cols) < 2:
        return HTMLResponse('<p class="error">Keep at least two features on. '
                            'Distance needs something to compare.</p>')

    p = dict(scaling=form.get("scaling", "robust"),
             mode=form.get("mode", "epochs"),
             mcs=int(form.get("mcs", 4)), ms=int(form.get("ms", 2)),
             eps=float(form.get("eps", 0.18)),
             epochs=int(form.get("epochs", 8)),
             batch=int(form.get("batch", 3000)),
             seed=int(form.get("seed", 0)))

    sig = hashlib.sha1(json.dumps([key, sorted(cols), p],
                                  sort_keys=True).encode()).hexdigest()[:12]
    cached = sig in RUNS
    if not cached:
        t0 = time.time()
        labels, _ = cluster(ds, cols, p["scaling"], p["mode"], p["mcs"],
                            p["ms"], p["eps"], p["epochs"], p["batch"], p["seed"])
        elapsed = time.time() - t0
        summary = summarise(ds.df, labels)
        noise = labels == -1
        conf = noise & (ds.df.n_beams.to_numpy() <= 4)
        rank = {int(c): i for i, c in enumerate(summary["cluster"])}
        RUNS[sig] = Run(sig, labels, summary, {
            "n": len(labels),
            "n_clusters": int(len(summary)),
            "clustered_pct": float((labels >= 0).mean() * 100),
            "noise": int(noise.sum()),
            "confined": int(conf.sum()),
            "seconds": elapsed,
            "n_features": len(cols),
        }, p, rank)
        if sig in HISTORY:
            HISTORY.remove(sig)
        HISTORY.insert(0, sig)
        del HISTORY[12:]

    run = RUNS[sig]
    resp = templates.TemplateResponse(request, "_results.html", {
        "run": run, "cached": cached, "rank": run.rank,
        "summary": run.summary.head(40).to_dict("records"),
        "history": [RUNS[h] for h in HISTORY if h in RUNS],
        "key": key,
    })
    # The client needs the size order to colour by rank; 12 ints is cheap.
    top = [int(c) for c in run.summary["cluster"].head(TOP_N)]
    resp.headers["HX-Trigger"] = json.dumps(
        {"clusterDone": {"run": run.run_id, "top": top}})
    return resp


@app.get("/labels.bin")
def labels_bin(run: str):
    r = RUNS.get(run)
    if r is None:
        return Response(status_code=404)
    return Response(r.labels.astype(np.int32).tobytes(),
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

    path = os.path.join(DATA_DIR, f"{file}.h5")
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


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    print(f"\n  Cluster Bench -> http://{a.host}:{a.port}\n")
    uvicorn.run("app:app" if a.reload else app, host=a.host, port=a.port,
                reload=a.reload, app_dir=HERE)
