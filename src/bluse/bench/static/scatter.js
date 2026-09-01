/* Cluster Bench scatter.
 *
 * The embedding is fetched once per dataset; only the label array changes when
 * hyperparameters change, so a re-cluster costs one Int32Array over the wire
 * and a recolour here. Colours crossfade over ~260 ms so you can see which
 * points moved between clusters rather than just seeing a new picture. */

/* A run routinely yields 200+ clusters. Cycling a palette over all of them
 * makes a 15,000-point RFI family and a 4-point singleton look equally
 * important. Colour is therefore assigned by RANK: the 12 largest clusters get
 * distinct hues, the long tail is drawn in one muted grey so it reads as
 * texture rather than noise. Kept in step with COLOURS in app.py so a table
 * swatch matches its points. */
const COLOURS = [
  '#e8734a', '#4aa3e8', '#7fc96b', '#c77dd6', '#e8c34a', '#5fd0c0',
  '#e06b8b', '#8f9ce8', '#b8d24a', '#4ac2e8', '#e89a4a', '#9ad67f'
];
const NOISE = '#39424f';
const MINOR = '#5c6779';
let rank = null;   // Map(label -> size rank), set by loadLabels
let focused = null;   // cluster label isolated by clicking a table row
let embSig = '';      // signature of the embedding currently on screen

let emb = null, labels = null, prevRGB = null, curRGB = null;
let values = null;   // Float32Array in [0,1], or null to colour by cluster
let view = { x: 0, y: 0, k: 1 };
let canvas, ctx, dpr = window.devicePixelRatio || 1;
let anim = null, animT = 1, grid = null, hovered = -1, currentRun = '';
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function colourFor(l) {
  if (l < 0) return NOISE;
  if (!rank) return COLOURS[Math.abs(l) % COLOURS.length];
  const i = rank.get(l);
  return i === undefined ? MINOR : COLOURS[i];
}

function hexToRGB(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}

function resize() {
  const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  draw();
}

function toScreen(i) {
  const w = canvas.width, h = canvas.height, pad = 34 * dpr;
  const x = emb[i * 2], y = emb[i * 2 + 1];
  return [pad + (x * view.k + view.x) * (w - 2 * pad),
          h - pad - (y * view.k + view.y) * (h - 2 * pad)];
}

/* Exact inverse of toScreen, so nearest() can turn a pixel box into a range of
 * grid cells. buildGrid() indexes by raw embedding coords in [0,1]. */
function toData(sx, sy) {
  const w = canvas.width, h = canvas.height, pad = 34 * dpr;
  return [(((sx - pad) / (w - 2 * pad)) - view.x) / view.k,
          (((h - pad - sy) / (h - 2 * pad)) - view.y) / view.k];
}

function draw() {
  if (!ctx) return;
  ctx.fillStyle = '#0f1319';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!emb) return;

  const n = emb.length / 2;
  const r = Math.max(1, 1.6 * dpr * Math.min(2.2, Math.sqrt(view.k)));

  /* Three passes, back to front: noise, then the muted long tail, then the
   * ranked clusters on top -- otherwise a 15,000-point family is buried under
   * whatever happens to be drawn last. */
  const tier = i => {
    if (focused !== null) return labels && labels[i] === focused ? 2 : 0;
    if (!labels || labels[i] < 0) return 0;
    return (rank && rank.get(labels[i]) === undefined) ? 1 : 2;
  };
  for (let pass = 0; pass < 3; pass++) {
    for (let i = 0; i < n; i++) {
      const t = tier(i);
      if (t !== pass) continue;
      const isNoise = t === 0;
      const [sx, sy] = toScreen(i);
      if (sx < -8 || sy < -8 || sx > canvas.width + 8 || sy > canvas.height + 8) continue;
      if (curRGB) {
        const j = i * 3;
        if (animT < 1 && prevRGB) {
          const t = animT;
          ctx.fillStyle = `rgb(${Math.round(prevRGB[j] + (curRGB[j] - prevRGB[j]) * t)},${
            Math.round(prevRGB[j + 1] + (curRGB[j + 1] - prevRGB[j + 1]) * t)},${
            Math.round(prevRGB[j + 2] + (curRGB[j + 2] - prevRGB[j + 2]) * t)})`;
        } else {
          ctx.fillStyle = `rgb(${curRGB[j]},${curRGB[j + 1]},${curRGB[j + 2]})`;
        }
      } else {
        ctx.fillStyle = NOISE;
      }
      let a = t === 0 ? 0.5 : t === 1 ? 0.45 : 0.85;
      if (focused !== null) a = labels[i] === focused ? 1 : 0.07;
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;

  if (hovered >= 0) {
    const [sx, sy] = toScreen(hovered);
    ctx.strokeStyle = '#f0a03c'; ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath(); ctx.arc(sx, sy, 7 * dpr, 0, Math.PI * 2); ctx.stroke();
  }
}

/* Diverging ramp, blue -> grey -> orange, for colour-by-value. */
function rampRGB(t) {
  const c = Math.max(0, Math.min(1, t));
  if (c < 0.5) {
    const u = c * 2;
    return [74 + (140 - 74) * u, 163 + (150 - 163) * u, 232 + (150 - 232) * u];
  }
  const u = (c - 0.5) * 2;
  return [140 + (232 - 140) * u, 150 + (115 - 150) * u, 150 + (74 - 150) * u];
}

function buildRGB() {
  const n = emb.length / 2;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    /* Colour by a raw feature value instead of by cluster. This is what makes
     * the rail's distance shares visible rather than tabular: colouring by
     * f02_abs_drift_n renders the zero-drift slab at once, and if colouring by
     * f01_frequency_n reproduces the cluster structure, that is a finding. */
    if (values && values.length === n) {
      const [r, g, b] = rampRGB(values[i]);
      out[i * 3] = r; out[i * 3 + 1] = g; out[i * 3 + 2] = b;
      continue;
    }
    const [r, g, b] = hexToRGB(colourFor(labels ? labels[i] : -1));
    out[i * 3] = r; out[i * 3 + 1] = g; out[i * 3 + 2] = b;
  }
  return out;
}

async function loadValues() {
  const sel = document.getElementById('colour-by');
  const col = sel ? sel.value : '';
  if (!col) { values = null; curRGB = buildRGB(); draw(); return; }
  setBusy(true);
  try {
    const res = await fetch('/values.bin?key=' + encodeURIComponent(window.DSKEY)
                            + '&col=' + encodeURIComponent(col));
    values = res.ok ? new Float32Array(await res.arrayBuffer()) : null;
    curRGB = buildRGB();
    draw();
  } finally { setBusy(false); }
}

function animateRecolour() {
  if (reduced) { animT = 1; draw(); return; }
  const t0 = performance.now();
  cancelAnimationFrame(anim);
  const step = () => {
    animT = Math.min(1, (performance.now() - t0) / 260);
    draw();
    if (animT < 1) anim = requestAnimationFrame(step);
  };
  animT = 0; step();
}

function buildGrid() {
  const n = emb.length / 2, cells = 90;
  grid = Array.from({ length: cells * cells }, () => []);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(cells - 1, Math.max(0, Math.floor(emb[i * 2] * cells)));
    const cy = Math.min(cells - 1, Math.max(0, Math.floor(emb[i * 2 + 1] * cells)));
    grid[cy * cells + cx].push(i);
  }
}

/* D-3: this was a full O(n) scan with a toScreen() per point on EVERY
 * mousemove -- 35,000 coordinate transforms per mouse move -- while
 * buildGrid() maintained a 90x90 index that nothing ever read. Now it reads
 * it: convert the cursor's hit box to data space and walk only those cells. */
function nearest(mx, my) {
  if (!emb) return -1;
  const r = 14 * dpr;
  let best = -1, bd = r * r;
  if (!grid) {                       // index not built yet: fall back
    const n = emb.length / 2;
    for (let i = 0; i < n; i++) {
      const [sx, sy] = toScreen(i);
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  const cells = 90;
  const [x0, y0] = toData(mx - r, my - r);
  const [x1, y1] = toData(mx + r, my + r);
  const cx0 = Math.max(0, Math.floor(Math.min(x0, x1) * cells));
  const cx1 = Math.min(cells - 1, Math.floor(Math.max(x0, x1) * cells));
  const cy0 = Math.max(0, Math.floor(Math.min(y0, y1) * cells));
  const cy1 = Math.min(cells - 1, Math.floor(Math.max(y0, y1) * cells));
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

/* The embedding now projects exactly the matrix HDBSCAN sees, so it depends on
 * the feature set and the scaling as well as the method. The server stamps each
 * one with a signature; when a run comes back under a different signature the
 * geometry on screen is stale and has to be refetched before the labels. */
function embedQuery() {
  const q = new URLSearchParams();
  q.set('key', window.DSKEY);
  q.set('method', (document.getElementById('embed-method') || {}).value || 'pca');
  const f = document.getElementById('cluster-form');
  if (f) {
    const fd = new FormData(f);
    q.set('scaling', fd.get('scaling') || 'robust');
    fd.getAll('feat').forEach(v => q.append('feat', v));
  }
  return q;
}

/* A re-cluster is TWO phases and htmx only knows about the first.
   POST /cluster returns in milliseconds; the client then refetches the
   embedding, and after a feature-set change that is a fresh projection --
   measured at 11 s for UMAP on a 35k sample of `all`, and up to ~60 s cold.
   With only hx-indicator driving the spinner the run looked like it had
   finished instantly and then the plot changed by itself much later.

   Counted rather than boolean: loadEmbedding and loadLabels each claim busy and
   clusterDone wraps both, so the nesting must not clear the state early. */
let busyCount = 0;
function setBusy(on) {
  busyCount = Math.max(0, busyCount + (on ? 1 : -1));
  const b = busyCount > 0;
  for (const id of ['results', 'spin']) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('busy', b);
  }
}

async function loadEmbedding() {
  if (!window.DSKEY) return;
  setBusy(true);
  try {
    await loadEmbeddingInner();
  } finally {
    setBusy(false);
  }
}

async function loadEmbeddingInner() {
  /* Name the slow thing. A silent minute reads as a hang; "projecting (umap)"
     reads as work. */
  const method = (document.getElementById('embed-method') || {}).value || 'pca';
  document.getElementById('hud').innerHTML =
    `projecting (${method})…` + (method === 'umap' ? ' <span class="dim">up to a minute</span>' : '');
  const res = await fetch('/embedding.bin?' + embedQuery().toString());
  if (!res.ok) { document.getElementById('hud').innerHTML = ''; return; }
  embSig = res.headers.get('X-Emb-Sig') || '';
  emb = new Float32Array(await res.arrayBuffer());
  labels = null; curRGB = null; prevRGB = null;
  view = { x: 0, y: 0, k: 1 };
  buildGrid();
  /* Hide the placeholder as soon as there are points: leaving it up meant the
     "no clustering yet" copy sat on top of 35,000 rendered hits. */
  document.getElementById('empty').style.display = 'none';
  updateHud();
  resize();
}

async function loadLabels(run) {
  setBusy(true);
  try {
    currentRun = run;
    const res = await fetch(`/labels.bin?run=${run}`);
    if (!res.ok) return;
    labels = new Int32Array(await res.arrayBuffer());
    prevRGB = curRGB;
    curRGB = buildRGB();
    document.getElementById('empty').style.display = 'none';
    updateHud();
    animateRecolour();
  } finally {
    setBusy(false);
  }
}

function updateHud() {
  const n = emb ? emb.length / 2 : 0;
  let s = `<span><b>${n.toLocaleString()}</b> hits</span>`;
  if (labels) {
    let c = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] >= 0) c++;
    s += `<span><b>${c.toLocaleString()}</b> clustered</span>`;
    s += `<span><b>${(labels.length - c).toLocaleString()}</b> noise</span>`;
  }
  if (focused !== null) {
    let f = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] === focused) f++;
    s += `<span class="focus">cluster <b>${focused}</b> · ${f.toLocaleString()} pts`
       + ` · <span class="dim">esc to clear</span></span>`;
  }
  document.getElementById('hud').innerHTML = s;
}

function focusCluster(lab) {
  if (!labels || !emb) return;
  let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9, n = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== lab) continue;
    const x = emb[i * 2], y = emb[i * 2 + 1];
    minx = Math.min(minx, x); maxx = Math.max(maxx, x);
    miny = Math.min(miny, y); maxy = Math.max(maxy, y); n++;
  }
  if (!n) return;
  /* Zooming alone was not enough: you landed in a dense field with the target
     cluster indistinguishable from its neighbours. Isolate it too, and let a
     second click on the same row (or Escape) drop back to the full view. */
  if (focused === lab) { focused = null; view = { x: 0, y: 0, k: 1 }; updateHud(); draw(); return; }
  focused = lab;
  const sx = Math.max(maxx - minx, 0.02), sy = Math.max(maxy - miny, 0.02);
  view.k = Math.min(18, 0.7 / Math.max(sx, sy));
  view.x = 0.5 - ((minx + maxx) / 2) * view.k;
  view.y = 0.5 - ((miny + maxy) / 2) * view.k;
  updateHud();
  draw();
}

function setAllFeatures(on) {
  document.querySelectorAll('#cluster-form input[name=feat]')
          .forEach(c => { c.checked = on; });
}

/* ---- wiring ------------------------------------------------------------ */

window.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('plot');
  ctx = canvas.getContext('2d');
  window.addEventListener('resize', resize);
  /* The results panel swaps in only after the first cluster, which shrinks
     #plot-wrap. Without this the canvas kept its full-height size and the
     bottom ~40% of the scatter was drawn into a clipped, invisible region. */
  new ResizeObserver(resize).observe(canvas.parentElement);
  resize();

  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener('mousedown', e => { dragging = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    if (dragging) {
      view.x += (e.clientX - lx) / rect.width;
      view.y -= (e.clientY - ly) / rect.height;
      lx = e.clientX; ly = e.clientY;
      hideTip(); draw(); return;
    }
    const mx = (e.clientX - rect.left) * dpr, my = (e.clientY - rect.top) * dpr;
    const i = nearest(mx, my);
    if (i !== hovered) { hovered = i; draw(); }
    if (i >= 0) showTip(e.clientX - rect.left, e.clientY - rect.top, i);
    else hideTip();
  });
  canvas.addEventListener('mouseleave', () => { hovered = -1; hideTip(); draw(); });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = 1 - (e.clientY - rect.top) / rect.height;
    const f = e.deltaY < 0 ? 1.16 : 1 / 1.16;
    const nk = Math.max(0.6, Math.min(60, view.k * f));
    view.x = fx - (fx - view.x) * (nk / view.k);
    view.y = fy - (fy - view.y) * (nk / view.k);
    view.k = nk;
    draw();
  }, { passive: false });

  canvas.addEventListener('click', () => {
    if (hovered < 0 || !window.DSKEY) return;
    fetch(`/hit?key=${encodeURIComponent(window.DSKEY)}&i=${hovered}&run=${currentRun}`)
      .then(r => r.text())
      .then(html => { const el = document.getElementById('hit'); if (el) el.innerHTML = html; });
  });

  document.body.addEventListener('clusterDone', async e => {
    /* Claim busy for the whole handler. htmx drops .htmx-request the moment
       the POST resolves, which is before any of this has run; without an outer
       claim the indicator would flicker off between the two awaits below. */
    setBusy(true);
    try {
      rank = e.detail.top ? new Map(e.detail.top.map((l, i) => [l, i])) : null;
      focused = null;
      if (e.detail.emb && e.detail.emb !== embSig) await loadEmbedding();
      await loadLabels(e.detail.run);
    } finally {
      setBusy(false);
    }
  });
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && focused !== null) {
      focused = null; view = { x: 0, y: 0, k: 1 }; updateHud(); draw();
    }
  });
});

function showTip(x, y, i) {
  const t = document.getElementById('tip');
  const lab = labels ? (labels[i] < 0 ? 'noise' : 'cluster ' + labels[i]) : '—';
  t.textContent = `${lab}\nindex ${i}`;
  t.style.left = Math.min(x + 14, canvas.clientWidth - 130) + 'px';
  t.style.top = (y + 14) + 'px';
  t.style.opacity = 1;
}
function hideTip() { const t = document.getElementById('tip'); if (t) t.style.opacity = 0; }
