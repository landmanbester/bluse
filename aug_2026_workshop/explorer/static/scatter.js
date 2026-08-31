/* Cluster Bench scatter.
 *
 * The embedding is fetched once per dataset; only the label array changes when
 * hyperparameters change, so a re-cluster costs one Int32Array over the wire
 * and a recolour here. Colours crossfade over ~260 ms so you can see which
 * points moved between clusters rather than just seeing a new picture. */

/* 20 hues chosen to stay separable on the slate background. Kept in step with
 * PALETTE in app.py so a table swatch matches its points. */
const COLOURS = [
  '#e8734a', '#4aa3e8', '#7fc96b', '#c77dd6', '#e8c34a', '#5fd0c0',
  '#e06b8b', '#8f9ce8', '#b8d24a', '#4ac2e8', '#e89a4a', '#9ad67f',
  '#d67fae', '#7ad6c2', '#e8e04a', '#6b8fe0', '#d6a17f', '#4ae89a',
  '#e84a6b', '#a4e84a'
];
const NOISE = '#39424f';

let emb = null, labels = null, prevRGB = null, curRGB = null;
let view = { x: 0, y: 0, k: 1 };
let canvas, ctx, dpr = window.devicePixelRatio || 1;
let anim = null, animT = 1, grid = null, hovered = -1, currentRun = '';
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function colourFor(l) { return l < 0 ? NOISE : COLOURS[Math.abs(l) % COLOURS.length]; }

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

function draw() {
  if (!ctx) return;
  ctx.fillStyle = '#0f1319';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!emb) return;

  const n = emb.length / 2;
  const r = Math.max(1, 1.6 * dpr * Math.min(2.2, Math.sqrt(view.k)));

  /* Noise first so clustered points sit on top of it. */
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < n; i++) {
      const isNoise = !labels || labels[i] < 0;
      if ((pass === 0) !== isNoise) continue;
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
      ctx.globalAlpha = isNoise ? 0.5 : 0.85;
      ctx.fillRect(sx - r, sy - r, r * 2, r * 2);
    }
  }
  ctx.globalAlpha = 1;

  if (hovered >= 0) {
    const [sx, sy] = toScreen(hovered);
    ctx.strokeStyle = '#f0a03c'; ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath(); ctx.arc(sx, sy, 7 * dpr, 0, Math.PI * 2); ctx.stroke();
  }
}

function buildRGB() {
  const n = emb.length / 2;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = hexToRGB(colourFor(labels ? labels[i] : -1));
    out[i * 3] = r; out[i * 3 + 1] = g; out[i * 3 + 2] = b;
  }
  return out;
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

function nearest(mx, my) {
  if (!emb) return -1;
  let best = -1, bd = (14 * dpr) ** 2;
  const n = emb.length / 2;
  const stride = n > 120000 ? 2 : 1;
  for (let i = 0; i < n; i += stride) {
    const [sx, sy] = toScreen(i);
    const d = (sx - mx) ** 2 + (sy - my) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

async function loadEmbedding() {
  if (!window.DSKEY) return;
  const method = (document.getElementById('embed-method') || {}).value || 'pca';
  document.getElementById('hud').innerHTML = 'loading embedding…';
  const res = await fetch(`/embedding.bin?key=${encodeURIComponent(window.DSKEY)}&method=${method}`);
  if (!res.ok) { document.getElementById('hud').innerHTML = ''; return; }
  emb = new Float32Array(await res.arrayBuffer());
  labels = null; curRGB = null; prevRGB = null;
  view = { x: 0, y: 0, k: 1 };
  buildGrid();
  updateHud();
  resize();
}

async function loadLabels(run) {
  currentRun = run;
  const res = await fetch(`/labels.bin?run=${run}`);
  if (!res.ok) return;
  labels = new Int32Array(await res.arrayBuffer());
  prevRGB = curRGB;
  curRGB = buildRGB();
  document.getElementById('empty').style.display = 'none';
  updateHud();
  animateRecolour();
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
  const sx = Math.max(maxx - minx, 0.02), sy = Math.max(maxy - miny, 0.02);
  view.k = Math.min(18, 0.7 / Math.max(sx, sy));
  view.x = 0.5 - ((minx + maxx) / 2) * view.k;
  view.y = 0.5 - ((miny + maxy) / 2) * view.k;
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

  document.body.addEventListener('clusterDone', e => loadLabels(e.detail.run));
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
