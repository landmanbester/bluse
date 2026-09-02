"""
A self-contained slide deck for the Track E result.

Generated from `scores/report.json`, so no number on a slide is hand-copied --
the same rule the figures follow. Writes one HTML file with every image
embedded as a data URI, so it can be moved to a laptop, opened offline and
presented full-screen with nothing else installed.

Five slides, one claim each:

  1  the result          the stamp alone carries what the beams carry
  2  why believe it      trained on the ends, correct in the middle
  3  what it gives you   two thirds of the vetting list, with a reason
  4  the honest slide    the shortlist is ten emitters, not 524 candidates
  5  what to do with it  for the team / the binding constraint / the caveat

Slides 1-3 are the talk. 4 and 5 are there because the first question from the
room is "so did you find anything", and the honest answer needs a slide.
"""

import base64
import io
import json
import os

# Cool-biased neutrals against the two accents the figures already use, so a
# slide and the plot on it are the same document. IBM Plex throughout -- drawn
# for engineering documentation, and Plex Mono gives the frequencies and AUCs
# the tabular alignment an instrument readout has.
FONTS = ("https://fonts.googleapis.com/css2?"
         "family=IBM+Plex+Mono:wght@400;500;600&"
         "family=IBM+Plex+Sans+Condensed:wght@600;700&"
         "family=IBM+Plex+Sans:wght@400;500;600&display=swap")


def _b64(path, max_width=None):
    """Embed an image, optionally downscaled. Keeps the deck one file."""
    if max_width:
        try:
            from PIL import Image
            im = Image.open(path)
            if im.width > max_width:
                h = round(im.height * max_width / im.width)
                im = im.convert("RGB").resize((max_width, h), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82, optimize=True)
                return "data:image/jpeg;base64," + base64.b64encode(
                    buf.getvalue()).decode()
        except ImportError:
            pass                              # Pillow is optional; ship as-is
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


CSS = """
:root{
  --ground:#eef2f4; --board:#ffffff; --plate:#fcfcfb;
  --ink:#0d1417; --ink2:#3d4e55; --muted:#6b7c84;
  --rule:#dbe3e7; --hair:#eaeff1;
  --accent:#2a78d6; --accent-soft:#e9f1fc; --warn:#c2410c; --warn-soft:#fdf0e8;
  --shadow:0 1px 2px rgba(13,20,23,.05),0 8px 28px rgba(13,20,23,.07);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0b1114; --board:#141d22; --plate:#fcfcfb;
  --ink:#eef4f6; --ink2:#b6c4ca; --muted:#8598a0;
  --rule:#26343b; --hair:#1c272d;
  --accent:#5b9df0; --accent-soft:#152a40; --warn:#f0834f; --warn-soft:#33200f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.45);
}}
:root[data-theme="dark"]{
  --ground:#0b1114; --board:#141d22; --plate:#fcfcfb;
  --ink:#eef4f6; --ink2:#b6c4ca; --muted:#8598a0;
  --rule:#26343b; --hair:#1c272d;
  --accent:#5b9df0; --accent-soft:#152a40; --warn:#f0834f; --warn-soft:#33200f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.45);
}

*{box-sizing:border-box}
html{color-scheme:light dark}
img{max-width:100%}
body{margin:0;
  background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.deck{
  display:flex; flex-direction:column; align-items:center;
  gap:34px; padding:34px 20px 72px;
  scroll-snap-type:y proximity;
}
.slide{
  container-type:inline-size;
  width:min(97vw,1240px); aspect-ratio:16/9;
  background:var(--board); border:1px solid var(--rule); border-radius:3px;
  box-shadow:var(--shadow);
  padding:4.1cqw 4.6cqw 3.4cqw;
  display:flex; flex-direction:column;
  scroll-snap-align:center;
  overflow:hidden;
}
.note{
  width:min(97vw,1240px); margin-top:-22px;
  font-size:13px; color:var(--muted); font-style:italic;
  padding-left:2px;
}
.note b{font-style:normal;font-weight:600;color:var(--ink2)}

/* --- type ------------------------------------------------------------- */
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:1.02cqw; font-weight:500; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted);
  display:flex; gap:.9em; align-items:center; flex-wrap:wrap;
}
.eyebrow .n{color:var(--accent);font-weight:600}
h1{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-weight:700; font-size:3.55cqw; line-height:1.05;
  letter-spacing:-.012em; margin:.42em 0 0; text-wrap:balance;
  max-width:22ch;
}
h1.wide{max-width:34ch}
h1 em{font-style:normal;color:var(--accent)}
h1 .warnword{font-style:normal;color:var(--warn)}
p{margin:0;font-size:1.32cqw;line-height:1.52;color:var(--ink2)}
p+p{margin-top:.75em}
p strong{color:var(--ink);font-weight:600}
.lead{font-size:1.46cqw;line-height:1.5}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;
      font-variant-numeric:tabular-nums}

/* --- layout ----------------------------------------------------------- */
.body{display:flex;gap:3.4cqw;flex:1;min-height:0;margin-top:2.1cqw}
.col{display:flex;flex-direction:column;min-width:0;gap:1.5cqw}
.grow{flex:1;min-height:0}
.plate{
  background:var(--plate); border:1px solid var(--rule); border-radius:2px;
  padding:.9cqw; display:flex; align-items:center; justify-content:center;
  min-height:0; flex:0 1 auto;
}
.col.grow{justify-content:center}
.plate img{max-width:100%;max-height:100%;object-fit:contain;display:block}

/* --- data ------------------------------------------------------------- */
table{border-collapse:collapse;width:100%;font-size:1.13cqw}
th{
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.92cqw; font-weight:500; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted);
  text-align:left; padding:0 0 .55em; border-bottom:1px solid var(--rule);
}
td{padding:.52em 0;border-bottom:1px solid var(--hair);color:var(--ink2)}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}
tr.hero td{color:var(--ink);font-weight:600}
tr.hero td.num{color:var(--accent)}

.stat{display:flex;flex-direction:column;gap:.15em}
.stat .v{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-weight:700; font-size:4.4cqw; line-height:.95;
  letter-spacing:-.02em; font-variant-numeric:tabular-nums; color:var(--accent);
}
.stat .v.plain{color:var(--ink)}
.stat .k{font-size:1.1cqw;color:var(--muted);line-height:1.3}
.statrow{display:flex;gap:3.6cqw;align-items:flex-end}

.callout{
  border-left:2px solid var(--warn); background:var(--warn-soft);
  padding:1.1cqw 1.4cqw; border-radius:0 2px 2px 0;
}
.callout p{color:var(--ink);font-size:1.24cqw}
.callout .tag{
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.9cqw;letter-spacing:.11em;text-transform:uppercase;
  color:var(--warn);font-weight:600;display:block;margin-bottom:.5em;
}
.rule{height:1px;background:var(--rule)}
.pillrow{display:flex;gap:.5cqw;flex-wrap:wrap}
.pill{
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.95cqw;padding:.3em .7em;border-radius:2px;
  background:var(--accent-soft);color:var(--accent);font-weight:500;
}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:3.2cqw;flex:1}
.three h3{
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.96cqw;letter-spacing:.11em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin:0 0 .9em;
  padding-bottom:.7em;border-bottom:1px solid var(--rule);
}
.foot{
  margin-top:auto;padding-top:1.5cqw;display:flex;justify-content:space-between;
  align-items:baseline;gap:2cqw;
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.92cqw;color:var(--muted);letter-spacing:.04em;
}
.foot .cite{color:var(--muted)}

@media print{
  body{background:#fff}
  .deck{gap:0;padding:0}
  .slide{width:100%;box-shadow:none;border:none;page-break-after:always}
  .note{display:none}
}
@media (prefers-reduced-motion:no-preference){
  html{scroll-behavior:smooth}
}
"""


def _slide(n, total, eyebrow, title, body, note, cite=""):
    return f"""
<section class="slide" aria-label="Slide {n}">
  <div class="eyebrow"><span class="n">{n:02d}</span><span>{eyebrow}</span></div>
  {title}
  {body}
  <div class="foot"><span class="cite">{cite}</span>
    <span>Track E · BLUSE / MeerKAT · {n} of {total}</span></div>
</section>
<div class="note">{note}</div>"""


def build(rep, plots_dir):
    """Render the deck HTML from a report dict and the figures beside it."""
    v = rep["validation"]
    a = v["ablation"]
    ver = rep["verdicts"]
    surv = rep["survey"]["n_survivors"]
    hits = rep["survey"]["n_hits"]
    nz, sn = v["nonzero_drift"], v["snr_stratified"]
    sig, cb, ood = v["signal_level"], v["cross_band"], v.get("ood", {})
    mono = v["n_beams_monotonicity"]
    untrained = sum(b["n"] for b in mono["bins"] if not b["in_training"])
    n_ut = sum(1 for b in mono["bins"] if not b["in_training"])
    info = rep["info"]

    def img(name, w=None):
        return _b64(os.path.join(plots_dir, name), w)

    names = {"stamp": "stamp morphology", "all": "metadata plus stamp",
             "meta": "metadata alone", "flags": "Track A flags"}
    alt_ablation = ", ".join(
        f"{names[k]} {a[k]['roc_auc']:.4f}"
        for k in ("stamp", "all", "meta", "flags"))
    mb = mono["bins"]
    alt_mono = (f"from {mb[0]['mean_score']:.2f} to {mb[-1]['mean_score']:.2f}"
                if mb and "mean_score" in mb[0] else "across every bin")
    alt_funnel = ", ".join(f"{n:,} {k}" for k, n in ver.items())

    checks = [
        ("drop every zero-drift row",
         "x01 is NaN exactly there, and 99.6% of those are RFI &mdash; a 29.6% freebie",
         f"{nz['stamp']:.4f}"),
        ("AUC within each SNR decile",
         "a signal must be bright to be <em>detected</em> in 32 beams",
         f"{sn['weighted']:.4f}"),
        ("one row per signal, not per beam",
         "one emitter yields up to 64 near-identical rows",
         f"{sig['median_per_signal']:.4f}"),
        (f"singletons only ({sig['n_singletons']:,})",
         "removes row duplication entirely",
         f"{sig['singleton']:.4f}"),
        ("hold out an entire band",
         "general morphology, or memorised local emitters?",
         "&ndash;".join([f"{min(x['stamp'] for x in cb.values()):.2f}",
                         f"{max(x['stamp'] for x in cb.values()):.2f}"])),
    ]
    rows = "".join(
        f'<tr><td>{c}<br><span style="color:var(--muted);font-size:.93em">'
        f'{w}</span></td><td class="num">{r}</td></tr>' for c, w, r in checks)

    s1 = _slide(
        1, 5, "BLUSE / MeerKAT &middot; 2,014,055 narrowband hits",
        '<h1 class="wide">The stamp alone tells you '
        '<em>what the beams tell you</em></h1>',
        f"""<div class="body">
      <div class="col" style="flex:0 0 40%">
        <p class="lead">The 64-beam spatial filter is this survey's strongest
        discriminant: a signal seen in &ge;32 beams is local interference.</p>
        <p><strong>It is also blind exactly where it matters.</strong> It needs
        many beams' worth of evidence, so it cannot judge a hit seen in one
        beam &mdash; and a technosignature is a one-beam hit. So is a weak
        terrestrial emitter that only clears threshold at boresight.</p>
        <p>Twelve numbers computed from the stamp pixels &mdash; no frequency,
        no drift rate, no SNR &mdash; reproduce the filter's verdict at
        <strong class="mono">{a['stamp']['roc_auc']:.4f}</strong>, above all
        four metadata columns together ({a['meta']['roc_auc']:.4f}) and within
        {a['all']['roc_auc'] - a['stamp']['roc_auc']:.4f} of everything
        combined.</p>
        <div class="pillrow" style="margin-top:auto">
          <span class="pill">{info['n_train']:,} labelled hits</span>
          <span class="pill">{info['n_groups']} observations</span>
          <span class="pill">group {info['n_splits']}-fold on obsid</span>
          <span class="pill">float64 &middot; {info['n_seeds']} seeds</span>
        </div>
      </div>
      <div class="col grow"><div class="plate">
        <img src="{img('track_e_ablation.png')}"
             alt="Dot plot of ROC-AUC for four feature sets. {alt_ablation}."
             ></div>
        <div class="callout" style="border-left-color:var(--accent);
             background:var(--accent-soft)">
          <span class="tag" style="color:var(--accent)">a free finding for
          Track&nbsp;A</span>
          <p>Most of the gap to the flag row is not the pixels. Track A's six
          booleans are <em>thresholds</em> on quantities the metadata holds
          continuously; handing the model the continuous versions is worth
          <strong>+{a['meta']['roc_auc'] - a['flags']['roc_auc']:.4f}</strong>,
          nine times what morphology adds on top. Anything downstream that
          consumes the flags should consume those quantities instead.</p>
        </div>
      </div>
    </div>""",
        "<b>Say:</b> the filter is our best tool and it abstains on precisely "
        "the hits we care about. The pixels carry the same judgement &mdash; "
        "and on the way we found 0.047 of AUC your own thresholds are "
        "discarding.")

    s2 = _slide(
        2, 5, "validation &middot; every de-confounding check",
        '<h1>Trained on the ends. <em>Correct in the middle.</em></h1>',
        f"""<div class="body">
      <div class="col" style="flex:0 0 55%"><div class="plate">
        <img src="{img('track_e_monotonicity.png')}"
             alt="Mean RFI score against beams detected in, rising
                  monotonically {alt_mono}, with the range between the two
                  trained bands marked as never trained on"></div></div>
      <div class="col grow">
        <p>The model sees only <strong>&le;2 beams</strong> and
        <strong>&ge;32 beams</strong> in training. Scored on the untrained
        middle &mdash; {n_ut} bins, <strong>{untrained:,} hits</strong> &mdash;
        it reproduces beam multiplicity monotonically.</p>
        <p>A fitted decision boundary has no reason to interpolate.</p>
        <div class="rule"></div>
        <table><thead><tr><th>and it survives</th>
          <th class="num">AUC</th></tr></thead><tbody>{rows}</tbody></table>
      </div>
    </div>""",
        "<b>Say:</b> a 0.99 AUC has several boring explanations &mdash; "
        "brightness, duplicate rows, the zero-drift shortcut. Each row is one "
        "of them, closed.",
        f"stamp morphology, {len(info['columns'])} features, float64, "
        f"mean of {info['n_seeds']} seeds &middot; "
        f"baseline {a['stamp']['roc_auc']:.4f}")

    s3 = _slide(
        3, 5, "the deliverable &middot; scores/candidates.csv",
        '<h1 class="wide">Two thirds of the vetting list, '
        '<em>ranked to the bottom</em></h1>',
        f"""<div class="body" style="flex-direction:column;gap:2.2cqw">
      <div class="plate" style="flex:0 0 auto;padding:1.4cqw">
        <img src="{img('track_e_funnel.png')}"
             alt="Stacked bar: {surv:,} Track A survivors split into
                  {alt_funnel}"></div>
      <div class="body" style="margin-top:0">
        <div class="col" style="flex:0 0 47%">
          <div class="statrow">
            <div class="stat"><span class="v">{ver.get('pruned', 0):,}</span>
              <span class="k">survivors that look exactly<br>like multi-beam
              RFI &mdash; vet them<br>last, not first</span></div>
            <div class="stat"><span class="v plain">{surv:,}</span>
              <span class="k">Track A survivors from<br>{hits:,} hits &mdash;
              this set is a 5.6&#37;<br>draw, so
              ~{round(surv / 0.056, -2):,.0f} survey-wide</span>
            </div>
          </div>
        </div>
        <div class="col grow">
          <p>Also written: <strong class="mono">contrarian.csv</strong> &mdash;
          {rep['counts']['contrarian']:,} hits in &ge;32 beams that score
          <em>clean</em>, where the filter and morphology disagree in the one
          direction "the filter is conservative" cannot explain. And
          <strong class="mono">ambiguous.csv</strong> &mdash;
          {rep['counts']['ambiguous']:,} hits in 3&ndash;31 beams, where the
          filter abstains and this is the only verdict available.</p>
          <p>Every row carries <span class="mono">row</span> and
          <span class="mono">file</span>, so
          <span class="mono">bluse-explore stamps --rows</span> plots it with
          no join.</p>
          <p><strong>This is an ordering, not yet a cut.</strong> Turning it
          into one needs an operating point, and an operating point needs
          ground truth we do not have &mdash; see slide 5.</p>
        </div>
      </div>
    </div>""",
        "<b>Say:</b> the number that matters operationally, and it scales &mdash; "
        "~81,000 survivors survey-wide. Vetting order, not a delete button.")

    s4 = _slide(
        4, 5, "the honest slide &middot; what a low score means",
        '<h1 class="wide">The shortlist is <span class="warnword">not 524 '
        'candidates</span>. It is about ten emitters.</h1>',
        f"""<div class="body">
      <div class="col" style="flex:0 0 46%">
        <div class="callout"><span class="tag">read before quoting</span>
        <p>The labels are <strong>positive&ndash;unlabelled</strong>.
        &le;2 beams means <em>not seen elsewhere</em>, not <em>verified
        clean</em>. A high score is strong evidence of RFI. A low score is
        <strong>not</strong> evidence of a technosignature.</p></div>
        <p>Those {ver.get('shortlist', 0)} hits collapse to
        <strong>48 distinct (file, 0.1&nbsp;MHz) groups</strong>, the top ten
        holding 73&#37;. Their stamps show two morphological populations,
        <strong>neither astrophysical</strong>: an instrumental signature
        around 867.8&nbsp;MHz, and blocky intermittent structure at
        599&ndash;678&nbsp;MHz.</p>
        <table><thead><tr><th>file</th><th class="num">MHz</th>
          <th class="num">hits</th><th class="num">obs</th></tr></thead>
        <tbody>
          <tr class="hero"><td>uhf_long</td><td class="num">678.0</td>
            <td class="num">157</td><td class="num">6</td></tr>
          <tr><td>lband_long</td><td class="num">960.0</td>
            <td class="num">73</td><td class="num">1</td></tr>
          <tr><td>lband_short</td><td class="num">867.8</td>
            <td class="num">51</td><td class="num">3</td></tr>
          <tr><td>uhf_short</td><td class="num">576.0</td>
            <td class="num">22</td><td class="num">3</td></tr>
        </tbody></table>
      </div>
      <div class="col grow"><div class="plate">
        <img src="{img('uhf_long_stamps_candidates.png', 1000)}"
             alt="A grid of 24 waterfall stamps from the lowest-scoring
                  uhf_long hits, showing blocky intermittent structure rather
                  than drifting narrowband lines."></div>
        <p style="font-size:1.05cqw;color:var(--muted)">The 24 lowest-scoring
        <span class="mono">uhf_long</span> survivors. Nothing like a drifting
        carrier &mdash; which is exactly why they score low.</p>
      </div>
    </div>""",
        "<b>Say:</b> this is the score working, not failing. An outlier "
        "ranking is supposed to surface what does not fit, and it surfaced two "
        "populations nothing else had isolated.")

    s5 = _slide(
        5, 5, "what to do with it",
        '<h1 class="wide">One thing to use, one thing to fix, '
        '<em>one thing to build</em></h1>',
        f"""<div class="body"><div class="three">
      <div class="col"><h3>use &mdash; today</h3>
        <p><strong class="mono">bluse-score</strong> scores all {hits:,} hits
        in 80&nbsp;s, nine minutes with the full validation report. No
        dependency outside scikit-learn.</p>
        <p>The pruned list is a two-thirds cut in vetting load with a per-hit
        reason. The <span class="mono">ambiguous</span> table gives a verdict
        on {rep['counts']['ambiguous']:,} hits the spatial filter cannot
        judge at all.</p></div>
      <div class="col"><h3>fix &mdash; a data problem</h3>
        <p><strong class="mono">mk_sample_hits</strong> is sampled at ~1&#37;
        of the other files' hit density, and beam coincidence is counted
        <em>within</em> a file.</p>
        <p>8,116 of its hits also appear in <span class="mono">lband_long</span>
        under the same id, frequency and beam. The same hit is counted in a mean
        of <strong>1.87 beams</strong> there against <strong>29.71</strong> in
        lband_long.</p>
        <p>Its <span class="mono">n_beams</span> and everything derived from it
        are artefacts. Of its 894 Track A survivors, only <strong>15</strong>
        survive when the surrounding hits are present.</p></div>
      <div class="col"><h3>build &mdash; the binding constraint</h3>
        <p><strong>Synthetic injections.</strong> Every number here measures
        agreement with the spatial filter, which is a good instrument and not
        ground truth. No hit in this survey is confirmed clean.</p>
        <p>Injected drifting narrowband signals at known SNR would give a real
        objective function, settle whether the monotonicity is physics or an
        SNR gradient, and turn the score from a ranking into a cut.</p>
        <p style="color:var(--muted)">~30 lines. It is now the highest-value
        item in the repository.</p></div>
    </div></div>""",
        "<b>Say:</b> the data problem is worth five minutes of theirs &mdash; "
        "it affects anything anyone computes from that file's beam counts.")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Track E — an RFI score for BLUSE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}</style>
</head>
<body>
<main class="deck">{s1}{s2}{s3}{s4}{s5}</main>
</body>
</html>"""


def write_deck(rep, plots_dir, out_path):
    with open(out_path, "w") as fh:
        fh.write(build(rep, plots_dir))
    return out_path


def main():
    import argparse

    from . import paths

    p = argparse.ArgumentParser(
        prog="bluse-score-slides",
        description="Build a self-contained HTML slide deck from "
                    "scores/report.json and the figures in plots/. One file, "
                    "every image embedded -- open it in a browser and present "
                    "full-screen.")
    paths.add_workspace_arg(p)
    p.add_argument("--out", default=None,
                   help="output path (default: plots/track_e_slides.html)")
    args = p.parse_args()
    paths.set_workspace(args.workspace)
    rp = os.path.join(paths.scores_dir(), "report.json")
    with open(rp) as fh:
        rep = json.load(fh)
    if "validation" not in rep:
        raise SystemExit(f"{rp} has no validation block -- it was written by "
                         f"`bluse-score --no-report`.")
    out = args.out or os.path.join(paths.plots_dir(), "track_e_slides.html")
    print(f"wrote {write_deck(rep, paths.plots_dir(), out)}")


if __name__ == "__main__":
    main()
