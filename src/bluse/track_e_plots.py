"""
Figures for Track E.

Drawn from `scores/report.json` and the catalogues beside it, so a figure can
never disagree with the numbers that were measured -- there is one source and
the plots read it.

Three form choices worth stating, because the obvious alternative is worse in
each case:

  * The ablation is a DOT PLOT, not ROC curves. Four curves at 0.94-0.99 all
    hug the top-left corner and separate nowhere a projector can show. It is
    also not a bar chart: every value sits between 0.93 and 0.99, so a
    zero-anchored bar wastes the plot and a truncated one misleads. A dot plot
    carries no area, so a non-zero axis is honest.
  * The SNR figure is TWO STACKED PANELS, not one plot with two y-scales. AUC
    and base rate have different units and the alignment of two scales is
    arbitrary -- it would invent a relationship. Shared x, two panels.
  * The monotonicity figure uses EMPHASIS, not two categorical hues. The story
    is the five bins the model never trained on; those carry the accent and the
    trained bins recede to grey.
"""

import json
import os

import numpy as np

# Validated categorical slots (light surface #fcfcfb): blue / orange / aqua.
# Aqua sits below 3:1 against the surface, so anything drawn in it carries a
# direct label -- the relief rule.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Ordinal ramp for the funnel: one hue, monotone, light end clear of the surface.
RAMP = ["#86b6ef", "#3987e5", "#184f95"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GREY, GRID, SURFACE = "#c9c8c3", "#e8e7e3", "#fcfcfb"


def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 11,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.titlepad": 10, "figure.dpi": 160,
    })
    return plt


def _clean(ax, grid="y"):
    """Hairline recessive chrome: no box, solid grid one shade off the surface."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# --- the money plot --------------------------------------------------------

# Bin -> representative n_beams, for a log x-axis. The bins are ranges; the
# midpoint is the honest single position for one.
_BIN_X = {"(0, 1]": 1, "(1, 2]": 2, "(2, 3]": 3, "(3, 5]": 4, "(5, 9]": 7,
          "(9, 17]": 13, "(17, 32]": 24, "(32, 49]": 40, "(49, 65]": 57}


def fig_monotonicity(rep, path):
    """
    The single figure worth showing. Trained only on the ends; correct in the
    middle.
    """
    plt = _rc()
    bins = rep["validation"]["n_beams_monotonicity"]["bins"]
    x = np.array([_BIN_X[b["bin"]] for b in bins], float)
    y = np.array([b["mean_score"] for b in bins])
    # dtype=bool is load-bearing: JSON may hand back 1/0, and an int
    # array turns x[tr] into fancy indexing and ~tr into bitwise NOT.
    tr = np.array([b["in_training"] for b in bins], dtype=bool)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _clean(ax)
    for lo, hi in ((0.8, 2.35), (31.0, 72)):
        ax.axvspan(lo, hi, color=GRID, alpha=0.65, zorder=0, linewidth=0)
    ax.plot(x, y, "-", color=BLUE, linewidth=2, zorder=2, alpha=0.55)
    ax.plot(x[tr], y[tr], "o", ms=9, color=GREY, mec=SURFACE, mew=2, zorder=3)
    ax.plot(x[~tr], y[~tr], "o", ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=4)

    for xi, yi in zip(x[~tr], y[~tr]):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=10,
                    color=INK, fontweight="bold")
    # Three region labels in one row, so the shading reads as one statement
    # rather than as decoration. The middle one is the point of the figure and
    # is the only one in the accent hue.
    n_untrained = sum(b["n"] for b in bins if not b["in_training"])
    ax.text(1.45, 1.05, "trained\n(≤2 beams)", ha="center", va="bottom",
            fontsize=9.5, color=MUTED, linespacing=1.4)
    ax.text(47, 1.05, "trained\n(≥32 beams)", ha="center", va="bottom",
            fontsize=9.5, color=MUTED, linespacing=1.4)
    ax.text(8.7, 1.05, f"never trained on\n{n_untrained:,} hits", ha="center",
            va="bottom", fontsize=10, color=BLUE, fontweight="bold",
            linespacing=1.4)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 9, 17, 32, 64])
    ax.set_xticklabels(["1", "2", "3", "5", "9", "17", "32", "64"])
    ax.set_xlim(0.8, 72)
    ax.set_ylim(0, 1.19)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("beams the signal was detected in")
    ax.set_ylabel("mean RFI score")
    ax.set_title("The score orders beam counts it was never trained on")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- what the information is worth -----------------------------------------

_ABL = {"flags": "Track A flags\n(6 hand-built)",
        "meta": "metadata\n(4 features)",
        "all": "metadata + stamp\n(16 features)",
        "stamp": "stamp morphology\n(12 features)"}
DEFAULT_SET = "stamp"


def fig_ablation(rep, path):
    """
    Ordered by score, with the SHIPPED DEFAULT emphasised -- which is not the
    highest. all-16 scores 0.9911 against stamp's 0.9899, and stamp is the
    default anyway because it cannot relearn the RFI frequency mask. Sorting by
    value keeps the chart honest about the ordering; the annotation keeps it
    honest about which one we run.
    """
    plt = _rc()
    a = rep["validation"]["ablation"]
    order = sorted(_ABL, key=lambda k: a[k]["roc_auc"])
    labels = [_ABL[k] for k in order]
    vals = [a[k]["roc_auc"] for k in order]
    y = np.arange(len(vals))
    hero = order.index(DEFAULT_SET)

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    _clean(ax, grid="x")
    for i, v in enumerate(vals):
        c = BLUE if i == hero else GREY
        ax.plot([0.90, v], [i, i], "-", color=c, linewidth=2,
                alpha=0.35 if i != hero else 0.5, zorder=2)
        ax.plot(v, i, "o", ms=11 if i == hero else 9, color=c,
                mec=SURFACE, mew=2, zorder=3)
        ax.annotate(f"{v:.4f}", (v, i), textcoords="offset points",
                    xytext=(13, 0), va="center", fontsize=10.5, color=INK,
                    fontweight="bold" if i == hero else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xlim(0.90, 1.008)
    ax.set_xticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    # Left-anchored in the empty band below the highlighted row. Centring it on
    # the dot puts it straight through the next row's value label.
    ax.text(0.9015, hero - 0.42,
            "the shipped default — it cannot relearn the RFI frequency mask",
            ha="left", va="top", fontsize=9, color=MUTED)
    ax.set_xlabel("ROC-AUC, group 5-fold on obsid, mean of 3 seeds   →")
    ax.set_title("Stamp morphology against the classical filter's own flags")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- is it a brightness meter? ---------------------------------------------

def fig_snr(rep, path):
    plt = _rc()
    d = rep["validation"]["snr_stratified"]["deciles"]
    w = rep["validation"]["snr_stratified"]["weighted"]
    x = np.arange(len(d))
    auc = [r["roc_auc"] for r in d]
    base = [r["base_rate"] for r in d]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True,
                                 height_ratios=[1.5, 1])
    _clean(a1); _clean(a2)
    a1.axhline(w, color=GREY, linewidth=1.4, zorder=1)
    a1.annotate(f"n-weighted mean {w:.3f}", (0, w),
                textcoords="offset points", xytext=(0, -17), ha="left",
                fontsize=9.5, color=MUTED)
    a1.plot(x, auc, "-o", color=BLUE, linewidth=2, ms=8, mec=SURFACE, mew=2)
    a1.set_ylim(0.5, 1.03)
    a1.set_ylabel("ROC-AUC within decile")
    a1.set_title("Within an SNR decile the label cannot be brightness")

    a2.plot(x, base, "-o", color=ORANGE, linewidth=2, ms=8, mec=SURFACE, mew=2)
    a2.set_ylim(0.55, 1.03)
    a2.set_ylabel("P(RFI) in decile")
    a2.set_xlabel("SNR decile, faintest → brightest")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{r['decile'] + 1}" for r in d])
    a2.annotate("almost nothing left to\nseparate up here", (8.5, 0.995),
                textcoords="offset points", xytext=(0, -34), ha="center",
                fontsize=9.5, color=MUTED, linespacing=1.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- which morphology says RFI ---------------------------------------------

def fig_importance(rep, path):
    plt = _rc()
    imp = rep["validation"]["importance"][::-1]
    y = np.arange(len(imp))
    v = [d["mean"] for d in imp]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _clean(ax, grid="x")
    ax.barh(y, v, height=0.62, color=BLUE, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([d["column"].replace("_n", "") for d in imp], fontsize=9.5)
    for i, val in enumerate(v):
        if val > max(v) * 0.05:
            ax.annotate(f"{val:.3f}", (val, i), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, max(v) * 1.14)
    ax.set_xlabel("drop in ROC-AUC when the column is permuted")
    ax.set_title("Drift-trajectory coherence carries the result")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- the deliverable --------------------------------------------------------

def fig_funnel(rep, path):
    plt = _rc()
    v = rep["verdicts"]
    order = [("pruned", "look exactly like\nmulti-beam RFI"),
             ("uncertain", "no clear verdict\neither way"),
             ("shortlist", "do not resemble\nthis survey's RFI")]
    vals = [v.get(k, 0) for k, _ in order]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    left = 0.0
    gap = total * 0.002                          # 2px-equivalent surface gap
    for (k, sub), val, c in zip(order, vals, RAMP):
        ax.barh([0], [val - gap], left=left, height=0.42, color=c, zorder=2)
        mid = left + val / 2
        ax.annotate(f"{val:,}", (mid, 0.30), ha="center", va="bottom",
                    fontsize=13, color=INK, fontweight="bold")
        ax.annotate(f"{k}   {val / total:.0%}", (mid, 0.235), ha="center",
                    va="bottom", fontsize=10, color=INK2)
        ax.annotate(sub, (mid, -0.30), ha="center", va="top", fontsize=9,
                    color=MUTED, linespacing=1.35)
        left += val
    # Pad the axis: the labels are centred over their segments and the last
    # segment is narrow, so its text is wider than the bar it names and would
    # otherwise be clipped by the figure edge.
    ax.set_xlim(-total * 0.03, total * 1.03)
    ax.set_ylim(-0.75, 0.75)
    ax.axis("off")
    ax.set_title(f"{total:,} Track A survivors, sorted by what they look like",
                 color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


FIGURES = {"monotonicity": fig_monotonicity, "ablation": fig_ablation,
           "snr": fig_snr, "importance": fig_importance, "funnel": fig_funnel}


def all_figures(rep, outdir, prefix="track_e"):
    """Draw every figure. Returns the paths written, in reading order."""
    out = []
    for name, fn in FIGURES.items():
        p = os.path.join(outdir, f"{prefix}_{name}.png")
        out.append(fn(rep, p))
    return out


def main():
    """Redraw from an existing report, without refitting."""
    import argparse

    from . import paths

    p = argparse.ArgumentParser(
        prog="bluse-score-plots",
        description="Redraw the Track E figures from scores/report.json, "
                    "without refitting. Use it to iterate on a figure; "
                    "`bluse-score` draws them as part of a run.")
    paths.add_workspace_arg(p)
    p.add_argument("--report", default=None,
                   help="path to report.json (default: scores/report.json in "
                        "the workspace)")
    args = p.parse_args()
    paths.set_workspace(args.workspace)
    rp = args.report or os.path.join(paths.scores_dir(), "report.json")
    with open(rp) as fh:
        rep = json.load(fh)
    if "validation" not in rep:
        raise SystemExit(f"{rp} has no validation block -- it was written by "
                         f"`bluse-score --no-report`, which skips the "
                         f"measurements every figure draws from.")
    for path in all_figures(rep, paths.plots_dir()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
