#!/usr/bin/env python3
"""
explore.py -- visual exploration of the BLUSE workshop HDF5 stamp files.

The data are seticore "stamp" files flattened into a columnar HDF5 layout: one
row per narrowband hit, with N scalar metadata columns plus a `data` cube of
per-hit time-frequency cutouts (the waterfall around the detection).

    bluse-explore info
    bluse-explore meta   data/sband_short.h5
    bluse-explore stamps data/sband_short.h5 --sort snr --n 24
    bluse-explore obs    data/sband_short.h5
    bluse-explore coincidence data/sband_short.h5

File arguments resolve against the workspace data directory, so a bare
`sband_short.h5` works from anywhere. Every plotting command writes a PNG into
<workspace>/plots/ and prints the path. See `bluse.paths` for how the workspace
is located.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from . import paths
from .paths import resolve_files

# Columns that are one scalar per hit. `data` and `tstartts` are handled apart:
# `data` is the stamp cube, `tstartts` is a nested compound type.
SCALAR_COLS = [
    "id", "index", "beam", "coarseChannel", "startChannel", "numChannels",
    "numTimesteps", "frequency", "driftRate", "driftSteps", "snr", "power",
    "incoherentPower", "ra", "dec", "fch1", "foff", "tsamp", "tstart",
    "fileoffset", "telescopeId",
]
STRING_COLS = ["sourceName", "obsid", "filename"]

PAD_VALUE = -1.0  # stamps are right-aligned in a 120-channel buffer; the unused
                  # leading columns are filled with exactly -1.


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def decode(arr):
    """h5py object-dtype string column -> numpy array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def load_meta(h, cols=None):
    """Read the scalar metadata columns into a dict of arrays. Cheap: no stamps."""
    cols = cols or SCALAR_COLS
    return {c: h[c][:] for c in cols if c in h}


def clean_stamp(cube_row):
    """
    Strip the -1 padding from one stamp and return (image, n_valid_channels).

    `data` rows are (1, n_time, 120); only `numChannels` of those 120 columns
    hold real data. Rather than trusting a stored width, detect the pad columns
    directly -- that stays correct whichever end they sit at.
    """
    img = np.asarray(cube_row)
    while img.ndim > 2:
        img = img[0]
    valid = ~np.all(img == PAD_VALUE, axis=0)
    if valid.any():
        img = img[:, valid]
    return img, int(valid.sum())


def norm_stamp(img):
    """Per-stamp normalisation for display: divide by the median, guard zeros."""
    med = np.median(img)
    if not np.isfinite(med) or med <= 0:
        med = np.abs(img).max() or 1.0
    out = img / med
    return np.clip(out, 1e-3, None)


def ensure_plot_dir():
    return paths.subdir("plots", create=True)


def savefig(fig, name):
    path = os.path.join(ensure_plot_dir(), name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ----------------------------------------------------------------------------
# cmd: info
# ----------------------------------------------------------------------------

def cmd_info(args):
    files = resolve_files(args.files)
    print(f"{len(files)} file(s)\n")
    total = 0
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f) / 1e9
        try:
            with h5py.File(f, "r") as h:
                n = h["snr"].shape[0]
                total += n
                d = h["data"]
                snr = h["snr"][:]
                dr = h["driftRate"][:]
                fr = h["frequency"][:]
                ip = h["incoherentPower"][:]
                T = int(h["numTimesteps"][0])
                tsamp = float(h["tsamp"][0])
                foff_hz = float(h["foff"][0]) * 1e6
                obs = np.unique(decode(h["obsid"][:]))
                nsrc = len(np.unique(decode(h["sourceName"][:])))
                nbeam = len(np.unique(h["beam"][:]))

                print(f"{name}   ({size:.2f} GB)")
                print(f"   hits        {n:,}")
                print(f"   stamp cube  {d.shape}  dtype={d.dtype} "
                      f"chunks={d.chunks} compression={d.compression}")
                print(f"   time        {T} steps x {tsamp:.3f} s = {T * tsamp:.1f} s")
                print(f"   freq res    {foff_hz:.2f} Hz   band "
                      f"{fr.min():.1f}-{fr.max():.1f} MHz")
                print(f"   snr         {snr.min():.1f} .. {snr.max():.3g}  "
                      f"(median {np.median(snr):.1f})")
                print(f"   drift       +/-{np.abs(dr).max():.3f} Hz/s   "
                      f"exactly zero: {100 * np.mean(dr == 0):.1f}% of hits")
                print(f"   coverage    {len(obs)} observations, {nsrc} sources, "
                      f"{nbeam} beams")
                if np.all(ip == 0):
                    print(f"   NOTE        incoherentPower is identically zero "
                          f"-- the coherent/incoherent test is unavailable")
                print()
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}\n")
    print(f"total hits across files: {total:,}")


# ----------------------------------------------------------------------------
# cmd: meta
# ----------------------------------------------------------------------------

def cmd_meta(args):
    f = resolve_files([args.file])[0]
    name = os.path.basename(f).replace(".h5", "")
    with h5py.File(f, "r") as h:
        m = load_meta(h)
        n = len(m["snr"])

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{name}  --  {n:,} hits", fontsize=13)

    # SNR: log-log, because the range spans ~7 decades above a hard cut at 6.
    a = ax[0, 0]
    a.hist(np.log10(m["snr"]), bins=120, color="#3b6ea5")
    a.set_xlabel("log10(SNR)"); a.set_ylabel("hits"); a.set_yscale("log")
    a.set_title(f"SNR  (threshold = {m['snr'].min():.1f})")

    # Frequency occupancy -- where the RFI lives.
    a = ax[0, 1]
    a.hist(m["frequency"], bins=400, color="#a5533b")
    a.set_xlabel("frequency [MHz]"); a.set_ylabel("hits"); a.set_yscale("log")
    a.set_title("frequency occupancy")

    # Drift rate. The spike at exactly zero is the classic local-RFI signature.
    a = ax[0, 2]
    dr = m["driftRate"]
    a.hist(dr, bins=121, color="#4a7c59")
    a.set_xlabel("drift rate [Hz/s]"); a.set_ylabel("hits"); a.set_yscale("log")
    a.set_title(f"drift rate  ({100 * np.mean(dr == 0):.1f}% exactly 0)")

    # Hits per beam: a flat distribution means field-wide RFI, spikes mean
    # something beam-specific.
    a = ax[1, 0]
    beams, counts = np.unique(m["beam"], return_counts=True)
    a.bar(beams, counts, color="#7a5c9e", width=1.0)
    a.set_xlabel("beam"); a.set_ylabel("hits"); a.set_title("hits per beam")

    # Hits per coarse channel.
    a = ax[1, 1]
    cc, ccn = np.unique(m["coarseChannel"], return_counts=True)
    a.bar(cc, ccn, color="#9e7a5c", width=1.0)
    a.set_xlabel("coarse channel"); a.set_ylabel("hits")
    a.set_title("hits per coarse channel")

    # SNR vs drift: RFI tends to pile up in the zero-drift column.
    a = ax[1, 2]
    a.scatter(dr, m["snr"], s=1, alpha=0.15, color="#333333", rasterized=True)
    a.set_yscale("log"); a.set_xlabel("drift rate [Hz/s]"); a.set_ylabel("SNR")
    a.set_title("SNR vs drift rate")

    fig.tight_layout()
    savefig(fig, f"{name}_meta.png")


# ----------------------------------------------------------------------------
# cmd: stamps
# ----------------------------------------------------------------------------

def cmd_stamps(args):
    f = resolve_files([args.file])[0]
    name = os.path.basename(f).replace(".h5", "")
    n_show = args.n

    with h5py.File(f, "r") as h:
        snr = h["snr"][:]
        dr = h["driftRate"][:]
        n = len(snr)

        if getattr(args, "rows", None):
            # Explicit row selection, so a family or the candidate residue can
            # be inspected directly: the taxonomy writes `file` and `row`, and
            # this reads them straight back.
            sel = pd.read_csv(args.rows)
            if "row" not in sel.columns:
                sys.exit(f"{args.rows} has no `row` column; expected the "
                         f"output of bluse-cluster --match")
            if "file" in sel.columns:
                same = sel["file"].astype(str) == name
                if not same.any():
                    sys.exit(f"{args.rows} has no rows from {name} "
                             f"(files present: "
                             f"{', '.join(sorted(set(sel['file'].astype(str))))})")
                sel = sel[same]
            if "family" in sel.columns and args.family is not None:
                sel = sel[sel["family"] == args.family]
                if not len(sel):
                    sys.exit(f"no rows for family {args.family} in {args.rows}")
            order = sel["row"].to_numpy()[:n_show].astype(int)
            bad = order[(order < 0) | (order >= n)]
            if len(bad):
                sys.exit(f"row index {bad[0]} is outside {name} (0-{n - 1})")
            label = os.path.basename(args.rows)
            if args.family is not None:
                label += f", family {args.family}"
        elif args.sort == "snr":
            order = np.argsort(snr)[::-1][:n_show]
            label = "highest SNR"
        elif args.sort == "drift":
            order = np.argsort(np.abs(dr))[::-1][:n_show]
            label = "largest |drift rate|"
        elif args.sort == "nonzero-drift":
            cand = np.where(dr != 0)[0]
            order = cand[np.argsort(snr[cand])[::-1][:n_show]]
            label = "highest SNR with non-zero drift"
        else:
            rng = np.random.default_rng(args.seed)
            order = rng.choice(n, size=min(n_show, n), replace=False)
            label = "random"

        # h5py needs a sorted, unique index list for fancy selection.
        order = np.asarray(order)
        srt = np.sort(order)
        cubes = h["data"][srt]
        back = {int(v): i for i, v in enumerate(srt)}
        cubes = np.stack([cubes[back[int(i)]] for i in order])

        meta = {c: h[c][:][order] for c in
                ["frequency", "driftRate", "snr", "beam", "numChannels"]}
        src = decode(h["sourceName"][:])[order]
        tsamp = float(h["tsamp"][0])
        foff_hz = float(h["foff"][0]) * 1e6

    ncol = args.ncol
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"{name}  --  {label}  "
        f"(x: {foff_hz:.2f} Hz/channel,  y: {tsamp:.2f} s/step, time downward)",
        fontsize=12)

    for k, ax in enumerate(axes):
        if k >= len(order):
            ax.axis("off")
            continue
        img, nvalid = clean_stamp(cubes[k])
        ax.imshow(norm_stamp(img), aspect="auto", origin="upper",
                  cmap="viridis", norm=LogNorm(),
                  interpolation="nearest")
        ax.set_title(
            f"{meta['frequency'][k]:.6f} MHz\n"
            f"SNR {meta['snr'][k]:.3g}  dr {meta['driftRate'][k]:+.3f}  "
            f"b{meta['beam'][k]}",
            fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    # Name the file after the SELECTION, not after --sort, which is ignored
    # when rows are given explicitly -- otherwise every family lands on
    # <name>_stamps_random.png and silently overwrites the last one.
    if getattr(args, "rows", None):
        tag = os.path.splitext(os.path.basename(args.rows))[0]
        if args.family is not None:
            tag += f"_fam{args.family}"
    else:
        tag = args.sort
    savefig(fig, f"{name}_stamps_{tag}.png")

    print("\n  first few shown:")
    for k in range(min(6, len(order))):
        print(f"    idx={order[k]:>8d}  {meta['frequency'][k]:14.6f} MHz  "
              f"SNR={meta['snr'][k]:>11.4g}  drift={meta['driftRate'][k]:+.4f}  "
              f"beam={meta['beam'][k]:>2d}  nchan={meta['numChannels'][k]:>3d}  "
              f"src={src[k]}")


# ----------------------------------------------------------------------------
# cmd: obs -- one observation, beam vs frequency
# ----------------------------------------------------------------------------

def cmd_obs(args):
    f = resolve_files([args.file])[0]
    name = os.path.basename(f).replace(".h5", "")
    with h5py.File(f, "r") as h:
        obsid = decode(h["obsid"][:])
        uniq, counts = np.unique(obsid, return_counts=True)
        target = args.obsid or uniq[np.argmax(counts)]
        if target not in set(uniq):
            sys.exit(f"obsid {target!r} not present. Options include: "
                     f"{list(uniq[:5])}")
        m = obsid == target
        beam = h["beam"][:][m]
        freq = h["frequency"][:][m]
        snr = h["snr"][:][m]
        dr = h["driftRate"][:][m]

    print(f"  observation {target}: {m.sum():,} hits, "
          f"{len(np.unique(beam))} beams populated")

    fig, ax = plt.subplots(2, 1, figsize=(15, 9),
                           gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{name}  --  {target}  ({m.sum():,} hits)", fontsize=12)

    # A vertical stripe here = one frequency seen in every beam = RFI.
    sc = ax[0].scatter(freq, beam, c=np.log10(snr), s=4, cmap="magma",
                       alpha=0.7, rasterized=True)
    ax[0].set_xlabel("frequency [MHz]"); ax[0].set_ylabel("beam")
    ax[0].set_title("beam vs frequency -- vertical stripes are field-wide RFI")
    fig.colorbar(sc, ax=ax[0], label="log10(SNR)")

    # How many distinct beams share each frequency (1 Hz tolerance)?
    nb = beams_per_frequency(freq, beam, tol_hz=args.tol)
    ax[1].hist(nb, bins=np.arange(0.5, 66.5, 1), color="#a5533b")
    ax[1].set_xlabel(f"number of distinct beams within {args.tol:g} Hz")
    ax[1].set_ylabel("hits"); ax[1].set_yscale("log")
    ax[1].set_title("multi-beam coincidence -- few beams = plausibly on-sky")

    fig.tight_layout()
    savefig(fig, f"{name}_obs.png")


def beams_per_frequency(freq_mhz, beam, tol_hz=1.0):
    """
    For each hit, count how many DISTINCT beams have a hit within tol_hz.

    This is the cheapest and most powerful RFI discriminant available from
    metadata alone: a real sky signal lands in one or a few coherent beams,
    while local interference lights up the whole field. Compare the multibeam
    spatial filtering in Tremblay et al. 2026 (K2-18b, VLA+MeerKAT).

    O(n log n) via a sorted sweep.
    """
    f_hz = np.asarray(freq_mhz, dtype=np.float64) * 1e6
    order = np.argsort(f_hz)
    fs, bs = f_hz[order], np.asarray(beam)[order]
    lo = np.searchsorted(fs, fs - tol_hz, side="left")
    hi = np.searchsorted(fs, fs + tol_hz, side="right")
    out = np.empty(len(fs), dtype=np.int32)
    for i in range(len(fs)):
        out[i] = len(np.unique(bs[lo[i]:hi[i]]))
    res = np.empty_like(out)
    res[order] = out
    return res


# ----------------------------------------------------------------------------
# cmd: beams -- why the aggregate hits-per-beam histogram has steps
# ----------------------------------------------------------------------------

def cmd_beams(args):
    """
    Explain the step structure in the hits-per-beam histogram.

    Aggregated over a whole file, hits-per-beam is flat up to some beam index
    and then steps down. That is not a beamformer defect: BLUSE assigns one
    coherent beam per catalogue target inside the primary field of view, and
    a sparse patch of sky simply has fewer targets to assign. Beams are then
    filled contiguously from 0, so a pointing with 49 targets populates beams
    0-48 and leaves 49-63 empty. Summed over many pointings, the steps appear
    at whatever beam counts are common in the sample.

    Four panels: the aggregate histogram with its steps; how many observations
    reach each beam index; beams-formed against galactic latitude, which is the
    underlying cause; and the beam-vs-observation occupancy map.
    """
    f = resolve_files([args.file])[0]
    name = os.path.basename(f).replace(".h5", "")
    with h5py.File(f, "r") as h:
        beam = h["beam"][:]
        obs = decode(h["obsid"][:])
        src = decode(h["sourceName"][:])
        ra = h["ra"][:]
        dec = h["dec"][:]

    uo = np.unique(obs)
    n_formed, gal_b, bijective, contiguous = [], [], [], []
    occupancy = np.zeros((len(uo), 64), dtype=bool)

    for i, o in enumerate(uo):
        m = obs == o
        b, s = beam[m], src[m]
        ub = np.unique(b)
        occupancy[i, ub] = True
        n_formed.append(ub.max() + 1)
        # one beam per target? count distinct (beam, source) pairs
        pairs = len(np.unique(np.char.add(b.astype(str), s)))
        bijective.append(pairs == len(ub) and len(ub) == len(np.unique(s)))
        contiguous.append(len(ub) == ub.max() + 1)
        gal_b.append(abs(galactic_latitude(ra[m].mean() * 15.0, dec[m].mean())))

    n_formed = np.array(n_formed)
    gal_b = np.array(gal_b)

    print(f"  {name}: {len(uo)} observations")
    print(f"    beam<->source is 1:1 in {sum(bijective)}/{len(uo)} observations")
    print(f"    beam indices contiguous from 0 in {sum(contiguous)}/{len(uo)}")
    print(f"    beams formed: min {n_formed.min()}, max {n_formed.max()}, "
          f"median {int(np.median(n_formed))}")
    vals, cnts = np.unique(n_formed, return_counts=True)
    print("    distribution: " + "  ".join(f"{v}:{c}" for v, c in zip(vals, cnts)))

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"{name}  --  why hits-per-beam has steps", fontsize=13)

    a = ax[0, 0]
    beams, counts = np.unique(beam, return_counts=True)
    a.bar(beams, counts, width=1.0, color="#7a5c9e")
    for edge in np.unique(n_formed):
        if 0 < edge < 64:
            a.axvline(edge - 0.5, color="#c0392b", ls="--", lw=1, alpha=0.8)
    a.set_xlabel("beam"); a.set_ylabel("hits")
    a.set_title("aggregate hits per beam\n(dashed: beam counts present in sample)")

    a = ax[0, 1]
    reach = occupancy.sum(axis=0)
    a.bar(np.arange(64), reach, width=1.0, color="#3b6ea5")
    a.set_xlabel("beam"); a.set_ylabel("observations reaching this beam")
    a.set_title("every step is an observation that formed fewer beams")

    a = ax[1, 0]
    a.scatter(gal_b, n_formed, s=45, alpha=0.75, color="#4a7c59",
              edgecolor="k", linewidth=0.4)
    a.set_xlabel("|galactic latitude| of field centre [deg]")
    a.set_ylabel("beams formed")
    a.set_title("the cause: sparse sky has fewer targets to point at")

    a = ax[1, 1]
    a.imshow(occupancy[np.argsort(n_formed)], aspect="auto",
             cmap="Greys", interpolation="nearest")
    a.set_xlabel("beam"); a.set_ylabel("observation (sorted by beams formed)")
    a.set_title("beam occupancy per observation")

    fig.tight_layout()
    savefig(fig, f"{name}_beams.png")


def galactic_latitude(ra_deg, dec_deg):
    """Galactic latitude in degrees. Good enough for a diagnostic plot."""
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    ngp_ra, ngp_dec = np.radians(192.8595), np.radians(27.1284)
    sb = (np.sin(ngp_dec) * np.sin(d)
          + np.cos(ngp_dec) * np.cos(d) * np.cos(r - ngp_ra))
    return np.degrees(np.arcsin(np.clip(sb, -1, 1)))


# ----------------------------------------------------------------------------
# cmd: coincidence
# ----------------------------------------------------------------------------

def cmd_coincidence(args):
    f = resolve_files([args.file])[0]
    name = os.path.basename(f).replace(".h5", "")
    with h5py.File(f, "r") as h:
        obsid = decode(h["obsid"][:])
        beam = h["beam"][:]
        freq = h["frequency"][:]
        snr = h["snr"][:]
        dr = h["driftRate"][:]

    nb = np.zeros(len(freq), dtype=np.int32)
    for o in np.unique(obsid):
        m = obsid == o
        nb[m] = beams_per_frequency(freq[m], beam[m], tol_hz=args.tol)

    n = len(nb)
    print(f"\n  {name}: {n:,} hits across {len(np.unique(obsid))} observations")
    print(f"  multi-beam coincidence within {args.tol:g} Hz:")
    for thr in (1, 2, 4, 8, 16, 32):
        k = int(np.sum(nb <= thr))
        print(f"    hits in <= {thr:>2d} beams : {k:>9,}  ({100 * k / n:5.2f}%)")
    surv = (nb <= args.max_beams) & (dr != 0)
    print(f"\n  combined cut (<= {args.max_beams} beams AND non-zero drift): "
          f"{int(surv.sum()):,} hits ({100 * surv.mean():.3f}%)")
    if surv.any():
        print(f"  survivors: SNR {snr[surv].min():.1f} .. {snr[surv].max():.4g}, "
              f"median {np.median(snr[surv]):.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{name}  --  multi-beam coincidence "
                 f"({args.tol:g} Hz tolerance)", fontsize=12)
    ax[0].hist(nb, bins=np.arange(0.5, 66.5, 1), color="#3b6ea5")
    ax[0].set_yscale("log"); ax[0].set_xlabel("distinct beams within tolerance")
    ax[0].set_ylabel("hits"); ax[0].set_title("coincidence multiplicity")

    ax[1].scatter(nb + np.random.uniform(-.3, .3, n), snr, s=1, alpha=0.1,
                  color="#222222", rasterized=True)
    ax[1].set_yscale("log"); ax[1].set_xlabel("distinct beams within tolerance")
    ax[1].set_ylabel("SNR"); ax[1].set_title("SNR vs multiplicity")

    fig.tight_layout()
    savefig(fig, f"{name}_coincidence.png")


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="schema and summary for each file")
    s.add_argument("files", nargs="*")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("meta", help="metadata distribution plots")
    s.add_argument("file")
    s.set_defaults(func=cmd_meta)

    s = sub.add_parser("stamps", help="grid of stamp waterfalls")
    s.add_argument("file")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--ncol", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sort", default="random",
                   choices=["random", "snr", "drift", "nonzero-drift"])
    s.add_argument("--rows", default=None,
                   help="CSV with a `row` column -- plot exactly those stamps "
                        "instead of sorting. Takes the candidates or "
                        "interesting file written by bluse-cluster --match, so "
                        "the residue that matches no documented RFI band can "
                        "be eyeballed. Filtered to this .h5 by the `file` "
                        "column when present")
    s.add_argument("--family", type=int, default=None,
                   help="with --rows, restrict to one family id")
    s.set_defaults(func=cmd_stamps)

    s = sub.add_parser("obs", help="beam vs frequency for one observation")
    s.add_argument("file")
    s.add_argument("--obsid", default=None)
    s.add_argument("--tol", type=float, default=1.0, help="Hz")
    s.set_defaults(func=cmd_obs)

    s = sub.add_parser("beams", help="explain the hits-per-beam step structure")
    s.add_argument("file")
    s.set_defaults(func=cmd_beams)

    s = sub.add_parser("coincidence", help="multi-beam coincidence statistics")
    s.add_argument("file")
    s.add_argument("--tol", type=float, default=1.0, help="Hz")
    s.add_argument("--max-beams", type=int, default=4)
    s.set_defaults(func=cmd_coincidence)

    paths.add_workspace_arg(p)

    args = p.parse_args()
    paths.set_workspace(args.workspace)
    args.func(args)


if __name__ == "__main__":
    main()
