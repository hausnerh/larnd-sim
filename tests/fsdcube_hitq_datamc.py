#!/usr/bin/env python3
# =============================================================================
# fsdcube_hitq_datamc.py  --  standalone re-implementation of mjkramer's FSDCube
# data/MC "Hit Q for all dt" workflow:
#     https://gist.github.com/mjkramer/a00a0dde7dccba948bedb8cbbe448585  (cells ~5-13)
#
# This does NOT import that notebook -- it copies the WORKFLOW so you can run and
# modify it yourself. It reads the SAME FLOW files (real data + mkramer's larnd-sim)
# with plain h5py (following the charge/events -> calib_prompt_hits references, so NO
# h5flow dependency), builds per-pixel hit lists PER EVENT, and histograms the per-hit charge
# Q (50 bins, 0-50 ke-) -- then overlays data vs sim for prc2 | prc16, and
# ADDS the split the notebook was missing: all hits / single-hit-pixel hits /
# multi-hit-pixel hits, to see which population carries the mid-Q shoulder.
#
# --norm picks the normalization (one plot set per scheme, under <outdir>/<scheme>/):
#   single  -- divide each dataset by its # single-hit pixels (m==1): every dataset then has
#              the SAME single-hit count, so the n==1 curves coincide by construction and any
#              residual in the mult2/3+/all populations is a SHAPE effect, not a normalization one.
#   shower  -- divide by the # showers (selected events with hits): entries per shower.
#   density -- area-normalized, shape only (the original behaviour).
# The pixel hit-multiplicity x-axis is capped by --mult-max (default 20).
#
# Run on NERSC (where the FLOW files and h5flow live), e.g.:
#   python tests/fsdcube_hitq_datamc.py --outdir hitq_out                       # all 3 norms
#   python tests/fsdcube_hitq_datamc.py --norm single --prc 2 16 --sim-r 3 --max-files 20 --outdir hitq_out
#
# Needs: h5py, numpy, matplotlib (all in the larnd env). No h5flow, no GPU, no larnd-sim import.
# =============================================================================
import argparse, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # NERSC /dvs_ro read-only CFS can't take HDF5's lock
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- geometry + binning (verbatim from the notebook) ------------------------
PITCH = 0.372
TILE_X0, TILE_Y0, TILE_Z0 = 48.0675, 133.92, 23.808     # TILE_CENTER
NX, NY = 128, 80
QBINS, QMIN, QMAX = 50, 0, 50                            # ke-
CUTS = dict(adc_min=0.2e6, adc_max=0.5e6)               # event-level cut on charge/events 'ADC'

# DUNE official plot-style palette (Okabe-Ito variant, from dune-plot-style's dune.mplstyle):
# black is data (solid); the rest are the MC colors (dashed), cycled if several MC series.
DUNE_COLORS = ["#000000", "#D55E00", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#F0E442"]
DATA_COLOR, MC_COLORS = DUNE_COLORS[0], DUNE_COLORS[1:]
GRID_COLOR = "#b2b2b2"                                   # dune.mplstyle grid.color

# default file locations (mkramer's; override on the CLI if they move)
SIM_BASE_DEF = "/pscratch/sd/m/mkramer/test/larnd-sim.segFFE3/output/skim_more"
DATA_BASE_DEF = ("/dvs_ro/cfs/cdirs/dunepro/www/data/nd-production/FSDCube"
                 "/Reflow_FSDCube_v3/run-ndlar-flow/Reflow_FSDCube_v3.flow"
                 "/FLOW/Feb2026/cold/scan")


def pixel2id_abs(z, y):
    """Map hit (z, y) [cm] -> an integer pixel id, so hits on the same pad group together
    (mirrors the notebook: z is the 128-pad axis, y the 80-pad axis)."""
    x0 = TILE_Z0 - (NX / 2 - 0.5) * PITCH
    y0 = TILE_Y0 - (NY / 2 - 0.5) * PITCH
    xr = np.round((np.asarray(z) - x0) / PITCH).astype(np.int64)
    yr = np.round((np.asarray(y) - y0) / PITCH).astype(np.int64)
    return xr + NX * yr                                  # pixel2id_rel


def accumulate_q(files, adc_min, adc_max, max_events=None, verbose=True):
    """Loop FLOW files/events with plain h5py (follow the charge/events -> calib_prompt_hits
    references), apply the ADC event cut, and return (q, n, m, qlast, nev): q and n are PER-HIT (hit charge
    ke-, and the hit-multiplicity of that hit's pixel within its event, so any Q population is a slice
    -- n==1 single-hit, n==2 two-hit, all hits, ...); m and qlast are PER-PIXEL (one entry per fired
    pixel) -- m = how many hits that pixel saw, qlast = the charge of that pixel's LAST hit (largest
    drift time within the event). nev = # selected showers (events passing the ADC cut that have hits).
    Multiplicity is per event (a pad hit in two events = two m==1 pixels).

    Reference layout (standard ndlar_flow): charge/events/ref/charge/calib_prompt_hits/ref is
    (n_links, 2) with col0 = event index, col1 = hit index; .../ref_region[e] = (start, stop) slices
    that ref array for event e; hits live in charge/calib_prompt_hits/data (fields z, y, Q)."""
    import h5py
    REFG = "charge/events/ref/charge/calib_prompt_hits"
    qacc, nacc, macc, lacc = [], [], [], []   # per-hit Q, per-hit mult, per-pixel mult, per-pixel last-hit Q
    nev = 0                                    # # selected showers (events with hits) -> per-shower norm
    for fi, path in enumerate(files):
        try:
            try:
                fh = h5py.File(str(path), "r", locking=False)    # read-only CFS: skip the lock
            except TypeError:                                     # older h5py without the locking kwarg
                fh = h5py.File(str(path), "r")
            with fh as h:
                adc = np.asarray(h["charge/events/data"]["ADC"])
                sel = np.ones(adc.shape, bool)
                if adc_min:
                    sel &= adc >= adc_min
                if adc_max:
                    sel &= adc <= adc_max
                idx = np.nonzero(sel)[0]
                if max_events:
                    idx = idx[:max_events]
                reg = h[f"{REFG}/ref_region"]
                r_start = np.asarray(reg["start"]); r_stop = np.asarray(reg["stop"])
                hit_of_ref = np.asarray(h[f"{REFG}/ref"][:, 1])        # ref col1 = hit index
                hd = h["charge/calib_prompt_hits/data"]
                z_all = np.asarray(hd["z"]); y_all = np.asarray(hd["y"])
                q_all = np.asarray(hd["Q"]); t_all = np.asarray(hd["t_drift"])
                for e in idx:
                    st, sp = int(r_start[e]), int(r_stop[e])
                    if sp <= st:
                        continue
                    nev += 1
                    hi = hit_of_ref[st:sp]
                    pid = pixel2id_abs(z_all[hi], y_all[hi])
                    q = q_all[hi].astype(float)            # 'Q' already ke- (no scaling, per notebook)
                    t = t_all[hi].astype(float)
                    _, inv, counts = np.unique(pid, return_inverse=True, return_counts=True)
                    qacc.append(q)
                    nacc.append(counts[inv].astype(np.int16))    # each HIT's pixel multiplicity
                    macc.append(counts.astype(np.int16))         # each PIXEL's multiplicity (one entry/pixel)
                    # charge of each pixel's LAST hit = the hit with the largest drift time in this event
                    o = np.lexsort((t, pid)); pid_s, q_s = pid[o], q[o]   # sort by pixel, then time
                    last = np.ones(pid_s.size, bool); last[:-1] = pid_s[:-1] != pid_s[1:]
                    lacc.append(q_s[last])                       # one entry per pixel = its max-t hit's Q
        except Exception as e:
            print(f"  !! skip {path}: {e}")
            continue
        if verbose:
            print(f"  [{fi + 1}/{len(files)}] {Path(path).name}: "
                  f"all={sum(a.size for a in qacc)}", flush=True)
    cat = lambda L, dt=float: np.concatenate(L) if L else np.zeros(0, dt)
    return cat(qacc), cat(nacc, np.int16), cat(macc, np.int16), cat(lacc), nev


def sim_files(sim_base, prc, r, var=""):
    cfg = f"prc{prc}_r{r}{var}_patch_noDrop"
    return cfg, sorted((Path(sim_base) / cfg / "flow").rglob("*.FLOW.hdf5"))


def data_files(data_base, prc):
    return sorted((Path(data_base) / f"lt_prc{prc}").rglob("*.hdf5"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prc", type=int, nargs="+", default=[2, 16], help="parity configs to plot (panels)")
    ap.add_argument("--sim-r", type=int, nargs="+", default=[3, 4],
                    help="sim readout radius/radii to overlay as separate MC curves, e.g. 3 4 (default both)")
    ap.add_argument("--sim-var", default="", help="sim variant suffix, e.g. _noFFE (default none)")
    ap.add_argument("--data-base", default=DATA_BASE_DEF)
    ap.add_argument("--sim-base", default=SIM_BASE_DEF)
    ap.add_argument("--adc-min", type=float, default=CUTS["adc_min"])
    ap.add_argument("--adc-max", type=float, default=CUTS["adc_max"])
    ap.add_argument("--max-files", type=int, default=None, help="cap files per sample (quick tests)")
    ap.add_argument("--max-events", type=int, default=None, help="cap selected events per file")
    ap.add_argument("--outdir", default="hitq_out")
    ap.add_argument("--norm", nargs="+", default=["single", "shower", "density"],
                    choices=["single", "shower", "density"],
                    help="normalization scheme(s); one plot set per scheme in <outdir>/<scheme>/. "
                         "single = divide each dataset by its # single-hit pixels (entries / single-hit "
                         "pixel) so every dataset has the SAME single-hit count -> residuals are shape, "
                         "not normalization; shower = divide by # showers (entries / shower); "
                         "density = area-normalized, shape only (default: all three)")
    ap.add_argument("--mult-max", type=int, default=20,
                    help="cap the pixel hit-multiplicity x-axis at this value (default 20)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = lambda L: L[: args.max_files] if args.max_files else L

    # gather per-hit (Q, multiplicity, last-hit Q) for data + each sim radius, per prc
    store = {}
    for prc in args.prc:
        df = cap(data_files(args.data_base, prc))
        print(f"\nprc{prc}: {len(df)} data files")
        print(" data:")
        store[(prc, "data")] = accumulate_q(df, args.adc_min, args.adc_max, args.max_events)
        for r in args.sim_r:
            scfg, sf = sim_files(args.sim_base, prc, r, args.sim_var)
            sf = cap(sf)
            print(f" sim r{r} ({scfg}): {len(sf)} files")
            store[(prc, "sim", r)] = accumulate_q(sf, args.adc_min, args.adc_max, args.max_events)

    # one plotting series per prc: data (solid black) + each sim radius (dashed, distinct DUNE colour)
    def series(prc):
        out = [(f"data prc{prc}", DATA_COLOR, "-", store[(prc, "data")])]
        for i, r in enumerate(args.sim_r):
            out.append((f"sim r{r} prc{prc}", MC_COLORS[i % len(MC_COLORS)], "--", store[(prc, "sim", r)]))
        return out

    BINW = (QMAX - QMIN) / QBINS

    def counts(q):
        h, edges = np.histogram(q, bins=QBINS, range=(QMIN, QMAX))
        return 0.5 * (edges[1:] + edges[:-1]), h.astype(float)

    nsingle = lambda tup: max(int((tup[2] == 1).sum()), 1)   # # single-hit pixels (m == 1)
    nshower = lambda tup: max(int(tup[4]), 1)                 # # showers (selected events with hits)

    # Per-dataset normalization WEIGHT for a scheme. density is per-histogram (area = 1, shape only);
    # single/shower are ONE per-dataset scalar applied to every population the same way, so the different
    # multiplicity slices stay on a common scale -> you can tell a pure normalization offset from a real
    # shape difference. 'single' pins the n==1 Q area (and the mult==1 bin) to 1 for every dataset by
    # construction: that is the "same number of single-hit pixels in each dataset" the study wants.
    def weight(scheme, tup, hsum):
        if scheme == "single":
            return 1.0 / nsingle(tup)                    # entries / single-hit pixel
        if scheme == "shower":
            return 1.0 / nshower(tup)                    # entries / shower
        return (1.0 / (hsum * BINW)) if hsum else 0.0    # density (area 1), per histogram

    YLAB = {"single": "Entries / single-hit pixel", "shower": "Entries / shower", "density": "Density"}
    NOTE = {"single": "norm: same # single-hit pixels per dataset",
            "shower": "norm: per shower", "density": "area-normalized (shape only)"}

    POPS = [("mult1",     "single-hit pixels (n = 1)", lambda q, n: q[n == 1]),
            ("mult2",     "two-hit pixels (n = 2)",    lambda q, n: q[n == 2]),
            ("mult3plus", "3+-hit pixels (n >= 3)",    lambda q, n: q[n >= 3]),
            ("multAll",   "all hits",                  lambda q, n: q)]

    # pixel hit-multiplicity x-axis: run out to the data max but cap at --mult-max (default 20)
    NM_raw = max((int(tup[2].max()) for prc in args.prc for _, _, _, tup in series(prc) if tup[2].size),
                 default=1)
    NM = min(NM_raw, args.mult_max)
    mbins = np.arange(0.5, NM + 1.5); mctr = np.arange(1, NM + 1)
    xticks = list(range(1, NM + 1, 1 if NM <= 20 else max(1, NM // 20)))

    for scheme in args.norm:
        sdir = os.path.join(args.outdir, scheme)
        os.makedirs(sdir, exist_ok=True)

        # ---- one figure per pixel-multiplicity population, DATA (solid) + SIM (dashed) on the SAME
        #      canvas for a natural comparison; prc configs as side-by-side panels ----
        for slug, title, selfn in POPS:
            fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                     squeeze=False, sharey=True)
            for ax, prc in zip(axes[0], args.prc):
                for lbl, col, ls, tup in series(prc):
                    ctr, h = counts(selfn(tup[0], tup[1]))
                    ax.step(ctr, h * weight(scheme, tup, h.sum()), where="mid",
                            ls=ls, color=col, lw=1.8, label=lbl)
                ax.set_title(f"prc{prc}"); ax.set_xlabel(R"Hit Q [ke$^-$]")
                ax.set_ylabel(YLAB[scheme]); ax.set_xlim(QMIN, QMAX)
                ax.legend(fontsize=8); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
            fig.suptitle(f"FSDCube Hit Q -- {title} -- data vs sim  [{NOTE[scheme]}]", fontsize=13)
            fig.tight_layout(); fig.savefig(f"{sdir}/hitq_split_{slug}.png", dpi=120)
            print(f"wrote {sdir}/hitq_split_{slug}.png")

        # ---- pixel hit-multiplicity: for each fired pixel, how many hits it saw -- data vs MC ----
        fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                 squeeze=False, sharey=True)
        for ax, prc in zip(axes[0], args.prc):
            for lbl, col, ls, tup in series(prc):
                h, _ = np.histogram(tup[2], bins=mbins); h = h.astype(float)
                # density -> fraction of fired pixels; single/shower -> the per-dataset scalar
                w = (1.0 / h.sum() if h.sum() else 0.0) if scheme == "density" else weight(scheme, tup, h.sum())
                ax.step(mctr, h * w, where="mid", ls=ls, color=col, lw=1.8, label=lbl)
            ax.set_title(f"prc{prc}"); ax.set_xlabel("pixel hit multiplicity")
            ax.set_ylabel("Fraction of fired pixels" if scheme == "density" else YLAB[scheme])
            ax.set_xlim(0.5, NM + 0.5); ax.set_xticks(xticks)
            ax.legend(fontsize=8); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
        fig.suptitle(f"FSDCube pixel hit-multiplicity -- data vs sim  [{NOTE[scheme]}]", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{sdir}/hitq_pixmult.png", dpi=120)
        print(f"wrote {sdir}/hitq_pixmult.png")

        # ---- charge of each pixel's LAST hit -- data vs MC ----
        fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                 squeeze=False, sharey=True)
        for ax, prc in zip(axes[0], args.prc):
            for lbl, col, ls, tup in series(prc):
                ctr, h = counts(tup[3])                   # per-pixel last-hit charge
                ax.step(ctr, h * weight(scheme, tup, h.sum()), where="mid",
                        ls=ls, color=col, lw=1.8, label=lbl)
            ax.set_title(f"prc{prc}"); ax.set_xlabel(R"last-hit Q [ke$^-$]")
            ax.set_ylabel(YLAB[scheme]); ax.set_xlim(QMIN, QMAX)
            ax.legend(fontsize=8); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
        fig.suptitle(f"FSDCube per-pixel last-hit charge -- data vs sim  [{NOTE[scheme]}]", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{sdir}/hitq_lasthit.png", dpi=120)
        print(f"wrote {sdir}/hitq_lasthit.png")


if __name__ == "__main__":
    main()
