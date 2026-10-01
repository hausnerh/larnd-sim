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
# Q (50 bins, 0-50 ke-, density) -- then overlays data vs sim for prc2 | prc16, and
# ADDS the split the notebook was missing: all hits / single-hit-pixel hits /
# multi-hit-pixel hits, to see which population carries the mid-Q shoulder.
#
# Run on NERSC (where the FLOW files and h5flow live), e.g.:
#   python tests/fsdcube_hitq_datamc.py --outdir hitq_out
#   python tests/fsdcube_hitq_datamc.py --prc 2 16 --sim-r 3 --max-files 20 --outdir hitq_out
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
    references), apply the ADC event cut, and return (q, n, m): q and n are PER-HIT (hit charge ke-,
    and the hit-multiplicity of that hit's pixel within its event, so any Q population is a slice --
    n==1 single-hit, n==2 two-hit, all hits, ...); m is PER-PIXEL (one entry per fired pixel = how
    many hits that pixel saw), for the pixel hit-multiplicity distribution. Multiplicity is per event
    (a pad hit in two events = two n==1 entries / two m==1 pixels).

    Reference layout (standard ndlar_flow): charge/events/ref/charge/calib_prompt_hits/ref is
    (n_links, 2) with col0 = event index, col1 = hit index; .../ref_region[e] = (start, stop) slices
    that ref array for event e; hits live in charge/calib_prompt_hits/data (fields z, y, Q)."""
    import h5py
    REFG = "charge/events/ref/charge/calib_prompt_hits"
    qacc, nacc, macc = [], [], []   # per-hit Q, per-hit parent multiplicity, per-PIXEL multiplicity
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
                z_all = np.asarray(hd["z"]); y_all = np.asarray(hd["y"]); q_all = np.asarray(hd["Q"])
                for e in idx:
                    st, sp = int(r_start[e]), int(r_stop[e])
                    if sp <= st:
                        continue
                    hi = hit_of_ref[st:sp]
                    pid = pixel2id_abs(z_all[hi], y_all[hi])
                    q = q_all[hi].astype(float)            # 'Q' already ke- (no scaling, per notebook)
                    _, inv, counts = np.unique(pid, return_inverse=True, return_counts=True)
                    qacc.append(q)
                    nacc.append(counts[inv].astype(np.int16))    # each HIT's pixel multiplicity
                    macc.append(counts.astype(np.int16))         # each PIXEL's multiplicity (one entry/pixel)
        except Exception as e:
            print(f"  !! skip {path}: {e}")
            continue
        if verbose:
            print(f"  [{fi + 1}/{len(files)}] {Path(path).name}: "
                  f"all={sum(a.size for a in qacc)}", flush=True)
    cat = lambda L, dt=float: np.concatenate(L) if L else np.zeros(0, dt)
    return cat(qacc), cat(nacc, np.int16), cat(macc, np.int16)


def _density(q):
    h, edges = np.histogram(q, bins=QBINS, range=(QMIN, QMAX))
    w = edges[1] - edges[0]
    s = h.sum()
    ctr = 0.5 * (edges[1:] + edges[:-1])
    return ctr, (h / (s * w) if s else h.astype(float))   # density (area = 1)


def sim_files(sim_base, prc, r, var=""):
    cfg = f"prc{prc}_r{r}{var}_patch_noDrop"
    return cfg, sorted((Path(sim_base) / cfg / "flow").rglob("*.FLOW.hdf5"))


def data_files(data_base, prc):
    return sorted((Path(data_base) / f"lt_prc{prc}").rglob("*.hdf5"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prc", type=int, nargs="+", default=[2, 16], help="parity configs to plot (panels)")
    ap.add_argument("--sim-r", type=int, default=3, help="sim readout radius r3/r4 (default 3)")
    ap.add_argument("--sim-var", default="", help="sim variant suffix, e.g. _noFFE (default none)")
    ap.add_argument("--data-base", default=DATA_BASE_DEF)
    ap.add_argument("--sim-base", default=SIM_BASE_DEF)
    ap.add_argument("--adc-min", type=float, default=CUTS["adc_min"])
    ap.add_argument("--adc-max", type=float, default=CUTS["adc_max"])
    ap.add_argument("--max-files", type=int, default=None, help="cap files per sample (quick tests)")
    ap.add_argument("--max-events", type=int, default=None, help="cap selected events per file")
    ap.add_argument("--outdir", default="hitq_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = lambda L: L[: args.max_files] if args.max_files else L

    # gather per-hit (Q, pixel-multiplicity) for data + sim, per prc
    store = {}
    for prc in args.prc:
        df = cap(data_files(args.data_base, prc))
        scfg, sf = sim_files(args.sim_base, prc, args.sim_r, args.sim_var)
        sf = cap(sf)
        print(f"\nprc{prc}: {len(df)} data files, {len(sf)} sim files ({scfg})")
        print(" data:")
        store[(prc, "data")] = accumulate_q(df, args.adc_min, args.adc_max, args.max_events)
        print(" sim:")
        store[(prc, "sim")] = accumulate_q(sf, args.adc_min, args.adc_max, args.max_events)
        store[(prc, "simcfg")] = scfg

    # ---- one figure per pixel-multiplicity population, DATA (solid) + SIM (dashed) on the SAME
    #      canvas for a natural comparison; prc configs as side-by-side panels ----
    POPS = [("mult1",     "single-hit pixels (n = 1)", lambda q, n: q[n == 1]),
            ("mult2",     "two-hit pixels (n = 2)",    lambda q, n: q[n == 2]),
            ("mult3plus", "3+-hit pixels (n >= 3)",    lambda q, n: q[n >= 3]),
            ("multAll",   "all hits",                  lambda q, n: q)]
    for slug, title, selfn in POPS:
        fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                 squeeze=False, sharey=True)
        for ax, prc in zip(axes[0], args.prc):
            # DATA = solid black; SIM = dashed, DUNE palette colour (cycle if several MC series)
            qd, nd, _ = store[(prc, "data")]; qs, ns, _ = store[(prc, "sim")]
            ax.step(*_density(selfn(qd, nd)), where="mid", ls="-",
                    color=DATA_COLOR, lw=1.8, label=f"data prc{prc}")
            ax.step(*_density(selfn(qs, ns)), where="mid", ls="--",
                    color=MC_COLORS[0], lw=1.8, label=f"sim prc{prc} ({store[(prc,'simcfg')]})")
            ax.set_title(f"prc{prc}"); ax.set_xlabel(R"Hit Q [ke$^-$]")
            ax.set_ylabel("Density"); ax.set_xlim(QMIN, QMAX)
            ax.legend(fontsize=8); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
        fig.suptitle(f"FSDCube Hit Q -- {title} -- data vs sim", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{args.outdir}/hitq_split_{slug}.png", dpi=120)
        print(f"wrote {args.outdir}/hitq_split_{slug}.png")

    # ---- pixel hit-multiplicity: for each fired pixel, how many hits it saw -- data vs MC ----
    NM = 10
    mbins = np.arange(0.5, NM + 1.5); mctr = np.arange(1, NM + 1)
    fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                             squeeze=False, sharey=True)
    for ax, prc in zip(axes[0], args.prc):
        for src, ls, col in (("data", "-", DATA_COLOR), ("sim", "--", MC_COLORS[0])):
            m = store[(prc, src)][2]
            h, _ = np.histogram(np.clip(m, 1, NM), bins=mbins); h = h.astype(float); s = h.sum()
            lbl = f"{src} prc{prc}" + ("" if src == "data" else f" ({store[(prc,'simcfg')]})")
            ax.step(mctr, h / s if s else h, where="mid", ls=ls, color=col, lw=1.8, label=lbl)
        ax.set_title(f"prc{prc}"); ax.set_xlabel(f"pixel hit multiplicity  (>= {NM} in last bin)")
        ax.set_ylabel("Fraction of fired pixels"); ax.set_xlim(0.5, NM + 0.5)
        ax.set_xticks(range(1, NM + 1)); ax.legend(fontsize=8); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
    fig.suptitle("FSDCube pixel hit-multiplicity -- data vs sim", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/hitq_pixmult.png", dpi=120)
    print(f"wrote {args.outdir}/hitq_pixmult.png")


if __name__ == "__main__":
    main()
