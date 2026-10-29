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
# multi-hit-pixel hits, to see which population carries the mid-Q shoulder. It also writes
# hitq_dt.png -- the time between consecutive hits on a pixel (the --dt-field gap between a
# multi-hit pad's successive hits), data vs MC, which probes the FEE reset/re-fire timing.
#
# --norm picks the normalization (one plot set per scheme, under <outdir>/<scheme>/):
#   single  -- divide each dataset by its # single-hit pixels (m==1): every dataset then has
#              the SAME single-hit count, so the n==1 curves coincide by construction and any
#              residual in the mult2/3+/all populations is a SHAPE effect, not a normalization one.
#   shower  -- divide by the # showers (selected events with hits): entries per shower.
#   density -- area-normalized, shape only (the original behaviour).
# The pixel hit-multiplicity x-axis is capped by --mult-max (default 20).
#
# --grid lays a FEE threshold/reset re-sweep (a ti_mult_grid.npz from threshold_induction_study.py
# --mult-grid) ON TOP of the data/MC curves, same bins + same normalization, so you can compare the
# simulated scan directly against the real kramer curves. The sim nodes are dotted, ONE consistent
# line style + weight, coloured by the scanned value (the nominal node is tagged '(nom)'). Read it under
# --norm single: pinning every curve's n==1 area to 1 cancels the rate/livetime offset between the two
# pipelines and leaves SHAPE, which is what makes a sim scan comparable to the real data/MC at all.
# The sweep is PER-PRC: pass one grid per --prc panel ('prc=path' or paths aligned with --prc) so prc2
# and prc16 each get their OWN sweep; a single grid is drawn on every panel (and warns that it is).
#
# --thr-bins ALSO remakes the whole plot set split by each pixel's OWN discriminator threshold, read from
# the real per-pixel map (--thr-map, default larndsim/bin/thresholds_fsdcube.npz -- the same map the
# fsd_cube larnd-sim config loads, so mkramer's MC already carries it: NO re-sim needed). One subdir
# <outdir>/<scheme>/thrbin_<lo-hi>/ per 1-ke- bin centered on the sweep values (--thr-bin-width,
# --thr-centers) + under/overflow; the mono sweep stays the full family in each so a bin's data/MC can be
# read against the mono node at that same threshold. Tests whether the data/MC shoulder is a per-pixel
# threshold-SPREAD effect rather than a mono-threshold one. A mapping-sanity line (single-hit Q vs that
# pixel's threshold) flags any pixel-id convention mismatch between pixel2id_abs and the map.
#
# Run on NERSC (where the FLOW files and h5flow live), e.g.:
#   python tests/fsdcube_hitq_datamc.py --outdir hitq_out                       # all 3 norms
#   python tests/fsdcube_hitq_datamc.py --norm single --prc 2 16 --sim-r 3 --max-files 20 --outdir hitq_out
#   python tests/fsdcube_hitq_datamc.py --norm single --grid 2=$ND_WORK/grid_prc2/ti_mult_grid.npz \
#          16=$ND_WORK/grid_prc16/ti_mult_grid.npz --outdir hitq_out   # per-prc sweep, one per panel
#   python tests/fsdcube_hitq_datamc.py --norm single --thr-bins --grid 2=g2.npz 16=g16.npz --outdir hitq_out
#          # + per-pixel-threshold-binned plot sets (bin centers default to the grids' scan values)
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
DT_UNIT = "ticks"                                        # axis label for the time-between-hits plot
                                                        # (the Δt of --dt-field; t_drift is in ticks)

# DUNE official plot-style palette (Okabe-Ito variant, from dune-plot-style's dune.mplstyle):
# black is data (solid); the rest are the MC colors (dashed), cycled if several MC series.
DUNE_COLORS = ["#000000", "#D55E00", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#F0E442"]
DATA_COLOR, MC_COLORS = DUNE_COLORS[0], DUNE_COLORS[1:]
GRID_COLOR = "#b2b2b2"                                   # dune.mplstyle grid.color

# the REAL per-pixel discriminator threshold map (electrons) -- the same map the fsd_cube
# larnd-sim config loads (larndsim/config/config.yaml: PIXEL_THRESHOLDS_FILE), so mkramer's MC
# was simulated with it and the detector is modelled by it. Used to bin hits by their pixel's
# own threshold. Keys are pixel ids on the SAME convention as pixel2id_abs (xr + NX*yr).
THR_MAP_DEF = str(Path(__file__).resolve().parents[1] / "larndsim" / "bin" / "thresholds_fsdcube.npz")

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


def load_thr_map(path):
    """Load the per-pixel threshold npz (keys=pixel ids, values=thresholds e-, default scalar)
    into a dense lookup array in ke- (index = pixel id), so thr_of_id[pid] is O(1).

    Returns (thr_dense_ke, default_ke). thr_dense has length max(key)+1 (10240 for the full
    128x80 FSDCube), prefilled with the default so any in-range id resolves even if the map
    were sparse. pixel2id_abs ids out of [0, len) fall back to the default via thr_lookup()."""
    d = np.load(path, allow_pickle=True)
    keys = np.asarray(d["keys"], np.int64)
    vals = np.asarray(d["values"], float) / 1000.0                     # e- -> ke-
    default = float(np.asarray(d["default"], float).ravel()[0]) / 1000.0
    dense = np.full(int(keys.max()) + 1, default, float)
    dense[keys] = vals
    return dense, default


def thr_lookup(pid, thr_dense, default):
    """Per-pixel threshold (ke-) for pixel ids `pid`, with out-of-range ids -> default."""
    pid = np.asarray(pid, np.int64)
    out = np.full(pid.shape, default, float)
    ok = (pid >= 0) & (pid < thr_dense.size)
    out[ok] = thr_dense[pid[ok]]
    return out


def accumulate_q(files, adc_min, adc_max, max_events=None, verbose=True,
                 thr_dense=None, thr_default=5.0, t_field="t_drift"):
    """Loop FLOW files/events with plain h5py (follow the charge/events -> calib_prompt_hits
    references), apply the ADC event cut, and return
    (q, n, m, qlast, nev, thr_hit, thr_pix, dt, dt_thr):
    q and n are PER-HIT (hit charge ke-, and the hit-multiplicity of that hit's pixel within its
    event, so any Q population is a slice -- n==1 single-hit, n==2 two-hit, all hits, ...); m and
    qlast are PER-PIXEL (one entry per fired pixel) -- m = how many hits that pixel saw, qlast = the
    charge of that pixel's LAST hit (largest drift time within the event). nev = # selected showers
    (events passing the ADC cut that have hits). Multiplicity is per event (a pad hit in two events
    = two m==1 pixels).

    thr_hit (aligned 1:1 with q/n) and thr_pix (aligned 1:1 with m/qlast) are each fired pixel's
    OWN discriminator threshold (ke-) looked up from thr_dense via pixel2id_abs -- the lever for
    the per-pixel-threshold binning. Empty when thr_dense is None.

    dt is PER-HIT-PAIR: for every pixel hit more than once in an event, the `t_field` gap between
    its consecutive hits (sorted in time) -- i.e. how long after one hit the same pad fires again.
    One entry per adjacent pair, so an m-hit pixel contributes m-1 entries; single-hit pixels
    contribute none. dt_thr (aligned 1:1 with dt) is that pixel's own threshold (ke-), so the
    time-between-hits plot can be split by --thr-bins too; empty when thr_dense is None.

    Reference layout (standard ndlar_flow): charge/events/ref/charge/calib_prompt_hits/ref is
    (n_links, 2) with col0 = event index, col1 = hit index; .../ref_region[e] = (start, stop) slices
    that ref array for event e; hits live in charge/calib_prompt_hits/data (fields z, y, Q, t_drift)."""
    import h5py
    REFG = "charge/events/ref/charge/calib_prompt_hits"
    qacc, nacc, macc, lacc = [], [], [], []   # per-hit Q, per-hit mult, per-pixel mult, per-pixel last-hit Q
    thacc, tpacc = [], []                      # per-hit threshold, per-pixel threshold (ke-)
    dtacc, dtthacc = [], []                     # per-pair Δt-between-hits, and that pair's pixel threshold
    do_thr = thr_dense is not None
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
                q_all = np.asarray(hd["Q"]); t_all = np.asarray(hd[t_field])
                for e in idx:
                    st, sp = int(r_start[e]), int(r_stop[e])
                    if sp <= st:
                        continue
                    nev += 1
                    hi = hit_of_ref[st:sp]
                    pid = pixel2id_abs(z_all[hi], y_all[hi])
                    q = q_all[hi].astype(float)            # 'Q' already ke- (no scaling, per notebook)
                    t = t_all[hi].astype(float)
                    uniq, inv, counts = np.unique(pid, return_inverse=True, return_counts=True)
                    qacc.append(q)
                    nacc.append(counts[inv].astype(np.int16))    # each HIT's pixel multiplicity
                    macc.append(counts.astype(np.int16))         # each PIXEL's multiplicity (one entry/pixel)
                    # charge of each pixel's LAST hit = the hit with the largest drift time in this event
                    o = np.lexsort((t, pid)); pid_s, q_s, t_s = pid[o], q[o], t[o]   # sort by pixel, then time
                    last = np.ones(pid_s.size, bool); last[:-1] = pid_s[:-1] != pid_s[1:]
                    lacc.append(q_s[last])                       # one entry per pixel = its max-t hit's Q
                    # time between consecutive hits on the SAME pixel: adjacent sorted pairs that
                    # stay on one pixel (one entry per pair, so m-hit pixel -> m-1 gaps, all >= 0).
                    if pid_s.size > 1:
                        sm = pid_s[1:] == pid_s[:-1]
                        if sm.any():
                            dtacc.append((t_s[1:] - t_s[:-1])[sm])
                            if do_thr:
                                dtthacc.append(thr_lookup(pid_s[1:][sm], thr_dense, thr_default))
                    if do_thr:
                        # per-HIT threshold aligns with q (look up each hit's pixel); per-PIXEL
                        # threshold aligns with macc(counts) AND lacc -- both are in np.unique(pid)
                        # (sorted-unique) order, which is exactly `uniq`.
                        thacc.append(thr_lookup(pid, thr_dense, thr_default))
                        tpacc.append(thr_lookup(uniq, thr_dense, thr_default))
        except Exception as e:
            print(f"  !! skip {path}: {e}")
            continue
        if verbose:
            print(f"  [{fi + 1}/{len(files)}] {Path(path).name}: "
                  f"all={sum(a.size for a in qacc)}", flush=True)
    cat = lambda L, dt=float: np.concatenate(L) if L else np.zeros(0, dt)
    return (cat(qacc), cat(nacc, np.int16), cat(macc, np.int16), cat(lacc), nev,
            cat(thacc), cat(tpacc), cat(dtacc), cat(dtthacc))


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
    ap.add_argument("--dt-field", default="t_drift",
                    help="calib_prompt_hits field used as a hit's time, for the time-between-hits "
                         "plot (default t_drift)")
    ap.add_argument("--dt-max", type=float, default=None,
                    help="upper x-limit (same units as --dt-field) for the time-between-hits plot "
                         "(default: auto = 99th percentile of all Δt, shared across data/MC/prc)")
    ap.add_argument("--dt-bins", type=int, default=50,
                    help="number of bins for the time-between-hits plot (default 50)")
    ap.add_argument("--dt-zoom-max", type=float, default=100.0,
                    help="upper x-limit for the ZOOMED companion plot hitq_dt_zoom.png, which shows "
                         "the fine structure near Δt=0 at native resolution (default 100 ticks)")
    ap.add_argument("--dt-zoom-binw", type=float, default=1.0,
                    help="bin width for the zoomed time-between-hits plot (default 1 tick)")
    ap.add_argument("--grid", nargs="+", default=None,
                    help="overlay a FEE threshold/reset re-sweep (a ti_mult_grid.npz from "
                         "tests/threshold_induction_study.py --mult-grid) ON TOP of the data/MC curves, "
                         "same bins + same normalization. Sim nodes are dotted and coloured by the "
                         "scanned value (nominal tagged '(nom)'). Read it under --norm single. "
                         "PER-PRC: pass one grid PER --prc panel so each gets its OWN sweep, either as "
                         "'prc=path' tokens (e.g. --grid 2=g2.npz 16=g16.npz) or as bare paths aligned "
                         "with --prc order; a single bare path is drawn on every panel (and warns).")
    ap.add_argument("--grid-scan", choices=["threshold", "reset"], default="threshold",
                    help="which grid axis to lay down as the overlay family (default threshold)")
    ap.add_argument("--grid-fixed", type=float, default=None,
                    help="value to hold the OTHER grid axis at (default: the grid's base threshold/reset). "
                         "threshold scan -> fixes reset; reset scan -> fixes threshold.")
    ap.add_argument("--grid-reset-per-prc", action="store_true",
                    help="threshold scan only: fix the overlaid sim reset to the node MATCHING each data "
                         "prc. prc is a periodic-reset WINDOW, so cycles = round(prc * --prc-step-us / grid "
                         "ts) -> prc2 = 128 cyc (12.8 us), prc16 = 1024 cyc (102.4 us), prc8 = 512 (nominal). "
                         "Each panel is then compared to the sim read out at ITS own reset instead of the "
                         "grid's base, which is the whole point of the prc comparison. Overrides --grid-fixed. "
                         "Lets one shared grid serve every prc: --grid one.npz (drawn on all panels).")
    ap.add_argument("--prc-step-us", type=float, default=6.4,
                    help="microseconds per prc step for --grid-reset-per-prc (default 6.4; prc8 = 51.2 us = "
                         "the nominal fsd_cube reset = 512 cycles at ts=0.1 us).")
    ap.add_argument("--grid-max-curves", type=int, default=6,
                    help="cap how many scan nodes are overlaid, evenly subsampled with endpoints kept "
                         "(default 6)")
    ap.add_argument("--thr-bins", action="store_true",
                    help="ALSO make the full plot set split by each pixel's OWN discriminator threshold "
                         "(from --thr-map): one subdir <outdir>/<norm>/thrbin_<lo-hi>/ per 1 ke- bin "
                         "centered on the threshold-scan values + under/overflow. Tests whether the "
                         "data/MC shape is explained by a per-pixel threshold spread rather than a "
                         "mono-threshold. mkramer's MC already carries this map, so no re-sim is needed.")
    ap.add_argument("--thr-map", default=THR_MAP_DEF,
                    help="per-pixel threshold npz (keys/values/default) for --thr-bins "
                         f"(default: the fsd_cube map {Path(THR_MAP_DEF).name})")
    ap.add_argument("--thr-bin-width", type=float, default=1.0,
                    help="width (ke-) of each per-pixel-threshold bin, centered on the scan values "
                         "(default 1.0)")
    ap.add_argument("--thr-centers", type=float, nargs="+", default=None,
                    help="bin centers (ke-) for --thr-bins (default: the --grid threshold-scan values, "
                         "else 2.5 3.75 5.0 6.25 7.5)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = lambda L: L[: args.max_files] if args.max_files else L

    thr_dense = thr_def = None
    if args.thr_bins:
        thr_dense, thr_def = load_thr_map(args.thr_map)
        print(f"per-pixel threshold map: {args.thr_map}  ({thr_dense.size} pixels, "
              f"{thr_dense.min():.2f}-{thr_dense.max():.2f} ke-, default {thr_def:.2f} ke-)")

    # gather per-hit (Q, multiplicity, last-hit Q[, thresholds]) for data + each sim radius, per prc
    store = {}
    for prc in args.prc:
        df = cap(data_files(args.data_base, prc))
        print(f"\nprc{prc}: {len(df)} data files")
        print(" data:")
        store[(prc, "data")] = accumulate_q(df, args.adc_min, args.adc_max, args.max_events,
                                            thr_dense=thr_dense, thr_default=thr_def or 5.0,
                                            t_field=args.dt_field)
        for r in args.sim_r:
            scfg, sf = sim_files(args.sim_base, prc, r, args.sim_var)
            sf = cap(sf)
            print(f" sim r{r} ({scfg}): {len(sf)} files")
            store[(prc, "sim", r)] = accumulate_q(sf, args.adc_min, args.adc_max, args.max_events,
                                                  thr_dense=thr_dense, thr_default=thr_def or 5.0,
                                                  t_field=args.dt_field)

    # time-between-hits x-axis: shared across data/MC/prc so the curves are comparable. Auto =
    # 99th percentile of ALL Δt (robust to a long tail), overridable with --dt-max.
    _alldt = [store[k][7] for k in store if store[k][7].size]
    _alldt = np.concatenate(_alldt) if _alldt else np.zeros(0)
    dt_hi = float(args.dt_max) if args.dt_max else (float(np.percentile(_alldt, 99)) if _alldt.size else 10.0)
    dt_hi = max(dt_hi, 1e-9)
    dtbins = np.linspace(0.0, dt_hi, args.dt_bins + 1)
    dtctr = 0.5 * (dtbins[1:] + dtbins[:-1])
    print(f"\ntime-between-hits: field={args.dt_field}, {int(_alldt.size)} pairs total, "
          f"x-axis [0, {dt_hi:.3g}] {DT_UNIT} in {args.dt_bins} bins")
    # zoomed companion (hitq_dt_zoom.png): the fine structure right at the floor, where the reset
    # cadence lives, at native resolution -- default 0..100 ticks in 1-tick bins. Fixed (not
    # percentile-driven) so data/MC are compared bin-for-bin, and density-normalized to each
    # dataset's GLOBAL pair count so the y-value is the true fraction of pairs in that 1-tick bin.
    dtz_w = max(float(args.dt_zoom_binw), 1e-9)
    dtzbins = np.arange(0.0, float(args.dt_zoom_max) + dtz_w, dtz_w)
    dtzctr = 0.5 * (dtzbins[1:] + dtzbins[:-1])
    # directly answer "is the minimum Δt different data vs MC?": floor + low percentiles per dataset
    print(f"  min / P0.1 / P1 / P5 Δt per dataset ({DT_UNIT}):")
    for k in sorted(store, key=lambda kk: (kk[0], kk[1] != "data", kk)):
        d = store[k][7]
        tag = "data" if k[1] == "data" else f"sim r{k[2]}"
        if d.size:
            lo = np.percentile(d, [0.1, 1.0, 5.0])
            print(f"    prc{k[0]} {tag:<8}: min={d.min():.3g}  P0.1={lo[0]:.3g}  "
                  f"P1={lo[1]:.3g}  P5={lo[2]:.3g}   (n={d.size})")
        else:
            print(f"    prc{k[0]} {tag:<8}: (no multi-hit pixels)")

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

    # ---- optional overlay: a FEE threshold/reset re-sweep (ti_mult_grid.npz) on the SAME axes ----
    # The grid npz is self-describing (saved by threshold_induction_study.py --mult-grid), so read it
    # with bare numpy -- no GPU, no study import, still the plain-h5py env. Each selected node becomes a
    # dotted curve weighted EXACTLY like the data/MC (same weight()); under --norm single that pins every
    # curve's n==1 area to 1, so the sim scan is shape-comparable to the real curves despite being a
    # completely different sample + pipeline. node_phq/phn = per-hit, node_n = per-pixel mult, node_ql =
    # per-pixel last-hit Q (newer grids only). Charges are electrons in the grid -> /1000 to ke-.
    def build_grid_nodes(path, prc=None):
        """Load ONE ti_mult_grid.npz -> the list of overlay node dicts for the selected scan.

        With --grid-reset-per-prc, a threshold scan fixes the reset to the node matching this panel's
        `prc` (a reset window: cycles = round(prc * --prc-step-us / grid ts)), so each data prc is
        compared to the sim read out at its OWN reset rather than the grid's base."""
        nodes = []
        g = np.load(path, allow_pickle=True)
        g_thr = np.asarray(g["node_thr"], float); g_rst = np.asarray(g["node_rst"], int)
        base_thr = float(g["meta_base_threshold"]); base_rst = int(g["meta_base_reset"])
        ts_meta = float(g["meta_ts"]) if "meta_ts" in g.files else 0.1
        nsh_grid = max(int(g["meta_n_shower"]), 1)
        samp = str(g["meta_sample"].item()) if "meta_sample" in g.files else "showers"
        has_ql = "node_ql" in g.files
        if args.grid_scan == "threshold":
            if args.grid_reset_per_prc and prc is not None:
                fixed = int(round(prc * args.prc_step_us / ts_meta))   # prc window -> reset cycles
            elif args.grid_fixed is not None:
                fixed = int(args.grid_fixed)
            else:
                fixed = base_rst
            idx = sorted((i for i in range(len(g_thr)) if g_rst[i] == fixed), key=lambda i: g_thr[i])
            labof = lambda i: f"sim {g_thr[i]/1000:.2f} ke$^-$" + (" (nom)" if g_thr[i] == base_thr else "")
            fixnote = (f"reset {'off' if fixed < 0 else f'{fixed} cyc = {fixed*ts_meta:.1f} us'}"
                       + (f" (prc{prc})" if args.grid_reset_per_prc and prc is not None else ""))
            if not idx and args.grid_reset_per_prc and prc is not None:
                print(f"  !! --grid-reset-per-prc: grid has no node at reset {fixed} cyc for prc{prc} "
                      f"(resets present: {sorted(set(int(r) for r in g_rst))}) -> this panel gets NO overlay")
        else:
            fixed = float(args.grid_fixed) if args.grid_fixed is not None else base_thr
            idx = sorted((i for i in range(len(g_thr)) if g_thr[i] == fixed),
                         key=lambda i: (np.inf if g_rst[i] < 0 else g_rst[i]))
            labof = lambda i: ("sim reset off" if g_rst[i] < 0 else f"sim {g_rst[i]} cyc") \
                + (" (nom)" if g_rst[i] == base_rst else "")
            fixnote = f"threshold {fixed/1000:.2f} ke$^-$"
        if 0 < args.grid_max_curves < len(idx):     # subsample, keeping the two endpoints
            keep = sorted(set(np.linspace(0, len(idx) - 1, args.grid_max_curves).round().astype(int)))
            idx = [idx[k] for k in keep]
        cmap = plt.get_cmap("plasma")
        for k, i in enumerate(idx):
            nn = np.asarray(g["node_n"][i], np.int16)
            qli = np.asarray(g["node_ql"][i], float) if has_ql else np.empty(0)
            nodes.append(dict(
                label=labof(i), color=cmap(0.06 + 0.82 * k / max(len(idx) - 1, 1)),
                is_nom=bool(g_thr[i] == base_thr and g_rst[i] == base_rst),
                thr=g_thr[i] / 1000.0,                        # node mono-threshold (ke-)
                phq=np.asarray(g["node_phq"][i], float) / 1000.0,
                phn=np.asarray(g["node_phn"][i], np.int16), n=nn,
                ql=(qli / 1000.0 if qli.size else None),
                nsingle=max(int((nn == 1).sum()), 1), nsh=nsh_grid))
        print(f"grid overlay [{path}]: {len(nodes)} sim nodes "
              f"({args.grid_scan} scan @ {fixnote}, sample={samp}"
              f"{', no last-hit Q' if not has_ql else ''})")
        return nodes

    # ---- map --grid tokens -> {prc: path} so EACH prc panel gets its OWN sweep. Tokens are either
    # 'prc=path' (explicit), bare paths aligned with --prc order, or a single bare path drawn on all. ----
    def parse_grid_arg(tokens, prcs):
        if not tokens:
            return {}
        if all("=" in t and t.split("=", 1)[0].strip().lstrip("-").isdigit() for t in tokens):
            m = {int(t.split("=", 1)[0]): t.split("=", 1)[1] for t in tokens}
            miss = [p for p in prcs if p not in m]
            if miss:
                print(f"  !! --grid: no sweep given for prc {miss} -> those panels get no overlay")
            return {p: m[p] for p in prcs if p in m}
        if len(tokens) == 1:
            if len(prcs) > 1:
                print(f"  !! one --grid for {len(prcs)} prc panels -> the SAME sweep on all; pass one "
                      f"per prc ('prc=path' or {len(prcs)} paths) for panel-specific sweeps.")
            return {p: tokens[0] for p in prcs}
        if len(tokens) == len(prcs):
            return {p: tokens[i] for i, p in enumerate(prcs)}
        raise SystemExit(f"--grid: {len(tokens)} paths for {len(prcs)} prc; pass 1, {len(prcs)}, "
                         f"or 'prc=path' tokens.")

    grid_by_prc = parse_grid_arg(args.grid, args.prc)
    # Build per prc (not cached by path): with --grid-reset-per-prc the selected reset node depends on
    # the prc, so one shared grid yields a DIFFERENT overlay per panel (prc2 -> reset 128, prc16 -> 1024).
    grid_nodes_by_prc = {prc: build_grid_nodes(grid_by_prc[prc], prc)
                         for prc in args.prc if prc in grid_by_prc}
    all_grid_nodes = [gn for prc in args.prc for gn in grid_nodes_by_prc.get(prc, [])]

    def grid_weight(scheme, gn, hsum):               # mirrors weight(), for a grid node
        if scheme == "single":
            return 1.0 / gn["nsingle"]
        if scheme == "shower":
            return 1.0 / gn["nsh"]
        return (1.0 / (hsum * BINW)) if hsum else 0.0

    # threshold/reset-scan curves: ONE consistent dotted style + line weight across the whole
    # family, with only the colour varying by scanned value (the nominal is still tagged in the
    # legend label, not by a heavier line) -- keeps the data/kramer-sim styling untouched.
    gstyle = lambda gn: dict(ls=":", lw=1.6, color=gn["color"], alpha=0.95, zorder=3)
    GSUF = "  (+ sim scan)" if all_grid_nodes else ""
    LEG_FS = 7 if all_grid_nodes else 8

    # pixel hit-multiplicity x-axis: run out to the data (and overlaid sim) max, capped at --mult-max
    NM_raw = max([int(tup[2].max()) for prc in args.prc for _, _, _, tup in series(prc) if tup[2].size]
                 + [int(gn["n"].max()) for gn in all_grid_nodes if gn["n"].size],
                 default=1)
    NM = min(NM_raw, args.mult_max)
    mbins = np.arange(0.5, NM + 1.5); mctr = np.arange(1, NM + 1)
    xticks = list(range(1, NM + 1, 1 if NM <= 20 else max(1, NM // 20)))

    # The whole plot set for one normalization scheme, with every data/MC series passed through `sub`
    # (identity for the full sample; a per-pixel-threshold mask for a threshold bin). The dotted mono
    # sweep stays the FULL family in every panel so a threshold-bin's data/MC can be read against the
    # mono node at that same threshold. `tag` -> a subdir; `tnote` -> appended to each suptitle.
    def emit_plots(scheme, sub, tag="", tnote=""):
        sdir = os.path.join(args.outdir, scheme, tag) if tag else os.path.join(args.outdir, scheme)
        os.makedirs(sdir, exist_ok=True)

        # ---- one figure per pixel-multiplicity population, DATA (solid) + SIM (dashed) + sweep ----
        for slug, title, selfn in POPS:
            fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                     squeeze=False, sharey=True)
            for ax, prc in zip(axes[0], args.prc):
                for lbl, col, ls, tup0 in series(prc):
                    tup = sub(tup0)
                    ctr, h = counts(selfn(tup[0], tup[1]))
                    ax.step(ctr, h * weight(scheme, tup, h.sum()), where="mid",
                            ls=ls, color=col, lw=1.8, label=lbl)
                for gn in grid_nodes_by_prc.get(prc, []):   # this prc's dotted sim scan nodes
                    qg = selfn(gn["phq"], gn["phn"])
                    if qg.size:
                        cg, hg = counts(qg)
                        ax.step(cg, hg * grid_weight(scheme, gn, hg.sum()), where="mid",
                                label=gn["label"], **gstyle(gn))
                ax.set_title(f"prc{prc}"); ax.set_xlabel(R"Hit Q [ke$^-$]")
                ax.set_ylabel(YLAB[scheme]); ax.set_xlim(QMIN, QMAX)
                ax.legend(fontsize=LEG_FS); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
            fig.suptitle(f"FSDCube Hit Q -- {title} -- data vs sim  [{NOTE[scheme]}]{GSUF}{tnote}",
                         fontsize=13)
            fig.tight_layout(); fig.savefig(f"{sdir}/hitq_split_{slug}.png", dpi=120)
            plt.close(fig)
            print(f"wrote {sdir}/hitq_split_{slug}.png")

        # ---- pixel hit-multiplicity: for each fired pixel, how many hits it saw -- data vs MC ----
        fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                 squeeze=False, sharey=True)
        for ax, prc in zip(axes[0], args.prc):
            for lbl, col, ls, tup0 in series(prc):
                tup = sub(tup0)
                h, _ = np.histogram(tup[2], bins=mbins); h = h.astype(float)
                # density -> fraction of fired pixels; single/shower -> the per-dataset scalar
                w = (1.0 / h.sum() if h.sum() else 0.0) if scheme == "density" else weight(scheme, tup, h.sum())
                ax.step(mctr, h * w, where="mid", ls=ls, color=col, lw=1.8, label=lbl)
            for gn in grid_nodes_by_prc.get(prc, []):   # this prc's dotted sim scan nodes
                hg, _ = np.histogram(gn["n"], bins=mbins); hg = hg.astype(float)
                wg = (1.0 / hg.sum() if hg.sum() else 0.0) if scheme == "density" \
                    else grid_weight(scheme, gn, hg.sum())
                ax.step(mctr, hg * wg, where="mid", label=gn["label"], **gstyle(gn))
            ax.set_title(f"prc{prc}"); ax.set_xlabel("pixel hit multiplicity")
            ax.set_ylabel("Fraction of fired pixels" if scheme == "density" else YLAB[scheme])
            ax.set_xlim(0.5, NM + 0.5); ax.set_xticks(xticks)
            ax.legend(fontsize=LEG_FS); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
        fig.suptitle(f"FSDCube pixel hit-multiplicity -- data vs sim  [{NOTE[scheme]}]{GSUF}{tnote}",
                     fontsize=13)
        fig.tight_layout(); fig.savefig(f"{sdir}/hitq_pixmult.png", dpi=120)
        plt.close(fig)
        print(f"wrote {sdir}/hitq_pixmult.png")

        # ---- charge of each pixel's LAST hit -- data vs MC ----
        fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                 squeeze=False, sharey=True)
        for ax, prc in zip(axes[0], args.prc):
            for lbl, col, ls, tup0 in series(prc):
                tup = sub(tup0)
                ctr, h = counts(tup[3])                   # per-pixel last-hit charge
                ax.step(ctr, h * weight(scheme, tup, h.sum()), where="mid",
                        ls=ls, color=col, lw=1.8, label=lbl)
            for gn in grid_nodes_by_prc.get(prc, []):   # this prc's dotted sim nodes (last-hit Q)
                if gn["ql"] is None:
                    continue
                cg, hg = counts(gn["ql"])
                ax.step(cg, hg * grid_weight(scheme, gn, hg.sum()), where="mid",
                        label=gn["label"], **gstyle(gn))
            ax.set_title(f"prc{prc}"); ax.set_xlabel(R"last-hit Q [ke$^-$]")
            ax.set_ylabel(YLAB[scheme]); ax.set_xlim(QMIN, QMAX)
            ax.legend(fontsize=LEG_FS); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
        fig.suptitle(f"FSDCube per-pixel last-hit charge -- data vs sim  [{NOTE[scheme]}]{GSUF}{tnote}",
                     fontsize=13)
        fig.tight_layout(); fig.savefig(f"{sdir}/hitq_lasthit.png", dpi=120)
        plt.close(fig)
        print(f"wrote {sdir}/hitq_lasthit.png")

        # ---- time between consecutive hits on a pixel -- data vs MC ----
        # For every pixel hit more than once in an event, the Δt (in --dt-field units) between its
        # adjacent-in-time hits: how long after one hit the same pad fires again. Only multi-hit
        # pixels contribute, so it is empty when no pixel fires twice. No sweep overlay here -- the
        # ti_mult_grid carries per-hit CHARGE but no per-hit TIME, so there is nothing to draw.
        # Drawn twice: the full-range view, then a zoomed companion at the floor. The zoom's density
        # is normalized to each dataset's GLOBAL pair count (not the in-window sum) so a 1-tick bin's
        # height is the true fraction of all that dataset's pairs -- directly comparable data vs MC.
        for _bins, _ctr, _fname, _zlab, _gnorm in (
                (dtbins, dtctr, "hitq_dt.png", "", False),
                (dtzbins, dtzctr, "hitq_dt_zoom.png",
                 f"  (zoom 0-{args.dt_zoom_max:g} {DT_UNIT}, {args.dt_zoom_binw:g}-{DT_UNIT} bins)", True)):
            fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6),
                                     squeeze=False, sharey=True)
            for ax, prc in zip(axes[0], args.prc):
                for lbl, col, ls, tup0 in series(prc):
                    tup = sub(tup0)
                    hh, _ = np.histogram(tup[7], bins=_bins); hh = hh.astype(float)
                    # density -> fraction of hit pairs; single/shower -> the per-dataset scalar
                    if scheme == "density":
                        _tot = (float(tup[7].size) if _gnorm else hh.sum())
                        w = (1.0 / _tot) if _tot else 0.0
                    else:
                        w = weight(scheme, tup, hh.sum())
                    ax.step(_ctr, hh * w, where="mid", ls=ls, color=col, lw=1.8, label=lbl)
                ax.set_title(f"prc{prc}"); ax.set_xlabel(f"Δt between hits on a pixel [{DT_UNIT}]")
                ax.set_ylabel("Fraction of hit pairs" if scheme == "density" else YLAB[scheme])
                ax.set_xlim(_bins[0], _bins[-1])
                ax.legend(fontsize=LEG_FS); ax.grid(alpha=.5, color=GRID_COLOR, lw=0.6)
            fig.suptitle(f"FSDCube time between consecutive hits on a pixel -- data vs sim  "
                         f"[{NOTE[scheme]}]{tnote}{_zlab}", fontsize=13)
            fig.tight_layout(); fig.savefig(f"{sdir}/{_fname}", dpi=120)
            plt.close(fig)
            print(f"wrote {sdir}/{_fname}")

    # ---- per-pixel-threshold bins (optional): 1-ke- windows centered on the scan values + under/over.
    # `sub` masks a dataset tuple to the pixels whose OWN threshold falls in [lo,hi): per-hit arrays
    # (q,n,thr_hit) by thr_hit[tup[5]], per-pixel arrays (m,qlast,thr_pix) by thr_pix[tup[6]]; nev (the
    # shower count, tup[4]) is left whole. nsingle/nshower then come out bin-local / per-dataset exactly
    # as weight() needs them. ----
    thr_bins = []
    if args.thr_bins:
        centers = (sorted(args.thr_centers) if args.thr_centers
                   else sorted({gn["thr"] for gn in all_grid_nodes}) if all_grid_nodes
                   else [2.5, 3.75, 5.0, 6.25, 7.5])
        W = args.thr_bin_width
        edges0 = centers[0] - W / 2.0
        edges1 = centers[-1] + W / 2.0
        thr_bins = [("under", -np.inf, edges0, f"thr < {edges0:.2f} ke-")]
        thr_bins += [(f"{c - W / 2:.2f}-{c + W / 2:.2f}", c - W / 2, c + W / 2,
                      f"thr in [{c - W / 2:.2f}, {c + W / 2:.2f}) ke- (center {c:.2f})") for c in centers]
        thr_bins.append(("over", edges1, np.inf, f"thr >= {edges1:.2f} ke-"))

        # mapping-sanity: among single-hit (n==1) hits, the fraction recorded BELOW their pixel's own
        # threshold. If pixel2id_abs and the map share a pixel-id convention this is small (noise/width
        # only); if the convention is scrambled, Q and threshold decouple and this spikes.
        print("\nper-pixel-threshold mapping sanity (single-hit Q vs that pixel's threshold):")
        for key, tup in store.items():
            q, n, thr_h = tup[0], tup[1], tup[5]
            s = n == 1
            if s.any():
                frac_lo = float(np.mean(q[s] < 0.8 * thr_h[s]))
                print(f"  {key}: median Q={np.median(q[s]):.2f} ke-, median thr={np.median(thr_h[s]):.2f}"
                      f" ke-, frac(Q<0.8*thr)={frac_lo:.3f}")
        cov = []  # coverage: fraction of per-pixel entries that land in SOME center bin (not a gap)
        for key, tup in store.items():
            tp = tup[6]
            if tp.size:
                inbin = np.zeros(tp.size, bool)
                for _, lo, hi, _ in thr_bins[1:-1]:
                    inbin |= (tp >= lo) & (tp < hi)
                cov.append(float(inbin.mean()))
        if cov:
            print(f"  center-bin coverage (per-pixel): {np.mean(cov):.3f} "
                  f"(1 - this = pixels in inter-bin gaps or the under/over tails)")

    def bin_sub(lo, hi):
        def sub(tup):
            hm = (tup[5] >= lo) & (tup[5] < hi) if tup[5].size else np.zeros(tup[0].shape, bool)
            pm = (tup[6] >= lo) & (tup[6] < hi) if tup[6].size else np.zeros(tup[2].shape, bool)
            dm = (tup[8] >= lo) & (tup[8] < hi) if tup[8].size else np.zeros(tup[7].shape, bool)
            return (tup[0][hm], tup[1][hm], tup[2][pm], tup[3][pm], tup[4],
                    tup[5][hm] if tup[5].size else tup[5], tup[6][pm] if tup[6].size else tup[6],
                    tup[7][dm], tup[8][dm] if tup[8].size else tup[8])
        return sub

    for scheme in args.norm:
        emit_plots(scheme, lambda t: t)                     # full sample (all pixels)
        for tag, lo, hi, note in thr_bins:                  # per-pixel-threshold bins (if --thr-bins)
            emit_plots(scheme, bin_sub(lo, hi), tag=f"thrbin_{tag}", tnote=f"  [{note}]")


if __name__ == "__main__":
    main()
