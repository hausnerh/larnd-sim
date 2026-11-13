#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_calib.py -- turn the Bi-207 pixel-hitmap into CALIBRATION-TIME answers:
#   Q1  how many hits does it take to calibrate one pixel?        -> from the K-570 line width
#   Q2  how long must a source sit in the detector to calibrate   -> coverage(emitted-decays) curve,
#       EVERY pixel (1 day? 1 week?)                                  with a time axis for a chosen activity
#   Q3  do 2-3 sources (or a 2D arrangement) cut that time?       -> superpose the illumination kernel
#
# GEOMETRY (fsd_cube, thesis convention -- see memory fsdcube-geometry):
#   Y = anode VERTICAL, LONG  = 47.6 cm, 128 px  (= larnd-sim x, TPC_BORDERS[0][0])
#   Z = anode HORIZONTAL, short = 29.76 cm, 80 px (= larnd-sim y, TPC_BORDERS[0][1])
#   X = DRIFT (= larnd-sim z), perpendicular to the anode. The detector is TALLER in Y than wide in Z.
#   So spreading sources along Y (the long axis) is the high-value move; we also test a 2D grid.
#
# WHAT COUNTS AS A CALIBRATION HIT (the key correction):
#   Only hits in the Bi-207 LINE window [--line-lo, --line-hi] ke- calibrate a pixel -- the K-570
#   conversion electron (8.06/9.53 ke-) and the full-energy 570 keV absorption line. The Bi-207 source
#   ALSO sprays a Compton CONTINUUM across the plane (the ~40% of hits that form the halo beyond the
#   one pixel under the source); those are the WRONG energy and cannot set a pixel's gain, so they must
#   NOT count toward "calibrated". We therefore build the calibration rate from the line-window hits
#   (Hc), and plot the all-hits profile alongside so the halo's non-line nature is visible. (The window
#   still admits some Compton-in-window background -- a per-pixel peak-over-continuum fit would refine
#   it -- so the line-window coverage is an optimistic bound on the usable rate.)
#
# MODEL:
#   Each pixel p has a calibration rate r_p = Hc[p]/N_generated [line-hits per EMITTED decay]. In time T
#   at source activity A [Bq] it collects n_p = r_p*A*T line-hits, reaching the target N_target at
#   T_p = N_target/(r_p*A). Everything scales with the emitted-decay count D = A*T, so we draw coverage
#   vs D and mark times for a reference activity. Dark-corner rates below the sim's 1/N_gen resolution
#   are filled from the azimuthal radial profile r(d) (flagged -- a longer sim would measure them).
#   Multi-source = superpose the same radial kernel at each source position.
#
# SOURCE: Eckert & Ziegler Bi-207 thin-window conversion-ELECTRON source (catalogue sec 3, electron
#   emitters): 37 kBq nominal, 5 mm active deposit in a 25 mm Al frame, Mylar 0.9 mg/cm2 or Ti
#   2.4 mg/cm2 window -> the K-570/K-1064 conversion electrons ESCAPE (window dE < 1%), so the sharp
#   local charge lines our BARE-electron sim produces are physical. Codes BIRB1083 (Mylar), BIRB1082
#   (Ti), BIRB4391 (Mylar, activity-calibrated). Default reference 37 kBq.
#   Caveats vs sim: the 5 mm deposit spreads the conversion line over ~1-2 pixels (we model a point);
#   a wall-mounted foil emits ~2pi not 4pi. NB pileup: at 37 kBq the pixel under the source sees
#   ~4.5 kHz (~1 hit/drift), so a strong source over-illuminates the core -- an argument for multi-source.
#
# Reuses the decode/fit in check_bi207_larnd.py (same directory) -- no logic is duplicated.
#
# Usage (NERSC, larnd env, login node -- CPU, seconds):
#   python tests/bi207/check_bi207_calib.py $ND_WORK/bi207_fsd.LARND.h5 \
#       --edep $ND_WORK/bi207.EDEPSIM.h5  --outdir $ND_WORK/bi207_calib
#   # --edep gives N_generated (unique primary events). If you know it, pass --n-generated N instead.
#   # --activity 37000 reference source strength [Bq] for the time marks (GF-007 range 185..3.7e6).
#   # --line-lo/--line-hi  the calibration line window [ke-] (default 5.5..12.4 = the K-570 FWHM region).
# =============================================================================
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_bi207_larnd as bq   # noqa: E402  -- reuse the decode + Gaussian fit

SECONDS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0, "week": 604800.0,
           "month": 2.6298e6, "year": 3.1557e7}


def count_generated(edep):
    """Unique PRIMARY-event count = number of thrown Bi-207 decays. Every thrown decay records a
    primary trajectory, so unique event_id over 'trajectories' is the emitted total. 'segments' only
    exist for depositing decays, so it is a fallback that UNDERCOUNTS. Returns (N, field, dataset)."""
    with bq.open_ro(edep) as h:
        for key in ("trajectories", "segments"):
            if key not in h:
                continue
            names = getattr(h[key].dtype, "names", None) or ()
            for f in ("event_id", "eventID", "event_ids", "eventId"):
                if f in names:
                    ev = np.asarray(h[key][f][:])
                    ev = ev.reshape(ev.shape[0], -1)[:, 0] if ev.ndim > 1 else ev
                    return int(np.unique(ev).size), f, key
    return None, None, None


def pixel_coordinate_grids(detector, nx, ny):
    """Yc[nx], Zc[ny] pixel-center coordinates from detsim.get_pixel_coordinates, in thesis axes:
    Yc = larnd-x = anode VERTICAL/long (128); Zc = larnd-y = anode HORIZONTAL/short (80)."""
    from larndsim.pixels_from_track import pixel2id
    from larndsim.detsim import get_pixel_coordinates
    Yc = np.array([get_pixel_coordinates(pixel2id(i, 0, 0))[0] for i in range(nx)], float)  # larnd-x -> Y
    Zc = np.array([get_pixel_coordinates(pixel2id(0, j, 0))[1] for j in range(ny)], float)  # larnd-y -> Z
    return Yc, Zc


def radial_model(R, Yc, Zc, sY, sZ, pitch, n_ann=28):
    """Azimuthally-averaged rate r(d) [hits/decay] about source (sY,sZ), INCLUDING the un-hit pixels in
    each annulus (so it is the true areal rate at radius d). Fit an exponential A*exp(-d/lam) to the
    outer annuli for extrapolation past the measured support. R has shape (nY,nZ). Returns a callable
    r_model(d) and diagnostics."""
    DD = np.hypot(Yc[:, None] - sY, Zc[None, :] - sZ).ravel()
    RR = R.ravel()
    d_max = float(DD.max())
    edges = np.linspace(0.0, d_max + 1e-6, n_ann + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_ann, np.nan)
    for k in range(n_ann):
        m = (DD >= edges[k]) & (DD < edges[k + 1])
        if m.any():
            mean[k] = RR[m].mean()
    good = np.isfinite(mean) & (cen > pitch)          # drop the d~0 core from the falloff fit
    pos = good & (mean > 0)
    if pos.sum() >= 3:
        d_fit, y_fit = cen[pos], np.log(mean[pos])
        A1 = np.polyfit(d_fit, y_fit, 1)              # log(mean) ~ intercept + slope*d
        lam = -1.0 / A1[0] if A1[0] < 0 else 1e9
        A0 = float(np.exp(A1[1]))
    else:
        lam, A0 = 1e9, float(np.nanmax(mean[good])) if good.any() else 1e-6
    core = float(RR[DD <= pitch].mean()) if (DD <= pitch).any() else float(np.nanmax(mean))
    cen_g, mean_g = cen[np.isfinite(mean)], mean[np.isfinite(mean)]

    def r_model(d):
        d = np.atleast_1d(np.asarray(d, float))
        out = np.where(d <= cen_g[0], core, np.interp(d, cen_g, mean_g, right=np.nan))
        ext = A0 * np.exp(-d / lam)                   # exponential tail past the measured support
        out = np.where(np.isnan(out), ext, out)
        out = np.maximum(out, ext)                    # never below the smooth tail (kills interp zeros)
        return np.maximum(out, 1e-12)
    return r_model, dict(cen=cen, mean=mean, A0=A0, lam=lam, core=core, d_max=d_max)


def main():
    ap = argparse.ArgumentParser(description="Bi-207 per-pixel calibration time + multi-source study.")
    ap.add_argument("larnd", help="larnd-sim FEE output (…_fsd.LARND.h5)")
    ap.add_argument("--edep", default=None, help="EDEPSIM (…_fsd.EDEPSIM.h5) to count EMITTED decays")
    ap.add_argument("--n-generated", type=int, default=None, help="emitted Bi-207 decays (overrides --edep)")
    ap.add_argument("--activity", type=float, default=37000.0, help="reference source activity [Bq] (E&Z Bi-207 e- source nominal 37 kBq)")
    ap.add_argument("--n-target", type=float, default=1000.0, help="line-hits/pixel that count as 'calibrated'")
    ap.add_argument("--line-lo", type=float, default=5.5, help="calibration line window low edge [ke-]")
    ap.add_argument("--line-hi", type=float, default=12.4, help="calibration line window high edge [ke-]")
    ap.add_argument("--outdir", default="bi207_calib")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    detector, _ = bq.load_fsd_cube_config()
    slope, intercept, dtype = bq.adc_linear(detector)

    # ---- decode (reuse base functions) ----
    with bq.open_ro(args.larnd) as h:
        p = np.asarray(h["packets"][:])
        dmask = p["packet_type"] == dtype
        adc = p["dataword"][dmask].astype(np.float64)
        q_ke = (adc + 0.5 - intercept) / (slope * 1.0e3)
        ev = bq.event_id_per_data_packet(h, dmask)
        pix = bq.hit_pixel_ids(h, dmask, detector)
    if pix is None:
        sys.exit("could not map packets to pixels")
    from larndsim.pixels_from_track import id2pixel
    pid, nx, ny = pix                                   # nx = Y (larnd-x, 128); ny = Z (larnd-y, 80)
    iY = np.full(pid.shape, -1, np.int64)
    iZ = np.full(pid.shape, -1, np.int64)
    for i in np.nonzero(pid >= 0)[0]:
        a, b, _ = id2pixel(int(pid[i]))
        iY[i], iZ[i] = int(a), int(b)
    valid = (iY >= 0) & (iY < nx) & (iZ >= 0) & (iZ < ny)
    inwin = (q_ke >= args.line_lo) & (q_ke <= args.line_hi)      # the calibration line window
    H = np.zeros((nx, ny), np.int64)                             # ALL hits (for the comparison profile)
    np.add.at(H, (iY[valid], iZ[valid]), 1)
    Hc = np.zeros((nx, ny), np.int64)                            # LINE hits (what calibrates)
    vln = valid & inwin
    np.add.at(Hc, (iY[vln], iZ[vln]), 1)
    npix = nx * ny
    n_lit, n_lit_c = int((H > 0).sum()), int((Hc > 0).sum())
    ndet = int(np.unique(ev[ev >= 0]).size) if ev is not None else -1

    # ---- emitted-decay count (needed to turn rate -> time) ----
    ngen, gfield, gset = (args.n_generated, "cli", "cli") if args.n_generated else (None, None, None)
    if ngen is None and args.edep:
        ngen, gfield, gset = count_generated(args.edep)
    lower_bound = False
    if ngen is None:
        ngen, lower_bound = max(ndet, 1), True
        print("!! WARNING: no --edep/--n-generated; using DETECTED decays as N_generated.")
        print("   Times are LOWER BOUNDS (coverage UPPER bounds). Pass --edep for the real number.")
    print(f"pixels: {nx}(Y)x{ny}(Z)={npix}; all-hit lit {n_lit} ({100*n_lit/npix:.1f}%); "
          f"line-hit lit {n_lit_c} ({100*n_lit_c/npix:.1f}%); hits {int(dmask.sum())} "
          f"({int(vln.sum())} in line window {args.line_lo:.1f}-{args.line_hi:.1f} ke-); "
          f"detected decays {ndet}; EMITTED decays {ngen} (from {gset}.{gfield})")

    # ---- source projection = brightest LINE pixel; rate grids ----
    si, sj = np.unravel_index(int(np.argmax(Hc)), Hc.shape)
    Yc, Zc = pixel_coordinate_grids(detector, nx, ny)
    sY, sZ = float(Yc[si]), float(Zc[sj])
    Y0, Y1, Z0, Z1 = float(Yc.min()), float(Yc.max()), float(Zc.min()), float(Zc.max())
    pitch = float(detector.PIXEL_PITCH)
    R_all = H.astype(float) / ngen
    R_cal = Hc.astype(float) / ngen                              # calibration (line-window) rate
    print(f"source projection (brightest line pixel): Y {sY:.1f} (long), Z {sZ:.1f} (short) cm  (Hc={int(Hc[si,sj])})")

    r_all, diag_all = radial_model(R_all, Yc, Zc, sY, sZ, pitch)
    r_cal, diag_cal = radial_model(R_cal, Yc, Zc, sY, sZ, pitch)
    print(f"line-rate radial falloff: core {diag_cal['core']:.3e} line-hits/decay, exp tail lambda "
          f"{diag_cal['lam']:.1f} cm (measured out to d={diag_cal['d_max']:.0f} cm)")

    # usable (line) fraction vs distance -- shows the halo is mostly wrong-energy
    DD = np.hypot(Yc[:, None] - sY, Zc[None, :] - sZ)
    frac_lines = []
    for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 12), (12, 30)]:
        m = (DD >= lo) & (DD < hi)
        a, c = int(H[m].sum()), int(Hc[m].sum())
        frac_lines.append(f"  d {lo:2d}-{hi:<2d} cm: {a:4d} hits, {c:4d} in line window "
                          f"({100*c/max(a,1):3.0f}% usable)")

    # ---- Q1: hits to calibrate one pixel, from the K-570 line ----
    edges = bq.adc_aligned_edges(min(0.0, float(q_ke.min())), 30.0, slope, intercept, 1)
    fit = bq.fit_gaussian_peak(q_ke, edges, 4.0, 14.0)
    q1_lines = []
    if fit is not None:
        mu, dmu, sig = fit["mu"], fit["dmu"], fit["sigma"]
        neff = (sig / dmu) ** 2 if dmu > 0 else float("nan")
        f_peak = float(((q_ke >= fit["win"][0]) & (q_ke <= fit["win"][1])).mean())
        q1_lines.append(f"K-570 line: mu {mu:.2f} ke-, sigma {sig:.2f} ke-, fit dmu {dmu:.3f} "
                        f"(N_eff {neff:.0f} peak hits); {100*f_peak:.0f}% of hits land in the peak window")
        for pct in (5.0, 3.0, 2.0, 1.0):
            delta = pct / 100.0 * mu
            n_peak = neff * (dmu / delta) ** 2
            n_tot = n_peak / max(f_peak, 1e-6)
            q1_lines.append(f"  gain to {pct:3.0f}%:  ~{n_peak:6.0f} peak hits  (~{n_tot:7.0f} total hits/pixel)")

    # ---- arrangements: source positions as (Y, Z) = (vertical-long, horizontal-short) ----
    cZ = 0.5 * (Z0 + Z1)
    arr = {
        "1 source (center)":        [(0.5 * (Y0 + Y1), cZ)],
        "2 sources (y=1/3,2/3)":    [(Y0 + (Y1 - Y0) / 3, cZ), (Y0 + 2 * (Y1 - Y0) / 3, cZ)],
        "3 sources (y row)":        [(Y0 + (Y1 - Y0) / 6, cZ), (0.5 * (Y0 + Y1), cZ),
                                     (Y0 + 5 * (Y1 - Y0) / 6, cZ)],
        "2x2 grid":                 [(Y0 + (Y1 - Y0) / 4, Z0 + (Z1 - Z0) / 4),
                                     (Y0 + 3 * (Y1 - Y0) / 4, Z0 + (Z1 - Z0) / 4),
                                     (Y0 + (Y1 - Y0) / 4, Z0 + 3 * (Z1 - Z0) / 4),
                                     (Y0 + 3 * (Y1 - Y0) / 4, Z0 + 3 * (Z1 - Z0) / 4)],
    }

    def rate_grid(sources):
        tot = np.zeros((nx, ny))
        for (sYi, sZi) in sources:
            d = np.hypot(Yc[:, None] - sYi, Zc[None, :] - sZi)
            tot += r_cal(d.ravel()).reshape(nx, ny)             # calibration (line) rate
        return tot

    rates = {k: rate_grid(v) for k, v in arr.items()}
    D_need = {k: (args.n_target / np.maximum(v.ravel(), 1e-12)) for k, v in rates.items()}

    A = args.activity
    day, week = SECONDS["day"] * A, SECONDS["week"] * A

    def coverage_at(k, D):
        return float(np.mean(D_need[k] <= D))

    # ---- report ----
    out = []
    out.append(f"Bi-207 per-pixel calibration study  (N_target {args.n_target:.0f} LINE-hits/pixel)")
    out.append(f"input {args.larnd}")
    out.append(f"geometry: Y vertical/long 47.6cm/{nx}px; Z horizontal/short 29.76cm/{ny}px; X drift")
    out.append(f"calibration line window: {args.line_lo:.1f}-{args.line_hi:.1f} ke- (K-570 8.06/9.53 + 570 full-energy)")
    out.append(f"source: E&Z Bi-207 thin-window e- source (BIRB4391, 37 kBq nominal); reference {A:.0f} Bq")
    out.append(f"plane {nx}x{ny}={npix} px; emitted decays N_gen={ngen}"
               + ("  [LOWER-BOUND: detected count used]" if lower_bound else "") + "\n")
    out.append("Which hits calibrate? (line window vs all hits, by distance from the source):")
    out.extend(frac_lines)
    out.append("  -> the halo beyond ~2 cm is mostly Compton continuum (wrong energy); only the line-")
    out.append("     window hits set a pixel's gain, so coverage is built from those.\n")
    out.append("Q1 -- hits to calibrate one pixel (from the K-570 line width):")
    out.extend(["  " + l for l in q1_lines])
    out.append("")
    out.append("Q2/Q3 -- coverage vs exposure (fraction of %d pixels reaching N_target LINE-hits):" % npix)
    out.append(f"  {'arrangement':24s} {'1 day':>8s} {'1 week':>8s}   {'D for 90%':>11s} {'D for 99%':>11s}   "
               f"t99 @37kBq /@370kBq")
    for k in arr:
        c_day, c_week = coverage_at(k, day), coverage_at(k, week)
        Ds = np.sort(D_need[k])
        d90 = Ds[min(int(0.90 * npix), npix - 1)]
        d99 = Ds[min(int(0.99 * npix), npix - 1)]
        t99_37 = d99 / 37000.0 / SECONDS["day"]      # days @ 37 kBq
        t99_370 = d99 / 370000.0 / SECONDS["day"]    # days @ 370 kBq
        out.append(f"  {k:24s} {100*c_day:7.1f}% {100*c_week:7.1f}%   {d90:11.2e} {d99:11.2e}   "
                   f"{t99_37:6.1f}d / {t99_370:5.2f}d")
    for ln in out:
        print(ln)

    # =========================== plots ===========================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # (1) coverage vs emitted decays (no title; time marks for the reference activity)
    D_axis = np.logspace(np.log10(max(1e3, args.n_target)), 12, 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for k in arr:
        ax.plot(D_axis, [100 * coverage_at(k, D) for D in D_axis], lw=2, label=k)
    for t, nm, ls in [(day, "1 day", "--"), (week, "1 week", ":")]:
        ax.axvline(t, color="0.45", ls=ls, lw=1)
        ax.text(t, 50, f" {nm} @ {A:.0f} Bq", rotation=90, va="center", ha="right", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("emitted decays  D = A x T   (time marks shown for the %g Bq reference)" % A)
    ax.set_ylabel(f"% of {npix} pixels with >= {args.n_target:.0f} line-hits")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="center right")
    for nm in ("hour", "day", "week", "month", "year"):
        t = SECONDS[nm] * A
        if D_axis[0] < t < D_axis[-1]:
            ax.annotate(nm, (t, 100), xytext=(t, 101.5), ha="center", fontsize=7, color="0.55",
                        annotation_clip=False)
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "bi207_coverage_vs_exposure.png")
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    print(f"wrote {f1}")

    # (2) time-to-calibrate maps: 1 source vs 2 sources (days @ reference activity); Y vertical/long
    for k, tag in (("1 source (center)", "1src"), ("2 sources (y=1/3,2/3)", "2src")):
        Tp = (D_need[k].reshape(nx, ny)) / (A * SECONDS["day"])
        fig, ax = plt.subplots(figsize=(4.8, 7.2))
        im = ax.imshow(Tp, origin="lower", aspect="equal", extent=[Z0, Z1, Y0, Y1],
                       cmap="viridis_r", norm=LogNorm(vmin=max(1e-3, Tp.min()), vmax=Tp.max()))
        ax.set_title(f"Days to {args.n_target:.0f} line-hits/pixel @ {A:.0f} Bq\n{k}", fontsize=10)
        ax.set_xlabel(f"Z (cm)  [{ny} px, short, horizontal]")
        ax.set_ylabel(f"Y (cm)  [{nx} px, long, vertical]")
        fig.colorbar(im, ax=ax).set_label("days to calibrate (log)")
        for (sYi, sZi) in arr[k]:
            ax.plot(sZi, sYi, marker="*", ms=16, color="crimson", mec="white", mew=0.8)
        fig.tight_layout()
        fn = os.path.join(args.outdir, f"bi207_calibtime_{tag}.png")
        fig.savefig(fn, dpi=140)
        plt.close(fig)
        print(f"wrote {fn}")

    # (3) radial profile: ALL hits vs LINE-window hits + the line model (shows the halo is non-line)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(diag_all["cen"], diag_all["mean"], "o", mfc="none", color="#888888", label="all hits (incl. Compton halo)")
    ax.plot(diag_cal["cen"], diag_cal["mean"], "o", color="#4477aa", label="line-window hits (calibration)")
    dd = np.linspace(0, diag_cal["d_max"] * 1.4, 300)
    ax.plot(dd, r_cal(dd), "-", color="crimson", lw=2, label="line-rate model (interp + exp tail)")
    ax.axhline(1.0 / ngen, color="0.5", ls="--", lw=1, label="1 hit / full sim (resolution floor)")
    ax.set_yscale("log")
    ax.set_xlabel("distance from source projection d [cm]")
    ax.set_ylabel("rate r(d) [hits / emitted decay]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f3 = os.path.join(args.outdir, "bi207_radial_profile.png")
    fig.savefig(f3, dpi=140)
    plt.close(fig)
    print(f"wrote {f3}")

    # ---- text summary + npz dump ----
    txt = os.path.join(args.outdir, "bi207_calib.txt")
    with open(txt, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {txt}")
    npz = os.path.join(args.outdir, "bi207_calib.npz")
    np.savez_compressed(npz, H_all=H, H_line=Hc, Yc=Yc, Zc=Zc, ngen=ngen, source=(sY, sZ),
                        n_target=args.n_target, activity=A, line_window=(args.line_lo, args.line_hi),
                        radial_cen=diag_cal["cen"], radial_mean_line=diag_cal["mean"],
                        radial_mean_all=diag_all["mean"], radial_A0=diag_cal["A0"],
                        radial_lam=diag_cal["lam"], radial_core=diag_cal["core"])
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
