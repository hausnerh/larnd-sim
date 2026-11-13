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
# MODEL (all computed, nothing hand-waved):
#   Each pixel p has an illumination rate r_p = H[p]/N_generated  [hits per EMITTED decay]. In wall-clock
#   time T at source activity A [Bq], it collects  n_p = r_p * A * T  hits, so it reaches the calibration
#   target N_target at  T_p = N_target / (r_p * A).  Because everything scales with the product A*T, the
#   honest independent variable is the number of EMITTED decays D = A*T; we draw the curve vs D and add a
#   second axis labelled in time for a reference activity (default 1 kBq).
#
#   The lit pixels give r_p directly. The dark corners have r_p below what a finite sim can resolve
#   (< 1/N_generated), so we fit the AZIMUTHAL radial profile r(d) (d = distance from the source
#   projection) and extrapolate -- flagged as a MODEL, since a 5000-decay sim cannot measure r < 2e-4.
#   A longer sim (more decays) would replace the extrapolation with a direct measurement of the tail.
#
#   Multi-source = superpose the SAME radial kernel at each source position and recompute coverage.
#
# Reuses the decode/fit in check_bi207_larnd.py (same directory) -- no logic is duplicated.
#
# Usage (NERSC, larnd env, login node -- CPU, seconds):
#   python tests/bi207/check_bi207_calib.py $ND_WORK/bi207_fsd.LARND.h5 \
#       --edep $ND_WORK/bi207.EDEPSIM.h5  --outdir $ND_WORK/bi207_calib
#   # --edep gives N_generated (unique primary events). If you know it, pass --n-generated N instead.
#   # --activity 1000  sets the reference source strength [Bq] for the time axis (default 1000 = 1 kBq).
#   # --n-target 1000  hits/pixel considered "calibrated" (default 1000; see the Q1 block for what % that is).
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
    each annulus (so it is the true areal rate at radius d, not just over lit pixels). Fit an exponential
    A*exp(-d/lam) to the outer annuli for extrapolation past the measured support. R has shape (nY,nZ).
    Returns a callable r_model(d) and diagnostics."""
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
    ap.add_argument("--activity", type=float, default=1000.0, help="reference source activity [Bq] for the time axis")
    ap.add_argument("--n-target", type=float, default=1000.0, help="hits/pixel that count as 'calibrated'")
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
    H = np.zeros((nx, ny), np.int64)                    # H[iY, iZ]
    np.add.at(H, (iY[valid], iZ[valid]), 1)
    npix = nx * ny
    n_lit = int((H > 0).sum())
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
    print(f"pixels: {nx}(Y)x{ny}(Z)={npix}; lit {n_lit} ({100*n_lit/npix:.1f}%); hits {int(dmask.sum())}; "
          f"detected decays {ndet}; EMITTED decays {ngen} (from {gset}.{gfield})")

    # ---- source projection = brightest pixel; rate grid ----
    si, sj = np.unravel_index(int(np.argmax(H)), H.shape)
    Yc, Zc = pixel_coordinate_grids(detector, nx, ny)
    sY, sZ = float(Yc[si]), float(Zc[sj])
    Y0, Y1, Z0, Z1 = float(Yc.min()), float(Yc.max()), float(Zc.min()), float(Zc.max())
    pitch = float(detector.PIXEL_PITCH)
    R = H.astype(float) / ngen                                   # measured hits/decay per pixel
    print(f"source projection (brightest pixel): Y {sY:.1f} (long), Z {sZ:.1f} (short) cm  (H={int(H[si,sj])})")

    r_model, diag = radial_model(R, Yc, Zc, sY, sZ, pitch)
    print(f"radial falloff: core {diag['core']:.3e} hits/decay, exp tail lambda {diag['lam']:.1f} cm "
          f"(measured out to d={diag['d_max']:.0f} cm)")

    # ---- Q1: hits to calibrate one pixel, from the K-570 line ----
    edges = bq.adc_aligned_edges(min(0.0, float(q_ke.min())), 30.0, slope, intercept, 1)
    fit = bq.fit_gaussian_peak(q_ke, edges, 4.0, 14.0)
    q1_lines = []
    if fit is not None:
        mu, dmu, sig = fit["mu"], fit["dmu"], fit["sigma"]
        neff = (sig / dmu) ** 2 if dmu > 0 else float("nan")    # effective counts behind this fit
        inwin = (q_ke >= fit["win"][0]) & (q_ke <= fit["win"][1])
        f_peak = float(inwin.mean())                            # fraction of hits in the peak window
        q1_lines.append(f"K-570 line: mu {mu:.2f} ke-, sigma {sig:.2f} ke-, fit dmu {dmu:.3f} "
                        f"(N_eff {neff:.0f} peak hits); {100*f_peak:.0f}% of hits land in the peak window")
        for pct in (5.0, 3.0, 2.0, 1.0):
            delta = pct / 100.0 * mu
            n_peak = neff * (dmu / delta) ** 2                  # peak hits for gain precision = pct%
            n_tot = n_peak / max(f_peak, 1e-6)                 # total hits on the pixel
            q1_lines.append(f"  gain to {pct:3.0f}%:  ~{n_peak:6.0f} peak hits  (~{n_tot:7.0f} total hits/pixel)")
    for ln in q1_lines:
        print(ln)

    # ---- arrangements: source positions as (Y, Z) = (vertical-long, horizontal-short) ----
    cZ = 0.5 * (Z0 + Z1)                                        # center of the SHORT axis
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
        """Per-pixel total rate [hits/decay] = sum over sources of the radial kernel (per source)."""
        tot = np.zeros((nx, ny))
        for (sYi, sZi) in sources:
            d = np.hypot(Yc[:, None] - sYi, Zc[None, :] - sZi)
            tot += r_model(d.ravel()).reshape(nx, ny)
        return tot

    rates = {k: rate_grid(v) for k, v in arr.items()}
    D_need = {k: (args.n_target / np.maximum(v.ravel(), 1e-12)) for k, v in rates.items()}

    A = args.activity
    day = SECONDS["day"] * A
    week = SECONDS["week"] * A

    def coverage_at(k, D):
        return float(np.mean(D_need[k] <= D))

    # ---- report Q2/Q3 ----
    out_txt = []
    out_txt.append(f"Bi-207 per-pixel calibration study  (N_target {args.n_target:.0f} hits/pixel, "
                   f"activity {A:.0f} Bq reference)")
    out_txt.append(f"input {args.larnd}")
    out_txt.append(f"geometry: Y vertical/long 47.6cm/{nx}px; Z horizontal/short 29.76cm/{ny}px; X drift")
    out_txt.append(f"plane {nx}x{ny}={npix} pixels; emitted decays N_gen={ngen}"
                   + ("  [LOWER-BOUND: detected count used]" if lower_bound else "") + "\n")
    out_txt.extend(["Q1 -- hits to calibrate one pixel (from the K-570 line width):"] + ["  " + l for l in q1_lines])
    out_txt.append("")
    out_txt.append("Q2/Q3 -- coverage vs exposure (fraction of the %d pixels reaching N_target):" % npix)
    out_txt.append(f"  {'arrangement':24s} {'1 day':>9s} {'1 week':>9s}   {'D for 90%':>12s} {'D for 99%':>12s}   t@1kBq(90/99%)")
    for k in arr:
        c_day, c_week = coverage_at(k, day), coverage_at(k, week)
        Ds = np.sort(D_need[k])
        d90 = Ds[min(int(0.90 * npix), npix - 1)]
        d99 = Ds[min(int(0.99 * npix), npix - 1)]
        t90 = d90 / (1000.0 * SECONDS["day"])    # days @ 1 kBq
        t99 = d99 / (1000.0 * SECONDS["day"])
        out_txt.append(f"  {k:24s} {100*c_day:7.1f}% {100*c_week:7.1f}%   {d90:12.2e} {d99:12.2e}   "
                       f"{t90:6.1f}d / {t99:6.1f}d")
    for ln in out_txt:
        print(ln)

    # =========================== plots ===========================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # (1) coverage vs emitted decays, with a time axis for the reference activity
    D_axis = np.logspace(np.log10(max(1e3, args.n_target)), 11, 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for k in arr:
        cov = np.array([coverage_at(k, D) for D in D_axis])
        ax.plot(D_axis, 100 * cov, lw=2, label=k)
    ax.axvline(day, color="0.4", ls="--", lw=1)
    ax.text(day, 5, f" 1 day @ {A:.0f} Bq", rotation=90, va="bottom", ha="right", fontsize=8, color="0.3")
    ax.axvline(week, color="0.4", ls=":", lw=1)
    ax.text(week, 5, f" 1 week @ {A:.0f} Bq", rotation=90, va="bottom", ha="right", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("emitted decays  D = A x T")
    ax.set_ylabel(f"% of {npix} pixels with >= {args.n_target:.0f} hits")
    ax.set_ylim(0, 101)
    ax.set_title(f"Bi-207 pixel-plane calibration coverage vs exposure\n"
                 f"(N_target {args.n_target:.0f} hits/pixel; dark-corner tail is a radial-model extrapolation)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    for nm in ("hour", "day", "week", "month", "year"):
        t = SECONDS[nm] * A
        if D_axis[0] < t < D_axis[-1]:
            ax.annotate(nm, (t, 100), xytext=(t, 103), ha="center", fontsize=7, color="0.5",
                        annotation_clip=False)
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "bi207_coverage_vs_exposure.png")
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    print(f"wrote {f1}")

    # (2) time-to-calibrate maps: 1 source vs 2 sources (days @ reference activity); Y vertical/long
    for k, tag in (("1 source (center)", "1src"), ("2 sources (y=1/3,2/3)", "2src")):
        Tp = (D_need[k].reshape(nx, ny)) / (A * SECONDS["day"])   # days @ reference activity, Tp[iY,iZ]
        fig, ax = plt.subplots(figsize=(4.8, 7.2))
        im = ax.imshow(Tp, origin="lower", aspect="equal", extent=[Z0, Z1, Y0, Y1],
                       cmap="viridis_r", norm=LogNorm(vmin=max(1e-3, Tp.min()), vmax=Tp.max()))
        ax.set_title(f"Days to reach {args.n_target:.0f} hits/pixel @ {A:.0f} Bq\n{k}", fontsize=10)
        ax.set_xlabel(f"Z (cm)  [{ny} px, short, horizontal]")
        ax.set_ylabel(f"Y (cm)  [{nx} px, long, vertical]")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("days to calibrate (log)")
        for (sYi, sZi) in arr[k]:
            ax.plot(sZi, sYi, marker="*", ms=16, color="crimson", mec="white", mew=0.8)
        fig.tight_layout()
        fn = os.path.join(args.outdir, f"bi207_calibtime_{tag}.png")
        fig.savefig(fn, dpi=140)
        plt.close(fig)
        print(f"wrote {fn}")

    # (3) radial profile: measured annuli + the model (validation of the extrapolation)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(diag["cen"], diag["mean"], "o", color="#4477aa", label="measured annulus mean")
    dd = np.linspace(0, diag["d_max"] * 1.4, 300)
    ax.plot(dd, r_model(dd), "-", color="crimson", lw=2, label="radial model (interp + exp tail)")
    ax.axhline(1.0 / ngen, color="0.5", ls="--", lw=1, label="1 hit / full sim (resolution floor)")
    ax.set_yscale("log")
    ax.set_xlabel("distance from source projection d [cm]")
    ax.set_ylabel("illumination rate r(d) [hits / emitted decay]")
    ax.set_title("Bi-207 illumination radial profile (sets the dark-corner extrapolation)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f3 = os.path.join(args.outdir, "bi207_radial_profile.png")
    fig.savefig(f3, dpi=140)
    plt.close(fig)
    print(f"wrote {f3}")

    # ---- text summary + npz dump (small; lets the model be re-run locally) ----
    txt = os.path.join(args.outdir, "bi207_calib.txt")
    with open(txt, "w") as f:
        f.write("\n".join(out_txt) + "\n")
    print(f"wrote {txt}")
    npz = os.path.join(args.outdir, "bi207_calib.npz")
    np.savez_compressed(npz, H=H, Yc=Yc, Zc=Zc, ngen=ngen, source=(sY, sZ),
                        n_target=args.n_target, activity=A,
                        radial_cen=diag["cen"], radial_mean=diag["mean"],
                        radial_A0=diag["A0"], radial_lam=diag["lam"], radial_core=diag["core"])
    print(f"wrote {npz}")


if __name__ == "__main__":
    main()
