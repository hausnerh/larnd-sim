#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_larnd.py -- read the FEE output DIRECTLY from the larnd-sim LARND file and
# look for the Bi-207 K-570 calibration line + pixel coverage, bypassing ndlar_flow.
#
# WHY: the LARND file IS the front-end-electronics response. Each LArPix-v3 DATA packet
# (packet_type == detector.PACKET_TYPE == 1 for fsd_cube) carries a digitized ADC in 'dataword'.
# larnd-sim's fee.digitize() maps collected charge Q[e-] -> ADC via, after the mV/e units cancel and
# with V_CM handled:  ADC = floor( intercept + slope*Q[e-] ), a line. We invert it (+0.5 for the floor
# midpoint) to recover per-pixel collected charge. For fsd_cube ADC ~= 289 + Q[ke-]. Constants come
# from the loaded fsd_cube config -- nothing hard-coded.
#
# Writes (each its own file):
#   bi207_perhit_q.png     per-PIXEL-HIT charge (one entry per data packet) + a Gaussian fit to K-570.
#   bi207_perdecay_q.png   per-DECAY charge, summed across the pixels one decay lit + a K-570 fit.
#   bi207_multiplicity.png pixel-hits per decay.
#   bi207_pixel_hitmap.png the ANODE pixel plane (128 cols x 80 rows), LOG-colored by hits/pixel, i.e.
#                          which pixels this source placement illuminates (drift is perpendicular to
#                          this plane -- the pixel position comes from the hit's chip/channel, not from
#                          any reco x/y/z). This is the map you need to decide per-pixel calibration.
# plus bi207_larnd.txt. Charge histograms bin on the native ADC grid (one bin per ADC count) so there
# is no bin/quantization aliasing.
#
# Usage (NERSC, larnd conda env; login node is fine -- tiny, CPU-only, no container/GPU):
#   python tests/bi207/check_bi207_larnd.py $ND_WORK/bi207_fsd.LARND.h5 --outdir $ND_WORK/bi207_larnd
# =============================================================================
import argparse
import os
import sys

import numpy as np

# larnd-sim Box-recombination + drift-attenuation model (physics.py constants), so we can turn a
# deposited energy into the collected charge the FEE should see, consistently for any line.
BOX_ALPHA = 0.93
BOX_BETA = 0.207
W_ION = 23.6e-6      # MeV per ionization electron
LAR_RHO = 1.38       # g/cm^3
E_FIELD = 0.25       # kV/cm (FSDCube half-nominal)
DEDX = 2.5           # MeV/cm, representative for these ~0.5-1 MeV electrons
DRIFT_ATTEN = 0.667  # exp(-t_drift/tau) for ~full drift (47.75 cm, tau 1100 us) from a cathode source

# Bi-207 K-570 handle. The nuclear transition is 569.7 keV; its K-shell internal-conversion electron
# carries KE = 569.7 - 88(Pb K binding) = 481.7 keV. If the Pb K X-ray (~88 keV) is reabsorbed locally
# (it usually is, over ~mm in LAr) the pixel sees the FULL 569.7 keV; if it escapes, only 481.7 keV.
K570_E_BARE = 0.4817   # MeV, conversion electron alone
K570_E_FULL = 0.5697   # MeV, electron + locally-absorbed Pb K X-ray (the full transition)


def edep_to_ke(e_mev):
    """Deposited energy [MeV] -> collected charge [ke-] via larnd-sim's Box recombination + drift
    attenuation (the same model that sets the reference lines). csi, recomb per physics.box_model."""
    csi = BOX_BETA * DEDX / (E_FIELD * LAR_RHO)
    recomb = np.log(BOX_ALPHA + csi) / csi
    n_e = recomb * e_mev / W_ION
    return n_e * DRIFT_ATTEN / 1.0e3


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def load_fsd_cube_config():
    """Load the fsd_cube config; return (detector consts module, pixel-layout yaml path)."""
    from larndsim import consts
    from larndsim.config import get_config
    from larndsim.consts import detector
    cfg = get_config("fsd_cube")

    def one(k):
        v = cfg.get(k)
        return v[0] if isinstance(v, list) else v

    consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"),
                           one("RESPONSE"), one("SIM_PROPERTIES"))
    return detector, one("PIXEL_LAYOUT")


def adc_linear(detector):
    """ADC = intercept + slope*Q[e-]  (mV cancels; e == 1). Returns (slope, intercept, data_pkt_type)."""
    denom = (detector.V_REF - detector.V_CM) * detector.ADC_SCALE_FACTOR
    slope = detector.GAIN * detector.ADC_COUNTS / denom
    intercept = (detector.V_PEDESTAL - detector.V_CM) * detector.ADC_COUNTS / denom + detector.ADC_OFFSET
    return float(slope), float(intercept), int(detector.PACKET_TYPE)


def adc_aligned_edges(qlo_ke, qhi_ke, slope, intercept, nadc=1):
    """Bin edges aligned to the ADC grid: each bin spans exactly `nadc` ADC counts, so a single-hit
    charge (quantized to one ADC count) sits bin-centered and there is NO aliasing beat with the
    ~1 ke-/count quantization. Edges in ke-."""
    s = slope * 1.0e3                      # ADC counts per ke-
    a_lo = int(np.floor(intercept + s * qlo_ke))
    a_hi = int(np.ceil(intercept + s * qhi_ke))
    if a_hi <= a_lo:
        a_hi = a_lo + 1
    step = max(1, int(nadc))
    a = np.arange(a_lo, a_hi + step, step)
    return (a - intercept) / s


def fit_gaussian_peak(values, edges, win_lo, win_hi):
    """Fit a Gaussian to the peak inside [win_lo, win_hi], using only the bins where the peak rises
    ABOVE its own half-maximum (the FWHM region), with Poisson (sqrt N) errors. Returns a dict with
    mu, dmu (uncertainty on the mean), sigma, fwhm, amplitude, and the fit window -- or None."""
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        print(f"  (scipy unavailable: {exc!r} -> skipping the Gaussian fit)")
        return None
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    inwin = (centers >= win_lo) & (centers <= win_hi)
    if inwin.sum() < 4:
        return None
    cc = counts[inwin].astype(float)
    xx = centers[inwin]
    ipk = int(np.argmax(cc))
    a0, x0 = cc[ipk], xx[ipk]
    half = a0 / 2.0
    above = cc >= half
    lo = ipk
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = ipk
    while hi < len(cc) - 1 and above[hi + 1]:
        hi += 1
    # widen by one bin each side if the FWHM core is too thin to constrain 3 parameters
    lo = max(0, lo - (1 if hi - lo < 3 else 0))
    hi = min(len(cc) - 1, hi + (1 if hi - lo < 3 else 0))
    xf, yf = xx[lo:hi + 1], cc[lo:hi + 1]
    if xf.size < 4:
        return None
    fwhm0 = max(xf[-1] - xf[0], edges[1] - edges[0])
    yerr = np.sqrt(np.maximum(yf, 1.0))

    def gauss(x, amp, mu, sig):
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    try:
        popt, pcov = curve_fit(gauss, xf, yf, p0=[a0, x0, fwhm0 / 2.355],
                               sigma=yerr, absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        print(f"  (Gaussian fit did not converge: {exc!r})")
        return None
    amp, mu, sig = popt
    dmu = float(np.sqrt(abs(pcov[1][1])))
    return dict(amp=float(amp), mu=float(mu), dmu=dmu, sigma=float(abs(sig)),
                fwhm=float(2.3548 * abs(sig)), win=(float(xf[0]), float(xf[-1])))


def event_id_per_data_packet(h, data_mask):
    """Per-packet truth decay id straight from mc_packets_assn['event_ids']. Returns int64 aligned to
    packets[data_mask], or None."""
    try:
        if "mc_packets_assn" not in h:
            print("  (no mc_packets_assn in the file -> per-decay grouping unavailable)")
            return None
        assn = h["mc_packets_assn"]
        names = getattr(assn.dtype, "names", None) or ()
        field = next((f for f in ("event_ids", "event_id") if f in names), None)
        if field is None:
            print(f"  (mc_packets_assn has no event-id field; fields={names} -> grouping unavailable)")
            return None
        ev = np.asarray(assn[field][:])
        ev = ev.reshape(ev.shape[0], -1)[:, 0].astype(np.int64)
        if ev.shape[0] != data_mask.shape[0]:
            print(f"  (mc_packets_assn len {ev.shape[0]} != packets {data_mask.shape[0]})")
            return None
        return ev[data_mask]
    except Exception as exc:
        print(f"  (per-decay grouping via mc_packets_assn unavailable: {exc!r})")
        return None


def build_pixel_map(detector):
    """Authoritative (chip, channel) -> PHYSICAL pixel map, mirroring fee.py's forward assignment
    EXACTLY so the hitmap position matches what simulate_pixels wrote.

    fee.py (line ~248) sets each packet's (chip_id, channel_id) =
        PIXEL_CONNECTION_DICT[ rotate_tile((pix_x % NPT[0], pix_y % NPT[1]), tile_id) ],
    with tile_id = TILE_MAP[anode][pix_x//NPT[0]][pix_y//NPT[1]] and rotate_tile flipping pix_x/pix_y
    by the tile orientation. So the raw chip_channel_to_position index is the POST-rotate_tile value --
    inverting it naively misplaces hits (a centered source landed at col 125 instead of 64). We invert
    correctly by iterating every physical pixel, computing its (chip, channel) the same way, and keying
    on that. Physical x = TPC_BORDERS[plane][0][0] + pix_x*pitch (pixels_from_track: pix_x = pitches
    from the border). Returns (ccmap, nx, ny, pitch, (x0,x1), (y0,y1)) or None.
    """
    try:
        from larndsim.fee import rotate_tile
        nx, ny = int(detector.N_PIXELS[0]), int(detector.N_PIXELS[1])
        npt0, npt1 = int(detector.N_PIXELS_PER_TILE[0]), int(detector.N_PIXELS_PER_TILE[1])
        pcd = detector.PIXEL_CONNECTION_DICT
        tmap = detector.TILE_MAP
        B = np.asarray(detector.TPC_BORDERS)[0]
        x0, x1 = float(min(B[0])), float(max(B[0]))      # larnd-x (anode horizontal == thesis Z)
        y0, y1 = float(min(B[1])), float(max(B[1]))      # larnd-y (anode vertical  == thesis Y)
        pitch = float(detector.PIXEL_PITCH)
        ccmap = {}
        for px in range(nx):
            tx = px // npt0
            for py in range(ny):
                ty = py // npt1
                try:
                    tile_id = tmap[0][tx][ty]            # single anode plane -> index 0
                    cc = pcd.get(tuple(rotate_tile((px % npt0, py % npt1), tile_id)))
                except Exception:
                    cc = None
                if cc is not None:
                    ccmap[(int(cc[0]), int(cc[1]))] = (px, py)
        if not ccmap:
            print("  (built an empty chip/channel->pixel map -> skipping heatmap)")
            return None
        return ccmap, nx, ny, pitch, (x0, x1), (y0, y1)
    except Exception as exc:
        print(f"  (authoritative pixel map unavailable: {exc!r} -> skipping the hit heatmap)")
        return None


def packet_chip_channel(h, data_mask):
    """(chip, channel) int64 arrays for the data packets, or None if the fields are absent."""
    p = np.asarray(h["packets"][:])
    names = p.dtype.names or ()
    chip_f = next((f for f in ("chip_id", "chipid", "chip") if f in names), None)
    chan_f = next((f for f in ("channel_id", "channelid", "channel") if f in names), None)
    if chip_f is None or chan_f is None:
        print(f"  (packets lack chip/channel fields; have {names} -> skipping the hit heatmap)")
        return None
    return (p[chip_f][data_mask].astype(np.int64), p[chan_f][data_mask].astype(np.int64))


def main():
    ap = argparse.ArgumentParser(description="Decode the larnd-sim FEE (LARND) output to charge, fit "
                                             "the Bi-207 K-570 line, and map pixel coverage. No ndlar_flow.")
    ap.add_argument("larnd", help="larnd-sim output .h5 (…_fsd.LARND.h5)")
    ap.add_argument("--outdir", default="bi207_larnd", help="output directory for the pngs + txt")
    ap.add_argument("--qmax", type=float, default=30.0, help="charge-axis max [ke-]")
    ap.add_argument("--nadc", type=int, default=1, help="ADC counts per charge-histogram bin (>=1)")
    ap.add_argument("--fit-lo", type=float, default=4.0, help="low edge of the K-570 fit search window [ke-]")
    ap.add_argument("--fit-hi", type=float, default=14.0, help="high edge of the K-570 fit search window [ke-]")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    detector, pixel_layout_path = load_fsd_cube_config()
    slope, intercept, data_type = adc_linear(detector)
    print(f"fsd_cube FEE: ADC = {intercept:.3f} + {slope*1e3:.4f} * Q[ke-]   "
          f"(pedestal ADC {intercept:.1f}; ~{1.0/(slope*1e3):.2f} ke- per ADC count; data packet_type={data_type})")

    q_bare = edep_to_ke(K570_E_BARE)       # 481.7 keV conversion electron alone
    q_full = edep_to_ke(K570_E_FULL)       # + locally-absorbed Pb K X-ray (full 569.7 keV transition)
    print(f"K-570 predicted collected charge: {q_bare:.2f} ke- (bare 482 keV e-), "
          f"{q_full:.2f} ke- (570 keV, +Pb K X-ray)")

    pmap = build_pixel_map(detector)

    with open_ro(args.larnd) as h:
        if "packets" not in h:
            sys.exit("no 'packets' dataset in the LARND file")
        p = np.asarray(h["packets"][:])
        pt = p["packet_type"]
        dmask = pt == data_type
        ndata = int(dmask.sum())
        print(f"packets: {p.size}  data (type {data_type}): {ndata}")
        if ndata == 0:
            sys.exit("no data packets -- nothing to decode")
        adc = p["dataword"][dmask].astype(np.float64)
        q_ke = (adc + 0.5 - intercept) / (slope * 1.0e3)   # unclipped; sub-pedestal shows slightly <0
        ev = event_id_per_data_packet(h, dmask)
        chch = packet_chip_channel(h, dmask) if pmap is not None else None

    # ---- per-decay sum + multiplicity ----
    per_evt_q = per_evt_n = None
    if ev is not None and (ev >= 0).any():
        good = ev >= 0
        uev, inv = np.unique(ev[good], return_inverse=True)
        per_evt_q = np.zeros(uev.size)
        np.add.at(per_evt_q, inv, q_ke[good])
        per_evt_n = np.bincount(inv, minlength=uev.size)
        print(f"grouped {int(good.sum())} hits into {uev.size} decays "
              f"(median {np.median(per_evt_n):.0f} hits/decay, max {per_evt_n.max()})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    def draw_k570(ax):
        ax.axvline(q_bare, color="darkorange", ls="--", lw=1.2)
        ax.text(q_bare, ax.get_ylim()[1] * 0.98, "K-570: 482 keV e-", rotation=90,
                va="top", ha="right", fontsize=8, color="darkorange")
        ax.axvline(q_full, color="crimson", ls="--", lw=1.2)
        ax.text(q_full, ax.get_ylim()[1] * 0.98, "K-570: 570 keV (+Pb X-ray)", rotation=90,
                va="top", ha="right", fontsize=8, color="crimson")

    def spectrum(values, title, xlabel, ylabel, color, fname):
        edges = adc_aligned_edges(min(0.0, values.min()), args.qmax, slope, intercept, args.nadc)
        fit = fit_gaussian_peak(values, edges, args.fit_lo, args.fit_hi)
        fig, ax = plt.subplots(figsize=(6.8, 4.9))
        ax.hist(values, bins=edges, histtype="stepfilled", color=color, alpha=0.9, edgecolor=color)
        ax.set_xlim(min(0.0, float(edges[0])), args.qmax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        draw_k570(ax)
        if fit is not None:
            xs = np.linspace(fit["win"][0] - fit["sigma"], fit["win"][1] + fit["sigma"], 200)
            ax.plot(xs, fit["amp"] * np.exp(-0.5 * ((xs - fit["mu"]) / fit["sigma"]) ** 2),
                    color="black", lw=2)
            ax.axvspan(fit["win"][0], fit["win"][1], color="gold", alpha=0.15)
            dev_full = (fit["mu"] - q_full) / fit["dmu"] if fit["dmu"] > 0 else float("nan")
            dev_bare = (fit["mu"] - q_bare) / fit["dmu"] if fit["dmu"] > 0 else float("nan")
            ax.text(0.97, 0.78,
                    f"K-570 Gaussian fit (FWHM region)\n"
                    f"mu = {fit['mu']:.2f} +/- {fit['dmu']:.2f} ke-\n"
                    f"sigma = {fit['sigma']:.2f} ke-  (FWHM {fit['fwhm']:.2f})\n"
                    f"vs 570 keV ({q_full:.2f}): {dev_full:+.1f} sigma\n"
                    f"vs 482 keV ({q_bare:.2f}): {dev_bare:+.1f} sigma",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
        fig.tight_layout()
        out = os.path.join(args.outdir, fname)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")
        return fit

    fit_hit = spectrum(q_ke, f"Per-pixel-hit collected charge   (N={ndata} hits)",
                       "Q per pixel-hit [ke-]", f"hits / {args.nadc} ADC count",
                       "#4477aa", "bi207_perhit_q.png")

    fit_dec = None
    if per_evt_q is not None:
        fit_dec = spectrum(per_evt_q,
                           f"Per-decay charge, summed across its pixels   (N={per_evt_q.size} decays)",
                           "sum Q over the decay's pixel-hits [ke-]", f"decays / {args.nadc} ADC count",
                           "#228833", "bi207_perdecay_q.png")

        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        mx = int(max(1, per_evt_n.max()))
        ax.hist(per_evt_n, bins=np.arange(0.5, mx + 1.5), histtype="stepfilled",
                color="#aa7744", alpha=0.9, edgecolor="#aa7744")
        ax.set_title(f"Pixel-hits per decay   (frac single-pixel {np.mean(per_evt_n == 1):.2f})")
        ax.set_xlabel("pixel-hits in the decay")
        ax.set_ylabel("decays")
        ax.set_xticks(range(1, mx + 1))
        fig.tight_layout()
        out = os.path.join(args.outdir, "bi207_multiplicity.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    # ---- pixel-plane hit heatmap (authoritative physical pixel; LOG color) ----
    n_pix_hit = frac_plane = None
    cc_cent = rr_cent = float("nan")
    if chch is not None and pmap is not None:
        ccmap, nx, ny, pitch, (x0, x1), (y0, y1) = pmap   # nx=larnd-x(Z,128), ny=larnd-y(Y,80)
        chip, chan = chch
        pcol = np.full(chip.size, -1, dtype=np.int64)     # physical pix_x (larnd-x / thesis Z)
        prow = np.full(chip.size, -1, dtype=np.int64)     # physical pix_y (larnd-y / thesis Y)
        for i in range(chip.size):
            pos = ccmap.get((int(chip[i]), int(chan[i])))
            if pos is not None:
                pcol[i], prow[i] = pos
        valid = (pcol >= 0) & (prow >= 0)
        H = np.zeros((nx, ny), dtype=np.int64)
        np.add.at(H, (pcol[valid], prow[valid]), 1)
        n_pix_hit = int((H > 0).sum())
        frac_plane = n_pix_hit / float(nx * ny)
        hit_counts = H[H > 0]

        px, py = (x1 - x0) / nx, (y1 - y0) / ny
        def cm(c, r):                                     # pixel index -> physical (Z,Y) cm
            return x0 + (c + 0.5) * px, y0 + (r + 0.5) * py
        if valid.any():
            cc_cent, rr_cent = float(np.average(pcol[valid])), float(np.average(prow[valid]))
        z_cen, y_cen = cm(cc_cent, rr_cent)
        zc, yc = (x0 + x1) / 2, (y0 + y1) / 2             # anode-plane center = centered-source projection
        order = np.argsort(H, axis=None)[::-1]
        inv = {v: k for k, v in ccmap.items()}            # (pix_x,pix_y) -> (chip,channel)
        print(f"pixel coverage: {n_pix_hit}/{nx*ny} pixels hit ({100*frac_plane:.1f}%); "
              f"hits/hit-pixel median {np.median(hit_counts):.0f}, max {int(hit_counts.max())}")
        print(f"hit centroid: pix ({cc_cent:.1f},{rr_cent:.1f}) = Z {z_cen:.1f} cm, Y {y_cen:.1f} cm  "
              f"[anode center Z {zc:.1f}, Y {yc:.1f}; a centered cathode source lands here]")
        print("hottest pixels (pix_x,pix_y -> Z,Y cm; hits; chip/channel):")
        for idx in order[:8]:
            c, r = np.unravel_index(idx, H.shape)
            if H[c, r] == 0:
                break
            zz, yy = cm(int(c), int(r))
            cc = inv.get((int(c), int(r)))
            cid, ch = (cc[0], cc[1]) if cc is not None else (-1, -1)
            print(f"    ({int(c):3d},{int(r):3d}) -> Z {zz:5.1f}, Y {yy:6.1f}  hits={int(H[c,r]):4d}  "
                  f"chip={cid} channel={ch}")

        # thesis Fig 4.3 anode (Y-Z) plane: Y [cm] (119-148.8, larnd-y, 80 px) VERTICAL, Z [cm]
        # (0-47.6, larnd-x, 128 px) HORIZONTAL; drift (X) perpendicular. Physical pixel from the
        # authoritative chip/channel map, so a centered source lands at the center.
        fig, ax = plt.subplots(figsize=(5.6, 6.8))
        im = ax.imshow(H.T, origin="lower", aspect="auto", extent=[x0, x1, y0, y1],
                       cmap="viridis", norm=LogNorm(vmin=1, vmax=max(2, int(hit_counts.max()))))
        ax.set_title(f"Anode (Y-Z) pixel-plane hit coverage\n"
                     f"{n_pix_hit}/{nx*ny} pixels hit ({100*frac_plane:.1f}%); drift (X) perpendicular; "
                     f"pitch {pitch*10:.2f} mm", fontsize=10)
        ax.set_xlabel(f"Z (cm)   [{nx} pixels, anode horizontal]")
        ax.set_ylabel(f"Y (cm)   [{ny} pixels, anode vertical]")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("hits on the pixel (log)")
        ax.plot(zc, yc, marker="+", ms=15, mew=1.8, color="crimson",
                label="anode center (centered-source projection)")
        if np.isfinite(cc_cent):
            ax.plot(z_cen, y_cen, marker="x", ms=11, mew=1.8, color="white", label="hit centroid")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        fig.tight_layout()
        out = os.path.join(args.outdir, "bi207_pixel_hitmap.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    # ---- text summary ----
    txt = os.path.join(args.outdir, "bi207_larnd.txt")
    with open(txt, "w") as f:
        f.write("Bi-207 FEE response decoded directly from the LARND (larnd-sim FEE output)\n")
        f.write(f"input: {args.larnd}\n")
        f.write(f"ADC = {intercept:.4f} + {slope:.6e} * Q[e-]   (data packet_type {data_type}; "
                f"{args.nadc} ADC count/bin)\n")
        f.write(f"K-570 predicted collected charge: {q_bare:.2f} ke- (bare 482 keV e-), "
                f"{q_full:.2f} ke- (570 keV, +Pb K X-ray)\n")
        f.write(f"data packets (hits): {ndata}\n")
        f.write(f"per-hit Q[ke-]: min {q_ke.min():.2f}  median {np.median(q_ke):.2f}  "
                f"max {q_ke.max():.2f}  mean {q_ke.mean():.2f}\n")
        if fit_hit is not None:
            f.write(f"per-hit K-570 Gaussian fit (FWHM region {fit_hit['win'][0]:.1f}-{fit_hit['win'][1]:.1f} ke-): "
                    f"mu = {fit_hit['mu']:.3f} +/- {fit_hit['dmu']:.3f} ke-, sigma {fit_hit['sigma']:.3f} "
                    f"(FWHM {fit_hit['fwhm']:.2f})\n")
        if per_evt_q is not None:
            f.write(f"\nper-decay summed Q[ke-]: median {np.median(per_evt_q):.2f}  max {per_evt_q.max():.2f}\n")
            f.write(f"hit multiplicity/decay: median {np.median(per_evt_n):.0f}  max {int(per_evt_n.max())}  "
                    f"frac single-pixel {np.mean(per_evt_n == 1):.2f}\n")
            if fit_dec is not None:
                f.write(f"per-decay K-570 Gaussian fit: mu = {fit_dec['mu']:.3f} +/- {fit_dec['dmu']:.3f} ke-, "
                        f"sigma {fit_dec['sigma']:.3f} (FWHM {fit_dec['fwhm']:.2f})\n")
        if n_pix_hit is not None:
            f.write(f"\npixel-plane coverage (this source placement): {n_pix_hit} pixels hit "
                    f"({100*frac_plane:.1f}% of the {nx}x{ny} plane)\n")
            f.write(f"hit centroid: Z {z_cen:.1f} cm, Y {y_cen:.1f} cm; anode center "
                    f"Z {(x0+x1)/2:.1f}, Y {(y0+y1)/2:.1f}\n")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
