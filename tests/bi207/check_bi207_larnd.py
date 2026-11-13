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
#   bi207_pixel_hitmap.png the ANODE (Y-Z) pixel plane (128 Y x 80 Z), LOG-colored by hits/pixel, i.e.
#                          which pixels this source placement illuminates (drift X is perpendicular).
#                          Each packet's pixel is found through larnd-sim's OWN pipeline -- the sim's
#                          mc_packets_assn gives the packet's segment, pixels_from_track indexing +
#                          pixel2id give its pixel, and detsim.get_pixel_coordinates gives the center --
#                          NOT by inverting chip/channel (a standalone PIXEL_CONNECTION_DICT load does
#                          not reproduce simulate_pixels' chip map for this config). The map you need to
#                          decide per-pixel calibration.
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
# 1064 complex. The 976 keV K-conversion electron is the MORE intense CE line (~7%/decay vs ~1.5%
# for 482 keV: the 1063.7 keV M4 transition has a large internal-conversion coefficient). The 1064
# full-energy gamma peak (17.8 ke-) is weak because 1 MeV gammas Compton-scatter far more than they
# photo-absorb -- so the clean second line is the 976 keV CE at 16.3 ke-, not the 1064 gamma peak.
K1064_E_BARE = 0.9757  # MeV, 976 keV conversion electron alone
K1064_E_FULL = 1.0637  # MeV, full 1063.7 keV transition absorbed


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


def hit_pixel_ids(h, data_mask, detector):
    """Pixel each data packet fired, found through larnd-sim's OWN pipeline (not by inverting
    chip/channel, which a standalone load does not reproduce for this config):
      packet --mc_packets_assn--> its leading segment --pixels_from_track indexing--> (i_x,i_y)
             --pixels_from_track.pixel2id--> pixel_id.
    We then read the pixel CENTER from detsim.get_pixel_coordinates (done by the caller). Reusing the
    sim's own functions guarantees the hitmap matches what simulate_pixels wrote. Returns int64 pixel
    ids aligned to packets[data_mask] (-1 where unresolved), plus (nx, ny, plane_of_hit) -- or None.
    """
    try:
        from larndsim.pixels_from_track import pixel2id
        if "mc_packets_assn" not in h or "segments" not in h:
            print("  (no mc_packets_assn/segments in the LARND -> cannot build the hitmap)")
            return None
        assn = np.asarray(h["mc_packets_assn"][:])[data_mask]
        if "segment_ids" not in (assn.dtype.names or ()):
            print(f"  (mc_packets_assn has no segment_ids; fields={assn.dtype.names})")
            return None
        sid = np.asarray(assn["segment_ids"]).reshape(assn.shape[0], -1)[:, 0].astype(np.int64)
        seg = h["segments"]
        sn = seg.dtype.names or ()

        def col(ax):
            if ax in sn:
                return np.asarray(seg[ax][:], float)
            if f"{ax}_start" in sn and f"{ax}_end" in sn:
                return 0.5 * (np.asarray(seg[f"{ax}_start"][:], float) + np.asarray(seg[f"{ax}_end"][:], float))
            return None
        sx, sy = col("x"), col("y")
        if sx is None or sy is None:
            print(f"  (segments lack x/y; fields={sn} -> cannot build the hitmap)")
            return None
        pitch = float(detector.PIXEL_PITCH)
        # fsd_cube is a single anode plane -> plane 0 (the LARND 'segments' pixel_plane field is not a
        # reliable plane index here). borders[0] = the one TPC's x/y range.
        b = np.asarray(detector.TPC_BORDERS)[0]
        bx, by = float(b[0][0]), float(b[1][0])
        pid = np.full(sid.shape, -1, dtype=np.int64)
        ok = (sid >= 0) & (sid < sx.size)
        for i in np.nonzero(ok)[0]:
            s = int(sid[i])
            # pixels_from_track.py:125-126 -- the pixel index is pitches from the TPC border
            i_x = int((sx[s] - bx) // pitch)
            i_y = int((sy[s] - by) // pitch)
            pid[i] = pixel2id(i_x, i_y, 0)
        nx, ny = int(detector.N_PIXELS[0]), int(detector.N_PIXELS[1])
        return pid, nx, ny
    except Exception as exc:
        print(f"  (hit-pixel mapping unavailable: {exc!r} -> skipping the hit heatmap)")
        return None


# Bi-207 mono-energetic lines that could anchor a per-pixel calibration (energy of the emitted
# particle [MeV], label, kind). Conversion electrons (e-) are LOCAL (range <1 px -> they only light
# the source pixel, but give a clean mono line there). Full-energy gamma peaks (g) are DELOCALIZED
# (the photon travels, so they light the whole plane, but sit on a Compton pedestal). intensities are
# per 100 decays. The Pb-K X-ray (~75 keV) is near threshold; listed so the truth spectrum shows it.
BI207_LINES = [
    ("Pb-K X-ray", 0.0750, "x", 0.0),    # ~75-88 keV fluorescence, ~1.2 ke- (near threshold)
    ("K-570 CE",   0.4817, "e", 1.5),    # 8.06 ke-   (K internal conversion of the 570 keV level)
    ("L-570 CE",   0.5537, "e", 0.44),   # ~9.3 ke-
    ("570 gamma",  0.5697, "g", 97.8),   # 9.53 ke-   (full-energy photo-absorption)
    ("K-1064 CE",  0.9757, "e", 7.0),    # 16.3 ke-   (the MORE intense CE line)
    ("L-1064 CE",  1.0478, "e", 1.8),    # ~17.5 ke-
    ("1064 gamma", 1.0637, "g", 74.5),   # 17.8 ke-   (weak photopeak: 1 MeV g mostly Compton)
    ("1770 gamma", 1.7703, "g", 6.87),   # ~29.6 ke-  (rare)
]


def hit_primary_origin(h, data_mask, edep_path):
    """TRUTH-match each data packet to the mono-energetic line that made it. Walk its leading
    segment's trajectory up the parent chain to the particle emitted directly by the decaying nucleus
    (the conversion electron, or the decay gamma whose Compton/photo electron deposited the charge),
    and return that primary's (pdg, E_start[MeV]) -- independent of the reconstructed charge. Returns
    (pdg, E0) int64/float64 arrays aligned to packets[data_mask] (pdg 0 / E0 nan where unresolved),
    or None. This is the 'truth' answer to 'does this pixel really see the line, or just charge in a
    window?': a conversion-electron origin is the real line; a gamma origin is the delocalized halo."""
    try:
        if "mc_packets_assn" not in h or "segments" not in h:
            return None
        assn = np.asarray(h["mc_packets_assn"][:])[data_mask]
        if "segment_ids" not in (assn.dtype.names or ()):
            return None
        sid = np.asarray(assn["segment_ids"]).reshape(assn.shape[0], -1)[:, 0].astype(np.int64)
        seg = h["segments"]
        sn = seg.dtype.names or ()
        if not {"event_id", "traj_id"} <= set(sn):
            print("  (segments lack event_id/traj_id -> truth origin unavailable)")
            return None
        seg_ev = np.asarray(seg["event_id"][:]).astype(np.int64)
        seg_tj = np.asarray(seg["traj_id"][:]).astype(np.int64)
        with open_ro(edep_path) as he:
            if "trajectories" not in he:
                print("  (no 'trajectories' in --edep -> truth origin unavailable)")
                return None
            tr = np.asarray(he["trajectories"][:])
        tn = tr.dtype.names or ()
        need = {"event_id", "traj_id", "parent_id", "pdg_id", "E_start"}
        if not need <= set(tn):
            print(f"  (trajectories missing {need - set(tn)} -> truth origin unavailable)")
            return None
        tev = tr["event_id"].astype(np.int64); ttj = tr["traj_id"].astype(np.int64)
        tpar = tr["parent_id"].astype(np.int64); tpdg = tr["pdg_id"].astype(np.int64)
        tE = tr["E_start"].astype(np.float64)
        idx = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(tev, ttj))}

        def emitted(ev, tj):
            # climb to the particle whose parent is the decaying nucleus (pdg>1e6) / primary / absent;
            # return (pdg, E_start[MeV], traj_id) of that directly-emitted particle
            cur = (ev, tj)
            for _ in range(40):
                if cur not in idx:
                    break
                i = idx[cur]; par = int(tpar[i]); pkey = (ev, par)
                par_nucleus = pkey in idx and abs(int(tpdg[idx[pkey]])) > 1000000
                if par < 0 or pkey not in idx or par_nucleus:
                    return int(tpdg[i]), float(tE[i]), int(ttj[i])
                cur = pkey
            return 0, float("nan"), -1

        pdg = np.zeros(sid.shape, np.int64); E0 = np.full(sid.shape, np.nan)
        atid = np.full(sid.shape, -1, np.int64)      # traj_id of the directly-emitted primary (particle key)
        ok = (sid >= 0) & (sid < seg_ev.size)
        cache = {}
        for i in np.nonzero(ok)[0]:
            key = (int(seg_ev[sid[i]]), int(seg_tj[sid[i]]))
            if key not in cache:
                cache[key] = emitted(*key)
            pdg[i], E0[i], atid[i] = cache[key]
        return pdg, E0, atid
    except Exception as exc:
        print(f"  (truth origin unavailable: {exc!r})")
        return None


def main():
    ap = argparse.ArgumentParser(description="Decode the larnd-sim FEE (LARND) output to charge, fit "
                                             "the Bi-207 K-570 line, and map pixel coverage. No ndlar_flow.")
    ap.add_argument("larnd", help="larnd-sim output .h5 (…_fsd.LARND.h5)")
    ap.add_argument("--edep", default=None, help="EDEPSIM .h5 (trajectories) -> TRUTH-match the per-line "
                    "maps: tag each hit by its directly-emitted primary (conversion e- vs decay gamma), "
                    "instead of only its reconstructed charge window")
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
    q_bare2 = edep_to_ke(K1064_E_BARE)     # 976 keV conversion electron (the clean second line)
    q_full2 = edep_to_ke(K1064_E_FULL)     # full 1063.7 keV transition
    print(f"K-570 predicted collected charge: {q_bare:.2f} ke- (bare 482 keV e-), "
          f"{q_full:.2f} ke- (570 keV, +Pb K X-ray)")

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
        pixinfo = hit_pixel_ids(h, dmask, detector)

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

    # ---- pixel-plane hit heatmap: pixel from larnd-sim's own pixelization (hit_pixel_ids), pixel
    #      center from detsim.get_pixel_coordinates, index from pixels_from_track.id2pixel; LOG color ----
    n_pix_hit = frac_plane = None
    Y_cen = Z_cen = Yctr = Zctr = float("nan")
    if pixinfo is not None:
        from larndsim.pixels_from_track import id2pixel, pixel2id
        from larndsim.detsim import get_pixel_coordinates
        pid, nx, ny = pixinfo                              # nx=larnd-x=thesis-Y(128,long); ny=larnd-y=thesis-Z(80,short)
        B = np.asarray(detector.TPC_BORDERS)[0]
        Y0, Y1 = sorted((float(B[0][0]), float(B[0][1])))  # larnd-x = thesis Y (anode VERTICAL, long, 47.6 cm)
        Z0, Z1 = sorted((float(B[1][0]), float(B[1][1])))  # larnd-y = thesis Z (anode HORIZONTAL, short, 29.76 cm)
        pitch = float(detector.PIXEL_PITCH)
        iY = np.full(pid.shape, -1, dtype=np.int64)        # pixel index along Y (larnd-x)
        iZ = np.full(pid.shape, -1, dtype=np.int64)        # pixel index along Z (larnd-y)
        cY = np.full(pid.shape, np.nan)                    # pixel-center Y [cm] (vertical, long)
        cZ = np.full(pid.shape, np.nan)                    # pixel-center Z [cm] (horizontal, short)
        for i in np.nonzero(pid >= 0)[0]:
            a, b, _pl = id2pixel(int(pid[i]))
            iY[i], iZ[i] = int(a), int(b)
            yy, zz = get_pixel_coordinates(int(pid[i]))    # returns (larnd-x, larnd-y) = (Y, Z)
            cY[i], cZ[i] = float(yy), float(zz)
        valid = (iY >= 0) & (iY < nx) & (iZ >= 0) & (iZ < ny)
        H = np.zeros((nx, ny), dtype=np.int64)             # H[iY, iZ]
        np.add.at(H, (iY[valid], iZ[valid]), 1)
        n_pix_hit = int((H > 0).sum())
        frac_plane = n_pix_hit / float(nx * ny)
        hit_counts = H[H > 0] if (H > 0).any() else np.array([0])
        Yctr, Zctr = (Y0 + Y1) / 2, (Z0 + Z1) / 2          # anode center = centered-source projection
        if valid.any():
            Y_cen, Z_cen = float(np.median(cY[valid])), float(np.median(cZ[valid]))
        order = np.argsort(H, axis=None)[::-1]
        print(f"pixel coverage: {n_pix_hit}/{nx*ny} pixels hit ({100*frac_plane:.1f}%); "
              f"hits/hit-pixel median {np.median(hit_counts):.0f}, max {int(hit_counts.max())}  "
              f"({int(valid.sum())}/{pid.size} hits localized)")
        print(f"hit centroid (pixel centers): Y {Y_cen:.1f} cm, Z {Z_cen:.1f} cm  "
              f"[anode center Y {Yctr:.1f}, Z {Zctr:.1f}; a centered cathode source lands here]")
        print("hottest pixels (Y,Z cm from get_pixel_coordinates; hits):")
        for idx in order[:8]:
            c, r = np.unravel_index(idx, H.shape)
            if H[c, r] == 0:
                break
            yy, zz = get_pixel_coordinates(pixel2id(int(c), int(r), 0))
            print(f"    Y {yy:5.1f}, Z {zz:6.1f}  hits={int(H[c, r]):4d}")

        # thesis Fig 4.3 anode (Y-Z) plane: Y [cm] (larnd-x, 128 px, 47.6 cm) VERTICAL and LONG; Z [cm]
        # (larnd-y, 80 px, 29.76 cm) HORIZONTAL and short; drift (X=larnd-z) perpendicular. The detector
        # is taller in Y than wide in Z -- aspect='equal' shows that 1:1. Pixel from the sim's own
        # pixelization pipeline, so a centered source lands at the center.
        fig, ax = plt.subplots(figsize=(4.8, 7.2))
        im = ax.imshow(H, origin="lower", aspect="equal", extent=[Z0, Z1, Y0, Y1],
                       cmap="viridis", norm=LogNorm(vmin=1, vmax=max(2, int(hit_counts.max()))))
        ax.set_title(f"Anode (Y-Z) pixel-plane hit coverage\n"
                     f"{n_pix_hit}/{nx*ny} pixels hit ({100*frac_plane:.1f}%); drift (X) perpendicular; "
                     f"pitch {pitch*10:.2f} mm", fontsize=9)
        ax.set_xlabel(f"Z (cm)   [{ny} pixels, anode horizontal, short]")
        ax.set_ylabel(f"Y (cm)   [{nx} pixels, anode vertical, long]")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("hits on the pixel (log)")
        ax.plot(Zctr, Yctr, marker="+", ms=15, mew=1.8, color="crimson",
                label="anode center (centered-source projection)")
        if np.isfinite(Y_cen):
            ax.plot(Z_cen, Y_cen, marker="x", ms=11, mew=1.8, color="white", label="hit centroid")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        fig.tight_layout()
        out = os.path.join(args.outdir, "bi207_pixel_hitmap.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

        # ---- per-LINE hitmaps: which pixels see each Bi-207 charge line (for cross-checking gains) ----
        # Charge lines (ke-), via the same box+drift model as the reference lines:
        #   570 complex  = K-570 conversion e- (482 keV -> 8.06 ke-) + full 570 keV absorption (9.53)
        #   1064 complex = K-1064 conversion e- (976 keV -> 16.3 ke-) + full 1064 keV absorption (17.8)
        # A pixel that shows BOTH gives two known charge points -> a per-pixel gain/linearity cross-check.
        # q_ke, iY, iZ and `valid` are all aligned per data-hit, so a Q-window selects that line's hits.
        from matplotlib.colors import BoundaryNorm
        lines = [("570", "K-570 e- + 570 full", 6.0, 11.0),
                 ("1064", "K-1064 e- + 1064 full", 14.0, 20.0)]
        nseen = np.zeros((nx, ny), dtype=np.int64)
        print("per-line pixel coverage (cross-check lines):")
        for tag, name, lo, hi in lines:
            m = valid & (q_ke >= lo) & (q_ke <= hi)
            Hl = np.zeros((nx, ny), dtype=np.int64)
            np.add.at(Hl, (iY[m], iZ[m]), 1)
            nseen += (Hl > 0).astype(np.int64)
            nlit, nhit = int((Hl > 0).sum()), int(Hl.sum())
            print(f"    {name} ({lo:.0f}-{hi:.0f} ke-): {nhit} hits on {nlit} pixels")
            if nhit == 0:
                continue
            fig, ax = plt.subplots(figsize=(4.8, 7.2))
            im = ax.imshow(Hl, origin="lower", aspect="equal", extent=[Z0, Z1, Y0, Y1],
                           cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, int(Hl.max()))))
            ax.set_title(f"Bi-207 {name}\n{lo:.0f}-{hi:.0f} ke-: {nhit} hits on {nlit} pixels", fontsize=9)
            ax.set_xlabel(f"Z (cm)  [{ny} px, short, horizontal]")
            ax.set_ylabel(f"Y (cm)  [{nx} px, long, vertical]")
            fig.colorbar(im, ax=ax).set_label("hits on the pixel (log)")
            ax.plot(Zctr, Yctr, marker="+", ms=14, mew=1.6, color="cyan")
            fig.tight_layout()
            o = os.path.join(args.outdir, f"bi207_line_{tag}.png")
            fig.savefig(o, dpi=140)
            plt.close(fig)
            print(f"wrote {o}")
        # cross-check map: how many of the lines each pixel sees (>=2 -> a per-pixel gain cross-check)
        both = int((nseen >= 2).sum())
        print(f"    pixels seeing >=2 lines (cross-checkable): {both}")
        fig, ax = plt.subplots(figsize=(4.8, 7.2))
        cmap = plt.get_cmap("viridis", len(lines) + 1)
        im = ax.imshow(nseen, origin="lower", aspect="equal", extent=[Z0, Z1, Y0, Y1],
                       cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, len(lines) + 1.5), cmap.N))
        ax.set_title(f"Bi-207 charge lines seen per pixel\n{both} pixels see >=2 (gain cross-check)", fontsize=9)
        ax.set_xlabel(f"Z (cm)  [{ny} px, short, horizontal]")
        ax.set_ylabel(f"Y (cm)  [{nx} px, long, vertical]")
        fig.colorbar(im, ax=ax, ticks=range(len(lines) + 1)).set_label("number of Bi-207 charge lines seen")
        fig.tight_layout()
        out = os.path.join(args.outdir, "bi207_lines_seen.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

        # ---- TRUTH line census: which mono-energetic lines does the source give, and where? ----
        if getattr(args, "edep", None):
            with open_ro(args.larnd) as h2:
                origin = hit_primary_origin(h2, dmask, args.edep)
            if origin is not None and origin[0].shape == valid.shape:
                pdg0, E0, atid = origin
                iY0, iZ0 = (Yctr - Y0) / pitch, (Zctr - Z0) / pitch   # source pixel in index space
                e_lines = [(lab, E) for lab, E, k, _ in BI207_LINES if k == "e"]
                g_lines = [(lab, E) for lab, E, k, _ in BI207_LINES if k in ("g", "x")]

                def match(pg, E):
                    if not np.isfinite(E):
                        return ""
                    if abs(pg) == 11:
                        lab, EE = min(e_lines, key=lambda le: abs(le[1] - E))
                        return lab if abs(EE - E) < 0.10 else "e-other"
                    if pg == 22:
                        lab, EE = min(g_lines, key=lambda le: abs(le[1] - E))
                        return lab if abs(EE - E) < 0.12 else "g-other"
                    return "other"

                line = np.array([match(int(pdg0[i]), float(E0[i])) for i in range(pdg0.size)], dtype=object)
                res = int((valid & (line == "")).sum())
                print(f"TRUTH line census ({int(valid.sum())-res} of {int(valid.sum())} localized hits "
                      f"traced to a primary; {res} unresolved):")
                print(f"  {'line':12s} {'emitted':>9s} {'collected':>10s} {'hits':>6s} {'pixels':>7s} "
                      f"{'rms':>6s}  role")
                census = []
                for lab, E, k, inten in BI207_LINES:
                    m = valid & (line == lab)
                    n = int(m.sum())
                    if n == 0:
                        continue
                    qpix = int(np.unique(iY[m] * ny + iZ[m]).size)
                    rr = pitch * float(np.sqrt(np.mean((iY[m] - iY0) ** 2 + (iZ[m] - iZ0) ** 2)))
                    role = "LOCAL e- line" if k == "e" else ("delocalized g peak" if k == "g" else "near-threshold")
                    census.append((lab, edep_to_ke(E), n, qpix, rr, role))
                    print(f"  {lab:12s} {E*1e3:6.0f} keV {edep_to_ke(E):7.2f} ke- {n:6d} {qpix:7d} "
                          f"{rr:5.1f}cm  {role}")
                for lab in ("e-other", "g-other", "other"):
                    n = int((valid & (line == lab)).sum())
                    if n:
                        print(f"  {lab:12s} {'':>9s} {'':>10s} {n:6d}")
                print("  -> LOCAL (conversion-electron) lines give a clean mono peak but only on the source")
                print("     pixel; delocalized gamma full-energy peaks light the plane but ride a Compton floor.")

                # Plot A: the source's mono-energetic lines, from the emitted-primary truth energy (delta
                # lines, before detector smearing). e- vs gamma separated; catalogue energies marked.
                fin = valid & np.isfinite(E0)
                if int(fin.sum()) > 0:
                    eE = E0[fin & (np.abs(pdg0) == 11)] * 1e3
                    gE = E0[fin & (pdg0 == 22)] * 1e3
                    fig, ax = plt.subplots(figsize=(7.4, 4.6))
                    be = np.arange(0, 1860, 15)
                    ax.hist(gE, bins=be, histtype="stepfilled", color="#4477aa", alpha=0.8, label="gamma-origin")
                    ax.hist(eE, bins=be, histtype="stepfilled", color="#cc6677", alpha=0.8, label="e--origin (CE)")
                    for lab, E, k, _ in BI207_LINES:
                        ax.axvline(E * 1e3, color="0.4", ls=":", lw=0.9)
                        ax.text(E * 1e3, ax.get_ylim()[1] * 0.97, lab, rotation=90, va="top",
                                ha="right", fontsize=7, color="0.3")
                    ax.set_yscale("log")
                    ax.set_title("Bi-207 mono-energetic lines (truth emitted-primary energy per hit)", fontsize=9)
                    ax.set_xlabel("emitted primary energy [keV]")
                    ax.set_ylabel("hits / 15 keV")
                    ax.legend(loc="upper right", fontsize=8)
                    fig.tight_layout()
                    o = os.path.join(args.outdir, "bi207_truth_lines.png")
                    fig.savefig(o, dpi=140); plt.close(fig); print(f"wrote {o}")

                # Plot B: the reconstructed CHARGE spectrum decomposed by truth origin -- shows which lines
                # survive as fittable peaks vs melt into the Compton continuum (what we can calibrate off).
                order = ["K-570 CE", "570 gamma", "L-570 CE", "K-1064 CE", "1064 gamma", "L-1064 CE",
                         "1770 gamma", "Pb-K X-ray"]
                cols = {"K-570 CE": "#ee6677", "570 gamma": "#4477aa", "L-570 CE": "#f4a0a8",
                        "K-1064 CE": "#228833", "1064 gamma": "#66ccee", "L-1064 CE": "#99ddbb",
                        "1770 gamma": "#aa3377", "Pb-K X-ray": "#bbbbbb"}
                present = [L for L in order if int((valid & (line == L)).sum()) > 0]
                if present:
                    edges = adc_aligned_edges(0.0, args.qmax, slope, intercept, args.nadc)
                    stacks = [q_ke[valid & (line == L)] for L in present]
                    rest = valid & ~np.isin(line.astype(str), np.array(present + [""], dtype=str))
                    if int(rest.sum()) > 0:
                        stacks.append(q_ke[rest]); present = present + ["Compton/other"]
                        cols["Compton/other"] = "0.75"
                    fig, ax = plt.subplots(figsize=(7.6, 4.9))
                    ax.hist(stacks, bins=edges, stacked=True, histtype="stepfilled",
                            color=[cols[L] for L in present], label=present)
                    for lab, E, k, _ in BI207_LINES:
                        ax.axvline(edep_to_ke(E), color="0.4", ls=":", lw=0.8)
                    ax.set_xlim(0, args.qmax)
                    ax.set_title("Collected charge decomposed by TRUTH origin (which lines are fittable peaks)", fontsize=9)
                    ax.set_xlabel("Q per pixel-hit [ke-]")
                    ax.set_ylabel(f"hits / {args.nadc} ADC count")
                    ax.legend(loc="upper right", fontsize=7, ncol=2)
                    fig.tight_layout()
                    o = os.path.join(args.outdir, "bi207_charge_by_truth.png")
                    fig.savefig(o, dpi=140); plt.close(fig); print(f"wrote {o}")

                # Plot C: 570 complex -- charge-window selection vs TRUTH conversion-electron origin, to
                # show the window map's spread is gamma, not the local K-570 electron line.
                def amap(selmask):
                    M = np.zeros((nx, ny), np.int64)
                    np.add.at(M, (iY[selmask], iZ[selmask]), 1)
                    return M
                H_win = amap(valid & (q_ke >= 6.0) & (q_ke <= 11.0))
                H_tce = amap(valid & (line == "K-570 CE"))
                fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 7.0), sharey=True)
                for ax, M, ttl in [(axA, H_win, f"charge window 6-11 ke-\n{int((H_win>0).sum())} px, {int(H_win.sum())} hits"),
                                   (axB, H_tce, f"TRUTH 482 keV conversion e-\n{int((H_tce>0).sum())} px, {int(H_tce.sum())} hits")]:
                    vmax = max(2, int(M.max()))
                    im = ax.imshow(M, origin="lower", aspect="equal", extent=[Z0, Z1, Y0, Y1],
                                   cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax))
                    ax.set_title(ttl, fontsize=9); ax.set_xlabel("Z (cm)  [short]")
                    ax.plot(Zctr, Yctr, "+", ms=13, mew=1.5, color="crimson")
                    fig.colorbar(im, ax=ax)
                axA.set_ylabel("Y (cm)  [long, vertical]")
                fig.suptitle("Per-line map: reconstructed charge window vs TRUTH line origin", fontsize=10)
                fig.tight_layout()
                o = os.path.join(args.outdir, "bi207_line_570_truth.png")
                fig.savefig(o, dpi=140); plt.close(fig); print(f"wrote {o}")

                # ---- multiplicity: does a line split across pixels, so the peak is in the PARTICLE-
                #      summed Q rather than a single hit? Group each line's hits by their emitting primary
                #      (event_id, ancestor traj_id) and sum Q across that one particle's pixels. If
                #      hits/particle>1 and summed Q lands on the line while 1-pixel Q falls short, the line
                #      is charge-shared and must be clustered before fitting the gain.
                if ev is not None:
                    pk = ev.astype(np.int64) * 1000000 + atid        # unique key per emitted primary
                    base = valid & (atid >= 0) & (ev >= 0)
                    print("per-line multiplicity (one emitted particle's pixels, and its Q single vs summed):")
                    print(f"  {'line':12s} {'parts':>6s} {'hits/part':>9s} {'multi%':>6s} "
                          f"{'1pix Q':>7s} {'summedQ':>8s}   line")
                    for lab, E, k, _ in BI207_LINES:
                        s = base & (line == lab)
                        if int(s.sum()) == 0:
                            continue
                        keys, invp = np.unique(pk[s], return_inverse=True)
                        nper = np.bincount(invp)
                        sq = np.zeros(keys.size); np.add.at(sq, invp, q_ke[s])
                        one = nper == 1
                        q1 = float(np.mean(sq[one])) if one.any() else float("nan")
                        print(f"  {lab:12s} {keys.size:6d} {nper.mean():9.2f} {100*np.mean(nper > 1):6.0f} "
                              f"{q1:7.2f} {np.mean(sq):8.2f}   {edep_to_ke(E):.1f} ke-")
                    print("  -> hits/part ~1 and 1pix Q ~ summedQ  => the line sits in single hits (no clustering"
                          " needed); hits/part>1 with summedQ > 1pix Q => charge-shared, cluster before fitting.")
            elif origin is None:
                print("  (truth line census skipped: origin unavailable -- check --edep has 'trajectories')")

        # ---- near-source vs far charge spectrum: can we SEE the 1064 lines? ----
        # The conversion electrons (482 & 976 keV) range out in <1 pixel, so they pile onto the
        # source pixel; the 570/1064 gammas Compton-scatter across the whole plane. Cutting on
        # radius from the source projection therefore ENRICHES the monoenergetic CE lines relative
        # to the gamma continuum -- the best chance to resolve the 976 keV CE (16.3 ke-) as a clean
        # SECOND calibration line. Far-field hits are the gamma Compton floor for contrast.
        dsrc = np.hypot(cY - Yctr, cZ - Zctr)
        near = valid & (dsrc < 1.0)                    # within ~2-3 pixels of the source
        far = valid & (dsrc > 4.0)                     # pure gamma halo
        if int(near.sum()) > 0:
            edges = adc_aligned_edges(0.0, 22.0, slope, intercept, args.nadc)
            qn, qf = q_ke[near], q_ke[far]
            fit1064 = fit_gaussian_peak(qn, edges, 13.5, 19.0)
            n1064 = int(((qn >= 14.0) & (qn <= 19.0)).sum())
            print(f"near-source spectrum ({int(near.sum())} hits, d<1cm) vs far halo "
                  f"({int(far.sum())} hits, d>4cm):")
            print(f"    976 keV CE expected at {q_bare2:.1f} ke-, 1064 full at {q_full2:.1f} ke-; "
                  f"{n1064} near hits in 14-19 ke-")
            if fit1064 is not None:
                dev = (fit1064["mu"] - q_bare2) / fit1064["dmu"] if fit1064["dmu"] > 0 else float("nan")
                print(f"    1064-region Gaussian: mu {fit1064['mu']:.1f} +/- {fit1064['dmu']:.1f} ke-, "
                      f"sigma {fit1064['sigma']:.1f} ({(fit1064['mu']-q_bare2):+.1f} ke- vs 976 CE, {dev:+.1f} sigma)")
            else:
                print("    1064-region fit did not converge (too few hits -> needs the larger sample)")
            fig, ax = plt.subplots(figsize=(7.0, 4.9))
            ax.hist(qn, bins=edges, histtype="stepfilled", color="#aa3377", alpha=0.85,
                    edgecolor="#aa3377", label=f"near source (d<1 cm, N={int(near.sum())})")
            if int(far.sum()) > 0:                     # scale the far halo to the near-source area for shape
                wf = np.full(qf.shape, near.sum() / max(far.sum(), 1))
                ax.hist(qf, bins=edges, weights=wf, histtype="step", color="#888888", lw=1.6,
                        label=f"far halo (d>4 cm, scaled, N={int(far.sum())})")
            for qx, lbl, col in [(q_bare, "482 CE", "darkorange"), (q_full, "570 full", "crimson"),
                                 (q_bare2, "976 CE", "teal"), (q_full2, "1064 full", "navy")]:
                ax.axvline(qx, color=col, ls="--", lw=1.2)
                ax.text(qx, ax.get_ylim()[1] * 0.98, lbl, rotation=90, va="top", ha="right",
                        fontsize=8, color=col)
            if fit1064 is not None:
                xs = np.linspace(13.5, 19.0, 120)
                ax.plot(xs, fit1064["amp"] * np.exp(-0.5 * ((xs - fit1064["mu"]) / fit1064["sigma"]) ** 2),
                        color="black", lw=1.8)
            ax.set_xlim(0, 22)
            ax.set_title("Near-source vs far charge spectrum: is the 976 keV CE line (16.3 ke-) visible?", fontsize=9)
            ax.set_xlabel("Q per pixel-hit [ke-]")
            ax.set_ylabel(f"hits / {args.nadc} ADC count")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            out = os.path.join(args.outdir, "bi207_nearfar_spectrum.png")
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
            f.write(f"hit centroid: Y {Y_cen:.1f} cm, Z {Z_cen:.1f} cm; anode center "
                    f"Y {Yctr:.1f}, Z {Zctr:.1f}\n")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
