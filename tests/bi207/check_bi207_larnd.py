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


def hit_drift_fraction(h, data_mask, detector):
    """Per data-hit drift fraction f in [0,1] (0 = anode, 1 = cathode), from the leading segment's drift
    (larnd-z) coordinate. Returns a float array aligned to packets[data_mask] (nan where unresolved),
    or None. Lets us undo the depth-dependent drift attenuation (q = q0 * DRIFT_ATTEN**f) so the gamma
    full-energy charge lines -- smeared across interaction depth -- collapse back to sharp peaks:
    q_cathode = q * DRIFT_ATTEN**(1-f)."""
    try:
        if "mc_packets_assn" not in h or "segments" not in h:
            return None
        assn = np.asarray(h["mc_packets_assn"][:])[data_mask]
        if "segment_ids" not in (assn.dtype.names or ()):
            return None
        sid = np.asarray(assn["segment_ids"]).reshape(assn.shape[0], -1)[:, 0].astype(np.int64)
        seg = h["segments"]; sn = seg.dtype.names or ()
        if "z" in sn:
            sz = np.asarray(seg["z"][:], float)
        elif "z_start" in sn and "z_end" in sn:
            sz = 0.5 * (np.asarray(seg["z_start"][:], float) + np.asarray(seg["z_end"][:], float))
        else:
            return None
        b = np.asarray(detector.TPC_BORDERS)[0]
        za, zc = float(b[2][0]), float(b[2][1])          # drift (larnd-z) range: [anode, cathode]
        f = np.full(sid.shape, np.nan)
        ok = (sid >= 0) & (sid < sz.size)
        f[ok] = (za - sz[sid[ok]]) / (za - zc)
        return np.clip(f, 0.0, 1.0)
    except Exception as exc:
        print(f"  (drift fraction unavailable: {exc!r})")
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
    """TRUTH-match each data packet to the mono-energetic line that made it. Attribute the hit to the
    particle contributing the most CHARGE (sum of mc_packets_assn 'fraction' over that particle's
    segments -- NOT the single top segment, since a high-energy track is split into many small segments
    while a co-located soft delta/Auger dumps its energy in one), then walk that particle up its parent
    chain to the one emitted directly by the decaying nucleus (the conversion electron, or the decay
    gamma whose Compton/photo electron deposited the charge). Return that primary's (pdg, E_start[MeV],
    traj_id), aligned to packets[data_mask] (pdg 0 / E0 nan where unresolved), or None. The 'truth'
    answer to 'is this really the line, or just charge in a window?': e- origin = real local line,
    gamma origin = delocalized halo."""
    try:
        if "mc_packets_assn" not in h or "segments" not in h:
            return None
        assn = np.asarray(h["mc_packets_assn"][:])[data_mask]
        if "segment_ids" not in (assn.dtype.names or ()):
            return None
        segids = np.asarray(assn["segment_ids"]).reshape(assn.shape[0], -1).astype(np.int64)   # (n, K)
        if "fraction" in (assn.dtype.names or ()):
            frac = np.abs(np.asarray(assn["fraction"]).reshape(assn.shape[0], -1).astype(np.float64))
        else:
            frac = (segids >= 0).astype(np.float64)        # equal weight if no fraction stored
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

        seg_cache = {}                               # segment index -> (pdg, E_start, emitted traj_id)
        def seg_primary(sidx):
            r = seg_cache.get(sidx)
            if r is None:
                r = emitted(int(seg_ev[sidx]), int(seg_tj[sidx])) if 0 <= sidx < seg_ev.size \
                    else (0, float("nan"), -1)
                seg_cache[sidx] = r
            return r

        n, K = segids.shape
        pdg = np.zeros(n, np.int64); E0 = np.full(n, np.nan); atid = np.full(n, -1, np.int64)
        aev = np.full(n, -1, np.int64)               # event id (segment frame) of the emitted quantum
        for i in range(n):
            acc = {}                                 # emitted traj_id -> [summed fraction, pdg, E_start]
            ev_i = -1
            for j in range(K):
                s = int(segids[i, j])
                if s < 0:
                    continue
                if ev_i < 0 and 0 <= s < seg_ev.size:
                    ev_i = int(seg_ev[s])            # all a packet's segments share one event
                pg, Es, tid = seg_primary(s)
                if tid < 0:
                    continue
                a = acc.get(tid)
                if a is None:
                    acc[tid] = [float(frac[i, j]), pg, Es]
                else:
                    a[0] += float(frac[i, j])
            if acc:
                tid_best = max(acc, key=lambda t: acc[t][0])
                _, pg, Es = acc[tid_best]
                pdg[i], E0[i], atid[i] = pg, Es, tid_best
                aev[i] = ev_i
        return pdg, E0, atid, aev
    except Exception as exc:
        print(f"  (truth origin unavailable: {exc!r})")
        return None


def emitted_deposited_energy(h, edep_path):
    """Total TRUTH deposited energy [MeV] per emitted quantum, keyed (event_id, emitted_traj_id):
    sum segment dE over every segment whose ancestry roots to that directly-emitted particle (the
    decay gamma or the conversion electron). This is DRIFT-INDEPENDENT, so it cleanly tags a full
    gamma absorption (photopeak) vs a Compton escape -- which the measured charge cannot, because
    drift attenuation (0.667-1.0x) also pulls full absorptions down into the Compton region.
    Returns {} if the truth tables are unavailable."""
    try:
        if "segments" not in h:
            return {}
        seg = h["segments"]; sn = seg.dtype.names or ()
        if not {"event_id", "traj_id", "dE"} <= set(sn):
            return {}
        seg_ev = np.asarray(seg["event_id"][:]).astype(np.int64)
        seg_tj = np.asarray(seg["traj_id"][:]).astype(np.int64)
        seg_dE = np.asarray(seg["dE"][:]).astype(np.float64)
        with open_ro(edep_path) as he:
            if "trajectories" not in he:
                return {}
            tr = np.asarray(he["trajectories"][:])
        tn = tr.dtype.names or ()
        if not {"event_id", "traj_id", "parent_id", "pdg_id"} <= set(tn):
            return {}
        tev = tr["event_id"].astype(np.int64); ttj = tr["traj_id"].astype(np.int64)
        tpar = tr["parent_id"].astype(np.int64); tpdg = tr["pdg_id"].astype(np.int64)
        idx = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(tev, ttj))}
        root_cache = {}
        def root(ev, tj):                                   # emitted-quantum key for a trajectory
            key = (ev, tj)
            r = root_cache.get(key)
            if r is not None:
                return r
            cur = key
            for _ in range(40):
                if cur not in idx:
                    break
                i = idx[cur]; par = int(tpar[i]); pkey = (ev, par)
                par_nucleus = pkey in idx and abs(int(tpdg[idx[pkey]])) > 1000000
                if par < 0 or pkey not in idx or par_nucleus:
                    root_cache[key] = (ev, int(ttj[i])); return root_cache[key]
                cur = pkey
            root_cache[key] = key; return key
        Edep = {}
        for s in range(seg_ev.size):
            rk = root(int(seg_ev[s]), int(seg_tj[s]))
            Edep[rk] = Edep.get(rk, 0.0) + float(seg_dE[s])
        return Edep
    except Exception as exc:
        print(f"  (emitted deposited energy unavailable: {exc!r})")
        return {}


def stack_by_label(ax, values, labels, order, colmap, edges):
    """Stacked step-filled histogram of `values` grouped by `labels` (object array), in `order`,
    colored by colmap. Returns the present labels (those with >0 entries)."""
    pres = [L for L in order if int((labels == L).sum()) > 0]
    if pres:
        ax.hist([values[labels == L] for L in pres], bins=edges, stacked=True, histtype="stepfilled",
                color=[colmap[L] for L in pres], label=[f"{L} ({int((labels == L).sum())})" for L in pres])
    return pres


def pixel_sum_label(q, ev, iY, iZ, mask, nx, ny, labels):
    """Group masked hits into (event,pixel) cells, sum their charge, and label each cell by its
    HIGHEST-charge hit. Returns (cell_sum_charge, cell_label). This is the within-pixel Q-sum (merges
    the 2-3 packets one deposit fires on a pixel) -- NOT a cross-pixel cluster sum."""
    qv = q[mask]; evv = ev[mask].astype(np.int64); iy = iY[mask]; iz = iZ[mask]; lb = labels[mask]
    if qv.size == 0:
        return np.array([]), np.array([], dtype=object)
    cell = evv * (nx * ny) + (iy * ny + iz)
    uc, cinv = np.unique(cell, return_inverse=True)
    csum = np.zeros(uc.size); np.add.at(csum, cinv, qv)
    clab = np.empty(uc.size, dtype=object)
    for k in np.argsort(qv):                 # ascending -> the highest-charge hit labels the cell
        clab[cinv[k]] = lb[k]
    return csum, clab


def truth_dE_per_decay(edep_path, outdir):
    """TRUTH energy deposited in the LAr per decay -- straight from the EDEPSIM segments (dE summed
    per event_id), INDEPENDENT of the pixel readout. The EDEP runs in the large module0 LAr, which
    contains the full Bi-207 cascade, so the per-decay total piles at the Pb-207 LEVEL energies, not
    at an isolated 1.064: the 1064 gamma is emitted in cascade WITH a 570 (1633 keV level -> 1064 g
    -> 570 level -> 570 g), so a contained 1064 decay deposits 1.633 MeV, and a 1770 decay 2.340 MeV.
    Compare the equivalent-charge top axis to the readout per-event-total (which peaks near 0.57
    MeV-equiv ~ 9.5 ke-): the shortfall is the charge the small fsd_cube readout footprint misses."""
    import matplotlib.pyplot as plt
    try:
        with open_ro(edep_path) as he:
            if "segments" not in he:
                print("  (no 'segments' in --edep -> truth per-decay dE unavailable)")
                return
            seg = he["segments"]; sn = seg.dtype.names or ()
            if not {"event_id", "dE"} <= set(sn):
                print(f"  (segments lack event_id/dE; have {sn} -> truth per-decay dE unavailable)")
                return
            seg_ev = np.asarray(seg["event_id"][:]).astype(np.int64)
            seg_dE = np.asarray(seg["dE"][:]).astype(np.float64)
    except Exception as exc:
        print(f"  (truth per-decay dE unavailable: {exc!r})")
        return
    if seg_ev.size == 0:
        print("  (no segments in --edep)"); return
    uev, inv = np.unique(seg_ev, return_inverse=True)
    etot = np.zeros(uev.size); np.add.at(etot, inv, seg_dE)      # MeV deposited in LAr per decay
    ke_per_mev = edep_to_ke(1.0)                                 # linear MeV->ke- (full-drift collected)
    # Pb-207 de-excitation levels Bi-207 feeds; full containment piles the per-decay total here.
    LEVELS = [(0.5697, "570 level"), (1.0637, "1064 g (570 esc.)"), (1.6334, "1064+570"),
              (1.7702, "1770 g (570 esc.)"), (2.3399, "1770+570")]
    emax = 2.6
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.hist(etot, bins=np.linspace(0.0, emax, 261), histtype="stepfilled",
            color="#3b6ea5", alpha=0.92, edgecolor="#1f3a5f", linewidth=0.4)
    ytop = ax.get_ylim()[1]
    for e, lab in LEVELS:
        ax.axvline(e, color="0.3", ls=":", lw=1.0)
        ax.text(e, ytop * 0.985, f"{lab}  {e:.3f}", rotation=90, va="top", ha="right",
                fontsize=7.5, color="0.2")
    # EC leaves a K-shell vacancy in the daughter Pb; the ~75 keV Pb Ka X-ray (or Auger) is usually
    # absorbed locally, so each nuclear line has a companion ~75 keV HIGHER -> the taller "+Kx" peaks.
    XRAY = 0.0749
    for e, lab in [(0.5697, "570+Kx"), (1.6334, "1064+570+Kx"), (2.3399, "1770+570+Kx")]:
        ax.axvline(e + XRAY, color="seagreen", ls="--", lw=1.0, alpha=0.85)
        ax.text(e + XRAY, ytop * 0.60, f"{lab} {e+XRAY:.3f}", rotation=90, va="top", ha="left",
                fontsize=7, color="seagreen")
    ax.set_xlim(0, emax)
    ax.set_xlabel("TRUTH energy deposited in LAr per decay [MeV]")
    ax.set_ylabel("decays")
    ax.set_title("Truth per-decay deposited energy (EDEPSIM segments, big module0 LAr)")
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * ke_per_mev, lambda q: q / ke_per_mev))
    secax.set_xlabel("equivalent fully-collected charge [ke-]  (edep_to_ke, full drift)")
    fig.tight_layout()
    out = os.path.join(outdir, "bi207_truth_dE_per_decay.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    print(f"  truth per-decay dE ({etot.size} decays with any LAr deposit): median "
          f"{np.median(etot):.3f} MeV, mean {np.mean(etot):.3f} MeV, max {etot.max():.3f} MeV")
    for lo, hi, name in [(0.50, 0.75, "570        [0.50-0.75]"),
                         (1.55, 1.80, "1064+570   [1.55-1.80]"),
                         (2.25, 2.50, "1770+570   [2.25-2.50]")]:
        n = int(((etot >= lo) & (etot < hi)).sum())
        print(f"    {name} MeV: {n:5d} ({100.0*n/etot.size:4.1f}%)   "
              f"~{0.5*(lo+hi)*ke_per_mev:.1f} ke- if fully collected")
    print("  -> readout per-decay median ~9.45 ke- ~ 0.57 MeV-equiv: the fsd_cube readout collects about "
          "ONE 570's worth; the 1.633/2.340 cascades DO deposit in LAr but spill the small readout footprint.")


def truth_dE_incube(fsd_edep_path, detector, outdir):
    """IMPOSE THE FSD-CUBE BOUNDARIES on the truth. Using the TRANSLATED (_fsd) segments -- which live
    in the cube frame and still carry EVERY deposit (bi207_to_fsd keeps them all; simulate_pixels drops
    the out-of-volume ones only at drift) -- sum dE per decay (a) over ALL segments [= big-volume total]
    and (b) only over segments INSIDE detector.TPC_BORDERS [= what the cube contains]. The gap is energy
    deposited in LAr but OUTSIDE the ~30x48 cm cube: the 1.633/2.340 MeV cascade peaks drain toward the
    ~0.57 blob the readout actually sees -- the direct proof that the loss is the footprint, not the FEE."""
    import matplotlib.pyplot as plt
    try:
        with open_ro(fsd_edep_path) as he:
            if "segments" not in he:
                print(f"  (no 'segments' in {fsd_edep_path} -> in-cube truth unavailable)"); return
            seg = he["segments"]; sn = seg.dtype.names or ()
            if not {"event_id", "dE"} <= set(sn):
                print("  (translated segments lack event_id/dE -> in-cube truth unavailable)"); return
            def col(ax):
                if ax in sn:
                    return np.asarray(seg[ax][:], float)
                if f"{ax}_start" in sn and f"{ax}_end" in sn:
                    return 0.5 * (np.asarray(seg[f"{ax}_start"][:], float) + np.asarray(seg[f"{ax}_end"][:], float))
                return None
            sev = np.asarray(seg["event_id"][:]).astype(np.int64)
            sde = np.asarray(seg["dE"][:]).astype(np.float64)
            sx, sy, sz = col("x"), col("y"), col("z")
    except Exception as exc:
        print(f"  (in-cube truth unavailable: {exc!r})"); return
    if sx is None or sy is None or sz is None:
        print("  (translated segments lack x/y/z -> in-cube truth unavailable)"); return
    B = np.asarray(detector.TPC_BORDERS)[0]
    inside = np.ones(sde.size, bool)
    for v, ax in ((sx, 0), (sy, 1), (sz, 2)):
        lo, hi = sorted((float(B[ax, 0]), float(B[ax, 1])))
        inside &= (v >= lo) & (v <= hi)
    uev, inv = np.unique(sev, return_inverse=True)
    full = np.zeros(uev.size); np.add.at(full, inv, sde)
    incube = np.zeros(uev.size); np.add.at(incube, inv, np.where(inside, sde, 0.0))
    ke_per_mev = edep_to_ke(1.0)
    LEVELS = [(0.5697, "570"), (1.6334, "1064+570"), (2.3399, "1770+570")]
    emax = 2.6
    bins = np.linspace(0.0, emax, 261)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.hist(full, bins=bins, histtype="stepfilled", color="#c9d6e5", edgecolor="#8aa0b8",
            label=f"deposited in LAr (big volume)   med {np.median(full):.2f} MeV")
    ax.hist(incube, bins=bins, histtype="step", color="#b2182b", lw=1.7,
            label=f"inside FSD-cube boundaries      med {np.median(incube):.2f} MeV")
    ytop = ax.get_ylim()[1]
    for e, lab in LEVELS:
        ax.axvline(e, color="0.35", ls=":", lw=1.0)
        ax.text(e, ytop * 0.985, f"{lab} {e:.3f}", rotation=90, va="top", ha="right", fontsize=7.5, color="0.25")
    ax.set_xlim(0, emax)
    ax.set_xlabel("TRUTH energy deposited per decay [MeV]")
    ax.set_ylabel("decays")
    ax.set_title("Truth per-decay energy: big LAr volume vs INSIDE the FSD-cube boundaries")
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * ke_per_mev, lambda q: q / ke_per_mev))
    secax.set_xlabel("equivalent fully-collected charge [ke-]")
    ax.legend(loc="center right", fontsize=8); fig.tight_layout()
    out = os.path.join(outdir, "bi207_truth_dE_incube.png")
    fig.savefig(out, dpi=130); plt.close(fig); print(f"wrote {out}")
    retain = 100.0 * incube.sum() / full.sum() if full.sum() > 0 else float("nan")
    print(f"  in-cube vs big-volume: total energy retained {retain:.1f}%; per-decay median "
          f"{np.median(incube):.3f} vs {np.median(full):.3f} MeV ({int((incube < 0.05).sum())} decays "
          f"deposit <0.05 MeV inside the cube)")
    for lo, hi, name in [(0.50, 0.75, "570        [0.50-0.75]"),
                         (1.55, 1.80, "1064+570   [1.55-1.80]"),
                         (2.25, 2.50, "1770+570   [2.25-2.50]")]:
        nf = int(((full >= lo) & (full < hi)).sum()); ni = int(((incube >= lo) & (incube < hi)).sum())
        print(f"    {name} MeV: big-volume {nf:5d} -> in-cube {ni:5d}")
    print("  -> the cascade peaks drain into the low-energy blob the readout sees: the energy is deposited "
          "in LAr but OUTSIDE the ~30x48 cm cube, so no summing/clustering can recover it.")
    return uev, incube


def readout_vs_truth_incube(uev_tr, incube_mev, uev_ro, q_ro, slope, intercept, nadc, outdir):
    """Reconcile the cube-contained TRUTH energy with what the READOUT actually recorded. For the decays
    whose in-cube truth energy sits in the 570 / 1064 / 1633 bands, histogram the readout per-event TOTAL
    charge. If collection were perfect each band would stand at its edep_to_ke mark; instead the 1064/1633
    bands smear DOWN and broad because the delocalized gamma splits across pixels and the per-pixel
    fragments that fall below threshold are dropped (plus depth-dependent drift attenuation). THIS is why
    the cube contains ~18 ke- of 1064 in ~500 decays yet no sharp 1064 readout peak survives."""
    import matplotlib.pyplot as plt
    tmap = {int(e): float(v) for e, v in zip(uev_tr, incube_mev)}
    tro = np.array([tmap.get(int(e), np.nan) for e in uev_ro])
    bands = [("570  truth-in-cube [0.50-0.75]", 0.50, 0.75, "#1f6fb2"),
             ("1064 truth-in-cube [1.00-1.25]", 1.00, 1.25, "#d1495b"),
             ("1633 truth-in-cube [1.55-1.90]", 1.55, 1.90, "#6a3d9a")]
    emax = 42.0
    edges = adc_aligned_edges(0.0, emax, slope, intercept, nadc)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    plotted = False
    for name, lo, hi, c in bands:
        sel = np.isfinite(tro) & (tro >= lo) & (tro < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        plotted = True
        ax.hist(q_ro[sel], bins=edges, histtype="step", color=c, lw=1.8,
                label=f"{name} MeV: {n} decays, readout med {np.median(q_ro[sel]):.1f} ke-")
        mark = edep_to_ke(0.5 * (lo + hi))     # where full collection at ~full drift would put it
        ax.axvline(mark, color=c, ls="--", lw=1.1, alpha=0.7)
        ax.text(mark, ax.get_ylim()[1] * 0.98, f"{mark:.0f}", color=c, fontsize=7.5, ha="right", va="top")
    ax.set_xlim(0, emax)
    ax.set_xlabel("readout per-event TOTAL charge [ke-]")
    ax.set_ylabel(f"decays / {nadc} ADC count")
    ax.set_title("Readout per-event charge, conditioned on TRUTH cube-contained energy", fontsize=10)
    if plotted:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "bi207_readout_vs_truth_incube.png")
    fig.savefig(out, dpi=130); plt.close(fig); print(f"wrote {out}")
    for name, lo, hi, c in bands:
        sel = np.isfinite(tro) & (tro >= lo) & (tro < hi)
        if sel.any():
            print(f"  {name} MeV: {int(sel.sum())} decays -> readout median {np.median(q_ro[sel]):.1f}, "
                  f"mean {np.mean(q_ro[sel]):.1f} ke-  (perfect ~{edep_to_ke(0.5*(lo+hi)):.0f} ke-)")
    print("  -> the 1064/1633 truth-contained decays record FAR below their mark and broad: delocalized "
          "gamma -> multi-pixel split -> sub-threshold fragments dropped. The charge is lost per-pixel, "
          "not off the cube, for these -- so a lower threshold / pixel-sum helps THIS population (unlike the escapers).")


def anode_vs_fee(anode_by_ev, uev_ro, q_ro, uev_tr, incube_mev, slope, intercept, nadc, outdir):
    """TRUE charge reaching the anode per decay = sum of segments['n_electrons'] -- larnd-sim's OWN
    Birks recombination x drift attenuation, written back after quench+drift, BEFORE the FEE (no
    threshold, no ADC quantization, no induction integration). Compared with the FEE per-event readout
    this splits the shortfall into two gaps: truth->anode = the recombination model (my edep_to_ke is
    Box + fixed dE/dx; larnd uses Birks + real dE/dx, so the marks need not sit on the anode peaks), and
    anode->FEE = threshold/induction loss. If the anode spectrum shows a 1064 structure the FEE smears,
    the loss is the per-pixel threshold (recoverable); if the anode is ALSO smeared, it is the physics."""
    import matplotlib.pyplot as plt
    anode_ro = np.array([anode_by_ev.get(int(e), np.nan) for e in uev_ro])   # ke- at anode, hit decays
    all_anode = np.fromiter(anode_by_ev.values(), dtype=float)
    fin = np.isfinite(anode_ro)
    emax = 42.0
    edges = adc_aligned_edges(0.0, emax, slope, intercept, nadc)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    # SAME decays on both curves (the ones that fired): anode = charge before the FEE, FEE = after. The
    # gap between them is purely the threshold/induction loss, with the population mismatch removed.
    ax.hist(anode_ro[fin], bins=edges, histtype="stepfilled", color="#9ecae1", edgecolor="#4a7fb0",
            label=f"anode charge in-cube, same {int(fin.sum())} hit decays (pre-FEE)")
    ax.hist(q_ro, bins=edges, histtype="step", color="#b2182b", lw=1.7,
            label=f"FEE readout total (same {q_ro.size} decays)")
    for e_mev, nm in [(0.5697, "570"), (1.0637, "1064"), (1.6334, "1064+570")]:
        x = edep_to_ke(e_mev); ax.axvline(x, color="0.4", ls=":", lw=1.0)
        ax.text(x, ax.get_ylim()[1] * 0.98, f"{nm} {x:.0f} (Box)", rotation=90, va="top", ha="right",
                fontsize=7, color="0.3")
    ax.set_xlim(0, emax)
    ax.set_xlabel("charge per decay [ke-]")
    ax.set_ylabel(f"decays / {nadc} ADC count")
    ax.set_title("TRUE charge at the anode INSIDE the cube (pre-FEE n_electrons) vs FEE readout", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    out = os.path.join(outdir, "bi207_anode_charge.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    print(f"  anode charge (n_electrons, pre-FEE): all-decay median {np.median(all_anode):.1f} ke-, "
          f"max {all_anode.max():.1f}; FEE hit-decay median {np.median(q_ro):.1f} ke-")
    tmap = {int(e): float(v) for e, v in zip(uev_tr, incube_mev)}
    tro = np.array([tmap.get(int(e), np.nan) for e in uev_ro])
    for name, lo, hi in [("570 ", 0.50, 0.75), ("1064", 1.00, 1.25), ("1633", 1.55, 1.90)]:
        sel = np.isfinite(tro) & (tro >= lo) & (tro < hi) & np.isfinite(anode_ro)
        if sel.any():
            tmid = 0.5 * (lo + hi)
            print(f"  in-cube {name} [{lo}-{hi}] MeV ({int(sel.sum())}): truth~{edep_to_ke(tmid):.1f} "
                  f"-> anode {np.median(anode_ro[sel]):.1f} -> FEE {np.median(q_ro[sel]):.1f} ke- (medians)")
    print("  -> truth->anode gap = recombination (edep_to_ke Box vs larnd Birks); anode->FEE gap = "
          "threshold/induction. Tells us whether the 'misaligned' FEE Q is a FEE effect or my reference.")


def source_placement_scan(fsd_edep_path, detector, outdir):
    """Where should the source sit to contain the most cascade energy? The EDEP cloud was generated in a
    big (fully-containing) LAr volume, so we hold each decay's COMPLETE deposit cloud; re-translating it
    to a trial source point inside the cube and re-clipping to TPC_BORDERS gives that placement's
    containment WITHOUT re-running the sim. Compares cathode-center (current) to a long-face (top/bottom)
    mount, a short-face mount, and the geometric center (the mount-free ceiling). Prints energy retained
    and how many decays keep the full 1064+570 (1.633 MeV) cascade -- the handle on whether moving the
    source rescues a 1064 line, or whether the short transverse dimension caps it regardless."""
    import matplotlib.pyplot as plt
    try:
        with open_ro(fsd_edep_path) as he:
            if "segments" not in he:
                print("  (no 'segments' in _fsd edep -> placement scan unavailable)"); return
            seg = he["segments"]; sn = seg.dtype.names or ()
            def col(ax):
                if ax in sn:
                    return np.asarray(seg[ax][:], float)
                if f"{ax}_start" in sn and f"{ax}_end" in sn:
                    return 0.5 * (np.asarray(seg[f"{ax}_start"][:], float) + np.asarray(seg[f"{ax}_end"][:], float))
                return None
            ev = np.asarray(seg["event_id"][:]).astype(np.int64)
            dE = np.asarray(seg["dE"][:]).astype(np.float64)
            X, Y, Z = col("x"), col("y"), col("z")
    except Exception as exc:
        print(f"  (placement scan unavailable: {exc!r})"); return
    if X is None or Y is None or Z is None:
        print("  (translated segments lack x/y/z -> placement scan unavailable)"); return
    Bc = np.asarray(detector.TPC_BORDERS)[0].astype(float)
    lo = np.minimum(Bc[:, 0], Bc[:, 1]); hi = np.maximum(Bc[:, 0], Bc[:, 1])
    mid = 0.5 * (lo + hi); length = hi - lo
    cathode_z = float(Bc[2, 1])                                  # bi207_to_fsd mounts on z1 = cathode
    long_ax = int(np.argmax(length[:2])); short_ax = 1 - long_ax  # of the two plane axes 0,1
    cur = mid.copy(); cur[2] = cathode_z                          # current source: plane-centered, cathode
    def at(ax_set):                                              # build a target from {axis: value}
        t = mid.copy()
        for a, v in ax_set.items():
            t[a] = v
        return t
    cands = {
        "cathode center (current)": cur,
        f"long-face center ({length[long_ax]:.0f}cm axis)": at({long_ax: hi[long_ax]}),
        f"short-face center ({length[short_ax]:.0f}cm axis)": at({short_ax: hi[short_ax]}),
        "cube center (ideal ceiling)": mid.copy(),
    }
    coords = np.stack([X, Y, Z], axis=1)
    _, inv = np.unique(ev, return_inverse=True); nd = int(inv.max()) + 1
    emax = 2.6; bins = np.linspace(0.0, emax, 261)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    results = {}
    for name, tsrc in cands.items():
        shift = tsrc - cur
        c2 = coords + shift
        inside = np.all((c2 >= lo) & (c2 <= hi), axis=1)
        etot = np.zeros(nd); np.add.at(etot, inv, np.where(inside, dE, 0.0))
        retain = 100.0 * float(np.where(inside, dE, 0.0).sum()) / float(dE.sum())
        n1633 = int(((etot >= 1.55) & (etot < 1.80)).sum())
        results[name] = (retain, float(np.median(etot)), n1633)
        ax.hist(etot, bins=bins, histtype="step", lw=1.6,
                label=f"{name}: {retain:.0f}% kept, 1.633-band {n1633}")
    for e, lab in [(0.5697, "570"), (1.6334, "1064+570"), (2.3399, "1770+570")]:
        ax.axvline(e, color="0.6", ls=":", lw=0.9)
        ax.text(e, ax.get_ylim()[1] * 0.98, lab, rotation=90, va="top", ha="right", fontsize=7, color="0.4")
    ax.set_xlim(0, emax)
    ax.set_xlabel("in-cube TRUTH energy per decay [MeV]")
    ax.set_ylabel("decays")
    ax.set_title("Source-placement scan (sim-free re-translation of the truth cloud)", fontsize=9.5)
    ax.legend(fontsize=7.5); fig.tight_layout()
    out = os.path.join(outdir, "bi207_source_placement.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    for name, (r, m, n) in results.items():
        print(f"  {name:36s}: {r:4.1f}% energy kept, median {m:.3f} MeV, 1064+570 [1.55-1.80] {n} decays")
    print("  -> read the 1.633-band counts: if the long-face mount ~= cathode but the cube center jumps, "
          "the SHORT transverse dim caps containment and a surface mount can't beat it; if the long-face "
          "mount itself jumps, moving the source there genuinely helps.")


def main():
    ap = argparse.ArgumentParser(description="Decode the larnd-sim FEE (LARND) output to charge, fit "
                                             "the Bi-207 K-570 line, and map pixel coverage. No ndlar_flow.")
    ap.add_argument("larnd", help="larnd-sim output .h5 (…_fsd.LARND.h5)")
    ap.add_argument("--edep", default=None, help="EDEPSIM .h5 (trajectories) -> TRUTH-match the per-line "
                    "maps: tag each hit by its directly-emitted primary (conversion e- vs decay gamma), "
                    "instead of only its reconstructed charge window")
    ap.add_argument("--fsd-edep", default=None, help="TRANSLATED (_fsd) EDEPSIM .h5 (cube frame) for the "
                    "in-cube boundary clip; default = the LARND path with .LARND.->.EDEPSIM.")
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
        # TRUE charge reaching the anode (pre-FEE): larnd writes the quenched+drifted tracks back as
        # 'segments', so segments['n_electrons'] = Birks-recombined, drift-attenuated charge, no threshold.
        anode_by_ev = None
        if "segments" in h:
            sg = h["segments"]; sgn = sg.dtype.names or ()
            if "n_electrons" in sgn and "event_id" in sgn:
                def scol(ax):
                    if ax in sgn:
                        return np.asarray(sg[ax][:], float)
                    if f"{ax}_start" in sgn and f"{ax}_end" in sgn:
                        return 0.5 * (np.asarray(sg[f"{ax}_start"][:], float) + np.asarray(sg[f"{ax}_end"][:], float))
                    return None
                s_ne = np.asarray(sg["n_electrons"][:], float)
                s_ev = np.asarray(sg["event_id"][:]).astype(np.int64)
                sX, sY, sZ = scol("x"), scol("y"), scol("z")
                # Restrict to the fsd_cube READOUT FOOTPRINT -- same pixel-index + drift range larnd uses
                # (hit_pixel_ids / hit_drift_fraction) -- so the anode charge is comparable to the FEE and
                # NOT inflated by the half of the cascade that escapes the small cube.
                incube = np.ones(s_ne.size, bool)
                if sX is not None and sY is not None:
                    Bb = np.asarray(detector.TPC_BORDERS)[0]
                    pitch = float(detector.PIXEL_PITCH)
                    nx0, ny0 = int(detector.N_PIXELS[0]), int(detector.N_PIXELS[1])
                    i_x = np.floor((sX - float(Bb[0][0])) / pitch).astype(int)
                    i_y = np.floor((sY - float(Bb[1][0])) / pitch).astype(int)
                    incube &= (i_x >= 0) & (i_x < nx0) & (i_y >= 0) & (i_y < ny0)
                    if sZ is not None:
                        zlo, zhi = sorted((float(Bb[2][0]), float(Bb[2][1])))
                        incube &= (sZ >= zlo) & (sZ <= zhi)
                ne_in = np.where(incube, s_ne, 0.0)
                ue_a, inv_a = np.unique(s_ev, return_inverse=True)
                a_tot = np.zeros(ue_a.size); np.add.at(a_tot, inv_a, ne_in)
                anode_by_ev = {int(e): float(v) / 1.0e3 for e, v in zip(ue_a, a_tot)}
                av = np.fromiter(anode_by_ev.values(), dtype=float)
                print(f"anode charge from segments['n_electrons'] (INSIDE readout footprint): {av.size} decays, "
                      f"{100*incube.mean():.0f}% of segments in-cube, median {np.median(av):.1f} ke-, "
                      f"max {av.max():.1f} ke- (Birks recomb x drift, pre-FEE: no threshold)")

    # ---- per-decay sum + multiplicity ----
    per_evt_q = per_evt_n = uev = None
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

    # ---- TRUTH deposited energy per decay (readout-independent; the intrinsic Bi-207 containment) ----
    if args.edep:
        truth_dE_per_decay(args.edep, args.outdir)
        fsd_edep = args.fsd_edep or args.larnd.replace(".LARND.", ".EDEPSIM.")
        if fsd_edep != args.larnd and os.path.exists(fsd_edep):
            tr = truth_dE_incube(fsd_edep, detector, args.outdir)
            if tr is not None and per_evt_q is not None and uev is not None:
                uev_tr, incube_mev = tr
                readout_vs_truth_incube(uev_tr, incube_mev, uev, per_evt_q,
                                        slope, intercept, args.nadc, args.outdir)
                if anode_by_ev is not None:
                    anode_vs_fee(anode_by_ev, uev, per_evt_q, uev_tr, incube_mev,
                                 slope, intercept, args.nadc, args.outdir)
            source_placement_scan(fsd_edep, detector, args.outdir)
        else:
            print(f"  (translated _fsd EDEPSIM not found at {fsd_edep}; pass --fsd-edep for the in-cube clip)")

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
                Edep_emitted = emitted_deposited_energy(h2, args.edep) if origin is not None else {}
            if origin is not None and origin[0].shape == valid.shape:
                pdg0, E0, atid, aev = origin
                iY0, iZ0 = (Yctr - Y0) / pitch, (Zctr - Z0) / pitch   # source pixel in index space
                # EDEPSIM trajectory E_start is TOTAL energy (KE + rest mass); the line catalogue is
                # KINETIC, so subtract the e-/e+ rest mass (0.511 MeV) before matching. Gammas (massless)
                # are unchanged. Without this every electron line is mis-mapped by 0.511 MeV.
                Ek = np.where(np.abs(pdg0) == 11, E0 - 0.511, E0)
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

                line = np.array([match(int(pdg0[i]), float(Ek[i])) for i in range(pdg0.size)], dtype=object)
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
                fin = valid & np.isfinite(Ek)
                if int(fin.sum()) > 0:
                    eE = Ek[fin & (np.abs(pdg0) == 11)] * 1e3    # kinetic (rest mass removed)
                    gE = Ek[fin & (pdg0 == 22)] * 1e3
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

                edges = adc_aligned_edges(0.0, args.qmax, slope, intercept, args.nadc)
                def mark_lines(ax, compton=False):
                    for lab, E, k, _ in BI207_LINES:
                        ax.axvline(edep_to_ke(E), color="0.4", ls=":", lw=0.8)
                    if compton:
                        qce = edep_to_ke(0.3936)                 # 570 Compton edge (394 keV)
                        ax.axvline(qce, color="crimson", ls="--", lw=1.0)
                        ax.text(qce, ax.get_ylim()[1] * 0.5, "570 Compton edge", rotation=90,
                                va="center", ha="right", fontsize=7, color="crimson")
                def finish(ax, title, psum):
                    ax.set_xlim(0, args.qmax)
                    ax.set_title(title, fontsize=9)
                    ax.set_xlabel("Q per-pixel SUM [ke-]" if psum else "Q per pixel-hit [ke-]")
                    ax.set_ylabel(("pixel-deposits" if psum else "hits") + f" / {args.nadc} ADC count")
                    ax.legend(loc="upper right", fontsize=7, ncol=2)
                    fig.tight_layout()

                # Plot B: reconstructed CHARGE decomposed by truth QUANTUM -- which lines survive as
                # fittable peaks vs melt into the Compton continuum. Per-hit AND per-pixel-Q-sum (the
                # sum of the packets one decay drops on a pixel), so we can see whether summing sharpens
                # the high-energy (1064) region.
                qorder = ["K-570 CE", "570 gamma", "L-570 CE", "K-1064 CE", "1064 gamma", "L-1064 CE",
                          "1770 gamma", "Pb-K X-ray", "Compton/other"]
                qcols = {"K-570 CE": "#ee6677", "570 gamma": "#4477aa", "L-570 CE": "#f4a0a8",
                         "K-1064 CE": "#228833", "1064 gamma": "#66ccee", "L-1064 CE": "#99ddbb",
                         "1770 gamma": "#aa3377", "Pb-K X-ray": "#bbbbbb", "Compton/other": "0.78"}
                ls = line.astype(str)
                qlabel = np.full(valid.shape, "", dtype=object)
                qlabel[valid] = "Compton/other"                  # e-other / g-other / untraced -> gray
                for L in qorder[:-1]:
                    qlabel[valid & (ls == L)] = L
                for psum, tag in [(False, ""), (True, "_psum")]:
                    fig, ax = plt.subplots(figsize=(7.6, 4.9))
                    if psum:
                        vals, labs = pixel_sum_label(q_ke, ev, iY, iZ, valid, nx, ny, qlabel)
                    else:
                        vals, labs = q_ke[valid], qlabel[valid]
                    if stack_by_label(ax, vals, labs, qorder, qcols, edges):
                        mark_lines(ax)
                        finish(ax, "Collected charge by truth QUANTUM" + (" (per-pixel Q-sum)" if psum else ""), psum)
                        o = os.path.join(args.outdir, f"bi207_charge_by_truth{tag}.png")
                        fig.savefig(o, dpi=140); print(f"wrote {o}")
                    plt.close(fig)

                # Plot B2: measured Q by INTERACTION MECHANISM. Truth deposited energy (drift-independent)
                # splits gamma hits into FULL absorption (photopeak) vs PARTIAL / Compton escape, beside
                # the true conversion-electron lines; the untraced remainder (Compton e- whose gamma was
                # not saved as a trajectory, + soft deltas) is the gray 'Compton/other'. Per-hit AND
                # per-pixel-Q-sum. The gamma full/partial split is the robust part; CE tagging is indicative.
                if Edep_emitted:
                    edh = np.array([Edep_emitted.get((int(aev[i]), int(atid[i])), np.nan)
                                    for i in range(atid.size)])
                    isg = valid & (pdg0 == 22) & np.isfinite(edh) & np.isfinite(Ek)
                    is_ce = valid & np.isin(ls, np.array(["K-570 CE", "L-570 CE", "K-1064 CE", "L-1064 CE"]))
                    # Compton ESCAPE is negligible here (big 3D LAr volume contains the gamma via multi-
                    # site scatter), so the lower hump is NOT a Compton edge. 'untraced e-' = gamma e-
                    # whose parent gamma was not saved as a trajectory (NOT Compton escape).
                    proc = np.full(valid.shape, "", dtype=object)
                    proc[valid] = "untraced e-"
                    proc[is_ce] = "conv e- (CE line)"
                    proc[isg & (edh < 0.9 * Ek)] = "gamma Compton escape"
                    proc[isg & (edh >= 0.9 * Ek)] = "gamma fully absorbed"
                    porder = ["gamma fully absorbed", "gamma Compton escape", "conv e- (CE line)", "untraced e-"]
                    pcol = {"gamma fully absorbed": "#4477aa", "gamma Compton escape": "#88ccee",
                            "conv e- (CE line)": "#cc6677", "untraced e-": "0.8"}
                    for psum, tag in [(False, ""), (True, "_psum")]:
                        fig, ax = plt.subplots(figsize=(7.6, 4.9))
                        if psum:
                            vals, labs = pixel_sum_label(q_ke, ev, iY, iZ, valid, nx, ny, proc)
                        else:
                            vals, labs = q_ke[valid], proc[valid]
                        if stack_by_label(ax, vals, labs, porder, pcol, edges):
                            mark_lines(ax)
                            finish(ax, "Collected charge by INTERACTION MECHANISM (truth dE)"
                                   + (" (per-pixel Q-sum)" if psum else ""), psum)
                            o = os.path.join(args.outdir, f"bi207_charge_by_process{tag}.png")
                            fig.savefig(o, dpi=140); print(f"wrote {o}")
                        plt.close(fig)
                    lo_m = valid & (q_ke >= 5.0) & (q_ke < 7.5)      # lower hump
                    hi_m = valid & (q_ke >= 8.5) & (q_ke < 10.5)     # upper hump (570 per-pixel peak)
                    print("  double-peak decomposition -- per-hit mechanism counts in each hump:")
                    for p in porder:
                        if int((proc == p).sum()):
                            print(f"    {p:26s} lower[5-7.5] {int((lo_m & (proc == p)).sum()):6d}   "
                                  f"upper[8.5-10.5] {int((hi_m & (proc == p)).sum()):6d}")

                # Plot D: per-event SPATIAL-CLUSTER charge (sum over the 8-connected pixels one decay
                # lights) -- the cross-pixel energy sum, and the real test of whether the 1064 line is
                # recoverable. A 570 keV photo-absorption lands on ~1 pixel (short photo-e range) so it
                # already peaks per-pixel; a 1064 keV one makes a ~1 MeV photo-e that spreads charge over
                # 2-3 pixels, so per-pixel it FRAGMENTS and only the cluster sum rebuilds its 17.8 ke-
                # band. DIAGNOSTIC (truth-free clustering), NOT the per-pixel threshold-fit quantity.
                try:
                    from scipy import ndimage as _ndi
                    _have_ndi = True
                except Exception:
                    _have_ndi = False
                if _have_ndi and int(valid.sum()) > 0:
                    evv = ev[valid].astype(np.int64); iyv = iY[valid]; izv = iZ[valid]
                    qv = q_ke[valid]; lbv = qlabel[valid]
                    clid = np.full(qv.size, -1, np.int64); base = 0
                    struct = np.ones((3, 3), int)
                    for e in np.unique(evv):
                        sel = np.where(evv == e)[0]
                        img = np.zeros((nx, ny), bool); img[iyv[sel], izv[sel]] = True
                        lab, nlab = _ndi.label(img, structure=struct)
                        clid[sel] = base + lab[iyv[sel], izv[sel]]
                        base += nlab + 1
                    uc, cinv = np.unique(clid, return_inverse=True)
                    csum = np.zeros(uc.size); np.add.at(csum, cinv, qv)
                    clab = np.empty(uc.size, dtype=object)
                    for k in np.argsort(qv):
                        clab[cinv[k]] = lbv[k]
                    fig, ax = plt.subplots(figsize=(7.6, 4.9))
                    if stack_by_label(ax, csum, clab, qorder, qcols, edges):
                        mark_lines(ax)
                        ax.set_xlim(0, args.qmax); ax.set_title(
                            "Per-event SPATIAL-CLUSTER charge (cross-pixel energy sum, by quantum)", fontsize=9)
                        ax.set_xlabel("cluster Q [ke-]"); ax.set_ylabel(f"clusters / {args.nadc} ADC count")
                        ax.legend(loc="upper right", fontsize=7, ncol=2); fig.tight_layout()
                        o = os.path.join(args.outdir, "bi207_charge_by_cluster.png")
                        fig.savefig(o, dpi=140); print(f"wrote {o}")
                        b570 = int(((csum >= 7.5) & (csum < 11.0)).sum())
                        b1064 = int(((csum >= 14.0) & (csum < 20.0)).sum())
                        print(f"  cluster-sum: {uc.size} clusters from {int(valid.sum())} hits "
                              f"(adjacency merges little -- deposits are sparse/separated); "
                              f"bands 570[7.5-11] {b570}, 1064[14-20] {b1064}")
                    plt.close(fig)

                # Plot E: per-EVENT TOTAL charge -- the WHOLE readout summed over every pixel in a decay,
                # no cuts, tagged by the event's DOMINANT truth quantum. This is the cross-pixel sum with
                # NO clustering and NO per-pixel/per-decay cap (those cannot clip it). Detection efficiency
                # is low (median 1 hit/decay -- most gammas escape the small cube), so most events are a
                # single interacting quantum and this ~= the per-quantum energy. A 1064 peak at 17.8 ke-
                # appears ONLY for the fraction that fully absorbs AND stays above threshold; whatever the
                # scattered photon carries out of the cube is gone (physical, not a reconstruction limit).
                if int(valid.sum()) > 0:
                    evv = ev[valid]; qv = q_ke[valid]; lbv = qlabel[valid]
                    ue, einv = np.unique(evv, return_inverse=True)
                    etot = np.zeros(ue.size); np.add.at(etot, einv, qv)
                    elab = np.empty(ue.size, dtype=object); best = np.full(ue.size, -1.0)
                    for L in qorder:                              # dominant quantum = most summed charge
                        m = lbv == L
                        if not m.any():
                            continue
                        s = np.zeros(ue.size); np.add.at(s, einv[m], qv[m])
                        bett = s > best; best[bett] = s[bett]; elab[bett] = L
                    emax = max(args.qmax, 42.0)
                    edges_e = adc_aligned_edges(0.0, emax, slope, intercept, args.nadc)
                    fig, ax = plt.subplots(figsize=(7.6, 4.9))
                    if stack_by_label(ax, etot, elab, qorder, qcols, edges_e):
                        mark_lines(ax); ax.set_xlim(0, emax)
                        # where a FULLY-CONTAINED decay would pile (truth cascade sums -> ke-): shows the
                        # readout reaches nowhere near these, so it is footprint loss, not a short x-axis.
                        for e_mev, nm in [(0.5697, "570 full"), (1.6334, "1064+570 full"),
                                          (2.3399, "1770+570 full")]:
                            xq = edep_to_ke(e_mev)
                            if xq <= emax:
                                ax.axvline(xq, color="purple", ls="--", lw=1.0, alpha=0.7)
                                ax.text(xq, ax.get_ylim()[1]*0.60, f"{nm} ({xq:.0f})", rotation=90,
                                        va="top", ha="right", fontsize=7, color="purple")
                        ax.set_title("Per-event TOTAL charge (whole readout, no cuts) by dominant truth quantum", fontsize=9)
                        ax.set_xlabel("event total Q [ke-]"); ax.set_ylabel(f"events / {args.nadc} ADC count")
                        ax.legend(loc="upper right", fontsize=7, ncol=2); fig.tight_layout()
                        o = os.path.join(args.outdir, "bi207_charge_by_event_total.png")
                        fig.savefig(o, dpi=140); print(f"wrote {o}")
                        print(f"  per-event TOTAL ({ue.size} decays with hits): "
                              + ", ".join(f"{nm}[{lo}-{hi}] {int(((etot >= lo) & (etot < hi)).sum())}"
                                          for lo, hi, nm in [(7.5, 11.0, "570"), (15.0, 20.0, "1064"), (26.0, 33.0, "1770")]))
                    plt.close(fig)

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

                # ---- multiplicity (ALWAYS within ONE pixel -- the definition the threshold fits use):
                #      one charge deposit on a pixel can fire several PACKETS as the FEE crosses threshold
                #      and resets. multiplicity = number of packets ON THAT PIXEL; the pixel's TRUE charge is
                #      their SUM, and that single-pixel sum is the input to a per-pixel gain/threshold fit.
                #      (Charge landing on NEIGHBOURING pixels is a separate effect -- NOT multiplicity.)
                if ev is not None:
                    good2 = valid & (ev >= 0)
                    pixid = (iY * ny + iZ).astype(np.int64)
                    cell = ev.astype(np.int64) * 10000000 + pixid       # one pixel in one decay = one deposit
                    uc, cinv = np.unique(cell[good2], return_inverse=True)
                    mult = np.bincount(cinv)                            # packets on that pixel = MULTIPLICITY
                    cell_q = np.zeros(uc.size); np.add.at(cell_q, cinv, q_ke[good2])   # single-pixel summed Q
                    rep = np.full(uc.size, -1, np.int64)               # a representative hit per pixel-deposit
                    rep[cinv] = np.nonzero(good2)[0]
                    cell_line = line[rep]
                    print(f"within-pixel multiplicity: {int((mult > 1).sum())}/{uc.size} lit pixels fired >1 "
                          f"packet (max {int(mult.max())}); the per-pixel SUM is the charge the threshold fit uses.")
                    print(f"  {'line':12s} {'pixels':>6s} {'mult>1%':>7s} {'<mult>':>6s} {'1pktQ':>6s} {'sumQ':>6s}  line")
                    for lab, E, k, _ in BI207_LINES:
                        m = cell_line == lab
                        n = int(m.sum())
                        if n == 0:
                            continue
                        one = m & (mult == 1)
                        q1 = float(np.mean(cell_q[one])) if one.any() else float("nan")
                        print(f"  {lab:12s} {n:6d} {100*np.mean(mult[m] > 1):7.0f} {np.mean(mult[m]):6.2f} "
                              f"{q1:6.2f} {np.mean(cell_q[m]):6.2f}  {edep_to_ke(E):.1f}")
                    print("  -> <mult> ~1 and 1pktQ ~ sumQ => one packet per pixel (fit the hit directly);"
                          " <mult>>1 => sum each pixel's packets before the threshold fit.")
                    # spectrum: per-packet (single hit) Q vs per-pixel summed Q -- how multiplicity moves the lines
                    edges = adc_aligned_edges(0.0, args.qmax, slope, intercept, args.nadc)
                    fig, ax = plt.subplots(figsize=(7.2, 4.8))
                    ax.hist(q_ke[good2], bins=edges, histtype="step", color="#888888", lw=1.4,
                            label="per packet (single hit)")
                    ax.hist(cell_q, bins=edges, histtype="step", color="#cc3311", lw=1.9,
                            label="per pixel (its packets summed)")
                    for lab, E, k, _ in BI207_LINES:
                        ax.axvline(edep_to_ke(E), color="0.6", ls=":", lw=0.7)
                    ax.set_xlim(0, args.qmax)
                    ax.set_title("Per-packet vs per-pixel-summed charge (within-pixel multiplicity -> threshold-fit input)", fontsize=9)
                    ax.set_xlabel("Q [ke-]"); ax.set_ylabel(f"entries / {args.nadc} ADC count")
                    ax.legend(loc="upper right", fontsize=8)
                    fig.tight_layout()
                    o = os.path.join(args.outdir, "bi207_perpixel_sum.png")
                    fig.savefig(o, dpi=140); plt.close(fig); print(f"wrote {o}")
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
