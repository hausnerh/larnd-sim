#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_larnd.py -- read the FEE output DIRECTLY from the larnd-sim LARND file and
# look for the Bi-207 calibration lines, bypassing ndlar_flow.
#
# WHY: the LARND file IS the front-end-electronics response. Each LArPix-v3 DATA packet
# (packet_type == detector.PACKET_TYPE == 1 for fsd_cube) carries a digitized ADC in 'dataword'.
# larnd-sim's fee.digitize() maps collected charge Q[e-] -> ADC via, after the mV/e units cancel and
# with V_CM handled:
#     ADC = floor( (Q*GAIN + (V_PEDESTAL - V_CM)) * ADC_COUNTS/((V_REF - V_CM)*ADC_SCALE_FACTOR)
#                  + ADC_OFFSET )
# i.e. ADC = intercept + slope*Q, a line. We invert it (adding 0.5 for the floor midpoint) to recover
# the per-pixel collected charge. For fsd_cube this is ADC ~= 289 + Q[ke-]. Constants come from the
# loaded fsd_cube config -- nothing hard-coded.
#
# This writes FOUR separate images (each its own file, so they can be used independently):
#   bi207_perhit_q.png     per-PIXEL-HIT charge: one entry per data packet (every hit).
#   bi207_perdecay_q.png   per-DECAY charge: summed across the pixels one decay (event_id) lit.
#   bi207_multiplicity.png pixel-hits per decay (how often a decay spreads across >1 pixel).
#   bi207_pixel_hitmap.png the pixel plane, colored by how many hits each pixel saw (coverage from
#                          THIS source placement -- the map you need to decide per-pixel calibration).
# plus bi207_larnd.txt (numbers). Charge histograms are binned on the native ADC grid (one bin per
# ADC count) so there is no bin/quantization aliasing.
#
# The ndlar_flow fsdcube reco chain is what normally calibrates these packets into calib_prompt_hits,
# but it chokes on this sparse, ~500 s-span single-source calibration MC (get_anchor_unix_ts hang;
# builder OOM). Reading the ADC straight from the LARND answers "does the simulated hardware preserve
# the Bi-207 peaks, and which pixels see them?" without that chain.
#
# Usage (NERSC, larnd conda env; login node is fine -- tiny, CPU-only, no container/GPU):
#   python tests/bi207/check_bi207_larnd.py $ND_WORK/bi207_fsd.LARND.h5 --outdir $ND_WORK/bi207_larnd
# =============================================================================
import argparse
import os
import sys

import numpy as np

# predicted COLLECTED charge of the Bi-207 handles at fsd_cube (0.25 kV/cm, tau 1100 us, ~full drift
# from a cathode source), from check_bi207_fee.py's Box-recombination + attenuation model [ke-]. These
# are BARE conversion-electron energies pushed through W_ion -> Box recombination -> drift attenuation;
# they exclude the coincident Pb K X-ray (~88 keV), which is why the observed K-570 peak sits a little
# ABOVE its 8 ke- guide. Treat them as ~0.5 ke- guides, not exact.
PRED_LINES = [
    (8.0,  "K-570 (482 keV)"),
    (16.0, "K-1064 (976 keV)"),
    (17.5, "L-1064 (1048 keV)"),
    (29.0, "full cascade (1.72 MeV)"),
]


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
    """Histogram bin edges aligned to the ADC grid: each bin spans exactly `nadc` ADC counts, so a
    single-hit charge (which is quantized to one ADC count, value at count+0.5) sits bin-centered and
    there is NO aliasing beat between bin width and the ~1 ke-/count quantization. Edges in ke-."""
    s = slope * 1.0e3                      # ADC counts per ke-
    a_lo = int(np.floor(intercept + s * qlo_ke))
    a_hi = int(np.ceil(intercept + s * qhi_ke))
    if a_hi <= a_lo:
        a_hi = a_lo + 1
    step = max(1, int(nadc))
    a = np.arange(a_lo, a_hi + step, step)
    return (a - intercept) / s             # ADC count -> ke- (lower edge of that count's cell)


def event_id_per_data_packet(h, data_mask):
    """Per-packet truth decay id straight from mc_packets_assn['event_ids'] (one (1,)i8 per packet,
    same length/order as 'packets'). Returns int64 aligned to packets[data_mask], or None."""
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
            print(f"  (mc_packets_assn len {ev.shape[0]} != packets {data_mask.shape[0]} "
                  f"-> grouping unavailable)")
            return None
        return ev[data_mask]
    except Exception as exc:
        print(f"  (per-decay grouping via mc_packets_assn unavailable: {exc!r})")
        return None


def load_pixel_index_map(pixel_layout_path):
    """Return (kmap, nx, ny, pitch_cm): kmap[chip*1000+channel] = (ix, iy) pixel indices on the plane.
    fsd_cube is a single tile, so (chip, channel) alone locate the pixel. None on any failure."""
    try:
        import yaml
        with open(pixel_layout_path) as f:
            lay = yaml.load(f, Loader=yaml.FullLoader)
        ccp = lay["chip_channel_to_position"]            # {chip*1000+channel: [ix, iy]}
        kmap = {int(k): (int(v[0]), int(v[1])) for k, v in ccp.items()}
        pos = np.array(list(kmap.values()), dtype=np.int64)
        nx = int(pos[:, 0].max()) + 1
        ny = int(pos[:, 1].max()) + 1
        pitch = float(lay.get("pixel_pitch", 3.72)) / 10.0   # mm -> cm
        return kmap, nx, ny, pitch
    except Exception as exc:
        print(f"  (pixel-plane map unavailable: {exc!r} -> skipping the hit heatmap)")
        return None


def pixel_indices(h, data_mask, kmap):
    """Map each data packet to (ix, iy) via (chip_id, channel_id). Returns (ix, iy) int64 arrays with
    -1 where the packet has no field or an unmapped chip/channel. None if the fields are absent."""
    p = np.asarray(h["packets"][:])
    names = p.dtype.names or ()
    chip_f = next((f for f in ("chip_id", "chipid", "chip") if f in names), None)
    chan_f = next((f for f in ("channel_id", "channelid", "channel") if f in names), None)
    if chip_f is None or chan_f is None:
        print(f"  (packets lack chip/channel fields; have {names} -> skipping the hit heatmap)")
        return None
    chip = p[chip_f][data_mask].astype(np.int64)
    chan = p[chan_f][data_mask].astype(np.int64)
    key = chip * 1000 + chan
    ix = np.full(key.size, -1, dtype=np.int64)
    iy = np.full(key.size, -1, dtype=np.int64)
    for i, k in enumerate(key):                           # N ~ 1500, a plain loop is plenty
        pos = kmap.get(int(k))
        if pos is not None:
            ix[i], iy[i] = pos
    return ix, iy


def _draw_lines(ax, qmax):
    for q, lab in PRED_LINES:
        if q <= qmax:
            ax.axvline(q, color="crimson", ls="--", lw=1)
            ax.text(q, ax.get_ylim()[1] * 0.97, lab, rotation=90, va="top", ha="right",
                    fontsize=7, color="crimson")


def main():
    ap = argparse.ArgumentParser(description="Decode the larnd-sim FEE (LARND) output to charge and "
                                             "look for the Bi-207 lines + pixel coverage, no ndlar_flow.")
    ap.add_argument("larnd", help="larnd-sim output .h5 (…_fsd.LARND.h5)")
    ap.add_argument("--outdir", default="bi207_larnd", help="output directory for the pngs + txt")
    ap.add_argument("--qmax", type=float, default=45.0, help="charge-axis max [ke-]")
    ap.add_argument("--nadc", type=int, default=1, help="ADC counts per charge-histogram bin (>=1)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    detector, pixel_layout_path = load_fsd_cube_config()
    slope, intercept, data_type = adc_linear(detector)
    print(f"fsd_cube FEE: ADC = {intercept:.3f} + {slope*1e3:.4f} * Q[ke-]   "
          f"(pedestal ADC {intercept:.1f}; ~{1.0/(slope*1e3):.2f} ke- per ADC count; data packet_type={data_type})")

    kmap_info = load_pixel_index_map(pixel_layout_path)

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
        # unclipped inverse (keep the exact ADC grid; sub-pedestal hits show slightly < 0 ke- rather
        # than piling artificially at 0). floor-midpoint +0.5 centers each ADC count's value.
        q_ke = (adc + 0.5 - intercept) / (slope * 1.0e3)
        ev = event_id_per_data_packet(h, dmask)
        pix = pixel_indices(h, dmask, kmap_info[0]) if kmap_info is not None else None

    n_subped = int((q_ke < 0).sum())

    # ---- per-decay sum + multiplicity (truth grouping, if available) ----
    per_evt_q = per_evt_n = None
    if ev is not None and (ev >= 0).any():
        good = ev >= 0
        uev, inv = np.unique(ev[good], return_inverse=True)
        per_evt_q = np.zeros(uev.size)
        np.add.at(per_evt_q, inv, q_ke[good])
        per_evt_n = np.bincount(inv, minlength=uev.size)
        print(f"grouped {int(good.sum())} hits into {uev.size} decays "
              f"(median {np.median(per_evt_n):.0f} hits/decay, max {per_evt_n.max()})")

    # ---- plots (one file each) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_hist(values, edges, title, xlabel, ylabel, color, fname, lines=True):
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        ax.hist(values, bins=edges, histtype="stepfilled", color=color, alpha=0.9, edgecolor=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(0.0, float(edges[0])), args.qmax)
        if lines:
            _draw_lines(ax, args.qmax)
        fig.tight_layout()
        out = os.path.join(args.outdir, fname)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    # per-pixel-hit charge (every data packet)
    edges_hit = adc_aligned_edges(min(0.0, q_ke.min()), args.qmax, slope, intercept, args.nadc)
    save_hist(q_ke, edges_hit,
              f"Per-pixel-hit collected charge   (N={ndata} hits)",
              "Q per pixel-hit [ke-]", f"hits / {args.nadc} ADC count",
              "#4477aa", "bi207_perhit_q.png")

    # per-decay summed charge (sum across the decay's pixels)
    if per_evt_q is not None:
        edges_dec = adc_aligned_edges(min(0.0, per_evt_q.min()), args.qmax, slope, intercept, args.nadc)
        save_hist(per_evt_q, edges_dec,
                  f"Per-decay charge, summed across its pixels   (N={per_evt_q.size} decays)",
                  "sum Q over the decay's pixel-hits [ke-]", f"decays / {args.nadc} ADC count",
                  "#228833", "bi207_perdecay_q.png")

        # hit multiplicity per decay
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

    # pixel-plane hit heatmap
    n_pix_hit = frac_plane = None
    if pix is not None and kmap_info is not None:
        _, nx, ny, pitch = kmap_info
        ix, iy = pix
        valid = (ix >= 0) & (iy >= 0)
        H = np.zeros((nx, ny), dtype=np.int64)
        np.add.at(H, (ix[valid], iy[valid]), 1)
        n_pix_hit = int((H > 0).sum())
        frac_plane = n_pix_hit / float(nx * ny)
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        # imshow with x = pixel-plane x, y = pixel-plane y; H is [ix, iy] so transpose for (row=y,col=x)
        im = ax.imshow(H.T, origin="lower", aspect="equal",
                       extent=[0, nx * pitch, 0, ny * pitch], cmap="viridis")
        ax.set_title(f"Pixel-plane hit coverage   ({n_pix_hit}/{nx*ny} pixels hit, {100*frac_plane:.1f}%)")
        ax.set_xlabel("pixel-plane x [cm]")
        ax.set_ylabel("pixel-plane y [cm]")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("hits on the pixel")
        # mark the plane center (where a centered cathode source projects)
        ax.plot(nx * pitch / 2, ny * pitch / 2, marker="+", ms=12, mew=1.5, color="crimson")
        fig.tight_layout()
        out = os.path.join(args.outdir, "bi207_pixel_hitmap.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")
        hit_counts = H[H > 0]
        print(f"pixel coverage: {n_pix_hit}/{nx*ny} pixels hit ({100*frac_plane:.1f}%); "
              f"hits/hit-pixel median {np.median(hit_counts):.0f}, max {int(hit_counts.max())}")

    # ---- text summary ----
    txt = os.path.join(args.outdir, "bi207_larnd.txt")
    with open(txt, "w") as f:
        f.write("Bi-207 FEE response decoded directly from the LARND (larnd-sim FEE output)\n")
        f.write(f"input: {args.larnd}\n")
        f.write(f"ADC = {intercept:.4f} + {slope:.6e} * Q[e-]   (data packet_type {data_type}; "
                f"{args.nadc} ADC count/bin)\n")
        f.write(f"data packets (hits): {ndata}   (sub-pedestal, Q<0: {n_subped})\n")
        f.write(f"per-hit Q[ke-]: min {q_ke.min():.2f}  median {np.median(q_ke):.2f}  "
                f"max {q_ke.max():.2f}  mean {q_ke.mean():.2f}\n")
        for lo, hi, lab in [(6, 10, "K-570 ~8"), (14, 19, "K/L-1064 ~16-18"), (26, 32, "cascade ~29")]:
            n = int(((q_ke >= lo) & (q_ke < hi)).sum())
            f.write(f"  hits in [{lo},{hi}) ke- ({lab}): {n}\n")
        if per_evt_q is not None:
            f.write(f"\nper-decay summed Q[ke-]: median {np.median(per_evt_q):.2f}  max {per_evt_q.max():.2f}\n")
            f.write(f"hit multiplicity/decay: median {np.median(per_evt_n):.0f}  max {int(per_evt_n.max())}  "
                    f"frac single-pixel {np.mean(per_evt_n == 1):.2f}\n")
            for lo, hi, lab in [(6, 10, "K-570 ~8"), (14, 19, "K/L-1064 ~16-18"), (26, 32, "cascade ~29")]:
                n = int(((per_evt_q >= lo) & (per_evt_q < hi)).sum())
                f.write(f"  decays with sum Q in [{lo},{hi}) ke- ({lab}): {n}\n")
        if n_pix_hit is not None:
            f.write(f"\npixel-plane coverage (this source placement): {n_pix_hit} pixels hit "
                    f"({100*frac_plane:.1f}% of the plane)\n")
    print(f"wrote {txt}")
    print("\nFour images written: per-hit Q, per-decay Q, multiplicity, pixel hitmap. The hitmap shows\n"
          "which pixels this source placement illuminates -- i.e. how many could be calibrated per-pixel\n"
          "from one position, and where you'd need to move/raster the source to cover the rest.")


if __name__ == "__main__":
    main()
