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
# the per-pixel collected charge. For fsd_cube this is ADC ~= 289 + Q[ke-], so a hit's charge in ke-
# is roughly (ADC - 289). All constants are read from the loaded fsd_cube config -- nothing hard-coded.
#
# The ndlar_flow fsdcube reco chain (event building / hit building / calibration) is what normally
# turns these packets into calibrated calib_prompt_hits, but it chokes on this sparse, ~500 s-span
# single-source calibration MC (get_anchor_unix_ts hang; builder OOM). Reading the ADC straight from
# the LARND answers "does the simulated hardware preserve the Bi-207 peaks?" without that chain.
#
# Usage (NERSC, larnd conda env; login node is fine -- tiny, CPU-only, no container/GPU):
#   python tests/bi207/check_bi207_larnd.py $ND_WORK/bi207_fsd.LARND.h5 --outdir $ND_WORK/bi207_larnd
# =============================================================================
import argparse
import os
import sys

import numpy as np

# predicted COLLECTED charge of the Bi-207 handles at fsd_cube (0.25 kV/cm, tau 1100 us, ~full drift
# from a cathode source), from check_bi207_fee.py's box-recombination + attenuation model [ke-].
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


def load_fsd_cube_detector():
    """Load the fsd_cube config and return the detector consts module (GAIN, V_REF, ... populated)."""
    from larndsim import consts
    from larndsim.config import get_config
    from larndsim.consts import detector
    cfg = get_config("fsd_cube")

    def one(k):
        v = cfg.get(k)
        return v[0] if isinstance(v, list) else v

    consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"),
                           one("RESPONSE"), one("SIM_PROPERTIES"))
    return detector


def adc_linear(detector):
    """ADC = intercept + slope*Q[e-]  (mV cancels; e == 1). Returns (slope, intercept, data_pkt_type)."""
    denom = (detector.V_REF - detector.V_CM) * detector.ADC_SCALE_FACTOR
    slope = detector.GAIN * detector.ADC_COUNTS / denom
    intercept = (detector.V_PEDESTAL - detector.V_CM) * detector.ADC_COUNTS / denom + detector.ADC_OFFSET
    return float(slope), float(intercept), int(detector.PACKET_TYPE)


def event_id_per_data_packet(h, data_mask):
    """Per-packet truth decay id straight from mc_packets_assn['event_ids'].

    larnd-sim writes mc_packets_assn (one compound row per packet, same length/order as 'packets')
    with an 'event_ids' field of shape (1,): the generating event (decay) id of that packet. We read
    it directly -- no join through the 'segments'/'tracks' table -- and mask to the data packets.
    Returns an int64 array aligned to packets[data_mask], or None if the field is not present.
    """
    try:
        if "mc_packets_assn" not in h:
            print("  (no mc_packets_assn in the file -> per-decay grouping unavailable)")
            return None
        assn = h["mc_packets_assn"]
        names = getattr(assn.dtype, "names", None) or ()
        # This larnd-sim stores the per-packet decay id directly as 'event_ids'. Older/other builds
        # may call it 'event_id'; accept either. If neither is present, say so (don't guess a join).
        field = next((f for f in ("event_ids", "event_id") if f in names), None)
        if field is None:
            print(f"  (mc_packets_assn has no event-id field; fields={names} -> grouping unavailable)")
            return None
        ev = np.asarray(assn[field][:])                 # (Npackets, 1) or (Npackets,)
        ev = ev.reshape(ev.shape[0], -1)[:, 0].astype(np.int64)
        if ev.shape[0] != data_mask.shape[0]:
            print(f"  (mc_packets_assn len {ev.shape[0]} != packets {data_mask.shape[0]} "
                  f"-> grouping unavailable)")
            return None
        return ev[data_mask]
    except Exception as exc:
        print(f"  (per-decay grouping via mc_packets_assn unavailable: {exc!r})")
        return None


def main():
    ap = argparse.ArgumentParser(description="Decode the larnd-sim FEE (LARND) output to charge and "
                                             "look for the Bi-207 lines, without ndlar_flow.")
    ap.add_argument("larnd", help="larnd-sim output .h5 (…_fsd.LARND.h5)")
    ap.add_argument("--outdir", default="bi207_larnd", help="output directory for png + txt")
    ap.add_argument("--qmax", type=float, default=45.0, help="charge-axis max [ke-]")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    detector = load_fsd_cube_detector()
    slope, intercept, data_type = adc_linear(detector)
    print(f"fsd_cube FEE: ADC = {intercept:.3f} + {slope*1e3:.4f} * Q[ke-]   "
          f"(pedestal ADC {intercept:.1f}; ~{1.0/(slope*1e3):.2f} ke- per ADC count; data packet_type={data_type})")

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
        q_ke = np.maximum((adc + 0.5 - intercept) / slope, 0.0) / 1.0e3   # e- -> ke-
        ev = event_id_per_data_packet(h, dmask)

    # ---- per-event sum + multiplicity (truth grouping, if available) ----
    per_evt_q = per_evt_n = None
    if ev is not None and (ev >= 0).any():
        good = ev >= 0
        uev, inv = np.unique(ev[good], return_inverse=True)
        per_evt_q = np.zeros(uev.size)
        np.add.at(per_evt_q, inv, q_ke[good])
        per_evt_n = np.bincount(inv, minlength=uev.size)
        print(f"grouped {int(good.sum())} hits into {uev.size} events "
              f"(median {np.median(per_evt_n):.0f} hits/event, max {per_evt_n.max()})")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npan = 3 if per_evt_q is not None else 1
    fig, axes = plt.subplots(1, npan, figsize=(6.2 * npan, 4.6))
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, args.qmax, 90)

    def draw_lines(ax):
        for q, lab in PRED_LINES:
            if q <= args.qmax:
                ax.axvline(q, color="crimson", ls="--", lw=1)
                ax.text(q, ax.get_ylim()[1] * 0.96, lab, rotation=90, va="top", ha="right",
                        fontsize=7, color="crimson")

    ax = axes[0]
    ax.hist(q_ke, bins=bins, histtype="stepfilled", color="#4477aa", alpha=0.85)
    ax.set_title(f"per-hit collected charge  (N={ndata})")
    ax.set_xlabel("Q per pixel-hit [ke-]"); ax.set_ylabel("hits")
    draw_lines(ax)

    if per_evt_q is not None:
        ax = axes[1]
        ax.hist(per_evt_q, bins=bins, histtype="stepfilled", color="#228833", alpha=0.85)
        ax.set_title(f"per-decay summed charge  (N={per_evt_q.size})")
        ax.set_xlabel("sum Q over the decay's hits [ke-]"); ax.set_ylabel("decays")
        draw_lines(ax)

        ax = axes[2]
        mx = int(max(1, per_evt_n.max()))
        ax.hist(per_evt_n, bins=np.arange(0.5, mx + 1.5), histtype="stepfilled", color="#aa7744", alpha=0.85)
        ax.set_title("hit multiplicity per decay")
        ax.set_xlabel("pixel-hits in the decay"); ax.set_ylabel("decays")

    fig.suptitle("Bi-207 FEE response (decoded straight from the LARND, no ndlar_flow)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(args.outdir, "bi207_larnd.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")

    # ---- text summary ----
    txt = os.path.join(args.outdir, "bi207_larnd.txt")
    with open(txt, "w") as f:
        f.write("Bi-207 FEE response decoded directly from the LARND (larnd-sim FEE output)\n")
        f.write(f"input: {args.larnd}\n")
        f.write(f"ADC = {intercept:.4f} + {slope:.6e} * Q[e-]   (data packet_type {data_type})\n")
        f.write(f"data packets (hits): {ndata}\n")
        f.write(f"per-hit Q[ke-]: min {q_ke.min():.2f}  median {np.median(q_ke):.2f}  "
                f"max {q_ke.max():.2f}  mean {q_ke.mean():.2f}\n")
        for lo, hi, lab in [(6, 10, "K-570 ~8"), (14, 19, "K/L-1064 ~16-18"), (26, 32, "cascade ~29")]:
            n = int(((q_ke >= lo) & (q_ke < hi)).sum())
            f.write(f"  hits in [{lo},{hi}) ke- ({lab}): {n}\n")
        if per_evt_q is not None:
            f.write(f"\nper-decay summed Q[ke-]: median {np.median(per_evt_q):.2f}  max {per_evt_q.max():.2f}\n")
            f.write(f"hit multiplicity/decay: median {np.median(per_evt_n):.0f}  max {int(per_evt_n.max())}  "
                    f"frac single-hit {np.mean(per_evt_n == 1):.2f}\n")
            for lo, hi, lab in [(6, 10, "K-570 ~8"), (14, 19, "K/L-1064 ~16-18"), (26, 32, "cascade ~29")]:
                n = int(((per_evt_q >= lo) & (per_evt_q < hi)).sum())
                f.write(f"  decays with sum Q in [{lo},{hi}) ke- ({lab}): {n}\n")
    print(f"wrote {txt}")
    print("\nRead the per-hit panel for whether charge lands at the predicted lines; the per-decay\n"
          "panel sums split hits back together (the hit-multiplicity panel shows how often a decay\n"
          "is spread across >1 pixel at the 3.72 mm pitch).")


if __name__ == "__main__":
    main()
