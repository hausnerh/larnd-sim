#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_fee.py -- STAGE 3: does the simulated FRONT-END ELECTRONICS preserve the
# Bi-207 peaks? Reads the ndlar_flow FLOW file (the Bi-207 edep after simulate_pixels + flow) and
# makes the FEE analogues of the edep dE spectra -- the digitized per-hit charge Q (ke-) -- split by
# pixel hit-multiplicity (all / single-hit / 2+ / last-hit), plus the per-DECAY summed Q (the FEE
# analogue of "total dE per decay"), and the per-pixel hit-multiplicity (the hit-splitting probe).
#
# It overlays the PREDICTED charge of each Bi-207 line, propagated edep -> collected charge with
# larnd-sim's OWN model: n_e = (E/W_ion) * R_box(dE/dx) ; then lifetime attenuation over the full
# cathode->anode drift, exp(-t_drift/tau). Shown as a marker with a band from the dE/dx range,
# because recombination depends on dE/dx so a line is a smear, not a delta. These are APPROXIMATE
# locators ("is the single-hit Q peak near the K-1064 line?"), not a fit.
#
# FLOW schema (same as tests/fsdcube_hitq_datamc.py): events charge/events/data; hits via
# charge/events/ref/charge/calib_prompt_hits/{ref_region,ref}; hits in charge/calib_prompt_hits/data
# (fields z,y,Q[ke-],t_drift). No ADC event cut here (each event is one decay, not a cosmic shower).
#
# Usage (NERSC):
#   python tests/bi207/check_bi207_fee.py $ND_WORK/bi207_fsd.FLOW.hdf5 --outdir $ND_WORK/bi207_fee
# =============================================================================
import argparse
import os
import sys

import numpy as np

# Bi-207 conversion-electron KE lines (MeV) and the gamma full-energy (for the per-decay panel).
CE_LINES = {"K-570": 0.4817, "L-570": 0.5538, "K-1064": 0.9757, "L-1064": 1.0478, "K-1770": 1.682}
CASCADE_FULL = 1.633 + 0.088    # 1064+570 + Pb K X-ray, the per-decay full-energy deposit (MeV)

# larnd-sim physics (larndsim/consts/physics.py); detector values default to fsd_cube, CLI-overridable.
W_ION = 23.6e-6          # MeV per ionization electron
BOX_ALPHA, BOX_BETA = 0.93, 0.207     # Box recombination (BETA in (kV/cm)(g/cm^2)/MeV)
RHO_LAR = 1.38           # g/cm^3

# geometry/binning mirror tests/fsdcube_hitq_datamc.py so the Q axis matches the data/MC study.
PITCH = 0.372
TILE_Y0, TILE_Z0 = 133.92, 23.808
NX, NY = 128, 80


def pixel2id_abs(z, y):
    x0 = TILE_Z0 - (NX / 2 - 0.5) * PITCH
    y0 = TILE_Y0 - (NY / 2 - 0.5) * PITCH
    xr = np.round((np.asarray(z) - x0) / PITCH).astype(np.int64)
    yr = np.round((np.asarray(y) - y0) / PITCH).astype(np.int64)
    return xr + NX * yr


def box_recomb(dedx, efield):
    """Box-model recombination survival fraction for a given dE/dx (MeV/cm) and E field (kV/cm)."""
    xi = BOX_BETA * dedx / (RHO_LAR * efield)
    return np.log(BOX_ALPHA + xi) / xi


def predict_q(E_mev, efield, tau_us, drift_cm, vdrift, dedx):
    """Predicted collected charge (ke-) for depositing energy E at the cathode: ionization *
    recombination(dE/dx) * lifetime attenuation over the full drift."""
    n_e = (E_mev / W_ION) * box_recomb(dedx, efield)       # electrons after recombination
    atten = np.exp(-(drift_cm / vdrift) / tau_us)          # cathode source -> full drift
    return n_e * atten / 1000.0                            # ke-


def accumulate(path, t_field="t_drift", max_events=None):
    """Per-hit Q (ke-) + that hit's pixel-multiplicity, per-pixel multiplicity, per-pixel last-hit Q,
    and per-DECAY summed Q. Follows the charge/events -> calib_prompt_hits references (no h5flow)."""
    import h5py
    REFG = "charge/events/ref/charge/calib_prompt_hits"
    try:
        h = h5py.File(path, "r", locking=False)
    except TypeError:
        h = h5py.File(path, "r")
    q_hit, n_hit, m_pix, q_last, q_decay = [], [], [], [], []
    with h:
        nevt = int(h["charge/events/data"].shape[0])
        idx = range(nevt if not max_events else min(nevt, max_events))
        reg = h[f"{REFG}/ref_region"]
        r_start = np.asarray(reg["start"]); r_stop = np.asarray(reg["stop"])
        hit_of_ref = np.asarray(h[f"{REFG}/ref"][:, 1])
        hd = h["charge/calib_prompt_hits/data"]
        z_all = np.asarray(hd["z"]); y_all = np.asarray(hd["y"])
        q_all = np.asarray(hd["Q"]).astype(float); t_all = np.asarray(hd[t_field]).astype(float)
        for e in idx:
            st, sp = int(r_start[e]), int(r_stop[e])
            if sp <= st:
                continue
            hi = hit_of_ref[st:sp]
            pid = pixel2id_abs(z_all[hi], y_all[hi])
            q = q_all[hi]; t = t_all[hi]
            uniq, inv, counts = np.unique(pid, return_inverse=True, return_counts=True)
            q_hit.append(q)
            n_hit.append(counts[inv].astype(np.int16))
            m_pix.append(counts.astype(np.int16))
            o = np.lexsort((t, pid)); pid_s, q_s, t_s = pid[o], q[o], t[o]
            last = np.ones(pid_s.size, bool); last[:-1] = pid_s[:-1] != pid_s[1:]
            q_last.append(q_s[last])
            q_decay.append(q.sum())
    cat = lambda L, dt=float: np.concatenate(L) if L else np.zeros(0, dt)
    return (cat(q_hit), cat(n_hit, np.int16), cat(m_pix, np.int16),
            cat(q_last), np.asarray(q_decay, float))


def main():
    ap = argparse.ArgumentParser(description="FEE Q spectra for a Bi-207 fsd_cube flow file.")
    ap.add_argument("path", help="Bi-207 FLOW .hdf5 (after simulate_pixels + ndlar_flow)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--qmax", type=float, default=40.0, help="per-hit Q axis max (ke-)")
    ap.add_argument("--qbins", type=int, default=80)
    ap.add_argument("--efield", type=float, default=0.25, help="drift field kV/cm (fsd_cube)")
    ap.add_argument("--lifetime", type=float, default=1100.0, help="electron lifetime us (fsd_cube)")
    ap.add_argument("--drift", type=float, default=47.75, help="full drift length cm (cathode source)")
    ap.add_argument("--vdrift", type=float, default=0.10711, help="drift velocity cm/us (fsd_cube)")
    ap.add_argument("--dedx", type=float, default=2.5, help="nominal dE/dx MeV/cm for the recomb estimate")
    ap.add_argument("--dedx-band", type=float, nargs=2, default=(1.5, 4.0),
                    help="dE/dx range (MeV/cm) for the predicted-Q band")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()
    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.path)) or ".")
    os.makedirs(outdir, exist_ok=True)

    qh, nh, mp, ql, qdec = accumulate(args.path, max_events=args.max_events)
    if qh.size == 0:
        sys.exit("no hits found in the flow file (check the dataset paths / that flow ran).")

    # predicted line positions (ke-) + band from the dE/dx range
    pk = {}
    lo_dedx, hi_dedx = args.dedx_band
    for name, E in CE_LINES.items():
        c = predict_q(E, args.efield, args.lifetime, args.drift, args.vdrift, args.dedx)
        b = sorted(predict_q(E, args.efield, args.lifetime, args.drift, args.vdrift, d)
                   for d in (lo_dedx, hi_dedx))
        pk[name] = (E, c, b[0], b[1])
    full_c = predict_q(CASCADE_FULL, args.efield, args.lifetime, args.drift, args.vdrift, args.dedx)

    # text summary
    lines = [f"# file: {args.path}",
             f"# hits={qh.size}  decays-with-hits={qdec.size}  fired-pixels={mp.size}",
             f"# recomb Box(alpha={BOX_ALPHA}, beta={BOX_BETA}) @ E={args.efield} kV/cm, "
             f"dE/dx={args.dedx} MeV/cm; lifetime atten exp(-{args.drift/args.vdrift:.0f}us/"
             f"{args.lifetime:.0f}us)={np.exp(-(args.drift/args.vdrift)/args.lifetime):.3f}",
             "", "[predicted collected Q per Bi-207 line] (single pixel, full drift)",
             f"  {'line':8s} {'E[keV]':>8s} {'Q[ke-]':>8s}  band[ke-]"]
    for name, (E, c, blo, bhi) in pk.items():
        lines.append(f"  {name:8s} {E*1e3:8.1f} {c:8.1f}  {blo:.1f}-{bhi:.1f}")
    lines += [f"  {'FULL':8s} {CASCADE_FULL*1e3:8.1f} {full_c:8.1f}  (1064+570+PbK, per-decay)",
              "", "[measured Q medians]",
              f"  all hits        : median {np.median(qh):.2f} ke-  ({qh.size} hits)",
              f"  single-hit (n=1): median {np.median(qh[nh==1]) if (nh==1).any() else float('nan'):.2f}"
              f" ke-  ({int((nh==1).sum())} hits)",
              f"  2+-hit (n>=2)   : median {np.median(qh[nh>=2]) if (nh>=2).any() else float('nan'):.2f}"
              f" ke-  ({int((nh>=2).sum())} hits)",
              f"  per-decay sum   : median {np.median(qdec):.2f} ke-  ({qdec.size} decays)",
              f"  pixel hit mult  : mean {mp.mean():.2f}, frac single {np.mean(mp==1):.3f}"]
    txt = os.path.join(outdir, "bi207_fee.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    def mark_lines(a, keys, ymax):
        for name in keys:
            E, c, blo, bhi = pk[name]
            a.axvspan(blo, bhi, color="0.85", zorder=0)
            a.axvline(c, color="k", ls="--", lw=0.9)
            a.text(c, ymax, f" {name}", rotation=90, va="top", ha="left", fontsize=7)

    # Panel 1: per-hit Q by pixel multiplicity
    edges = np.linspace(0, args.qmax, args.qbins + 1)
    for sel, lab, col in [(nh >= 1, "all hits", "C0"), (nh == 1, "single-hit (n=1)", "C1"),
                          (nh >= 2, "2+-hit (n>=2)", "C2")]:
        ax[0].hist(qh[sel], bins=edges, histtype="step", lw=1.6, color=col, label=lab)
    mark_lines(ax[0], ["K-570", "K-1064", "L-1064"], ax[0].get_ylim()[1])
    ax[0].set_title("Per-hit Q by pixel multiplicity\n(dashed = predicted line Q, band = dE/dx range)")
    ax[0].set_xlabel("hit Q [ke$^-$]"); ax[0].set_ylabel("hits"); ax[0].legend(fontsize=8)

    # Panel 2: per-decay summed Q (FEE analogue of total dE per decay)
    de = np.linspace(0, max(args.qmax, float(np.percentile(qdec, 99.5)) if qdec.size else args.qmax), 120)
    ax[1].hist(qdec, bins=de, histtype="step", lw=1.6, color="C3")
    y1 = ax[1].get_ylim()[1]
    ax[1].axvline(full_c, color="k", ls="--", lw=1.0)
    ax[1].text(full_c, y1*0.98, f" full cascade ~{full_c:.0f} ke-", rotation=90, va="top", ha="left", fontsize=7)
    ax[1].set_title("Total collected Q per decay"); ax[1].set_xlabel("sum hit Q per decay [ke$^-$]")
    ax[1].set_ylabel("decays")

    # Panel 3: pixel hit-multiplicity (the hit-splitting probe)
    nm = int(min(mp.max(), 15)) if mp.size else 1
    ax[2].hist(mp, bins=np.arange(0.5, nm + 1.5), histtype="step", lw=1.6, color="C4")
    ax[2].set_title("Pixel hit multiplicity (hit splitting)")
    ax[2].set_xlabel("hits on a pixel per decay"); ax[2].set_ylabel("fired pixels")
    ax[2].set_xticks(range(1, nm + 1))

    fig.tight_layout()
    out = os.path.join(outdir, "bi207_fee.png")
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
