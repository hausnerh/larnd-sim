#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207.py -- validate a Bi-207 edep-sim generation BEFORE investing in
# the larnd-sim/FLOW/calibration chain. Reads the dumpTree HDF5 (segments [+ trajectories])
# and checks that Geant4's /gps/ion 83 207 + RadioactiveDecay actually produced the Bi-207
# cascade, by looking for the known lines and reporting the deposit spectrum + time range.
#
# Validates three things:
#   1) DECAY (trajectories): the primary electrons should pile up at the internal-conversion
#      KE lines and the photons at the gamma lines -> proof the generator is correct.
#   2) DEPOSITS (segments): per-decay total dE and per-segment dE -- what larnd-sim will read.
#   3) TIME (segments t/t0): Bi-207 T1/2=31.55 yr; analogue radioactive decay can stamp
#      products at astronomically large times. If t is huge, see the GOTCHA in bi207_sim.sh
#      (we reset t in STAGE 2; just confirm deposits EXIST and energies are right here).
#
# Bi-207 reference (keV): conversion-electron KE  481.7 553.8 565.8 | 975.7 1047.8 1059.8 | 1682
#                         gamma energies          569.7             | 1063.7              | 1770.2
#
# Usage:
#   python tests/bi207/check_bi207.py bi207.EDEPSIM.h5 --outdir bi207_check
# =============================================================================
import argparse
import os
import sys

import numpy as np

ME = 0.510999  # MeV, electron rest mass (edep-sim trajectory E_start may be total E or KE)
E_LINES = [0.4817, 0.5538, 0.5658, 0.9757, 1.0478, 1.0598, 1.682]   # conversion-e- KE (MeV)
G_LINES = [0.5697, 1.0637, 1.7702]                                  # gamma energies (MeV)
WIN = 0.015  # MeV, +/- window for line-counting


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _get(h, name):
    return np.asarray(h[name][:]) if name in h else None


def line_counts(e, lines, label, offset=0.0):
    """Fraction/count of `e` within +/-WIN of each line (optionally shifted by `offset`)."""
    print(f"  {label} line occupancy (+/-{WIN*1e3:.0f} keV, energies shifted by {offset*1e3:+.0f} keV):")
    tot = e.size or 1
    for L in lines:
        c = int(np.sum(np.abs(e - (L + offset)) <= WIN))
        bar = "#" * min(40, int(40 * c / tot / 0.1))  # scale bar to ~10% full
        print(f"    {L*1e3:7.1f} keV : {c:7d} ({100*c/tot:5.2f}%) {bar}")


def main():
    ap = argparse.ArgumentParser(description="Validate a Bi-207 edep-sim generation.")
    ap.add_argument("path", help="dumpTree edep-sim HDF5 (segments [+ trajectories])")
    ap.add_argument("--outdir", default=None, help="where to save plots (default: alongside input)")
    ap.add_argument("--emax", type=float, default=2.0, help="energy-axis max (MeV)")
    args = ap.parse_args()

    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.path)) or ".")
    os.makedirs(outdir, exist_ok=True)

    with open_ro(args.path) as h:
        print(f"# file: {args.path}")
        print(f"# datasets: {list(h.keys())}")
        seg = _get(h, "segments")
        trj = _get(h, "trajectories")

    # ---- 1) DECAY validation from trajectories ----
    e_e = g_e = None
    if trj is not None and trj.size:
        pdg = trj["pdg_id"]
        Es = trj["E_start"].astype(float)   # MeV; convention (KE vs total) checked below
        e_e = Es[pdg == 11]
        g_e = Es[pdg == 22]
        print(f"\n[1] DECAY  (trajectories={trj.size}: e-={e_e.size}, gamma={g_e.size})")
        if e_e.size:
            # edep-sim E_start may be TOTAL energy; conversion lines are KE. Try both and
            # report whichever window catches them (the non-empty one is the convention).
            print("  electrons -- if the KE rows are empty but the +511 rows fill, E_start is TOTAL energy:")
            line_counts(e_e, E_LINES, "e- as-KE", offset=0.0)
            line_counts(e_e, E_LINES, "e- as-total(+me)", offset=ME)
        if g_e.size:
            line_counts(g_e, G_LINES, "gamma", offset=0.0)
    else:
        print("\n[1] DECAY: no 'trajectories' dataset (run dumpTree.py with --keep_all_dets to keep it).")

    # ---- 2) DEPOSITS + 3) TIME from segments ----
    if seg is None or not seg.size:
        sys.exit("\n[2] no 'segments' dataset -- edep produced no energy deposits "
                 "(check the HALF-LIFE/time gotcha in bi207_sim.sh).")
    dE = seg["dE"].astype(float)
    ev = seg["event_id"]
    nev = int(np.unique(ev).size)
    per_ev = np.array([dE[ev == e].sum() for e in np.unique(ev)])
    tcol = "t" if "t" in seg.dtype.names else ("t0" if "t0" in seg.dtype.names else None)
    print(f"\n[2] DEPOSITS  (segments={seg.size}, decays with deposits={nev})")
    print(f"    total dE/decay: median={np.median(per_ev):.3f} MeV  max={per_ev.max():.3f} MeV")
    print(f"    per-segment dE: median={np.median(dE)*1e3:.1f} keV")
    if tcol is not None:
        t = seg[tcol].astype(float)
        print(f"[3] TIME  ({tcol}): min={t.min():.3g}  max={t.max():.3g}  "
              f"(us if sane; HUGE => half-life gotcha, reset in STAGE 2)")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    edges = np.linspace(0, args.emax, 200)
    if e_e is not None and e_e.size:
        ax[0, 0].hist(e_e, bins=edges, histtype="step", color="C0")
        for L in E_LINES:
            ax[0, 0].axvline(L, color="r", ls=":", lw=0.8)
            ax[0, 0].axvline(L + ME, color="orange", ls=":", lw=0.6)
        ax[0, 0].set_title("primary e- E_start (red=KE lines, orange=+me)")
        ax[0, 0].set_xlabel("MeV")
    if g_e is not None and g_e.size:
        ax[0, 1].hist(g_e, bins=edges, histtype="step", color="C2")
        for L in G_LINES:
            ax[0, 1].axvline(L, color="r", ls=":", lw=0.8)
        ax[0, 1].set_title("primary gamma E_start (red=gamma lines)")
        ax[0, 1].set_xlabel("MeV")
    ax[1, 0].hist(per_ev, bins=np.linspace(0, max(args.emax, per_ev.max()), 200),
                  histtype="step", color="C3")
    ax[1, 0].set_title("total deposited dE per decay"); ax[1, 0].set_xlabel("MeV")
    ax[1, 1].hist(dE * 1e3, bins=200, histtype="step", color="C4")
    ax[1, 1].set_title("per-segment dE"); ax[1, 1].set_xlabel("keV"); ax[1, 1].set_yscale("log")
    fig.tight_layout()
    out = os.path.join(outdir, "bi207_check.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
