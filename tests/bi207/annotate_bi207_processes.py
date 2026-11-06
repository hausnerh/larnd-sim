#!/usr/bin/env python3
# =============================================================================
# tests/bi207/annotate_bi207_processes.py -- decompose the Bi-207 deposited-energy
# spectra by the PROCESS that produced each deposit, so the peaks in the total-dE and
# per-segment-dE distributions can be read off by origin instead of guessed.
#
# HOW IT ATTRIBUTES A DEPOSIT (truth, not fitting):
#   Every ionizing deposit is made by an electron. We walk that electron's trajectory
#   parentage UP (through delta rays and Compton-scatter chains) until we reach the first
#   particle that is a DIRECT DAUGHTER of the decaying nucleus -- i.e. a de-excitation
#   gamma (pdg 22, E in {569.7, 1063.7, 1770.2} keV) or a conversion/Auger electron
#   (pdg 11, emitted off the ion). That root quantum labels the deposit:
#       - gamma 1063.7 keV  -> Compton/photoelectric electrons it liberated
#       - gamma  569.7 keV  -> ditto
#       - gamma 1770.2 keV  -> ditto (weak crossover)
#       - conversion e- near a K/L/M line of the 1064 or 570 transition
#       - Auger / atomic e- (low-KE electron off the ion/atom; Pb K X-ray ~75-88 keV)
#   This is the SAME per-event parentage keying that check_bi207.py already validated
#   (parent_id references traj_id within each event: event_id*1e9 + traj_id).
#
# OUTPUT (bi207_processes.png):
#   (A) total dE per DECAY, stacked by decay topology (which transition converted), with
#       dotted lines at the clean nuclear energies 0.570 / 1.064 / 1.633 MeV. A conversion
#       decay deposits that transition's full energy locally -> it feeds the full peak;
#       a gamma that escapes removes its energy -> the escape shoulders.
#   (B) per-segment dE, stacked by originating process (log y), with the Pb K X-ray
#       photo-absorption energies (~75 / 87 keV) marked.
# Also prints the deposited-energy budget by process (what carries the charge).
#
# Usage:
#   python tests/bi207/annotate_bi207_processes.py bi207.EDEPSIM.h5 --outdir bi207_check
# =============================================================================
import argparse
import os
import sys

import numpy as np

ME = 0.510999  # MeV
BIG = 1_000_000_000

# Conversion-electron KE lines (MeV), grouped by the transition they convert.
CE_1064 = (0.9757, 1.0478, 1.0598)   # K, L, M of the 1063.7 keV transition
CE_570 = (0.4817, 0.5538, 0.5658)    # K, L, M of the  569.7 keV transition
CE_1770 = (1.682,)                   # K of the 1770.2 keV crossover
G570, G1064, G1770 = 0.5697, 1.0637, 1.7702
CE_WIN, G_WIN = 0.020, 0.035         # MeV matching windows

# Category -> (label, color). Order controls the stack order (low energy at bottom).
SEG_CATS = [
    ("ce570",    "conv. e$^-$ (570 keV transition)",   "#1f77b4"),
    ("ce1064",   "conv. e$^-$ (1064 keV transition)",  "#2ca02c"),
    ("ce1770",   "conv. e$^-$ (1770 keV transition)",  "#17becf"),
    ("auger",    "Auger / atomic e$^-$ (off nucleus)", "#9467bd"),
    ("xray",     "Pb K X-ray photo-e$^-$ (~75-88 keV)","#e377c2"),
    ("comp570",  "Compton/PE from 570 keV $\\gamma$",  "#ff7f0e"),
    ("comp1064", "Compton/PE from 1064 keV $\\gamma$", "#d62728"),
    ("comp1770", "Compton/PE from 1770 keV $\\gamma$", "#8c564b"),
    ("untraced", "untraced $\\gamma$-induced",         "#7f7f7f"),
]
PB_K = 0.088  # Pb K-shell binding (MeV): energy deposited when the EC/conversion K-vacancy
              # cascade (X-ray + Auger) is absorbed locally. Shifts the deposited peaks up.
DECAY_CATS = [
    ("conv1064", "1064 keV converted",  "#2ca02c"),
    ("conv570",  "570 keV converted",   "#1f77b4"),
    ("conv1770", "1770 keV converted",  "#17becf"),
    ("gamma",    "$\\gamma$-only (no conversion e$^-$)", "#ff7f0e"),
]


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _get(h, name):
    return np.asarray(h[name][:]) if name in h else None


def _seg_link_field(seg):
    """Find the field linking a segment to its trajectory (varies by dumpTree version)."""
    for c in ("traj_id", "trackID", "track_id", "file_traj_id", "local_traj_id", "segment_id"):
        if c in seg.dtype.names:
            return c
    return None


def build_origin(trj):
    """Return origin(key)->(root_pdg, root_E): the first ancestor that is a direct daughter
    of the decaying nucleus, climbing through deltas and Compton-scatter chains."""
    ev = trj["event_id"].astype(np.int64)
    tid = trj["traj_id"].astype(np.int64)
    par = trj["parent_id"].astype(np.int64)
    pdg = trj["pdg_id"]
    E = trj["E_start"].astype(float)
    key = ev * BIG + tid
    idx_of = {int(k): i for i, k in enumerate(key)}

    def origin(k):
        """Returns (root_pdg, root_E, on_ion). on_ion=True means the climb stopped on a
        STORED nucleus/ion parent (reliable: a genuine conversion/Auger e- or de-excitation
        gamma). on_ion=False means the parent was not stored (an untraceable gamma chain)."""
        k = int(k)
        guard = 0
        while True:
            i = idx_of.get(k)
            if i is None:
                return (None, None, False)
            pk = (k // BIG) * BIG + int(par[i])     # parent key in the SAME event
            j = idx_of.get(pk)
            if j is None:
                return (int(pdg[i]), float(E[i]), False)   # parent not stored -> untraceable
            if pdg[j] > BIG:
                return (int(pdg[i]), float(E[i]), True)    # parent IS the decaying nucleus/ion
            k = pk
            guard += 1
            if guard > 400:
                return (int(pdg[i]), float(E[i]), False)

    return origin, idx_of


def categorize(root_pdg, root_E, on_ion):
    """Map a root quantum (pdg, E, on_ion) to a segment-process category key.

    on_ion distinguishes a genuine direct daughter of the nucleus (conversion/Auger e- or
    de-excitation gamma -- reliable) from an electron whose parent photon was not stored
    (an untraceable Compton/PE chain)."""
    if root_pdg is None:
        return "untraced"
    if root_pdg == 22:                         # traced to a photon
        if abs(root_E - G1064) < G_WIN:
            return "comp1064"
        if abs(root_E - G570) < G_WIN:
            return "comp570"
        if abs(root_E - G1770) < G_WIN:
            return "comp1770"
        if root_E < 0.12:                      # Pb K/L fluorescence X-ray, photo-absorbed
            return "xray"
        return "untraced"                      # mid-energy scattered gamma, line lost
    if root_pdg == 11:
        ke = root_E - ME                       # edep E_start is TOTAL energy; CE lines are KE
        if on_ion:                             # electron emitted directly off the nucleus
            if any(abs(ke - L) < CE_WIN for L in CE_1064):
                return "ce1064"
            if any(abs(ke - L) < CE_WIN for L in CE_570):
                return "ce570"
            if any(abs(ke - L) < CE_WIN for L in CE_1770):
                return "ce1770"
            return "auger"                     # Auger / low-E atomic electron off the nucleus
        return "untraced"                      # parent photon not stored -> cannot assign a line
    return "untraced"


def main():
    ap = argparse.ArgumentParser(description="Annotate Bi-207 dE spectra by process.")
    ap.add_argument("path", help="dumpTree edep-sim HDF5 (segments + trajectories)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--emax", type=float, default=2.0, help="total-dE x-axis max (MeV)")
    ap.add_argument("--seg-emax", type=float, default=1.1, help="per-segment x-axis max (MeV)")
    args = ap.parse_args()
    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.path)) or ".")
    os.makedirs(outdir, exist_ok=True)

    with open_ro(args.path) as h:
        seg = _get(h, "segments")
        trj = _get(h, "trajectories")
    if seg is None or not seg.size:
        sys.exit("no 'segments' dataset -- nothing to decompose.")
    if trj is None or not trj.size:
        sys.exit("no 'trajectories' dataset -- cannot attribute processes "
                 "(re-run dumpTree keeping trajectories).")
    print(f"# segment fields: {seg.dtype.names}")
    print(f"# trajectory fields: {trj.dtype.names}")

    dE = seg["dE"].astype(float)                       # MeV
    sev = seg["event_id"].astype(np.int64)
    link = _seg_link_field(seg)
    if link is None:
        sys.exit("segments carry no trajectory-link field (looked for traj_id/trackID/...). "
                 "Tell me which field links a segment to its trajectory and I'll wire it in; "
                 "the fields are printed above.")
    print(f"# using segment->trajectory link: '{link}'")

    origin, _ = build_origin(trj)

    # ---- attribute every segment to a process category ----
    skey = sev * BIG + seg[link].astype(np.int64)
    # cache origin per unique trajectory key (many segments share one electron)
    cat_of = {}
    seg_cat = np.empty(seg.size, dtype=object)
    for n, k in enumerate(skey):
        c = cat_of.get(int(k))
        if c is None:
            c = categorize(*origin(k))
            cat_of[int(k)] = c
        seg_cat[n] = c

    seg_keys = [c for c, _, _ in SEG_CATS]
    seg_color = {c: col for c, _, col in SEG_CATS}
    seg_label = {c: lab for c, lab, _ in SEG_CATS}

    # ---- deposited-energy budget by process (collected -> written to the .txt summary) ----
    summary = [f"# file: {args.path}",
               f"# segment->trajectory link: '{link}'",
               f"# segments={seg.size}  trajectories={trj.size}",
               "", "[dE budget by process]  (fraction of ALL deposited energy)"]
    tot = dE.sum()
    for c in seg_keys:
        m = seg_cat == c
        e = dE[m].sum()
        if e > 0:
            summary.append(f"  {seg_label[c]:38s}: {100*e/tot:6.2f}%  ({int(m.sum())} segments)")

    # ---- per-decay topology: which transition converted in each decay ----
    uev = np.unique(sev)
    per_ev_dE = {int(e): 0.0 for e in uev}
    for e, d in zip(sev, dE):
        per_ev_dE[int(e)] += d
    ev_has = {c: set() for c in ("ce1064", "ce570", "ce1770")}
    for k, c in cat_of.items():
        if c in ev_has:
            ev_has[c].add(k // BIG)

    def decay_cat(e):
        if e in ev_has["ce1064"]:
            return "conv1064"
        if e in ev_has["ce570"]:
            return "conv570"
        if e in ev_has["ce1770"]:
            return "conv1770"
        return "gamma"

    dkeys = [c for c, _, _ in DECAY_CATS]
    dcolor = {c: col for c, _, col in DECAY_CATS}
    dlabel = {c: lab for c, lab, _ in DECAY_CATS}
    tot_by_decay = {c: [] for c in dkeys}
    for e in uev:
        tot_by_decay[decay_cat(int(e))].append(per_ev_dE[int(e)])
    summary += ["", "[decays by topology]"]
    for c in dkeys:
        summary.append(f"  {dlabel[c]:40s}: {len(tot_by_decay[c]):6d} decays")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: total dE per decay, stacked by decay topology
    edges = np.linspace(0, args.emax, 220)
    stacks = [np.array(tot_by_decay[c]) for c in dkeys]
    ax[0].hist(stacks, bins=edges, stacked=True, histtype="stepfilled",
               color=[dcolor[c] for c in dkeys], label=[dlabel[c] for c in dkeys],
               edgecolor="none", alpha=0.9)
    ymaxA = ax[0].get_ylim()[1]
    for E0, txt in [(G570, "570"), (G1064, "1064"), (1.6334, "1064+570")]:
        ax[0].axvline(E0, color="0.65", ls=":", lw=0.7)            # bare nuclear (X-ray escapes)
        ax[0].axvline(E0 + PB_K, color="k", ls="--", lw=1.0)       # + Pb K X-ray = observed peak
        ax[0].text(E0 + PB_K, ymaxA*0.98, f" {txt}+PbK = {int(round((E0+PB_K)*1e3))} keV",
                   rotation=90, va="top", ha="left", fontsize=8, color="k")
    ax[0].set_title("Total dE per decay, by which transition converted\n"
                    "(conversion e$^-$ deposit locally -> full peak; escaping $\\gamma$ -> shoulders)")
    ax[0].set_xlabel("total deposited energy per decay  [MeV]")
    ax[0].set_ylabel("decays")
    ax[0].legend(fontsize=8, loc="upper left")

    # Panel B: per-segment dE, stacked by originating process (log y)
    segedges = np.linspace(0, args.seg_emax*1e3, 220)   # keV
    present = [c for c in seg_keys if (seg_cat == c).any()]
    seg_stacks = [dE[seg_cat == c]*1e3 for c in present]
    ax[1].hist(seg_stacks, bins=segedges, stacked=True, histtype="stepfilled",
               color=[seg_color[c] for c in present], label=[seg_label[c] for c in present],
               edgecolor="none", alpha=0.9)
    ax[1].set_yscale("log")
    yb = ax[1].get_ylim()[0]
    # Short-range conversion e- deposit in ~1 segment -> the blue/green spikes ARE these lines.
    # Labels sit at the BOTTOM so the legend never covers them.
    for L in CE_570:
        ax[1].axvline(L*1e3, color="#1f77b4", ls=":", lw=0.7)
    for L in CE_1064:
        ax[1].axvline(L*1e3, color="#2ca02c", ls=":", lw=0.7)
    ax[1].text(481.7, yb*1.4, " 482 (K-570)", rotation=90, va="bottom", ha="left", fontsize=7, color="#1f77b4")
    ax[1].text(975.7, yb*1.4, " 976 (K-1064)", rotation=90, va="bottom", ha="left", fontsize=7, color="#2ca02c")
    for E in (75.0, 87.3):                          # Pb K X-ray photo-absorption
        ax[1].axvline(E, color="k", ls=":", lw=0.7)
    ax[1].text(80, yb*1.4, " Pb K X-ray", rotation=90, va="bottom", ha="left", fontsize=7, color="k")
    ax[1].set_title("Per-segment dE, by originating process")
    ax[1].set_xlabel("per-segment deposited energy  [keV]")
    ax[1].set_ylabel("segments")
    ax[1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = os.path.join(outdir, "bi207_processes.png")
    fig.savefig(out, dpi=120)
    txt = os.path.join(outdir, "bi207_processes.txt")
    with open(txt, "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"wrote {out}")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
