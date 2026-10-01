#!/usr/bin/env python3
# =============================================================================
# fsdcube_hitq_datamc.py  --  standalone re-implementation of mjkramer's FSDCube
# data/MC "Hit Q for all dt" workflow:
#     https://gist.github.com/mjkramer/a00a0dde7dccba948bedb8cbbe448585  (cells ~5-13)
#
# This does NOT import that notebook -- it copies the WORKFLOW so you can run and
# modify it yourself. It reads the SAME FLOW files (real data + mkramer's larnd-sim)
# with plain h5py (following the charge/events -> calib_prompt_hits references, so NO
# h5flow dependency), builds per-pixel hit lists PER EVENT, and histograms the per-hit charge
# Q (50 bins, 0-50 ke-, density) -- then overlays data vs sim for prc2 | prc16, and
# ADDS the split the notebook was missing: all hits / single-hit-pixel hits /
# multi-hit-pixel hits, to see which population carries the mid-Q shoulder.
#
# Run on NERSC (where the FLOW files and h5flow live), e.g.:
#   python tests/fsdcube_hitq_datamc.py --outdir hitq_out
#   python tests/fsdcube_hitq_datamc.py --prc 2 16 --sim-r 3 --max-files 20 --outdir hitq_out
#
# Needs: h5py, numpy, matplotlib (all in the larnd env). No h5flow, no GPU, no larnd-sim import.
# =============================================================================
import argparse, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- geometry + binning (verbatim from the notebook) ------------------------
PITCH = 0.372
TILE_X0, TILE_Y0, TILE_Z0 = 48.0675, 133.92, 23.808     # TILE_CENTER
NX, NY = 128, 80
QBINS, QMIN, QMAX = 50, 0, 50                            # ke-
CUTS = dict(adc_min=0.2e6, adc_max=0.5e6)               # event-level cut on charge/events 'ADC'

# default file locations (mkramer's; override on the CLI if they move)
SIM_BASE_DEF = "/pscratch/sd/m/mkramer/test/larnd-sim.segFFE3/output/skim_more"
DATA_BASE_DEF = ("/dvs_ro/cfs/cdirs/dunepro/www/data/nd-production/FSDCube"
                 "/Reflow_FSDCube_v3/run-ndlar-flow/Reflow_FSDCube_v3.flow"
                 "/FLOW/Feb2026/cold/scan")


def pixel2id_abs(z, y):
    """Map hit (z, y) [cm] -> an integer pixel id, so hits on the same pad group together
    (mirrors the notebook: z is the 128-pad axis, y the 80-pad axis)."""
    x0 = TILE_Z0 - (NX / 2 - 0.5) * PITCH
    y0 = TILE_Y0 - (NY / 2 - 0.5) * PITCH
    xr = np.round((np.asarray(z) - x0) / PITCH).astype(np.int64)
    yr = np.round((np.asarray(y) - y0) / PITCH).astype(np.int64)
    return xr + NX * yr                                  # pixel2id_rel


def accumulate_q(files, adc_min, adc_max, max_events=None, verbose=True):
    """Loop FLOW files/events with plain h5py (follow the charge/events -> calib_prompt_hits
    references), apply the ADC event cut, group hits by pixel WITHIN EACH EVENT, and return per-hit
    Q (ke-) for three populations: (all, single-hit-pixel, multi-hit-pixel). Multiplicity is counted
    per event (a pad hit in two events = two single-hit entries).

    Reference layout (standard ndlar_flow): charge/events/ref/charge/calib_prompt_hits/ref is
    (n_links, 2) with col0 = event index, col1 = hit index; .../ref_region[e] = (start, stop) slices
    that ref array for event e; hits live in charge/calib_prompt_hits/data (fields z, y, Q)."""
    import h5py
    REFG = "charge/events/ref/charge/calib_prompt_hits"
    allq, singq, multq = [], [], []
    for fi, path in enumerate(files):
        try:
            with h5py.File(str(path), "r") as h:
                adc = np.asarray(h["charge/events/data"]["ADC"])
                sel = np.ones(adc.shape, bool)
                if adc_min:
                    sel &= adc >= adc_min
                if adc_max:
                    sel &= adc <= adc_max
                idx = np.nonzero(sel)[0]
                if max_events:
                    idx = idx[:max_events]
                reg = h[f"{REFG}/ref_region"]
                r_start = np.asarray(reg["start"]); r_stop = np.asarray(reg["stop"])
                hit_of_ref = np.asarray(h[f"{REFG}/ref"][:, 1])        # ref col1 = hit index
                hd = h["charge/calib_prompt_hits/data"]
                z_all = np.asarray(hd["z"]); y_all = np.asarray(hd["y"]); q_all = np.asarray(hd["Q"])
                for e in idx:
                    st, sp = int(r_start[e]), int(r_stop[e])
                    if sp <= st:
                        continue
                    hi = hit_of_ref[st:sp]
                    pid = pixel2id_abs(z_all[hi], y_all[hi])
                    q = q_all[hi].astype(float)            # 'Q' already ke- (no scaling, per notebook)
                    allq.append(q)
                    order = np.argsort(pid, kind="stable")
                    pid_s, q_s = pid[order], q[order]
                    _, start, counts = np.unique(pid_s, return_index=True, return_counts=True)
                    for a0, ct in zip(start, counts):
                        (singq if ct == 1 else multq).append(q_s[a0:a0 + ct])
        except Exception as e:
            print(f"  !! skip {path}: {e}")
            continue
        if verbose:
            print(f"  [{fi + 1}/{len(files)}] {Path(path).name}: "
                  f"all={sum(a.size for a in allq)}", flush=True)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    return cat(allq), cat(singq), cat(multq)


def _density(q):
    h, edges = np.histogram(q, bins=QBINS, range=(QMIN, QMAX))
    w = edges[1] - edges[0]
    s = h.sum()
    ctr = 0.5 * (edges[1:] + edges[:-1])
    return ctr, (h / (s * w) if s else h.astype(float))   # density (area = 1)


def sim_files(sim_base, prc, r, var=""):
    cfg = f"prc{prc}_r{r}{var}_patch_noDrop"
    return cfg, sorted((Path(sim_base) / cfg / "flow").rglob("*.FLOW.hdf5"))


def data_files(data_base, prc):
    return sorted((Path(data_base) / f"lt_prc{prc}").rglob("*.hdf5"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prc", type=int, nargs="+", default=[2, 16], help="parity configs to plot (panels)")
    ap.add_argument("--sim-r", type=int, default=3, help="sim readout radius r3/r4 (default 3)")
    ap.add_argument("--sim-var", default="", help="sim variant suffix, e.g. _noFFE (default none)")
    ap.add_argument("--data-base", default=DATA_BASE_DEF)
    ap.add_argument("--sim-base", default=SIM_BASE_DEF)
    ap.add_argument("--adc-min", type=float, default=CUTS["adc_min"])
    ap.add_argument("--adc-max", type=float, default=CUTS["adc_max"])
    ap.add_argument("--max-files", type=int, default=None, help="cap files per sample (quick tests)")
    ap.add_argument("--max-events", type=int, default=None, help="cap selected events per file")
    ap.add_argument("--outdir", default="hitq_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = lambda L: L[: args.max_files] if args.max_files else L

    # gather per-hit Q (all/single/multi) for data + sim, per prc
    store = {}
    for prc in args.prc:
        df = cap(data_files(args.data_base, prc))
        scfg, sf = sim_files(args.sim_base, prc, args.sim_r, args.sim_var)
        sf = cap(sf)
        print(f"\nprc{prc}: {len(df)} data files, {len(sf)} sim files ({scfg})")
        print(" data:")
        store[(prc, "data")] = accumulate_q(df, args.adc_min, args.adc_max, args.max_events)
        print(" sim:")
        store[(prc, "sim")] = accumulate_q(sf, args.adc_min, args.adc_max, args.max_events)
        store[(prc, "simcfg")] = scfg

    # ---- Figure 1: reproduce cell 13 -- all-hits per-hit Q, data (solid) vs sim (dashed) ----
    fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6), squeeze=False)
    for ax, prc in zip(axes[0], args.prc):
        for src, ls in (("data", "-"), ("sim", "--")):
            c, y = _density(store[(prc, src)][0])
            ax.step(c, y, where="mid", ls=ls,
                    label=f"{src} prc{prc}" + ("" if src == "data" else f" ({store[(prc,'simcfg')]})"))
        ax.set_title(f"Q for all dt -- prc{prc}"); ax.set_xlabel(R"Hit Q [ke$^-$]")
        ax.set_ylabel("Density"); ax.set_xlim(QMIN, QMAX); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/hitq_datamc.png", dpi=120)
    print(f"\nwrote {args.outdir}/hitq_datamc.png")

    # ---- Figure 2: the shoulder split -- all / single-hit-pix / multi-hit-pix, data & sim ----
    for src in ("data", "sim"):
        fig, axes = plt.subplots(1, len(args.prc), figsize=(6.4 * len(args.prc), 4.6), squeeze=False, sharey=True)
        for ax, prc in zip(axes[0], args.prc):
            qa, qs, qm = store[(prc, src)]
            for q, lab, col in ((qa, "all hits", "0.25"),
                                (qs, "single-hit pixels", "C0"),
                                (qm, "multi-hit pixels", "C1")):
                c, y = _density(q)
                ax.step(c, y, where="mid", color=col, label=lab)
            tag = f"prc{prc}" + ("" if src == "data" else f"  {store[(prc,'simcfg')]}")
            ax.set_title(f"{src} {tag}"); ax.set_xlabel(R"Hit Q [ke$^-$]")
            ax.set_ylabel("Density"); ax.set_xlim(QMIN, QMAX); ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.suptitle(f"Hit Q split by pixel multiplicity -- {src}", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{args.outdir}/hitq_split_{src}.png", dpi=120)
        print(f"wrote {args.outdir}/hitq_split_{src}.png")


if __name__ == "__main__":
    main()
