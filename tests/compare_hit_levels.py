#!/usr/bin/env python3
# =============================================================================
# tests/compare_hit_levels.py -- compare the RAW FEE hit level (charge/raw_hits)
# with the reconstructed level (charge/calib_prompt_hits) in an ndlar_flow FLOW
# file, to localize where the data/MC single-hit-Q + hit-multiplicity differences
# enter: the FEE, or the prompt-hit reconstruction.
#
# The data/MC plotter (tests/fsdcube_hitq_datamc.py) reads ONLY calib_prompt_hits.
# raw_hits is the FEE hit (one per data packet; x_pix/y_pix/z_pix, ADC); calib_prompt_hits
# is its per-hit calibration (ADC -> Q [ke-]). If the two levels give the SAME per-pixel
# multiplicity, the reconstruction preserves hit identity and the data/MC difference is in
# the FEE, not the grouping. Multiplicity is counted per event with each level's OWN pixel
# key (raw: the x/y/z_pix index tuple; calib: the x/y/z position tuple), so no cross-level
# coordinate assumption is made. Charge is reported in each level's native unit (raw: ADC
# counts; calib: Q in ke-), plus the ADC->ke- gain if the two levels line up 1:1 by row.
#
# Usage:
#   python tests/compare_hit_levels.py \
#     '/pscratch/.../prc2_r4_patch_noDrop/flow/*.FLOW.hdf5'          # all matched files
#   python tests/compare_hit_levels.py <file|glob|dir> --max-files 4 --max-events 0
#   python tests/compare_hit_levels.py <...> --adc-min 2e5 --adc-max 5e5   # event cut
# =============================================================================
import argparse
import glob
import os
import sys

import numpy as np

# default event-level charge/events 'ADC' cut -- same window the plotter uses
ADC_MIN, ADC_MAX = 0.2e6, 0.5e6
LEVELS = {
    "raw_hits":          dict(coords=("x_pix", "y_pix", "z_pix"), q="ADC", unit="ADC"),
    "calib_prompt_hits": dict(coords=("x", "y", "z"),             q="Q",   unit="ke-"),
}


def resolve(path):
    if any(c in path for c in "*?["):
        hits = sorted(glob.glob(path))
    elif os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "*.FLOW.hdf5"), recursive=True) or
                      glob.glob(os.path.join(path, "**", "*.hdf5"), recursive=True))
    else:
        hits = [path]
    hits = [p for p in hits if os.path.exists(p)]
    if not hits:
        sys.exit(f"!! no file matched: {path}")
    return hits


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def event_mask(h, adc_min, adc_max, max_events):
    adc = np.asarray(h["charge/events/data"]["ADC"])
    sel = np.ones(adc.shape, bool)
    if adc_min:
        sel &= adc >= adc_min
    if adc_max:
        sel &= adc <= adc_max
    idx = np.nonzero(sel)[0]
    return idx[:max_events] if max_events else idx


def read_level(h, idx, level):
    """Per-hit multiplicity (of that hit's pixel, within its event) and per-hit charge,
    plus each pixel's single-hit charge, read the plotter's way for one hit level."""
    spec = LEVELS[level]
    refg = f"charge/events/ref/charge/{level}"
    reg = h[f"{refg}/ref_region"]
    r_start, r_stop = np.asarray(reg["start"]), np.asarray(reg["stop"])
    hit_of_ref = np.asarray(h[f"{refg}/ref"][:, 1])      # ref col1 = hit index
    hd = h[f"charge/{level}/data"]
    cols = [np.asarray(hd[c]) for c in spec["coords"]]
    q_all = np.asarray(hd[spec["q"]], float)
    per_hit_mult, single_q = [], []
    for e in idx:
        st, sp = int(r_start[e]), int(r_stop[e])
        if sp <= st:
            continue
        hi = hit_of_ref[st:sp]
        key = np.stack([c[hi] for c in cols], axis=1)    # one row per hit -> pixel key
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        per_hit_mult.append(counts[inv])
        single_q.append(q_all[hi][counts[inv] == 1])     # hits whose pixel fired once
    mult = np.concatenate(per_hit_mult) if per_hit_mult else np.zeros(0, int)
    sq = np.concatenate(single_q) if single_q else np.zeros(0, float)
    return mult, sq, q_all.size


def summarize(level, mult, sq, unit):
    n = mult.size
    if n == 0:
        print(f"  {level:18s}: no hits"); return
    frac1 = (mult == 1).mean()
    # per-pixel multiplicity fractions (one weight per pixel = 1/mult for each of its hits)
    wpix = 1.0 / mult
    bins = np.arange(1, 9)
    pm = np.array([wpix[mult == m].sum() for m in bins]); pm /= pm.sum()
    hist = "  ".join(f"m{m}={p:.3f}" for m, p in zip(bins, pm))
    print(f"  {level:18s}: hits={n:>7d}  single-hit frac={frac1:.3f}  "
          f"single-hit Q peak/mean={_peak(sq):.1f}/{sq.mean():.2f} {unit}")
    print(f"  {'':18s}  pixel-mult: {hist}")


def _peak(x, lo=0, hi=50, w=1.0):
    if x.size == 0:
        return float("nan")
    edges = np.arange(lo, hi + w, w)
    c, _ = np.histogram(x[np.isfinite(x)], bins=edges)
    return 0.5 * (edges[c.argmax()] + edges[c.argmax() + 1]) if c.sum() else float("nan")


def main():
    ap = argparse.ArgumentParser(description="raw_hits vs calib_prompt_hits in a FLOW file.")
    ap.add_argument("path", help="FLOW file, glob, or directory")
    ap.add_argument("--max-files", type=int, default=0, help="limit # files (0 = all matched)")
    ap.add_argument("--max-events", type=int, default=0, help="limit # events per file (0 = all)")
    ap.add_argument("--adc-min", type=float, default=ADC_MIN)
    ap.add_argument("--adc-max", type=float, default=ADC_MAX)
    args = ap.parse_args()

    files = resolve(args.path)
    if args.max_files:
        files = files[:args.max_files]
    print(f"# {len(files)} file(s); event ADC cut [{args.adc_min:.3g},{args.adc_max:.3g}]\n")

    agg = {lv: (np.zeros(0, int), np.zeros(0, float)) for lv in LEVELS}
    rows = {lv: 0 for lv in LEVELS}
    for p in files:
        with open_ro(p) as h:
            idx = event_mask(h, args.adc_min, args.adc_max, args.max_events)
            for lv in LEVELS:
                if lv not in h["charge"]:
                    print(f"  !! {lv} not in {os.path.basename(p)}"); continue
                m, sq, nrow = read_level(h, idx, lv)
                agg[lv] = (np.concatenate([agg[lv][0], m]), np.concatenate([agg[lv][1], sq]))
                rows[lv] += nrow

    print("TOTAL raw dataset rows:  " +
          "  ".join(f"{lv}={rows[lv]}" for lv in LEVELS) +
          ("   (equal => 1:1, reco preserves hit identity)"
           if len(set(rows.values())) == 1 else "   (UNEQUAL => reco changes hit count!)"))
    print()
    for lv, spec in LEVELS.items():
        summarize(lv, agg[lv][0], agg[lv][1], spec["unit"])


if __name__ == "__main__":
    main()
