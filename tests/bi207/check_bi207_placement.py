#!/usr/bin/env python3
# =============================================================================
# tests/bi207/check_bi207_placement.py -- diagnose WHERE the Bi-207 source actually sits in the
# fsd_cube frame, because the anode hitmap shows the source projecting to the Z edge (col ~125)
# instead of the intended anode-plane center (col 64). Pure diagnosis: reads the EDEPSIM file(s),
# prints positions, computes the offset. No GPU/container; login node, seconds.
#
# bi207_to_fsd.py shifts every segment by (cathode_center - src), where src is the mean PRIMARY
# TRAJECTORY vertex. If the dumpTree stores segment coordinates in a frame offset from the trajectory
# frame, the segments land at (target + that offset) -- off-center. This script measures that offset
# directly by comparing, in the SAME file, the primary trajectory vertex to where the dense (short-
# range conversion-electron) deposits actually are.
#
# Usage (NERSC, larnd env, login node):
#   python tests/bi207/check_bi207_placement.py \
#       --translated $ND_WORK/bi207_fsd.EDEPSIM.h5  --input $ND_WORK/bi207.EDEPSIM.h5
#   (either file may be omitted; --input is what reveals the trajectory-vs-segment frame offset.)
# =============================================================================
import argparse
import os

import numpy as np

BIG = 1_000_000_000


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def stat(a):
    a = np.asarray(a, float).ravel()
    if a.size == 0:
        return "EMPTY"
    return (f"min {a.min():8.2f}  med {np.median(a):8.2f}  mean {a.mean():8.2f}  "
            f"max {a.max():8.2f}  (n={a.size})")


def seg_center_xyz(seg):
    """Best per-segment (x,y,z): the 'x'/'y'/'z' center fields if present, else the start/end mean."""
    out = {}
    for ax in ("x", "y", "z"):
        if ax in seg.dtype.names:
            out[ax] = np.asarray(seg[ax], float)
        elif f"{ax}_start" in seg.dtype.names and f"{ax}_end" in seg.dtype.names:
            out[ax] = 0.5 * (np.asarray(seg[f"{ax}_start"], float) + np.asarray(seg[f"{ax}_end"], float))
        else:
            out[ax] = None
    return out


def primary_vertex(trj):
    """Mean primary vertex (parent_id==-1, else the decaying ion pdg>1e9). None if unavailable."""
    if trj is None or "xyz_start" not in trj.dtype.names:
        return None
    if "parent_id" in trj.dtype.names:
        m = trj["parent_id"] == -1
        if not m.any() and "pdg_id" in trj.dtype.names:
            m = trj["pdg_id"] > BIG
    elif "pdg_id" in trj.dtype.names:
        m = trj["pdg_id"] > BIG
    else:
        return None
    if not np.asarray(m).any():
        return None
    return np.asarray(trj["xyz_start"], float)[np.asarray(m)].mean(axis=0)


def dense_deposit_point(xyz, dE=None):
    """The densest deposit point per axis = the histogram mode. Short-range conversion electrons pile
    at the source, so this estimates the source point robustly against the Compton halo."""
    pt = {}
    for ax in ("x", "y", "z"):
        v = xyz.get(ax)
        if v is None or v.size == 0:
            pt[ax] = float("nan")
            continue
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            pt[ax] = lo
            continue
        nb = max(10, min(200, int((hi - lo) / 0.2)))   # ~2 mm bins
        w = np.asarray(dE, float) if dE is not None and np.size(dE) == v.size else None
        c, e = np.histogram(v, bins=nb, weights=w)
        pt[ax] = float(0.5 * (e[np.argmax(c)] + e[np.argmax(c) + 1]))
    return pt


def load_fsd_cube_borders(tpc=0):
    from larndsim import consts
    from larndsim.config import get_config
    from larndsim.consts import detector
    cfg = get_config("fsd_cube")

    def one(k):
        v = cfg.get(k)
        return v[0] if isinstance(v, list) else v

    consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"),
                           one("RESPONSE"), one("SIM_PROPERTIES"))
    return np.asarray(detector.TPC_BORDERS)[tpc]


def report_file(tag, path):
    """Print segment/trajectory positions for one file; return (seg_xyz dict, primary vertex, dE)."""
    if not path or not os.path.exists(path):
        print(f"[{tag}] (no file at {path})")
        return None, None, None
    with open_ro(path) as h:
        seg = np.asarray(h["segments"][:]) if "segments" in h else None
        trj = np.asarray(h["trajectories"][:]) if "trajectories" in h else None
    print(f"[{tag}] {path}")
    if seg is None or not seg.size:
        print(f"[{tag}]   no/empty segments")
        return None, None, None
    xyz = seg_center_xyz(seg)
    dE = np.asarray(seg["dE"], float) if "dE" in seg.dtype.names else None
    for ax in ("x", "y", "z"):
        print(f"[{tag}]   segment {ax}: {stat(xyz[ax]) if xyz[ax] is not None else 'absent'}")
    pv = primary_vertex(trj)
    if pv is not None:
        print(f"[{tag}]   primary trajectory vertex (mean): x {pv[0]:.2f}  y {pv[1]:.2f}  z {pv[2]:.2f}")
    dense = dense_deposit_point(xyz, dE)
    print(f"[{tag}]   densest deposit point (source est): "
          f"x {dense['x']:.2f}  y {dense['y']:.2f}  z {dense['z']:.2f}")
    if pv is not None:
        off = [dense[a] - pv[i] for i, a in enumerate(("x", "y", "z"))]
        print(f"[{tag}]   >>> deposit-vs-trajectory offset: "
              f"dx {off[0]:+.2f}  dy {off[1]:+.2f}  dz {off[2]:+.2f} cm "
              f"(nonzero = the segment frame is offset from the trajectory frame)")
    return xyz, pv, dE


def main():
    ap = argparse.ArgumentParser(description="Diagnose the Bi-207 source placement in the fsd_cube frame.")
    ap.add_argument("--translated", default=None, help="fsd_cube-frame EDEPSIM (…_fsd.EDEPSIM.h5)")
    ap.add_argument("--input", default=None, help="original module0-frame EDEPSIM (…bi207.EDEPSIM.h5)")
    ap.add_argument("--tpc", type=int, default=0)
    args = ap.parse_args()

    try:
        B = load_fsd_cube_borders(args.tpc)
        x0, x1 = sorted((float(B[0][0]), float(B[0][1])))
        y0, y1 = sorted((float(B[1][0]), float(B[1][1])))
        z0, z1 = float(B[2][0]), float(B[2][1])          # z0 anode, z1 cathode (drifting.py)
        print(f"fsd_cube TPC {args.tpc} borders:  x[{x0:.2f},{x1:.2f}]  y[{y0:.2f},{y1:.2f}]  "
              f"z[{z0:.2f},{z1:.2f}] (anode z0, cathode z1)")
        print(f"intended source target (anode-plane CENTER on the cathode): "
              f"x {(x0+x1)/2:.2f}  y {(y0+y1)/2:.2f}  z {z1:.2f}  "
              f"[x=mid -> col 64, y=mid -> row 40]\n")
    except Exception as exc:
        print(f"(could not load fsd_cube borders: {exc!r})\n")
        x0 = x1 = None

    report_file("translated", args.translated)
    print()
    report_file("input", args.input)

    print("\nINTERPRETATION:")
    print("- In the TRANSLATED file the densest deposit x is where the source landed. If it is near")
    print("  the x edge (~47) rather than mid (~23.8 = col 64), the source is mis-centered in Z.")
    print("- In the INPUT file, a nonzero deposit-vs-trajectory offset is the root cause: bi207_to_fsd")
    print("  centers the TRAJECTORY vertex, but the SEGMENTS carry that offset, so they land off-center.")
    print("  The fix is to reference the segment deposits (not the trajectory vertex) when centering.")


if __name__ == "__main__":
    main()
