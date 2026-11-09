#!/usr/bin/env python3
# =============================================================================
# tests/bi207/bi207_to_fsd.py -- STAGE 2 translate: move a Bi-207 edep generation (made in the
# module0 GDML at an interior point) so the SOURCE POINT lands on the FSD-cube cathode, in the
# fsd_cube detector frame, ready for simulate_pixels(--config fsd_cube).
#
# WHY A RIGID SHIFT IS CORRECT HERE: the Bi-207 source is isotropic (/gps/ang iso), so the
# deposit cloud has no preferred axis -- a translation (no rotation) preserves it. simulate_pixels
# compares segment x/y/z directly to the fsd_cube TPC_BORDERS (drifting.py), so the deposits must
# already be in that frame. We shift every segment coordinate by (cathode_center - source_point).
# Deposits that then fall outside the (smaller) fsd_cube volume are dropped by the drift step --
# physically, those gammas escape the cube, which is exactly what a real cube source would do.
#
# SOURCE ON THE CATHODE: a point source centered on the cathode PLANE emits isotropically, so
# ~half the cloud is on the drift side (collected) and ~half goes into the cathode (z beyond the
# cathode -> outside -> dropped). Use --inset to push the source a few mm into the LAr instead.
#
# The cathode center + frame are read from the loaded fsd_cube config at runtime (not hard-coded):
# TPC_BORDERS[tpc] = [[x0,x1],[y0,y1],[z0,z1]] with the ANODE at z0 and the CATHODE at z1
# (drifting.py: z_anode = TPC_BORDERS[plane][2][0]); cathode center = (mid x, mid y, z1).
#
# Usage (on NERSC, in the larnd env):
#   python tests/bi207/bi207_to_fsd.py $ND_WORK/bi207.EDEPSIM.h5 $ND_WORK/bi207_fsd.EDEPSIM.h5
#   # then: python cli/simulate_pixels.py --input_filename=... --output_filename=...LARND.h5 \
#   #            --config=fsd_cube --rand_seed=1   ; then ndlar_flow -> *.FLOW.hdf5
# =============================================================================
import argparse
import os
import sys

import numpy as np
from numpy.lib import recfunctions as rfn

SEG_XYZ = ("x_start", "y_start", "z_start", "x_end", "y_end", "z_end", "x", "y", "z")
TRJ_XYZ = ("xyz_start", "xyz_end")        # (N,3) subarray fields
BIG = 1_000_000_000


def open_ro(path):
    import h5py
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def fsd_cube_cathode_center(tpc):
    """Load the fsd_cube config and return (cathode_center xyz, drift-axis index, borders)."""
    from larndsim.config import get_config
    from larndsim import consts
    from larndsim.consts import detector
    cfg = get_config("fsd_cube")
    def one(k):
        v = cfg.get(k)
        return v[0] if isinstance(v, list) else v
    consts.load_properties(one("DET_PROPERTIES"), one("PIXEL_LAYOUT"),
                           one("RESPONSE"), one("SIM_PROPERTIES"))
    B = np.asarray(detector.TPC_BORDERS)[tpc]      # [[x0,x1],[y0,y1],[z0,z1]]
    center = np.array([0.5*(B[0, 0] + B[0, 1]),
                       0.5*(B[1, 0] + B[1, 1]),
                       B[2, 1]], float)            # z1 = cathode
    return center, 2, B


def main():
    ap = argparse.ArgumentParser(description="Translate a Bi-207 edep onto the fsd_cube cathode.")
    ap.add_argument("inp", help="input dumpTree Bi-207 EDEPSIM .h5 (module0 frame)")
    ap.add_argument("out", help="output EDEPSIM .h5 in the fsd_cube frame")
    ap.add_argument("--tpc", type=int, default=0, help="fsd_cube TPC index (default 0)")
    ap.add_argument("--inset", type=float, default=0.0,
                    help="push the source this many cm INTO the LAr from the cathode (toward the "
                         "anode) so more of the cloud is collected (default 0 = on the cathode plane)")
    ap.add_argument("--source", type=float, nargs=3, default=None,
                    help="override the source point (cm, input frame); default = mean primary vertex")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite existing {args.out}")

    import h5py
    with open_ro(args.inp) as h:
        names = list(h.keys())
        trj = np.asarray(h["trajectories"][:]) if "trajectories" in h else None
        seg = np.asarray(h["segments"][:]) if "segments" in h else None
    if seg is None or not seg.size:
        sys.exit("no 'segments' dataset in input.")

    # ---- source point: the decay vertex (primary = parent_id == -1), averaged over events ----
    if args.source is not None:
        src = np.asarray(args.source, float)
    elif trj is not None and "parent_id" in trj.dtype.names and "xyz_start" in trj.dtype.names:
        prim = trj["parent_id"] == -1
        if not prim.any():                         # fall back: the decaying ion (pdg>1e9)
            prim = trj["pdg_id"] > BIG
        src = np.asarray(trj["xyz_start"][prim], float).mean(axis=0)
    else:
        src = np.array([seg["x"].mean(), seg["y"].mean(), seg["z"].mean()], float)

    center, drift_ax, B = fsd_cube_cathode_center(args.tpc)
    target = center.copy()
    # inset moves the source toward the anode (z0) from the cathode (z1) along the drift axis
    z0, z1 = float(B[drift_ax, 0]), float(B[drift_ax, 1])
    target[drift_ax] = z1 - np.sign(z1 - z0) * args.inset
    shift = target - src
    print(f"source point (input frame)   : {src}")
    print(f"fsd_cube cathode center      : {center}   (TPC {args.tpc}, borders x{B[0]} y{B[1]} z{B[2]})")
    print(f"target (with inset {args.inset} cm): {target}")
    print(f"shift applied                : {shift}")

    # ---- write a translated copy ----
    axis_of = {"x": 0, "y": 1, "z": 2}
    seg2 = seg.copy()
    for f in SEG_XYZ:
        if f in seg2.dtype.names:
            seg2[f] = seg2[f] + shift[axis_of[f[0]]]    # f[0] is x/y/z -> drift/plane axis

    # simulate_pixels reads tracks['file_traj_id'] (a file-global unique trajectory id). The
    # /gps/ion -> dumpTree path emits only per-event 'traj_id', not 'file_traj_id', so synthesize
    # one: a dense 0..K-1 id per unique (event_id, traj_id) pair, unique across the whole file.
    if "file_traj_id" not in seg2.dtype.names:
        if "traj_id" in seg2.dtype.names:
            ev = (seg2["event_id"].astype(np.int64) if "event_id" in seg2.dtype.names
                  else np.zeros(seg2.size, np.int64))
            tj = seg2["traj_id"].astype(np.int64)
            _, ftid = np.unique(np.stack([ev, tj], axis=1), axis=0, return_inverse=True)
            ftid = np.ravel(ftid)                       # numpy>=2 may return a 2-D inverse
        else:
            ftid = np.arange(seg2.size)
        seg2 = rfn.append_fields(seg2, "file_traj_id", ftid.astype(np.int32), usemask=False)
        print(f"added file_traj_id ({len(np.unique(ftid))} unique trajectories)")

    # G4RadioactiveDecay samples each decay at a time drawn from Bi-207's ~31.6 yr half-life, so
    # every segment time carries a ~1e15 us (tens-of-years) offset -- and in float32 at that
    # magnitude all sub-second precision is already gone. simulate_pixels then rejects every segment
    # as "highly-delayed" (t0 > MAX_SEGMENT_T0) -> "Empty input dataset, exiting". Each edep event
    # is one prompt decay (cascade sub-ns vs us drift), so zero all segment time fields: the
    # calibration model is one trigger per decay at t0 = 0.
    SEG_T = ("t0", "t0_start", "t0_end", "t", "t_start", "t_end")
    zeroed = [tf for tf in SEG_T if tf in seg2.dtype.names]
    if zeroed:
        t0_was = float(seg2["t0"].max()) if "t0" in seg2.dtype.names else float("nan")
        for tf in zeroed:
            seg2[tf] = 0
        print(f"zeroed segment times {zeroed} (decay-time offset was up to {t0_was:.3e} us "
              f"~ {t0_was/3.156e13:.1f} yr = Bi-207 half-life)")

    trj2 = None
    if trj is not None:
        trj2 = trj.copy()
        for f in TRJ_XYZ:
            if f in trj2.dtype.names:
                v = np.asarray(trj2[f], float)
                v[:, 0] += shift[0]; v[:, 1] += shift[1]; v[:, 2] += shift[2]
                trj2[f] = v
        for tf in ("t_start", "t_end", "t"):      # same decay-time offset on the trajectories
            if tf in trj2.dtype.names:
                trj2[tf] = 0

    # fraction that will survive inside the fsd_cube volume (what simulate_pixels will keep)
    inside = np.ones(seg2.size, bool)
    for ax, f in ((0, "x"), (1, "y"), (2, "z")):
        if f in seg2.dtype.names:
            lo, hi = sorted((float(B[ax, 0]), float(B[ax, 1])))
            inside &= (seg2[f] >= lo) & (seg2[f] <= hi)
    print(f"segments inside fsd_cube after shift: {int(inside.sum())}/{seg2.size} "
          f"({100*inside.mean():.1f}%)  (the rest escape the cube and are dropped at drift)")

    with h5py.File(args.inp, "r") as hin, h5py.File(args.out, "w") as hout:
        for name in names:
            if name == "segments":
                d = hout.create_dataset("segments", data=seg2)
            elif name == "trajectories" and trj2 is not None:
                d = hout.create_dataset("trajectories", data=trj2)
            else:
                d = hout.create_dataset(name, data=np.asarray(hin[name][:]))
            for k, v in hin[name].attrs.items():
                d.attrs[k] = v
        for k, v in hin.attrs.items():
            hout.attrs[k] = v
    print(f"\nwrote {args.out}")
    print("next: simulate_pixels --config=fsd_cube on this file, then ndlar_flow.")


if __name__ == "__main__":
    main()
