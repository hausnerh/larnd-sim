#!/usr/bin/env python
"""
test_issue1_charge_conservation.py
==================================

Empirical check of "issue 1": per-pixel charge accumulation in
``larndsim.detsim.tracks_current_mc`` may fail to conserve charge for
segments that sit on/near a pixel boundary, because of the half-bin
rounding in ``get_closest_waveform`` interacting with the bipolar
response tail.

WHAT THIS SCRIPT DOES
---------------------
It builds *synthetic single-segment* inputs and runs the REAL CUDA
kernels:

    quench  ->  drift  ->  tracks_current_mc  ->  sum_pixel_signals

then sums the resulting per-pixel signal over all pixels and ticks.

IMPORTANT ABOUT UNITS / METRIC
------------------------------
tracks_current_mc writes  signals = charge * response  where
charge ~ n_electrons. The response table is NOT normalized to 1 per
electron (its central time-integral is ~20 for response_44_v2a_full),
so sum(signals)/n_electrons is an arbitrary response-dependent CONSTANT
-- it is NOT expected to be 1, and an earlier version of this script
that divided by E_CHARGE (copied from a stale in-repo test for a
different, removed kernel) produced meaningless ~1e17 numbers.

Issue 1 is a claim about POSITION DEPENDENCE, not absolute scale. So
the metric here is the collected sum NORMALIZED to its own value with
the segment on a pixel CENTER. A flat curve == charge collected
position-independently (issue 1 not reproduced). Structure localized
near offset 0.5 (the pixel boundary) == a real position-dependent
effect consistent with issue 1.

A sanity diagnostic runs first: a centered, well-diffused segment must
give a stable POSITIVE collected sum. If it doesn't, the harness is
unsound and the script says so instead of pretending to a conclusion.

It runs:

  (A) Offset scan: a short segment stepped across one pixel pitch at
      several diffusion widths; each row self-normalized to its center.

  (B) Boundary/center ratio vs. diffusion width. If a boundary effect
      exists it should wash out as diffusion grows.

OUTPUTS
-------
  issue1_offset_scan.png       - collected/deposited vs transverse offset
  issue1_diffusion_scan.png    - centered vs boundary vs diffusion
  issue1_results.npz           - raw arrays for further analysis

REQUIREMENTS
------------
  * A CUDA-capable GPU (numba.cuda.is_available() must be True)
  * cupy, numba, numpy, matplotlib
  * Run from the top level of a larnd-sim checkout. The defaults match
    the `module0` config, so this is usually enough:

        python test_issue1_charge_conservation.py

    or fully specified:

        python test_issue1_charge_conservation.py \
            --response      larndsim/bin/response_44_v2a_full.npz \
            --detector      larndsim/detector_properties/module0.yaml \
            --pixel-layout  larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml \
            --sim-properties larndsim/simulation_properties/singles_sim.yaml

NOTE ON SCOPE
-------------
This exercises the charge-induction stage only (tracks_current_mc +
sum_pixel_signals), not the FEE digitization, because issue 1 lives in
the current binning. A deficit here is upstream of and independent from
the discriminator logic.
"""

import argparse
import sys
from math import ceil

import numpy as np


# ---------------------------------------------------------------------------
# Track/segment record dtype.
#
# This MUST match the schema larnd-sim actually consumes. The authoritative
# definition is `segments_dtype` in cli/dumpTree.py. We inline a copy here
# rather than importing it, because cli/dumpTree.py does `from ROOT import ...`
# at module scope and ROOT is generally not installed on a GPU compute node.
#
# If this ever drifts again, the symptom is a Numba TypingError of the form
# "Field '<name>' was not found in record" -- re-sync the list below with
# cli/dumpTree.py:segments_dtype (keep align=True).
# ---------------------------------------------------------------------------
SEGMENTS_DTYPE = np.dtype([
    ("event_id", "u4"), ("vertex_id", "u8"), ("file_vertex_id", "u8"),
    ("segment_id", "u4"), ("z_end", "f4"), ("traj_id", "i4"),
    ("file_traj_id", "u4"), ("tran_diff", "f4"), ("z_start", "f4"),
    ("x_end", "f4"), ("y_end", "f4"), ("n_electrons", "u4"),
    ("pdg_id", "i4"), ("x_start", "f4"), ("y_start", "f4"),
    ("t_start", "f4"), ("t0_start", "f8"), ("t0_end", "f8"),
    ("t0", "f8"), ("dx", "f4"), ("long_diff", "f4"),
    ("pixel_plane", "i4"), ("t_end", "f4"), ("dEdx", "f4"),
    ("dE", "f4"), ("t", "f4"), ("y", "f4"), ("x", "f4"),
    ("z", "f4"), ("n_photons", "f4"),
], align=True)


def make_blank_tracks(n):
    # np.recarray view so both t["field"] and t[i]["field"] work, matching
    # how the kernels index the array.
    return np.zeros(n, dtype=SEGMENTS_DTYPE).view(np.recarray)


def build_segment(detector, x_center, y_center, *,
                  half_len_y=0.15,    # cm; segment runs along +/- y
                  dEdx=2.0,           # MeV/cm
                  drift_cm=10.0,      # cm from the anode (realistic mid-TPC)
                  forced_diff=None):
    """One straight segment parallel to the anode, centered at (x_center,
    y_center), running along y.

    The segment is placed at a *realistic* drift distance (~10 cm), not
    hard against the anode. An earlier version used a 0.05 cm margin to
    minimise intrinsic diffusion, but that put the collection time in the
    very early part of the bipolar response table where the kernel sums
    mostly negative lobes -> unstable / negative total charge, which is a
    harness artifact, not physics. With a realistic drift the response is
    sampled in its valid region; we instead control diffusion explicitly
    via `forced_diff` when we need it small.
    """
    t = make_blank_tracks(1)

    z_anode = detector.TPC_BORDERS[0][2][0]
    # sign of (cathode - anode) tells us which way "into the drift volume" is
    z_cathode = detector.TPC_BORDERS[0][2][1]
    into = np.sign(z_cathode - z_anode)
    z_pos = z_anode + into * drift_cm

    t["x_start"] = x_center
    t["x_end"] = x_center
    t["y_start"] = y_center - half_len_y
    t["y_end"] = y_center + half_len_y
    t["z_start"] = z_pos
    t["z_end"] = z_pos
    t["x"] = x_center
    t["y"] = y_center
    t["z"] = z_pos
    t["dx"] = np.sqrt((t["x_end"] - t["x_start"])**2 +
                      (t["y_end"] - t["y_start"])**2 +
                      (t["z_end"] - t["z_start"])**2)
    t["dEdx"] = dEdx
    t["dE"] = dEdx * t["dx"]
    t["pdg_id"] = 13          # muon-like: a real ionizing particle
    t["segment_id"] = 0       # read by detsim backtracking; single segment
    t["event_id"] = 0
    t["traj_id"] = 0
    t["pixel_plane"] = 0
    t["t0"] = 0
    t["t0_start"] = 0
    t["t0_end"] = 0
    # quench/drift overwrite these; seed sane non-zero values
    t["tran_diff"] = 1e-2
    t["long_diff"] = 1e-2
    return t, forced_diff


def run_one(detector, physics, detsim, drifting, quenching, pixels_from_track,
            sim, create_xoroshiro128p_states,
            tracks, response, forced_diff=None):
    """Run quench->drift->tracks_current_mc->sum_pixel_signals for a single
    one-segment `tracks` array and return (collected_e, deposited_e).
    """
    tracks = np.copy(tracks)
    tpb = 128
    bpg = ceil(tracks.shape[0] / tpb)

    # Explicitly stage arrays on the device. quench/drift mutate `tracks`
    # in place, so we keep a single device copy and read fields back when
    # needed. This mirrors cli/simulate_pixels.py and removes the
    # host-array auto-transfer ambiguity (the "Host array used in CUDA
    # kernel" warnings) -- those are perf-only here, but being explicit
    # guarantees written results actually come back.
    from numba import cuda as _cuda
    d_tracks = _cuda.to_device(tracks)

    quenching.quench[bpg, tpb](d_tracks, physics.BOX)
    drifting.drift[bpg, tpb](d_tracks)

    tracks = d_tracks.copy_to_host()
    if forced_diff is not None:
        # Override diffusion to isolate the offset effect from drift distance.
        tracks["tran_diff"] = forced_diff
        tracks["long_diff"] = forced_diff
        d_tracks = _cuda.to_device(tracks)

    deposited_e = float(np.sum(tracks["n_electrons"]))
    if deposited_e <= 0:
        return 0.0, 0.0

    MAX_PIXELS = 110
    MAX_ACTIVE_PIXELS = 50
    active_pixels = np.full((tracks.shape[0], MAX_ACTIVE_PIXELS), -1,
                            dtype=np.int32)
    neighboring_pixels = np.full((tracks.shape[0], MAX_PIXELS), -1,
                                 dtype=np.int32)
    neighboring_radius = np.zeros((tracks.shape[0], MAX_PIXELS),
                                  dtype=np.float32)
    n_pixels_list = np.zeros(shape=(tracks.shape[0]), dtype=np.int64)

    d_active = _cuda.to_device(active_pixels)
    d_neigh = _cuda.to_device(neighboring_pixels)
    d_radius = _cuda.to_device(neighboring_radius)
    d_npix = _cuda.to_device(n_pixels_list)

    pixels_from_track.get_pixels[bpg, tpb](d_tracks,
                                           d_active,
                                           d_neigh,
                                           d_radius,
                                           d_npix)
    neighboring_pixels = d_neigh.copy_to_host()

    # Size the signal time axis exactly as cli/simulate_pixels.py does.
    # detector.TIME_TICKS was removed on recent develop; the production
    # code now computes a per-batch max_signal_time and ceils it.
    # RESPONSE_MAX_TIME / DRIFT_MAX_TIME are globals set by
    # detector.load_response() / set_detector_properties().
    long_diff_max = float(np.max(tracks["long_diff"]))
    t_span = float(np.max(tracks["t_end"] - tracks["t0"]))
    diff_pad = (long_diff_max / detector.V_DRIFT
                * detector.DIFF_N_SIGMAS)
    if detector.RESPONSE_MAX_TIME > detector.DRIFT_MAX_TIME:
        max_signal_time = (t_span + diff_pad
                           + detector.RESPONSE_MAX_TIME
                           - detector.DRIFT_MAX_TIME)
    else:
        max_signal_time = t_span + diff_pad
    signals_ticks = max(ceil(max_signal_time / detector.TIME_SAMPLING), 1)

    signals = np.zeros((tracks.shape[0],
                        neighboring_pixels.shape[1],
                        signals_ticks), dtype=np.float32)

    tpb3 = (1, 1, 64)
    bpg3 = (ceil(signals.shape[0] / tpb3[0]),
            ceil(signals.shape[1] / tpb3[1]),
            ceil(signals.shape[2] / tpb3[2]))

    n_states = int(np.prod(tpb3) * bpg3[0] * bpg3[1] * bpg3[2])
    rng_states = create_xoroshiro128p_states(max(n_states, 1024), seed=12345)

    d_signals = _cuda.to_device(signals)
    # response came from detector.load_response() and is already a CuPy
    # device array; Numba accepts it via the CUDA array interface.
    detsim.tracks_current_mc[bpg3, tpb3](d_signals,
                                         d_neigh,
                                         d_tracks,
                                         response,
                                         rng_states)
    signals = d_signals.copy_to_host()

    # UNITS: tracks_current_mc writes
    #     signals = charge * response,   charge = n_electrons * frac / nstep
    # i.e. signals is ALREADY in electron-like units scaled by the
    # response table's internal normalization. There is NO physical
    # current here and NO amps->coulombs conversion. (The old
    # tests/testTracksCurrent.py multiplied by TIME_SAMPLING/E_CHARGE;
    # that was for a different, removed `tracks_current` kernel that
    # integrated a real current. Applying it here inflates the result
    # by ~1/E_CHARGE ~ 1e19 and is wrong.)
    #
    # The response table is NOT normalized to 1 per electron (its
    # central time-integral is ~20 for response_44_v2a_full.npz), so
    # collected/deposited is a response-dependent CONSTANT, not 1.
    # Issue 1 is about POSITION DEPENDENCE, not absolute scale, so we
    # return the raw collected sum and let the caller normalize each
    # scan to its own pixel-center value.
    collected = float(np.sum(signals))
    return collected, deposited_e


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # NOTE: defaults match the current `module0` entry in
    # larndsim/config/config.yaml. On recent develop, load_properties()
    # takes FOUR files and the response file is the bundled .npz
    # (not the legacy bare response_44.npy, which has no 'time_tick'/'response'
    # keys and will fail in detector.load_response()).
    ap.add_argument("--response",
                    default="larndsim/bin/response_44_v2a_full.npz",
                    help="bundled .npz response (keys: response, time_tick, ...)")
    ap.add_argument("--detector",
                    default="larndsim/detector_properties/module0.yaml")
    ap.add_argument("--pixel-layout",
                    default="larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
    ap.add_argument("--sim-properties",
                    default="larndsim/simulation_properties/singles_sim.yaml",
                    help="simulation-properties YAML (4th arg to load_properties)")
    ap.add_argument("--n-offsets", type=int, default=41,
                    help="points across one pixel pitch in the offset scan")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    # Import larnd-sim only after arg parsing so --help works without a GPU.
    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available. Run this on a GPU node.")

    from numba.cuda.random import create_xoroshiro128p_states
    from larndsim import consts
    # Current signature: load_properties(detprop, pixel, response, sim).
    # This also sets RESPONSE_SAMPLING / RESPONSE_BIN_SIZE globals.
    consts.load_properties(args.detector, args.pixel_layout,
                           args.response, args.sim_properties)
    from larndsim.consts import detector, physics, sim
    from larndsim import detsim, drifting, quenching, pixels_from_track

    # Use the canonical loader, NOT a raw np.load: it extracts the
    # 'response' array, chops/pads it to the detector drift length, and
    # (critically) sets the global detector.RESPONSE_MAX_TIME that
    # tracks_current_mc reads. A raw np.load leaves RESPONSE_MAX_TIME=None
    # and the kernel's time-window check then misbehaves.
    response = detector.load_response(args.response)
    pitch = detector.PIXEL_PITCH

    # A pixel center sits at:  border + (i + 0.5) * pitch  (i integer).
    # Put our reference near the middle of plane 0 and define:
    #   offset = 0     -> segment on a pixel CENTER
    #   offset = 0.5   -> segment on a pixel BOUNDARY (corner/edge symmetry)
    bx = detector.TPC_BORDERS[0][0][0]
    by = detector.TPC_BORDERS[0][1][0]
    i0 = int((detector.N_PIXELS[0] // 2))
    j0 = int((detector.N_PIXELS[1] // 2))
    x_center_pixel = bx + (i0 + 0.5) * pitch
    y_fixed = by + (j0 + 0.5) * pitch

    # ---- Sanity diagnostic FIRST -------------------------------------
    # Before any position scan, confirm a centered, well-diffused segment
    # gives a stable, positive collected sum. If this is negative or wild,
    # the harness is unsound and the position scans are meaningless.
    print("Sanity check: centered segment, diffusion = 0.05 cm")
    s_tr, _ = build_segment(detector, x_center_pixel, y_fixed)
    s_coll, s_dep = run_one(detector, physics, detsim, drifting,
                            quenching, pixels_from_track, sim,
                            create_xoroshiro128p_states,
                            s_tr, response, forced_diff=0.05)
    ratio_const = s_coll / s_dep if s_dep > 0 else float("nan")
    print("  deposited n_electrons : %.4e" % s_dep)
    print("  collected sum(signals): %.4e" % s_coll)
    print("  ratio (response-scaled constant, NOT expected to be 1): %.4f"
          % ratio_const)
    if not (np.isfinite(s_coll) and s_coll > 0):
        print("  !! WARNING: non-positive collected charge for a centered")
        print("     segment. The harness is unsound; position scans below")
        print("     should NOT be interpreted as evidence about issue 1.")
    print()

    # ---- Scan A: transverse offset across one pitch, several diffusions ----
    # METRIC: raw collected sum(signals). The absolute value is an
    # arbitrary response-normalized constant, so each diffusion row is
    # later normalized to ITS OWN value at the pixel center. Issue 1 is a
    # claim about POSITION DEPENDENCE: a flat normalized curve == charge
    # is collected position-independently; a dip or spike near offset 0.5
    # (pixel boundary) == position-dependent mis-collection.
    diffs_A = [0.01, 0.02, 0.05, 0.10, 0.20]  # cm; all > 0 (0 was unstable)
    offsets = np.linspace(0.0, 1.0, args.n_offsets)  # in units of pitch
    collA = np.full((len(diffs_A), len(offsets)), np.nan)

    print("Scan A: offset across one pixel pitch")
    for di, d in enumerate(diffs_A):
        for oi, off in enumerate(offsets):
            xc = x_center_pixel + off * pitch
            tracks, _ = build_segment(detector, xc, y_fixed)
            coll, dep = run_one(detector, physics, detsim, drifting,
                                quenching, pixels_from_track, sim,
                                create_xoroshiro128p_states,
                                tracks, response, forced_diff=d)
            collA[di, oi] = coll
        print("  diffusion=%.3f cm done" % d)

    # Normalize each row to its pixel-center (offset=0) value.
    normA = np.full_like(collA, np.nan)
    for di in range(len(diffs_A)):
        base = collA[di, 0]
        if np.isfinite(base) and base != 0:
            normA[di] = collA[di] / base

    # ---- Scan B: centered vs boundary segment vs diffusion width ----
    # Plot the RATIO boundary/center at each diffusion. If charge is
    # collected position-independently this is ~1 at all diffusions.
    diffs_B = np.array([0.01, 0.02, 0.03, 0.05, 0.075,
                        0.10, 0.15, 0.20, 0.30, 0.40])
    coll_center = np.full(len(diffs_B), np.nan)
    coll_bound = np.full(len(diffs_B), np.nan)

    print("Scan B: centered vs boundary vs diffusion")
    for k, d in enumerate(diffs_B):
        tr_c, _ = build_segment(detector, x_center_pixel, y_fixed)
        tr_b, _ = build_segment(detector, x_center_pixel + 0.5 * pitch,
                                y_fixed)
        c_coll, _ = run_one(detector, physics, detsim, drifting,
                            quenching, pixels_from_track, sim,
                            create_xoroshiro128p_states,
                            tr_c, response, forced_diff=d)
        b_coll, _ = run_one(detector, physics, detsim, drifting,
                            quenching, pixels_from_track, sim,
                            create_xoroshiro128p_states,
                            tr_b, response, forced_diff=d)
        coll_center[k] = c_coll
        coll_bound[k] = b_coll
        rB = b_coll / c_coll if (np.isfinite(c_coll) and c_coll != 0) \
            else float("nan")
        print("  diffusion=%.3f cm  boundary/center = %.3f" % (d, rB))

    bound_over_center = coll_bound / coll_center

    # ---- Save raw numbers ----
    out_npz = f"{args.outdir}/issue1_results.npz"
    np.savez(out_npz,
             offsets=offsets, diffs_A=np.array(diffs_A),
             collA=collA, normA=normA,
             diffs_B=diffs_B, coll_center=coll_center,
             coll_bound=coll_bound, bound_over_center=bound_over_center,
             ratio_const=ratio_const, pitch=pitch)
    print("wrote", out_npz)

    # ---- Plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for di, d in enumerate(diffs_A):
        ax.plot(offsets, normA[di], marker="o", ms=3,
                label=f"diffusion = {d:.3f} cm")
    ax.axhline(1.0, color="k", lw=0.8, ls="--",
               label="position-independent (ideal)")
    ax.axvline(0.5, color="r", lw=0.8, ls=":")
    ax.set_xlabel("transverse offset of segment  [pixel pitch]")
    ax.set_ylabel("collected charge  /  collected at pixel center")
    ax.set_title("Issue 1: position dependence of collected charge\n"
                 "(flat = OK; structure near 0.5 = boundary effect)")
    ax.text(0.5, ax.get_ylim()[0], "  pixel boundary",
            color="r", va="bottom", ha="left", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = f"{args.outdir}/issue1_offset_scan.png"
    fig.savefig(f1, dpi=140)
    print("wrote", f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diffs_B, bound_over_center, marker="s", color="C1")
    ax.axhline(1.0, color="k", lw=0.8, ls="--",
               label="position-independent (ideal)")
    ax.set_xlabel("diffusion width applied  [cm]")
    ax.set_ylabel("collected(boundary)  /  collected(center)")
    ax.set_title("Issue 1: boundary/center ratio vs. diffusion\n"
                 "(should approach 1 as diffusion grows if it's an edge effect)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = f"{args.outdir}/issue1_diffusion_scan.png"
    fig.savefig(f2, dpi=140)
    print("wrote", f2)

    # ---- Console verdict ----
    print("\n================ SUMMARY ================")
    print("Response-normalization constant (centered, d=0.05):")
    print("  collected/deposited = %.3f  (arbitrary; not expected ~1)"
          % ratio_const)
    print()
    if not (np.isfinite(s_coll) and s_coll > 0):
        print("HARNESS UNSOUND: centered sanity case was non-positive.")
        print("Do not interpret the plots as evidence about issue 1.")
        print("=========================================")
        return
    # Use the smallest-diffusion row as the cleanest position probe.
    row = normA[0]
    finite = row[np.isfinite(row)]
    if finite.size:
        dev = np.nanmax(np.abs(row - 1.0))
        near = np.nanmax(np.abs(row[(offsets > 0.4) & (offsets < 0.6)] - 1.0))
        print("Position dependence (diffusion = %.3f cm row):" % diffs_A[0])
        print("  max |norm - 1| over full pitch : %.3f" % dev)
        print("  max |norm - 1| near boundary   : %.3f" % near)
        if near > 0.15 and near >= 0.5 * dev:
            print("  --> Significant position dependence concentrated near")
            print("      the pixel boundary. Consistent with issue 1.")
        elif dev > 0.15:
            print("  --> Position dependence present but not boundary-")
            print("      localized; cause unclear, needs investigation.")
        else:
            print("  --> Collected charge is position-independent to")
            print("      <15%%. Issue 1 not reproduced under these settings.")
    print("=========================================")


if __name__ == "__main__":
    main()
