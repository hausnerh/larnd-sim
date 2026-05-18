#!/usr/bin/env python
"""
test_charge_sharing.py
======================

Simpler, more physically intuitive demonstration of issue 1.

PHYSICS PICTURE
---------------
A 10 cm long track runs parallel to the y-axis at a fixed x position.
We scan the track's x position from one pixel center to the next,
crossing one pixel boundary in the middle. At each x position we
measure the total charge collected on THREE pixel columns:

    L = pixel column at x_index - 1   (left neighbor)
    C = pixel column at x_index       (central pixel)
    R = pixel column at x_index + 1   (right neighbor)

"column" = sum over all y-pixels in that x-column. Because the track
is long in y, this integrates over many y-pixels and removes y-axis
discretization as a variable. The only thing changing across the scan
is the source x.

EXPECTED BEHAVIOR
-----------------
Q_L(x_src), Q_C(x_src), Q_R(x_src) should form three smooth S-curves
that hand off charge between columns as x_src crosses pixel
boundaries. The TOTAL Q_L + Q_C + Q_R should be FLAT across the scan
(charge conservation: total collected = total deposited, modulo
response normalization).

If issue 1 is real, Q_total has small (~few %) BUMPS at exactly
x_src = ±pitch/2 (the pixel boundaries). Those bumps are the smoking
gun.

OUTPUTS
-------
  charge_sharing_per_pixel.png   - Q_L, Q_C, Q_R vs x_src
  charge_sharing_total.png       - Q_total vs x_src  (the money plot)
  charge_sharing_normalized.png  - Q_total / Q_total(x_src=0)
  charge_sharing_results.npz     - raw arrays

REQUIREMENTS
------------
  * CUDA GPU
  * larnd-sim install (run from repo root)

USAGE
-----
    python tests/test_charge_sharing.py

Defaults to the module0 config.
"""

import argparse
import sys
from math import ceil

import numpy as np


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
    return np.zeros(n, dtype=SEGMENTS_DTYPE).view(np.recarray)


def build_track(detector, x_src, y_center=0.0, half_len_y=5.0,
                drift_cm=10.0, dEdx=2.0):
    """One 10 cm track parallel to the y-axis at x = x_src."""
    t = make_blank_tracks(1)
    z_anode = detector.TPC_BORDERS[0][2][0]
    z_cathode = detector.TPC_BORDERS[0][2][1]
    into = np.sign(z_cathode - z_anode)
    z_pos = z_anode + into * drift_cm
    t["x_start"] = x_src
    t["x_end"] = x_src
    t["y_start"] = y_center - half_len_y
    t["y_end"] = y_center + half_len_y
    t["z_start"] = z_pos
    t["z_end"] = z_pos
    t["x"] = x_src
    t["y"] = y_center
    t["z"] = z_pos
    t["dx"] = 2.0 * half_len_y
    t["dEdx"] = dEdx
    t["dE"] = dEdx * t["dx"]
    t["pdg_id"] = 13
    t["segment_id"] = 0
    t["event_id"] = 0
    t["traj_id"] = 0
    t["pixel_plane"] = 0
    t["t0"] = 0
    t["t0_start"] = 0
    t["t0_end"] = 0
    t["tran_diff"] = 1e-2
    t["long_diff"] = 1e-2
    return t


def run_one(detector, physics, detsim, drifting, quenching,
            pixels_from_track, sim, create_xoroshiro128p_states,
            tracks, response, *, forced_diff=None, seed=42,
            MAX_PIXELS=800, MAX_ACTIVE_PIXELS=60):
    """Run the kernel chain and return (signals, neighboring_pixels,
    deposited_e). signals.shape = (1, MAX_PIXELS, n_ticks).
    neighboring_pixels[0, ipix] = pixel ID."""
    tracks = np.copy(tracks)
    tpb = 128
    bpg = ceil(tracks.shape[0] / tpb)

    from numba import cuda as _cuda
    d_tracks = _cuda.to_device(tracks)
    quenching.quench[bpg, tpb](d_tracks, physics.BOX)
    drifting.drift[bpg, tpb](d_tracks)
    tracks = d_tracks.copy_to_host()
    if forced_diff is not None:
        tracks["tran_diff"] = forced_diff
        tracks["long_diff"] = forced_diff
    d_tracks = _cuda.to_device(tracks)

    deposited_e = float(np.sum(tracks["n_electrons"]))
    if deposited_e <= 0:
        return None, None, 0.0

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

    pixels_from_track.get_pixels[bpg, tpb](d_tracks, d_active, d_neigh,
                                           d_radius, d_npix)
    neighboring_pixels = d_neigh.copy_to_host()
    n_pixels_list_h = d_npix.copy_to_host()
    if n_pixels_list_h[0] >= MAX_PIXELS:
        print(f"  WARNING: n_pixels {n_pixels_list_h[0]} reached "
              f"MAX_PIXELS {MAX_PIXELS}; raise MAX_PIXELS.")

    # Time-axis sizing as in cli/simulate_pixels.py
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
    rng_states = create_xoroshiro128p_states(max(n_states, 1024), seed=seed)
    d_signals = _cuda.to_device(signals)
    detsim.tracks_current_mc[bpg3, tpb3](d_signals, d_neigh, d_tracks,
                                         response, rng_states)
    signals = d_signals.copy_to_host()
    return signals, neighboring_pixels, deposited_e


def column_charges(signals, neighboring_pixels, id2pixel_fn,
                   target_x_indices):
    """For each x_index in target_x_indices, sum signals across all
    pixels with that pixel-x index and across all time ticks.

    Returns:
        col_totals (dict): {x_index: total_charge}
        all_pixel_total (float): sum over ALL pixels in halo (for
            "leak past 3-column" diagnostic)
    """
    col_totals = {ix: 0.0 for ix in target_x_indices}
    all_pixel_total = 0.0
    for ipix in range(neighboring_pixels.shape[1]):
        pid = int(neighboring_pixels[0, ipix])
        if pid < 0:
            continue
        ix, iy, plane = id2pixel_fn(pid)
        pix_sum = float(signals[0, ipix].sum())
        all_pixel_total += pix_sum
        if ix in col_totals:
            col_totals[ix] += pix_sum
    return col_totals, all_pixel_total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response",
                    default="larndsim/bin/response_44_v2a_full.npz")
    ap.add_argument("--detector",
                    default="larndsim/detector_properties/module0.yaml")
    ap.add_argument("--pixel-layout",
                    default="larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
    ap.add_argument("--sim-properties",
                    default="larndsim/simulation_properties/singles_sim.yaml")
    ap.add_argument("--n-x", type=int, default=41,
                    help="x_src points across [-pitch, +pitch]")
    ap.add_argument("--n-reps", type=int, default=15,
                    help="seed replicas per point")
    ap.add_argument("--half-len-y", type=float, default=5.0,
                    help="half-length of the y track [cm]")
    ap.add_argument("--diffusion", type=float, default=0.01,
                    help="forced diffusion sigma [cm]; smaller = sharper")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available.")

    from numba.cuda.random import create_xoroshiro128p_states
    from larndsim import consts
    consts.load_properties(args.detector, args.pixel_layout,
                           args.response, args.sim_properties)
    from larndsim.consts import detector, physics, sim
    from larndsim import detsim, drifting, quenching, pixels_from_track

    response = detector.load_response(args.response)
    pitch = detector.PIXEL_PITCH
    bx = detector.TPC_BORDERS[0][0][0]
    by = detector.TPC_BORDERS[0][1][0]
    # Reference: central pixel of plane 0
    i0 = int(detector.N_PIXELS[0] // 2)
    j0 = int(detector.N_PIXELS[1] // 2)
    x_center_pixel = bx + (i0 + 0.5) * pitch
    y_center = by + (j0 + 0.5) * pitch
    # Three target columns in absolute pixel x-index
    target_x_indices = [i0 - 1, i0, i0 + 1]

    print(f"Config: pitch={pitch:.4f} cm, "
          f"x_center_pixel={x_center_pixel:.4f}, "
          f"y_center={y_center:.4f}")
    print(f"Target x columns (absolute pixel index): "
          f"{target_x_indices} (i.e. L=-1, C=0, R=+1 relative)")
    print(f"Track: 2 * {args.half_len_y} = "
          f"{2*args.half_len_y} cm long in y")
    print(f"Scan: x_src in [-pitch, +pitch] relative to x_center, "
          f"{args.n_x} points, {args.n_reps} seeds each, "
          f"diffusion = {args.diffusion} cm")

    id2pixel_fn = pixels_from_track.id2pixel.py_func

    common = dict(
        detector=detector, physics=physics, detsim=detsim,
        drifting=drifting, quenching=quenching,
        pixels_from_track=pixels_from_track, sim=sim,
        create_xoroshiro128p_states=create_xoroshiro128p_states,
        response=response,
    )

    # x_src scan, RELATIVE to x_center_pixel
    x_offsets = np.linspace(-pitch, +pitch, args.n_x)
    q_L = np.zeros((len(x_offsets), args.n_reps))
    q_C = np.zeros((len(x_offsets), args.n_reps))
    q_R = np.zeros((len(x_offsets), args.n_reps))
    q_all = np.zeros((len(x_offsets), args.n_reps))
    deposited = np.zeros(len(x_offsets))

    for ix, xoff in enumerate(x_offsets):
        x_src = x_center_pixel + xoff
        tracks = build_track(detector, x_src, y_center=y_center,
                             half_len_y=args.half_len_y)
        for r in range(args.n_reps):
            sigs, neigh, dep = run_one(
                tracks=tracks, forced_diff=args.diffusion,
                seed=10000 + ix * 100 + r, **common)
            deposited[ix] = dep
            if sigs is None:
                continue
            cols, all_sum = column_charges(
                sigs, neigh, id2pixel_fn, target_x_indices)
            q_L[ix, r] = cols[target_x_indices[0]]
            q_C[ix, r] = cols[target_x_indices[1]]
            q_R[ix, r] = cols[target_x_indices[2]]
            q_all[ix, r] = all_sum
        print(f"  x_src/pitch = {xoff/pitch:+.3f}: "
              f"Q_L={q_L[ix].mean():.3e}  "
              f"Q_C={q_C[ix].mean():.3e}  "
              f"Q_R={q_R[ix].mean():.3e}  "
              f"Q_all={q_all[ix].mean():.3e}")

    # Stats
    L_m = q_L.mean(axis=1); L_se = q_L.std(axis=1, ddof=1) / np.sqrt(args.n_reps)
    C_m = q_C.mean(axis=1); C_se = q_C.std(axis=1, ddof=1) / np.sqrt(args.n_reps)
    R_m = q_R.mean(axis=1); R_se = q_R.std(axis=1, ddof=1) / np.sqrt(args.n_reps)
    A_m = q_all.mean(axis=1); A_se = q_all.std(axis=1, ddof=1) / np.sqrt(args.n_reps)
    T_m = L_m + C_m + R_m
    T_se = np.sqrt(L_se**2 + C_se**2 + R_se**2)

    # ---- Save raw ----
    np.savez(f"{args.outdir}/charge_sharing_results.npz",
             x_offsets=x_offsets, pitch=pitch,
             q_L=q_L, q_C=q_C, q_R=q_R, q_all=q_all,
             deposited=deposited)
    print(f"wrote {args.outdir}/charge_sharing_results.npz")

    # ---- Plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: per-column S-curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x_offsets / pitch, L_m, yerr=L_se,
                marker="o", ms=4, capsize=2, color="C0",
                label="Q_L (left column, i_x = -1)")
    ax.errorbar(x_offsets / pitch, C_m, yerr=C_se,
                marker="s", ms=4, capsize=2, color="C2",
                label="Q_C (central column, i_x = 0)")
    ax.errorbar(x_offsets / pitch, R_m, yerr=R_se,
                marker="^", ms=4, capsize=2, color="C3",
                label="Q_R (right column, i_x = +1)")
    ax.axvline(-0.5, color="k", lw=0.6, ls=":",
               label="pixel boundaries")
    ax.axvline(+0.5, color="k", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xlabel("track x position  [pixel pitch from central pixel]")
    ax.set_ylabel("collected charge on column  [response units]")
    ax.set_title("Charge sharing across three pixel columns\n"
                 "Track is 10 cm along y at fixed x; each column = "
                 "sum over all y pixels")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/charge_sharing_per_pixel.png", dpi=140)
    print(f"wrote {args.outdir}/charge_sharing_per_pixel.png")
    plt.close(fig)

    # Plot 2: total (THE money plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x_offsets / pitch, T_m, yerr=T_se,
                marker="o", ms=4, capsize=2, color="C1",
                label="Q_L + Q_C + Q_R (sum of 3 columns)")
    ax.errorbar(x_offsets / pitch, A_m, yerr=A_se,
                marker="s", ms=3, capsize=2, color="C4", alpha=0.6,
                label="Q over ALL halo pixels (leak diagnostic)")
    ax.axvline(-0.5, color="r", lw=0.8, ls=":",
               label="pixel boundary (issue-1 bump location)")
    ax.axvline(+0.5, color="r", lw=0.8, ls=":")
    ax.set_xlabel("track x position  [pixel pitch]")
    ax.set_ylabel("total collected charge  [response units]")
    ax.set_title("Total collected vs x_src — flat ⇒ charge conserved;\n"
                 "bumps at ±0.5 pitch ⇒ issue-1 boundary excess "
                 "(NN-undersampling of the response)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/charge_sharing_total.png", dpi=140)
    print(f"wrote {args.outdir}/charge_sharing_total.png")
    plt.close(fig)

    # Plot 3: normalized to centre
    ic = int(np.argmin(np.abs(x_offsets)))   # closest to x_offset=0
    norm = T_m / T_m[ic]
    norm_se = T_se / T_m[ic]
    norm_all = A_m / A_m[ic]
    norm_all_se = A_se / A_m[ic]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x_offsets / pitch, norm, yerr=norm_se,
                marker="o", capsize=2, color="C1",
                label="3-column sum / 3-column sum(x=0)")
    ax.errorbar(x_offsets / pitch, norm_all, yerr=norm_all_se,
                marker="s", capsize=2, color="C4", alpha=0.6,
                label="all-halo sum / all-halo sum(x=0)")
    ax.axhline(1.0, color="k", lw=0.6, ls="--",
               label="conservation (ideal)")
    ax.axvline(-0.5, color="r", lw=0.8, ls=":")
    ax.axvline(+0.5, color="r", lw=0.8, ls=":",
               label="pixel boundaries")
    # Annotate peak excess at each boundary
    boundary_idx_left = int(np.argmin(np.abs(x_offsets + pitch/2)))
    boundary_idx_right = int(np.argmin(np.abs(x_offsets - pitch/2)))
    bump_L = (norm[boundary_idx_left] - 1) * 100
    bump_R = (norm[boundary_idx_right] - 1) * 100
    ax.text(0.02, 0.97,
            f"bump at -0.5 pitch: {bump_L:+.2f}%\n"
            f"bump at +0.5 pitch: {bump_R:+.2f}%",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    ax.set_xlabel("track x position  [pixel pitch]")
    ax.set_ylabel("collected / collected(x_src = 0)")
    ax.set_title("Charge conservation check\n"
                 "Bumps at ±0.5 pitch are the issue-1 boundary excess; "
                 "compare to expected +2.6%")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/charge_sharing_normalized.png", dpi=140)
    print(f"wrote {args.outdir}/charge_sharing_normalized.png")
    plt.close(fig)

    # Console summary
    print("\n" + "=" * 60)
    print("Boundary-bump magnitudes (3-column sum, normalized to x=0):")
    print(f"  at x_src = -0.5 pitch: {bump_L:+.2f}%")
    print(f"  at x_src = +0.5 pitch: {bump_R:+.2f}%")
    print(f"Mid-pixel deviations (should be small):")
    quart_L = int(np.argmin(np.abs(x_offsets + pitch/4)))
    quart_R = int(np.argmin(np.abs(x_offsets - pitch/4)))
    print(f"  at x_src = -0.25 pitch: {(norm[quart_L]-1)*100:+.2f}%")
    print(f"  at x_src = +0.25 pitch: {(norm[quart_R]-1)*100:+.2f}%")
    print(f"Leak past 3 columns at x=0: "
          f"{(A_m[ic] - T_m[ic]) / A_m[ic] * 100:+.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
