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
Builds *synthetic single-segment* inputs and runs the REAL CUDA kernels:

    quench  ->  drift  ->  tracks_current_mc

then sums the resulting per-pixel signal over all pixels and ticks.

This is a *machinery exerciser*, not just a single test. In addition to
the original offset/diffusion scans for issue 1, it runs cross-checks
of the surrounding kernels (linearity, pitch periodicity, symmetry,
drift on/off, diffusion vs. drift distance, spatial spread, time
profile, RNG noise floor) so that any anomaly is judged against an
explicit baseline rather than a single point estimate.

IMPORTANT ABOUT UNITS / METRIC
------------------------------
tracks_current_mc writes  signals = charge * response  where
charge ~ n_electrons. The response table is NOT normalized to 1 per
electron (its central time-integral is ~20 for response_44_v2a_full),
so sum(signals)/n_electrons is an arbitrary response-dependent CONSTANT
-- it is NOT expected to be 1. An earlier version of this script
divided by E_CHARGE (copied from a stale in-repo test for a different,
removed kernel); that inflated results by ~1/E_CHARGE ~ 1e19 and was
wrong. Issue 1 is a claim about POSITION DEPENDENCE, not absolute
scale, so each scan is self-normalized to its own pixel-center value.

A sanity diagnostic runs first: a centered, well-diffused segment must
give a stable POSITIVE collected sum. If it does not, the harness is
declared unsound and the rest is skipped.

OUTPUTS
-------
  issue1_offset_scan.png        - collected vs transverse offset (Scan A)
  issue1_diffusion_scan.png     - boundary/center vs diffusion (Scan B)
  issue1_noise_floor.png        - RNG-only stochastic floor
  issue1_drift_onoff.png        - drift kernel on vs off
  issue1_diffusion_vs_drift.png - tran_diff growth with drift distance
  issue1_linearity.png          - collected vs deposited (dEdx sweep)
  issue1_pitch_periodicity.png  - offsets 0, 1, 2 pitches
  issue1_symmetry.png           - +/- offset around center
  issue1_spatial_spread.png     - pixel multiplicity & RMS vs diffusion
  issue1_time_profile.png       - sum vs tick at center
  issue1_results.npz            - raw arrays for downstream analysis

REQUIREMENTS
------------
  * A CUDA-capable GPU (numba.cuda.is_available() must be True)
  * cupy, numba, numpy, matplotlib
  * Run from the top level of a larnd-sim checkout. Defaults match the
    `module0` config:

        python tests/test_issue1_charge_conservation.py

NOTE ON SCOPE
-------------
This exercises the charge-induction stage only (quench, drift,
tracks_current_mc), not sum_pixel_signals or FEE digitization. Issue 1
lives in the current binning, upstream of and independent from the
discriminator logic.
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
    return np.zeros(n, dtype=SEGMENTS_DTYPE).view(np.recarray)


def build_segment(detector, x_center, y_center, *,
                  half_len_y=0.15,    # cm; segment runs along +/- y
                  dEdx=2.0,           # MeV/cm
                  drift_cm=10.0,      # cm from anode (realistic mid-TPC)
                  forced_diff=None):
    """One straight segment parallel to the anode, centered at
    (x_center, y_center), running along y.

    Realistic drift distance (~10 cm by default) keeps the collection
    time in the valid central region of the response table. An earlier
    version used a 0.05 cm margin to minimize intrinsic diffusion, but
    that sampled the very early bipolar lobe and produced negative
    totals -- harness artifact, not physics.
    """
    t = make_blank_tracks(1)

    z_anode = detector.TPC_BORDERS[0][2][0]
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
    t["pdg_id"] = 13          # muon-like
    t["segment_id"] = 0
    t["event_id"] = 0
    t["traj_id"] = 0
    t["pixel_plane"] = 0
    t["t0"] = 0
    t["t0_start"] = 0
    t["t0_end"] = 0
    t["tran_diff"] = 1e-2
    t["long_diff"] = 1e-2
    return t, forced_diff


def _populate_drift_fields_manually(detector, tracks):
    """Replicate drifting.drift() output WITHOUT the kernel: deterministic
    drift_time and zero diffusion. Used by the drift-off scan and any
    `skip_drift=True` runs.

    Lifetime attenuation is also skipped here (set lifetime_red=1) so the
    drift-on/off comparison isolates the *diffusion* contribution rather
    than confounding it with attenuation.
    """
    for i in range(tracks.shape[0]):
        # pick plane: same bbox check drifting.drift uses
        plane_idx = detector.DEFAULT_PLANE_INDEX
        for ip, plane in enumerate(detector.TPC_BORDERS):
            if (plane[0][0] - 2e-2 <= tracks[i]["x"] <= plane[0][1] + 2e-2
                and plane[1][0] - 2e-2 <= tracks[i]["y"] <= plane[1][1] + 2e-2
                and min(plane[2][1] - 2e-2, plane[2][0] - 2e-2)
                    <= tracks[i]["z"]
                    <= max(plane[2][1] + 2e-2, plane[2][0] + 2e-2)):
                plane_idx = ip
                break
        tracks[i]["pixel_plane"] = plane_idx
        if plane_idx == detector.DEFAULT_PLANE_INDEX:
            continue

        z_anode = detector.TPC_BORDERS[plane_idx][2][0]
        drift_distance = abs(tracks[i]["z"] - z_anode)
        drift_start = abs(min(tracks[i]["z_start"], tracks[i]["z_end"])
                          - z_anode)
        drift_end = abs(max(tracks[i]["z_start"], tracks[i]["z_end"])
                        - z_anode)
        drift_time = drift_distance / detector.V_DRIFT
        # no lifetime attenuation in drift-off mode
        tracks[i]["long_diff"] = 0.0
        tracks[i]["tran_diff"] = 0.0
        tracks[i]["t"] = drift_time + tracks[i]["t0"]
        tracks[i]["t_start"] = (min(drift_start, drift_end)
                                / detector.V_DRIFT + tracks[i]["t0"])
        tracks[i]["t_end"] = (max(drift_start, drift_end)
                              / detector.V_DRIFT + tracks[i]["t0"])


def run_one(detector, physics, detsim, drifting, quenching, pixels_from_track,
            sim, create_xoroshiro128p_states,
            tracks, response, *,
            forced_diff=None, seed=12345, skip_drift=False,
            return_signals=False):
    """quench -> [drift|manual] -> tracks_current_mc for a single
    one-segment tracks array.

    Returns:
        if return_signals=False: (collected_e, deposited_e)
        if return_signals=True:  (collected_e, deposited_e, signals,
                                   pixel_sum)
            where pixel_sum[ipix] = sum over time ticks of signals on
            the ipix-th neighboring pixel of segment 0.
    """
    tracks = np.copy(tracks)
    tpb = 128
    bpg = ceil(tracks.shape[0] / tpb)

    from numba import cuda as _cuda
    d_tracks = _cuda.to_device(tracks)

    quenching.quench[bpg, tpb](d_tracks, physics.BOX)
    if not skip_drift:
        drifting.drift[bpg, tpb](d_tracks)

    tracks = d_tracks.copy_to_host()

    if skip_drift:
        _populate_drift_fields_manually(detector, tracks)

    if forced_diff is not None:
        tracks["tran_diff"] = forced_diff
        tracks["long_diff"] = forced_diff

    d_tracks = _cuda.to_device(tracks)

    deposited_e = float(np.sum(tracks["n_electrons"]))
    if deposited_e <= 0:
        if return_signals:
            return 0.0, 0.0, None, None
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

    # Size signal time axis as cli/simulate_pixels.py does.
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
    detsim.tracks_current_mc[bpg3, tpb3](d_signals,
                                         d_neigh,
                                         d_tracks,
                                         response,
                                         rng_states)
    signals = d_signals.copy_to_host()

    collected = float(np.sum(signals))
    if return_signals:
        pixel_sum = signals[0].sum(axis=-1)  # per pixel, integrated in time
        return collected, deposited_e, signals, pixel_sum
    return collected, deposited_e


def run_ensemble(detector, physics, detsim, drifting, quenching,
                 pixels_from_track, sim, create_xoroshiro128p_states,
                 tracks, response, *, n_reps, base_seed=1000, **kwargs):
    """Run `run_one` n_reps times with distinct seeds. Returns
    (mean, stderr, all) of `collected` over the ensemble."""
    vals = np.zeros(n_reps)
    for r in range(n_reps):
        c, _ = run_one(detector, physics, detsim, drifting, quenching,
                       pixels_from_track, sim, create_xoroshiro128p_states,
                       tracks, response, seed=base_seed + r, **kwargs)
        vals[r] = c
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n_reps), vals


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("wrote", path)


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
    ap.add_argument("--n-offsets", type=int, default=21,
                    help="points across one pixel pitch in offset scan")
    ap.add_argument("--n-reps", type=int, default=5,
                    help="seed replicas per data point (ensemble size)")
    ap.add_argument("--skip-extras", action="store_true",
                    help="run only the core issue-1 scans (A, B, sanity)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available. Run this on a GPU node.")

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
    i0 = int((detector.N_PIXELS[0] // 2))
    j0 = int((detector.N_PIXELS[1] // 2))
    x_center_pixel = bx + (i0 + 0.5) * pitch
    y_fixed = by + (j0 + 0.5) * pitch

    print(f"Configuration: pitch={pitch:.4f} cm, "
          f"V_DRIFT={detector.V_DRIFT:.4e} cm/us, "
          f"RESPONSE_MAX_TIME={detector.RESPONSE_MAX_TIME}, "
          f"DRIFT_MAX_TIME={detector.DRIFT_MAX_TIME}, "
          f"TIME_SAMPLING={detector.TIME_SAMPLING}, "
          f"DIFF_N_SIGMAS={detector.DIFF_N_SIGMAS}")
    print(f"n_reps={args.n_reps}, n_offsets={args.n_offsets}, "
          f"skip_extras={args.skip_extras}")
    print()

    common = dict(
        detector=detector, physics=physics, detsim=detsim,
        drifting=drifting, quenching=quenching,
        pixels_from_track=pixels_from_track, sim=sim,
        create_xoroshiro128p_states=create_xoroshiro128p_states,
        response=response,
    )

    # ====================================================================
    # SANITY GATE
    # ====================================================================
    print("Sanity check: centered segment, diffusion = 0.05 cm, "
          f"{args.n_reps} seeds")
    s_tr, _ = build_segment(detector, x_center_pixel, y_fixed)
    s_mean, s_se, s_vals = run_ensemble(tracks=s_tr, n_reps=args.n_reps,
                                        forced_diff=0.05, **common)
    s_dep = float(np.sum(s_tr["dE"]))  # before quench; rough scale only
    # actually-deposited electrons: re-run quench-only for the print
    _, dep0 = run_one(tracks=s_tr, forced_diff=0.05, seed=42, **common)
    ratio_const = s_mean / dep0 if dep0 > 0 else float("nan")
    print("  deposited n_electrons        : %.4e" % dep0)
    print("  collected sum(signals) mean  : %.4e" % s_mean)
    print("  collected sum(signals) stderr: %.4e (rel %.2e)"
          % (s_se, s_se / s_mean if s_mean != 0 else float("nan")))
    print("  per-seed values              :",
          ", ".join("%.3e" % v for v in s_vals))
    print("  ratio (response-scaled constant, NOT expected to be 1): %.4f"
          % ratio_const)
    sanity_ok = bool(np.isfinite(s_mean) and s_mean > 0)
    if not sanity_ok:
        print("  !! HARNESS UNSOUND: non-positive centered collected sum.")
        print("     Scans below will run but should NOT be interpreted")
        print("     as evidence about issue 1.")
    print()

    results = dict(ratio_const=ratio_const, pitch=pitch,
                   sanity_mean=s_mean, sanity_stderr=s_se,
                   sanity_vals=s_vals)

    # ====================================================================
    # SCAN A: transverse offset across one pitch, several diffusions
    # (ensembled)
    # ====================================================================
    diffs_A = [0.01, 0.02, 0.05, 0.10, 0.20]
    offsets = np.linspace(0.0, 1.0, args.n_offsets)
    collA_mean = np.full((len(diffs_A), len(offsets)), np.nan)
    collA_stderr = np.full_like(collA_mean, np.nan)

    print("Scan A: offset across one pitch (with seed ensemble)")
    for di, d in enumerate(diffs_A):
        for oi, off in enumerate(offsets):
            xc = x_center_pixel + off * pitch
            tracks, _ = build_segment(detector, xc, y_fixed)
            m, se, _ = run_ensemble(tracks=tracks, n_reps=args.n_reps,
                                    forced_diff=d, **common)
            collA_mean[di, oi] = m
            collA_stderr[di, oi] = se
        print("  diffusion=%.3f cm done" % d)

    normA = np.full_like(collA_mean, np.nan)
    normA_err = np.full_like(collA_mean, np.nan)
    for di in range(len(diffs_A)):
        base = collA_mean[di, 0]
        base_se = collA_stderr[di, 0]
        if np.isfinite(base) and base != 0:
            normA[di] = collA_mean[di] / base
            # ratio error: relative-errors-in-quadrature
            rel = np.where(collA_mean[di] != 0,
                           collA_stderr[di] / np.abs(collA_mean[di]), 0)
            rel0 = base_se / abs(base) if base else 0
            normA_err[di] = np.abs(normA[di]) * np.sqrt(rel**2 + rel0**2)

    results.update(offsets=offsets, diffs_A=np.array(diffs_A),
                   collA_mean=collA_mean, collA_stderr=collA_stderr,
                   normA=normA, normA_err=normA_err)

    # ====================================================================
    # SCAN B: boundary/center ratio vs diffusion (ensembled)
    # ====================================================================
    diffs_B = np.array([0.01, 0.02, 0.03, 0.05, 0.075,
                        0.10, 0.15, 0.20, 0.30, 0.40])
    coll_center = np.full(len(diffs_B), np.nan)
    coll_center_se = np.full_like(coll_center, np.nan)
    coll_bound = np.full_like(coll_center, np.nan)
    coll_bound_se = np.full_like(coll_center, np.nan)

    print("Scan B: centered vs boundary, ensembled vs diffusion")
    for k, d in enumerate(diffs_B):
        tr_c, _ = build_segment(detector, x_center_pixel, y_fixed)
        tr_b, _ = build_segment(detector,
                                x_center_pixel + 0.5 * pitch, y_fixed)
        cm, cse, _ = run_ensemble(tracks=tr_c, n_reps=args.n_reps,
                                  forced_diff=d, **common)
        bm, bse, _ = run_ensemble(tracks=tr_b, n_reps=args.n_reps,
                                  forced_diff=d, **common)
        coll_center[k] = cm
        coll_center_se[k] = cse
        coll_bound[k] = bm
        coll_bound_se[k] = bse
        rB = bm / cm if (np.isfinite(cm) and cm != 0) else float("nan")
        print("  diffusion=%.3f cm  b/c = %.3f +/- %.3f"
              % (d, rB,
                 abs(rB) * np.sqrt((bse / bm)**2 + (cse / cm)**2)
                 if (bm and cm) else float("nan")))

    bound_over_center = coll_bound / coll_center
    results.update(diffs_B=diffs_B,
                   coll_center=coll_center, coll_center_se=coll_center_se,
                   coll_bound=coll_bound, coll_bound_se=coll_bound_se,
                   bound_over_center=bound_over_center)

    # ====================================================================
    # PLOTS (always emit Scans A, B)
    # ====================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for di, d in enumerate(diffs_A):
        ax.errorbar(offsets, normA[di], yerr=normA_err[di],
                    marker="o", ms=3, capsize=2,
                    label=f"diffusion = {d:.3f} cm")
    ax.axhline(1.0, color="k", lw=0.8, ls="--",
               label="position-independent (ideal)")
    ax.axvline(0.5, color="r", lw=0.8, ls=":")
    ax.set_xlabel("transverse offset of segment  [pixel pitch]")
    ax.set_ylabel("collected charge  /  collected at pixel center")
    ax.set_title("Issue 1: position dependence of collected charge\n"
                 "(flat = OK; structure near 0.5 = boundary effect)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, f"{args.outdir}/issue1_offset_scan.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    rB_err = np.abs(bound_over_center) * np.sqrt(
        (coll_bound_se / coll_bound)**2 + (coll_center_se / coll_center)**2)
    ax.errorbar(diffs_B, bound_over_center, yerr=rB_err,
                marker="s", color="C1", capsize=2)
    ax.axhline(1.0, color="k", lw=0.8, ls="--",
               label="position-independent (ideal)")
    ax.set_xlabel("diffusion width applied  [cm]")
    ax.set_ylabel("collected(boundary)  /  collected(center)")
    ax.set_title("Issue 1: boundary/center ratio vs. diffusion")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, f"{args.outdir}/issue1_diffusion_scan.png")

    # ====================================================================
    # EXTRA MACHINERY CHECKS (skip if --skip-extras)
    # ====================================================================
    if not args.skip_extras:
        # -------- RNG noise floor: fixed geometry x many seeds --------
        # Quantifies stochastic baseline. Position-dependence signal must
        # exceed this floor to count.
        print("Extra: RNG noise floor (centered, d=0.05, many seeds)")
        n_floor_seeds = max(20, args.n_reps * 4)
        floor_vals = np.zeros(n_floor_seeds)
        tr_f, _ = build_segment(detector, x_center_pixel, y_fixed)
        for r in range(n_floor_seeds):
            c, _ = run_one(tracks=tr_f, forced_diff=0.05,
                           seed=5000 + r, **common)
            floor_vals[r] = c
        rel_floor = floor_vals.std(ddof=1) / floor_vals.mean()
        print("  mean=%.4e  std=%.4e  rel-std=%.3f%%"
              % (floor_vals.mean(), floor_vals.std(ddof=1),
                 rel_floor * 100))
        results.update(floor_vals=floor_vals, floor_rel_std=rel_floor)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(floor_vals, bins=15, color="C0", alpha=0.8)
        ax.axvline(floor_vals.mean(), color="k", ls="--",
                   label=f"mean={floor_vals.mean():.3e}")
        ax.set_xlabel("collected sum(signals)  [arb]")
        ax.set_ylabel("count")
        ax.set_title(f"RNG noise floor: same geometry, "
                     f"{n_floor_seeds} seeds (rel-std {rel_floor*100:.2f}%)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_noise_floor.png")

        # -------- Drift on vs off --------
        # Compare full drift() (real diffusion, lifetime) vs hand-populated
        # drift fields (zero diffusion, no attenuation). Difference == net
        # effect of the drift kernel's smearing + attenuation.
        print("Extra: drift kernel ON vs OFF (centered)")
        drift_cms = [1.0, 5.0, 10.0, 20.0]
        on_means = []
        off_means = []
        for dc in drift_cms:
            tr, _ = build_segment(detector, x_center_pixel, y_fixed,
                                  drift_cm=dc)
            m_on, _, _ = run_ensemble(tracks=tr, n_reps=args.n_reps,
                                      skip_drift=False, **common)
            m_off, _, _ = run_ensemble(tracks=tr, n_reps=args.n_reps,
                                       skip_drift=True, **common)
            on_means.append(m_on)
            off_means.append(m_off)
            print("  drift=%.1f cm: ON=%.3e  OFF=%.3e  ratio(off/on)=%.3f"
                  % (dc, m_on, m_off,
                     m_off / m_on if m_on else float("nan")))
        on_means = np.array(on_means)
        off_means = np.array(off_means)
        results.update(drift_cms=np.array(drift_cms),
                       drift_on_means=on_means,
                       drift_off_means=off_means)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(drift_cms, on_means, "o-", label="drift ON (real)")
        ax.plot(drift_cms, off_means, "s--", label="drift OFF (no diff)")
        ax.set_xlabel("drift distance  [cm]")
        ax.set_ylabel("collected sum(signals)  [arb]")
        ax.set_title("Drift kernel on/off vs drift distance")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_drift_onoff.png")

        # -------- tran_diff growth vs drift distance --------
        # drift() sets tran_diff = sqrt(2 * TRAN_DIFF * drift_time).
        # Check empirically that the field comes back populated as
        # expected. This is a direct correctness check of drifting.drift,
        # independent of detsim.
        print("Extra: tran_diff growth with drift distance (drift kernel)")
        td_drift_cms = np.linspace(0.5, 25.0, 12)
        td_vals = np.zeros_like(td_drift_cms)
        ld_vals = np.zeros_like(td_drift_cms)
        for k, dc in enumerate(td_drift_cms):
            tr, _ = build_segment(detector, x_center_pixel, y_fixed,
                                  drift_cm=dc)
            # quench+drift only; read fields back
            from numba import cuda as _cuda
            d_tr = _cuda.to_device(np.copy(tr))
            quenching.quench[1, 128](d_tr, physics.BOX)
            drifting.drift[1, 128](d_tr)
            t_back = d_tr.copy_to_host()
            td_vals[k] = float(t_back["tran_diff"][0])
            ld_vals[k] = float(t_back["long_diff"][0])
        # expected: tran_diff = sqrt(2*TRAN_DIFF * drift_cm / V_DRIFT)
        td_pred = np.sqrt(2 * detector.TRAN_DIFF
                          * td_drift_cms / detector.V_DRIFT)
        ld_pred = np.sqrt(2 * detector.LONG_DIFF
                          * td_drift_cms / detector.V_DRIFT)
        print("  drift_cm    tran_diff(actual)  tran_diff(pred)  ratio")
        for k, dc in enumerate(td_drift_cms):
            print("  %8.2f       %10.5f       %10.5f      %.4f"
                  % (dc, td_vals[k], td_pred[k],
                     td_vals[k] / td_pred[k] if td_pred[k] else float("nan")))
        results.update(td_drift_cms=td_drift_cms,
                       td_vals=td_vals, td_pred=td_pred,
                       ld_vals=ld_vals, ld_pred=ld_pred)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(td_drift_cms, td_vals, "o", label="tran_diff (kernel)")
        ax.plot(td_drift_cms, td_pred, "-",
                label="tran_diff (predicted sqrt scaling)")
        ax.plot(td_drift_cms, ld_vals, "s", label="long_diff (kernel)")
        ax.plot(td_drift_cms, ld_pred, "--",
                label="long_diff (predicted sqrt scaling)")
        ax.set_xlabel("drift distance  [cm]")
        ax.set_ylabel("diffusion width  [cm]")
        ax.set_title("Diffusion vs drift distance (drift kernel sanity)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_diffusion_vs_drift.png")

        # -------- Linearity in deposited charge --------
        # tracks_current_mc is linear in n_electrons; collected should
        # scale as dEdx within RNG floor (since same recombination, same
        # geometry).
        print("Extra: linearity (collected vs dEdx)")
        dedx_grid = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        lin_dep = np.zeros_like(dedx_grid)
        lin_coll = np.zeros_like(dedx_grid)
        for k, de in enumerate(dedx_grid):
            tr, _ = build_segment(detector, x_center_pixel, y_fixed,
                                  dEdx=de)
            m, _, _ = run_ensemble(tracks=tr, n_reps=args.n_reps,
                                   forced_diff=0.05, **common)
            _, dep = run_one(tracks=tr, forced_diff=0.05, seed=42, **common)
            lin_dep[k] = dep
            lin_coll[k] = m
            print("  dEdx=%.2f  dep=%.3e  coll=%.3e  coll/dep=%.4f"
                  % (de, dep, m, m / dep if dep else float("nan")))
        results.update(dedx_grid=dedx_grid,
                       lin_dep=lin_dep, lin_coll=lin_coll)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lin_dep, lin_coll, "o-")
        # fit line through origin via dep[1] (avoid dedx=0)
        if lin_dep[1] > 0:
            slope = lin_coll[1] / lin_dep[1]
            ax.plot(lin_dep, slope * lin_dep, "k--",
                    label=f"linear (slope={slope:.3e})")
        ax.set_xlabel("deposited n_electrons (post-quench)")
        ax.set_ylabel("collected sum(signals)")
        ax.set_title("Linearity check: collected vs deposited")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_linearity.png")

        # -------- Pitch periodicity --------
        # Offsets of 0, 1, 2 pitches sample equivalent positions
        # relative to the pixel grid; collected should match within RNG
        # floor.
        print("Extra: pitch periodicity (offsets 0, 1, 2 pitches)")
        per_offsets_pitch = [0.0, 1.0, 2.0]
        per_means = []
        per_ses = []
        for off in per_offsets_pitch:
            tr, _ = build_segment(detector,
                                  x_center_pixel + off * pitch, y_fixed)
            m, se, _ = run_ensemble(tracks=tr, n_reps=args.n_reps,
                                    forced_diff=0.05, **common)
            per_means.append(m)
            per_ses.append(se)
            print("  +%.0f pitch: %.4e +/- %.2e" % (off, m, se))
        results.update(periodicity_offsets=np.array(per_offsets_pitch),
                       periodicity_means=np.array(per_means),
                       periodicity_stderr=np.array(per_ses))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(per_offsets_pitch, per_means, yerr=per_ses,
                    marker="o", capsize=3)
        if per_means[0]:
            ax.axhline(per_means[0], color="k", ls="--",
                       label="offset=0 reference")
        ax.set_xlabel("offset  [pixel pitch]")
        ax.set_ylabel("collected sum(signals)")
        ax.set_title("Pitch periodicity (equivalent grid positions)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_pitch_periodicity.png")

        # -------- Symmetry around pixel center --------
        # +/- offset around pixel center should give equal collected
        # within floor (pure spatial reflection of geometry).
        print("Extra: +/- offset symmetry around pixel center")
        sym_offs = [-0.25, -0.10, 0.0, 0.10, 0.25]
        sym_means = []
        sym_ses = []
        for off in sym_offs:
            tr, _ = build_segment(detector,
                                  x_center_pixel + off * pitch, y_fixed)
            m, se, _ = run_ensemble(tracks=tr, n_reps=args.n_reps,
                                    forced_diff=0.05, **common)
            sym_means.append(m)
            sym_ses.append(se)
            print("  off=%+.2f pitch: %.4e +/- %.2e" % (off, m, se))
        results.update(sym_offs=np.array(sym_offs),
                       sym_means=np.array(sym_means),
                       sym_stderr=np.array(sym_ses))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(sym_offs, sym_means, yerr=sym_ses,
                    marker="o", capsize=3)
        ax.set_xlabel("offset  [pixel pitch]")
        ax.set_ylabel("collected sum(signals)")
        ax.set_title("Symmetry: +/- offset around pixel center")
        ax.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_symmetry.png")

        # -------- Spatial spread (pixel multiplicity / RMS) --------
        # Number of pixels receiving charge, and RMS spread, should
        # grow with diffusion.
        print("Extra: spatial spread vs diffusion (centered)")
        sp_diffs = [0.005, 0.02, 0.05, 0.10, 0.20, 0.40]
        n_active = []
        rms_pix = []
        for d in sp_diffs:
            tr, _ = build_segment(detector, x_center_pixel, y_fixed)
            _, _, _, pix_sum = run_one(tracks=tr, forced_diff=d,
                                       seed=12345, return_signals=True,
                                       **common)
            if pix_sum is None:
                n_active.append(np.nan)
                rms_pix.append(np.nan)
                continue
            mass = np.abs(pix_sum)
            n_act = int((mass > 0).sum())
            if mass.sum() > 0:
                idx = np.arange(len(mass))
                mean_idx = (idx * mass).sum() / mass.sum()
                rms = np.sqrt(((idx - mean_idx)**2 * mass).sum() / mass.sum())
            else:
                rms = np.nan
            n_active.append(n_act)
            rms_pix.append(rms)
            print("  d=%.3f cm  n_active_pix=%d  rms(pix idx)=%.3f"
                  % (d, n_act, rms))
        results.update(spread_diffs=np.array(sp_diffs),
                       spread_n_active=np.array(n_active),
                       spread_rms=np.array(rms_pix))

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(sp_diffs, n_active, "o-", color="C0",
                 label="active pixel count")
        ax2.plot(sp_diffs, rms_pix, "s--", color="C3",
                 label="RMS of pixel-charge distrib")
        ax1.set_xlabel("diffusion width  [cm]")
        ax1.set_ylabel("active pixels", color="C0")
        ax2.set_ylabel("RMS (pixel index units)", color="C3")
        ax1.set_title("Spatial spread of collected charge vs diffusion")
        ax1.grid(alpha=0.3)
        _save(fig, f"{args.outdir}/issue1_spatial_spread.png")

        # -------- Time profile of collected signal --------
        # sum over pixels vs tick. Peak should sit near drift_time/
        # TIME_SAMPLING.
        print("Extra: time profile of signal at center (d=0.05)")
        tr, _ = build_segment(detector, x_center_pixel, y_fixed)
        _, _, sig, _ = run_one(tracks=tr, forced_diff=0.05, seed=999,
                               return_signals=True, **common)
        if sig is not None:
            tprof = sig[0].sum(axis=0)  # sum over pixels -> [ticks]
            t_ticks = np.arange(len(tprof)) * detector.TIME_SAMPLING
            results.update(time_profile=tprof, time_axis=t_ticks)
            drift_time_pred = 10.0 / detector.V_DRIFT
            peak_tick = int(np.argmax(np.abs(tprof)))
            print("  drift_time predicted = %.3f us, peak at tick %d "
                  "(t=%.3f us)" % (drift_time_pred, peak_tick,
                                   t_ticks[peak_tick]))
            # negative-lobe accounting
            pos = sig[sig > 0].sum()
            neg = sig[sig < 0].sum()
            print("  sum(pos)=%.3e  sum(neg)=%.3e  "
                  "frac_neg=%.3f" % (pos, neg,
                                     abs(neg) / (pos + abs(neg))
                                     if (pos or neg) else float("nan")))
            results.update(neg_lobe_pos=pos, neg_lobe_neg=neg)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(t_ticks, tprof)
            ax.axvline(drift_time_pred, color="r", ls=":",
                       label=f"drift_time_pred={drift_time_pred:.3f} us")
            ax.set_xlabel("time  [us]")
            ax.set_ylabel("sum over pixels of signal")
            ax.set_title("Time profile of induced signal (centered, d=0.05)")
            ax.legend()
            ax.grid(alpha=0.3)
            _save(fig, f"{args.outdir}/issue1_time_profile.png")

    # ====================================================================
    # SAVE RAW + VERDICT
    # ====================================================================
    out_npz = f"{args.outdir}/issue1_results.npz"
    # filter out None entries and non-ndarray-friendly objects
    np.savez(out_npz, **{k: np.asarray(v) for k, v in results.items()
                         if v is not None})
    print("wrote", out_npz)

    print("\n================ SUMMARY ================")
    print("Sanity (centered, d=0.05): collected = %.3e +/- %.2e (n=%d)"
          % (s_mean, s_se, args.n_reps))
    print("Ratio constant (response-scaled): %.3f" % ratio_const)
    if not sanity_ok:
        print("HARNESS UNSOUND: centered sanity case was non-positive.")
        print("Do not interpret position scans as evidence about issue 1.")
        print("=========================================")
        return

    if not args.skip_extras:
        print("RNG noise floor rel-std: %.2f%%" % (rel_floor * 100))

    # Issue-1 verdict from finest-diffusion row of Scan A.
    row = normA[0]
    row_err = normA_err[0]
    finite = np.isfinite(row)
    if finite.any():
        dev = np.nanmax(np.abs(row - 1.0))
        near_mask = (offsets > 0.4) & (offsets < 0.6) & finite
        near = np.nanmax(np.abs(row[near_mask] - 1.0)) if near_mask.any() \
            else 0.0
        # use ensemble error to gate the verdict
        max_err = np.nanmax(row_err[finite])
        print("Issue-1 position dependence (diffusion = %.3f cm row):"
              % diffs_A[0])
        print("  max |norm - 1| over full pitch : %.3f" % dev)
        print("  max |norm - 1| near boundary   : %.3f" % near)
        print("  typical ensemble error band    : %.3f" % max_err)
        if dev < 3 * max_err:
            print("  --> Variation comparable to RNG floor. Cannot")
            print("      claim position dependence.")
        elif near > 0.15 and near >= 0.5 * dev:
            print("  --> Significant position dependence concentrated")
            print("      near pixel boundary. Consistent with issue 1.")
        elif dev > 0.15:
            print("  --> Position dependence present but not boundary-")
            print("      localized; cause unclear, needs investigation.")
        else:
            print("  --> Collected charge position-independent to <15%%.")
            print("      Issue 1 not reproduced under these settings.")
    print("=========================================")


if __name__ == "__main__":
    main()
