#!/usr/bin/env python
"""
test_tricell_calibration.py
===========================

Tricell-style dQ/dx calibration study on simulated cosmic muons in
larnd-sim. For each valid Z-shape tricell extracted from a muon's
pixel readout, we compute THREE independent dQ/dx estimates on the
middle pixel and compare:

  A — Truth: energy deposited in argon × recomb / W_ION, sliced to the
              pixel pillar by geometric clip of the input segment.
  B — Drift-on-pad: number of electrons that PHYSICALLY arrive on the
                    pixel pad after drift + diffusion, computed per
                    substep with the same Gaussian σ the kernel uses,
                    including all RNG fluctuations. Captures Far Field
                    Effects (inward + outward crosstalk).
  C — Pixel readout: kernel response output (sum of signals on the
                     pixel over all time ticks). Includes induced
                     current, cross-induction, etc.

The "tricell" is a Z-shape pattern of 5 pixels:
  - Three contiguous pixels in column w_1 (orthogonal axis v),
  - One "entry-witness" pixel in column w_0 (last w_0 hit before
    track enters w_1),
  - One "exit-witness" pixel in column w_2 (first w_2 hit after
    track exits w_1).
The middle of the three w_1 pixels is the calibration target. ds_C is
reconstructed from the witness pixel positions + timing:

    Δt = t_5 − t_1
    Δx_drift = V_DRIFT * Δt
    Δv = v_5 − v_1                       (pixel coord along v axis)
    Δw = w_5 − w_1 = ±2·pitch            (column index span)
    L_path = sqrt(Δx_drift² + Δv² + Δw²)
    ds_C_recon = L_path * (pitch_v / Δv)

(Equivalently, pitch / cos_v where cos_v is the v-component of the unit
direction vector.) The same `ds_C_3D` (analytic geometric clip from the
input segment) is used as the denominator for all three (A, B, C) so
they're directly comparable.

OUTPUTS
-------
  tricell_dQdx_vs_truth.png       - histograms of B/A, C/A
  tricell_ratio_vs_angle.png      - ratios vs θ_zenith
  tricell_ratio_vs_length.png     - ratios vs L_total
  tricell_ratio_vs_drift.png      - ratios vs drift_cm
  tricell_ds_recon_vs_truth.png   - scatter ds_recon vs ds_3D
  tricell_far_field.png           - B/A vs σ_T/pitch
  tricell_C_vs_B.png              - C/B per tricell
  tricell_w_axis_breakdown.png    - ratios split by w-axis
  tricell_fluctuations.png        - per-seed spread of B, C
  tricell_C_minus_B.png           - (C - B)/A scatter
  tricell_calibration_bias.png    - dQdx_recon_ds / A vs dQdx_truth_ds / A
  tricell_results.npz             - raw records

REQUIREMENTS
------------
  CUDA GPU, cupy, numba, numpy, matplotlib. Run from repo root.

USAGE
-----
    python tests/test_tricell_calibration.py
    python tests/test_tricell_calibration.py --n-tracks 200 --n-seeds 5
"""

import argparse
import sys
from math import ceil, cos, sin, sqrt, pi, acos

import numpy as np


# ---------------------------------------------------------------------------
# Track/segment record dtype, same schema used in prior harness scripts.
# Authoritative copy in cli/dumpTree.py:17.
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


# ---------------------------------------------------------------------------
# Muon track generation
# ---------------------------------------------------------------------------
def sample_cosmic_direction(rng, theta_min_deg=10, theta_max_deg=80):
    """Sample (θ_zenith, φ_azimuth) cosmic-like.
    Zenith: cos²θ distribution restricted to [θ_min, θ_max].
    Azimuth: uniform.
    Returns (θ_radians, φ_radians).
    """
    # Inverse-CDF for cos²θ between cos(θ_max) and cos(θ_min):
    # P(cos θ ≤ c) ∝ c³, so c = U^(1/3) within bounds.
    c_min = cos(np.radians(theta_max_deg))
    c_max = cos(np.radians(theta_min_deg))
    u = rng.uniform(c_min ** 3, c_max ** 3)
    cos_theta = u ** (1.0 / 3.0)
    theta = acos(cos_theta)
    phi = rng.uniform(0, 2 * pi)
    return theta, phi


def build_muon(detector, entry_xyz, direction_xyz, length, dEdx=2.0):
    """Single-segment muon track.

    Args:
        entry_xyz   : (x, y, z) of segment start in detector coords
        direction_xyz: unit 3-vector of track direction
        length      : total path length (cm)
        dEdx        : MeV/cm
    """
    t = make_blank_tracks(1)
    sx, sy, sz = entry_xyz
    dx, dy, dz = direction_xyz
    ex, ey, ez = sx + length * dx, sy + length * dy, sz + length * dz

    t["x_start"] = sx
    t["x_end"] = ex
    t["y_start"] = sy
    t["y_end"] = ey
    t["z_start"] = sz
    t["z_end"] = ez
    t["x"] = 0.5 * (sx + ex)
    t["y"] = 0.5 * (sy + ey)
    t["z"] = 0.5 * (sz + ez)
    t["dx"] = length
    t["dEdx"] = dEdx
    t["dE"] = dEdx * length
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


# ---------------------------------------------------------------------------
# Kernel chain
# ---------------------------------------------------------------------------
def run_kernel_chain(detector, physics, detsim, drifting, quenching,
                     pixels_from_track, sim, create_xoroshiro128p_states,
                     tracks, response, *, seed=42,
                     MAX_PIXELS=500, MAX_ACTIVE_PIXELS=80):
    """quench → drift → get_pixels → tracks_current_mc.

    Returns dict with:
        signals             : (n_tracks, MAX_PIXELS, n_ticks) array
        neighboring_pixels  : (n_tracks, MAX_PIXELS) pixel IDs
        active_pixels       : (n_tracks, MAX_ACTIVE_PIXELS) Bresenham pids
        n_electrons_post_drift : float
        tracks_after        : the tracks recarray after quench+drift
                              (has tran_diff, long_diff, n_electrons set)
    """
    tracks = np.copy(tracks)
    tpb = 128
    bpg = ceil(tracks.shape[0] / tpb)

    from numba import cuda as _cuda
    d_tracks = _cuda.to_device(tracks)
    quenching.quench[bpg, tpb](d_tracks, physics.BOX)
    drifting.drift[bpg, tpb](d_tracks)
    tracks_after = d_tracks.copy_to_host()

    if float(np.sum(tracks_after["n_electrons"])) <= 0:
        return None

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
    active_pixels = d_active.copy_to_host()
    n_pixels_host = d_npix.copy_to_host()
    if int(n_pixels_host[0]) >= MAX_PIXELS:
        print(f"  WARNING: n_pixels {int(n_pixels_host[0])} hit "
              f"MAX_PIXELS={MAX_PIXELS}; consider raising.")

    # Time-axis sizing matching cli/simulate_pixels.py
    long_diff_max = float(np.max(tracks_after["long_diff"]))
    t_span = float(np.max(tracks_after["t_end"] - tracks_after["t0"]))
    diff_pad = long_diff_max / detector.V_DRIFT * detector.DIFF_N_SIGMAS
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
    rng_states = create_xoroshiro128p_states(max(n_states, 1024),
                                             seed=seed)
    d_signals = _cuda.to_device(signals)
    detsim.tracks_current_mc[bpg3, tpb3](d_signals, d_neigh, d_tracks,
                                         response, rng_states)
    signals = d_signals.copy_to_host()

    return dict(
        signals=signals,
        neighboring_pixels=neighboring_pixels,
        active_pixels=active_pixels,
        n_electrons_post_drift=float(tracks_after["n_electrons"][0]),
        tracks_after=tracks_after,
    )


# ---------------------------------------------------------------------------
# Pixel coords
# ---------------------------------------------------------------------------
def pixel_centre(detector, pixel_id, id2pixel):
    """Return (x_centre, y_centre) of pixel pad (cm)."""
    i_x, i_y, plane = id2pixel(int(pixel_id))
    border_x = detector.TPC_BORDERS[int(plane)][0][0]
    border_y = detector.TPC_BORDERS[int(plane)][1][0]
    x_c = border_x + (int(i_x) + 0.5) * detector.PIXEL_PITCH
    y_c = border_y + (int(i_y) + 0.5) * detector.PIXEL_PITCH
    return x_c, y_c, int(i_x), int(i_y), int(plane)


# ---------------------------------------------------------------------------
# Geometric clip of segment through pixel pillar
# ---------------------------------------------------------------------------
def ds_segment_through_pillar(start, end, x_lo, x_hi, y_lo, y_hi):
    """Compute the 3D length of the segment (start → end) intersected
    with the infinite z-pillar over the rectangle [x_lo, x_hi] × [y_lo,
    y_hi]. Uses 2D Liang-Barsky clipping in xy and scales by full 3D
    segment length.
    """
    sx, sy, sz = start
    ex, ey, ez = end
    dxv = ex - sx
    dyv = ey - sy
    dzv = ez - sz
    full_len = sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
    if full_len == 0:
        return 0.0

    t_in, t_out = 0.0, 1.0
    # x slab
    if dxv == 0:
        if sx < x_lo or sx > x_hi:
            return 0.0
    else:
        tx0 = (x_lo - sx) / dxv
        tx1 = (x_hi - sx) / dxv
        if tx0 > tx1:
            tx0, tx1 = tx1, tx0
        t_in = max(t_in, tx0)
        t_out = min(t_out, tx1)
    # y slab
    if dyv == 0:
        if sy < y_lo or sy > y_hi:
            return 0.0
    else:
        ty0 = (y_lo - sy) / dyv
        ty1 = (y_hi - sy) / dyv
        if ty0 > ty1:
            ty0, ty1 = ty1, ty0
        t_in = max(t_in, ty0)
        t_out = min(t_out, ty1)

    if t_in >= t_out:
        return 0.0
    return full_len * (t_out - t_in)


# ---------------------------------------------------------------------------
# B: physical charge arriving on pad via per-substep RNG-kick MC
# ---------------------------------------------------------------------------
def drift_charge_per_pixel_with_rng(detector, segment_recarray,
                                     n_electrons_post_drift, rng):
    """Per-substep RNG-kick MC. For each substep along the track:
      pos = base + Gaussian(0, σ_T, σ_T, σ_L)
      determine which pixel (i_x, i_y) the kicked position lands in
      accumulate n_electrons_post_drift / nstep into that pixel.

    Returns dict {(i_x, i_y): n_electrons_on_pad}.
    """
    t = segment_recarray[0]
    sx, sy, sz = float(t["x_start"]), float(t["y_start"]), float(t["z_start"])
    ex, ey, ez = float(t["x_end"]), float(t["y_end"]), float(t["z_end"])
    seg_len = sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)

    nstep = max(int(round(seg_len / detector.MIN_STEP_SIZE)), 1)
    sigma_T = float(t["tran_diff"])
    sigma_L = float(t["long_diff"])

    # Base substep positions along the track
    s = (np.arange(nstep, dtype=np.float64) + 0.5) / nstep
    bx_arr = sx + s * (ex - sx)
    by_arr = sy + s * (ey - sy)
    # bz_arr = sz + s * (ez - sz)   # not needed for pad membership

    # Gaussian kicks (we only need x, y for pad determination; z kick
    # would affect time but we score by spatial pad membership only)
    kx = rng.normal(0, sigma_T, size=nstep)
    ky = rng.normal(0, sigma_T, size=nstep)
    # kz = rng.normal(0, sigma_L, size=nstep)  # unused for B

    pos_x = bx_arr + kx
    pos_y = by_arr + ky

    pitch = detector.PIXEL_PITCH
    border_x = detector.TPC_BORDERS[int(t["pixel_plane"])][0][0]
    border_y = detector.TPC_BORDERS[int(t["pixel_plane"])][1][0]
    i_x_arr = np.floor((pos_x - border_x) / pitch).astype(np.int64)
    i_y_arr = np.floor((pos_y - border_y) / pitch).astype(np.int64)

    n_e_per_substep = n_electrons_post_drift / nstep
    pixel_keys, counts = np.unique(
        np.stack([i_x_arr, i_y_arr], axis=-1), axis=0, return_counts=True)
    return {(int(k[0]), int(k[1])): float(c * n_e_per_substep)
            for k, c in zip(pixel_keys, counts)}


# ---------------------------------------------------------------------------
# Tricell finder
# ---------------------------------------------------------------------------
def build_hit_dict(neighboring_pixels, signals, id2pixel, q_min):
    """Return {(i_x, i_y): (ipix, Q, t_peak)} for hit pixels.

    ipix = index into neighboring_pixels for the kernel signals lookup.
    Q = sum(signals[0, ipix, :]) (raw, may be negative for fringe)
    t_peak = argmax(signals[0, ipix, :]) * TIME_SAMPLING
    Filtered by |Q| > q_min.
    """
    hits = {}
    n_neigh = neighboring_pixels.shape[1]
    for ipix in range(n_neigh):
        pid = int(neighboring_pixels[0, ipix])
        if pid < 0:
            continue
        i_x, i_y, plane = id2pixel(pid)
        sig = signals[0, ipix, :]
        Q = float(sig.sum())
        if abs(Q) < q_min:
            continue
        t_peak_tick = int(np.argmax(sig))
        hits[(int(i_x), int(i_y))] = (ipix, Q, t_peak_tick, pid)
    return hits


def find_zshape_tricells(hits, time_sampling, q_min_witness):
    """Find Z-shape tricells from a hit dict.

    Args:
        hits: {(i_x, i_y): (ipix, Q, t_peak_tick, pid)}
        time_sampling: us per tick

    Yields dicts, one per valid tricell:
        {
          'w_axis': 'x' or 'y',  # axis of w_1 column
          'v_axis': 'y' or 'x',  # axis along which 3 w_1 pixels stack
          'P_C': (i_x_C, i_y_C),
          'P_w1_lo': (...), 'P_w1_hi': (...),
          'P_w0': (...), 'P_w2': (...),
          't_w0', 't_w1_lo', 't_w1_mid', 't_w1_hi', 't_w2',  # in us
          'Q_C_raw': Q on the middle pixel,
        }
    """
    keys = list(hits.keys())
    for (ix_C, iy_C) in keys:
        ipix_C, Q_C, t_C_tick, pid_C = hits[(ix_C, iy_C)]
        # Try v=y, w_1=x_column: require (ix_C, iy_C±1) hit
        for v_axis, w_axis, lo_key, hi_key in [
            ('y', 'x', (ix_C, iy_C - 1), (ix_C, iy_C + 1)),
            ('x', 'y', (ix_C - 1, iy_C), (ix_C + 1, iy_C)),
        ]:
            if lo_key not in hits or hi_key not in hits:
                continue
            ipix_lo, Q_lo, t_lo_tick, pid_lo = hits[lo_key]
            ipix_hi, Q_hi, t_hi_tick, pid_hi = hits[hi_key]

            # Decide which is "earlier" in time → that's "lower side"
            # of the w_1 column traversal
            t_w1_min = min(t_lo_tick, t_C_tick, t_hi_tick)
            t_w1_max = max(t_lo_tick, t_C_tick, t_hi_tick)

            # Look for w_0 entry witness and w_2 exit witness
            if w_axis == 'x':
                w_col_C = ix_C
            else:
                w_col_C = iy_C
            w0_col = w_col_C - 1
            w2_col = w_col_C + 1

            # Scan all hits in columns w_0 and w_2 for the
            # last-before-t_w1_min and first-after-t_w1_max
            best_w0 = None  # (key, t_tick)
            best_w2 = None
            for (ix_h, iy_h), (ip_h, Q_h, t_h_tick, pid_h) in hits.items():
                if abs(Q_h) < q_min_witness:
                    continue
                col_h = ix_h if w_axis == 'x' else iy_h
                if col_h == w0_col:
                    if t_h_tick < t_w1_min:
                        if best_w0 is None or t_h_tick > best_w0[1]:
                            best_w0 = ((ix_h, iy_h), t_h_tick)
                elif col_h == w2_col:
                    if t_h_tick > t_w1_max:
                        if best_w2 is None or t_h_tick < best_w2[1]:
                            best_w2 = ((ix_h, iy_h), t_h_tick)

            # Also handle reversed direction (track moving in opposite z)
            best_w0_rev = None
            best_w2_rev = None
            for (ix_h, iy_h), (ip_h, Q_h, t_h_tick, pid_h) in hits.items():
                if abs(Q_h) < q_min_witness:
                    continue
                col_h = ix_h if w_axis == 'x' else iy_h
                if col_h == w0_col:
                    if t_h_tick > t_w1_max:
                        if best_w0_rev is None or t_h_tick < best_w0_rev[1]:
                            best_w0_rev = ((ix_h, iy_h), t_h_tick)
                elif col_h == w2_col:
                    if t_h_tick < t_w1_min:
                        if best_w2_rev is None or t_h_tick > best_w2_rev[1]:
                            best_w2_rev = ((ix_h, iy_h), t_h_tick)

            for w0, w2, direction in [
                (best_w0, best_w2, 'forward'),
                (best_w0_rev, best_w2_rev, 'reversed'),
            ]:
                if w0 is None or w2 is None:
                    continue
                yield dict(
                    w_axis=w_axis,
                    v_axis=v_axis,
                    direction=direction,
                    P_C=(ix_C, iy_C),
                    P_w1_lo=lo_key,
                    P_w1_hi=hi_key,
                    P_w0=w0[0],
                    P_w2=w2[0],
                    t_w0_tick=w0[1],
                    t_w1_lo_tick=t_lo_tick,
                    t_w1_mid_tick=t_C_tick,
                    t_w1_hi_tick=t_hi_tick,
                    t_w2_tick=w2[1],
                    ipix_C=ipix_C,
                    pid_C=pid_C,
                    Q_C_raw=Q_C,
                )


# ---------------------------------------------------------------------------
# ds_C reconstructed from witness pixels (user's formula)
# ---------------------------------------------------------------------------
def reconstruct_ds_witnesses(tricell, detector, time_sampling, id2pixel):
    """Compute ds_C_recon from entry and exit witness pixels.

    user's formulation:
        Δt   = t_w2 − t_w0
        Δx   = V_DRIFT * Δt
        Δv   = v_w2 − v_w0   (v is the axis the 3 w_1 pixels stack along)
        Δz   = w_w2 − w_w0   (w is the column axis)
        L    = sqrt(Δx² + Δv² + Δz²)
        ds_C = L * (pitch / Δv)

    Returns ds_C_recon (cm) or None if Δv = 0.
    """
    pitch = detector.PIXEL_PITCH
    v_drift = detector.V_DRIFT
    v_axis = tricell['v_axis']

    border_x = detector.TPC_BORDERS[0][0][0]
    border_y = detector.TPC_BORDERS[0][1][0]

    ix_w0, iy_w0 = tricell['P_w0']
    ix_w2, iy_w2 = tricell['P_w2']
    x_w0 = border_x + (ix_w0 + 0.5) * pitch
    y_w0 = border_y + (iy_w0 + 0.5) * pitch
    x_w2 = border_x + (ix_w2 + 0.5) * pitch
    y_w2 = border_y + (iy_w2 + 0.5) * pitch

    delta_t = (tricell['t_w2_tick'] - tricell['t_w0_tick']) * time_sampling
    delta_drift = v_drift * delta_t   # = drift coordinate span

    if v_axis == 'y':
        delta_v = y_w2 - y_w0
        delta_w = x_w2 - x_w0
    else:  # v_axis == 'x'
        delta_v = x_w2 - x_w0
        delta_w = y_w2 - y_w0

    if delta_v == 0:
        return None, None
    L_path = sqrt(delta_drift ** 2 + delta_v ** 2 + delta_w ** 2)
    ds_recon = L_path * (pitch / abs(delta_v))
    return ds_recon, L_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    ap.add_argument("--n-tracks", type=int, default=100,
                    help="number of independent muon tracks to throw")
    ap.add_argument("--lengths", type=float, nargs="+",
                    default=[2.0, 5.0, 10.0],
                    help="track lengths to scan (cm)")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="kernel seeds per track")
    ap.add_argument("--drift-min", type=float, default=2.0,
                    help="min drift distance midpoint (cm)")
    ap.add_argument("--drift-max", type=float, default=28.0,
                    help="max drift distance midpoint (cm)")
    ap.add_argument("--dedx", type=float, default=2.0)
    ap.add_argument("--q-min", type=float, default=10.0,
                    help="minimum |Q| (response units) for a pixel to "
                         "count as hit")
    ap.add_argument("--q-min-witness", type=float, default=10.0,
                    help="minimum |Q| for witness pixels")
    ap.add_argument("--rng-seed-master", type=int, default=20260519)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--verbose", action="store_true")
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
    id2pixel = pixels_from_track.id2pixel.py_func

    response = detector.load_response(args.response)
    pitch = detector.PIXEL_PITCH
    border_x = detector.TPC_BORDERS[0][0][0]
    border_y = detector.TPC_BORDERS[0][1][0]
    z_anode = detector.TPC_BORDERS[0][2][0]
    z_cathode = detector.TPC_BORDERS[0][2][1]
    into = np.sign(z_cathode - z_anode)
    n_pix_x = int(detector.N_PIXELS[0])
    n_pix_y = int(detector.N_PIXELS[1])

    # Allowed entry region: at least 3 pitches from every wall, AND
    # entry+exit of the longest track to also stay inside. We'll re-check
    # per-track to be safe.
    margin_pitches = 3
    x_lo_allowed = border_x + margin_pitches * pitch
    x_hi_allowed = border_x + (n_pix_x - margin_pitches) * pitch
    y_lo_allowed = border_y + margin_pitches * pitch
    y_hi_allowed = border_y + (n_pix_y - margin_pitches) * pitch

    print(f"Configuration: pitch={pitch:.4f}, V_DRIFT={detector.V_DRIFT:.4e}, "
          f"TIME_SAMPLING={detector.TIME_SAMPLING}, MIN_STEP_SIZE="
          f"{detector.MIN_STEP_SIZE}")
    print(f"N_PIXELS = ({n_pix_x}, {n_pix_y}), TPC drift extent "
          f"({z_anode:.2f}, {z_cathode:.2f})")
    print(f"Allowed (x, y) entry box: x ∈ [{x_lo_allowed:.2f}, "
          f"{x_hi_allowed:.2f}], y ∈ [{y_lo_allowed:.2f}, "
          f"{y_hi_allowed:.2f}]")
    print(f"Scan: {args.n_tracks} tracks × {len(args.lengths)} lengths × "
          f"{args.n_seeds} seeds → "
          f"{args.n_tracks * len(args.lengths) * args.n_seeds} total runs")

    master_rng = np.random.default_rng(args.rng_seed_master)
    common = dict(
        detector=detector, physics=physics, detsim=detsim,
        drifting=drifting, quenching=quenching,
        pixels_from_track=pixels_from_track, sim=sim,
        create_xoroshiro128p_states=create_xoroshiro128p_states,
        response=response,
    )

    records = []
    n_attempts = 0
    n_valid_tricells = 0

    for L in args.lengths:
        for itrk in range(args.n_tracks):
            # Sample direction & entry
            theta, phi = sample_cosmic_direction(master_rng)
            # Detector coords: z = drift axis, x and y on anode plane
            # Direction unit vector: zenith from z-axis
            dirx = sin(theta) * cos(phi)
            diry = sin(theta) * sin(phi)
            dirz = cos(theta) * into   # into the drift volume

            # Choose drift_cm at the MIDPOINT of the track
            drift_cm = master_rng.uniform(
                max(args.drift_min, abs(dirz) * L / 2 + 1),
                args.drift_max - abs(dirz) * L / 2 - 1)
            z_mid = z_anode + into * drift_cm

            # Entry point: choose midpoint, work outward
            x_mid = master_rng.uniform(x_lo_allowed, x_hi_allowed)
            y_mid = master_rng.uniform(y_lo_allowed, y_hi_allowed)
            sx = x_mid - 0.5 * L * dirx
            sy = y_mid - 0.5 * L * diry
            sz = z_mid - 0.5 * L * dirz
            ex = x_mid + 0.5 * L * dirx
            ey = y_mid + 0.5 * L * diry

            # Check entry/exit inside allowed box
            if not (x_lo_allowed <= sx <= x_hi_allowed and
                    x_lo_allowed <= ex <= x_hi_allowed and
                    y_lo_allowed <= sy <= y_hi_allowed and
                    y_lo_allowed <= ey <= y_hi_allowed):
                continue

            entry_xyz = (sx, sy, sz)
            direction = (dirx, diry, dirz)
            tracks = build_muon(detector, entry_xyz, direction, L, args.dedx)
            n_attempts += 1

            for iseed in range(args.n_seeds):
                seed = args.rng_seed_master + itrk * 10000 + iseed * 17
                result = run_kernel_chain(tracks=tracks, seed=seed, **common)
                if result is None:
                    continue
                signals = result['signals']
                neigh = result['neighboring_pixels']
                n_e_post = result['n_electrons_post_drift']
                tracks_after = result['tracks_after']

                # B: pre-compute per-pixel landed charge
                py_rng = np.random.default_rng(seed)
                b_dict = drift_charge_per_pixel_with_rng(
                    detector, tracks_after, n_e_post, py_rng)

                hits = build_hit_dict(neigh, signals, id2pixel,
                                      args.q_min)

                # Find tricells
                tricells = list(find_zshape_tricells(
                    hits, detector.TIME_SAMPLING, args.q_min_witness))

                # Time-monotonicity filter for forward-direction set
                # already enforced in finder. Now compute records.
                for tri in tricells:
                    ds_recon, L_path = reconstruct_ds_witnesses(
                        tri, detector, detector.TIME_SAMPLING, id2pixel)
                    if ds_recon is None:
                        continue

                    # Middle pixel coords
                    ix_C, iy_C = tri['P_C']
                    x_C = border_x + (ix_C + 0.5) * pitch
                    y_C = border_y + (iy_C + 0.5) * pitch

                    # ds_C_3D: geometric clip of input segment
                    # through P_C's pillar
                    ds_3D = ds_segment_through_pillar(
                        (sx, sy, sz),
                        (ex, ey, sz + L * dirz),
                        x_C - pitch / 2, x_C + pitch / 2,
                        y_C - pitch / 2, y_C + pitch / 2)
                    if ds_3D <= 0:
                        continue

                    # A: truth charge in middle pixel
                    Q_A = n_e_post * (ds_3D / L)

                    # B: landed charge on middle pixel
                    Q_B = b_dict.get((ix_C, iy_C), 0.0)

                    # C: raw response charge on middle pixel
                    Q_C_raw = tri['Q_C_raw']

                    dQdx_A = Q_A / ds_3D
                    dQdx_B = Q_B / ds_3D
                    dQdx_C_raw_truth_ds = Q_C_raw / ds_3D
                    dQdx_C_raw_recon_ds = Q_C_raw / ds_recon

                    records.append(dict(
                        track_id=itrk,
                        seed=iseed,
                        L_total=L,
                        theta=theta,
                        phi=phi,
                        drift_cm=drift_cm,
                        w_axis=tri['w_axis'],
                        v_axis=tri['v_axis'],
                        direction=tri['direction'],
                        i_x_C=ix_C,
                        i_y_C=iy_C,
                        ds_C_3D=ds_3D,
                        ds_C_recon=ds_recon,
                        L_path_witnesses=L_path,
                        Q_A=Q_A,
                        Q_B=Q_B,
                        Q_C_raw=Q_C_raw,
                        dQdx_A=dQdx_A,
                        dQdx_B=dQdx_B,
                        dQdx_C_raw_truth_ds=dQdx_C_raw_truth_ds,
                        dQdx_C_raw_recon_ds=dQdx_C_raw_recon_ds,
                        sigma_T=float(tracks_after["tran_diff"][0]),
                        sigma_L=float(tracks_after["long_diff"][0]),
                        n_e_post=n_e_post,
                    ))
                    n_valid_tricells += 1

            if (itrk + 1) % 25 == 0 or args.verbose:
                print(f"  L={L:.1f}cm track {itrk+1}/{args.n_tracks}: "
                      f"{n_valid_tricells} valid tricells so far")

    if not records:
        print("No valid tricells. Run with larger --n-tracks or check "
              "q_min.")
        return

    print(f"\nTotal: {n_valid_tricells} valid tricells from "
          f"{n_attempts} accepted tracks "
          f"({len(args.lengths) * args.n_tracks} attempted).")

    # ---- Save raw ----
    rec_arr = {k: np.array([r[k] for r in records]) for k in records[0]}
    out_npz = f"{args.outdir}/tricell_results.npz"
    np.savez(out_npz, **rec_arr)
    print(f"wrote {out_npz}")

    # ---- Plots ----
    _make_plots(records, args.outdir, pitch)


def _make_plots(records, outdir, pitch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dQdx_A = np.array([r['dQdx_A'] for r in records])
    dQdx_B = np.array([r['dQdx_B'] for r in records])
    dQdx_C = np.array([r['dQdx_C_raw_truth_ds'] for r in records])
    dQdx_C_recon = np.array([r['dQdx_C_raw_recon_ds'] for r in records])
    theta = np.array([r['theta'] for r in records])
    L_total = np.array([r['L_total'] for r in records])
    drift_cm = np.array([r['drift_cm'] for r in records])
    sigma_T = np.array([r['sigma_T'] for r in records])
    w_axis = np.array([r['w_axis'] for r in records])
    ds_3D = np.array([r['ds_C_3D'] for r in records])
    ds_recon = np.array([r['ds_C_recon'] for r in records])

    safe = lambda num, den: np.where(den != 0, num / den, np.nan)
    rBA = safe(dQdx_B, dQdx_A)
    rCA = safe(dQdx_C, dQdx_A)
    rC_recon_A = safe(dQdx_C_recon, dQdx_A)
    rCB = safe(dQdx_C, dQdx_B)

    def _save(fig, name):
        fig.tight_layout()
        path = f"{outdir}/{name}"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        plt.close(fig)

    # 1. dQdx vs truth histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 3, 60)
    ax.hist(rBA, bins=bins, alpha=0.55, label=f"B/A: μ={np.nanmean(rBA):.3f} σ={np.nanstd(rBA):.3f}")
    ax.hist(rCA, bins=bins, alpha=0.55,
            label=f"C/A truth-ds: μ={np.nanmean(rCA):.3f} σ={np.nanstd(rCA):.3f}")
    ax.hist(rC_recon_A, bins=bins, alpha=0.55,
            label=f"C/A recon-ds: μ={np.nanmean(rC_recon_A):.3f} σ={np.nanstd(rC_recon_A):.3f}")
    ax.axvline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("ratio to truth dQ/dx (A)")
    ax.set_ylabel("tricell count")
    ax.set_title("dQ/dx ratios vs truth — tricell calibration test")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "tricell_dQdx_vs_truth.png")

    # 2. Ratio vs angle
    fig, ax = plt.subplots(figsize=(9, 5))
    theta_deg = np.degrees(theta)
    bins_theta = np.linspace(theta_deg.min(), theta_deg.max(), 8)
    for ratio, label, color in [
        (rBA, "B/A", "C0"),
        (rCA, "C/A truth-ds", "C1"),
        (rC_recon_A, "C/A recon-ds", "C2"),
    ]:
        means, stderrs, centres = _binned_stats(theta_deg, ratio, bins_theta)
        ax.errorbar(centres, means, yerr=stderrs, marker="o",
                    capsize=3, label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("θ_zenith [deg]")
    ax.set_ylabel("ratio to dQ/dx truth")
    ax.set_title("dQ/dx ratio vs track zenith angle")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tricell_ratio_vs_angle.png")

    # 3. Ratio vs length
    fig, ax = plt.subplots(figsize=(9, 5))
    unique_L = sorted(set(L_total))
    for ratio, label, color in [(rBA, "B/A", "C0"),
                                 (rCA, "C/A truth-ds", "C1"),
                                 (rC_recon_A, "C/A recon-ds", "C2")]:
        means = []
        stderrs = []
        for L in unique_L:
            mask = L_total == L
            vals = ratio[mask & np.isfinite(ratio)]
            means.append(np.mean(vals) if len(vals) else np.nan)
            stderrs.append(np.std(vals) / np.sqrt(max(len(vals), 1))
                           if len(vals) else np.nan)
        ax.errorbar(unique_L, means, yerr=stderrs, marker="o", capsize=3,
                    label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("track length [cm]")
    ax.set_ylabel("ratio to dQ/dx truth")
    ax.set_title("dQ/dx ratio vs track length")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tricell_ratio_vs_length.png")

    # 4. Ratio vs drift
    fig, ax = plt.subplots(figsize=(9, 5))
    bins_drift = np.linspace(drift_cm.min(), drift_cm.max(), 8)
    for ratio, label, color in [(rBA, "B/A", "C0"),
                                 (rCA, "C/A truth-ds", "C1"),
                                 (rC_recon_A, "C/A recon-ds", "C2")]:
        means, stderrs, centres = _binned_stats(drift_cm, ratio, bins_drift)
        ax.errorbar(centres, means, yerr=stderrs, marker="o", capsize=3,
                    label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("drift distance at tricell midpoint [cm]")
    ax.set_ylabel("ratio to dQ/dx truth")
    ax.set_title("dQ/dx ratio vs drift distance")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tricell_ratio_vs_drift.png")

    # 5. ds recon vs truth
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ds_3D, ds_recon, s=4, alpha=0.4)
    lim = (0, max(ds_3D.max(), ds_recon.max()) * 1.1)
    ax.plot(lim, lim, "k--", lw=0.6, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ds_C_3D (geometric truth) [cm]")
    ax.set_ylabel("ds_C_recon (from witness timing) [cm]")
    ax.set_title("Path length reconstruction vs truth")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tricell_ds_recon_vs_truth.png")

    # 6. Far Field: B/A vs σ_T/pitch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sigma_T / pitch, rBA, s=4, alpha=0.4)
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("σ_T / pitch  (transverse diffusion in pitch units)")
    ax.set_ylabel("dQ/dx_B / dQ/dx_A")
    ax.set_title("Far Field Effect: charge-on-pad ratio vs diffusion width")
    ax.grid(alpha=0.3)
    _save(fig, "tricell_far_field.png")

    # 7. C/B scatter
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(dQdx_B, dQdx_C, s=4, alpha=0.4)
    mx = max(dQdx_B.max(), dQdx_C.max())
    ax.plot([0, mx], [0, mx], "k--", lw=0.6, label="C = B")
    ax.set_xlabel("dQ/dx_B (drift on pad)")
    ax.set_ylabel("dQ/dx_C (response readout)")
    ax.set_title("Response model vs direct drift-on-pad charge")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tricell_C_vs_B.png")

    # 8. w-axis breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["B/A", "C/A truth-ds", "C/A recon-ds"]
    x_axis_means = []
    y_axis_means = []
    for ratio in (rBA, rCA, rC_recon_A):
        mask_x = w_axis == 'x'
        mask_y = w_axis == 'y'
        x_axis_means.append(np.nanmean(ratio[mask_x]) if mask_x.sum() else np.nan)
        y_axis_means.append(np.nanmean(ratio[mask_y]) if mask_y.sum() else np.nan)
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, x_axis_means, 0.4, label="w_1 = x column")
    ax.bar(xpos + 0.2, y_axis_means, 0.4, label="w_1 = y column")
    ax.axhline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean ratio")
    ax.set_title("Tricell ratios split by w_1 axis")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "tricell_w_axis_breakdown.png")

    # 9. Fluctuations: per-(track,tricell) stderr across seeds
    # group records by (track_id, i_x_C, i_y_C, w_axis)
    groups = {}
    for r in records:
        key = (r['track_id'], r['L_total'], r['i_x_C'], r['i_y_C'],
               r['w_axis'])
        groups.setdefault(key, []).append(r)

    a_means = []
    b_stds = []
    c_stds = []
    for key, rs in groups.items():
        if len(rs) < 2:
            continue
        a_means.append(np.mean([r['dQdx_A'] for r in rs]))
        b_stds.append(np.std([r['dQdx_B'] for r in rs]))
        c_stds.append(np.std([r['dQdx_C_raw_truth_ds'] for r in rs]))

    if a_means:
        fig, ax = plt.subplots(figsize=(9, 5))
        a_means = np.array(a_means)
        b_stds = np.array(b_stds)
        c_stds = np.array(c_stds)
        ax.scatter(a_means, b_stds / a_means, s=6, alpha=0.5,
                   label="B stderr / A", color="C0")
        ax.scatter(a_means, c_stds / a_means, s=6, alpha=0.5,
                   label="C stderr / A", color="C1")
        ax.set_xlabel("dQ/dx_A (truth)")
        ax.set_ylabel("per-tricell relative stderr across seeds")
        ax.set_title("Practical reliability: per-tricell seed-to-seed spread")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, "tricell_fluctuations.png")

    # 10. (C - B)/A scatter vs σ_T/pitch
    fig, ax = plt.subplots(figsize=(9, 5))
    delta = (dQdx_C - dQdx_B) / dQdx_A
    ax.scatter(sigma_T / pitch, delta, s=4, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("σ_T / pitch")
    ax.set_ylabel("(C - B) / A")
    ax.set_title("Induced-current contribution beyond direct drift charge")
    ax.grid(alpha=0.3)
    _save(fig, "tricell_C_minus_B.png")

    # 11. Calibration bias: dQ_dx_C / A for truth-ds vs recon-ds
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 3, 60)
    ax.hist(rCA, bins=bins, alpha=0.55,
            label=f"C/A truth-ds: μ={np.nanmean(rCA):.3f} σ={np.nanstd(rCA):.3f}",
            color="C1")
    ax.hist(rC_recon_A, bins=bins, alpha=0.55,
            label=f"C/A recon-ds: μ={np.nanmean(rC_recon_A):.3f} σ={np.nanstd(rC_recon_A):.3f}",
            color="C2")
    ax.axvline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("ratio to truth dQ/dx (A)")
    ax.set_ylabel("tricell count")
    ax.set_title(
        "Calibration bias: truth-ds vs recon-ds denominators\n"
        "(difference = timing-reconstruction contribution to bias)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "tricell_calibration_bias.png")

    # ---- Console summary ----
    print("\n" + "=" * 60)
    print(f"Records: {len(records)}, "
          f"tracks with ≥1 tricell: "
          f"{len(set((r['track_id'], r['L_total']) for r in records))}")
    print(f"<B/A>           = {np.nanmean(rBA):.3f} ± {np.nanstd(rBA)/np.sqrt(len(records)):.3f}")
    print(f"<C/A truth-ds>  = {np.nanmean(rCA):.3f} ± {np.nanstd(rCA)/np.sqrt(len(records)):.3f}")
    print(f"<C/A recon-ds>  = {np.nanmean(rC_recon_A):.3f} ± {np.nanstd(rC_recon_A)/np.sqrt(len(records)):.3f}")
    print(f"<C/B>           = {np.nanmean(rCB):.3f} ± {np.nanstd(rCB)/np.sqrt(len(records)):.3f}")
    print(f"<ds_recon/ds_3D>= {np.nanmean(ds_recon/ds_3D):.3f}")
    print("=" * 60)


def _binned_stats(x, y, bins):
    idx = np.digitize(x, bins) - 1
    means, stderrs, centres = [], [], []
    for i in range(len(bins) - 1):
        mask = (idx == i) & np.isfinite(y)
        vals = y[mask]
        if len(vals) >= 2:
            means.append(np.mean(vals))
            stderrs.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            stderrs.append(np.nan)
        centres.append(0.5 * (bins[i] + bins[i + 1]))
    return np.array(means), np.array(stderrs), np.array(centres)


if __name__ == "__main__":
    main()
