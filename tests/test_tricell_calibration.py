#!/usr/bin/env python
"""
test_tricell_calibration.py
===========================

Tricell-style dQ/dx calibration study on simulated cosmic muons in
larnd-sim. For each valid Z-shape tricell extracted from a muon's
pixel readout, we compute THREE independent dQ/dx estimates on the
middle pixel and compare:

  TRUTH       — energy deposited in argon × recomb / W_ION, sliced to
                 the pixel pillar by geometric clip of the input
                 segment.
  DRIFT_LANDED— physical electrons that arrive on the pixel pad after
                 drift + diffusion, computed per substep with the same
                 Gaussian σ the kernel uses, including all RNG
                 fluctuations. Captures Far Field Effects: inward +
                 outward diffusion crosstalk.
  KERNEL      — kernel response readout (sum of signals on the pixel
                 over all time ticks). Includes induced current, cross-
                 induction, etc.

The "tricell" is a Z-shape pattern of 5 pixels:
  - Three contiguous pixels in column w_1 (along orthogonal axis v),
  - One "entry-witness" pixel in column w_0 (last w_0 hit before the
    track enters w_1),
  - One "exit-witness" pixel in column w_2 (first w_2 hit after the
    track exits w_1).
The middle of the three w_1 pixels is the calibration target. ds is
reconstructed from witness pixel positions + timing, scaled to the
portion of the path that lies INSIDE the w_1 column:

    delta_t            = t_w2_witness − t_w0_witness
    delta_drift        = V_DRIFT * delta_t
    delta_traversal    = v_center_w2_witness − v_center_w0_witness
                          (uses witness centers as proxies for the
                           v-coord where the track crossed the
                           w_0/w_1 and w_1/w_2 column boundaries)
    delta_column       = 1 * pixel_pitch   (exact, the width of w_1)
    L_inside_w1        = sqrt(delta_drift² + delta_traversal² + delta_column²)
    ds_middle_pixel    = L_inside_w1 * pixel_pitch / |delta_traversal|

Note delta_column is ONE pitch (the column width), not two: only the
portion of the track inside w_1 contributes to ds for the middle
pixel. The witnesses themselves sit outside w_1 (in w_0 and w_2) and
their full center-to-center w-separation is two pitches, but the path
through w_1 covers just one.

The same `ds_middle_pixel_geometric_truth` (analytic 3D clip from the
input segment) is used as the denominator for all three measurements so
they're directly comparable.

OUTPUTS
-------
  tricell_ratios_to_truth_hist.png
  tricell_ratios_vs_zenith_angle.png
  tricell_ratios_vs_track_length.png
  tricell_ratios_vs_drift_distance.png
  tricell_ds_reconstructed_vs_truth.png
  tricell_drift_landed_vs_diffusion_width.png
  tricell_kernel_vs_drift_landed.png
  tricell_ratios_by_column_axis.png
  tricell_seed_fluctuation_reliability.png
  tricell_kernel_minus_drift_landed.png
  tricell_calibration_bias_truth_vs_recon_ds.png
  tricell_results.npz

REQUIREMENTS
------------
  CUDA GPU, cupy, numba, numpy, matplotlib. Run from repo root.
"""

import argparse
import sys
from math import ceil, cos, sin, sqrt, pi, acos

import numpy as np


# ---------------------------------------------------------------------------
# Track/segment record dtype, inlined from cli/dumpTree.py:17.
# Same 30-field schema used in the other tests in this directory.
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


def make_blank_track_recarray(n_segments):
    return np.zeros(n_segments, dtype=SEGMENTS_DTYPE).view(np.recarray)


# ---------------------------------------------------------------------------
# Muon track generation
# ---------------------------------------------------------------------------
def sample_cosmic_direction(rng,
                            zenith_min_deg=10.0,
                            zenith_max_deg=80.0):
    """Sample cosmic-like (zenith θ, azimuth φ) angles.

    Zenith uses a cos²θ angular distribution, restricted to
    [zenith_min, zenith_max]. Azimuth is uniform on [0, 2π).

    Returns (zenith_radians, azimuth_radians).
    """
    # cos²θ distribution: dN/dcos θ ∝ cos² θ, so cumulative CDF is
    # ∝ cos³ θ. Sample uniformly in cos³θ within bounds.
    cos_zenith_min = cos(np.radians(zenith_max_deg))
    cos_zenith_max = cos(np.radians(zenith_min_deg))
    sample_uniform = rng.uniform(cos_zenith_min ** 3, cos_zenith_max ** 3)
    cos_zenith = sample_uniform ** (1.0 / 3.0)
    zenith = acos(cos_zenith)
    azimuth = rng.uniform(0, 2 * pi)
    return zenith, azimuth


def build_muon_segment(start_xyz, direction_xyz, track_length_cm,
                       dEdx_MeV_per_cm=2.0):
    """One-segment muon track from start point + unit direction vector
    + total path length. Returns a 1-entry tracks recarray."""
    tracks = make_blank_track_recarray(1)
    start_x, start_y, start_z = start_xyz
    direction_x, direction_y, direction_z = direction_xyz
    end_x = start_x + track_length_cm * direction_x
    end_y = start_y + track_length_cm * direction_y
    end_z = start_z + track_length_cm * direction_z

    tracks["x_start"] = start_x
    tracks["x_end"] = end_x
    tracks["y_start"] = start_y
    tracks["y_end"] = end_y
    tracks["z_start"] = start_z
    tracks["z_end"] = end_z
    tracks["x"] = 0.5 * (start_x + end_x)
    tracks["y"] = 0.5 * (start_y + end_y)
    tracks["z"] = 0.5 * (start_z + end_z)
    tracks["dx"] = track_length_cm
    tracks["dEdx"] = dEdx_MeV_per_cm
    tracks["dE"] = dEdx_MeV_per_cm * track_length_cm
    tracks["pdg_id"] = 13                      # muon
    tracks["segment_id"] = 0
    tracks["event_id"] = 0
    tracks["traj_id"] = 0
    tracks["pixel_plane"] = 0
    tracks["t0"] = 0
    tracks["t0_start"] = 0
    tracks["t0_end"] = 0
    tracks["tran_diff"] = 1e-2
    tracks["long_diff"] = 1e-2
    return tracks


# ---------------------------------------------------------------------------
# Run kernel chain on a single-segment track
# ---------------------------------------------------------------------------
def run_kernel_chain(detector, physics, sim,
                     detsim, drifting, quenching, pixels_from_track,
                     create_xoroshiro128p_states,
                     tracks, response_table, *,
                     kernel_rng_seed=42,
                     max_pixels_per_track=500,
                     max_active_pixels_per_track=80):
    """quench → drift → get_pixels → tracks_current_mc.

    Returns dict with:
      signals                 : (n_tracks, max_pixels_per_track, n_ticks)
      neighboring_pixels      : (n_tracks, max_pixels_per_track) pixel IDs
      active_pixels           : (n_tracks, max_active_pixels_per_track)
      n_electrons_post_drift  : float (electrons after quench+drift)
      tracks_after_drift      : the tracks recarray after quench+drift
    """
    tracks = np.copy(tracks)
    threads_per_block = 128
    blocks_per_grid = ceil(tracks.shape[0] / threads_per_block)

    from numba import cuda as _cuda
    device_tracks = _cuda.to_device(tracks)
    quenching.quench[blocks_per_grid, threads_per_block](
        device_tracks, physics.BOX)
    drifting.drift[blocks_per_grid, threads_per_block](device_tracks)
    tracks_after_drift = device_tracks.copy_to_host()

    if float(np.sum(tracks_after_drift["n_electrons"])) <= 0:
        return None

    active_pixels = np.full(
        (tracks.shape[0], max_active_pixels_per_track), -1,
        dtype=np.int32)
    neighboring_pixels = np.full(
        (tracks.shape[0], max_pixels_per_track), -1, dtype=np.int32)
    neighboring_radius = np.zeros(
        (tracks.shape[0], max_pixels_per_track), dtype=np.float32)
    n_pixels_used = np.zeros(shape=(tracks.shape[0]), dtype=np.int64)
    device_active = _cuda.to_device(active_pixels)
    device_neighbors = _cuda.to_device(neighboring_pixels)
    device_radius = _cuda.to_device(neighboring_radius)
    device_n_pixels = _cuda.to_device(n_pixels_used)

    pixels_from_track.get_pixels[blocks_per_grid, threads_per_block](
        device_tracks, device_active, device_neighbors,
        device_radius, device_n_pixels)
    neighboring_pixels = device_neighbors.copy_to_host()
    active_pixels = device_active.copy_to_host()
    n_pixels_used_host = device_n_pixels.copy_to_host()
    if int(n_pixels_used_host[0]) >= max_pixels_per_track:
        print(f"  WARNING: halo size {int(n_pixels_used_host[0])} hit "
              f"max_pixels_per_track={max_pixels_per_track}; "
              f"raise this limit if you see this repeatedly.")

    # Time-axis sizing matches cli/simulate_pixels.py
    long_diff_max = float(np.max(tracks_after_drift["long_diff"]))
    t_span = float(np.max(
        tracks_after_drift["t_end"] - tracks_after_drift["t0"]))
    diffusion_pad_us = (long_diff_max / detector.V_DRIFT
                        * detector.DIFF_N_SIGMAS)
    if detector.RESPONSE_MAX_TIME > detector.DRIFT_MAX_TIME:
        max_signal_time_us = (
            t_span + diffusion_pad_us
            + detector.RESPONSE_MAX_TIME - detector.DRIFT_MAX_TIME)
    else:
        max_signal_time_us = t_span + diffusion_pad_us
    signal_n_ticks = max(
        ceil(max_signal_time_us / detector.TIME_SAMPLING), 1)

    signals = np.zeros((tracks.shape[0],
                        neighboring_pixels.shape[1],
                        signal_n_ticks), dtype=np.float32)
    threads_per_block_3d = (1, 1, 64)
    blocks_per_grid_3d = (
        ceil(signals.shape[0] / threads_per_block_3d[0]),
        ceil(signals.shape[1] / threads_per_block_3d[1]),
        ceil(signals.shape[2] / threads_per_block_3d[2]),
    )
    n_rng_states = int(
        np.prod(threads_per_block_3d)
        * blocks_per_grid_3d[0]
        * blocks_per_grid_3d[1]
        * blocks_per_grid_3d[2])
    rng_states = create_xoroshiro128p_states(
        max(n_rng_states, 1024), seed=kernel_rng_seed)
    device_signals = _cuda.to_device(signals)
    detsim.tracks_current_mc[blocks_per_grid_3d, threads_per_block_3d](
        device_signals, device_neighbors, device_tracks,
        response_table, rng_states)
    signals = device_signals.copy_to_host()

    return dict(
        signals=signals,
        neighboring_pixels=neighboring_pixels,
        active_pixels=active_pixels,
        n_electrons_post_drift=float(
            tracks_after_drift["n_electrons"][0]),
        tracks_after_drift=tracks_after_drift,
    )


# ---------------------------------------------------------------------------
# Pixel-coordinate helpers
# ---------------------------------------------------------------------------
def pixel_center_coords(detector, pixel_id, id2pixel_fn):
    """Return (x_center_cm, y_center_cm, i_x, i_y, plane_id) of a pixel
    pad given its integer pixel ID."""
    i_x, i_y, plane = id2pixel_fn(int(pixel_id))
    border_x = detector.TPC_BORDERS[int(plane)][0][0]
    border_y = detector.TPC_BORDERS[int(plane)][1][0]
    pitch = detector.PIXEL_PITCH
    x_center_cm = border_x + (int(i_x) + 0.5) * pitch
    y_center_cm = border_y + (int(i_y) + 0.5) * pitch
    return x_center_cm, y_center_cm, int(i_x), int(i_y), int(plane)


def pixel_indices_to_center(detector, i_x, i_y, plane=0):
    """Inverse mapping (i_x, i_y) → (x_center, y_center) in cm."""
    border_x = detector.TPC_BORDERS[plane][0][0]
    border_y = detector.TPC_BORDERS[plane][1][0]
    pitch = detector.PIXEL_PITCH
    return (border_x + (i_x + 0.5) * pitch,
            border_y + (i_y + 0.5) * pitch)


# ---------------------------------------------------------------------------
# Geometric clip of a 3D segment through a pixel's pillar
# ---------------------------------------------------------------------------
def segment_length_through_pixel_pillar(
        segment_start_xyz, segment_end_xyz,
        pillar_x_lo, pillar_x_hi, pillar_y_lo, pillar_y_hi):
    """3D length of the segment intersected with the infinite z-pillar
    over the rectangle (pillar_x_lo, pillar_x_hi) × (pillar_y_lo,
    pillar_y_hi). 2D Liang-Barsky in xy, scaled by full 3D length."""
    start_x, start_y, start_z = segment_start_xyz
    end_x, end_y, end_z = segment_end_xyz
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    delta_z = end_z - start_z
    full_length = sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    if full_length == 0:
        return 0.0

    t_enter, t_exit = 0.0, 1.0
    # x slab clipping
    if delta_x == 0:
        if not (pillar_x_lo <= start_x <= pillar_x_hi):
            return 0.0
    else:
        t_x_lo = (pillar_x_lo - start_x) / delta_x
        t_x_hi = (pillar_x_hi - start_x) / delta_x
        if t_x_lo > t_x_hi:
            t_x_lo, t_x_hi = t_x_hi, t_x_lo
        t_enter = max(t_enter, t_x_lo)
        t_exit = min(t_exit, t_x_hi)
    # y slab clipping
    if delta_y == 0:
        if not (pillar_y_lo <= start_y <= pillar_y_hi):
            return 0.0
    else:
        t_y_lo = (pillar_y_lo - start_y) / delta_y
        t_y_hi = (pillar_y_hi - start_y) / delta_y
        if t_y_lo > t_y_hi:
            t_y_lo, t_y_hi = t_y_hi, t_y_lo
        t_enter = max(t_enter, t_y_lo)
        t_exit = min(t_exit, t_y_hi)

    if t_enter >= t_exit:
        return 0.0
    return full_length * (t_exit - t_enter)


# ---------------------------------------------------------------------------
# Drift-landed charge per pixel (per-substep RNG-kick MC)
# ---------------------------------------------------------------------------
def landed_charge_per_pixel_with_rng_kicks(
        detector, tracks_after_drift, n_electrons_post_drift, rng):
    """Per-substep RNG-kick MC of "where electrons land on the anode
    after drift + diffusion".

    Each substep is treated as a point charge at base_position +
    Gaussian_kick (same σ_T, σ_L as the kernel). We accumulate
    electrons-per-substep into the pixel whose pad rectangle contains
    the kicked position. Sums over the entire track, including
    substeps whose base position sits over neighbouring pixels but
    whose kick lands them on this one (inward Far Field crosstalk).

    Returns {(pixel_i_x, pixel_i_y): n_electrons_landed}.
    """
    track = tracks_after_drift[0]
    start_x = float(track["x_start"])
    start_y = float(track["y_start"])
    start_z = float(track["z_start"])
    end_x = float(track["x_end"])
    end_y = float(track["y_end"])
    end_z = float(track["z_end"])
    segment_length_cm = sqrt(
        (end_x - start_x) ** 2
        + (end_y - start_y) ** 2
        + (end_z - start_z) ** 2)

    n_substeps = max(
        int(round(segment_length_cm / detector.MIN_STEP_SIZE)), 1)
    sigma_transverse_cm = float(track["tran_diff"])
    sigma_longitudinal_cm = float(track["long_diff"])

    # Substep base positions along the track center-line
    substep_fraction = (np.arange(n_substeps, dtype=np.float64) + 0.5) \
        / n_substeps
    base_x = start_x + substep_fraction * (end_x - start_x)
    base_y = start_y + substep_fraction * (end_y - start_y)
    # base_z not needed: we score by (x, y) pad membership only.

    # Gaussian kicks per substep (independent draws)
    kick_x = rng.normal(0, sigma_transverse_cm, size=n_substeps)
    kick_y = rng.normal(0, sigma_transverse_cm, size=n_substeps)
    landed_x = base_x + kick_x
    landed_y = base_y + kick_y

    pitch = detector.PIXEL_PITCH
    border_x = detector.TPC_BORDERS[int(track["pixel_plane"])][0][0]
    border_y = detector.TPC_BORDERS[int(track["pixel_plane"])][1][0]
    pixel_i_x_per_substep = np.floor(
        (landed_x - border_x) / pitch).astype(np.int64)
    pixel_i_y_per_substep = np.floor(
        (landed_y - border_y) / pitch).astype(np.int64)

    electrons_per_substep = n_electrons_post_drift / n_substeps
    unique_pixel_keys, substep_counts = np.unique(
        np.stack([pixel_i_x_per_substep, pixel_i_y_per_substep],
                 axis=-1),
        axis=0, return_counts=True)
    return {(int(key[0]), int(key[1])):
            float(count * electrons_per_substep)
            for key, count in zip(unique_pixel_keys, substep_counts)}


# ---------------------------------------------------------------------------
# Pixel-hit dict from kernel output
# ---------------------------------------------------------------------------
def build_hit_pixel_dict(neighboring_pixels, signals, id2pixel_fn,
                         minimum_pixel_charge_threshold):
    """Build a dict of pixels with significant kernel response charge.

    Returns {(i_x, i_y): {
        'halo_index': index into neighboring_pixels for signal lookup,
        'collected_charge': sum(signals[0, halo_index, :]),
        'peak_tick': argmax(signals[0, halo_index, :]),
        'pixel_id': raw pixel ID,
    }}.

    Pixels with |collected_charge| < threshold are excluded.
    """
    hit_dict = {}
    n_halo_pixels = neighboring_pixels.shape[1]
    for halo_index in range(n_halo_pixels):
        pixel_id = int(neighboring_pixels[0, halo_index])
        if pixel_id < 0:
            continue
        i_x, i_y, plane_id = id2pixel_fn(pixel_id)
        signal_trace = signals[0, halo_index, :]
        collected_charge = float(signal_trace.sum())
        if abs(collected_charge) < minimum_pixel_charge_threshold:
            continue
        peak_tick = int(np.argmax(signal_trace))
        hit_dict[(int(i_x), int(i_y))] = dict(
            halo_index=halo_index,
            collected_charge=collected_charge,
            peak_tick=peak_tick,
            pixel_id=pixel_id,
        )
    return hit_dict


# ---------------------------------------------------------------------------
# Z-shape tricell finder
# ---------------------------------------------------------------------------
def find_zshape_tricells(hit_pixel_dict, minimum_witness_charge_threshold):
    """Find Z-shape tricells.

    For each candidate center pixel P_C:
      1. Require three contiguous v-pixels in column w_1
         (where w_1 = x or y).
      2. Find the "entry-witness" pixel in column w_0 = w_1 − 1
         whose peak time is *latest before* the w_1 trio's earliest
         peak (forward-direction) OR *earliest after* the latest
         (reversed-direction).
      3. Likewise the "exit-witness" pixel in column w_2 = w_1 + 1.

    Yields one dict per valid tricell.
    """
    pixel_keys = list(hit_pixel_dict.keys())

    for (i_x_center, i_y_center) in pixel_keys:
        center_record = hit_pixel_dict[(i_x_center, i_y_center)]

        for tricell_orientation in [
            # column_axis = the axis along which we cross 3 columns
            # traversal_axis = the axis the 3 w_1 pixels stack along
            dict(column_axis='x', traversal_axis='y',
                 lo_key=(i_x_center, i_y_center - 1),
                 hi_key=(i_x_center, i_y_center + 1)),
            dict(column_axis='y', traversal_axis='x',
                 lo_key=(i_x_center - 1, i_y_center),
                 hi_key=(i_x_center + 1, i_y_center)),
        ]:
            lo_key = tricell_orientation['lo_key']
            hi_key = tricell_orientation['hi_key']
            if lo_key not in hit_pixel_dict:
                continue
            if hi_key not in hit_pixel_dict:
                continue
            lo_record = hit_pixel_dict[lo_key]
            hi_record = hit_pixel_dict[hi_key]

            w1_min_peak_tick = min(lo_record['peak_tick'],
                                   center_record['peak_tick'],
                                   hi_record['peak_tick'])
            w1_max_peak_tick = max(lo_record['peak_tick'],
                                   center_record['peak_tick'],
                                   hi_record['peak_tick'])

            column_axis = tricell_orientation['column_axis']
            traversal_axis = tricell_orientation['traversal_axis']
            if column_axis == 'x':
                w1_column_index = i_x_center
            else:
                w1_column_index = i_y_center
            w0_column_index = w1_column_index - 1
            w2_column_index = w1_column_index + 1

            # Search all hit pixels for entry/exit witnesses in
            # columns w_0 and w_2. Track may run either direction in
            # drift (z), so try both orderings.
            best_forward_w0 = None
            best_forward_w2 = None
            best_reversed_w0 = None
            best_reversed_w2 = None
            for (i_x_hit, i_y_hit), hit in hit_pixel_dict.items():
                if abs(hit['collected_charge']) < \
                        minimum_witness_charge_threshold:
                    continue
                column_of_hit = (
                    i_x_hit if column_axis == 'x' else i_y_hit)
                peak_tick_hit = hit['peak_tick']
                if column_of_hit == w0_column_index:
                    if peak_tick_hit < w1_min_peak_tick:
                        # forward direction: w_0 is before w_1
                        if (best_forward_w0 is None
                                or peak_tick_hit > best_forward_w0['peak_tick']):
                            best_forward_w0 = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)
                    if peak_tick_hit > w1_max_peak_tick:
                        if (best_reversed_w0 is None
                                or peak_tick_hit < best_reversed_w0['peak_tick']):
                            best_reversed_w0 = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)
                elif column_of_hit == w2_column_index:
                    if peak_tick_hit > w1_max_peak_tick:
                        if (best_forward_w2 is None
                                or peak_tick_hit < best_forward_w2['peak_tick']):
                            best_forward_w2 = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)
                    if peak_tick_hit < w1_min_peak_tick:
                        if (best_reversed_w2 is None
                                or peak_tick_hit > best_reversed_w2['peak_tick']):
                            best_reversed_w2 = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)

            for witness_w0, witness_w2, direction_label in [
                (best_forward_w0, best_forward_w2, 'forward'),
                (best_reversed_w0, best_reversed_w2, 'reversed'),
            ]:
                if witness_w0 is None or witness_w2 is None:
                    continue
                yield dict(
                    column_axis=column_axis,
                    traversal_axis=traversal_axis,
                    drift_direction=direction_label,
                    center_pixel_key=(i_x_center, i_y_center),
                    center_halo_index=center_record['halo_index'],
                    center_collected_charge=center_record['collected_charge'],
                    center_pixel_id=center_record['pixel_id'],
                    w1_lo_key=lo_key,
                    w1_hi_key=hi_key,
                    w0_witness_key=witness_w0['key'],
                    w2_witness_key=witness_w2['key'],
                    peak_tick_w0_witness=witness_w0['peak_tick'],
                    peak_tick_w1_lo=lo_record['peak_tick'],
                    peak_tick_w1_center=center_record['peak_tick'],
                    peak_tick_w1_hi=hi_record['peak_tick'],
                    peak_tick_w2_witness=witness_w2['peak_tick'],
                )


# ---------------------------------------------------------------------------
# Reconstruct ds through middle pixel from witness pixel geometry+timing
# ---------------------------------------------------------------------------
def reconstruct_ds_from_witnesses(tricell, detector, time_sampling_us):
    """Compute ds through the middle pixel from witness pixels.

    The witnesses bound the track's path through the w_1 column. We
    use them as proxies for the boundary crossings into and out of w_1:

      delta_t            = t_w2_witness − t_w0_witness
      delta_drift        = V_DRIFT * delta_t
      delta_traversal    = v_center_w2_witness − v_center_w0_witness
                            (approximates the v-distance traveled WHILE
                             INSIDE w_1; uses witness centers as proxies
                             for where the track crossed w_0/w_1 and
                             w_1/w_2 boundaries in v)
      delta_column       = 1 * pixel_pitch
                            (EXACT: the w-distance traveled while inside
                             w_1 is exactly the width of one column. We
                             do NOT use 2·pitch between witness centers
                             because most of that distance is inside
                             w_0 and w_2, not w_1.)
      L_inside_w1        = sqrt(delta_drift² + delta_traversal² + delta_column²)
      ds_middle_pixel    = L_inside_w1 * pixel_pitch / |delta_traversal|

    Returns (ds_middle_pixel_cm, L_inside_w1_cm) or (None, None) if
    delta_traversal == 0.

    Caveat: using witness pixel CENTERS as the v-coord for the
    boundary crossings is an approximation. The true v at the
    w_0/w_1 boundary lies somewhere within ±pitch/2 of the w_0
    witness center; same for w_2. For tracks with shallow v-slope
    this is accurate; for steeply v-moving tracks the approximation
    can drift by a fraction of a pitch.
    """
    pitch_cm = detector.PIXEL_PITCH
    v_drift = detector.V_DRIFT
    traversal_axis = tricell['traversal_axis']

    i_x_w0, i_y_w0 = tricell['w0_witness_key']
    i_x_w2, i_y_w2 = tricell['w2_witness_key']
    x_w0, y_w0 = pixel_indices_to_center(detector, i_x_w0, i_y_w0)
    x_w2, y_w2 = pixel_indices_to_center(detector, i_x_w2, i_y_w2)

    delta_t_us = (tricell['peak_tick_w2_witness']
                  - tricell['peak_tick_w0_witness']) * time_sampling_us
    delta_drift_cm = v_drift * delta_t_us

    if traversal_axis == 'y':
        delta_traversal_cm = y_w2 - y_w0
    else:  # traversal_axis == 'x'
        delta_traversal_cm = x_w2 - x_w0

    # Path inside w_1 column traverses exactly ONE pixel pitch in the
    # column-direction (w), not two. Sign follows w_2 > w_0.
    delta_column_cm = pitch_cm

    if delta_traversal_cm == 0:
        return None, None
    length_inside_w1_cm = sqrt(
        delta_drift_cm ** 2
        + delta_traversal_cm ** 2
        + delta_column_cm ** 2)
    ds_middle_pixel_cm = (length_inside_w1_cm
                          * pitch_cm / abs(delta_traversal_cm))
    return ds_middle_pixel_cm, length_inside_w1_cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument(
        "--response",
        default="larndsim/bin/response_44_v2a_full.npz")
    arg_parser.add_argument(
        "--detector",
        default="larndsim/detector_properties/module0.yaml")
    arg_parser.add_argument(
        "--pixel-layout",
        default="larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
    arg_parser.add_argument(
        "--sim-properties",
        default="larndsim/simulation_properties/singles_sim.yaml")
    arg_parser.add_argument(
        "--n-tracks", type=int, default=100,
        help="number of independent muon tracks per length")
    arg_parser.add_argument(
        "--track-lengths-cm", type=float, nargs="+",
        default=[2.0, 5.0, 10.0],
        help="track lengths to scan (cm)")
    arg_parser.add_argument(
        "--n-seeds-per-track", type=int, default=3,
        help="kernel RNG seeds per track")
    arg_parser.add_argument(
        "--drift-distance-min-cm", type=float, default=2.0)
    arg_parser.add_argument(
        "--drift-distance-max-cm", type=float, default=28.0)
    arg_parser.add_argument(
        "--track-dEdx-MeV-per-cm", type=float, default=2.0)
    arg_parser.add_argument(
        "--minimum-pixel-charge-threshold", type=float, default=10.0,
        help="minimum |collected charge| (response units) for a pixel "
             "to count as hit")
    arg_parser.add_argument(
        "--minimum-witness-charge-threshold", type=float, default=10.0,
        help="minimum |collected charge| for witness pixels")
    arg_parser.add_argument(
        "--master-rng-seed", type=int, default=20260519)
    arg_parser.add_argument(
        "--outdir", default=".")
    arg_parser.add_argument(
        "--verbose", action="store_true")
    args = arg_parser.parse_args()

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

    response_table = detector.load_response(args.response)
    pitch_cm = detector.PIXEL_PITCH
    border_x_cm = detector.TPC_BORDERS[0][0][0]
    border_y_cm = detector.TPC_BORDERS[0][1][0]
    anode_z_cm = detector.TPC_BORDERS[0][2][0]
    cathode_z_cm = detector.TPC_BORDERS[0][2][1]
    into_drift_volume = np.sign(cathode_z_cm - anode_z_cm)
    n_pixels_x = int(detector.N_PIXELS[0])
    n_pixels_y = int(detector.N_PIXELS[1])

    # Restrict entry+exit positions to be at least this many pitches
    # away from any TPC wall so the halo isn't truncated.
    wall_margin_pitches = 3
    x_entry_lo = border_x_cm + wall_margin_pitches * pitch_cm
    x_entry_hi = border_x_cm + (n_pixels_x - wall_margin_pitches) * pitch_cm
    y_entry_lo = border_y_cm + wall_margin_pitches * pitch_cm
    y_entry_hi = border_y_cm + (n_pixels_y - wall_margin_pitches) * pitch_cm

    print(f"Detector config: pitch={pitch_cm:.4f} cm, "
          f"V_DRIFT={detector.V_DRIFT:.4e} cm/us, "
          f"TIME_SAMPLING={detector.TIME_SAMPLING} us, "
          f"MIN_STEP_SIZE={detector.MIN_STEP_SIZE} cm")
    print(f"N_PIXELS = ({n_pixels_x}, {n_pixels_y}); "
          f"anode_z={anode_z_cm:.2f}, cathode_z={cathode_z_cm:.2f} cm")
    print(f"Track entry box: x ∈ [{x_entry_lo:.2f}, {x_entry_hi:.2f}], "
          f"y ∈ [{y_entry_lo:.2f}, {y_entry_hi:.2f}] cm")
    n_total_kernel_runs = (args.n_tracks * len(args.track_lengths_cm)
                           * args.n_seeds_per_track)
    print(f"Scan: {args.n_tracks} tracks × "
          f"{len(args.track_lengths_cm)} lengths × "
          f"{args.n_seeds_per_track} seeds = "
          f"{n_total_kernel_runs} kernel runs")

    master_rng = np.random.default_rng(args.master_rng_seed)
    kernel_common = dict(
        detector=detector, physics=physics, sim=sim,
        detsim=detsim, drifting=drifting, quenching=quenching,
        pixels_from_track=pixels_from_track,
        create_xoroshiro128p_states=create_xoroshiro128p_states,
        response_table=response_table,
    )

    tricell_records = []
    n_accepted_tracks = 0
    n_valid_tricells = 0

    for track_length_cm in args.track_lengths_cm:
        for track_index in range(args.n_tracks):
            # Sample direction (zenith from cos²θ, azimuth uniform)
            zenith_rad, azimuth_rad = sample_cosmic_direction(master_rng)
            direction_x = sin(zenith_rad) * cos(azimuth_rad)
            direction_y = sin(zenith_rad) * sin(azimuth_rad)
            direction_z = cos(zenith_rad) * into_drift_volume

            # Drift distance at the track midpoint
            half_drift_extent = abs(direction_z) * track_length_cm / 2
            drift_lo = max(args.drift_distance_min_cm,
                           half_drift_extent + 1)
            drift_hi = args.drift_distance_max_cm - half_drift_extent - 1
            if drift_lo >= drift_hi:
                continue
            midpoint_drift_cm = master_rng.uniform(drift_lo, drift_hi)
            midpoint_z_cm = anode_z_cm + into_drift_volume * midpoint_drift_cm

            # Midpoint on the anode plane, then back out to start/end
            midpoint_x_cm = master_rng.uniform(x_entry_lo, x_entry_hi)
            midpoint_y_cm = master_rng.uniform(y_entry_lo, y_entry_hi)
            start_x = midpoint_x_cm - 0.5 * track_length_cm * direction_x
            start_y = midpoint_y_cm - 0.5 * track_length_cm * direction_y
            start_z = midpoint_z_cm - 0.5 * track_length_cm * direction_z
            end_x = midpoint_x_cm + 0.5 * track_length_cm * direction_x
            end_y = midpoint_y_cm + 0.5 * track_length_cm * direction_y
            end_z = midpoint_z_cm + 0.5 * track_length_cm * direction_z

            # Reject if track exits the allowed (x, y) entry box
            if not (x_entry_lo <= start_x <= x_entry_hi
                    and x_entry_lo <= end_x <= x_entry_hi
                    and y_entry_lo <= start_y <= y_entry_hi
                    and y_entry_lo <= end_y <= y_entry_hi):
                continue

            track_recarray = build_muon_segment(
                (start_x, start_y, start_z),
                (direction_x, direction_y, direction_z),
                track_length_cm,
                args.track_dEdx_MeV_per_cm)
            n_accepted_tracks += 1

            for seed_index in range(args.n_seeds_per_track):
                kernel_seed = (args.master_rng_seed
                               + track_index * 10000
                               + seed_index * 17)
                kernel_result = run_kernel_chain(
                    tracks=track_recarray,
                    kernel_rng_seed=kernel_seed, **kernel_common)
                if kernel_result is None:
                    continue
                signals = kernel_result['signals']
                neighboring_pixels = kernel_result['neighboring_pixels']
                n_electrons_post_drift = kernel_result['n_electrons_post_drift']
                tracks_after_drift = kernel_result['tracks_after_drift']

                # B: drift-landed charge per pixel
                landed_charge_rng = np.random.default_rng(kernel_seed)
                drift_landed_per_pixel = landed_charge_per_pixel_with_rng_kicks(
                    detector, tracks_after_drift,
                    n_electrons_post_drift, landed_charge_rng)

                # C-hit pixel dict from kernel response
                hit_pixel_dict = build_hit_pixel_dict(
                    neighboring_pixels, signals, id2pixel,
                    args.minimum_pixel_charge_threshold)

                # Find Z-shape tricells
                tricells_found = list(find_zshape_tricells(
                    hit_pixel_dict,
                    args.minimum_witness_charge_threshold))

                for tricell in tricells_found:
                    ds_reconstructed_cm, length_inside_w1_cm = \
                        reconstruct_ds_from_witnesses(
                            tricell, detector, detector.TIME_SAMPLING)
                    if ds_reconstructed_cm is None:
                        continue

                    i_x_center, i_y_center = tricell['center_pixel_key']
                    x_center_cm, y_center_cm = pixel_indices_to_center(
                        detector, i_x_center, i_y_center)

                    # Geometric truth ds through the middle pixel's pillar
                    ds_geometric_truth_cm = segment_length_through_pixel_pillar(
                        (start_x, start_y, start_z),
                        (end_x, end_y, end_z),
                        x_center_cm - pitch_cm / 2,
                        x_center_cm + pitch_cm / 2,
                        y_center_cm - pitch_cm / 2,
                        y_center_cm + pitch_cm / 2)
                    if ds_geometric_truth_cm <= 0:
                        continue

                    q_truth = (n_electrons_post_drift
                               * (ds_geometric_truth_cm
                                  / track_length_cm))
                    q_drift_landed = drift_landed_per_pixel.get(
                        (i_x_center, i_y_center), 0.0)
                    q_kernel_response = tricell['center_collected_charge']

                    dQ_per_dx_truth = q_truth / ds_geometric_truth_cm
                    dQ_per_dx_drift_landed = (q_drift_landed
                                              / ds_geometric_truth_cm)
                    dQ_per_dx_kernel_truth_ds = (q_kernel_response
                                                 / ds_geometric_truth_cm)
                    dQ_per_dx_kernel_recon_ds = (q_kernel_response
                                                 / ds_reconstructed_cm)

                    tricell_records.append(dict(
                        track_index=track_index,
                        seed_index=seed_index,
                        track_length_cm=track_length_cm,
                        zenith_rad=zenith_rad,
                        azimuth_rad=azimuth_rad,
                        midpoint_drift_cm=midpoint_drift_cm,
                        column_axis=tricell['column_axis'],
                        traversal_axis=tricell['traversal_axis'],
                        drift_direction=tricell['drift_direction'],
                        i_x_center=i_x_center,
                        i_y_center=i_y_center,
                        ds_geometric_truth_cm=ds_geometric_truth_cm,
                        ds_reconstructed_cm=ds_reconstructed_cm,
                        length_inside_w1_cm=length_inside_w1_cm,
                        q_truth=q_truth,
                        q_drift_landed=q_drift_landed,
                        q_kernel_response=q_kernel_response,
                        dQ_per_dx_truth=dQ_per_dx_truth,
                        dQ_per_dx_drift_landed=dQ_per_dx_drift_landed,
                        dQ_per_dx_kernel_truth_ds=dQ_per_dx_kernel_truth_ds,
                        dQ_per_dx_kernel_recon_ds=dQ_per_dx_kernel_recon_ds,
                        sigma_transverse_cm=float(
                            tracks_after_drift["tran_diff"][0]),
                        sigma_longitudinal_cm=float(
                            tracks_after_drift["long_diff"][0]),
                        n_electrons_post_drift=n_electrons_post_drift,
                    ))
                    n_valid_tricells += 1

            if (track_index + 1) % 25 == 0 or args.verbose:
                print(f"  L={track_length_cm:.1f}cm, "
                      f"track {track_index + 1}/{args.n_tracks}: "
                      f"{n_valid_tricells} valid tricells so far")

    if not tricell_records:
        print("No valid tricells. Try more --n-tracks, or relax "
              "--minimum-pixel-charge-threshold.")
        return

    print(f"\nTotal: {n_valid_tricells} valid tricells from "
          f"{n_accepted_tracks} accepted tracks "
          f"(of {len(args.track_lengths_cm) * args.n_tracks} "
          f"attempted).")

    # ---- Save raw arrays ----
    record_arrays = {key: np.array([record[key]
                                    for record in tricell_records])
                     for key in tricell_records[0]}
    output_npz_path = f"{args.outdir}/tricell_results.npz"
    np.savez(output_npz_path, **record_arrays)
    print(f"wrote {output_npz_path}")

    # ---- Plots ----
    make_summary_plots(tricell_records, args.outdir, pitch_cm)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_summary_plots(tricell_records, outdir, pitch_cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dQ_per_dx_truth = np.array(
        [r['dQ_per_dx_truth'] for r in tricell_records])
    dQ_per_dx_drift_landed = np.array(
        [r['dQ_per_dx_drift_landed'] for r in tricell_records])
    dQ_per_dx_kernel_truth_ds = np.array(
        [r['dQ_per_dx_kernel_truth_ds'] for r in tricell_records])
    dQ_per_dx_kernel_recon_ds = np.array(
        [r['dQ_per_dx_kernel_recon_ds'] for r in tricell_records])
    zenith_rad = np.array([r['zenith_rad'] for r in tricell_records])
    track_length_cm = np.array(
        [r['track_length_cm'] for r in tricell_records])
    drift_cm = np.array(
        [r['midpoint_drift_cm'] for r in tricell_records])
    sigma_transverse_cm = np.array(
        [r['sigma_transverse_cm'] for r in tricell_records])
    column_axis = np.array(
        [r['column_axis'] for r in tricell_records])
    ds_truth_cm = np.array(
        [r['ds_geometric_truth_cm'] for r in tricell_records])
    ds_recon_cm = np.array(
        [r['ds_reconstructed_cm'] for r in tricell_records])

    def safe_ratio(numerator, denominator):
        return np.where(denominator != 0,
                        numerator / denominator, np.nan)

    ratio_drift_to_truth = safe_ratio(
        dQ_per_dx_drift_landed, dQ_per_dx_truth)
    ratio_kernel_truth_ds_to_truth = safe_ratio(
        dQ_per_dx_kernel_truth_ds, dQ_per_dx_truth)
    ratio_kernel_recon_ds_to_truth = safe_ratio(
        dQ_per_dx_kernel_recon_ds, dQ_per_dx_truth)
    ratio_kernel_to_drift_landed = safe_ratio(
        dQ_per_dx_kernel_truth_ds, dQ_per_dx_drift_landed)

    def save_figure(fig, filename):
        fig.tight_layout()
        path = f"{outdir}/{filename}"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        plt.close(fig)

    # 1. dQ/dx ratios to truth
    fig, ax = plt.subplots(figsize=(10, 5.5))
    histogram_bins = np.linspace(0, 3, 60)
    for ratio_array, label, color in [
        (ratio_drift_to_truth,
         "drift-landed / truth", "C0"),
        (ratio_kernel_truth_ds_to_truth,
         "kernel response / truth   (using truth ds)", "C1"),
        (ratio_kernel_recon_ds_to_truth,
         "kernel response / truth   (using reconstructed ds)", "C2"),
    ]:
        ax.hist(ratio_array, bins=histogram_bins, alpha=0.55,
                label=f"{label}: mean={np.nanmean(ratio_array):.3f}, "
                      f"std={np.nanstd(ratio_array):.3f}",
                color=color)
    ax.axvline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("ratio of measured dQ/dx to truth dQ/dx")
    ax.set_ylabel("tricell count")
    ax.set_title("Tricell dQ/dx ratios to truth")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ratios_to_truth_hist.png")

    # 2. Ratios vs zenith angle
    fig, ax = plt.subplots(figsize=(10, 5.5))
    zenith_deg = np.degrees(zenith_rad)
    angle_bins = np.linspace(zenith_deg.min(), zenith_deg.max(), 8)
    for ratio_array, label, color in [
        (ratio_drift_to_truth, "drift-landed / truth", "C0"),
        (ratio_kernel_truth_ds_to_truth,
         "kernel / truth (truth ds)", "C1"),
        (ratio_kernel_recon_ds_to_truth,
         "kernel / truth (reconstructed ds)", "C2"),
    ]:
        bin_means, bin_stderrs, bin_centers = binned_mean_and_stderr(
            zenith_deg, ratio_array, angle_bins)
        ax.errorbar(bin_centers, bin_means, yerr=bin_stderrs,
                    marker="o", capsize=3, label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("track zenith angle [deg]")
    ax.set_ylabel("dQ/dx ratio to truth")
    ax.set_title("Tricell dQ/dx ratio vs track zenith angle")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ratios_vs_zenith_angle.png")

    # 3. Ratios vs track length
    fig, ax = plt.subplots(figsize=(10, 5.5))
    unique_track_lengths = sorted(set(track_length_cm.tolist()))
    for ratio_array, label, color in [
        (ratio_drift_to_truth, "drift-landed / truth", "C0"),
        (ratio_kernel_truth_ds_to_truth,
         "kernel / truth (truth ds)", "C1"),
        (ratio_kernel_recon_ds_to_truth,
         "kernel / truth (reconstructed ds)", "C2"),
    ]:
        means_per_length = []
        stderrs_per_length = []
        for length_cm in unique_track_lengths:
            mask = track_length_cm == length_cm
            finite_values = ratio_array[mask & np.isfinite(ratio_array)]
            if len(finite_values):
                means_per_length.append(float(np.mean(finite_values)))
                stderrs_per_length.append(
                    float(np.std(finite_values))
                    / sqrt(max(len(finite_values), 1)))
            else:
                means_per_length.append(np.nan)
                stderrs_per_length.append(np.nan)
        ax.errorbar(unique_track_lengths, means_per_length,
                    yerr=stderrs_per_length, marker="o", capsize=3,
                    label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("total track length [cm]")
    ax.set_ylabel("dQ/dx ratio to truth")
    ax.set_title("Tricell dQ/dx ratio vs total track length")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ratios_vs_track_length.png")

    # 4. Ratios vs drift distance
    fig, ax = plt.subplots(figsize=(10, 5.5))
    drift_bins = np.linspace(drift_cm.min(), drift_cm.max(), 8)
    for ratio_array, label, color in [
        (ratio_drift_to_truth, "drift-landed / truth", "C0"),
        (ratio_kernel_truth_ds_to_truth,
         "kernel / truth (truth ds)", "C1"),
        (ratio_kernel_recon_ds_to_truth,
         "kernel / truth (reconstructed ds)", "C2"),
    ]:
        bin_means, bin_stderrs, bin_centers = binned_mean_and_stderr(
            drift_cm, ratio_array, drift_bins)
        ax.errorbar(bin_centers, bin_means, yerr=bin_stderrs,
                    marker="o", capsize=3, label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("drift distance at tricell midpoint [cm]")
    ax.set_ylabel("dQ/dx ratio to truth")
    ax.set_title("Tricell dQ/dx ratio vs drift distance")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ratios_vs_drift_distance.png")

    # 5. ds reconstructed vs truth
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ds_truth_cm, ds_recon_cm, s=4, alpha=0.4)
    plot_limit = max(ds_truth_cm.max(), ds_recon_cm.max()) * 1.1
    ax.plot([0, plot_limit], [0, plot_limit], "k--", lw=0.6,
            label="reconstructed = truth")
    ax.set_xlim(0, plot_limit)
    ax.set_ylim(0, plot_limit)
    ax.set_xlabel("ds through middle pixel, geometric truth [cm]")
    ax.set_ylabel("ds through middle pixel, reconstructed [cm]")
    ax.set_title("Reconstructed ds vs geometric truth")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ds_reconstructed_vs_truth.png")

    # 6. Drift-landed / truth vs σ_transverse / pitch (Far Field view)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(sigma_transverse_cm / pitch_cm, ratio_drift_to_truth,
               s=4, alpha=0.4)
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("transverse diffusion width / pixel pitch")
    ax.set_ylabel("drift-landed dQ/dx  /  truth dQ/dx")
    ax.set_title("Far Field effect: landed-charge calibration vs "
                 "diffusion width")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_drift_landed_vs_diffusion_width.png")

    # 7. Kernel response vs drift-landed (per tricell, dQ/dx)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(dQ_per_dx_drift_landed, dQ_per_dx_kernel_truth_ds,
               s=4, alpha=0.4)
    plot_max = max(dQ_per_dx_drift_landed.max(),
                   dQ_per_dx_kernel_truth_ds.max())
    ax.plot([0, plot_max], [0, plot_max], "k--", lw=0.6,
            label="kernel = drift-landed")
    ax.set_xlabel("drift-landed dQ/dx (physically arrives on pad)")
    ax.set_ylabel("kernel response dQ/dx (induced-current readout)")
    ax.set_title("Kernel response vs drift-landed charge")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_kernel_vs_drift_landed.png")

    # 8. Ratios split by column-axis orientation
    fig, ax = plt.subplots(figsize=(9, 5.5))
    label_pairs = [
        ("drift-landed / truth", ratio_drift_to_truth),
        ("kernel / truth (truth ds)", ratio_kernel_truth_ds_to_truth),
        ("kernel / truth (recon ds)", ratio_kernel_recon_ds_to_truth),
    ]
    means_x_axis = []
    means_y_axis = []
    for label, ratio_array in label_pairs:
        mask_x = column_axis == 'x'
        mask_y = column_axis == 'y'
        means_x_axis.append(
            float(np.nanmean(ratio_array[mask_x]))
            if mask_x.sum() else np.nan)
        means_y_axis.append(
            float(np.nanmean(ratio_array[mask_y]))
            if mask_y.sum() else np.nan)
    bar_positions = np.arange(len(label_pairs))
    ax.bar(bar_positions - 0.2, means_x_axis, 0.4,
           label="w_1 = x-column")
    ax.bar(bar_positions + 0.2, means_y_axis, 0.4,
           label="w_1 = y-column")
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([label for label, _ in label_pairs],
                       fontsize=9)
    ax.set_ylabel("mean ratio")
    ax.set_title("Tricell dQ/dx ratios split by w_1 column axis")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "tricell_ratios_by_column_axis.png")

    # 9. Per-tricell seed-to-seed fluctuation reliability
    record_groups_by_tricell = {}
    for record in tricell_records:
        group_key = (record['track_index'],
                     record['track_length_cm'],
                     record['i_x_center'],
                     record['i_y_center'],
                     record['column_axis'])
        record_groups_by_tricell.setdefault(group_key, []).append(record)

    truth_means = []
    drift_landed_stderrs = []
    kernel_stderrs = []
    for group_key, records in record_groups_by_tricell.items():
        if len(records) < 2:
            continue
        truth_means.append(np.mean(
            [r['dQ_per_dx_truth'] for r in records]))
        drift_landed_stderrs.append(np.std(
            [r['dQ_per_dx_drift_landed'] for r in records]))
        kernel_stderrs.append(np.std(
            [r['dQ_per_dx_kernel_truth_ds'] for r in records]))

    if truth_means:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        truth_means = np.array(truth_means)
        drift_landed_stderrs = np.array(drift_landed_stderrs)
        kernel_stderrs = np.array(kernel_stderrs)
        ax.scatter(truth_means,
                   drift_landed_stderrs / truth_means,
                   s=6, alpha=0.5,
                   label="drift-landed seed-stderr / truth",
                   color="C0")
        ax.scatter(truth_means,
                   kernel_stderrs / truth_means,
                   s=6, alpha=0.5,
                   label="kernel response seed-stderr / truth",
                   color="C1")
        ax.set_xlabel("truth dQ/dx")
        ax.set_ylabel("per-tricell relative stderr across seeds")
        ax.set_title("Seed-to-seed fluctuation reliability per tricell")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        save_figure(fig, "tricell_seed_fluctuation_reliability.png")

    # 10. (kernel − drift-landed) / truth vs diffusion width
    fig, ax = plt.subplots(figsize=(10, 5.5))
    delta_kernel_minus_drift_landed = (
        (dQ_per_dx_kernel_truth_ds - dQ_per_dx_drift_landed)
        / dQ_per_dx_truth)
    ax.scatter(sigma_transverse_cm / pitch_cm,
               delta_kernel_minus_drift_landed,
               s=4, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.6,
               label="kernel = drift-landed")
    ax.set_xlabel("transverse diffusion width / pixel pitch")
    ax.set_ylabel("(kernel − drift-landed) / truth")
    ax.set_title("Induced-current contribution beyond directly-landed "
                 "charge")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_kernel_minus_drift_landed.png")

    # 11. Calibration bias: truth-ds denominator vs recon-ds denominator
    fig, ax = plt.subplots(figsize=(10, 5.5))
    histogram_bins = np.linspace(0, 3, 60)
    ax.hist(ratio_kernel_truth_ds_to_truth, bins=histogram_bins,
            alpha=0.55,
            label=f"kernel / truth, using TRUTH ds: "
                  f"mean={np.nanmean(ratio_kernel_truth_ds_to_truth):.3f}",
            color="C1")
    ax.hist(ratio_kernel_recon_ds_to_truth, bins=histogram_bins,
            alpha=0.55,
            label=f"kernel / truth, using RECONSTRUCTED ds: "
                  f"mean={np.nanmean(ratio_kernel_recon_ds_to_truth):.3f}",
            color="C2")
    ax.axvline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("ratio of measured dQ/dx to truth dQ/dx")
    ax.set_ylabel("tricell count")
    ax.set_title(
        "Calibration bias from ds reconstruction\n"
        "(difference between distributions = "
        "timing-reconstruction contribution to bias)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_calibration_bias_truth_vs_recon_ds.png")

    # ---- Console summary ----
    n_records = len(tricell_records)
    print("\n" + "=" * 70)
    print(f"Total tricells:           {n_records}")
    print(f"Unique (track, length) pairs with ≥1 tricell: "
          f"{len(set((r['track_index'], r['track_length_cm']) for r in tricell_records))}")
    print(f"drift-landed / truth                       : "
          f"mean={np.nanmean(ratio_drift_to_truth):.3f}, "
          f"std={np.nanstd(ratio_drift_to_truth):.3f}")
    print(f"kernel / truth (using truth ds)            : "
          f"mean={np.nanmean(ratio_kernel_truth_ds_to_truth):.3f}, "
          f"std={np.nanstd(ratio_kernel_truth_ds_to_truth):.3f}")
    print(f"kernel / truth (using reconstructed ds)    : "
          f"mean={np.nanmean(ratio_kernel_recon_ds_to_truth):.3f}, "
          f"std={np.nanstd(ratio_kernel_recon_ds_to_truth):.3f}")
    print(f"kernel / drift-landed                      : "
          f"mean={np.nanmean(ratio_kernel_to_drift_landed):.3f}, "
          f"std={np.nanstd(ratio_kernel_to_drift_landed):.3f}")
    print(f"ds_reconstructed / ds_geometric_truth      : "
          f"mean={np.nanmean(ds_recon_cm / ds_truth_cm):.3f}")
    print("=" * 70)


def binned_mean_and_stderr(x_values, y_values, x_bins):
    """Bin y_values by x_values, return (means, standard_errors, bin_centers)."""
    bin_indices = np.digitize(x_values, x_bins) - 1
    means = []
    standard_errors = []
    bin_centers = []
    for bin_index in range(len(x_bins) - 1):
        mask = (bin_indices == bin_index) & np.isfinite(y_values)
        values_in_bin = y_values[mask]
        if len(values_in_bin) >= 2:
            means.append(float(np.mean(values_in_bin)))
            standard_errors.append(
                float(np.std(values_in_bin))
                / sqrt(len(values_in_bin)))
        else:
            means.append(np.nan)
            standard_errors.append(np.nan)
        bin_centers.append(0.5 * (x_bins[bin_index]
                                  + x_bins[bin_index + 1]))
    return (np.array(means),
            np.array(standard_errors),
            np.array(bin_centers))


if __name__ == "__main__":
    main()
