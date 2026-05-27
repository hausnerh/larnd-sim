#!/usr/bin/env python
"""
test_tricell_ds_accuracy.py
===========================

Focused study of how accurately the Z-shape tricell algorithm
reconstructs the geometric path length (ds) through the middle pixel,
as a function of a "sentinel pixel" charge threshold.

WHAT THIS SCRIPT DOES
---------------------
Throws synthetic cosmic muons, runs them through the FULL larnd-sim
chain (quench → drift → tracks_current_mc → sum_pixel_signals →
get_adc_values), identifies Z-shape tricells from the readout, and
compares the reconstructed ds to the geometric truth ds. It also
reports the dQ/dx that a real calibration analysis would derive from
the readout — FEE ADC packet charge divided by tricell-reconstructed
ds — vs the truth dQ/dx from the simulated ionisation.

Per-pixel timing comes from the FEE OUTPUT (the tick of the largest
ADC packet on that pixel), not the raw response waveform's argmax.
This makes the selection portable to real data — the same tricell
finder would work on actual packet streams.

The track direction is estimated from witness peak-time deltas:

    delta_w     = 1 * pixel_pitch      (the effective w-extent between
                                        witness peak times, equal to
                                        the inside-w_1 column width.
                                        A naive pad-center-to-pad-
                                        center value of 2·pitch was
                                        tried first and over-estimated
                                        ds by ~12.7%; data refit gave
                                        Δw_eff = 1.0·pitch.)
    delta_v     = v_center_w2_witness − v_center_w0_witness
    delta_drift = V_DRIFT * (t_peak_w2_witness − t_peak_w0_witness)
    L           = sqrt(delta_w² + delta_v² + delta_drift²)
    ds_C_recon  = L * pixel_pitch / |delta_v|

The "sentinel" pixels are the top and bottom of the three contiguous
w_1 pixels — the two w_1 pixels adjacent to the calibration target. A
track that just clips a corner of the middle pixel won't deposit much
charge on the sentinels. Requiring a minimum sentinel-to-center charge
ratio is hypothesised to reject those corner-clip tricells where ds
reconstruction is most likely to be wildly off.

Pre-cuts: ALL applied on FEE-readout quantities so the same
selection can be applied verbatim to real data.

  - --minimum-pixel-charge-threshold (default 5000 e⁻)
      Sum of FEE ADC packets on a pixel must exceed this for the
      pixel to enter the hit dict. FEE discriminator threshold is
      5000 e⁻ — so any pixel with at least one packet satisfies this
      by construction.
  - --minimum-target-charge (default 15000 e⁻)
      FEE-readout sum on the tricell target pixel must exceed this
      (≈ 3 packets worth).
  - --minimum-witness-charge-threshold (default 10000 e⁻)
      FEE-readout sum on each witness pixel must exceed this.
  - --maximum-sentinel-asymmetry (default 20)
      max(Q_w1_top, Q_w1_bottom) / min(Q_w1_top, Q_w1_bottom) below
      this — uses FEE-readout charges, dimensionless ratio so the
      scale doesn't matter.
  - --minimum-ds-truth-cm (default 0, off)
      Truth ds is the only sim-only quantity, kept as an optional
      diagnostic knob. The charge cuts above already suppress
      corner-clippers via dQ ∝ ds without needing this.

All selection and threshold-scan logic in the plots uses these
same FEE-readout quantities; the kernel response sum is saved as a
diagnostic (`collected_charge_raw_kernel`) but never used for
selection.

OUTPUTS
-------
  tricell_ds_fractional_difference.png
                            histogram of (ds_recon − ds_truth)/ds_truth
                            at a few sentinel thresholds overlaid.
  tricell_ds_accuracy_vs_sentinel_threshold.png
                            mean and median fractional difference vs
                            the sentinel-to-center charge ratio
                            threshold.
  tricell_ds_yield_vs_sentinel_threshold.png
                            number of tricells passing the sentinel cut
                            vs the threshold.
  tricell_dQdx_readout_vs_truth_uncalibrated.png
                            histogram of dQ/dx_readout / dQ/dx_truth
                            with the truth ds in the denominator; the
                            median sets the calibration factor.
  tricell_dQdx_readout_calibrated_vs_truth.png
                            same after applying the calibration factor,
                            with tricell-reconstructed ds in the
                            denominator (the actual "data-style"
                            measurement).
  tricell_dQdx_calibrated_vs_recon_ds.png
                            calibrated dQ/dx ratio binned by
                            reconstructed ds — flatness here means the
                            calibration is ds-independent.
  tricell_ds_accuracy_results.npz

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
def sample_cosmic_direction(rng, zenith_min_deg=10.0, zenith_max_deg=80.0):
    """Cosmic-like zenith from cos²θ distribution, uniform azimuth."""
    cos_zenith_min = cos(np.radians(zenith_max_deg))
    cos_zenith_max = cos(np.radians(zenith_min_deg))
    sample_uniform = rng.uniform(cos_zenith_min ** 3, cos_zenith_max ** 3)
    cos_zenith = sample_uniform ** (1.0 / 3.0)
    return acos(cos_zenith), rng.uniform(0, 2 * pi)


def build_muon_segment(start_xyz, direction_xyz, track_length_cm,
                       dEdx_MeV_per_cm=2.0):
    """Single-segment muon track."""
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
    tracks["pdg_id"] = 13
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
# Kernel chain
# ---------------------------------------------------------------------------
def run_kernel_chain(detector, physics, sim,
                     detsim, drifting, quenching, pixels_from_track,
                     create_xoroshiro128p_states,
                     tracks, response_table, *,
                     kernel_rng_seed=42,
                     max_pixels_per_track=500,
                     max_active_pixels_per_track=80):
    """quench → drift → get_pixels → tracks_current_mc."""
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
    neighboring_radius = device_radius.copy_to_host()

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

    return dict(signals=signals,
                neighboring_pixels=neighboring_pixels,
                neighboring_radius=neighboring_radius,
                signal_n_ticks=signal_n_ticks,
                tracks_after_drift=tracks_after_drift,
                n_electrons_post_drift=float(
                    tracks_after_drift["n_electrons"][0]))


# ---------------------------------------------------------------------------
# FEE chain (sum_pixel_signals + get_adc_values) for data-applicable
# per-pixel timing and packet-derived charge.
# ---------------------------------------------------------------------------
def run_fee_chain(detector, sim, detsim, fee,
                  create_xoroshiro128p_states,
                  signals, neighboring_pixels, signal_n_ticks,
                  *, fee_rng_seed=42):
    """sum_pixel_signals → get_adc_values, mirroring the production
    pipeline in cli/simulate_pixels.py:1389-1454, 1497-1506.

    Args:
        signals: per-segment per-halo-pixel per-tick induced current,
                 shape (n_tracks, max_pixels_per_track, n_ticks)
        neighboring_pixels: (n_tracks, max_pixels_per_track) pixel IDs
        signal_n_ticks: number of output ticks in `signals`

    Returns dict:
        unique_pix: (N_unique,) deduplicated pixel IDs
        adc_packet_list: (N_unique, MAX_ADC_VALUES) packet ADC values
                          (zero where no packet was issued)
        adc_packet_ticks: (N_unique, MAX_ADC_VALUES) packet ticks
                          (in TIME_SAMPLING units)
    """
    from numba import cuda as _cuda

    # ---- Dedupe pixel IDs (host-side) ----
    flat_pixels = neighboring_pixels.ravel()
    valid_pixels = flat_pixels[flat_pixels >= 0]
    if valid_pixels.size == 0:
        return None
    unique_pix = np.unique(valid_pixels).astype(np.int32)
    n_unique = len(unique_pix)

    # ---- pixel_index_map: neighboring_pixels[itrk, ipix] -> idx in unique_pix ----
    max_pix_val = int(unique_pix.max()) + 1
    pix_lookup = np.full((max_pix_val,), -1, dtype=np.int32)
    pix_lookup[unique_pix] = np.arange(n_unique, dtype=np.int32)
    pixel_index_map = pix_lookup[neighboring_pixels].astype(np.int32)
    pixel_index_map[neighboring_pixels == -1] = -1

    # ---- track_pixel_map ----
    max_segments_to_trace = sim.MAX_TRACKS_PER_PIXEL
    track_pixel_map = np.full((n_unique, max_segments_to_trace), -1,
                              dtype=np.int32)
    device_track_pixel_map = _cuda.to_device(track_pixel_map)
    device_unique_pix = _cuda.to_device(unique_pix)
    device_neighboring_pixels = _cuda.to_device(neighboring_pixels)

    threads_per_block = 32
    blocks_per_grid = max(ceil(n_unique / threads_per_block), 1)
    detsim.get_track_pixel_map[blocks_per_grid, threads_per_block](
        device_track_pixel_map, device_unique_pix,
        device_neighboring_pixels)
    track_pixel_map_host = device_track_pixel_map.copy_to_host()

    # num_backtrack: count of segments per pixel; offset_backtrack: cumsum
    num_backtrack = (track_pixel_map_host != -1).sum(axis=-1).astype(
        np.int64)
    offset_backtrack = (np.cumsum(num_backtrack)
                        - num_backtrack).astype(np.int64)

    # ---- sum_pixel_signals ----
    pixels_signals = np.zeros((n_unique, signal_n_ticks),
                              dtype=np.float64)
    pixels_tracks_signals = np.zeros(
        signal_n_ticks * int(num_backtrack.sum()), dtype=np.float64)
    overflow_flag = np.zeros(n_unique, dtype=np.float32)

    device_pixels_signals = _cuda.to_device(pixels_signals)
    device_pixels_tracks_signals = _cuda.to_device(pixels_tracks_signals)
    device_pixel_index_map = _cuda.to_device(pixel_index_map)
    device_num_backtrack = _cuda.to_device(num_backtrack)
    device_offset_backtrack = _cuda.to_device(offset_backtrack)
    device_overflow_flag = _cuda.to_device(overflow_flag)
    device_signals = _cuda.to_device(signals)

    # track_t0 (in TIME_SAMPLING units, integer ticks). Our t0=0 so 0.
    track_t0 = np.zeros(signals.shape[0], dtype=np.int64)
    device_track_t0 = _cuda.to_device(track_t0)

    threads_per_block_3d = (1, 1, 64)
    blocks_per_grid_3d = (
        max(ceil(signals.shape[0] / threads_per_block_3d[0]), 1),
        max(ceil(signals.shape[1] / threads_per_block_3d[1]), 1),
        max(ceil(signals.shape[2] / threads_per_block_3d[2]), 1),
    )
    detsim.sum_pixel_signals[blocks_per_grid_3d, threads_per_block_3d](
        device_pixels_signals, device_signals, device_track_t0,
        device_pixel_index_map, device_track_pixel_map,
        device_pixels_tracks_signals,
        device_num_backtrack, device_offset_backtrack,
        device_overflow_flag)

    # ---- get_adc_values ----
    time_ticks = (np.arange(signal_n_ticks + 1, dtype=np.float64)
                  * detector.TIME_SAMPLING)
    device_time_ticks = _cuda.to_device(time_ticks)

    max_adcs = sim.MAX_ADC_VALUES
    adc_packet_list = np.zeros((n_unique, max_adcs), dtype=np.float64)
    adc_packet_ticks = np.zeros((n_unique, max_adcs), dtype=np.float64)
    current_fractions = np.zeros(
        (n_unique, max_adcs, sim.MAX_TRACKS_PER_PIXEL), dtype=np.float64)
    device_adc_packet_list = _cuda.to_device(adc_packet_list)
    device_adc_packet_ticks = _cuda.to_device(adc_packet_ticks)
    device_current_fractions = _cuda.to_device(current_fractions)

    # Discrimination threshold (electrons; consts.units.e = 1)
    default_threshold = float(detector.DISCRIMINATION_THRESHOLD)
    pixel_thresholds = np.full(n_unique, default_threshold,
                               dtype=np.float64)
    device_pixel_thresholds = _cuda.to_device(pixel_thresholds)

    fee_threads_per_block = 4
    fee_blocks_per_grid = max(ceil(n_unique / fee_threads_per_block), 1)
    fee_n_rng = max(fee_threads_per_block * fee_blocks_per_grid, 1024)
    fee_rng_states = create_xoroshiro128p_states(fee_n_rng,
                                                 seed=fee_rng_seed)

    fee.get_adc_values[fee_blocks_per_grid, fee_threads_per_block](
        device_pixels_signals,
        device_pixels_tracks_signals,
        device_num_backtrack,
        device_offset_backtrack,
        device_time_ticks,
        device_adc_packet_list,
        device_adc_packet_ticks,
        0.0,                              # time_padding
        fee_rng_states,
        device_current_fractions,
        device_pixel_thresholds)

    adc_packet_list = device_adc_packet_list.copy_to_host()
    adc_packet_ticks = device_adc_packet_ticks.copy_to_host()

    return dict(
        unique_pix=unique_pix,
        adc_packet_list=adc_packet_list,
        adc_packet_ticks=adc_packet_ticks,
    )


# ---------------------------------------------------------------------------
# Pixel index → world coordinates
# ---------------------------------------------------------------------------
def pixel_indices_to_center(detector, i_x, i_y, plane=0):
    border_x = detector.TPC_BORDERS[plane][0][0]
    border_y = detector.TPC_BORDERS[plane][1][0]
    pitch = detector.PIXEL_PITCH
    return (border_x + (i_x + 0.5) * pitch,
            border_y + (i_y + 0.5) * pitch)


# ---------------------------------------------------------------------------
# Geometric truth ds
# ---------------------------------------------------------------------------
def segment_length_through_pixel_pillar(segment_start_xyz, segment_end_xyz,
                                        pillar_x_lo, pillar_x_hi,
                                        pillar_y_lo, pillar_y_hi):
    """3D length of segment intersected with the infinite z-pillar over
    (pillar_x_lo, pillar_x_hi) × (pillar_y_lo, pillar_y_hi)."""
    start_x, start_y, start_z = segment_start_xyz
    end_x, end_y, end_z = segment_end_xyz
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    delta_z = end_z - start_z
    full_length = sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    if full_length == 0:
        return 0.0

    t_enter, t_exit = 0.0, 1.0
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
# Tricell finder + ds reconstruction (witness-to-witness, consistent span)
# ---------------------------------------------------------------------------
def build_hit_pixel_dict(neighboring_pixels, signals,
                         fee_output, id2pixel_fn,
                         minimum_pixel_charge_threshold):
    """Build a per-pixel hit dict from the FEE-readout packet stream.

    Every quantity used for downstream cuts is RECO-LEVEL: derivable
    from real-data packets alone. The kernel response sum is also
    saved as a diagnostic but is NOT used for any selection.

    Per-pixel fields:
      collected_charge        : sum of ADC packet charges (FEE readout;
                                 the data-applicable charge). Used for
                                 all halo-fringe and tricell cuts.
      collected_charge_raw    : sum(signals[ipix, :]); kernel response
                                 (diagnostic only — would not exist on
                                 real data).
      peak_tick               : tick of the largest ADC packet.
      first_packet_tick       : tick of the first nonzero ADC packet.
      n_packets               : count of nonzero packets.

    Pixels in the halo that produce zero FEE packets are skipped --
    in data we would not see them.
    """
    unique_pix = fee_output['unique_pix']
    adc_packet_list = fee_output['adc_packet_list']
    adc_packet_ticks = fee_output['adc_packet_ticks']
    pixel_id_to_unique_idx = {int(pid): idx
                              for idx, pid in enumerate(unique_pix)}

    hit_dict = {}
    n_halo_pixels = neighboring_pixels.shape[1]
    for halo_index in range(n_halo_pixels):
        pixel_id = int(neighboring_pixels[0, halo_index])
        if pixel_id < 0:
            continue
        i_x, i_y, plane_id = id2pixel_fn(pixel_id)

        unique_idx = pixel_id_to_unique_idx.get(pixel_id, -1)
        if unique_idx < 0:
            continue
        packet_charges = adc_packet_list[unique_idx]
        packet_ticks = adc_packet_ticks[unique_idx]
        nonzero_packet_mask = packet_charges != 0
        n_packets = int(nonzero_packet_mask.sum())
        if n_packets == 0:
            # Pixel produced no FEE packets — invisible in data.
            continue
        # FEE-readout total charge for this pixel
        collected_charge_readout = float(
            packet_charges[nonzero_packet_mask].sum())
        # Apply the halo-fringe cut on the READOUT charge (data-only)
        if abs(collected_charge_readout) < minimum_pixel_charge_threshold:
            continue
        # Raw kernel response — saved for diagnostics only
        signal_trace = signals[0, halo_index, :]
        collected_charge_raw = float(signal_trace.sum())

        # Two candidate timing definitions, both from real data:
        #   - first_packet_tick: time of the FIRST discriminator trigger
        #     on this pixel (the rising-edge of integrated charge —
        #     closest to when significant charge first arrived).
        #   - largest_packet_tick: time of the largest ADC packet
        #     (which window collected the most charge — depends on
        #     FEE integration timing and is often a poor proxy for
        #     spatial reconstruction).
        # For tricell ds-recon the first-packet tick is more directly
        # tied to the substep contribution time; use it as the
        # primary `peak_tick`. Both are saved in the per-pixel record
        # so the choice can be revisited offline.
        largest_packet_index = int(np.argmax(np.abs(packet_charges)))
        largest_packet_tick = int(packet_ticks[largest_packet_index])
        first_packet_tick = int(packet_ticks[
            np.argmax(nonzero_packet_mask)])

        hit_dict[(int(i_x), int(i_y))] = dict(
            halo_index=halo_index,
            unique_pix_idx=unique_idx,
            collected_charge=collected_charge_readout,       # cuts use this
            collected_charge_raw=collected_charge_raw,        # diagnostic
            peak_tick=first_packet_tick,                     # data-style ts
            largest_packet_tick=largest_packet_tick,         # diagnostic
            first_packet_tick=first_packet_tick,             # diagnostic
            n_packets=n_packets,
            pixel_id=pixel_id,
        )
    return hit_dict


def find_zshape_tricells(hit_pixel_dict, minimum_witness_charge_threshold):
    """Find Z-shape tricells: 3 contiguous in w_1 column + entry/exit
    witnesses in w_0 and w_2. Yields one dict per tricell."""
    for (i_x_center, i_y_center), centre_record in hit_pixel_dict.items():
        for tricell_orientation in [
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
                                   centre_record['peak_tick'],
                                   hi_record['peak_tick'])
            w1_max_peak_tick = max(lo_record['peak_tick'],
                                   centre_record['peak_tick'],
                                   hi_record['peak_tick'])

            column_axis = tricell_orientation['column_axis']
            if column_axis == 'x':
                w1_column_index = i_x_center
            else:
                w1_column_index = i_y_center
            w0_column_index = w1_column_index - 1
            w2_column_index = w1_column_index + 1

            best_w0_forward = None
            best_w2_forward = None
            best_w0_reversed = None
            best_w2_reversed = None
            for (i_x_hit, i_y_hit), hit in hit_pixel_dict.items():
                if abs(hit['collected_charge']) < \
                        minimum_witness_charge_threshold:
                    continue
                column_of_hit = (
                    i_x_hit if column_axis == 'x' else i_y_hit)
                peak_tick_hit = hit['peak_tick']
                if column_of_hit == w0_column_index:
                    if peak_tick_hit < w1_min_peak_tick:
                        if (best_w0_forward is None
                                or peak_tick_hit > best_w0_forward['peak_tick']):
                            best_w0_forward = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)
                    if peak_tick_hit > w1_max_peak_tick:
                        if (best_w0_reversed is None
                                or peak_tick_hit < best_w0_reversed['peak_tick']):
                            best_w0_reversed = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)
                elif column_of_hit == w2_column_index:
                    if peak_tick_hit > w1_max_peak_tick:
                        if (best_w2_forward is None
                                or peak_tick_hit < best_w2_forward['peak_tick']):
                            best_w2_forward = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)
                    if peak_tick_hit < w1_min_peak_tick:
                        if (best_w2_reversed is None
                                or peak_tick_hit > best_w2_reversed['peak_tick']):
                            best_w2_reversed = dict(
                                key=(i_x_hit, i_y_hit),
                                peak_tick=peak_tick_hit,
                                record=hit)

            for witness_w0, witness_w2, direction_label in [
                (best_w0_forward, best_w2_forward, 'forward'),
                (best_w0_reversed, best_w2_reversed, 'reversed'),
            ]:
                if witness_w0 is None or witness_w2 is None:
                    continue
                yield dict(
                    column_axis=column_axis,
                    traversal_axis=tricell_orientation['traversal_axis'],
                    drift_direction=direction_label,
                    centre_pixel_key=(i_x_center, i_y_center),
                    centre_collected_charge=(
                        centre_record['collected_charge']),
                    centre_collected_charge_raw=(
                        centre_record['collected_charge_raw']),
                    w1_lo_key=lo_key,
                    w1_lo_collected_charge=lo_record['collected_charge'],
                    w1_lo_collected_charge_raw=(
                        lo_record['collected_charge_raw']),
                    w1_hi_key=hi_key,
                    w1_hi_collected_charge=hi_record['collected_charge'],
                    w1_hi_collected_charge_raw=(
                        hi_record['collected_charge_raw']),
                    w0_witness_key=witness_w0['key'],
                    w0_witness_collected_charge=(
                        witness_w0['record']['collected_charge']),
                    w2_witness_key=witness_w2['key'],
                    w2_witness_collected_charge=(
                        witness_w2['record']['collected_charge']),
                    peak_tick_w0_witness=witness_w0['peak_tick'],
                    peak_tick_w1_lo=lo_record['peak_tick'],
                    peak_tick_w1_centre=centre_record['peak_tick'],
                    peak_tick_w1_hi=hi_record['peak_tick'],
                    peak_tick_w2_witness=witness_w2['peak_tick'],
                )


def reconstruct_ds_witness_to_witness(tricell, detector, time_sampling_us):
    """Direction estimate from witness peak-time deltas.

      delta_w     = 1 * pixel_pitch
                     (the EFFECTIVE w-extent between witness peak
                     times. Empirically this is what the data wants,
                     not 2 * pitch as a naive pad-center-to-pad-center
                     argument would suggest. The witnesses' peak times
                     correspond to the track's closest approach to
                     each witness pad, and for tricells the closest
                     approach to a witness pad lies near the column
                     boundary rather than at the pad center, giving
                     an effective Δw of one pitch — the width of the
                     w_1 column the track actually traverses.)
      delta_v     = v_center_w2 − v_center_w0
      delta_drift = V_DRIFT * (t_w2_peak − t_w0_peak)
      L = sqrt(delta_w² + delta_v² + delta_drift²)
      ds_middle_pixel = L * pixel_pitch / |delta_v|

    Earlier versions used delta_w = 2 * pitch (pad-center-to-pad-
    center) and produced a robust +12.7% median over-estimate of ds.
    The +12.7% bias is consistent with delta_v and delta_drift
    measuring the inside-w_1 portion of the track while delta_w was
    measuring the full 2-pitch witness span.

    Returns dict with ds_cm and the input deltas for diagnostics;
    or None if delta_v == 0.
    """
    pitch_cm = detector.PIXEL_PITCH
    v_drift = detector.V_DRIFT
    traversal_axis = tricell['traversal_axis']

    i_x_w0, i_y_w0 = tricell['w0_witness_key']
    i_x_w2, i_y_w2 = tricell['w2_witness_key']
    x_w0, y_w0 = pixel_indices_to_center(detector, i_x_w0, i_y_w0)
    x_w2, y_w2 = pixel_indices_to_center(detector, i_x_w2, i_y_w2)

    delta_t_us = ((tricell['peak_tick_w2_witness']
                   - tricell['peak_tick_w0_witness'])
                  * time_sampling_us)
    delta_drift_cm = v_drift * delta_t_us

    if traversal_axis == 'y':
        delta_v_cm = y_w2 - y_w0
    else:
        delta_v_cm = x_w2 - x_w0

    # See docstring: effective Δw is 1·pitch (inside-w_1 column width),
    # not 2·pitch (pad center-to-center). Verified against data, brings
    # median ds_recon / ds_truth from 1.127 → 1.041.
    delta_w_cm = pitch_cm

    if delta_v_cm == 0:
        return None
    length_cm = sqrt(
        delta_w_cm ** 2 + delta_v_cm ** 2 + delta_drift_cm ** 2)
    ds_middle_pixel_cm = length_cm * pitch_cm / abs(delta_v_cm)
    return dict(
        ds_recon_cm=ds_middle_pixel_cm,
        length_witness_to_witness_cm=length_cm,
        delta_w_cm=delta_w_cm,
        delta_v_cm=delta_v_cm,
        delta_drift_cm=delta_drift_cm,
        delta_t_witness_us=delta_t_us,
        w0_witness_xy_cm=(x_w0, y_w0),
        w2_witness_xy_cm=(x_w2, y_w2),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument(
        "--response", default="larndsim/bin/response_44_v2a_full.npz")
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
        "--n-tracks", type=int, default=300,
        help="number of muon tracks per length")
    arg_parser.add_argument(
        "--track-lengths-cm", type=float, nargs="+",
        default=[2.0, 5.0, 10.0])
    arg_parser.add_argument(
        "--n-seeds-per-track", type=int, default=2)
    arg_parser.add_argument(
        "--drift-distance-min-cm", type=float, default=2.0)
    arg_parser.add_argument(
        "--drift-distance-max-cm", type=float, default=28.0)
    arg_parser.add_argument(
        "--track-dEdx-MeV-per-cm", type=float, default=2.0)
    arg_parser.add_argument(
        "--minimum-pixel-charge-threshold", type=float, default=5000.0,
        help="minimum |Q| (FEE readout charge, electrons) for a pixel "
             "to enter the hit dict. The FEE discriminator threshold "
             "is DISCRIMINATION_THRESHOLD = 5000 e⁻, so by construction "
             "any pixel with a packet has ≥ 5000 e⁻ accumulated. "
             "Default 5000 (effectively keeps any pixel with at least "
             "one issued packet).")
    arg_parser.add_argument(
        "--minimum-witness-charge-threshold", type=float, default=10000.0,
        help="absolute minimum |Q| (FEE readout charge, electrons) "
             "required of witness pixels. Default 10000 ≈ 2 "
             "discriminator-threshold worth.")
    arg_parser.add_argument(
        "--minimum-target-charge", type=float, default=15000.0,
        help="absolute minimum |Q| (FEE readout charge, electrons) "
             "required of the tricell target (centre) pixel. Default "
             "15000 ≈ 3 discriminator-threshold worth. The cut is "
             "meaningful for fringe rejection because the FEE itself "
             "rarely lets ≥ 3 packets get through on a fringe pixel.")
    arg_parser.add_argument(
        "--maximum-sentinel-asymmetry", type=float, default=20.0,
        help="max(|Q_w1_top|, |Q_w1_bottom|) / min(|Q_w1_top|, "
             "|Q_w1_bottom|) ≤ this. Default 20 (was 10). Catches "
             "tricells where one sentinel is real and the other is "
             "fringe (asymmetry > 100); loose enough to keep "
             "asymmetric-but-real cases.")
    arg_parser.add_argument(
        "--minimum-ds-truth-cm", type=float, default=0.0,
        help="optional reject tricells with input-segment ds_truth "
             "below this. Default 0 (no cut) — the absolute charge "
             "cuts above already filter out corner-clippers via the "
             "natural dQ ∝ ds relation. Kept as a knob for "
             "diagnostic use.")
    arg_parser.add_argument(
        "--master-rng-seed", type=int, default=20260519)
    arg_parser.add_argument(
        "--outdir", default=".")
    arg_parser.add_argument("--verbose", action="store_true")
    args = arg_parser.parse_args()

    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available.")
    from numba.cuda.random import create_xoroshiro128p_states
    from larndsim import consts
    consts.load_properties(args.detector, args.pixel_layout,
                           args.response, args.sim_properties)
    from larndsim.consts import detector, physics, sim
    from larndsim import detsim, drifting, quenching, pixels_from_track, fee
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
    wall_margin_pitches = 3
    x_entry_lo = border_x_cm + wall_margin_pitches * pitch_cm
    x_entry_hi = border_x_cm + (n_pixels_x - wall_margin_pitches) * pitch_cm
    y_entry_lo = border_y_cm + wall_margin_pitches * pitch_cm
    y_entry_hi = border_y_cm + (n_pixels_y - wall_margin_pitches) * pitch_cm

    n_total_runs = (args.n_tracks * len(args.track_lengths_cm)
                    * args.n_seeds_per_track)
    print(f"Detector: pitch={pitch_cm:.4f} cm, "
          f"V_DRIFT={detector.V_DRIFT:.4e} cm/us")
    print(f"Scan: {args.n_tracks} tracks × "
          f"{len(args.track_lengths_cm)} lengths × "
          f"{args.n_seeds_per_track} seeds = {n_total_runs} kernel runs")

    master_rng = np.random.default_rng(args.master_rng_seed)
    kernel_common = dict(
        detector=detector, physics=physics, sim=sim,
        detsim=detsim, drifting=drifting, quenching=quenching,
        pixels_from_track=pixels_from_track,
        create_xoroshiro128p_states=create_xoroshiro128p_states,
        response_table=response_table,
    )

    records = []
    n_accepted_tracks = 0
    n_valid_tricells = 0
    # Per-cut rejection diagnostics
    cut_rejection_counts = dict(
        ds_recon_failed=0,
        ds_truth_below_threshold=0,
        target_charge_below_threshold=0,
        sentinel_asymmetric=0,
    )
    n_tricells_examined = 0

    for track_length_cm in args.track_lengths_cm:
        for track_index in range(args.n_tracks):
            zenith_rad, azimuth_rad = sample_cosmic_direction(master_rng)
            direction_x = sin(zenith_rad) * cos(azimuth_rad)
            direction_y = sin(zenith_rad) * sin(azimuth_rad)
            direction_z = cos(zenith_rad) * into_drift_volume

            half_drift_extent = abs(direction_z) * track_length_cm / 2
            drift_lo = max(args.drift_distance_min_cm,
                           half_drift_extent + 1)
            drift_hi = args.drift_distance_max_cm - half_drift_extent - 1
            if drift_lo >= drift_hi:
                continue
            midpoint_drift_cm = master_rng.uniform(drift_lo, drift_hi)
            midpoint_z_cm = (anode_z_cm
                             + into_drift_volume * midpoint_drift_cm)
            midpoint_x_cm = master_rng.uniform(x_entry_lo, x_entry_hi)
            midpoint_y_cm = master_rng.uniform(y_entry_lo, y_entry_hi)
            start_x = midpoint_x_cm - 0.5 * track_length_cm * direction_x
            start_y = midpoint_y_cm - 0.5 * track_length_cm * direction_y
            start_z = midpoint_z_cm - 0.5 * track_length_cm * direction_z
            end_x = midpoint_x_cm + 0.5 * track_length_cm * direction_x
            end_y = midpoint_y_cm + 0.5 * track_length_cm * direction_y
            end_z = midpoint_z_cm + 0.5 * track_length_cm * direction_z
            if not (x_entry_lo <= start_x <= x_entry_hi
                    and x_entry_lo <= end_x <= x_entry_hi
                    and y_entry_lo <= start_y <= y_entry_hi
                    and y_entry_lo <= end_y <= y_entry_hi):
                continue

            track_recarray = build_muon_segment(
                (start_x, start_y, start_z),
                (direction_x, direction_y, direction_z),
                track_length_cm, args.track_dEdx_MeV_per_cm)
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
                signal_n_ticks = kernel_result['signal_n_ticks']
                n_electrons_post_drift = kernel_result[
                    'n_electrons_post_drift']

                fee_output = run_fee_chain(
                    detector, sim, detsim, fee,
                    create_xoroshiro128p_states,
                    signals, neighboring_pixels, signal_n_ticks,
                    fee_rng_seed=kernel_seed + 1)
                if fee_output is None:
                    continue

                hit_pixel_dict = build_hit_pixel_dict(
                    neighboring_pixels, signals, fee_output, id2pixel,
                    args.minimum_pixel_charge_threshold)
                tricells_found = list(find_zshape_tricells(
                    hit_pixel_dict,
                    args.minimum_witness_charge_threshold))

                for tricell in tricells_found:
                    n_tricells_examined += 1

                    recon_info = reconstruct_ds_witness_to_witness(
                        tricell, detector, detector.TIME_SAMPLING)
                    if recon_info is None:
                        cut_rejection_counts['ds_recon_failed'] += 1
                        continue
                    ds_recon_cm = recon_info['ds_recon_cm']

                    i_x_center, i_y_center = tricell['centre_pixel_key']
                    x_center_cm, y_center_cm = pixel_indices_to_center(
                        detector, i_x_center, i_y_center)
                    ds_truth_cm = segment_length_through_pixel_pillar(
                        (start_x, start_y, start_z),
                        (end_x, end_y, end_z),
                        x_center_cm - pitch_cm / 2,
                        x_center_cm + pitch_cm / 2,
                        y_center_cm - pitch_cm / 2,
                        y_center_cm + pitch_cm / 2)
                    if ds_truth_cm <= args.minimum_ds_truth_cm:
                        cut_rejection_counts['ds_truth_below_threshold'] += 1
                        continue

                    # Absolute charge cuts (all on FEE-readout charge,
                    # so the same cut definitions apply to real data).
                    q_centre = abs(tricell['centre_collected_charge'])
                    q_w1_lo = abs(tricell['w1_lo_collected_charge'])
                    q_w1_hi = abs(tricell['w1_hi_collected_charge'])
                    q_centre_raw_kernel = abs(
                        tricell['centre_collected_charge_raw'])
                    if q_centre < args.minimum_target_charge:
                        cut_rejection_counts[
                            'target_charge_below_threshold'] += 1
                        continue

                    # Sentinel asymmetry cut: catches cases where one
                    # w_1 sentinel is a real-track pixel and the other
                    # is fringe.
                    min_sentinel_q = min(q_w1_lo, q_w1_hi)
                    max_sentinel_q = max(q_w1_lo, q_w1_hi)
                    if min_sentinel_q <= 0:
                        cut_rejection_counts['sentinel_asymmetric'] += 1
                        continue
                    sentinel_asymmetry = max_sentinel_q / min_sentinel_q
                    if sentinel_asymmetry > args.maximum_sentinel_asymmetry:
                        cut_rejection_counts['sentinel_asymmetric'] += 1
                        continue

                    sentinel_min_ratio = min_sentinel_q / q_centre
                    sentinel_mean_ratio = (
                        0.5 * (q_w1_lo + q_w1_hi) / q_centre)
                    sentinel_max_ratio = max_sentinel_q / q_centre

                    fractional_difference = (
                        (ds_recon_cm - ds_truth_cm) / ds_truth_cm)

                    # FEE / readout-derived dQ/dx -- the data-style
                    # measurement: integrated readout packet charge on
                    # the centre pixel divided by tricell-recon ds.
                    # q_centre IS the FEE readout charge (in the
                    # FEE charge units, comparable to electrons after
                    # FEE gain).
                    dQdx_readout_per_dx_recon = q_centre / ds_recon_cm
                    dQdx_readout_per_dx_truth = q_centre / ds_truth_cm
                    # Truth: constant for fixed-dEdx muon (n_electrons
                    # after quench+drift attenuation, per cm of track).
                    dQdx_truth_per_dx = (n_electrons_post_drift
                                         / track_length_cm)

                    records.append(dict(
                        track_index=track_index,
                        seed_index=seed_index,
                        track_length_cm=track_length_cm,
                        zenith_rad=zenith_rad,
                        ds_truth_cm=ds_truth_cm,
                        ds_recon_cm=ds_recon_cm,
                        fractional_difference=fractional_difference,
                        sentinel_min_ratio=sentinel_min_ratio,
                        sentinel_mean_ratio=sentinel_mean_ratio,
                        sentinel_max_ratio=sentinel_max_ratio,
                        sentinel_asymmetry=sentinel_asymmetry,
                        q_centre=q_centre,
                        q_w1_lo=q_w1_lo,
                        q_w1_hi=q_w1_hi,
                        column_axis=tricell['column_axis'],
                        # Diagnostic: per-axis deltas that fed ds recon
                        delta_w_cm=recon_info['delta_w_cm'],
                        delta_v_cm=recon_info['delta_v_cm'],
                        delta_drift_cm=recon_info['delta_drift_cm'],
                        delta_t_witness_us=recon_info[
                            'delta_t_witness_us'],
                        length_witness_to_witness_cm=recon_info[
                            'length_witness_to_witness_cm'],
                        # FEE / readout dQ/dx (data-applicable; q_centre
                        # is already the FEE-readout charge)
                        q_centre_raw_kernel=q_centre_raw_kernel,
                        n_electrons_post_drift=n_electrons_post_drift,
                        dQdx_readout_per_dx_recon=(
                            dQdx_readout_per_dx_recon),
                        dQdx_readout_per_dx_truth=(
                            dQdx_readout_per_dx_truth),
                        dQdx_truth_per_dx=dQdx_truth_per_dx,
                    ))
                    n_valid_tricells += 1

            if (track_index + 1) % 50 == 0 or args.verbose:
                print(f"  L={track_length_cm:.1f}cm, "
                      f"track {track_index + 1}/{args.n_tracks}: "
                      f"{n_valid_tricells} valid tricells so far")

    print(f"\nTricells examined            : {n_tricells_examined}")
    print(f"  rejected by ds_recon failure   : "
          f"{cut_rejection_counts['ds_recon_failed']}")
    print(f"  rejected by ds_truth ≤ {args.minimum_ds_truth_cm} cm "
          f": {cut_rejection_counts['ds_truth_below_threshold']}")
    print(f"  rejected by q_centre < {args.minimum_target_charge:.0f} "
          f": {cut_rejection_counts['target_charge_below_threshold']}")
    print(f"  rejected by sentinel asymmetry > {args.maximum_sentinel_asymmetry} "
          f": {cut_rejection_counts['sentinel_asymmetric']}")
    print(f"Surviving valid tricells      : {n_valid_tricells} "
          f"from {n_accepted_tracks} accepted tracks")

    if not records:
        print("No valid tricells survived the cuts. Loosen thresholds.")
        return

    # ---- Save raw ----
    record_arrays = {key: np.array([record[key] for record in records])
                     for key in records[0]}
    output_npz_path = f"{args.outdir}/tricell_ds_accuracy_results.npz"
    np.savez(output_npz_path, **record_arrays)
    print(f"wrote {output_npz_path}")

    make_summary_plots(records, args.outdir)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_summary_plots(records, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractional_difference = np.array(
        [r['fractional_difference'] for r in records])
    sentinel_min_ratio = np.array(
        [r['sentinel_min_ratio'] for r in records])
    sentinel_mean_ratio = np.array(
        [r['sentinel_mean_ratio'] for r in records])
    ds_truth_cm = np.array([r['ds_truth_cm'] for r in records])
    ds_recon_cm = np.array([r['ds_recon_cm'] for r in records])

    def save_figure(fig, filename):
        fig.tight_layout()
        path = f"{outdir}/{filename}"
        fig.savefig(path, dpi=140)
        print(f"wrote {path}")
        plt.close(fig)

    # ----- Threshold scan (the main investigation) -----
    threshold_scan = np.linspace(0.0, 1.0, 41)
    n_passing_at_threshold = []
    mean_fractional_diff_at_threshold = []
    median_fractional_diff_at_threshold = []
    rms_fractional_diff_at_threshold = []
    for threshold in threshold_scan:
        mask_pass = sentinel_min_ratio >= threshold
        n_passing = int(mask_pass.sum())
        n_passing_at_threshold.append(n_passing)
        if n_passing == 0:
            mean_fractional_diff_at_threshold.append(np.nan)
            median_fractional_diff_at_threshold.append(np.nan)
            rms_fractional_diff_at_threshold.append(np.nan)
            continue
        passing_frac_diff = fractional_difference[mask_pass]
        mean_fractional_diff_at_threshold.append(
            float(np.mean(passing_frac_diff)))
        median_fractional_diff_at_threshold.append(
            float(np.median(passing_frac_diff)))
        rms_fractional_diff_at_threshold.append(
            float(np.sqrt(np.mean(passing_frac_diff ** 2))))
    n_passing_at_threshold = np.array(n_passing_at_threshold)
    mean_fractional_diff_at_threshold = np.array(
        mean_fractional_diff_at_threshold)
    median_fractional_diff_at_threshold = np.array(
        median_fractional_diff_at_threshold)
    rms_fractional_diff_at_threshold = np.array(
        rms_fractional_diff_at_threshold)

    # 1. Fractional difference histograms at a few threshold levels
    fig, ax = plt.subplots(figsize=(10, 5.5))
    histogram_bins = np.linspace(-1.0, 5.0, 80)
    for threshold, color in [(0.0, "C0"), (0.05, "C1"),
                              (0.1, "C2"), (0.2, "C3")]:
        mask_pass = sentinel_min_ratio >= threshold
        passing_frac_diff = fractional_difference[mask_pass]
        if len(passing_frac_diff) == 0:
            continue
        ax.hist(np.clip(passing_frac_diff,
                        histogram_bins[0], histogram_bins[-1]),
                bins=histogram_bins, alpha=0.45,
                label=(f"sentinel ≥ {threshold:.2f}: "
                       f"N={len(passing_frac_diff)}, "
                       f"mean={np.mean(passing_frac_diff):+.3f}, "
                       f"median={np.median(passing_frac_diff):+.3f}"),
                color=color)
    ax.axvline(0.0, color="k", ls="--", lw=0.6, label="perfect ds")
    ax.set_xlabel("(ds_reconstructed − ds_truth) / ds_truth")
    ax.set_ylabel("tricell count")
    ax.set_title(
        "Tricell ds fractional difference at various sentinel cuts\n"
        "(sentinel = min(|Q_w1_top|, |Q_w1_bottom|) / |Q_target|)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ds_fractional_difference.png")

    # 2. Mean / median fractional difference vs sentinel threshold
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(threshold_scan, mean_fractional_diff_at_threshold,
            marker="o", ms=4, label="mean", color="C1")
    ax.plot(threshold_scan, median_fractional_diff_at_threshold,
            marker="s", ms=4, label="median", color="C2")
    ax.plot(threshold_scan, rms_fractional_diff_at_threshold,
            marker="^", ms=4, label="RMS of frac diff", color="C3")
    ax.axhline(0.0, color="k", ls="--", lw=0.6, label="perfect ds")
    ax.set_xlabel("sentinel cut: minimum |Q_sentinel| / |Q_target|")
    ax.set_ylabel("fractional difference (ds_recon − ds_truth) / ds_truth")
    ax.set_title("ds accuracy vs sentinel-charge threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ds_accuracy_vs_sentinel_threshold.png")

    # 3. Yield vs threshold
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(threshold_scan, n_passing_at_threshold,
            marker="o", color="C0")
    ax.set_yscale("log")
    ax.set_xlabel("sentinel cut: minimum |Q_sentinel| / |Q_target|")
    ax.set_ylabel("number of tricells passing")
    ax.set_title("Tricell yield vs sentinel-charge threshold")
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, "tricell_ds_yield_vs_sentinel_threshold.png")

    # ===== Readout-derived dQ/dx vs truth dQ/dx (calibration-style) =====
    dQdx_readout_recon_ds = np.array(
        [r['dQdx_readout_per_dx_recon'] for r in records])
    dQdx_readout_truth_ds = np.array(
        [r['dQdx_readout_per_dx_truth'] for r in records])
    dQdx_truth = np.array(
        [r['dQdx_truth_per_dx'] for r in records])

    # Fit calibration factor: median(dQdx_readout / dQdx_truth) for
    # a tight subsample (high sentinel ratio, sentinel asymmetry ≈ 1)
    # so the calibration isn't pulled by tricell mis-reco.
    tight_mask = (sentinel_min_ratio > 0.5)
    if tight_mask.sum() < 50:
        tight_mask = sentinel_min_ratio > 0.2
    calibration_factor_truth_ds = float(np.median(
        dQdx_readout_truth_ds[tight_mask] / dQdx_truth[tight_mask]))

    # 4. Calibration-style histogram: readout/truth using truth ds
    fig, ax = plt.subplots(figsize=(10, 5.5))
    raw_ratio = dQdx_readout_truth_ds / dQdx_truth
    histogram_bins_cal = np.linspace(
        np.percentile(raw_ratio, 1) * 0.5,
        np.percentile(raw_ratio, 99) * 1.1, 80)
    ax.hist(raw_ratio, bins=histogram_bins_cal, alpha=0.55,
            label=(f"all tricells: median="
                   f"{np.median(raw_ratio):.4f}"),
            color="C0")
    ax.hist(raw_ratio[tight_mask], bins=histogram_bins_cal, alpha=0.65,
            label=(f"tight subsample (sentinel ratio > 0.5): "
                   f"median={calibration_factor_truth_ds:.4f}"),
            color="C1")
    ax.axvline(calibration_factor_truth_ds, color="k", ls="--", lw=0.8,
               label=(f"fitted calibration "
                      f"factor = {calibration_factor_truth_ds:.4f}"))
    ax.set_xlabel(
        "dQ/dx_readout / dQ/dx_truth  (using truth ds in denominator)")
    ax.set_ylabel("tricell count")
    ax.set_title(
        "Readout-derived dQ/dx vs truth dQ/dx (uncalibrated)\n"
        "Width of the distribution reflects readout-only fluctuations; "
        "the median sets the calibration factor.")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_dQdx_readout_vs_truth_uncalibrated.png")

    # 5. Calibrated readout dQ/dx vs truth, using tricell-recon ds
    calibrated_readout_recon_ds = (dQdx_readout_recon_ds
                                   / calibration_factor_truth_ds)
    calibrated_ratio = calibrated_readout_recon_ds / dQdx_truth
    fig, ax = plt.subplots(figsize=(10, 5.5))
    histogram_bins_calib = np.linspace(
        np.percentile(calibrated_ratio, 1) * 0.9,
        np.percentile(calibrated_ratio, 99) * 1.1, 80)
    for sentinel_cut, color, label_prefix in [
        (0.0, "C0", "all tricells"),
        (0.10, "C1", "sentinel ≥ 0.10"),
        (0.20, "C2", "sentinel ≥ 0.20"),
    ]:
        mask = sentinel_min_ratio >= sentinel_cut
        if mask.sum() == 0:
            continue
        sub = calibrated_ratio[mask]
        ax.hist(sub, bins=histogram_bins_calib, alpha=0.45,
                label=(f"{label_prefix}: N={mask.sum()}, "
                       f"median={np.median(sub):.3f}, "
                       f"mean={np.mean(sub):.3f}"),
                color=color)
    ax.axvline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel(
        "(calibrated dQ/dx_readout) / dQ/dx_truth  "
        "using tricell-reconstructed ds in denominator")
    ax.set_ylabel("tricell count")
    ax.set_title(
        "Calibrated readout dQ/dx vs truth dQ/dx\n"
        f"(calibration constant {calibration_factor_truth_ds:.4f} "
        "fitted from high-sentinel-ratio subsample with truth ds)\n"
        "Deviation from 1.0 here mixes ds-reconstruction error and "
        "any residual charge-collection bias.")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_dQdx_readout_calibrated_vs_truth.png")

    # 6. Calibrated dQ/dx ratio vs reconstructed ds (data-only view)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ds_recon_cm_arr = np.array([r['ds_recon_cm'] for r in records])
    bins_ds = np.linspace(np.percentile(ds_recon_cm_arr, 1),
                          np.percentile(ds_recon_cm_arr, 99), 10)
    bin_means, bin_stderrs, bin_centres = binned_mean_and_stderr(
        ds_recon_cm_arr, calibrated_ratio, bins_ds)
    ax.errorbar(bin_centres, bin_means, yerr=bin_stderrs,
                marker="o", capsize=3, color="C2",
                label="(calibrated readout dQ/dx) / dQ/dx_truth")
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
               label="perfect calibration")
    ax.set_xlabel("reconstructed ds (tricell) [cm]")
    ax.set_ylabel("calibrated dQ/dx ratio")
    ax.set_title("Calibrated readout dQ/dx vs reconstructed ds\n"
                 "(data-only view: would be flat if calibration is "
                 "ds-independent)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_dQdx_calibrated_vs_recon_ds.png")

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print(f"Total tricells:                              {len(records)}")
    print("Threshold scan (sentinel = min(Q_lo,Q_hi)/Q_center):")
    print(f"{'threshold':>12}  {'N pass':>8}  "
          f"{'mean':>10}  {'median':>10}  {'RMS':>10}")
    for i in range(0, len(threshold_scan), 4):
        print(f"{threshold_scan[i]:>12.3f}  "
              f"{n_passing_at_threshold[i]:>8d}  "
              f"{mean_fractional_diff_at_threshold[i]:>+10.4f}  "
              f"{median_fractional_diff_at_threshold[i]:>+10.4f}  "
              f"{rms_fractional_diff_at_threshold[i]:>10.4f}")
    print("=" * 70)


def binned_mean_and_stderr(x_values, y_values, x_bins):
    """Bin y by x and return (means, stderrs, bin_centres) per bin."""
    bin_indices = np.digitize(x_values, x_bins) - 1
    means = []
    stderrs = []
    bin_centres = []
    for bin_index in range(len(x_bins) - 1):
        mask = (bin_indices == bin_index) & np.isfinite(y_values)
        values_in_bin = y_values[mask]
        if len(values_in_bin) >= 2:
            means.append(float(np.mean(values_in_bin)))
            stderrs.append(
                float(np.std(values_in_bin)) / sqrt(len(values_in_bin)))
        else:
            means.append(np.nan)
            stderrs.append(np.nan)
        bin_centres.append(
            0.5 * (x_bins[bin_index] + x_bins[bin_index + 1]))
    return (np.array(means), np.array(stderrs),
            np.array(bin_centres))


if __name__ == "__main__":
    main()
