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

The witness peak times come from the FEE ADC packet ticks, which are
in MICROSECONDS (run_fee_chain builds time_ticks = arange *
TIME_SAMPLING, matching production simulate_pixels.py). An earlier
version multiplied the witness Δt by TIME_SAMPLING a second time,
shrinking delta_drift 10× and collapsing ds for near-vertical
(drift-dominated) tracks; that is now fixed.

POST-FIX Δw RE-FIT (run 3a197bf data, N≈2270 tricells): with the drift
term restored, delta_w = 1·pitch gives a near-centered result —
median(ds_recon/ds_truth) = 1.019 (centroid). Re-fitting delta_w to
zero the median wants delta_w ≈ 0 (centroid) / 0.3·pitch (peak_rate),
NOT 2·pitch — so the naive pad-center-to-pad-center value is firmly
ruled out, and 1·pitch is retained (physically the inside-w_1 column
width, and within ~2% of optimal). The reason a single delta_w cannot
do better: the residual is ZENITH-STRUCTURED, not a constant offset —
median fractional difference runs +9% for near-vertical (drift-
dominated, |Δdrift|≈4 cm) tracks down to −1% for near-horizontal ones.
delta_w only matters when Δdrift is small (horizontal), so tuning it
trades the two ends against each other instead of removing the
structure. That structure is the mixed-coordinate-sourcing effect
(pad-center Δv,Δw vs FEE-timing Δdrift); the principled fix is to
source all three witness coordinates consistently — deferred.

The per-pixel timing estimator is selectable via
--primary-timing-method {centroid,peak_rate,first,largest,median};
all five are saved to the npz regardless, so the choice can be
revisited offline. 'peak_rate' is the charge-arrival-rate (inverse
inter-packet spacing) estimator — the data-portable form of "find the
maximum of the Δq profile".

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

WITNESS-QUALITY CUTS (staged; default OFF so each can be turned on one
at a time and its effect on bias/RMS/yield is attributable). Witness
quality is the dominant ds-reconstruction error source: the whole
track direction — hence Δdrift, hence ds — is built from the two
witnesses, so a witness that is actually an induced-fringe pad
corrupts everything. All three cuts are computable from FEE packets
alone, so they port verbatim to real data.

  - --minimum-witness-packets (default 1, off)
      CUT 1, halo-vs-track. A pad the track CROSSES collects the full
      deposited charge → several packets clustered in time → reliable
      timing. A pad seeing only a neighbour's induced positive lobe
      (within MAX_RADIUS=2) emits 0–1 packets. Requiring ≥ 2 (try 3)
      packets per witness, on top of the charge floor set above the
      induced-lobe scale, selects real-collection witnesses. Polarity
      is NOT used — the negative induced lobe never crosses threshold,
      so it is invisible to the readout. Validated in sim against the
      geometric ds-through-witness-pillar truth label (NOT polarity).
  - --maximum-direction-disagreement-deg (default 10, ON)
      CUT 2, direction cross-check (highest single value, the
      production default). The three w_1 pads give an INDEPENDENT
      (x, y, z=t·V_DRIFT) direction; the two witnesses give another.
      Require them to agree within a few degrees, else drop — a fringe
      witness with a bogus peak tick throws the two apart. Staged scan:
      ds fractional-difference RMS 0.149 → 0.094 at ≤10°. CUT 1 is left
      OFF because it costs ~25% yield for negligible additional RMS gain
      once CUT 2 is on. Set 180 to disable CUT 2.
  - --maximum-implied-drift-cm (default 30) and
    --implied-zenith-min-deg / --implied-zenith-max-deg (default 2/88)
      CUT 3, physical plausibility — a loose safety net. Reject
      reconstructions whose implied |Δdrift| exceeds the drift window or
      whose implied zenith falls outside a wide band — catches late
      induced-lobe timing outliers. On a clean sim sample it rejects ~0;
      it is kept on to guard real-data outliers. Set 0 / 0 / 90 to
      disable.

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
  tricell_ds_bias_vs_zenith.png
                            ds fractional difference (median, mean,
                            RMS) binned by track zenith, with per-bin
                            yield. Exposes the dominant residual: low-
                            zenith (drift-aligned) tracks over-estimate
                            ds, pad-plane-aligned tracks are ~unbiased.
                            Flattening this slope is the next
                            calibration target.
  tricell_witness_halo_validation.png
                            sim-only validation of CUT 1: witness FEE
                            charge and packet count split by the
                            geometric truth label (ds through the
                            witness pad pillar > 0 ⇒ track-crossing,
                            = 0 ⇒ induced-only). Confirms the reco-only
                            charge+packet cuts keep real-collection
                            witnesses and reject induced-only fringe.
  tricell_direction_disagreement.png
                            validation of CUT 2: angle between the
                            witness-to-witness and three-w_1-pad track
                            directions, split by witness truth label.
                            A clean core + rejectable (mostly induced-
                            only) tail confirms the cut works.
  tricell_calibration_constant_uncertainty.png
                            the money plot: per-tricell calibration ratio
                            r = (q_centre/ds) / dQdx_truth for three ds
                            sources overlaid — PRODUCTION (global line-fit
                            pitch, what NDLAr reconstruction uses today),
                            TRICELL (local 5-pixel ds), and TRUTH-DS
                            (SIM-ONLY ceiling). A narrower distribution =
                            a tighter calibration constant. Demonstrates
                            whether the local tricell ds beats the global
                            track fit; legend reports median, fractional
                            RMS, the constant's standard error, and the
                            track count needed for 1% precision.
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
        adc_packet_ticks: (N_unique, MAX_ADC_VALUES) packet times
                          in MICROSECONDS (time_ticks is built as
                          arange * TIME_SAMPLING, so these are already
                          scaled to µs — not integer tick indices).
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
def per_pixel_charge_centroid_time(packet_ticks, packet_charges):
    """Charge-weighted centroid of packet times for one pixel.

    Captures the moment when the bulk of induced charge is arriving
    on the pad — close to the substep-closest-approach time for that
    pad — instead of the leading-edge or peak-window time.
    """
    valid = packet_charges > 0
    if not np.any(valid):
        return None
    return float(np.sum(packet_ticks[valid] * packet_charges[valid])
                 / np.sum(packet_charges[valid]))


def per_pixel_50pct_time(packet_ticks, packet_charges):
    """Tick at which 50% of the total integrated charge has been
    accumulated on the pixel (median arrival time across the packet
    stream). Linear interpolation between the two bracketing packets.

    More robust than the centroid against a single outlier packet.
    """
    valid = packet_charges > 0
    if not np.any(valid):
        return None
    # Sort packets chronologically.
    sort_idx = np.argsort(packet_ticks[valid])
    t = packet_ticks[valid][sort_idx]
    q = packet_charges[valid][sort_idx]
    cumulative_charge = np.cumsum(q)
    half_charge = 0.5 * cumulative_charge[-1]
    idx = int(np.searchsorted(cumulative_charge, half_charge))
    if idx == 0:
        return float(t[0])
    if idx >= len(t):
        return float(t[-1])
    c_before = cumulative_charge[idx - 1]
    c_after = cumulative_charge[idx]
    interp_frac = ((half_charge - c_before)
                   / (c_after - c_before)
                   if c_after > c_before else 0.0)
    return float(t[idx - 1] + interp_frac * (t[idx] - t[idx - 1]))


def per_pixel_peak_charge_rate_time(packet_ticks, packet_charges):
    """Time of peak charge-arrival RATE on a pixel.

    Data-portable realisation of the user's "look for the maximum of
    the Δq profile" idea, corrected for how LArPix actually stores
    packets.

    Each FEE packet is (q_n, t_n). In fee.get_adc_values the
    accumulator `q_sum` is RESET after every packet (fee.py:707), so
    q_n is the charge integrated since the previous reset — already a
    per-packet increment (the "Δq"), NOT a running total. Two facts
    then make charge RATE the right discriminator:

      * q_n is larger when the instantaneous induced current during
        the fixed ~1.8 µs integration window is larger, i.e. q_n
        peaks near the track's closest approach to the pad;
      * packets are emitted closer together in time when charge
        arrives faster (the inter-packet gap is set by how quickly
        q_sum re-crosses threshold, on top of the ~2.8 µs reset+busy
        dead time).

    So the charge-arrival rate between consecutive packets,

        rate_n = q_n / (t_n - t_{n-1}),

    is doubly peaked at closest approach (numerator up, denominator
    down). We return the midpoint time of the maximum-rate interval.

    NOTE on resolution: packet times are quantised to >= ~2.8 µs by
    the reset+busy dead time, and a typical witness emits only a
    handful of packets. So this estimator can only resolve Δt down to
    the packet spacing — it does NOT manufacture sub-packet timing.
    For witnesses dominated by diffusion halo (near-vertical tracks)
    there is no sharp closest-approach current peak to find and this
    will track the other estimators; the cure there is a quality cut,
    not a cleverer single-tick estimator.

    Ticks are in µs (see run_fee_chain). Returns a time in µs, or the
    single packet's tick if only one packet, or None if no positive
    packet.
    """
    valid = packet_charges > 0
    if not np.any(valid):
        return None
    t = packet_ticks[valid]
    q = packet_charges[valid]
    order = np.argsort(t)
    t = t[order]
    q = q[order]
    if len(t) == 1:
        return float(t[0])
    inter_packet_dt = np.diff(t)
    # Guard against zero/negative spacing (packets sharing a tick).
    inter_packet_dt = np.where(inter_packet_dt > 0,
                               inter_packet_dt, np.inf)
    charge_rate = q[1:] / inter_packet_dt          # rate over interval n
    peak_interval = int(np.argmax(charge_rate))
    return float(0.5 * (t[peak_interval] + t[peak_interval + 1]))


PRIMARY_TIMING_METHODS = (
    "centroid", "peak_rate", "first", "largest", "median")


def build_hit_pixel_dict(neighboring_pixels, signals,
                         fee_output, id2pixel_fn,
                         minimum_pixel_charge_threshold,
                         primary_timing_method="centroid"):
    """Build a per-pixel hit dict from the FEE-readout packet stream.

    Every quantity used for downstream cuts is RECO-LEVEL: derivable
    from real-data packets alone. The kernel response sum is also
    saved as a diagnostic but is NOT used for any selection.

    `primary_timing_method` selects which packet-timing estimator
    becomes the per-pixel `peak_tick` that drives witness ordering and
    ds reconstruction. All five estimators are always saved per pixel,
    so the choice can be revisited offline from the npz without
    re-running the kernel. Options (see PRIMARY_TIMING_METHODS):
      centroid  : charge-weighted mean packet time (default — status
                  quo, isolates the Δt units-bug fix when compared to
                  prior runs).
      peak_rate : time of peak charge-arrival rate (per_pixel_peak_
                  charge_rate_time).
      first     : first nonzero packet tick.
      largest   : tick of the largest |packet|.
      median    : 50%-integrated-charge tick.

    Per-pixel fields:
      collected_charge        : sum of ADC packet charges (FEE readout;
                                 the data-applicable charge). Used for
                                 all halo-fringe and tricell cuts.
      collected_charge_raw    : sum(signals[ipix, :]); kernel response
                                 (diagnostic only — would not exist on
                                 real data).
      peak_tick               : the selected primary timing (µs).
      centroid_tick           : charge-weighted centroid (µs).
      peak_rate_tick          : peak charge-rate time (µs).
      tick_50pct              : 50%-charge time (µs).
      first_packet_tick       : first nonzero packet tick (µs).
      largest_packet_tick     : largest-packet tick (µs).
      n_packets               : count of nonzero packets.

    Pixels in the halo that produce zero FEE packets are skipped --
    in data we would not see them.
    """
    if primary_timing_method not in PRIMARY_TIMING_METHODS:
        raise ValueError(
            f"primary_timing_method must be one of "
            f"{PRIMARY_TIMING_METHODS}, got {primary_timing_method!r}")
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

        # Four candidate timing definitions, all derivable from the
        # data packet stream:
        #   - first_packet_tick    : time of the FIRST discriminator
        #     trigger (leading edge of integrated charge; dominated by
        #     earliest-arriving substep — collapses across witnesses
        #     for vertical tracks).
        #   - largest_packet_tick  : tick of the largest ADC packet
        #     (window with the most charge — depends on FEE
        #     integration timing; poor spatial reconstruction).
        #   - centroid_tick        : charge-weighted centroid over all
        #     packets. Tracks the bulk-charge arrival, which for a
        #     given pixel is dominated by the closest-approach
        #     substep. This is the primary `peak_tick`.
        #   - tick_50pct           : tick at which 50% of total charge
        #     has accumulated. More robust to outlier packets than
        #     the centroid.
        largest_packet_index = int(np.argmax(np.abs(packet_charges)))
        largest_packet_tick = float(packet_ticks[largest_packet_index])
        first_packet_tick = float(packet_ticks[
            np.argmax(nonzero_packet_mask)])
        centroid_tick = per_pixel_charge_centroid_time(
            packet_ticks, packet_charges)
        tick_50pct = per_pixel_50pct_time(
            packet_ticks, packet_charges)
        peak_rate_tick = per_pixel_peak_charge_rate_time(
            packet_ticks, packet_charges)

        # Select the primary timing per the requested method, with a
        # graceful fallback to first_packet_tick if the estimator
        # returns None (e.g. zero-charge edge case).
        timing_candidates = dict(
            centroid=centroid_tick,
            peak_rate=peak_rate_tick,
            first=first_packet_tick,
            largest=largest_packet_tick,
            median=tick_50pct,
        )
        peak_tick = timing_candidates[primary_timing_method]
        if peak_tick is None:
            peak_tick = first_packet_tick

        hit_dict[(int(i_x), int(i_y))] = dict(
            halo_index=halo_index,
            unique_pix_idx=unique_idx,
            collected_charge=collected_charge_readout,       # cuts use this
            collected_charge_raw=collected_charge_raw,        # diagnostic
            peak_tick=peak_tick,                             # primary ts
            centroid_tick=centroid_tick,                     # diagnostic
            peak_rate_tick=peak_rate_tick,                   # diagnostic
            tick_50pct=tick_50pct,                           # diagnostic
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
                    w0_witness_collected_charge_raw=(
                        witness_w0['record']['collected_charge_raw']),
                    w0_witness_n_packets=(
                        witness_w0['record']['n_packets']),
                    w2_witness_key=witness_w2['key'],
                    w2_witness_collected_charge=(
                        witness_w2['record']['collected_charge']),
                    w2_witness_collected_charge_raw=(
                        witness_w2['record']['collected_charge_raw']),
                    w2_witness_n_packets=(
                        witness_w2['record']['n_packets']),
                    # Primary timing per pixel (used by ds-recon below)
                    peak_tick_w0_witness=witness_w0['peak_tick'],
                    peak_tick_w1_lo=lo_record['peak_tick'],
                    peak_tick_w1_centre=centre_record['peak_tick'],
                    peak_tick_w1_hi=hi_record['peak_tick'],
                    peak_tick_w2_witness=witness_w2['peak_tick'],
                    # Alternative per-pixel timings for offline reproc
                    w0_witness_centroid_tick=(
                        witness_w0['record']['centroid_tick']),
                    w2_witness_centroid_tick=(
                        witness_w2['record']['centroid_tick']),
                    w0_witness_peak_rate_tick=(
                        witness_w0['record']['peak_rate_tick']),
                    w2_witness_peak_rate_tick=(
                        witness_w2['record']['peak_rate_tick']),
                    w0_witness_50pct_tick=(
                        witness_w0['record']['tick_50pct']),
                    w2_witness_50pct_tick=(
                        witness_w2['record']['tick_50pct']),
                    w0_witness_first_packet_tick=(
                        witness_w0['record']['first_packet_tick']),
                    w2_witness_first_packet_tick=(
                        witness_w2['record']['first_packet_tick']),
                    w0_witness_largest_packet_tick=(
                        witness_w0['record']['largest_packet_tick']),
                    w2_witness_largest_packet_tick=(
                        witness_w2['record']['largest_packet_tick']),
                )


def reconstruct_ds_witness_to_witness(tricell, detector):
    """Direction estimate from witness peak-time deltas.

      delta_w     = 1 * pixel_pitch
                     (the EFFECTIVE w-extent between witness peak
                     times. Empirically this is what the data wanted,
                     not 2 * pitch as a naive pad-center-to-pad-center
                     argument would suggest. The witnesses' peak times
                     correspond to the track's closest approach to
                     each witness pad, and for tricells the closest
                     approach to a witness pad lies near the column
                     boundary rather than at the pad center, giving
                     an effective Δw of one pitch — the width of the
                     w_1 column the track actually traverses.

                     POST-UNITS-FIX RE-FIT (3a197bf data): with the
                     drift term restored, 1·pitch gives median
                     ds_recon/ds_truth = 1.019. Re-fitting to zero the
                     median wants delta_w ≈ 0–0.3·pitch, NOT 2·pitch —
                     pad-center-to-pad-center is ruled out. 1·pitch is
                     kept (within ~2% of optimal, physical column
                     width). The residual is zenith-structured (+9%
                     vertical → −1% horizontal), which no constant
                     delta_w can remove; see module docstring.)
      delta_v     = v_center_w2 − v_center_w0
      delta_drift = V_DRIFT * (t_w2_peak − t_w0_peak)
      L = sqrt(delta_w² + delta_v² + delta_drift²)
      ds_middle_pixel = L * pixel_pitch / |delta_v|

    UNITS: the per-pixel peak ticks come from `adc_packet_ticks`,
    which are already in MICROSECONDS (run_fee_chain builds
    `time_ticks = arange * TIME_SAMPLING`, mirroring production
    cli/simulate_pixels.py:210). So the witness time difference is
    taken directly in µs — NO extra * TIME_SAMPLING. An earlier
    version multiplied by TIME_SAMPLING here, double-converting and
    making delta_drift 10× too small; that suppressed the drift term
    and collapsed ds for near-vertical (drift-dominated) tricells.

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

    # peak ticks are already in µs — take the difference directly.
    delta_t_us = (tricell['peak_tick_w2_witness']
                  - tricell['peak_tick_w0_witness'])
    delta_drift_cm = v_drift * delta_t_us

    if traversal_axis == 'y':
        delta_v_cm = y_w2 - y_w0
    else:
        delta_v_cm = x_w2 - x_w0

    # See docstring: effective Δw is 1·pitch (inside-w_1 column width),
    # not 2·pitch (pad center-to-center). Post-units-fix data (3a197bf)
    # gives median ds_recon/ds_truth = 1.019 here; a re-fit prefers
    # ≈0–0.3·pitch, ruling out 2·pitch. Residual bias is zenith-
    # structured, not removable by any constant Δw.
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


def fit_global_track_direction(hit_pixel_dict, detector):
    """Global 3D line fit over ALL the track's hit pixels (production ds).

    This emulates the path-length (dx) that NDLAr production
    reconstruction assigns to a pixel today, so the local 5-pixel
    tricell ds can be compared against the method actually in use rather
    than against a naive baseline.

    METHOD (standard LArTPC calorimetric dQ/dx). Reconstruct the global
    track direction, then take the per-pixel path length as the pad
    pitch projected onto that direction: dx = pitch / |cos theta|, with
    cos theta the direction cosine along the traversal (v) axis. Here
    the direction is obtained from a principal-axis (PCA) line fit to
    the 3D hit cloud: for every hit pixel (i_x, i_y) build a point
        x, y = pixel_indices_to_center(detector, i_x, i_y)
        z     = peak_tick * V_DRIFT            (drift coordinate, cm)
    mean-center the cloud and take the principal right-singular vector
    (largest singular value) as the unit track direction. This is the
    "tracklet"-style global line fit: a single straight direction shared
    by the whole track, in contrast to the tricell's local 5-pixel
    estimate.

    PROVENANCE / REFERENCE (why this is "the production method"). The
    global-line-fit-pitch dQ/dx is the standard LArTPC calorimetry
    recipe: fit the track direction, then project the readout pitch onto
    it. For ND-LAr / Module-0 / the 2x2 prototype the reference
    implementation is the DUNE pixel-readout reconstruction chain, whose
    "tracklet" stage fits a principal-axis (PCA) line to the 3D hits and
    assigns dQ/dx along it:
      - DUNE ndlar_flow reconstruction chain (ND-LAr / 2x2):
            https://github.com/DUNE/ndlar_flow
      - Module-0 predecessor (module0_flow):
            https://github.com/peter-madigan/module0_flow
      - Pixelated-LArTPC cosmic dQ/dx via a fitted track (line fit +
        pitch-along-track):                     arXiv:2512.10830
      - Standard LArTPC charge/dE-dx-per-length calibration (pitch
        projected on the reconstructed track):  arXiv:1907.11736
                                                 (ArgoNeuT)
    NOTE: the exact ndlar_flow source file/line for the tracklet fit was
    not pinned down at authoring time (repo browse needed auth); the
    citation is the framework + repos + methodology papers, not a single
    line. Confirm the precise tracklet-reco path on the GPU node (which
    has the repo checked out) and tighten this citation if available.

    Unweighted PCA is used as the baseline (every hit pixel counts
    equally); charge-weighting the fit is a noted optional refinement.
    The fit faithfully includes induced-fringe hit pixels that passed
    the build_hit_pixel_dict charge floor, exactly as production would.

    Returns the unit direction [dir_x, dir_y, dir_drift] (np.ndarray),
    or None if fewer than 3 hit pixels or the fit is degenerate.
    """
    v_drift = detector.V_DRIFT
    points = []
    for (i_x, i_y), record in hit_pixel_dict.items():
        x_cm, y_cm = pixel_indices_to_center(detector, i_x, i_y)
        z_cm = record['peak_tick'] * v_drift
        points.append((x_cm, y_cm, z_cm))

    if len(points) < 3:
        return None

    point_cloud = np.asarray(points, dtype=float)
    centered = point_cloud - point_cloud.mean(axis=0)
    if not np.all(np.isfinite(centered)):
        return None
    try:
        _, singular_values, right_vectors = np.linalg.svd(
            centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size == 0 or singular_values[0] <= 1e-9:
        return None

    direction = right_vectors[0]
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return direction / norm


def witness_direction_cross_check_deg(tricell, detector):
    """Independent drift-vs-v slope estimates, compared for agreement.

    The track's slope in the (v, drift) plane — how far it drifts per
    unit of travel along the traversal axis v — is measured two ways,
    both from data-derivable pad v-coordinates and FEE peak ticks
    (drift = t_peak · V_DRIFT):

      * witnesses : Δv = v(w_2 witness) − v(w_0 witness),
                    Δdrift = V_DRIFT · (t_w2 − t_w0)
      * three w_1 pads : Δv = v(w_1 hi) − v(w_1 lo) = 2·pitch,
                         Δdrift = V_DRIFT · (t_hi − t_lo)

    For a straight track these slopes are equal, so the two 2-vectors
    (Δv, Δdrift) are parallel. A large angle flags a corrupted
    witness — e.g. an induced-fringe pad whose peak tick does not mark
    a real closest-approach — and is the single most powerful
    witness-quality discriminant.

    WHY (v, drift) AND NOT FULL 3D: the three w_1 pads all sit in the
    SAME w-column, so their w-coordinate is identical and their
    direction carries NO w-information (Δw ≡ 0). The witness-to-witness
    vector, by contrast, spans 2·pitch in w. Comparing full 3D vectors
    would therefore register a large, geometry-driven disagreement on
    EVERY tricell. The w-baseline is supplied solely by the witnesses
    and cannot be cross-checked here; the genuinely independent,
    comparable quantity is the drift-vs-v slope.

    Sign-agnostic (|cos|): the track may be traversed in either time
    order, so anti-parallel is treated as agreement.

    Returns the disagreement angle in degrees, or None if either slope
    vector is degenerate (zero length).
    """
    v_drift = detector.V_DRIFT
    traversal_axis = tricell['traversal_axis']

    def v_coordinate(pixel_key):
        i_x, i_y = pixel_key
        x_cm, y_cm = pixel_indices_to_center(detector, i_x, i_y)
        return y_cm if traversal_axis == 'y' else x_cm

    delta_v_witness = (v_coordinate(tricell['w2_witness_key'])
                       - v_coordinate(tricell['w0_witness_key']))
    delta_drift_witness = v_drift * (tricell['peak_tick_w2_witness']
                                     - tricell['peak_tick_w0_witness'])
    witness_slope_vec = np.array(
        [delta_v_witness, delta_drift_witness], dtype=float)

    delta_v_w1 = (v_coordinate(tricell['w1_hi_key'])
                  - v_coordinate(tricell['w1_lo_key']))
    delta_drift_w1 = v_drift * (tricell['peak_tick_w1_hi']
                                - tricell['peak_tick_w1_lo'])
    w1_slope_vec = np.array([delta_v_w1, delta_drift_w1], dtype=float)

    norm_witness = float(np.linalg.norm(witness_slope_vec))
    norm_w1 = float(np.linalg.norm(w1_slope_vec))
    if norm_witness == 0.0 or norm_w1 == 0.0:
        return None
    cos_angle = abs(float(np.dot(witness_slope_vec, w1_slope_vec))
                    / (norm_witness * norm_w1))
    cos_angle = min(1.0, max(0.0, cos_angle))
    return float(np.degrees(acos(cos_angle)))


def implied_geometry_from_recon(recon_info):
    """Data-derivable plausibility quantities of the reconstructed
    direction:

      implied_drift_cm   : |Δdrift| between the two witnesses (cm).
      implied_zenith_deg : angle of the witness-to-witness direction
                           from the drift axis, acos(|Δdrift| / L).
                           In this script's convention the drift axis
                           is the zenith axis, so this is the implied
                           track zenith.

    A near-vertical (drift-dominated) reconstruction gives a small
    zenith; a near-horizontal one gives ~90°. Returns None if the
    witness-to-witness length is zero.
    """
    length_cm = recon_info['length_witness_to_witness_cm']
    implied_drift_cm = abs(recon_info['delta_drift_cm'])
    if length_cm == 0.0:
        return None
    cos_zenith = min(1.0, implied_drift_cm / length_cm)
    return dict(
        implied_drift_cm=implied_drift_cm,
        implied_zenith_deg=float(np.degrees(acos(cos_zenith))),
    )


def load_records_from_npz(path):
    """Reload per-tricell records from a saved results npz.

    np.savez stored one parallel array per record field (see the save in
    main()); this reverses that into the list-of-dicts shape the summary
    helpers expect. Each value comes back as a numpy scalar, which the
    calibration helpers handle via float(...). Enables the offline
    --from-npz re-analysis path with no CUDA.
    """
    data = np.load(path, allow_pickle=True)
    field_names = list(data.files)
    columns = [data[name] for name in field_names]
    return [dict(zip(field_names, row)) for row in zip(*columns)]


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
    # --- Witness-quality cuts (staged; default OFF/permissive so each
    #     can be enabled one at a time and its effect on bias, RMS, and
    #     yield is individually attributable). All three are computed
    #     from FEE packets alone, so they port verbatim to real data.
    arg_parser.add_argument(
        "--minimum-witness-packets", type=int, default=1,
        help="CUT 1 (halo-vs-track): minimum number of FEE ADC packets "
             "required on EACH witness pixel. A pad the track actually "
             "crosses collects the full deposited charge → several "
             "packets clustered in time; a pad seeing only a neighbour's "
             "induced positive lobe emits 0–1 (often sub-threshold) "
             "packets. Combined with --minimum-witness-charge-threshold "
             "(set above the induced-lobe scale) this selects pads the "
             "track's charge truly reached. Default 1 (OFF by "
             "recommendation: the staged scan showed it costs ~25%% yield "
             "for negligible ds-RMS gain, since CUT 2 already removes the "
             "fringe-witness tail). Try 2, then 3 if a sample needs it.")
    arg_parser.add_argument(
        "--maximum-direction-disagreement-deg", type=float, default=10.0,
        help="CUT 2 (direction cross-check): reject a tricell if the "
             "witness-to-witness direction and the independent "
             "three-w_1-pad direction disagree by more than this angle "
             "(degrees, sign-agnostic). The single most powerful "
             "witness-quality discriminant — a fringe witness with a "
             "bogus peak tick throws the two directions apart. Default "
             "10 (ON — the production default; staged scan: ds-RMS "
             "0.149→0.094 at ≤10°). Set 180 to disable; 8 is tighter.")
    arg_parser.add_argument(
        "--maximum-implied-drift-cm", type=float, default=30.0,
        help="CUT 3a (plausibility): reject a tricell if the implied "
             "|Δdrift| between witnesses exceeds this (cm). Catches "
             "timing outliers (late induced-lobe peaks) that imply a "
             "drift longer than physically possible. Default 30 (≈ the "
             "drift-window length — a loose safety net; rejects ~0 on a "
             "clean sim sample but guards real-data outliers). Set 0 to "
             "disable.")
    arg_parser.add_argument(
        "--implied-zenith-min-deg", type=float, default=2.0,
        help="CUT 3b (plausibility): reject a tricell whose implied "
             "zenith (angle of the reconstructed direction from the "
             "drift axis) is below this. Default 2 (wide safety net). "
             "Set 0 to disable.")
    arg_parser.add_argument(
        "--implied-zenith-max-deg", type=float, default=88.0,
        help="CUT 3b (plausibility): reject a tricell whose implied "
             "zenith exceeds this. Default 88 (wide safety net; the "
             "generator draws zenith in [10°, 80°]). Set 90 to disable. "
             "A tighter band like [5, 85] rejects more unphysical "
             "near-horizontal/near-vertical reconstructions.")
    arg_parser.add_argument(
        "--primary-timing-method", default="centroid",
        choices=PRIMARY_TIMING_METHODS,
        help="which packet-timing estimator drives witness ordering "
             "and ds reconstruction. Default 'centroid' (status quo, so "
             "the Δt units-bug fix is isolated when comparing to prior "
             "runs). Use 'peak_rate' to try the charge-arrival-rate "
             "estimator. ALL methods are always saved in the npz, so "
             "this can also be re-evaluated offline without re-running.")
    arg_parser.add_argument(
        "--master-rng-seed", type=int, default=20260519)
    arg_parser.add_argument(
        "--outdir", default=".")
    arg_parser.add_argument(
        "--from-npz", default=None,
        help="Offline re-analysis: skip the GPU simulation, load saved "
             "per-tricell records from this tricell_ds_accuracy_results.npz "
             "and regenerate the summary table + plots. Works on a machine "
             "with no CUDA because ds_production_cm (the global line-fit "
             "pitch) is saved per tricell. Lets you iterate on the "
             "calibration plot/table without re-running the simulation.")
    arg_parser.add_argument("--verbose", action="store_true")
    args = arg_parser.parse_args()

    # Offline path: rebuild the table/plots straight from a saved npz,
    # no CUDA needed. Branch BEFORE importing numba.cuda so it runs on a
    # GPU-less dev box.
    if args.from_npz is not None:
        records = load_records_from_npz(args.from_npz)
        print(f"loaded {len(records)} records from {args.from_npz}")
        make_summary_plots(records, args.outdir)
        return

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
    print(f"Primary per-pixel timing method: "
          f"{args.primary_timing_method} "
          f"(all methods saved to npz)")

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
        witness_too_few_packets=0,
        direction_disagreement=0,
        implied_drift_too_large=0,
        implied_zenith_out_of_band=0,
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
                    args.minimum_pixel_charge_threshold,
                    primary_timing_method=args.primary_timing_method)
                tricells_found = list(find_zshape_tricells(
                    hit_pixel_dict,
                    args.minimum_witness_charge_threshold))

                # PRODUCTION ds (global line-fit pitch): fit ONE 3D line
                # through ALL the track's hit pixels. This is the NDLAr
                # production-style direction (see fit_global_track_direction
                # for provenance). It needs every hit pixel — which are NOT
                # in the per-tricell records — so it MUST be computed here
                # and saved per tricell; it cannot be recovered offline.
                global_track_direction = fit_global_track_direction(
                    hit_pixel_dict, detector)
                n_hits_in_global_fit = len(hit_pixel_dict)

                for tricell in tricells_found:
                    n_tricells_examined += 1

                    # CUT 1 (halo-vs-track): require enough FEE packets
                    # on each witness. A track-crossing pad collects
                    # the full deposited charge (several clustered
                    # packets); an induced-fringe pad emits 0–1. Both
                    # witness charges already passed the charge floor
                    # in find_zshape_tricells.
                    if (tricell['w0_witness_n_packets']
                            < args.minimum_witness_packets
                            or tricell['w2_witness_n_packets']
                            < args.minimum_witness_packets):
                        cut_rejection_counts['witness_too_few_packets'] += 1
                        continue

                    recon_info = reconstruct_ds_witness_to_witness(
                        tricell, detector)
                    if recon_info is None:
                        cut_rejection_counts['ds_recon_failed'] += 1
                        continue
                    ds_recon_cm = recon_info['ds_recon_cm']

                    # CUT 2 (direction cross-check): the witness-to-
                    # witness direction and the independent three-w_1-
                    # pad direction must agree within tolerance.
                    direction_disagreement_deg = (
                        witness_direction_cross_check_deg(
                            tricell, detector))
                    if (direction_disagreement_deg is not None
                            and direction_disagreement_deg
                            > args.maximum_direction_disagreement_deg):
                        cut_rejection_counts['direction_disagreement'] += 1
                        continue

                    # CUT 3 (plausibility): implied drift / zenith of
                    # the reconstructed direction must be physical.
                    implied_geom = implied_geometry_from_recon(recon_info)
                    implied_drift_cm = (
                        implied_geom['implied_drift_cm']
                        if implied_geom is not None else np.nan)
                    implied_zenith_deg = (
                        implied_geom['implied_zenith_deg']
                        if implied_geom is not None else np.nan)
                    if implied_geom is not None:
                        if (args.maximum_implied_drift_cm > 0.0
                                and implied_drift_cm
                                > args.maximum_implied_drift_cm):
                            cut_rejection_counts[
                                'implied_drift_too_large'] += 1
                            continue
                        if (implied_zenith_deg < args.implied_zenith_min_deg
                                or implied_zenith_deg
                                > args.implied_zenith_max_deg):
                            cut_rejection_counts[
                                'implied_zenith_out_of_band'] += 1
                            continue

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

                    # Geometric ds through each WITNESS pad pillar — the
                    # polarity-agnostic truth label for the halo-vs-
                    # track validation plot. ds > 0 ⇒ the muon segment
                    # actually crosses that witness pad (real
                    # collection); ds = 0 ⇒ induced-only / diffusion-
                    # edge witness. SIM-VALIDATION ONLY — never used as
                    # a cut (a real analysis has no access to it).
                    i_x_w0_wit, i_y_w0_wit = tricell['w0_witness_key']
                    i_x_w2_wit, i_y_w2_wit = tricell['w2_witness_key']
                    x_w0_wit_cm, y_w0_wit_cm = pixel_indices_to_center(
                        detector, i_x_w0_wit, i_y_w0_wit)
                    x_w2_wit_cm, y_w2_wit_cm = pixel_indices_to_center(
                        detector, i_x_w2_wit, i_y_w2_wit)
                    ds_truth_w0_witness_cm = (
                        segment_length_through_pixel_pillar(
                            (start_x, start_y, start_z),
                            (end_x, end_y, end_z),
                            x_w0_wit_cm - pitch_cm / 2,
                            x_w0_wit_cm + pitch_cm / 2,
                            y_w0_wit_cm - pitch_cm / 2,
                            y_w0_wit_cm + pitch_cm / 2))
                    ds_truth_w2_witness_cm = (
                        segment_length_through_pixel_pillar(
                            (start_x, start_y, start_z),
                            (end_x, end_y, end_z),
                            x_w2_wit_cm - pitch_cm / 2,
                            x_w2_wit_cm + pitch_cm / 2,
                            y_w2_wit_cm - pitch_cm / 2,
                            y_w2_wit_cm + pitch_cm / 2))

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

                    # PRODUCTION ds for THIS pixel: same pitch/projection
                    # formula as the tricell (apples-to-apples), but the
                    # direction comes from the GLOBAL line fit over all
                    # the track's hits instead of the local 5-pixel
                    # witnesses. ds = pitch / |dir_v|, with dir_v the
                    # direction cosine along the traversal (v) axis.
                    ds_production_cm = float('nan')
                    global_dir_v_component = float('nan')
                    if global_track_direction is not None:
                        v_index = (0 if tricell['traversal_axis'] == 'x'
                                   else 1)
                        global_dir_v_component = abs(
                            global_track_direction[v_index])
                        if global_dir_v_component > 1e-6:
                            ds_production_cm = (
                                pitch_cm / global_dir_v_component)

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
                        # --- Witness-quality diagnostics (CUT 1/2/3) ---
                        w0_witness_collected_charge=abs(
                            tricell['w0_witness_collected_charge']),
                        w2_witness_collected_charge=abs(
                            tricell['w2_witness_collected_charge']),
                        w0_witness_collected_charge_raw=(
                            tricell['w0_witness_collected_charge_raw']),
                        w2_witness_collected_charge_raw=(
                            tricell['w2_witness_collected_charge_raw']),
                        w0_witness_n_packets=(
                            tricell['w0_witness_n_packets']),
                        w2_witness_n_packets=(
                            tricell['w2_witness_n_packets']),
                        # Geometric truth labels for halo-vs-track
                        # validation (SIM-ONLY — never used as a cut).
                        ds_truth_w0_witness_cm=ds_truth_w0_witness_cm,
                        ds_truth_w2_witness_cm=ds_truth_w2_witness_cm,
                        # CUT 2 / CUT 3 reconstructed-direction quantities
                        direction_disagreement_deg=(
                            direction_disagreement_deg
                            if direction_disagreement_deg is not None
                            else np.nan),
                        implied_drift_cm=implied_drift_cm,
                        implied_zenith_deg=implied_zenith_deg,
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
                        # PRODUCTION ds (global line-fit pitch) — the
                        # NDLAr-production baseline the tricell ds is
                        # measured against in the calibration plot. NaN
                        # if the global fit failed (<3 hits / degenerate)
                        # or the track runs ⟂ to the traversal axis.
                        ds_production_cm=ds_production_cm,
                        n_hits_in_global_fit=n_hits_in_global_fit,
                        global_dir_v_component=global_dir_v_component,
                        # Alternative witness timings — save for offline
                        # reprocessing (recompute ds with different
                        # timing extractors and compare).
                        w0_centroid_tick=tricell[
                            'w0_witness_centroid_tick'],
                        w2_centroid_tick=tricell[
                            'w2_witness_centroid_tick'],
                        w0_peak_rate_tick=tricell[
                            'w0_witness_peak_rate_tick'],
                        w2_peak_rate_tick=tricell[
                            'w2_witness_peak_rate_tick'],
                        w0_50pct_tick=tricell[
                            'w0_witness_50pct_tick'],
                        w2_50pct_tick=tricell[
                            'w2_witness_50pct_tick'],
                        w0_first_tick=tricell[
                            'w0_witness_first_packet_tick'],
                        w2_first_tick=tricell[
                            'w2_witness_first_packet_tick'],
                        w0_largest_tick=tricell[
                            'w0_witness_largest_packet_tick'],
                        w2_largest_tick=tricell[
                            'w2_witness_largest_packet_tick'],
                    ))
                    n_valid_tricells += 1

            if (track_index + 1) % 50 == 0 or args.verbose:
                print(f"  L={track_length_cm:.1f}cm, "
                      f"track {track_index + 1}/{args.n_tracks}: "
                      f"{n_valid_tricells} valid tricells so far")

    print(f"\nTricells examined            : {n_tricells_examined}")
    print(f"  rejected by witness packets < {args.minimum_witness_packets} "
          f": {cut_rejection_counts['witness_too_few_packets']}")
    print(f"  rejected by ds_recon failure   : "
          f"{cut_rejection_counts['ds_recon_failed']}")
    print(f"  rejected by direction disagree > "
          f"{args.maximum_direction_disagreement_deg}° "
          f": {cut_rejection_counts['direction_disagreement']}")
    print(f"  rejected by implied drift > {args.maximum_implied_drift_cm} cm "
          f": {cut_rejection_counts['implied_drift_too_large']}")
    print(f"  rejected by implied zenith ∉ "
          f"[{args.implied_zenith_min_deg}, {args.implied_zenith_max_deg}]° "
          f": {cut_rejection_counts['implied_zenith_out_of_band']}")
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

    make_summary_plots(
        records, args.outdir,
        witness_charge_floor=args.minimum_witness_charge_threshold,
        witness_packet_floor=args.minimum_witness_packets,
        direction_disagreement_cut_deg=(
            args.maximum_direction_disagreement_deg),
        pitch_cm=pitch_cm)


# ---------------------------------------------------------------------------
# Calibration-constant uncertainty (production line-fit ds vs tricell ds)
# ---------------------------------------------------------------------------
# Method labels and the ds field each draws its per-tricell path length
# from. "production" = global line-fit pitch (NDLAr-production style);
# "tricell" = local 5-pixel ds; "truth" = SIM-ONLY ceiling.
CALIBRATION_METHOD_DS_FIELD = dict(
    production='ds_production_cm',
    tricell='ds_recon_cm',
    truth='ds_truth_cm')
CALIBRATION_METHOD_LABEL = dict(
    production='production (line fit)',
    tricell='tricell (5-pixel)',
    truth='truth-ds (SIM-ONLY)')
CALIBRATION_METHOD_ORDER = ['production', 'tricell', 'truth']


def _nan_calibration_summary(ratio_array=None, n=0):
    return dict(
        median_ratio=float('nan'),
        fractional_rms=float('nan'),
        standard_error=float('nan'),
        tracks_for_one_percent=float('nan'),
        ratio_array=(ratio_array if ratio_array is not None
                     else np.array([])),
        N=int(n))


def compute_calibration_ratio_summary(records):
    """Per-method calibration-constant precision from saved records alone.

    For each ds method we form the per-tricell calibration ratio
        r = (q_centre / ds) / dQdx_truth_per_dx
    i.e. how the readout dQ/dx built with that method's ds compares to
    the truth dQ/dx. A calibration constant is the sample statistic of
    r across many tricells; its uncertainty is the spread of r. Per
    method we report:
        median_ratio           = median(r)                  (the constant)
        fractional_rms         = RMS(r - median) / median    (rel. spread)
        standard_error         = fractional_rms / sqrt(N)    (precision)
        tracks_for_one_percent = (fractional_rms / 0.01) ** 2
    A SMALLER fractional_rms ⇒ a TIGHTER calibration constant.

    Uses only saved record fields (q_centre, dQdx_truth_per_dx, and each
    method's ds), so it runs identically on GPU records and on
    npz-reloaded records — no CUDA, no matplotlib.

    Methods compared:
      production : ds_production_cm (global line-fit pitch; data-style)
      tricell    : ds_recon_cm      (local 5-pixel ds;   data-style)
      truth      : ds_truth_cm      (SIM-ONLY ceiling; never a method)

    Guards: a method is reported as NaN if its ds field is absent, if
    fewer than 2 tricells have a usable (|ds|>1e-6, finite q and dQdx,
    dQdx>0) ratio, or if the resulting median ratio is non-positive.
    """
    summary = {}
    if not records:
        for method in CALIBRATION_METHOD_ORDER:
            summary[method] = _nan_calibration_summary()
        return summary

    q_centre = np.array(
        [float(r['q_centre']) for r in records], dtype=float)
    dQdx_truth = np.array(
        [float(r['dQdx_truth_per_dx']) for r in records], dtype=float)

    for method in CALIBRATION_METHOD_ORDER:
        ds_field = CALIBRATION_METHOD_DS_FIELD[method]
        if ds_field not in records[0]:
            summary[method] = _nan_calibration_summary()
            continue
        ds_values = np.array(
            [float(r[ds_field]) for r in records], dtype=float)
        usable = (np.isfinite(ds_values) & (np.abs(ds_values) > 1e-6)
                  & np.isfinite(q_centre) & np.isfinite(dQdx_truth)
                  & (dQdx_truth > 0))
        ratio = (q_centre[usable] / ds_values[usable]) / dQdx_truth[usable]
        ratio = ratio[np.isfinite(ratio)]
        n = int(ratio.size)
        if n < 2:
            summary[method] = _nan_calibration_summary(ratio, n)
            continue
        median_ratio = float(np.median(ratio))
        if not np.isfinite(median_ratio) or median_ratio <= 0:
            summary[method] = _nan_calibration_summary(ratio, n)
            continue
        fractional_rms = float(
            np.sqrt(np.mean((ratio - median_ratio) ** 2)) / median_ratio)
        summary[method] = dict(
            median_ratio=median_ratio,
            fractional_rms=fractional_rms,
            standard_error=fractional_rms / sqrt(n),
            tracks_for_one_percent=(fractional_rms / 0.01) ** 2,
            ratio_array=ratio,
            N=n)
    return summary


def print_calibration_table(summary):
    """Print the calibration-constant uncertainty table (no matplotlib).

    Emitted straight from the summary dict so it shows up in the run log
    even on a machine without a plotting backend. The headline compares
    the production (global line-fit) and tricell (local 5-pixel)
    fractional RMS: the smaller wins.
    """
    print("")
    print("CALIBRATION-CONSTANT UNCERTAINTY  (r = (q/ds)/dQdx_truth)")
    print(f"{'method':<24}{'median(r)':>11}{'fracRMS':>9}"
          f"{'sigma_const':>13}{'N_for_1%':>11}")
    for method in CALIBRATION_METHOD_ORDER:
        s = summary.get(method, _nan_calibration_summary())
        print(f"{CALIBRATION_METHOD_LABEL[method]:<24}"
              f"{s['median_ratio']:>11.4f}{s['fractional_rms']:>9.3f}"
              f"{s['standard_error']:>13.4f}"
              f"{s['tracks_for_one_percent']:>11.0f}"
              f"   (N={s['N']})")

    production = summary.get('production', _nan_calibration_summary())
    tricell = summary.get('tricell', _nan_calibration_summary())
    production_rms = production['fractional_rms']
    tricell_rms = tricell['fractional_rms']
    if (np.isfinite(production_rms) and np.isfinite(tricell_rms)
            and production_rms > 0):
        print(f"=> tricell fracRMS / production fracRMS = "
              f"{tricell_rms / production_rms:.2f}")
        print(f"   tricell needs {tricell['tracks_for_one_percent']:.0f} "
              f"tracks for 1% vs "
              f"{production['tracks_for_one_percent']:.0f} for production.")
        if tricell_rms < production_rms:
            print("   => TRICELL gives the tighter calibration constant "
                  "(local 5-pixel ds beats the global track fit).")
        elif tricell_rms > production_rms:
            print("   => production is tighter; the tricell ds does NOT "
                  "improve on the global track fit here.")
        else:
            print("   => production ≈ tricell; no measurable improvement.")
    else:
        print("=> comparison unavailable (a method had <2 usable "
              "tricells; loosen cuts or run more tracks).")
    print("")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_summary_plots(records, outdir,
                       witness_charge_floor=None,
                       witness_packet_floor=None,
                       direction_disagreement_cut_deg=None,
                       pitch_cm=None):
    # The calibration-constant uncertainty table is computed and printed
    # FIRST, from saved record fields only, so it appears in the run log
    # even on a machine without matplotlib (e.g. the offline --from-npz
    # path on a dev box). Plotting is best-effort below.
    calibration_summary = compute_calibration_ratio_summary(records)
    print_calibration_table(calibration_summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots "
              "(calibration table above was emitted from the records).")
        return

    # Pad pitch for the money-plot annotation. Offline (no detector
    # handy) fall back to the reconstructed Δw, which IS one pad pitch.
    if pitch_cm is None and records:
        delta_w_values = np.array(
            [float(r['delta_w_cm']) for r in records
             if np.isfinite(float(r.get('delta_w_cm', np.nan)))])
        if delta_w_values.size:
            pitch_cm = float(np.median(delta_w_values))

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

    # ===== ds bias vs zenith (the dominant residual structure) =====
    # After the Δt units fix the ds bias is no longer a flat offset —
    # it slopes with zenith: drift-dominated (drift-aligned) tracks
    # over-estimate ds, pad-plane-aligned tracks are ~unbiased. This
    # is the mixed-coordinate-sourcing residual (pad-center Δv,Δw vs
    # FEE-timing Δdrift) that no constant Δw can remove. The witness-
    # quality cuts are expected to trim the high-bias (low-zenith) end.
    zenith_deg = np.degrees(
        np.array([r['zenith_rad'] for r in records]))
    zenith_bins = np.linspace(10.0, 80.0, 15)
    zenith_bin_centers = 0.5 * (zenith_bins[:-1] + zenith_bins[1:])
    median_bias_per_bin = []
    mean_bias_per_bin = []
    stderr_bias_per_bin = []
    rms_bias_per_bin = []
    count_per_zenith_bin = []
    zenith_bin_index = np.digitize(zenith_deg, zenith_bins) - 1
    for bin_number in range(len(zenith_bins) - 1):
        in_bin = zenith_bin_index == bin_number
        values_in_bin = fractional_difference[in_bin]
        count_per_zenith_bin.append(int(in_bin.sum()))
        if in_bin.sum() >= 2:
            median_bias_per_bin.append(float(np.median(values_in_bin)))
            mean_bias_per_bin.append(float(np.mean(values_in_bin)))
            stderr_bias_per_bin.append(
                float(np.std(values_in_bin)) / sqrt(int(in_bin.sum())))
            rms_bias_per_bin.append(
                float(np.sqrt(np.mean(values_in_bin ** 2))))
        else:
            median_bias_per_bin.append(np.nan)
            mean_bias_per_bin.append(np.nan)
            stderr_bias_per_bin.append(np.nan)
            rms_bias_per_bin.append(np.nan)
    median_bias_per_bin = np.array(median_bias_per_bin)
    mean_bias_per_bin = np.array(mean_bias_per_bin)
    stderr_bias_per_bin = np.array(stderr_bias_per_bin)
    rms_bias_per_bin = np.array(rms_bias_per_bin)
    count_per_zenith_bin = np.array(count_per_zenith_bin)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax_count = ax.twinx()
    ax_count.bar(zenith_bin_centers, count_per_zenith_bin,
                 width=(zenith_bins[1] - zenith_bins[0]) * 0.9,
                 alpha=0.12, color="C0", zorder=0)
    ax_count.set_ylabel("tricells per bin", color="C0")
    ax_count.tick_params(axis="y", labelcolor="C0")
    ax.errorbar(zenith_bin_centers, mean_bias_per_bin,
                yerr=stderr_bias_per_bin, marker="o", ms=4, capsize=3,
                color="C1", label="mean ± stderr", zorder=3)
    ax.plot(zenith_bin_centers, median_bias_per_bin, marker="s", ms=4,
            color="C2", label="median", zorder=3)
    ax.plot(zenith_bin_centers, rms_bias_per_bin, marker="^", ms=4,
            color="C3", label="RMS", zorder=3)
    ax.axhline(0.0, color="k", ls="--", lw=0.6, label="perfect ds")
    ax.set_zorder(ax_count.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("track zenith [deg]  "
                  "(10° = drift-aligned, 80° = pad-plane-aligned)")
    ax.set_ylabel("(ds_recon − ds_truth) / ds_truth")
    ax.set_title(
        "Tricell ds bias vs zenith\n"
        "Drift-dominated (low-zenith) tracks over-estimate ds; "
        "pad-plane-aligned tracks ~unbiased.\n"
        "The mixed-coordinate-sourcing residual — flatten this to "
        "improve the calibration.")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    save_figure(fig, "tricell_ds_bias_vs_zenith.png")

    # ===== Witness-quality validation (CUT 1: halo-vs-track) =====
    # Sim-only check that the reco-only witness cuts (charge floor +
    # n_packets) keep the pads the track actually crossed and reject
    # the induced-only fringe pads. The truth label is the geometric
    # ds through each witness pad pillar — POLARITY-AGNOSTIC, and never
    # used as a cut (the negative induced lobe is invisible to the
    # readout, so net-integral/polarity is not a data-applicable
    # discriminant). Each tricell contributes its two witnesses.
    have_witness_diagnostics = (
        len(records) > 0
        and 'w0_witness_collected_charge' in records[0])
    if have_witness_diagnostics:
        witness_charge = np.concatenate([
            np.array([r['w0_witness_collected_charge'] for r in records]),
            np.array([r['w2_witness_collected_charge'] for r in records]),
        ])
        witness_n_packets = np.concatenate([
            np.array([r['w0_witness_n_packets'] for r in records]),
            np.array([r['w2_witness_n_packets'] for r in records]),
        ])
        witness_ds_truth = np.concatenate([
            np.array([r['ds_truth_w0_witness_cm'] for r in records]),
            np.array([r['ds_truth_w2_witness_cm'] for r in records]),
        ])
        track_crossing_mask = witness_ds_truth > 0.0
        induced_only_mask = ~track_crossing_mask

        fig, (ax_charge, ax_packets) = plt.subplots(
            1, 2, figsize=(13, 5.5))

        # Panel A: witness FEE charge, split by truth label.
        positive_charge = witness_charge[witness_charge > 0]
        if positive_charge.size > 0:
            charge_bins = np.linspace(
                0.0, np.percentile(positive_charge, 99), 60)
        else:
            charge_bins = np.linspace(0.0, 1.0, 60)
        ax_charge.hist(
            witness_charge[track_crossing_mask], bins=charge_bins,
            alpha=0.55, color="C2",
            label=(f"track-crossing (ds>0): "
                   f"N={int(track_crossing_mask.sum())}"))
        ax_charge.hist(
            witness_charge[induced_only_mask], bins=charge_bins,
            alpha=0.55, color="C3",
            label=(f"induced-only (ds=0): "
                   f"N={int(induced_only_mask.sum())}"))
        if witness_charge_floor is not None:
            ax_charge.axvline(
                witness_charge_floor, color="k", ls="--", lw=0.9,
                label=f"charge floor = {witness_charge_floor:.0f} e⁻")
        ax_charge.set_xlabel("witness FEE charge  Σq_packets  [e⁻]")
        ax_charge.set_ylabel("witness count")
        ax_charge.set_title("Witness charge by geometric truth label")
        ax_charge.legend(fontsize=8)
        ax_charge.grid(alpha=0.3)

        # Panel B: witness packet count, split by truth label.
        max_packets = int(max(witness_n_packets.max(), 1))
        packet_bins = np.arange(0.5, max_packets + 1.5, 1.0)
        ax_packets.hist(
            witness_n_packets[track_crossing_mask], bins=packet_bins,
            alpha=0.55, color="C2",
            label="track-crossing (ds>0)")
        ax_packets.hist(
            witness_n_packets[induced_only_mask], bins=packet_bins,
            alpha=0.55, color="C3",
            label="induced-only (ds=0)")
        if (witness_packet_floor is not None
                and witness_packet_floor > 1):
            ax_packets.axvline(
                witness_packet_floor - 0.5, color="k", ls="--", lw=0.9,
                label=f"packet floor = {witness_packet_floor}")
        ax_packets.set_xlabel("witness FEE packet count  n_packets")
        ax_packets.set_ylabel("witness count")
        ax_packets.set_title("Witness packets by geometric truth label")
        ax_packets.legend(fontsize=8)
        ax_packets.grid(alpha=0.3)

        fig.suptitle(
            "Witness halo-vs-track validation (sim truth label is "
            "geometric ds through the witness pad pillar; cuts are "
            "reco-only).\nGood separation ⇒ the charge+packet cuts keep "
            "real-collection witnesses and reject induced-only fringe.",
            fontsize=10)
        save_figure(fig, "tricell_witness_halo_validation.png")

        # ===== Witness-quality validation (CUT 2: direction) =====
        # Direction-disagreement distribution: should have a clean core
        # (good tricells) plus a rejectable tail (corrupted witnesses).
        # Split by whether BOTH witnesses are track-crossing vs at
        # least one induced-only, to confirm the tail is fringe-driven.
        direction_disagreement = np.array(
            [r['direction_disagreement_deg'] for r in records])
        both_cross_mask = (
            np.array([r['ds_truth_w0_witness_cm'] for r in records]) > 0.0
        ) & (
            np.array([r['ds_truth_w2_witness_cm'] for r in records]) > 0.0)
        finite_mask = np.isfinite(direction_disagreement)
        if finite_mask.any():
            fig, ax = plt.subplots(figsize=(10, 5.5))
            disagreement_bins = np.linspace(
                0.0, min(90.0, float(np.nanmax(
                    direction_disagreement[finite_mask])) + 1.0), 60)
            ax.hist(
                direction_disagreement[finite_mask & both_cross_mask],
                bins=disagreement_bins, alpha=0.55, color="C2",
                label=("both witnesses track-crossing: "
                       f"N={int((finite_mask & both_cross_mask).sum())}"))
            ax.hist(
                direction_disagreement[finite_mask & ~both_cross_mask],
                bins=disagreement_bins, alpha=0.55, color="C3",
                label=("≥1 witness induced-only: "
                       f"N={int((finite_mask & ~both_cross_mask).sum())}"))
            if (direction_disagreement_cut_deg is not None
                    and direction_disagreement_cut_deg < 180.0):
                ax.axvline(
                    direction_disagreement_cut_deg, color="k", ls="--",
                    lw=0.9,
                    label=(f"cut = "
                           f"{direction_disagreement_cut_deg:.0f}°"))
            ax.set_xlabel(
                "angle between witness-to-witness and three-w_1-pad "
                "directions [deg]")
            ax.set_ylabel("tricell count")
            ax.set_title(
                "Direction cross-check (CUT 2): witness-derived vs "
                "w_1-pad-derived track direction\n"
                "clean core = consistent tricells; tail = corrupted "
                "witnesses (mostly induced-only)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            save_figure(fig, "tricell_direction_disagreement.png")

    # ----- Money plot: calibration-constant uncertainty -----
    # Three overlaid histograms of the per-tricell calibration ratio
    # r = (q_centre/ds) / dQdx_truth, one per ds source. A narrower
    # distribution ⇒ a tighter calibration constant. The question the
    # plot answers: does the LOCAL tricell ds (C2) beat the GLOBAL
    # line-fit pitch PRODUCTION ds (C1) that NDLAr reconstruction uses
    # today? truth-ds (C0) is the SIM-ONLY ceiling.
    method_color = dict(production="C1", tricell="C2", truth="C0")
    ratio_arrays = {
        method: calibration_summary[method]['ratio_array']
        for method in CALIBRATION_METHOD_ORDER
        if calibration_summary[method]['N'] >= 2}
    if ratio_arrays:
        pooled = np.concatenate(list(ratio_arrays.values()))
        bin_low, bin_high = np.percentile(pooled, [1.0, 99.0])
        if not (np.isfinite(bin_low) and np.isfinite(bin_high)
                and bin_high > bin_low):
            bin_low, bin_high = float(pooled.min()), float(pooled.max())
        if bin_high <= bin_low:
            bin_high = bin_low + 1.0
        shared_bins = np.linspace(bin_low, bin_high, 51)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for method in CALIBRATION_METHOD_ORDER:
            s = calibration_summary[method]
            if s['N'] < 2:
                continue
            ax.hist(
                s['ratio_array'], bins=shared_bins, histtype="step",
                lw=2.0, color=method_color[method],
                label=(f"{CALIBRATION_METHOD_LABEL[method]}: "
                       f"median={s['median_ratio']:.4f}, "
                       f"fracRMS={s['fractional_rms']:.3f}, "
                       f"σ_const={s['standard_error']:.4f}, "
                       f"N(1%)={s['tracks_for_one_percent']:.0f} "
                       f"(N={s['N']})"))

        production_summary = calibration_summary['production']
        if np.isfinite(production_summary['median_ratio']):
            ax.axvline(
                production_summary['median_ratio'], color="C1", ls="--",
                lw=1.0, label="production median")

        production_rms = production_summary['fractional_rms']
        tricell_rms = calibration_summary['tricell']['fractional_rms']
        if (np.isfinite(production_rms) and np.isfinite(tricell_rms)
                and production_rms > 0):
            ratio_text = (
                f"tricell/production fracRMS = "
                f"{tricell_rms / production_rms:.2f}  "
                + ("(tricell tighter ✓)" if tricell_rms < production_rms
                   else "(production tighter)"))
        else:
            ratio_text = "comparison unavailable"

        pitch_text = (f"  [pad pitch = {pitch_cm:.4f} cm]"
                      if pitch_cm is not None else "")
        ax.set_xlabel(
            "calibration ratio  r = (q_centre / ds) / dQdx_truth")
        ax.set_ylabel("tricell count")
        ax.set_title(
            "Calibration-constant uncertainty: local tricell ds vs "
            "global line-fit (production) ds\n"
            "narrower = tighter calibration constant.  "
            + ratio_text + pitch_text
            + "\n(truth-ds is a SIM-ONLY ceiling; "
            "production = NDLAr-style global line-fit pitch)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        save_figure(fig, "tricell_calibration_constant_uncertainty.png")

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
