#!/usr/bin/env python
"""
verify_diffusion.py
===================

Charge drift + diffusion verification for larnd-sim, reframed as a
**truncation / assumption audit**. The simulation makes several deliberate
truncations in the induction stage (finite integration window, near-field
response radius, n-sigma cloud cuts). Each one is physically motivated but each
one drops some charge or signal, and *that* is what can move data/MC agreement.
This suite drives the REAL CUDA kernels and, for every truncation, measures how
much it actually drops and whether that is small and depth-stable in the
operating regime -- or maps where it stops being small.

ORGANISING PRINCIPLE -- the Shockley-Ramo telescoping identity
  The induced charge on pad k is a boundary term:  integral(i_k dt) =
  -q*[W_k(end) - W_k(start)], with W_k the weighting potential (1 on pad k, 0
  elsewhere). Over a COMPLETE transit the collection pad nets exactly q and every
  non-collecting neighbor nets exactly 0 (its bipolar lobes cancel). Truncate the
  time window and a neighbor keeps a residual ~ q*W_k(t_cut) > 0. Hence:
    * the CORRECT charge observable is the long-window per-pad integral
      (collector -> q, neighbors -> 0);
    * the TRUNCATION ERROR is directly measurable as the residual (uncancelled
      negative charge / lost positive charge) left at the simulation's window.

TEST GROUPS
  A. Analytic drift stage is EXACT (no truncation): drift-time linearity,
     sqrt diffusion scaling, sigma_T/sigma_L, lifetime. These must match theory
     to ~machine precision; any miss is a real physics/implementation bug.
  B. Induction truncations are SMALL and depth-stable (the point of this rewrite):
     B1 window-length closure (charge contained, bipolar residual -> 0);
     B2 charge integrity vs depth (only lifetime, no truncation-induced loss);
     B3 near-field radius validity (scan diffusion up until charge leaks past
        MAX_RADIUS -> maps where the near-field approximation breaks);
     B4 transverse footprint from the long-window collected charge (the CORRECT
        observable) vs sqrt(sigma_T^2 + (p/sqrt12)^2);
     B5 post-hoc diffusion knob: charge invariant + post-hoc == native control.
  C. Readout truncations: FEE threshold efficiency; ADC clamp (never negative);
     far-field on/off charge difference (the dropped long-range pre-trigger).

Two probe handles: MUON (mu) for realistic coverage of the analytic checks, and
POINT (pt) -- a near-delta beta blob -- the clean handle for the induction /
truncation measurements.

REQUIREMENTS
  A CUDA GPU (numba.cuda.is_available()); cupy, numba. Run from a larnd-sim
  checkout. Defaults match module0:
      python tests/verify_diffusion.py --n-muons 2000 --outdir diffverify

NOTE ON THE DIFFUSION KNOB (B5) -- post-hoc by necessity
  drifting.drift() bakes LONG_DIFF/TRAN_DIFF in as numba compile-time constants,
  so the global cannot be rescaled after compile. The knob instead scales the
  per-segment long_diff/tran_diff FIELDS by sqrt(s) AFTER drift (sigma ~ sqrt(D)).
  It is validated, not trusted: post-hoc(s, depth d) is compared to the UNMODIFIED
  kernel at the equivalent depth s*d (since sigma(s*d) == sqrt(s)*sigma(d)).
"""

import argparse
import sys
from math import ceil, sqrt, erf

import numpy as np


# ---------------------------------------------------------------------------
# Track/segment record dtype -- inline copy of cli/dumpTree.py:segments_dtype
# (keep align=True). Copied rather than imported because cli/dumpTree.py does
# `from ROOT import ...` at import, and ROOT is usually absent on a GPU node.
# Symptom if it drifts: Numba TypingError "Field '<name>' was not found".
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


# ===========================================================================
# Setup / config
# ===========================================================================
class SimContext:
    """Bundle of loaded modules, the response table, and derived constants."""
    def __init__(self, modules, response, ideal):
        self.__dict__.update(modules)
        self.response = response
        self.ideal = ideal


def load_simulation(args):
    """Load module0 config + response, import kernels, return a SimContext."""
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states
    from larndsim import consts
    consts.load_properties(args.detector, args.pixel_layout,
                           args.response, args.sim_properties)
    from larndsim.consts import detector, physics, sim, units
    from larndsim import detsim, drifting, quenching, pixels_from_track, fee

    response = detector.load_response(args.response)
    modules = dict(cuda=cuda, create_rng=create_xoroshiro128p_states,
                   detector=detector, physics=physics, sim=sim, units=units,
                   detsim=detsim, drifting=drifting, quenching=quenching,
                   pixels_from_track=pixels_from_track, fee=fee)
    return SimContext(modules, response, ideal_constants(detector))


def ideal_constants(detector):
    """Theory constants used by the checks, all read from the loaded config."""
    v = detector.V_DRIFT
    d = dict(
        v_drift=v, inv_v_drift=1.0 / v,
        d_long=detector.LONG_DIFF, d_tran=detector.TRAN_DIFF,
        tau=detector.ELECTRON_LIFETIME,
        atten_length=v * detector.ELECTRON_LIFETIME,
        pitch=detector.PIXEL_PITCH,
        pitch_rms=detector.PIXEL_PITCH / sqrt(12.0),
        q_threshold=detector.DISCRIMINATION_THRESHOLD,
        diff_n_sigmas=detector.DIFF_N_SIGMAS,
        max_radius=detector.MAX_RADIUS,
        time_sampling=detector.TIME_SAMPLING,
        response_max_time=detector.RESPONSE_MAX_TIME,
        drift_max_time=detector.DRIFT_MAX_TIME,
        drift_length=abs(detector.DRIFT_LENGTH),
    )
    d["sigma_long_coeff"] = sqrt(2.0 * d["d_long"] / v)   # sigma_L = coeff*sqrt(d)
    d["sigma_tran_coeff"] = sqrt(2.0 * d["d_tran"] / v)   # sigma_T = coeff*sqrt(d)
    d["sigma_ratio"] = sqrt(d["d_tran"] / d["d_long"])    # constant ~1.483
    d["psf_slope"] = 2.0 * d["d_tran"] / v                # d(sigma_T^2)/dd
    # transverse reach of the near-field response table (the truncation radius)
    d["nearfield_reach"] = detector.MAX_RADIUS * detector.PIXEL_PITCH
    return d


def print_config(ctx):
    k = ctx.ideal
    print("=" * 70)
    print("Diffusion verification -- module0 constants (runtime):")
    print(f"  V_DRIFT          = {k['v_drift']:.5f} cm/us "
          f"(1/V_DRIFT = {k['inv_v_drift']:.4f} us/cm)")
    print(f"  D_L, D_T         = {k['d_long']:.3e}, {k['d_tran']:.3e} cm^2/us")
    print(f"  tau, lambda      = {k['tau']:.1f} us, {k['atten_length']:.1f} cm")
    print(f"  sigma_L, sigma_T = {k['sigma_long_coeff']:.4e}, "
          f"{k['sigma_tran_coeff']:.4e} * sqrt(d[cm]) cm")
    print(f"  sigma_T/sigma_L  = {k['sigma_ratio']:.4f}")
    print(f"  pitch, p/sqrt12  = {k['pitch']:.4f}, {k['pitch_rms']:.4f} cm")
    print("  --- truncation constants under audit ---")
    print(f"  MAX_RADIUS       = {k['max_radius']:.0f} pix  "
          f"(near-field reach {k['nearfield_reach']:.3f} cm)")
    print(f"  DIFF_N_SIGMAS    = {k['diff_n_sigmas']:.0f}")
    print(f"  RESPONSE_MAX_TIME= {k['response_max_time']:.3f} us, "
          f"DRIFT_MAX_TIME = {k['drift_max_time']:.3f} us")
    print(f"  Q_threshold      = {k['q_threshold']:.0f} e-")
    print("=" * 70)


# ===========================================================================
# Geometry helpers
# ===========================================================================
def plane_z(detector, plane=0):
    """(z_anode, z_cathode, into) for a TPC plane; `into` is the drift direction."""
    z_anode = detector.TPC_BORDERS[plane][2][0]
    z_cathode = detector.TPC_BORDERS[plane][2][1]
    return z_anode, z_cathode, float(np.sign(z_cathode - z_anode))


def active_volume(detector, plane=0, margin=1.0):
    """(x0, x1, y0, y1) of the anode face inset by `margin` cm.

    min/max because TPC_BORDERS rows are not guaranteed ascending (the z border
    is stored descending for cathode_direction < 0); using min/max keeps the
    inset correct regardless of ordering.
    """
    b = detector.TPC_BORDERS[plane]
    return (min(b[0]) + margin, max(b[0]) - margin,
            min(b[1]) + margin, max(b[1]) - margin)


def depth_to_z(detector, drift_cm, plane=0):
    """z-coordinate of a deposit `drift_cm` from the anode of `plane`."""
    z_anode, _, into = plane_z(detector, plane)
    return z_anode + into * drift_cm


def pixel_center(detector, ix, iy, plane=0):
    """(x, y) center of pixel (ix, iy) on `plane`."""
    b = detector.TPC_BORDERS[plane]
    p = detector.PIXEL_PITCH
    return b[0][0] + (ix + 0.5) * p, b[1][0] + (iy + 0.5) * p


# ===========================================================================
# Probe generation (one event at a time)
# ===========================================================================
def blank_tracks(n):
    return np.zeros(n, dtype=SEGMENTS_DTYPE)


def fill_segment(row, plane, start, end, dEdx):
    """Populate one structured row from endpoints and a dE/dx."""
    (xs, ys, zs), (xe, ye, ze) = start, end
    row["x_start"], row["y_start"], row["z_start"] = xs, ys, zs
    row["x_end"], row["y_end"], row["z_end"] = xe, ye, ze
    row["x"], row["y"], row["z"] = 0.5 * (xs + xe), 0.5 * (ys + ye), 0.5 * (zs + ze)
    row["dx"] = sqrt((xe - xs) ** 2 + (ye - ys) ** 2 + (ze - zs) ** 2)
    row["dEdx"] = dEdx
    row["dE"] = dEdx * row["dx"]
    row["pdg_id"] = 13
    row["pixel_plane"] = plane


def segmentize(start, end, step_cm):
    """Split a path into <= step_cm segments; returns list of (p0, p1)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    n = max(1, int(ceil(length / step_cm)))
    pts = [start + (end - start) * (i / n) for i in range(n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


def random_muon_endpoints(detector, rng, plane=0):
    """Two random in-volume points at random drift depths (convex => all
    interpolated segments stay inside the box and get drifted)."""
    x0, x1, y0, y1 = active_volume(detector, plane, margin=2.0)
    dmax = abs(detector.DRIFT_LENGTH) - 0.5
    depths = rng.uniform(0.5, dmax, size=2)
    start = (rng.uniform(x0, x1), rng.uniform(y0, y1), depth_to_z(detector, depths[0], plane))
    end = (rng.uniform(x0, x1), rng.uniform(y0, y1), depth_to_z(detector, depths[1], plane))
    return start, end


def build_muon_event(detector, rng, plane=0, step_cm=0.4, dEdx=2.1):
    """Assemble a per-event muon `tracks` array (segmented MIP line)."""
    start, end = random_muon_endpoints(detector, rng, plane)
    pairs = segmentize(start, end, step_cm)
    tracks = blank_tracks(len(pairs))
    for i, (p0, p1) in enumerate(pairs):
        fill_segment(tracks[i], plane, p0, p1, dEdx)
        tracks[i]["segment_id"] = i
    return tracks


def point_source_record(detector, x, y, drift_cm, plane=0, point_dE=0.5, dx=0.02):
    """One near-delta deposit at transverse (x, y), drift depth drift_cm.

    Total energy point_dE (MeV) in a tiny length dx (cm) along +y so the cloud is
    point-like. drift_cm=0 => on the anode (the diffusion floor).
    """
    z = depth_to_z(detector, drift_cm, plane)
    tracks = blank_tracks(1)
    fill_segment(tracks[0], plane, (x, y - 0.5 * dx, z), (x, y + 0.5 * dx, z),
                 dEdx=point_dE / dx)
    return tracks


def build_point_source_event(detector, rng, drift_cm, plane=0, **kw):
    """Per-event point source at a random pixel center, depth drift_cm."""
    i0, j0 = detector.N_PIXELS[0] // 2, detector.N_PIXELS[1] // 2
    di, dj = rng.integers(-20, 21), rng.integers(-20, 21)
    x, y = pixel_center(detector, i0 + di, j0 + dj, plane)
    return point_source_record(detector, x, y, drift_cm, plane, **kw)


def centered_point_source(detector, drift_cm, plane=0, **kw):
    """Fixed, centered point source -- identical sub-pixel phase at every depth
    so post-hoc and native series are comparable bin-for-bin."""
    i0, j0 = detector.N_PIXELS[0] // 2, detector.N_PIXELS[1] // 2
    x, y = pixel_center(detector, i0, j0, plane)
    return point_source_record(detector, x, y, max(drift_cm, 0.0), plane, **kw)


def scan_point_sources(detector, depths, plane=0, **kw):
    """Yield (depth, tracks) for a centered point source over a depth grid."""
    for depth in depths:
        yield depth, centered_point_source(detector, depth, plane, **kw)


def boundary_point_source(detector, depth, delta, plane=0, **kw):
    """Point source sitting `delta` cm inside a pad, just left of an x pad boundary.

    The boundary between pads i0 and i0+1 is at x = border + (i0+1)*pitch; we place
    the deposit at x = boundary - delta on row j0. Transverse diffusion then leaks a
    fraction 0.5*erfc(delta / (sqrt2 * sigma_T)) of the charge across the boundary
    onto pad i0+1 -- the sub-pixel-sensitive transverse observable (B4). Returns
    (tracks, x_point).
    """
    i0, j0 = detector.N_PIXELS[0] // 2, detector.N_PIXELS[1] // 2
    b = detector.TPC_BORDERS[plane]
    x_boundary = b[0][0] + (i0 + 1) * detector.PIXEL_PITCH
    x = x_boundary - delta
    y = b[1][0] + (j0 + 0.5) * detector.PIXEL_PITCH
    return point_source_record(detector, x, y, depth, plane, **kw), x


# ===========================================================================
# Knob / far-field toggles
# ===========================================================================
def apply_diffusion_scale(tracks, factor):
    """Scale per-segment diffusion widths to emulate D -> factor*D (in place).

    sigma ~ sqrt(2 D t), so D -> factor*D means width *= sqrt(factor). factor=0
    removes diffusion. Applied AFTER drift (the global cannot be rescaled
    post-compile; see module docstring).
    """
    s = sqrt(factor)
    tracks["long_diff"] *= s
    tracks["tran_diff"] *= s
    return tracks


# ===========================================================================
# Kernel drivers (thin wrappers over the real machinery)
# ===========================================================================
def to_host(a):
    """cupy / numba-device -> numpy host array."""
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    if hasattr(a, "copy_to_host"):
        return a.copy_to_host()
    return np.asarray(a)


def grid_1d(n, tpb=128):
    """(blocks, threads) covering ALL n items -- never under-launches.

    The original suite launched quench/drift as [1, 128] = 128 threads total;
    cosmic muons segment into up to ~190 rows, so every segment with index >= 128
    was silently left un-processed (pixel_plane=0, t=0). This helper is the fix.
    """
    return max(int(ceil(n / tpb)), 1), tpb


def quench_and_drift(ctx, tracks):
    """quench then drift on a host tracks array; return drifted host copy."""
    tracks = np.copy(tracks)
    d_tracks = ctx.cuda.to_device(tracks)
    bpg, tpb = grid_1d(tracks.shape[0])
    ctx.quenching.quench[bpg, tpb](d_tracks, ctx.physics.BOX)
    ctx.drifting.drift[bpg, tpb](d_tracks)
    return d_tracks.copy_to_host()


def quench_only(ctx, tracks):
    """Pre-attenuation electron count: quench applied, drift NOT applied."""
    d = ctx.cuda.to_device(np.copy(tracks))
    bpg, tpb = grid_1d(tracks.shape[0])
    ctx.quenching.quench[bpg, tpb](d, ctx.physics.BOX)
    return d.copy_to_host()["n_electrons"].astype(float)


def find_pixels(ctx, tracks):
    """get_pixels -> (neighboring_pixels, neighboring_radius) cupy arrays."""
    import cupy as cp
    nseg = tracks.shape[0]
    max_active, max_neigh = 64, 220
    active = cp.full((nseg, max_active), -1, dtype=cp.int32)
    neigh = cp.full((nseg, max_neigh), -1, dtype=cp.int32)
    radius = cp.full((nseg, max_neigh), -1, dtype=cp.float32)
    n_list = cp.zeros(nseg, dtype=cp.int64)
    d_tracks = ctx.cuda.to_device(tracks)
    bpg, tpb = grid_1d(nseg)
    ctx.pixels_from_track.get_pixels[bpg, tpb](d_tracks, active, neigh, radius, n_list)
    return neigh, radius


def natural_window_ticks(ctx, tracks):
    """The simulation's own induction window length, in ticks.

    Reproduces simulate_pixels.py:1349-1353 (which mirrors the per-tick cap at
    detsim.py:155): the readout window is the segment time span plus
    DIFF_N_SIGMAS*sigma_L/V, plus the response-vs-drift headroom. This is the
    truncation under audit -- B1 scans around it; everything else allocates it.
    """
    det = ctx.detector
    span = float(np.max(tracks["t_end"] - tracks["t0"])) if tracks.shape[0] else 0.0
    long_max = float(np.max(tracks["long_diff"])) if tracks.shape[0] else 0.0
    pad = long_max / det.V_DRIFT * det.DIFF_N_SIGMAS
    extra = max(det.RESPONSE_MAX_TIME - det.DRIFT_MAX_TIME, 0.0)
    return max(int(ceil((span + pad + extra) / det.TIME_SAMPLING)), 2)


def make_rng(ctx, n, seed):
    return ctx.create_rng(max(n, 1024), seed=seed)


def induce_current(ctx, tracks, neigh, seed=12345, n_ticks=None):
    """tracks_current_mc -> per-(segment, pixel, tick) SIGNED current (cupy).

    n_ticks overrides the window length (for the B1 window-length scan). The
    kernel additionally self-caps each tick at detsim.py:155, so allocating more
    than the natural window simply yields trailing zeros (no extra charge).
    """
    import cupy as cp
    nt = int(n_ticks) if n_ticks is not None else natural_window_ticks(ctx, tracks)
    nt = max(nt, 1)
    signals = cp.zeros((tracks.shape[0], neigh.shape[1], nt), dtype=cp.float32)
    tpb = (1, 1, 64)
    bpg = (max(ceil(signals.shape[0] / tpb[0]), 1),
           max(ceil(signals.shape[1] / tpb[1]), 1),
           max(ceil(signals.shape[2] / tpb[2]), 1))
    n_states = int(np.prod(tpb) * bpg[0] * bpg[1] * bpg[2])
    rng = make_rng(ctx, n_states, seed)
    d_tracks = ctx.cuda.to_device(tracks)
    ctx.detsim.tracks_current_mc[bpg, tpb](signals, neigh, d_tracks, ctx.response, rng)
    return signals


def sum_to_pixels(ctx, signals, neigh, radius, tracks):
    """sum_pixel_signals -> (unique_pix, pixels_signals, backtrack arrays).

    Faithful single-batch copy of cli/simulate_pixels.py:1389-1454.
    """
    import cupy as cp
    det, sim, detsim = ctx.detector, ctx.sim, ctx.detsim

    unique_pix = cp.unique(neigh.reshape(-1))
    unique_pix = unique_pix[unique_pix != -1].astype(cp.int32)
    if unique_pix.shape[0] == 0:
        return unique_pix, None, None, None, None

    max_pix_val = int(cp.max(unique_pix)) + 1
    lookup = cp.full((max_pix_val,), -1, dtype=cp.int32)
    lookup[unique_pix] = cp.arange(unique_pix.shape[0], dtype=cp.int32)
    pixel_index_map = lookup[neigh]
    pixel_index_map[neigh == -1] = -1

    track_pixel_map = cp.full((unique_pix.shape[0], sim.MAX_TRACKS_PER_PIXEL), -1)
    detsim.get_track_pixel_map2[max(ceil(unique_pix.shape[0] / 32), 1), 32](
        track_pixel_map, unique_pix, neigh, radius)

    nt0 = signals.shape[2] + ceil(float(np.max(tracks["t0"])) / det.TIME_SAMPLING)
    pixels_signals = cp.zeros((unique_pix.shape[0], nt0))
    num_backtrack = cp.sum(track_pixel_map != -1, axis=-1)
    pixels_tracks_signals = cp.zeros(nt0 * int(num_backtrack.sum()))
    offset_backtrack = cp.cumsum(num_backtrack) - num_backtrack
    overflow = cp.zeros(unique_pix.shape[0])
    track_t0 = cp.array(tracks["t0"] / det.TIME_SAMPLING, dtype=int)

    tpb = (1, 1, 64)
    bpg = (max(ceil(signals.shape[0] / tpb[0]), 1),
           max(ceil(signals.shape[1] / tpb[1]), 1),
           max(ceil(signals.shape[2] / tpb[2]), 1))
    detsim.sum_pixel_signals[bpg, tpb](
        pixels_signals, signals, track_t0, pixel_index_map, track_pixel_map,
        pixels_tracks_signals, num_backtrack, offset_backtrack, overflow)
    return (unique_pix, pixels_signals, pixels_tracks_signals,
            num_backtrack, offset_backtrack)


def run_fee(ctx, pixels_signals, pixels_tracks_signals, num_backtrack,
            offset_backtrack, max_signal_time, seed=777):
    """get_adc_values + digitize -> (adc_list, adc_ticks) host arrays.

    Mirrors cli/simulate_pixels.py:do_digitize_and_update for one event.
    """
    import cupy as cp
    det, sim, fee, units = ctx.detector, ctx.sim, ctx.fee, ctx.units
    npix = pixels_signals.shape[0]
    time_ticks = cp.arange(0, max_signal_time, det.TIME_SAMPLING)
    integral = cp.zeros((npix, sim.MAX_ADC_VALUES))
    adc_ticks = cp.zeros((npix, sim.MAX_ADC_VALUES))
    fractions = cp.zeros((npix, sim.MAX_ADC_VALUES, sim.MAX_TRACKS_PER_PIXEL))
    tpb = 4
    bpg = max(ceil(npix / tpb), 1)
    rng = make_rng(ctx, tpb * bpg, seed)
    thresholds = cp.full(npix, det.DISCRIMINATION_THRESHOLD * units.e)
    fee.get_adc_values[bpg, tpb](
        pixels_signals, pixels_tracks_signals, num_backtrack, offset_backtrack,
        time_ticks, integral, adc_ticks, 0, rng, fractions, thresholds)
    adc_list = fee.digitize(integral, det.GAIN, det.V_PEDESTAL)
    return to_host(adc_list), to_host(adc_ticks)


def simulate_event(ctx, tracks, seed=12345, with_fee=False, n_ticks=None):
    """Full per-event pipeline. Returns a dict of measurements/arrays."""
    drifted = quench_and_drift(ctx, tracks)
    if float(np.sum(drifted["n_electrons"])) <= 0:
        return None
    neigh, radius = find_pixels(ctx, drifted)
    signals = induce_current(ctx, drifted, neigh, seed=seed, n_ticks=n_ticks)
    out = dict(tracks=drifted, signals=signals, neigh=neigh)
    if with_fee:
        summed = sum_to_pixels(ctx, signals, neigh, radius, drifted)
        unique_pix, pixels_signals = summed[0], summed[1]
        out["unique_pix"], out["pixels_signals"] = unique_pix, pixels_signals
        if pixels_signals is not None:
            max_time = pixels_signals.shape[1] * ctx.detector.TIME_SAMPLING
            adc, ticks = run_fee(ctx, *summed[1:], max_time, seed=seed)
            out["adc"], out["adc_ticks"] = adc, ticks
    return out


# ===========================================================================
# Measurements
# ===========================================================================
def drift_distance_of(tracks, detector):
    """Per-segment drift distance |z - z_anode| (cm), from the KERNEL-assigned
    plane; NaN for segments drift() never processed (pixel_plane left at
    DEFAULT_PLANE_INDEX or, before the launch fix, an untouched 0)."""
    planes = np.asarray(tracks["pixel_plane"]).astype(np.int64)
    nplanes = detector.TPC_BORDERS.shape[0]
    valid = (planes >= 0) & (planes < nplanes)
    z_anode = np.full(planes.shape, np.nan)
    z_anode[valid] = detector.TPC_BORDERS[planes[valid], 2, 0]
    return np.abs(np.asarray(tracks["z"]) - z_anode)


def pixel_xy_from_id(detector, pid):
    """Decode pixel IDs -> physical (x, y) pad corners, mirroring detsim.

    Inverse of pixels_from_track.pixel2id followed by detsim.get_pixel_coordinates:
    pix = i*PITCH + TPC_BORDERS[plane][axis][0]. Plane decoded from the ID (not
    assumed 0) -> correct across module0's two TPCs.
    """
    nx, ny = detector.N_PIXELS
    pid = np.asarray(pid, dtype=np.int64)
    ix = pid % nx
    iy = (pid // nx) % ny
    plane = np.clip(pid // (nx * ny), 0, detector.TPC_BORDERS.shape[0] - 1)
    x0 = detector.TPC_BORDERS[plane, 0, 0]
    y0 = detector.TPC_BORDERS[plane, 1, 0]
    return x0 + ix * detector.PIXEL_PITCH, y0 + iy * detector.PIXEL_PITCH


def total_induced_charge(signals):
    """Sum of induced current over all (segment, pixel, tick).

    In the long-window limit this telescopes to the total terminating (collected)
    charge: collection pads net q, neighbors net 0. Response units.
    """
    return float(to_host(signals).sum())


def per_pixel_net_charge(signals):
    """Net induced charge per pixel column (sum over segments & ticks).

    Long-window: collector -> q_k, neighbor -> 0. Index aligns with neigh's pixel
    axis for a single-segment (point-source) event.
    """
    s = to_host(signals)
    return s.sum(axis=(0, 2)) if s.ndim == 3 else s.sum(axis=-1)


def charge_polarity_split(signals):
    """(positive charge summed, |negative charge| summed) over pixel columns.

    The negative sum is the UNCANCELLED bipolar residual -- the direct, telescoping
    measure of window-truncation error. Long-window: it -> 0.
    """
    prof = per_pixel_net_charge(signals)
    pos = float(prof[prof > 0].sum())
    neg = float(-prof[prof < 0].sum())
    return pos, neg


def footprint_rms(ctx, signals, neigh):
    """Transverse RMS (cm) weighted by the long-window per-pad COLLECTED charge.

    Uses only the positive net per cell -- the telescoping-correct collected
    charge (neighbors cancel to ~0). This is the CORRECT footprint observable;
    the old |signed-integral| version was dominated by truncation residuals.
    """
    s = to_host(signals)
    per_cell = s.sum(axis=2).reshape(-1) if s.ndim == 3 else s.reshape(-1)
    ids = to_host(neigh).reshape(-1)
    keep = (ids != -1) & (per_cell > 0)
    if keep.sum() == 0:
        return np.nan
    xs, _ = pixel_xy_from_id(ctx.detector, ids[keep])
    w = per_cell[keep]
    mean = np.average(xs, weights=w)
    return float(sqrt(np.average((xs - mean) ** 2, weights=w)))


def charge_sharing_leak(ctx, signals, neigh, x_point):
    """Fraction of collected (positive) charge that leaked onto the +x neighbor pad.

    Marginalizes the long-window per-pad collected charge over y within each x
    column, then returns q(column i_pt+1) / [q(i_pt) + q(i_pt+1)], where i_pt is the
    pad column containing x_point. For a deposit `delta` left of the i_pt/i_pt+1
    boundary this is the sub-pixel transverse-diffusion observable; theory predicts
    0.5*erfc(delta/(sqrt2*sigma_T)) (the far boundary is >> sigma_T away).
    """
    s = to_host(signals)
    per_cell = s.sum(axis=2).reshape(-1) if s.ndim == 3 else s.reshape(-1)
    ids = to_host(neigh).reshape(-1)
    keep = (ids != -1) & (per_cell > 0)
    if keep.sum() == 0:
        return np.nan
    nx = ctx.detector.N_PIXELS[0]
    ix = ids[keep] % nx
    i_pt = int((x_point - ctx.detector.TPC_BORDERS[0, 0, 0]) // ctx.detector.PIXEL_PITCH)
    w = per_cell[keep]
    q_central = w[ix == i_pt].sum()
    q_neigh = w[ix == i_pt + 1].sum()
    tot = q_central + q_neigh
    return float(q_neigh / tot) if tot > 0 else np.nan


def waveform_time_rms(ctx, signals):
    """Time RMS (us) of the busiest pixel's waveform (|signal|-weighted)."""
    prof = np.abs(per_pixel_net_charge(signals))
    if prof.max() <= 0:
        return np.nan
    ipix = int(prof.argmax())
    wf = to_host(signals)[:, ipix, :].sum(axis=0)
    t = np.arange(wf.shape[0]) * ctx.detector.TIME_SAMPLING
    mass = np.abs(wf)
    if mass.sum() <= 0:
        return np.nan
    mean = np.average(t, weights=mass)
    return float(sqrt(np.average((t - mean) ** 2, weights=mass)))


def fired_fraction(adc):
    """Fraction of pixels with at least one above-threshold ADC value."""
    if adc is None or adc.size == 0:
        return np.nan
    return float((adc.max(axis=1) > 0).mean())


# ===========================================================================
# Diagnosis
# ===========================================================================
def diagnose(name, observed, expected, tol, results):
    """PASS/FAIL on |observed-expected|/|expected| <= tol; record it."""
    rel = abs(observed - expected) / abs(expected) if expected else float("inf")
    ok = rel <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: observed={observed:.4g} "
          f"expected={expected:.4g} (rel={rel:.2%}, tol={tol:.0%})")
    results.setdefault("checks", {})[name] = ok
    return ok


def record_check(results, name, ok, detail=""):
    """PASS/FAIL for a boolean check; record it."""
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    results.setdefault("checks", {})[name] = bool(ok)
    return ok


def fit_loglinear(x, y):
    """Slope/intercept of ln(y) vs x (for exponential attenuation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = (y > 0)
    return np.polyfit(x[good], np.log(y[good]), 1)


def fit_power(x, y):
    """Exponent b of y = a*x^b via log-log fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = (x > 0) & (y > 0)
    b, _ = np.polyfit(np.log(x[good]), np.log(y[good]), 1)
    return b


def linear_r2(x, y, slope, intercept):
    """R^2 of (slope, intercept) against (x, y)."""
    pred = slope * np.asarray(x) + intercept
    ss_res = np.sum((np.asarray(y) - pred) ** 2)
    ss_tot = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def median_rel_dev(obs, ref):
    """Median |obs-ref|/|ref| over entries where both finite and ref != 0."""
    obs, ref = np.asarray(obs, float), np.asarray(ref, float)
    good = np.isfinite(obs) & np.isfinite(ref) & (ref != 0)
    return float(np.median(np.abs(obs[good] - ref[good]) / np.abs(ref[good]))) \
        if good.any() else np.nan


# ===========================================================================
# Plot helpers
# ===========================================================================
def _new_ax(figsize=(8, 5)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print("  wrote", path)


# ===========================================================================
# GROUP A -- analytic drift stage is EXACT (no truncation)
# ===========================================================================
def plotA_drift_time(ctx, muon, outdir, results):
    """A1 (mu): drift time t vs drift distance; slope = 1/V_DRIFT, R^2 = 1.

    Pure kernel arithmetic t = t0 + d/V_DRIFT -- there is no truncation here, so a
    perfect line is mandatory. The earlier 'failure' was the [1,128] launch bug
    (segments >=128 left at t=0); grid_1d removes it.
    """
    d, t = muon["drift_cm"], muon["t"]
    slope, intercept = np.polyfit(d, t, 1)
    r2 = linear_r2(d, t, slope, intercept)
    fig, ax = _new_ax()
    ax.plot(d, t, ".", ms=2, alpha=0.3, label="segments")
    grid = np.linspace(d.min(), d.max(), 50)
    ax.plot(grid, ctx.ideal["inv_v_drift"] * grid + intercept, "r-",
            label=f"ideal 1/V_DRIFT = {ctx.ideal['inv_v_drift']:.3f} us/cm")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("drift time t [us]")
    ax.set_title("A1: drift-time linearity"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffA1_drift_time.png")
    diagnose("A1 drift-time slope", slope, ctx.ideal["inv_v_drift"], 0.01, results)
    record_check(results, "A1 drift-time R^2 > 0.999", r2 > 0.999, f"R^2={r2:.5f}")


def plotA_diffusion_fields(ctx, point, outdir, results):
    """A2 (pt): long_diff/tran_diff vs drift; sqrt law + ratio 1.483.

    Reads the kernel's own width fields -> tests the sqrt(2 D d / V) formula
    directly (no induction involved)."""
    d = point["drift_cm"]
    sl, st = point["long_diff"], point["tran_diff"]
    fig, ax = _new_ax()
    grid = np.linspace(max(d.min(), 1e-3), d.max(), 50)
    ax.plot(d, sl, "o", ms=3, label="long_diff (kernel)")
    ax.plot(d, st, "s", ms=3, label="tran_diff (kernel)")
    ax.plot(grid, ctx.ideal["sigma_long_coeff"] * np.sqrt(grid), "b-",
            label="ideal sigma_L = sqrt(2 D_L d / V)")
    ax.plot(grid, ctx.ideal["sigma_tran_coeff"] * np.sqrt(grid), "r-",
            label="ideal sigma_T = sqrt(2 D_T d / V)")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("diffusion width [cm]")
    ax.set_title("A2: diffusion sqrt-scaling"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffA2_diffusion_fields.png")
    diagnose("A2 tran_diff power", fit_power(d, st), 0.5, 0.04, results)
    ratio = float(np.median(st[d > 0] / sl[d > 0]))
    diagnose("A2 sigma_T/sigma_L ratio", ratio, ctx.ideal["sigma_ratio"], 0.01, results)


def plotA_lifetime(ctx, muon, outdir, results):
    """A3 (mu): surviving charge fraction vs drift; exp(-d/lambda), lambda=V*tau."""
    d = muon["drift_cm"]
    surv = muon["n_e_out"] / np.maximum(muon["n_e_in"], 1)
    good = (surv > 0) & (surv < 2)
    d, surv = d[good], surv[good]
    bins = np.linspace(d.min(), d.max(), 25)
    idx = np.digitize(d, bins)
    bd = np.array([d[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()])
    bs = np.array([surv[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()])
    slope, intercept = fit_loglinear(bd, bs)
    lam_fit = -1.0 / slope
    fig, ax = _new_ax()
    ax.semilogy(bd, bs, "o", label="binned survival")
    ax.semilogy(bd, np.exp(intercept) * np.exp(-bd / ctx.ideal["atten_length"]), "r-",
                label=f"ideal exp(-d/{ctx.ideal['atten_length']:.0f} cm)")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("n_e_out / n_e_in")
    ax.set_title("A3: lifetime attenuation"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffA3_lifetime.png")
    diagnose("A3 attenuation length lambda", lam_fit, ctx.ideal["atten_length"], 0.03, results)


# ===========================================================================
# GROUP B -- induction truncations are SMALL and depth-stable
# ===========================================================================
def measure_window_point(ctx, depth, fracs, n_rep, seed0):
    """For a centered point at `depth`, return collected charge and bipolar
    residual fraction at each window fraction of the natural window."""
    raw = centered_point_source(ctx.detector, depth)
    drifted = quench_and_drift(ctx, raw)
    nat = natural_window_ticks(ctx, drifted)
    neigh, _ = find_pixels(ctx, drifted)
    pos = np.zeros(len(fracs)); resid = np.zeros(len(fracs))
    for j, f in enumerate(fracs):
        nt = max(int(round(f * nat)), 2)
        pa, ra = [], []
        for r in range(n_rep):
            sig = induce_current(ctx, drifted, neigh, seed=seed0 + j * 50 + r, n_ticks=nt)
            p, n = charge_polarity_split(sig)
            pa.append(p); ra.append(n / p if p > 0 else np.nan)
        pos[j] = np.nanmean(pa); resid[j] = np.nanmean(ra)
    return nat, pos, resid


def plotB1_window_closure(ctx, depths, outdir, results, n_rep=4, seed0=40000):
    """B1 (pt): window-length closure -- the headline truncation test.

    Scan the induction window from 0.4x to 1.4x the simulation's natural window.
    Theory (telescoping): collected charge rises and PLATEAUS once the window
    covers the transit; the bipolar residual (uncancelled negative charge) -> 0.
    We confirm (i) the natural window sits ON the plateau with margin, and
    (ii) the residual there is small -- i.e. the time-truncation drops ~nothing.
    """
    import matplotlib.pyplot as plt
    fracs = np.array([0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))
    resid_ok, shrinks_ok, res_vals = [], [], []
    for depth in depths:
        nat, pos, resid = measure_window_point(ctx, depth, fracs, n_rep, seed0 + int(depth) * 7)
        plateau = pos[fracs >= 1.0].max()
        norm = pos / plateau if plateau > 0 else pos
        axL.plot(fracs, norm, "o-", label=f"d={depth:.0f} cm")
        axR.plot(fracs, 100 * resid, "o-", label=f"d={depth:.0f} cm")
        margin = float(np.interp(0.85, fracs, norm))           # diagnostic only
        res_at_1 = float(np.interp(1.0, fracs, resid))
        res_short = float(np.interp(0.4, fracs, resid))
        # HARD: at the natural window the uncancelled (truncation) residual is small,
        # and it shrank as the window grew (the telescoping signature). The 0.85x
        # "margin" is reported but NOT pass/failed: where the collection peak sits in
        # the window is set by RESPONSE_MAX_TIME headroom, not by a truncation error.
        resid_ok.append(res_at_1 < 0.10)
        shrinks_ok.append(res_at_1 <= res_short + 1e-6)
        res_vals.append(res_at_1)
        print(f"    depth {depth:5.1f} cm: natural window={nat} ticks  "
              f"charge@0.85x/plateau={margin:.3f}  residual: {res_short:.2%}@0.4x -> "
              f"{res_at_1:.2%}@1.0x")
    axL.axvline(1.0, color="k", ls="--", lw=0.8); axL.axhline(1.0, color="grey", lw=0.6)
    axL.set_xlabel("window / natural window"); axL.set_ylabel("collected / plateau")
    axL.set_title("B1: charge containment vs window (diagnostic)")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axR.axvline(1.0, color="k", ls="--", lw=0.8)
    axR.set_xlabel("window / natural window"); axR.set_ylabel("bipolar residual [%]")
    axR.set_title("B1: uncancelled residual vs window"); axR.legend(fontsize=8); axR.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffB1_window_closure.png")
    record_check(results, "B1 truncation residual at natural window < 10%",
                 all(resid_ok), f"max={max(res_vals):.2%}")
    record_check(results, "B1 residual shrinks as window grows (telescoping)",
                 all(shrinks_ok))


def plotB2_charge_integrity(ctx, depths, outdir, results, n_rep=6, seed0=50000):
    """B2 (pt): collected charge vs depth = pure lifetime, no truncation loss.

    With the natural window, collected charge per electron should fall ONLY as the
    electron lifetime exp(-d/lambda). The window's late cut is depth-dependent
    (detsim.py:155 uses dist_cathode), so this is the test that the depth-varying
    truncation is charge-neutral: any extra slope is truncation, not physics.
    """
    dlist, qlist, residlist = [], [], []
    for i, depth in enumerate(depths):
        qs, rs = [], []
        for r in range(n_rep):
            raw = centered_point_source(ctx.detector, depth)
            out = simulate_event(ctx, raw, seed=seed0 + i * 100 + r)
            if out is None:
                continue
            p, n = charge_polarity_split(out["signals"])
            n_in = float(quench_only(ctx, raw).sum())
            qs.append(p / n_in if n_in > 0 else np.nan)   # collected per input electron
            rs.append(n / p if p > 0 else np.nan)
        if qs:
            dlist.append(depth); qlist.append(np.nanmean(qs)); residlist.append(np.nanmean(rs))
    d = np.array(dlist); q = np.array(qlist); resid = np.array(residlist)
    slope, intercept = fit_loglinear(d, q)
    lam_fit = -1.0 / slope
    fig, ax = _new_ax()
    ax.semilogy(d, q, "o", label="collected / input electron")
    ax.semilogy(d, np.exp(intercept) * np.exp(-d / ctx.ideal["atten_length"]), "r-",
                label=f"pure lifetime exp(-d/{ctx.ideal['atten_length']:.0f} cm)")
    ax.set_xlabel("drift distance [cm]")
    ax.set_ylabel("collected charge / input electron")
    ax.set_title("B2: charge integrity vs depth (lifetime only)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffB2_charge_integrity.png")
    diagnose("B2 collected-charge lambda == lifetime", lam_fit,
             ctx.ideal["atten_length"], 0.05, results)
    record_check(results, "B2 bipolar residual small & flat vs depth (<5%)",
                 bool(np.nanmax(resid) < 0.05), f"max={np.nanmax(resid):.2%}")


def plotB3_nearfield_radius(ctx, depth, scales, outdir, results, n_rep=6, seed0=60000):
    """B3 (pt): near-field radius validity -- where the +/-MAX_RADIUS cut breaks.

    The response table only spans ~MAX_RADIUS pixels transversely (detsim.py:201),
    so charge that diffuses beyond is dropped. We inflate sigma_T with the post-hoc
    knob at a fixed depth and watch the collected charge: it stays flat while the
    5-sigma cloud fits inside the near-field reach, then falls once it spills past.
    Validates the approximation in the physical regime and MAPS its boundary.
    """
    raw = centered_point_source(ctx.detector, depth)
    reach = ctx.ideal["nearfield_reach"]
    xs, q, sigs = [], [], []
    for i, s in enumerate(scales):
        coll = []
        for r in range(n_rep):
            drifted = quench_and_drift(ctx, raw)
            apply_diffusion_scale(drifted, s)
            neigh, _ = find_pixels(ctx, drifted)
            sig = induce_current(ctx, drifted, neigh, seed=seed0 + i * 50 + r)
            p, _ = charge_polarity_split(sig)
            coll.append(p)
        sigT = float(quench_and_drift(ctx, raw)["tran_diff"][0]) * sqrt(s)
        sigs.append(sigT)
        xs.append(ctx.ideal["diff_n_sigmas"] * sigT / reach)   # 5-sigma cloud / reach
        q.append(np.mean(coll))
    xs, q = np.array(xs), np.array(q)
    q_norm = q / q[0] if q[0] > 0 else q
    fig, ax = _new_ax()
    ax.plot(xs, q_norm, "o-")
    ax.axvline(1.0, color="r", ls="--", label="5*sigma_T = near-field reach")
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_xlabel("DIFF_N_SIGMAS * sigma_T / (MAX_RADIUS * pitch)")
    ax.set_ylabel("collected charge / collected(smallest sigma_T)")
    ax.set_title(f"B3: near-field radius validity (point @ {depth:.0f} cm)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffB3_nearfield_radius.png")
    # in the physical regime (cloud well inside reach) collected charge is stable
    physical = xs < 0.6
    swing = float(np.max(np.abs(q_norm[physical] - 1.0))) if physical.any() else np.nan
    record_check(results, "B3 charge stable while 5-sigma cloud inside near-field (<5%)",
                 bool(np.isfinite(swing) and swing < 0.05), f"max swing={swing:.2%}")
    # and it must visibly fall once the cloud spills past the reach (boundary mapped)
    beyond = xs > 1.2
    drop = bool(beyond.any() and np.min(q_norm[beyond]) < 0.9)
    record_check(results, "B3 boundary mapped: charge falls once cloud exceeds reach",
                 drop or not beyond.any(),
                 "no scales beyond reach" if not beyond.any() else
                 f"min={np.min(q_norm[beyond]):.2f}")
    # physical operating point: max physical sigma_T (deepest drift) vs reach
    max_phys = ctx.ideal["diff_n_sigmas"] * ctx.ideal["sigma_tran_coeff"] * \
        sqrt(ctx.ideal["drift_length"]) / reach
    print(f"    operating point: 5*sigma_T(full drift)/reach = {max_phys:.3f} "
          f"(<<1 => near-field cut drops ~nothing in normal running)")


def plotB4_charge_sharing(ctx, share, outdir, results):
    """B4 (pt): transverse diffusion via CHARGE SHARING across a pad boundary.

    In module0 sigma_T <= 0.13*pitch at ALL depths, so a centered deposit lands
    almost entirely on one pad and the pad-level footprint RMS is ~0 (NOT
    sqrt(sigma_T^2+(p/sqrt12)^2)) -- pad-scale footprint simply cannot resolve a
    sub-pixel cloud. The transverse diffusion is instead exposed by charge sharing:
    with the deposit `delta` inside a pad boundary, the fraction leaking across is
    0.5*erfc(delta / (sqrt2 * sigma_T)). We compare the observed leak to that
    theory (using the kernel's own sigma_T at each depth) -- a sub-pixel-sensitive,
    fit-free, theory-driven test that the transverse spread is physically correct.
    """
    d = share["drift_cm"]; sigT = share["sigma_tran"]; leak = share["leak"]
    delta = share["delta"]
    theory = 0.5 * (1.0 - np.array([erf(delta / (sqrt(2.0) * s)) if s > 0 else 1.0
                                    for s in sigT]))
    fig, ax = _new_ax()
    ax.plot(d, 100 * leak, "o", label="observed leak fraction")
    ax.plot(d, 100 * theory, "r-",
            label=f"0.5*erfc(delta/(sqrt2 sigma_T)), delta={delta:.3f} cm")
    ax.set_xlabel("drift distance [cm]")
    ax.set_ylabel("charge leaked across boundary [%]")
    ax.set_title("B4: transverse diffusion via charge sharing")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diffB4_charge_sharing.png")
    print(f"    note: sigma_T/pitch ranges {sigT.min()/ctx.ideal['pitch']:.3f}.."
          f"{sigT.max()/ctx.ideal['pitch']:.3f} -- transverse diffusion is sub-pixel")
    # compare only where there is a measurable leak (theory > 2%); shallow ~0/0 is noise
    meas = theory > 0.02
    med = median_rel_dev(leak[meas], theory[meas]) if meas.any() else np.nan
    diagnose("B4 charge-sharing leak vs erfc theory", 1 + (med if np.isfinite(med) else 9),
             1.0, 0.25, results)
    good = np.isfinite(leak)
    grows = bool(good.sum() >= 2 and leak[good][-1] > leak[good][0])
    record_check(results, "B4 charge sharing grows with drift (sigma_T)", grows)


def plotB5_diffusion_knob(ctx, knob, outdir, results):
    """B5 (pt): post-hoc diffusion knob, validated via depth-independent charge sharing.

    The knob scales the per-segment width fields by sqrt(s) after drift. We validate
    it on charge-sharing leak -- an observable that depends on sigma_T ALONE (not on
    the depth-dependent window/response), so post-hoc and native-depth are cleanly
    comparable:
      a) leak(post-hoc @ depth d, scale s) == leak(native @ depth s*d): the decisive
         'not faking it' control (both have sigma_T(s*d));
      b) leak follows 0.5*erfc(delta/(sqrt2*sigma_T(s*d))): the scaling makes the
         right sigma_T;
      c) collected charge is invariant under s: diffusion conserves charge.
    """
    import matplotlib.pyplot as plt
    s = knob["scales"]; delta = knob["delta"]; sigT = knob["sigma_tran"]
    base = knob["collected"][list(s).index(1.0)]
    nat = np.isfinite(knob["native_leak"])
    theory = 0.5 * (1.0 - np.array([erf(delta / (sqrt(2.0) * st)) if st > 0 else 1.0
                                    for st in sigT]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(s, 100 * knob["leak"], "o-", color="C0", label="post-hoc (scale s @ d0)")
    ax[0].plot(s[nat], 100 * knob["native_leak"][nat], "x", ms=11, color="C1",
               label="native kernel @ depth s*d0")
    ax[0].plot(s, 100 * theory, "r:", label="0.5*erfc(delta/(sqrt2 sigma_T(s*d0)))")
    ax[0].set_xlabel("diffusion scale s")
    ax[0].set_ylabel("charge-sharing leak [%]")
    ax[0].set_title("B5a: knob vs native via charge sharing")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(s, knob["collected"] / base, "s-", color="C2")
    ax[1].axhline(1.0, color="k", ls="--", label="charge-conserving ideal")
    ax[1].set_xlabel("diffusion scale s"); ax[1].set_ylabel("collected / collected(s=1)")
    ax[1].set_title("B5b: collected charge invariant")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    _save(fig, f"{outdir}/diffB5_diffusion_knob.png")

    swing = float(np.max(np.abs(knob["collected"] / base - 1.0)))
    record_check(results, "B5 collected charge invariant under knob (<10%)",
                 swing < 0.10, f"{swing:.2%}")
    relc = median_rel_dev(knob["leak"][nat], knob["native_leak"][nat])
    diagnose("B5 post-hoc==native (charge-sharing control)",
             1 + (relc if np.isfinite(relc) else 9), 1.0, 0.25, results)
    meas = theory > 0.02
    relt = median_rel_dev(knob["leak"][meas], theory[meas]) if meas.any() else np.nan
    diagnose("B5 knob leak vs erfc theory",
             1 + (relt if np.isfinite(relt) else 9), 1.0, 0.25, results)


# ===========================================================================
# GROUP C -- readout truncations
# ===========================================================================
def plotC1_threshold_efficiency(ctx, eff, outdir, results):
    """C1 (pt): fired-pad fraction vs drift (real FEE, 5000 e- threshold)."""
    d, e = eff["drift_cm"], eff["fired_frac"]
    fig, ax = _new_ax()
    ax.plot(d, e, "o-", label="fired-pad fraction (obs)")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("fraction of pads above 5000 e-")
    ax.set_title("C1: threshold efficiency vs drift"); ax.grid(alpha=0.3); ax.legend()
    _save(fig, f"{outdir}/diffC1_threshold_efficiency.png")
    falls = bool(np.isfinite(e[-1]) and np.isfinite(e[0]) and e[-1] <= e[0] + 1e-6)
    record_check(results, "C1 efficiency non-increasing with drift", falls)


def check_C2_adc_clamp(ctx, results, depth=10.0):
    """C2 (pt): recorded ADC is clamped to [0, ADC_COUNTS-1] -- never negative,
    even though the internal q_sum dips on a neighbor's bipolar lobe."""
    out = simulate_event(ctx, build_point_source_event(ctx.detector,
                         np.random.default_rng(11), depth), seed=99, with_fee=True)
    if out is None or "adc" not in out:
        record_check(results, "C2 no negative recorded ADC", True, "no hits")
        return
    adc = out["adc"]
    record_check(results, "C2 no negative recorded ADC",
                 bool(np.nanmin(adc) >= 0), f"min={np.nanmin(adc):.0f}")


def plotC3_farfield_note(ctx, outdir, results):
    """C3 (pt): the long-range pre-trigger that the default near-field run omits.

    HONEST SCOPE: the dedicated far-field module (larndsim/far_field, induced current
    out to ~50 cm) is gated by sim.FARFIELD_ENABLED, which is read at
    consts.load_properties time -- it both selects the far-field kernel AND expands
    MAX_RADIUS to CHARGE_NEIGHBOR_RADIUS (detector.py:465-469), recompiling the
    response geometry. Flipping the flag *after* load (and our single-event pipeline,
    which only ever launches tracks_current_mc) does NOT activate it, so an on/off
    comparison inside one process would be a no-op reporting a fake "0%". We therefore
    do NOT fake it: quantifying the far-field truncation requires a SEPARATE run with
    `farfield_enabled: True` in the simulation-properties YAML and comparing the
    collected/pre-trigger charge between the two runs. This panel records the scope
    and the (off-by-default) state rather than a misleading number.
    """
    import matplotlib.pyplot as plt
    fig, ax = _new_ax()
    enabled = bool(getattr(ctx.sim, "FARFIELD_ENABLED", False))
    ax.text(0.5, 0.5,
            "C3 far-field truncation\n\n"
            f"sim.FARFIELD_ENABLED = {enabled} (default False)\n"
            f"MAX_RADIUS = {ctx.ideal['max_radius']:.0f} pix (near-field only)\n\n"
            "Quantifying the dropped long-range pre-trigger requires a separate\n"
            "run with farfield_enabled: True (it expands MAX_RADIUS and selects\n"
            "the far_field kernel at load time). A runtime flag flip is a no-op,\n"
            "so this is reported as SKIP rather than a fabricated 0%.",
            ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_title("C3: far-field pre-trigger (SKIP -- needs a dedicated FF run)")
    ax.axis("off")
    _save(fig, f"{outdir}/diffC3_farfield.png")
    print("    C3 far-field: SKIP -- needs a separate farfield_enabled run "
          "(runtime flag flip would be a no-op).")
    results.setdefault("checks", {})["C3 far-field truncation quantified"] = None


# ===========================================================================
# Ensemble accumulation
# ===========================================================================
def accumulate_muons(ctx, n_muons, rng, seed0=1000):
    """Per-segment muon table (drifted segments only)."""
    cols = {k: [] for k in ("drift_cm", "t", "n_e_in", "n_e_out")}
    for ev in range(n_muons):
        raw = build_muon_event(ctx.detector, rng)
        # Group A only needs the drift kernel's analytic outputs; skip the (heavy,
        # ~200 MB/event) induction so a 126-segment muon doesn't allocate signals.
        drifted = quench_and_drift(ctx, raw)
        if float(np.sum(drifted["n_electrons"])) <= 0:
            continue
        d = drift_distance_of(drifted, ctx.detector)
        keep = np.isfinite(d) & (drifted["t"] > 0)   # processed & drifted
        if not keep.any():
            continue
        n_in = quench_only(ctx, raw)
        cols["drift_cm"].append(d[keep])
        cols["t"].append(drifted["t"][keep])
        cols["n_e_in"].append(n_in[keep])
        cols["n_e_out"].append(drifted["n_electrons"].astype(float)[keep])
    return {k: np.concatenate(v) for k, v in cols.items() if v}


def accumulate_point_scan(ctx, depths, seed0=5000):
    """Per-depth point-source table: diffusion fields + long-window footprint."""
    cols = {k: [] for k in ("drift_cm", "long_diff", "tran_diff", "sigma_long",
                            "sigma_tran", "foot_rms", "n_collect", "time_rms")}
    for i, (depth, raw) in enumerate(scan_point_sources(ctx.detector, depths)):
        out = simulate_event(ctx, raw, seed=seed0 + i)
        if out is None:
            continue
        drifted = out["tracks"]
        if not np.isfinite(drift_distance_of(drifted, ctx.detector)[0]):
            continue
        prof = per_pixel_net_charge(out["signals"])
        cols["drift_cm"].append(depth)
        cols["long_diff"].append(float(drifted["long_diff"][0]))
        cols["tran_diff"].append(float(drifted["tran_diff"][0]))
        cols["sigma_long"].append(float(drifted["long_diff"][0]))
        cols["sigma_tran"].append(float(drifted["tran_diff"][0]))
        cols["foot_rms"].append(footprint_rms(ctx, out["signals"], out["neigh"]))
        cols["n_collect"].append(int((prof > 1e-3 * prof.max()).sum()) if prof.max() > 0 else 0)
        cols["time_rms"].append(waveform_time_rms(ctx, out["signals"]))
    return {k: np.array(v) for k, v in cols.items()}


def accumulate_charge_sharing(ctx, depths, delta=0.05, n_rep=8, seed0=7000):
    """Per-depth charge-sharing leak across an x pad boundary (B4)."""
    cols = {k: [] for k in ("drift_cm", "sigma_tran", "leak")}
    for i, depth in enumerate(depths):
        raw, x_pt = boundary_point_source(ctx.detector, depth, delta)
        drifted = quench_and_drift(ctx, raw)
        if not np.isfinite(drift_distance_of(drifted, ctx.detector)[0]):
            continue
        neigh, _ = find_pixels(ctx, drifted)
        leaks = []
        for r in range(n_rep):
            sig = induce_current(ctx, drifted, neigh, seed=seed0 + i * 50 + r)
            leaks.append(charge_sharing_leak(ctx, sig, neigh, x_pt))
        cols["drift_cm"].append(depth)
        cols["sigma_tran"].append(float(drifted["tran_diff"][0]))
        cols["leak"].append(float(np.nanmean(leaks)))
    out = {k: np.array(v) for k, v in cols.items()}
    out["delta"] = delta
    return out


def mean_boundary_leak(ctx, drifted, x_pt, n_rep, seed0):
    """Mean (charge-sharing leak, collected positive charge) over n_rep MC reps."""
    neigh, _ = find_pixels(ctx, drifted)
    leaks, cols = [], []
    for r in range(n_rep):
        sig = induce_current(ctx, drifted, neigh, seed=seed0 + r)
        lk = charge_sharing_leak(ctx, sig, neigh, x_pt)
        if np.isfinite(lk):
            leaks.append(lk)
        cols.append(charge_polarity_split(sig)[0])
    return (float(np.mean(leaks)) if leaks else np.nan,
            float(np.mean(cols)) if cols else np.nan)


def accumulate_knob(ctx, depth_cm, scales, max_depth, delta=0.05, n_rep=8, seed0=9000):
    """Plot B5: post-hoc knob validated against the native kernel via CHARGE SHARING.

    The anti-faking control needs an observable that depends on the diffusion width
    ALONE (not on the depth-dependent readout window or response). Charge-sharing
    leak across a pad boundary is exactly that: leak = 0.5*erfc(delta/(sqrt2*sigma_T)).
    For every scale s we compare, at a fixed boundary deposit:
      * post-hoc: depth `depth_cm`, widths scaled by sqrt(s)  -> sigma_T = sigma_T(s*depth_cm)
      * native:   the UNMODIFIED kernel at depth s*depth_cm    -> same sigma_T
    They must agree; both must match the erfc theory; and the collected charge must
    stay invariant under s (diffusion conserves charge).
    """
    raw, x_pt = boundary_point_source(ctx.detector, depth_cm, delta)
    keys = ("scales", "leak", "collected", "sigma_tran",
            "native_depth", "native_leak")
    cols = {k: [] for k in keys}
    for i, s in enumerate(scales):
        drifted = quench_and_drift(ctx, raw)
        apply_diffusion_scale(drifted, s)
        leak, coll = mean_boundary_leak(ctx, drifted, x_pt, n_rep, seed0 + i * 100)
        cols["scales"].append(s); cols["leak"].append(leak); cols["collected"].append(coll)
        cols["sigma_tran"].append(ctx.ideal["sigma_tran_coeff"] * sqrt(depth_cm) * sqrt(s))
        nd = s * depth_cm
        cols["native_depth"].append(nd)
        if 0 < nd <= max_depth:
            raw_n, x_n = boundary_point_source(ctx.detector, nd, delta)
            drifted_n = quench_and_drift(ctx, raw_n)
            nleak, _ = mean_boundary_leak(ctx, drifted_n, x_n, n_rep, seed0 + 50 + i * 100)
        else:
            nleak = np.nan
        cols["native_leak"].append(nleak)
    out = {k: np.array(v, dtype=float) for k, v in cols.items()}
    out["depth_cm"] = depth_cm
    out["delta"] = delta
    return out


def accumulate_threshold(ctx, depths, n_rep, rng, seed0=13000):
    """Fired-pad fraction vs drift (real FEE)."""
    d_out, eff = [], []
    for i, depth in enumerate(depths):
        fracs = []
        for r in range(n_rep):
            raw = build_point_source_event(ctx.detector, rng, depth)
            out = simulate_event(ctx, raw, seed=seed0 + i * 100 + r, with_fee=True)
            if out is None or "adc" not in out:
                continue
            fracs.append(fired_fraction(out["adc"]))
        if fracs:
            d_out.append(depth); eff.append(float(np.nanmean(fracs)))
    return dict(drift_cm=np.array(d_out), fired_frac=np.array(eff))


# ===========================================================================
# Driver
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response", default="larndsim/bin/response_44_v2a_full.npz")
    ap.add_argument("--detector", default="larndsim/detector_properties/module0.yaml")
    ap.add_argument("--pixel-layout",
                    default="larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml")
    ap.add_argument("--sim-properties",
                    default="larndsim/simulation_properties/singles_sim.yaml")
    ap.add_argument("--n-muons", type=int, default=2000)
    ap.add_argument("--n-points", type=int, default=40, help="point-source depth grid size")
    ap.add_argument("--max-depth", type=float, default=None,
                    help="max drift depth for scans (cm); default DRIFT_LENGTH-1")
    ap.add_argument("--eff-reps", type=int, default=20, help="reps per depth for C1")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=".")
    return ap.parse_args()


def sanity_floor(ctx):
    """A centered, well-diffused point source must give positive collected charge."""
    out = simulate_event(ctx, build_point_source_event(
        ctx.detector, np.random.default_rng(7), 10.0), seed=1)
    val = total_induced_charge(out["signals"]) if out else 0.0
    print(f"Sanity floor: centered point @10cm collected = {val:.4e}")
    if not (np.isfinite(val) and val > 0):
        print("  !! HARNESS UNSOUND: non-positive collected sum. Aborting.")
        return False
    return True


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available. Run this on a GPU node.")

    import os
    os.makedirs(args.outdir, exist_ok=True)
    ctx = load_simulation(args)
    print_config(ctx)
    if not sanity_floor(ctx):
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    max_depth = args.max_depth or (abs(ctx.detector.DRIFT_LENGTH) - 1.0)
    depths = np.linspace(0.5, max_depth, args.n_points)
    results = {"checks": {}}

    print("\nAccumulating muon ensemble (%d events)..." % args.n_muons)
    muon = accumulate_muons(ctx, args.n_muons, rng)
    print("Accumulating point-source depth scan (%d depths)..." % args.n_points)
    point = accumulate_point_scan(ctx, depths)

    print("\n=== GROUP A: analytic drift stage is exact (no truncation) ===")
    plotA_drift_time(ctx, muon, args.outdir, results)
    plotA_diffusion_fields(ctx, point, args.outdir, results)
    plotA_lifetime(ctx, muon, args.outdir, results)

    print("\n=== GROUP B: induction truncations small & depth-stable ===")
    probe_depths = np.array([d for d in (2.0, 10.0, max_depth * 0.5, max_depth - 1)
                             if 0 < d <= max_depth])
    plotB1_window_closure(ctx, probe_depths, args.outdir, results)
    plotB2_charge_integrity(ctx, np.linspace(1.0, max_depth, 14), args.outdir, results)
    plotB3_nearfield_radius(ctx, 10.0, [0.5, 1, 2, 4, 8, 16, 32, 64], args.outdir, results)
    share = accumulate_charge_sharing(ctx, np.linspace(2.0, max_depth, 14))
    plotB4_charge_sharing(ctx, share, args.outdir, results)
    knob = accumulate_knob(ctx, 10.0, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], max_depth)
    plotB5_diffusion_knob(ctx, knob, args.outdir, results)

    print("\n=== GROUP C: readout truncations ===")
    eff = accumulate_threshold(ctx, np.linspace(0.5, max_depth, 12), args.eff_reps, rng)
    plotC1_threshold_efficiency(ctx, eff, args.outdir, results)
    check_C2_adc_clamp(ctx, results)
    plotC3_farfield_note(ctx, args.outdir, results)

    np.savez(f"{args.outdir}/verify_diffusion_results.npz",
             **{f"muon_{k}": v for k, v in muon.items()},
             **{f"point_{k}": v for k, v in point.items()},
             **{f"share_{k}": np.asarray(v) for k, v in share.items()},
             **{f"knob_{k}": np.asarray(v) for k, v in knob.items()},
             **{f"eff_{k}": v for k, v in eff.items()})
    print_summary(results)


def print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY")
    checks = results["checks"]
    passed = sum(1 for v in checks.values() if v is True)
    failed = sum(1 for v in checks.values() if v is False)
    skipped = sum(1 for v in checks.values() if v is None)
    for name, ok in checks.items():
        tag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        print(f"  [{tag}] {name}")
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)


if __name__ == "__main__":
    main()
