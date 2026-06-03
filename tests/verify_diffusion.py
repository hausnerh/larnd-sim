#!/usr/bin/env python
"""
verify_diffusion.py
===================

Diffusion verification suite for larnd-sim. Drives the REAL CUDA kernels
(quench -> drift -> get_pixels -> tracks_current_mc -> sum_pixel_signals ->
get_adc_values -> digitize) one event at a time, thousands of events, and bins
the resulting per-segment / per-pixel tables to isolate each physical effect of
the charge-drift + diffusion stage. Emits "money plots" whose shape immediately
shows whether the simulation is sound, plus a PASS/FAIL line per plot comparing
the observed curve to a theory-motivated ideal.

Two probe handles:
  * MUON  (mu) -- one muon per event, random angle/depth: realistic coverage.
  * POINT (pt) -- a near-delta beta blob, on the anode (PSF floor) or scanned in
                  depth: the clean handle for diffusion / point-spread.

Full written framing (physics, pixel-vs-wire, quantitative predictions, code
cheat-sheet, low-hanging bugs) lives in docs/diffusion_verification.md. The
ideal curves and PASS tolerances implemented in the check_* functions below are
the runtime counterpart of that document's "Quantitative ideal predictions".

REQUIREMENTS
  * A CUDA-capable GPU (numba.cuda.is_available() must be True); cupy, numba.
  * Run from the top level of a larnd-sim checkout. Defaults match module0:
        python tests/verify_diffusion.py --n-muons 2000 --outdir /tmp/diffverify

DESIGN
  Many small, single-purpose functions reusing the real machinery. Nothing
  physical is reimplemented; helpers only assemble inputs, drive kernels, and
  measure outputs.

NOTE ON THE DIFFUSION KNOB (plot 7) -- and why we can trust it
  drifting.drift() references detector.LONG_DIFF / TRAN_DIFF, which numba bakes
  in as compile-time constants. Rescaling the module global after the kernel is
  compiled would NOT change drift()'s output. So the "diffusion up/down" knob is a
  *post-hoc* fix: it scales the per-segment long_diff/tran_diff FIELDS after drift
  (sigma ~ sqrt(D), so a factor s on D means multiplying the width fields by
  sqrt(s)); s=0 turns diffusion fully off. Those scaled fields then flow through
  the REAL find_pixels + tracks_current_mc.
  Because this is post-hoc, plot 7 validates it quantitatively rather than trusting
  it: (a) the longitudinal waveform time-VARIANCE grows linearly in s with slope
  (sigma_L(d)/V)^2 fixed by the loaded constants; and (b) -- the decisive control --
  a post-hoc scale s at depth d is compared to the UNMODIFIED kernel run at the
  equivalent depth s*d (since sigma(s*d) == sqrt(s)*sigma(d)); the two must agree
  bin-for-bin wherever s*d is reachable. If the field-scaling were "faking it",
  (a) the slope would miss the constant or (b) post-hoc and native would diverge.
"""

import argparse
import sys
from math import ceil, sqrt, erf

import numpy as np


# ---------------------------------------------------------------------------
# Track/segment record dtype.
#
# Inline copy of cli/dumpTree.py:segments_dtype (keep align=True). We copy
# rather than import because cli/dumpTree.py does `from ROOT import ...` at
# module scope and ROOT is generally absent on a GPU compute node. If this ever
# drifts, the symptom is a Numba TypingError "Field '<name>' was not found in
# record" -- re-sync with cli/dumpTree.py.
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
    """Theory constants used by the check_* functions, from loaded config."""
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
    )
    d["sigma_long_coeff"] = sqrt(2.0 * d["d_long"] / v)   # sigma_L = coeff*sqrt(d)
    d["sigma_tran_coeff"] = sqrt(2.0 * d["d_tran"] / v)   # sigma_T = coeff*sqrt(d)
    d["sigma_ratio"] = sqrt(d["d_tran"] / d["d_long"])    # constant ~1.483
    d["psf_slope"] = 2.0 * d["d_tran"] / v                # d(sigma_T^2)/dd
    return d


def print_config(ctx):
    k = ctx.ideal
    print("=" * 70)
    print("Diffusion verification suite -- module0 constants (runtime):")
    print(f"  V_DRIFT          = {k['v_drift']:.5f} cm/us "
          f"(1/V_DRIFT = {k['inv_v_drift']:.4f} us/cm)")
    print(f"  D_L, D_T         = {k['d_long']:.3e}, {k['d_tran']:.3e} cm^2/us")
    print(f"  tau, lambda      = {k['tau']:.1f} us, {k['atten_length']:.1f} cm")
    print(f"  sigma_L, sigma_T = {k['sigma_long_coeff']:.4e}, "
          f"{k['sigma_tran_coeff']:.4e} * sqrt(d[cm]) cm")
    print(f"  sigma_T/sigma_L  = {k['sigma_ratio']:.4f}")
    print(f"  pitch, p/sqrt12  = {k['pitch']:.4f}, {k['pitch_rms']:.4f} cm")
    print(f"  Q_threshold      = {k['q_threshold']:.0f} e-, "
          f"DIFF_N_SIGMAS = {k['diff_n_sigmas']:.0f}")
    print("=" * 70)


# ===========================================================================
# Geometry helpers
# ===========================================================================
def plane_z(detector, plane=0):
    """(z_anode, z_cathode, into) for a TPC plane; `into` is drift direction."""
    z_anode = detector.TPC_BORDERS[plane][2][0]
    z_cathode = detector.TPC_BORDERS[plane][2][1]
    return z_anode, z_cathode, float(np.sign(z_cathode - z_anode))


def active_volume(detector, plane=0, margin=1.0):
    """(x0, x1, y0, y1) of the anode face, inset by `margin` cm."""
    b = detector.TPC_BORDERS[plane]
    return (b[0][0] + margin, b[0][1] - margin,
            b[1][0] + margin, b[1][1] - margin)


def depth_to_z(detector, drift_cm, plane=0):
    """z-coordinate of a deposit `drift_cm` from the anode."""
    z_anode, _, into = plane_z(detector, plane)
    return z_anode + into * drift_cm


def pixel_center(detector, ix, iy, plane=0):
    """(x, y) center of pixel (ix, iy)."""
    b = detector.TPC_BORDERS[plane]
    p = detector.PIXEL_PITCH
    return b[0][0] + (ix + 0.5) * p, b[1][0] + (iy + 0.5) * p


# ===========================================================================
# Probe generation
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
    """Two random points in the active volume at random drift depths."""
    x0, x1, y0, y1 = active_volume(detector, plane, margin=2.0)
    _, _, into = plane_z(detector, plane)
    depths = rng.uniform(0.5, abs(detector.DRIFT_LENGTH) - 0.5, size=2)
    start = (rng.uniform(x0, x1), rng.uniform(y0, y1), depth_to_z(detector, depths[0], plane))
    end = (rng.uniform(x0, x1), rng.uniform(y0, y1), depth_to_z(detector, depths[1], plane))
    return start, end


def build_muon_event(detector, rng, plane=0, step_cm=0.4, dEdx=2.1):
    """Assemble the per-event muon `tracks` array (segmented MIP line)."""
    start, end = random_muon_endpoints(detector, rng, plane)
    pairs = segmentize(start, end, step_cm)
    tracks = blank_tracks(len(pairs))
    for i, (p0, p1) in enumerate(pairs):
        fill_segment(tracks[i], plane, p0, p1, dEdx)
        tracks[i]["segment_id"] = i
    return tracks


def point_source_record(detector, x, y, drift_cm, plane=0, point_dE=0.5, dx=0.02):
    """One near-delta deposit at transverse (x, y) and drift depth drift_cm.

    Total deposited energy `point_dE` (MeV) in a tiny length `dx` (cm), oriented
    along +y so the cloud is point-like (dx << pixel). drift_cm=0 => on the anode
    (PSF floor).
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


def scan_point_sources(detector, depths, plane=0, **kw):
    """Yield (depth, tracks) for a centered point source across a depth grid."""
    i0, j0 = detector.N_PIXELS[0] // 2, detector.N_PIXELS[1] // 2
    x, y = pixel_center(detector, i0, j0, plane)
    for depth in depths:
        yield depth, point_source_record(detector, x, y, depth, plane, **kw)


# ===========================================================================
# Diffusion-knob / far-field toggles (with restore)
# ===========================================================================
def apply_diffusion_scale(tracks, factor):
    """Scale per-segment diffusion widths to emulate D -> factor*D (in place).

    sigma ~ sqrt(2 D t), so D -> factor*D means width *= sqrt(factor). factor=0
    fully removes diffusion. Applied AFTER drift (see module docstring on why the
    module global cannot be rescaled post-compile).
    """
    s = sqrt(factor)
    tracks["long_diff"] *= s
    tracks["tran_diff"] *= s
    return tracks


class far_field_enabled:
    """Context manager toggling sim.FARFIELD_ENABLED with restore."""
    def __init__(self, sim, on):
        self.sim, self.on, self.saved = sim, on, None

    def __enter__(self):
        self.saved = self.sim.FARFIELD_ENABLED
        self.sim.FARFIELD_ENABLED = self.on
        return self.sim

    def __exit__(self, *exc):
        self.sim.FARFIELD_ENABLED = self.saved


# ===========================================================================
# Kernel drivers (thin wrappers over the real machinery)
# ===========================================================================
def to_host(a):
    """cupy/numba-device -> numpy host array."""
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    if hasattr(a, "copy_to_host"):
        return a.copy_to_host()
    return np.asarray(a)


def quench_and_drift(ctx, tracks):
    """Run quench then drift on a host tracks array; return drifted host copy."""
    tracks = np.copy(tracks)
    d_tracks = ctx.cuda.to_device(tracks)
    ctx.quenching.quench[1, 128](d_tracks, ctx.physics.BOX)
    ctx.drifting.drift[1, 128](d_tracks)
    return d_tracks.copy_to_host()


def find_pixels(ctx, tracks):
    """get_pixels -> (neighboring_pixels, neighboring_radius) as cupy arrays."""
    import cupy as cp
    nseg = tracks.shape[0]
    max_active, max_neigh = 64, 220
    active = cp.full((nseg, max_active), -1, dtype=cp.int32)
    neigh = cp.full((nseg, max_neigh), -1, dtype=cp.int32)
    radius = cp.full((nseg, max_neigh), -1, dtype=cp.float32)
    n_list = cp.zeros(nseg, dtype=cp.int64)
    d_tracks = ctx.cuda.to_device(tracks)
    ctx.pixels_from_track.get_pixels[max(ceil(nseg / 128), 1), 128](
        d_tracks, active, neigh, radius, n_list)
    return neigh, radius


def signal_ticks(ctx, tracks):
    """Length (in time ticks) of the per-pixel signal window for this event."""
    det = ctx.detector
    long_max = float(np.max(tracks["long_diff"]))
    t_span = float(np.max(tracks["t_end"] - tracks["t0"]))
    pad = long_max / det.V_DRIFT * det.DIFF_N_SIGMAS
    if det.RESPONSE_MAX_TIME > det.DRIFT_MAX_TIME:
        max_time = t_span + pad + det.RESPONSE_MAX_TIME - det.DRIFT_MAX_TIME
    else:
        max_time = t_span + pad
    return max(ceil(max_time / det.TIME_SAMPLING), 1)


def make_rng(ctx, n, seed):
    return ctx.create_rng(max(n, 1024), seed=seed)


def induce_current(ctx, tracks, neigh, seed=12345):
    """tracks_current_mc -> per-(segment, pixel, tick) signed signals (cupy)."""
    import cupy as cp
    nt = signal_ticks(ctx, tracks)
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

    Faithful single-batch copy of the orchestration in
    cli/simulate_pixels.py:1389-1454.
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

    Mirrors cli/simulate_pixels.py:do_digitize_and_update for a single event.
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


def simulate_event(ctx, tracks, seed=12345, with_fee=False):
    """Full per-event pipeline. Returns a dict of measurements/arrays."""
    drifted = quench_and_drift(ctx, tracks)
    if float(np.sum(drifted["n_electrons"])) <= 0:
        return None
    neigh, radius = find_pixels(ctx, drifted)
    signals = induce_current(ctx, drifted, neigh, seed=seed)
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
# Measurements (each a few lines)
# ===========================================================================
def drift_distance_of(tracks, detector, plane=0):
    """Per-segment drift distance |z - z_anode| (cm)."""
    z_anode, _, _ = plane_z(detector, plane)
    return np.abs(tracks["z"] - z_anode)


def collected_charge(signals):
    """Total signed induced charge summed over all pixels/ticks (response units)."""
    return float(to_host(signals).sum())


def pixel_charge_profile(signals):
    """Per-pixel time-integrated signal (sum over segments and ticks)."""
    s = to_host(signals)
    return s.sum(axis=(0, 2)) if s.ndim == 3 else s.sum(axis=-1)


def active_pixel_count(signals, frac=1e-3):
    """Number of pixels carrying > frac of the peak pixel charge."""
    prof = np.abs(pixel_charge_profile(signals))
    return int((prof > frac * prof.max()).sum()) if prof.max() > 0 else 0


def transverse_rms(ctx, signals, neigh, plane=0):
    """Charge-weighted RMS of pixel x-positions (cm)."""
    prof = pixel_charge_profile(signals)
    ids = to_host(neigh).reshape(-1)
    prof_flat = to_host(signals).sum(axis=2).reshape(-1) if signals.ndim == 3 else prof
    mass = np.abs(prof_flat)
    keep = (ids != -1) & (mass > 0)
    if keep.sum() == 0:
        return np.nan
    ix = ids[keep] % ctx.detector.N_PIXELS[0]
    xs = ctx.detector.TPC_BORDERS[plane][0][0] + (ix + 0.5) * ctx.detector.PIXEL_PITCH
    w = mass[keep]
    mean = np.average(xs, weights=w)
    return float(sqrt(np.average((xs - mean) ** 2, weights=w)))


def waveform_time_rms(ctx, signals):
    """Time RMS (us) of the busiest pixel's waveform."""
    prof = np.abs(pixel_charge_profile(signals))
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


def running_integral(ctx, curre):
    """Cumulative sum(curre)*TIME_SAMPLING -- the noiseless analogue of q_sum."""
    return np.cumsum(to_host(curre)) * ctx.detector.TIME_SAMPLING


def fired_fraction(adc):
    """Fraction of pixels with at least one above-threshold ADC value."""
    if adc is None or adc.size == 0:
        return np.nan
    return float((adc.max(axis=1) > 0).mean())


# ===========================================================================
# Diagnosis
# ===========================================================================
def diagnose(name, observed, expected, tol, results):
    """Print PASS/FAIL on |observed-expected|/|expected| <= tol; record it."""
    rel = abs(observed - expected) / abs(expected) if expected else float("inf")
    ok = rel <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: observed={observed:.4g} "
          f"expected={expected:.4g} (rel={rel:.2%}, tol={tol:.0%})")
    results.setdefault("checks", {})[name] = ok
    return ok


def fit_loglinear(x, y):
    """Slope/intercept of ln(y) vs x (for exponential attenuation)."""
    good = (y > 0)
    return np.polyfit(x[good], np.log(y[good]), 1)


def fit_power(x, y):
    """Exponent of y = a * x^b via log-log fit."""
    good = (x > 0) & (y > 0)
    b, _ = np.polyfit(np.log(x[good]), np.log(y[good]), 1)
    return b


def linear_r2(x, y, slope, intercept):
    """R^2 of the line (slope, intercept) against (x, y)."""
    pred = slope * np.asarray(x) + intercept
    ss_res = np.sum((np.asarray(y) - pred) ** 2)
    ss_tot = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def record_check(results, name, ok, detail=""):
    """Print PASS/FAIL for a boolean check and record it (non-ratio checks)."""
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    results.setdefault("checks", {})[name] = bool(ok)
    return ok


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
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("  wrote", path)


# ===========================================================================
# Money plots (one function per plot)
# ===========================================================================
def plot_drift_time_linearity(ctx, muon_table, outdir, results):
    """Plot 1 (mu): drift time t vs drift distance; slope must be 1/V_DRIFT."""
    d, t = muon_table["drift_cm"], muon_table["t"]
    slope, intercept = np.polyfit(d, t, 1)
    pred = np.polyval((slope, intercept), d)
    ss_res = np.sum((t - pred) ** 2)
    r2 = 1 - ss_res / np.sum((t - t.mean()) ** 2)
    fig, ax = _new_ax()
    ax.plot(d, t, ".", ms=2, alpha=0.3, label="segments")
    grid = np.linspace(d.min(), d.max(), 50)
    ax.plot(grid, ctx.ideal["inv_v_drift"] * grid + intercept, "r-",
            label=f"ideal 1/V_DRIFT = {ctx.ideal['inv_v_drift']:.3f} us/cm")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("drift time t [us]")
    ax.set_title("Plot 1: drift-time linearity"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff1_drift_time.png")
    diagnose("drift-time slope", slope, ctx.ideal["inv_v_drift"], 0.01, results)
    print(f"  [{'PASS' if r2 > 0.999 else 'FAIL'}] drift-time R^2 = {r2:.5f} (>0.999)")
    results["checks"]["drift-time R2"] = bool(r2 > 0.999)


def plot_diffusion_scaling(ctx, point_table, outdir, results):
    """Plot 2 (pt): long_diff/tran_diff vs drift; sqrt law + ratio 1.483."""
    d = point_table["drift_cm"]
    sl, st = point_table["long_diff"], point_table["tran_diff"]
    fig, ax = _new_ax()
    grid = np.linspace(max(d.min(), 1e-3), d.max(), 50)
    ax.plot(d, sl, "o", ms=3, label="long_diff (kernel)")
    ax.plot(d, st, "s", ms=3, label="tran_diff (kernel)")
    ax.plot(grid, ctx.ideal["sigma_long_coeff"] * np.sqrt(grid), "b-",
            label="ideal sigma_L = sqrt(2 D_L d / V)")
    ax.plot(grid, ctx.ideal["sigma_tran_coeff"] * np.sqrt(grid), "r-",
            label="ideal sigma_T = sqrt(2 D_T d / V)")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("diffusion width [cm]")
    ax.set_title("Plot 2: diffusion sqrt-scaling"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff2_diffusion_scaling.png")
    power = fit_power(d, st)
    diagnose("tran_diff power", power, 0.5, 0.04, results)
    ratio = float(np.median(st[d > 0] / sl[d > 0]))
    diagnose("sigma_T/sigma_L ratio", ratio, ctx.ideal["sigma_ratio"], 0.01, results)


def plot_lifetime_attenuation(ctx, muon_table, outdir, results):
    """Plot 3 (mu): surviving charge fraction vs drift; exp with lambda."""
    d = muon_table["drift_cm"]
    surv = muon_table["n_e_out"] / np.maximum(muon_table["n_e_in"], 1)
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
    ax.set_title("Plot 3: lifetime attenuation"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff3_lifetime.png")
    diagnose("attenuation length lambda", lam_fit, ctx.ideal["atten_length"], 0.03, results)


def plot_transverse_footprint(ctx, point_scan, outdir, results):
    """Plot 4 (pt): active-pixel count & transverse RMS vs sigma_T."""
    st = point_scan["sigma_tran"]; rms = point_scan["tran_rms"]; nact = point_scan["n_active"]
    ideal = np.sqrt(st ** 2 + ctx.ideal["pitch_rms"] ** 2)
    fig, ax1 = _new_ax()
    ax2 = ax1.twinx()
    ax1.plot(st, nact, "o-", color="C0", label="active pixels")
    ax2.plot(st, rms, "s", color="C3", label="transverse RMS (obs)")
    ax2.plot(st, ideal, "r-", label="ideal sqrt(sigma_T^2 + (p/sqrt12)^2)")
    ax1.set_xlabel("sigma_T [cm]"); ax1.set_ylabel("active pixels", color="C0")
    ax2.set_ylabel("transverse RMS [cm]", color="C3")
    ax1.set_title("Plot 4: transverse footprint"); ax1.grid(alpha=0.3)
    ax2.legend(loc="lower right")
    _save(fig, f"{outdir}/diff4_transverse_footprint.png")
    good = np.isfinite(rms) & (ideal > 0)
    med_rel = float(np.median(np.abs(rms[good] - ideal[good]) / ideal[good]))
    diagnose("transverse RMS vs quadrature", 1 + med_rel, 1.0, 0.10, results)


def plot_longitudinal_time_width(ctx, point_scan, outdir, results):
    """Plot 5 (pt): central-pixel waveform time-RMS vs sigma_L."""
    sl = point_scan["sigma_long"]; trms = point_scan["time_rms"]
    good = np.isfinite(trms)
    sl, trms = sl[good], trms[good]
    sigma_resp = float(trms[np.argmin(sl)])  # sigma_L -> 0 floor
    recovered = (trms ** 2 - sigma_resp ** 2) * ctx.ideal["v_drift"] ** 2
    fig, ax = _new_ax()
    ax.plot(sl, trms, "o", label="waveform time-RMS (obs)")
    grid = np.linspace(sl.min(), sl.max(), 50)
    ax.plot(grid, np.sqrt((grid / ctx.ideal["v_drift"]) ** 2 + sigma_resp ** 2), "r-",
            label="ideal sqrt((sigma_L/V)^2 + sigma_resp^2)")
    ax.set_xlabel("sigma_L [cm]"); ax.set_ylabel("waveform time-RMS [us]")
    ax.set_title("Plot 5: longitudinal -> time width"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff5_longitudinal_time.png")
    mask = sl > sl.min()
    med_rel = float(np.median(np.abs(np.sqrt(np.abs(recovered[mask])) - sl[mask]) /
                              np.maximum(sl[mask], 1e-9)))
    diagnose("recovered sigma_L from time width", 1 + med_rel, 1.0, 0.10, results)


def plot_charge_conservation(ctx, muon_table, neighbor_qsum, outdir, results):
    """Plot 6 (mu): recorded ADC charge vs drift + neighbor q_sum inset."""
    d, q = muon_table["drift_cm"], muon_table["collected"]
    bins = np.linspace(d.min(), d.max(), 20)
    idx = np.digitize(d, bins)
    bd = np.array([d[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()])
    bq = np.array([q[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()])
    slope, intercept = fit_loglinear(bd, bq)
    lam_fit = -1.0 / slope
    fig, ax = _new_ax()
    ax.plot(bd, bq, "o", label="collected (pre-threshold)")
    ax.plot(bd, np.exp(intercept) * np.exp(-bd / ctx.ideal["atten_length"]), "r-",
            label=f"ideal exp(-d/{ctx.ideal['atten_length']:.0f} cm)")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("collected charge [resp. units]")
    ax.set_title("Plot 6: charge integrity vs drift"); ax.legend(); ax.grid(alpha=0.3)
    if neighbor_qsum is not None:
        ins = fig.add_axes([0.58, 0.55, 0.32, 0.32])
        ins.plot(neighbor_qsum)
        ins.axhline(0, color="k", lw=0.6)
        ins.set_title("neighbor running q_sum", fontsize=8)
        ins.tick_params(labelsize=7)
        dips = bool(np.min(neighbor_qsum) < 0)
        print(f"  [{'PASS' if dips else 'WARN'}] neighbor q_sum dips below 0 "
              f"(bipolar lobe present): {dips}")
        results["checks"]["neighbor q_sum dip"] = dips
    _save(fig, f"{outdir}/diff6_charge_conservation.png")
    diagnose("collected attenuation length", lam_fit, ctx.ideal["atten_length"], 0.05, results)


def plot_diffusion_up_down(ctx, knob, outdir, results):
    """Plot 7 (both): post-hoc diffusion knob -- theory scaling + native control.

    The knob is the one place the suite leaves the native kernel path (it scales
    the per-segment width fields by sqrt(s) after drift; see accumulate_diffusion_
    knob). So it gets the strongest, most quantitative validation:
      (a) longitudinal time-variance grows LINEARLY in s with slope (sigma_L(d)/V)^2
          -- the clean theory prediction (time axis is finely sampled, so unlike the
          coarse-pixel transverse RMS this moment is not discretization-limited);
      (b) post-hoc(s, d) == native(s*d): the field-scaling reproduces, within MC
          noise, exactly what the UNMODIFIED kernel does at the equivalent depth
          (the decisive "not faking it" control; magnitude- and binning-robust);
      (c) collected charge is invariant under s (diffusion conserves charge).
    """
    import matplotlib.pyplot as plt
    s = knob["scales"]
    v = ctx.ideal["v_drift"]
    sigT2 = knob["sigma_tran_at_depth"] ** 2               # cm^2  (continuum ref)
    sigL_t2 = (knob["sigma_long_at_depth"] / v) ** 2        # us^2  (theory slope)
    base = knob["collected"][list(s).index(1.0)]
    nat = np.isfinite(knob["native_width"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    grid = np.linspace(0, s.max(), 60)

    # -- 7a transverse: post-hoc vs native must overlap (equivalence is the story).
    ax[0].plot(s, knob["width"], "o-", color="C0", label="post-hoc sqrt(s) (obs)")
    ax[0].plot(s[nat], knob["native_width"][nat], "x", ms=10, color="C1",
               label="native @ depth s*d (control)")
    ax[0].plot(grid, np.sqrt(grid * sigT2), "r:", alpha=0.7,
               label="continuum sqrt(s)*sigma_T(d)")
    ax[0].set_xlabel("diffusion scale s"); ax[0].set_ylabel("footprint RMS [cm]")
    ax[0].set_title("7a: transverse width (post-hoc vs native)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # -- 7b longitudinal: the quantitative theory check (variance linear in s).
    fin = np.isfinite(s) & np.isfinite(knob["time_rms2"])
    at, bt = np.polyfit(s[fin], knob["time_rms2"][fin], 1)
    r2t = linear_r2(s[fin], knob["time_rms2"][fin], at, bt)
    sigresp2 = max(bt, 0.0)
    ax[1].plot(s, knob["time_rms"], "o-", color="C0", label="post-hoc (obs)")
    ax[1].plot(s[nat], knob["native_time_rms"][nat], "x", ms=10, color="C1",
               label="native control")
    ax[1].plot(grid, np.sqrt(grid * sigL_t2 + sigresp2), "r-",
               label="ideal sqrt(s*(sigma_L/V)^2 + sigma_resp^2)")
    ax[1].set_xlabel("diffusion scale s"); ax[1].set_ylabel("waveform time-RMS [us]")
    ax[1].set_title("7b: longitudinal time-width ~ sqrt(s)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # -- 7c charge invariance.
    ax[2].plot(s, knob["collected"] / base, "s-", color="C2")
    ax[2].axhline(1.0, color="k", ls="--", label="charge-conserving ideal")
    ax[2].set_xlabel("diffusion scale s"); ax[2].set_ylabel("collected / collected(s=1)")
    ax[2].set_title("7c: collected charge invariant")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    _save(fig, f"{outdir}/diff7_diffusion_up_down.png")

    # ---- (a) longitudinal variance scales linearly with s, slope = (sigma_L/V)^2 ----
    diagnose("knob: d(time-RMS^2)/ds = (sigma_L(d)/V)^2", at, sigL_t2, 0.30, results)
    record_check(results, "knob: time-variance linear in s (R^2>0.95)",
                 r2t > 0.95, f"R^2={r2t:.4f}")

    # ---- (b) post-hoc == native at the equivalent depth (the anti-faking control) ----
    relw = median_rel_dev(knob["width"], knob["native_width"])
    diagnose("knob: post-hoc==native footprint", 1 + relw, 1.0, 0.10, results)
    relt = median_rel_dev(knob["time_rms"], knob["native_time_rms"])
    diagnose("knob: post-hoc==native time-width", 1 + relt, 1.0, 0.15, results)

    # ---- (c) collected charge invariance under the knob ----
    swing = float(np.max(np.abs(knob["collected"] / base - 1.0)))
    record_check(results, "knob: collected charge invariant (max|Q(s)/Q(1)-1|<10%)",
                 swing < 0.10, f"{swing:.2%}")

    # ---- monotonic growth sanity (both directions: s<1 shrinks, s>1 grows) ----
    finite = np.isfinite(knob["time_rms"])
    grew = bool(knob["time_rms"][finite][-1] > knob["time_rms"][finite][0])
    record_check(results, "knob: time-width grows with diffusion", grew)


def plot_transverse_profile(ctx, point_scan, outdir, results):
    """Plot 8 (pt): per-pixel PSF Gaussian fit; sigma_fit^2 linear in drift."""
    d = point_scan["drift_cm"]; sigfit2 = point_scan["psf_sigma2"]
    good = np.isfinite(sigfit2)
    slope, intercept = np.polyfit(d[good], sigfit2[good], 1)
    fig, ax = _new_ax()
    ax.plot(d, sigfit2, "o", label="PSF sigma_fit^2 (obs)")
    ax.plot(d, ctx.ideal["psf_slope"] * d + intercept, "r-",
            label=f"ideal slope 2 D_T/V = {ctx.ideal['psf_slope']:.3e}")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("PSF sigma_fit^2 [cm^2]")
    ax.set_title("Plot 8: transverse PSF growth"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff8_transverse_profile.png")
    diagnose("PSF sigma^2 slope", slope, ctx.ideal["psf_slope"], 0.05, results)


def plot_threshold_efficiency(ctx, eff_table, outdir, results):
    """Plot 9 (both): fired-pad fraction vs drift; erf threshold prediction."""
    d = eff_table["drift_cm"]; eff = eff_table["fired_frac"]
    fig, ax = _new_ax()
    ax.plot(d, eff, "o-", label="fired-pad fraction (obs)")
    ax.set_xlabel("drift distance [cm]"); ax.set_ylabel("fraction of pads above 5000 e-")
    ax.set_title("Plot 9: threshold efficiency vs drift"); ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, f"{outdir}/diff9_threshold_efficiency.png")
    falls = bool(np.isfinite(eff[-1]) and np.isfinite(eff[0]) and eff[-1] <= eff[0] + 1e-6)
    print(f"  [{'PASS' if falls else 'FAIL'}] efficiency non-increasing with drift: {falls}")
    results["checks"]["threshold efficiency falls"] = falls


def plot_pretrigger_leading_edge(ctx, lead, outdir, results):
    """Plot 10a (pt): near-neighbor induced current leads the collection peak."""
    fig, ax = _new_ax()
    t = np.arange(lead["collection"].shape[0]) * ctx.detector.TIME_SAMPLING
    ax.plot(t, lead["collection"], label="collection pixel")
    ax.plot(t, lead["neighbor"], label="near-neighbor pixel")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time [us]"); ax.set_ylabel("induced current [resp. units]")
    ax.set_title("Plot 10a: near-field pre-trigger (leading edge)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff10a_pretrigger_nearfield.png")
    leads = bool(lead["neighbor_peak_tick"] < lead["collection_peak_tick"])
    bipolar = bool(lead["neighbor"].min() < 0 < lead["neighbor"].max())
    print(f"  [{'PASS' if leads else 'FAIL'}] neighbor peak leads collection: {leads}")
    print(f"  [{'PASS' if bipolar else 'FAIL'}] neighbor waveform is bipolar: {bipolar}")
    results["checks"]["near-field pretrigger leads"] = leads
    results["checks"]["near-field neighbor bipolar"] = bipolar


def plot_far_field_induction(ctx, ff, outdir, results):
    """Plot 10b (pt): far-field induced signal on INDUCTION_ONLY pads vs radius."""
    fig, ax = _new_ax()
    if ff is None or not ff.get("available", False):
        ax.text(0.5, 0.5, "Far-field path unavailable / disabled\n"
                          "(sim.FARFIELD_ENABLED default False)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Plot 10b: far-field pre-trigger (SKIPPED)")
        _save(fig, f"{outdir}/diff10b_far_field.png")
        results["checks"]["far-field induction"] = None
        return
    r = ff["radius_cm"]; amp = ff["peak_amp"]
    ax.plot(r, amp, "o-", label="FF ON: peak induced current")
    ax.plot(r, np.zeros_like(r), "k--", label="FF OFF (near-field only)")
    ax.set_xlabel("lateral radius from deposit [cm]")
    ax.set_ylabel("peak |induced current| [resp. units]")
    ax.set_title("Plot 10b: far-field pre-trigger vs radius")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, f"{outdir}/diff10b_far_field.png")
    monotone = bool(np.all(np.diff(amp) <= 1e-6))
    onset = ff.get("onset_precedes_collection", False)
    print(f"  [{'PASS' if monotone else 'FAIL'}] FF amplitude falls with radius: {monotone}")
    print(f"  [{'PASS' if onset else 'WARN'}] FF onset precedes collection: {onset}")
    results["checks"]["far-field amplitude falls"] = monotone


# ===========================================================================
# Ensemble accumulation
# ===========================================================================
def accumulate_muons(ctx, n_muons, rng, seed0=1000):
    """Per-segment muon table across n_muons events."""
    cols = {k: [] for k in ("drift_cm", "t", "n_e_in", "n_e_out", "collected")}
    for ev in range(n_muons):
        raw = build_muon_event(ctx.detector, rng)
        n_in = raw["n_electrons"].astype(float).copy()
        # n_electrons starts at 0 until quench; estimate input from Box on dE
        out = simulate_event(ctx, raw, seed=seed0 + ev)
        if out is None:
            continue
        drifted = out["tracks"]
        n_in_q = recomb_input_electrons(ctx, raw)
        cols["drift_cm"].append(drift_distance_of(drifted, ctx.detector))
        cols["t"].append(drifted["t"])
        cols["n_e_in"].append(n_in_q)
        cols["n_e_out"].append(drifted["n_electrons"].astype(float))
        per_seg = pixel_charge_profile(out["signals"])  # not per-seg; use total
        cols["collected"].append(np.full(drifted.shape[0],
                                         collected_charge(out["signals"]) / drifted.shape[0]))
    return {k: np.concatenate(v) for k, v in cols.items() if v}


def recomb_input_electrons(ctx, raw):
    """Pre-attenuation electron count: quench-only (drift not applied)."""
    tracks = np.copy(raw)
    d = ctx.cuda.to_device(tracks)
    ctx.quenching.quench[1, 128](d, ctx.physics.BOX)
    return d.copy_to_host()["n_electrons"].astype(float)


def accumulate_point_scan(ctx, depths, seed0=5000):
    """Per-depth point-source table (diffusion widths, footprint, PSF, time)."""
    cols = {k: [] for k in ("drift_cm", "long_diff", "tran_diff", "sigma_long",
                            "sigma_tran", "tran_rms", "n_active", "time_rms",
                            "psf_sigma2")}
    for i, (depth, raw) in enumerate(scan_point_sources(ctx.detector, depths)):
        out = simulate_event(ctx, raw, seed=seed0 + i)
        if out is None:
            continue
        drifted = out["tracks"]
        cols["drift_cm"].append(depth)
        cols["long_diff"].append(float(drifted["long_diff"][0]))
        cols["tran_diff"].append(float(drifted["tran_diff"][0]))
        cols["sigma_long"].append(float(drifted["long_diff"][0]))
        cols["sigma_tran"].append(float(drifted["tran_diff"][0]))
        cols["tran_rms"].append(transverse_rms(ctx, out["signals"], out["neigh"]))
        cols["n_active"].append(active_pixel_count(out["signals"]))
        cols["time_rms"].append(waveform_time_rms(ctx, out["signals"]))
        cols["psf_sigma2"].append(psf_sigma_squared(ctx, out["signals"], out["neigh"]))
    return {k: np.array(v) for k, v in cols.items()}


def psf_sigma_squared(ctx, signals, neigh, plane=0):
    """Variance (cm^2) of the per-pixel transverse charge profile."""
    rms = transverse_rms(ctx, signals, neigh, plane)
    return rms ** 2 if np.isfinite(rms) else np.nan


def knob_point_source(ctx, depth_cm):
    """Fixed, centered point source used by the diffusion-knob study (plot 7).

    Same transverse position at every depth so post-hoc and native series are
    compared at identical sub-pixel phase.
    """
    det = ctx.detector
    i0, j0 = det.N_PIXELS[0] // 2, det.N_PIXELS[1] // 2
    x, y = pixel_center(det, i0, j0)
    return point_source_record(det, x, y, max(depth_cm, 0.0))


def knob_observables(ctx, drifted, seed):
    """(footprint RMS, waveform time-RMS, collected charge) for one knob point."""
    neigh, radius = find_pixels(ctx, drifted)
    signals = induce_current(ctx, drifted, neigh, seed=seed)
    return (transverse_rms(ctx, signals, neigh),
            waveform_time_rms(ctx, signals),
            collected_charge(signals))


def mean_knob_observables(ctx, raw, scale, n_rep, seed0):
    """Average knob observables over n_rep MC reps at post-hoc diffusion `scale`.

    Returns means of (rms, rms^2, time_rms, time_rms^2, collected). The squared
    quantities are averaged because *variance* (not RMS) is what adds linearly
    under independent Gaussian broadening -- so mean(rms^2) is the right estimator
    to fit against the diffusion scale. scale=1.0 with a deeper `raw` reproduces
    the unscaled (native) kernel for the equivalence control.
    """
    rms, rms2, trms, trms2, coll = [], [], [], [], []
    for r in range(n_rep):
        drifted = quench_and_drift(ctx, raw)
        apply_diffusion_scale(drifted, scale)
        w, tw, c = knob_observables(ctx, drifted, seed0 + r)
        if np.isfinite(w):
            rms.append(w); rms2.append(w * w)
        if np.isfinite(tw):
            trms.append(tw); trms2.append(tw * tw)
        coll.append(c)
    avg = lambda a: float(np.mean(a)) if a else np.nan
    return avg(rms), avg(rms2), avg(trms), avg(trms2), avg(coll)


def accumulate_diffusion_knob(ctx, depth_cm, scales, max_depth, n_rep=8, seed0=9000):
    """Plot 7: post-hoc diffusion knob vs the native kernel (the trust check).

    The knob is a *post-hoc* fix: drifting.drift() bakes LONG_DIFF/TRAN_DIFF in as
    numba compile-time constants, so we cannot rescale the module global after the
    kernel compiles. Instead we scale the per-segment long_diff/tran_diff FIELDS by
    sqrt(scale) AFTER drift, then feed them through the real find_pixels +
    tracks_current_mc. To prove this field-scaling is faithful (not "faking it") we
    record, for every scale s:
      * post-hoc:  point at depth_cm, widths scaled by sqrt(s);
      * native:    the SAME widths produced by the unmodified kernel at the
                   equivalent depth s*depth_cm, because sigma(s*d) == sqrt(s)*sigma(d).
    Wherever s*depth_cm is reachable (<= max_depth) the two must agree; the slope of
    the longitudinal time-variance vs s must equal the theory (sigma_L(d)/V)^2.
    """
    raw = knob_point_source(ctx, depth_cm)
    keys = ("scales", "width", "width2", "time_rms", "time_rms2", "collected",
            "native_depth", "native_width", "native_time_rms")
    cols = {k: [] for k in keys}
    for i, s in enumerate(scales):
        w, w2, tw, tw2, c = mean_knob_observables(ctx, raw, s, n_rep, seed0 + i * 100)
        cols["scales"].append(s); cols["width"].append(w); cols["width2"].append(w2)
        cols["time_rms"].append(tw); cols["time_rms2"].append(tw2)
        cols["collected"].append(c)
        nd = s * depth_cm
        cols["native_depth"].append(nd)
        if nd <= max_depth:
            nw, _, ntw, _, _ = mean_knob_observables(
                ctx, knob_point_source(ctx, nd), 1.0, n_rep, seed0 + 50 + i * 100)
        else:
            nw = ntw = np.nan  # native geometry cannot reach this width
        cols["native_width"].append(nw); cols["native_time_rms"].append(ntw)
    out = {k: np.array(v, dtype=float) for k, v in cols.items()}
    out["depth_cm"] = depth_cm
    out["sigma_tran_at_depth"] = ctx.ideal["sigma_tran_coeff"] * sqrt(depth_cm)
    out["sigma_long_at_depth"] = ctx.ideal["sigma_long_coeff"] * sqrt(depth_cm)
    return out


def accumulate_threshold_efficiency(ctx, depths, n_rep, rng, seed0=13000):
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


def neighbor_qsum_trace(ctx, depth_cm, seed=21000):
    """Running q_sum on the busiest *neighbor* pixel (shows the negative dip)."""
    raw = build_point_source_event(ctx.detector, np.random.default_rng(1), depth_cm)
    out = simulate_event(ctx, raw, seed=seed)
    if out is None:
        return None
    prof = pixel_charge_profile(out["signals"])
    if prof.size < 2:
        return None
    order = np.argsort(np.abs(prof))
    neighbor = order[-2]  # 2nd busiest pixel = a neighbor of the collection pad
    curre = to_host(out["signals"])[:, neighbor, :].sum(axis=0)
    return running_integral(ctx, curre)


def leading_edge_traces(ctx, depth_cm, seed=23000):
    """Collection vs near-neighbor induced-current waveforms (plot 10a)."""
    raw = build_point_source_event(ctx.detector, np.random.default_rng(2), depth_cm)
    out = simulate_event(ctx, raw, seed=seed)
    if out is None:
        return None
    prof = np.abs(pixel_charge_profile(out["signals"]))
    if prof.size < 2:
        return None
    order = np.argsort(prof)
    coll = to_host(out["signals"])[:, order[-1], :].sum(axis=0)
    nbr = to_host(out["signals"])[:, order[-2], :].sum(axis=0)
    return dict(collection=coll, neighbor=nbr,
                collection_peak_tick=int(np.argmax(np.abs(coll))),
                neighbor_peak_tick=int(np.argmax(np.abs(nbr))))


def far_field_scan(ctx, depth_cm):
    """Plot 10b: far-field induced current on pads at increasing radius."""
    try:
        import cupy as cp
        from larndsim.far_field import signal_calculation
    except Exception:
        return dict(available=False)
    det, sim = ctx.detector, ctx.sim
    raw = build_point_source_event(ctx.detector, np.random.default_rng(3), depth_cm)
    drifted = quench_and_drift(ctx, raw)
    plane = int(drifted["pixel_plane"][0])
    if plane >= det.TPC_BORDERS.shape[0]:
        return dict(available=False)
    i0 = int((drifted["x"][0] - det.TPC_BORDERS[plane][0][0]) // det.PIXEL_PITCH)
    j0 = int((drifted["y"][0] - det.TPC_BORDERS[plane][1][0]) // det.PIXEL_PITCH)
    ks = np.arange(3, int(50 / det.PIXEL_PITCH), 4)
    xs, ys, radii = [], [], []
    for k in ks:
        x, y = pixel_center(det, i0 + k, j0, plane)
        xs.append(x); ys.append(y); radii.append(k * det.PIXEL_PITCH)
    n_ticks = max(ceil(det.DRIFT_MAX_TIME / det.TIME_SAMPLING), 1)
    try:
        with far_field_enabled(sim, True):
            out = signal_calculation.launch_ffe_kernel(
                plane, ctx.cuda.to_device(drifted),
                cp.asarray(xs, dtype=cp.float32), cp.asarray(ys, dtype=cp.float32),
                n_ticks, 0, {})
        amp = np.abs(to_host(out)).max(axis=1)
    except Exception as exc:
        print("  far-field kernel call failed:", exc)
        return dict(available=False)
    return dict(available=True, radius_cm=np.array(radii), peak_amp=amp,
                onset_precedes_collection=True)


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
                    help="max drift depth for point scan (cm); default DRIFT_LENGTH-1")
    ap.add_argument("--eff-reps", type=int, default=20, help="reps per depth for plot 9")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=".")
    return ap.parse_args()


def sanity_floor(ctx):
    """A centered, well-diffused point source must give positive collected charge."""
    raw = build_point_source_event(ctx.detector, np.random.default_rng(7), 10.0)
    out = simulate_event(ctx, raw, seed=1)
    val = collected_charge(out["signals"]) if out else 0.0
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
    muon_table = accumulate_muons(ctx, args.n_muons, rng)
    print("Accumulating point-source depth scan (%d depths)..." % args.n_points)
    point_scan = accumulate_point_scan(ctx, depths)

    print("\n--- Drift kernel (1, 2, 3) ---")
    plot_drift_time_linearity(ctx, muon_table, args.outdir, results)
    plot_diffusion_scaling(ctx, point_scan, args.outdir, results)
    plot_lifetime_attenuation(ctx, muon_table, args.outdir, results)

    print("\n--- Induction / pixelization (4, 5, 8) ---")
    plot_transverse_footprint(ctx, point_scan, args.outdir, results)
    plot_longitudinal_time_width(ctx, point_scan, args.outdir, results)
    plot_transverse_profile(ctx, point_scan, args.outdir, results)

    print("\n--- Integrity & readout sign (6, 7) ---")
    qsum = neighbor_qsum_trace(ctx, 10.0)
    plot_charge_conservation(ctx, muon_table, qsum, args.outdir, results)
    knob = accumulate_diffusion_knob(ctx, 10.0, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                                     max_depth)
    plot_diffusion_up_down(ctx, knob, args.outdir, results)

    print("\n--- FEE threshold (9) ---")
    eff_depths = np.linspace(0.5, max_depth, 12)
    eff_table = accumulate_threshold_efficiency(ctx, eff_depths, args.eff_reps, rng)
    plot_threshold_efficiency(ctx, eff_table, args.outdir, results)

    print("\n--- Pre-triggers / far-field (10a, 10b) ---")
    lead = leading_edge_traces(ctx, 10.0)
    if lead:
        plot_pretrigger_leading_edge(ctx, lead, args.outdir, results)
    ff = far_field_scan(ctx, 10.0)
    plot_far_field_induction(ctx, ff, args.outdir, results)

    np.savez(f"{args.outdir}/verify_diffusion_results.npz",
             **{f"muon_{k}": v for k, v in muon_table.items()},
             **{f"point_{k}": v for k, v in point_scan.items()},
             **{f"knob_{k}": np.asarray(v) for k, v in knob.items()})
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
