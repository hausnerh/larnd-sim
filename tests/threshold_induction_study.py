#!/usr/bin/env python
"""
threshold_induction_study.py
============================

Disentangling **thresholding**, **periodic reset**, and **induction response**
in larnd-sim, using the charge spectrum of *single-hit pixels* in simulated EM
showers.

THE IDEA
  Simulate a dense, spatially-extended energy deposit (an EM-shower-like charge
  blob) and read it out through the REAL FEE. Then select pixels that record
  exactly ONE hit (one ADC sample) over the whole event. That single-hit sample
  is a mixture of two physically distinct populations:

    1. SHOWER (collection) hits -- a peripheral pad on which drifting electrons
       actually landed. It records the real (Landau-distributed) deposited
       charge: a HIGH-Q peak.
    2. INDUCTION hits -- a pad that collected NO charge but on which the bipolar
       induced current from a neighbour's collection momentarily crossed
       threshold before its negative lobe could cancel it. It records only that
       small transient: a LOW-Q peak, piled just above threshold.

  The two populations OVERLAP near threshold, so a free two-peak fit mis-assigns them.
  Instead we TEMPLATE the shower: with induction turned OFF (neighbour-pad response
  scaled to 0) the single-hit sample is PURE shower, so we fit that clean spectrum with
  a near-threshold bare LANDAU + a higher-Q Gaussian and FREEZE the Landau -- shape and
  crucially AMPLITUDE (the near-threshold shower is peripheral -> its weak neighbours
  induce little on it, so it is induction-immune). A single Landau (not a Landau*Gauss)
  keeps that amplitude well determined: the noise smear is sub-bin, so an extra Gaussian
  width would only add a degeneracy. We build the frozen Landau over the full (threshold
  x reset) grid and look for a CLOSED-FORM law for its parameters. The induction-ON fit
  is then: frozen shower Landau + a floating Gaussian (the core-adjacent bump, which IS
  induction-shadowed) + a floating induction Landau -- so the turn-on excess is cleanly
  attributed to induction. The per-pixel TRUTH label is a backtrack (did drifting charge
  land on the pad?) used only to colour the histograms. We ask:

    * Do threshold / periodic-reset / induction-response each move the extracted
      observables in CLEAR, MEASURABLE, and DISENTANGLED ways?

WHY THE THREE KNOBS SEPARATE (the hypothesis this script tests)
  * THRESHOLD sets where the induction peak sits: induction hits barely cross
    threshold, so their MPV rides the threshold. The shower peak (well above
    threshold) barely moves. -> threshold moves Ind. MPV, leaves Shower MPV.
  * INDUCTION RESPONSE sets how MANY induction hits there are (and how strong the
    transient is) without moving where the threshold turn-on sits. -> induction
    moves the induction *fraction/rate*, leaves both MPVs ~fixed.
  * PERIODIC RESET periodically clears the running integral, suppressing the
    slow-accumulation induction hits AND chopping real shower charge into more /
    smaller samples. -> reset moves both populations' rates with a distinct
    signature (it is the one knob that also lowers the shower yield).

  The "money plot" is a sensitivity matrix (knob x extracted observable): a
  near-diagonal structure is the quantitative statement that the three effects
  are separable from the single-hit Q spectrum alone.

HOW THE KNOBS ARE APPLIED (and why two are cheap, one needs a recompile)
  * Threshold      : `DISCRIMINATION_THRESHOLD` is handed to the FEE kernel as a
                     per-pixel ARRAY -> runtime-tunable, no recompile, no need to
                     re-run induction. (Scanned by re-running only the FEE stage.)
  * Induction      : scale the `response` table's neighbour-pad bins (Chebyshev
                     distance from pad centre > pitch/2). `response` is a kernel
                     ARGUMENT, so this is runtime-tunable too -- but it changes
                     the pre-FEE current, so the induction scan re-runs induction.
  * Periodic reset : `PERIODIC_RESET_CYCLES` is read as a module GLOBAL inside the
                     jitted `get_adc_values`, so numba bakes it at compile time
                     (exactly the constraint verify_diffusion notes for the
                     diffusion knob). We force a recompile by reloading the `fee`
                     module after setting the global.

  Because threshold and periodic reset only touch the (cheap) FEE stage, the
  expensive pre-FEE signal (drift -> get_pixels -> tracks_current_mc ->
  sum_pixel_signals) is computed ONCE per event at nominal induction and reused
  across both of those scans.

REQUIREMENTS
  A CUDA GPU (numba.cuda + cupy), scipy, matplotlib. Run from a larnd-sim
  checkout; config defaults to module0:
      python tests/threshold_induction_study.py --n-events 30 --outdir tistudy

NOTE ON "REALISTIC SHOWERS"
  No Geant4 input is wired in here, so the shower is a documented PARAMETRIC
  stand-in: deposits sampled from a gamma longitudinal profile with a laterally
  growing (core+tail) transverse spread. What matters for this study is the
  spatial charge STRUCTURE it reproduces -- a dense multi-hit core, a sparse
  single-hit periphery, and an induction halo -- not the exact cascade. Swap in
  real segments by replacing build_shower_event() and everything downstream is
  unchanged.
"""

import argparse
import importlib
import os
import sys
from math import sqrt, log, pi

import numpy as np

# Reuse the proven kernel drivers / plot helpers from the diffusion suite.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTS_DIR)
# ...and the repo root, so `import larndsim` (for the named-config registry) works when the
# script is run straight out of a checkout, not only when the package is pip-installed.
sys.path.insert(1, os.path.dirname(_TESTS_DIR))
import verify_diffusion as vd  # noqa: E402

SQRT2PI = sqrt(2.0 * pi)

# Liquid-argon EM constants for the parametric shower (approximate; only set the
# blob geometry, not any pass/fail number).
X0_LAR = 14.0        # radiation length (cm)
EC_LAR = 32.0        # e- critical energy (MeV)
W_ION_MEV = 23.6e-6  # mean ionisation energy (MeV/e-), for reference only


# ===========================================================================
# Parametric EM-shower charge blob (parametric stand-in -- see module docstring)
# ===========================================================================
def build_shower_event(ctx, rng, plane=0, energy_mev=300.0, n_dep=400,
                       step_cm=0.3, long_cm=12.0, sig_core=0.45, sig_tail=2.4,
                       tail_frac=0.30, grow=0.6, z_squash=0.45, depth0=None):
    """A `tracks` array of many short tracklets sampling an EM-shower-like blob.

    Longitudinal (along +x in the pixel plane): a gamma profile dE/dt ~ t^a e^{-bt}
    (t in radiation lengths), rescaled so the bulk fits inside `long_cm`. Transverse:
    a core+tail Gaussian mixture whose width grows with shower depth, with the spread
    in y (in the pixel plane -> lights up many pads, the single-hit periphery we want)
    larger than the spread in z (the drift direction -> kept modest by `z_squash` so
    the readout window stays bounded). Each deposit carries energy_mev/n_dep over a
    `step_cm` tracklet, so the local dE/dx is MIP-like and the dense core is built by
    deposit PILE-UP (many tracklets per pad) -- giving multi-hit core pads, a single-
    hit periphery, and an induction halo.
    """
    det = ctx.detector
    x0, x1, y0, y1 = vd.active_volume(det, plane, margin=3.0)
    dmax = abs(det.DRIFT_LENGTH) - 3.0
    # Shower starts a sixth of the way in x, centred in y, at a mid drift depth.
    sx = x0 + 0.15 * (x1 - x0)
    sy = 0.5 * (y0 + y1)
    d0 = depth0 if depth0 is not None else 0.5 * dmax
    sz = vd.depth_to_z(det, d0, plane)
    z_anode = vd.plane_z(det, plane)[0]

    # Gamma longitudinal profile -> normalised coordinate xi (~1 at shower max).
    tmax = max(1.0, log(energy_mev / EC_LAR) - 0.5)
    b = 0.5
    a = b * tmax + 1.0
    t = rng.gamma(a, 1.0 / b, size=n_dep)          # radiation lengths
    xi = t / tmax
    long_off = np.clip(xi, 0, 3.5) / 3.5 * long_cm

    # Transverse core+tail mixture, width grows with shower depth (later = wider).
    is_tail = rng.random(n_dep) < tail_frac
    width = np.where(is_tail, sig_tail, sig_core) * (1.0 + grow * np.clip(xi, 0, 3))
    dy = rng.normal(0.0, width)
    dz = rng.normal(0.0, width) * z_squash

    xpos = np.clip(sx + long_off, x0, x1)
    ypos = np.clip(sy + dy, y0, y1)
    # dz is a real offset along the drift direction -> a per-deposit drift depth.
    depth = np.clip(np.abs((sz + dz) - z_anode), 0.5, dmax)
    zpos = np.array([vd.depth_to_z(det, dd, plane) for dd in depth])

    tracks = vd.blank_tracks(n_dep)
    e_per = energy_mev / n_dep
    dedx = e_per / step_cm
    half = 0.5 * step_cm
    for i in range(n_dep):
        p0 = (xpos[i] - half, ypos[i], zpos[i])
        p1 = (xpos[i] + half, ypos[i], zpos[i])
        vd.fill_segment(tracks[i], plane, p0, p1, dedx)
        tracks[i]["pdg_id"] = 11
        tracks[i]["segment_id"] = i
    return tracks


def build_muon_event(ctx, rng, plane=0, theta_deg=45.0, length_cm=20.0,
                     dedx=2.1, step_cm=0.3):
    """A straight MIP muon track, parametrised by its angle to the PIXEL PLANE.

    theta_deg is the angle between the track and the pixel (anode) plane:
      * 0   -> track lies IN the plane at fixed drift depth: charge along the whole track
               arrives at ~the same time (ISOCHRONOUS). Lights up a clean line of collection
               pixels, each a temporally concentrated deposit (small sig_t_coll).
      * 90  -> track runs ALONG the drift axis at fixed (x,y): a single pixel column collects
               charge deposited at every drift depth, so it arrives spread over the drift time
               (large sig_t_coll); that column is multi-hit, and the single-hit sample is the
               transverse-neighbour induction.
      * 45  -> tilted: each pixel sees charge over a time window set by the tilt.
    The transverse position, depth and azimuth are randomised per event so different pixels
    are sampled while the topology (angle) is held fixed. MIP dE/dx ~ 2.1 MeV/cm."""
    from math import radians, cos, sin
    det = ctx.detector
    x0, x1, y0, y1 = vd.active_volume(det, plane, margin=3.0)
    dmax = abs(det.DRIFT_LENGTH) - 3.0
    th = radians(theta_deg)
    ph = rng.uniform(0.0, 2.0 * np.pi)
    dvec = np.array([cos(th) * cos(ph), cos(th) * sin(ph), sin(th)])   # (x, y, DEPTH)
    cx = rng.uniform(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0))
    cy = rng.uniform(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0))
    cdepth = rng.uniform(0.3 * dmax, 0.7 * dmax)

    n = max(int(length_cm / step_cm), 2)
    s = (np.arange(n) - 0.5 * n) * step_cm
    X, Y, D = cx + s * dvec[0], cy + s * dvec[1], cdepth + s * dvec[2]
    inb = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (D >= 0.5) & (D <= dmax)
    X, Y, D = X[inb], Y[inb], D[inb]
    if X.size < 2:
        return vd.blank_tracks(0)
    half = 0.5 * step_cm
    tracks = vd.blank_tracks(X.size)
    for i in range(X.size):
        z0 = vd.depth_to_z(det, float(np.clip(D[i] - half * dvec[2], 0.5, dmax)), plane)
        z1 = vd.depth_to_z(det, float(np.clip(D[i] + half * dvec[2], 0.5, dmax)), plane)
        p0 = (X[i] - half * dvec[0], Y[i] - half * dvec[1], z0)
        p1 = (X[i] + half * dvec[0], Y[i] + half * dvec[1], z1)
        vd.fill_segment(tracks[i], plane, p0, p1, dedx)
        tracks[i]["pdg_id"] = 13
        tracks[i]["segment_id"] = i
    return tracks


# ===========================================================================
# Induction-response knob -- scale neighbour-pad bins of the response table
# ===========================================================================
def induction_mask(ctx):
    """Boolean (Nx, Ny) mask: True where a response bin is OFF the collection pad.

    get_closest_waveform indexes the response at (x_dist, y_dist) = distance from
    the charge to a pad CENTRE. The charge's own pad spans |x_dist|,|y_dist| <=
    pitch/2 (Chebyshev <= pitch/2); any bin beyond that is the induced response
    felt by a NON-collecting neighbour. Scaling exactly those bins changes the
    induction strength while leaving every pad's own collection response intact.
    """
    det = ctx.detector
    resp = ctx.response
    nx, ny = resp.shape[0], resp.shape[1]
    bs = det.RESPONSE_BIN_SIZE
    xc = (np.arange(nx) + 0.5) * bs                 # bin-centre distances (cm)
    yc = (np.arange(ny) + 0.5) * bs
    half = det.PIXEL_PITCH / 2.0
    cheb = np.maximum(xc[:, None], yc[None, :])     # Chebyshev distance to pad centre
    return cheb > half


def set_induction(ctx, scale):
    """Point ctx.response at the response table with the neighbour-pad bins scaled by `scale`.

    scale=1 -> nominal; 0 -> no neighbour induction (collection only); >1 stronger.

    MEMORY: the fsd_cube response is ~1 GB on the GPU (MAX_RADIUS=12), and this is called once
    per induction scale, so the naive `resp = r0.copy(); resp[m] = resp[m]*scale` is what tips a
    tight GPU into OOM -- it allocates a full copy PLUS a boolean-mask gather AND a scatter, i.e.
    up to ~3 transient full-size buffers. Instead we multiply the pristine table by a broadcast
    (nx, ny, 1) factor straight into ONE reused work buffer: a single full-size buffer, no
    per-bin temporaries. The result is bit-identical to the old path (verified). scale==1.0 just
    aliases the pristine table -- no allocation at all.
    """
    xp = _xp_of(ctx._response0)
    r0 = ctx._response0
    if scale == 1.0:                        # nominal: alias the pristine table, no copy
        ctx.response = r0
        ctx.induction_scale = 1.0
        return
    # (nx, ny, 1, ...) multiplicative factor: `scale` on neighbour bins, 1 on each pad's own
    # bins; broadcasts over the response time axis. The fancy index is on the SMALL (nx, ny)
    # mask, not the big response. Rebuilt only when the scale changes.
    if getattr(ctx, "_induction_factor_scale", None) != scale:
        mask = xp.asarray(ctx._induction_mask)
        factor = xp.ones(mask.shape, dtype=r0.dtype)
        factor[mask] = r0.dtype.type(scale)
        ctx._induction_factor = factor.reshape(mask.shape + (1,) * (r0.ndim - mask.ndim))
        ctx._induction_factor_scale = scale
    work = getattr(ctx, "_response_work", None)
    if work is None or work.shape != r0.shape or work.dtype != r0.dtype:
        work = ctx._response_work = xp.empty_like(r0)
    xp.multiply(r0, ctx._induction_factor, out=work)   # in-place, no fancy-index temporary
    ctx.response = work
    ctx.induction_scale = scale


def _xp_of(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp
    except Exception:
        pass
    return np


# ===========================================================================
# Periodic-reset knob -- needs a recompile (kernel global), so reload `fee`
# ===========================================================================
def set_periodic_reset(ctx, cycles):
    """Set PERIODIC_RESET_CYCLES and force a fee recompile if it changed.

    numba bakes module globals into a jitted kernel at compile time, so the only
    portable way to change a kernel global between runs is to reload the module
    that defines the kernel (the next call then recompiles against the new value).
    No-op when the value is unchanged (avoids needless recompiles).
    """
    cur = getattr(ctx.detector, "PERIODIC_RESET_CYCLES", -1)
    if int(cur) == int(cycles):
        return
    ctx.detector.PERIODIC_RESET_CYCLES = int(cycles)
    importlib.reload(ctx.fee)   # reload returns the same module object, recompiled lazily


# ===========================================================================
# One event: expensive pre-FEE stage (cached) + cheap FEE stage (scanned)
# ===========================================================================
def collection_pixels_from_signals(signals, neigh, f_collect):
    """Truth BACKTRACK: the set of pixel IDs on which drifting charge actually LANDED.

    `signals[s, j, :]` is the signed induced current from segment s on the j-th pad
    of its neighbourhood (pixel id `neigh[s, j]`; -1 = padding). Integrated over
    time it is the NET charge that pad sees from that segment -- ~the collected
    electrons on a pad the diffused cloud lands on, ~0 on a pure-induction neighbour
    (the bipolar Ramo lobes telescope away). So a (segment, pad) COLLECTED charge
    iff its net is a real positive fraction (> f_collect) of the most-collecting pad
    for that same segment: a footprint/shape criterion that says "charge terminated
    here", independent of any absolute charge cut or the discriminator threshold.
    This is the per-pixel truth used to split single-hit pixels into collection
    (shower) vs pure-induction hits -- no arbitrary net-charge cut.

    NOTE this is a purely SPATIAL test ("did charge ever land here"); it says nothing about
    what actually made the hit fire, so a pad with sub-threshold real charge that was pushed
    over by a neighbour's transient is still labelled shower. See collected_charge_truth()
    for the timing information that makes the label causal.

    Returns (collect_ids, landed, net_sp, neigh_h) so callers can also build the per-pixel
    COLLECTED-charge truth (amount + arrival time), not just the pixel-id set."""
    import cupy as cp
    net_sp = cp.asnumpy(signals.sum(axis=2))          # (nseg, max_neigh) net per (seg,pad)
    neigh_h = cp.asnumpy(neigh)
    seg_peak = np.maximum(net_sp.max(axis=1), 0.0)    # per-segment peak collected net
    landed = (neigh_h >= 0) & (net_sp > 0) & (net_sp > f_collect * seg_peak[:, None])
    return np.unique(neigh_h[landed]), landed, net_sp, neigh_h


def collected_charge_truth(landed, net_sp, neigh_h, unique_pix, t_arrival, ts):
    """Per-pixel truth about the REAL charge that landed on it, from the collection
    (segment, pad) pairs only:

      q_coll   -- total collected charge on the pad (ELECTRONS: net_sp is summed
                  current-per-tick, so x TIME_SAMPLING converts to charge),
      t_coll   -- charge-weighted ARRIVAL TIME of that charge at the anode (us).
                  `t_arrival` must be the drift kernel's `tracks["t"]` (= drift_time + t0,
                  written by drifting.drift) -- NOT `t0`, which is the segment's GENERATION
                  time (~0 for our events). The FEE's `adc_ticks` live on the same clock
                  (hits appear at ~ the drift time), so dt = t_hit - t_coll is meaningful,
                  up to a small constant response delay -- use it RELATIVE, not absolute.
      sig_t    -- spread of that arrival time: small => one concentrated deposit,
                  large => charge dribbling in over many segments/depths.

    These are what distinguish "a pad with just enough charge to fire once" from "a pad
    under a large, temporally concentrated deposit".

    ALSO returns the NEIGHBOURHOOD arrival time:

      t_near   -- charge-weighted arrival time of ALL the charge in this pad's
                  neighbourhood, whether or not it landed on the pad itself.

    This one is defined for essentially EVERY pixel, including pure-induction pads that
    collect nothing (for which t_coll is undefined/NaN). It is the reference the induction
    hits need: an induction pad fires because a NEIGHBOUR's charge is arriving, so
    dt_near = t_hit - t_near tests whether the hit leads that arrival. Without it, induction
    pixels drop out of the timing comparison entirely and only the collection population is
    plotted."""
    npix = int(np.size(unique_pix))
    t_arr = np.asarray(t_arrival, float)
    q = np.zeros(npix); m1 = np.zeros(npix); m2 = np.zeros(npix)
    if landed.any():
        ids = neigh_h[landed]
        idx = np.searchsorted(unique_pix, ids)        # unique_pix is sorted (cp.unique)
        w = net_sp[landed].astype(np.float64) * float(ts)   # current-sum -> electrons
        seg = np.nonzero(landed)[0]                   # segment index of each landed pair
        t = t_arr[seg]
        q = np.bincount(idx, weights=w, minlength=npix)
        m1 = np.bincount(idx, weights=w * t, minlength=npix)
        m2 = np.bincount(idx, weights=w * t * t, minlength=npix)
    good = q > 0
    t_coll = np.full(npix, np.nan); sig_t = np.full(npix, np.nan)
    t_coll[good] = m1[good] / q[good]
    sig_t[good] = np.sqrt(np.clip(m2[good] / q[good] - t_coll[good] ** 2, 0.0, None))

    # --- neighbourhood arrival time: every (segment, pad) pair in the neighbourhood map,
    #     weighted by that segment's OWN collected charge (its peak-pad net) -- i.e. how
    #     much charge is driving the induction seen here.
    valid = neigh_h >= 0
    t_near = np.full(npix, np.nan)
    if valid.any():
        seg_q = np.maximum(net_sp.max(axis=1), 0.0) * float(ts)   # per-segment collected e-
        segn = np.nonzero(valid)[0]
        wn = seg_q[segn]
        keep = wn > 0
        if keep.any():
            idxn = np.searchsorted(unique_pix, neigh_h[valid][keep])
            wn = wn[keep]; tn = t_arr[segn[keep]]
            qn = np.bincount(idxn, weights=wn, minlength=npix)
            mn = np.bincount(idxn, weights=wn * tn, minlength=npix)
            gn = qn > 0
            t_near[gn] = mn[gn] / qn[gn]
    return q, t_coll, sig_t, t_near


def event_drift(ctx, tracks):
    """The induction-scale-INDEPENDENT pre-FEE work: quench -> drift -> get_pixels.

    Returns (drifted_host, neigh, radius) or None if the event drifts no electrons.
    Drift and pixel-finding depend only on the deposited charge geometry, NOT on the
    response/induction scale, so this is computed ONCE per event and reused across the
    nominal pass, the induction-off pass, and every induction-scan scale -- only
    induce_current + sum_pixel_signals downstream have to re-run per scale."""
    drifted = vd.quench_and_drift(ctx, tracks)
    if float(np.sum(drifted["n_electrons"])) <= 0:
        return None
    neigh, radius = vd.find_pixels(ctx, drifted)
    return drifted, neigh, radius


def event_induce(ctx, drifted, neigh, radius, seed, collect_frac=0.15):
    """Induce + sum on a cached (drifted, neigh, radius) -> the pre-FEE signal dict.

    The only induction-scale-DEPENDENT stage (callers vary ctx.response between calls).
    `is_collection[ip]` is the per-pixel TRUTH label from a backtrack
    (collection_pixels_from_signals): True if drifting charge landed on pixel ip (a
    shower/collection hit), False if it only ever saw induced current from a neighbour.
    `truth_e[ip]` (net integrated charge) is kept only for the diagnostic seed-check.
    """
    signals = vd.induce_current(ctx, drifted, neigh, seed=seed)
    collect_ids, landed, net_sp, neigh_h = collection_pixels_from_signals(
        signals, neigh, collect_frac)
    summed = vd.sum_to_pixels(ctx, signals, neigh, radius, drifted)
    unique_pix, pixels_signals = summed[0], summed[1]
    if pixels_signals is None:
        return None
    ts = ctx.detector.TIME_SAMPLING
    ps_host = vd.to_host(pixels_signals).astype(np.float32)
    uph = vd.to_host(unique_pix)
    npix = ps_host.shape[0]
    # pixel rows of pixels_signals align 1:1 with unique_pix, so the backtrack set
    # maps straight onto a per-pixel boolean collection/induction label.
    is_collection = np.isin(uph, collect_ids)
    # ... and the per-pixel COLLECTED-charge truth (amount + arrival time + spread), which
    # is what lets us (a) anatomise the two shower peaks and (b) add a causal timing test.
    q_coll, t_coll, sig_t_coll, t_near = collected_charge_truth(
        landed, net_sp, neigh_h, uph, drifted["t"], ts)
    # The study uses only the ADC (q_sum), never the per-track backtracking. The
    # backtracking array (pixels_tracks_signals, size nt0*sum(num_backtrack)) is by
    # far the largest per-event object and OOM'd the 200-event cache -- so we DROP
    # it: zero num_backtrack makes get_adc_values skip backtracking entirely while
    # adc_list (built from pixels_signals alone) is bit-for-bit unchanged.
    return dict(
        unique_pix=uph,
        pixels_signals=ps_host,
        pixels_tracks_signals=np.zeros(1, dtype=np.float64),
        num_backtrack=np.zeros(npix, dtype=np.int64),
        offset_backtrack=np.zeros(npix, dtype=np.int64),
        is_collection=is_collection,                          # per-pixel truth (backtrack)
        q_coll=q_coll,                                        # real charge landed on pad (e-)
        t_coll=t_coll,                                        # its arrival time (us, global frame)
        sig_t_coll=sig_t_coll,                                # arrival-time spread (us)
        t_near=t_near,                                        # neighbourhood arrival time (us)
        truth_e=ps_host.sum(axis=1, dtype=np.float64) * ts,   # net collected charge (e-)
        max_time=ps_host.shape[1] * ts,
    )


def _cache_mb(evs):
    """Host megabytes held by a list of cached pre-signal dicts.

    `pixels_signals` dominates: it is a dense (n_unique_pix x n_ticks) float32 array, and
    n_unique_pix grows with MAX_RADIUS**2. Worth printing, because MAX_RADIUS comes from the
    RESPONSE FILE rather than from anything set here, so swapping in a wider response table
    silently multiplies the cache -- which is how a configuration that ran yesterday OOMs
    today with no change to the command line.
    """
    tot = 0
    for e in evs:
        if not e:
            continue
        for k in ("pixels_signals", "unique_pix", "q_coll", "t_coll", "sig_t_coll",
                  "t_near", "truth_e", "is_collection"):
            v = e.get(k)
            if v is not None:
                tot += int(np.asarray(v).nbytes)
    return tot / 1024.0 / 1024.0


def event_presignals(ctx, tracks, seed, collect_frac=0.15):
    """Thin wrapper: event_drift then event_induce (for callers that don't reuse drift)."""
    d = event_drift(ctx, tracks)
    if d is None:
        return None
    return event_induce(ctx, d[0], d[1], d[2], seed, collect_frac)


# per-single-hit TRUTH auxiliaries carried alongside (q, is_collection): the hit time, the
# real charge that landed on the pad and when/how spread-out it arrived. These drive the
# two-peak anatomy and the causal (timing) shower-vs-induction separation.
_AUX_KEYS = ("t_hit", "t_coll", "dt", "q_coll", "sig_t_coll", "t_near", "dt_near")


def _aux_sel(aux, sel, n):
    """Apply a boolean/index selection to an aux dict, skipping entries that are absent
    (older spectra files predate some keys) so old files still analyse cleanly."""
    out = {}
    for k, v in (aux or {}).items():
        v = np.asarray(v, float)
        out[k] = v[sel] if v.size == n else np.empty(0)
    return out


_HIT_POPS = (1, 2)


def _empty_pop():
    """An empty (Q, truth, aux) population, shaped like a real one."""
    return np.empty(0), np.empty(0, bool), {k: np.empty(0) for k in _AUX_KEYS}


def event_fee_hits(ctx, ev, threshold_e, seed, pops=_HIT_POPS):
    """Run only the FEE on a cached pre-signal; split pixels by ADC MULTIPLICITY.

    Returns {n: (Q, truth, aux)} for each n in `pops`. n=1 is the single-hit sample every
    fit in this study uses. n=2 is the population those pixels MIGRATE INTO when a knob
    promotes them -- a higher threshold delays the crossing until a second sample fits, and
    a periodic reset landing mid-integration chops one hit into two. Because the single-hit
    cut *removes* the promoted pixels, reset is nearly invisible in the n=1 spectrum on its
    own; the migration between n=1 and n=2 is where its signature actually lives.

    Both populations come out of ONE FEE pass, so the second costs no extra GPU time.

    For n>1, Q is the SUM of the per-hit charges: get_adc_values resets the integrator after
    every ADC sample, so a pixel's total recorded charge is the sum of what its samples
    recorded. That puts n=2 on the SAME charge axis as n=1, which is what makes the
    migration readable -- a pixel that crosses over keeps roughly its charge, so the pair of
    spectra shows counts moving rather than two unrelated distributions. `aux` always refers
    to the FIRST hit (the discriminator's decision point), so the causal-timing keys mean
    the same thing in both populations.
    """
    import cupy as cp
    ctx.detector.DISCRIMINATION_THRESHOLD = float(threshold_e)
    adc, ticks = vd.run_fee(
        ctx,
        cp.asarray(ev["pixels_signals"]),
        cp.asarray(ev["pixels_tracks_signals"]),
        cp.asarray(ev["num_backtrack"]),
        cp.asarray(ev["offset_backtrack"]),
        ev["max_time"], seed=seed)
    adc = np.asarray(adc)                            # (npix, MAX_ADC_VALUES)
    ticks = np.asarray(ticks)                        # hit time (us), same grid as t_coll
    hit = adc > 0
    n_hits = hit.sum(axis=1)
    q_all = vd.adc_to_charge(ctx, adc)               # electrons; exactly 0 where adc <= 0
    out = {}
    # Every pixel that fired at all, whatever its multiplicity, as (summed Q, truth, n_hits).
    # The n=1/n=2 populations below are just slices of this, but they cannot represent 3+ hits
    # -- and a saturating pillar pixel reaches 20-30. At 6 bytes/pixel this is far cheaper
    # than the aux arrays (which are ~74% of the spectra file), so there is no reason to
    # throw the rest away.
    hit_any = n_hits >= 1
    out["all"] = (np.asarray(q_all[hit_any].sum(axis=1), float),
                  ev["is_collection"][hit_any],
                  np.asarray(n_hits[hit_any], np.int16))
    for n in pops:
        sel = n_hits == n
        if not sel.any():
            out[n] = _empty_pop()
            continue
        first = np.argmax(hit[sel], axis=1)          # column of the FIRST recorded sample
        q = q_all[sel].sum(axis=1)                   # total over this pixel's n hits
        t_hit = np.asarray(ticks[sel][np.arange(first.size), first], float)
        t_coll = np.asarray(ev["t_coll"][sel], float)
        t_near = np.asarray(ev["t_near"][sel], float)
        aux = dict(t_hit=t_hit, t_coll=t_coll, dt=t_hit - t_coll,
                   q_coll=np.asarray(ev["q_coll"][sel], float),
                   sig_t_coll=np.asarray(ev["sig_t_coll"][sel], float),
                   t_near=t_near, dt_near=t_hit - t_near)
        out[n] = (np.asarray(q, float), ev["is_collection"][sel], aux)
    return out


def _empty_all():
    """Empty all-multiplicity record, shaped like a real one."""
    return np.empty(0), np.empty(0, bool), np.empty(0, np.int16)


def event_fee_singlehits(ctx, ev, threshold_e, seed):
    """Single-hit (Q, truth, aux) only -- thin wrapper on `event_fee_hits`.

    Kept because the fits, the muon control samples and the template grid all want just the
    n=1 sample; only the scans need the paired populations.
    """
    return event_fee_hits(ctx, ev, threshold_e, seed, pops=(1,))[1]


# ===========================================================================
# Langaus model (Landau*Gauss) and the two-population fit
# ===========================================================================
class Langaus:
    """Landau (Moyal approximation) convolved with a Gaussian, on a fixed grid.

    The Moyal density exp(-(z+e^{-z})/2)/sqrt(2pi), z=(x-mpv)/eta, has its MODE at
    x=mpv and is the standard closed-form Landau stand-in; convolving with a
    Gaussian of width sigma_g gives the langaus used for ionisation spectra. The
    grid is fixed (data-range driven) so the model is smooth in its parameters --
    important for curve_fit's finite-difference Jacobian.
    """
    def __init__(self, xmin, xmax, n=4096):
        pad = 0.25 * (xmax - xmin) + 1.0
        self.g = np.linspace(max(0.0, xmin - pad), xmax + pad, n)
        self.dx = self.g[1] - self.g[0]

    def density(self, x, mpv, eta, sigma_g):
        eta = max(eta, 1e-6)
        # clip the lower tail: for z < -30 the Moyal density is already 0, and
        # np.exp(-z) there overflows float64 (harmless inf -> 0, but it warns)
        z = np.clip((self.g - mpv) / eta, -30.0, None)
        land = np.exp(-0.5 * (z + np.exp(-z))) / (eta * SQRT2PI)
        if sigma_g > self.dx:
            # cap the kernel half-width to the grid so convolve(mode="same")
            # always returns len(self.g) (else np.interp lengths mismatch when
            # curve_fit pushes sigma_g wider than the whole grid)
            half = min(int(4 * sigma_g / self.dx), (len(self.g) - 1) // 2)
            ks = np.arange(-half, half + 1) * self.dx
            ker = np.exp(-0.5 * (ks / sigma_g) ** 2)
            ker /= ker.sum()
            land = np.convolve(land, ker, mode="same")
        return np.interp(x, self.g, land)

    def comp(self, x, mpv, eta, sigma_g, area):
        return area * self.density(x, mpv, eta, sigma_g)

    def landau(self, x, mpv, eta, area):
        """Bare Landau (Moyal), NO Gaussian convolution. Used for the near-threshold
        peaks: the electronics-noise smear (sigma ~ 500-650 e-) is below the LSB-scale
        bin width here, so the Gaussian width is unresolvable and only adds an eta<->sigma
        degeneracy that destabilises the peak AMPLITUDE -- the quantity the induction
        extraction depends on most. One width (eta) => a well-determined area."""
        return area * self.density(x, mpv, eta, 0.0)


def _adc_lsb(q):
    """Charge quantum (e-/ADC-count) inferred from the recorded-Q grid.

    adc_to_charge maps integer ADC counts to charge, so the recorded Q values lie
    on an evenly-spaced grid whose step is the LSB. The median spacing of the
    sorted unique values recovers it (robust to the occasional gap)."""
    u = np.unique(q[np.isfinite(q) & (q > 0)])
    if u.size < 3:
        return 0.0
    d = np.diff(u)
    d = d[d > 1e-6]
    return float(np.median(d)) if d.size else 0.0


def _hist(q, threshold_e, nbins=45, qmax=None):
    """Histogram single-hit Q on the ADC-LSB grid -> (centres, counts, width, edges).

    Bin width is an integer multiple k of the LSB and edges are offset by LSB/2,
    so each bin contains exactly k ADC levels -- this removes the comb artifact you
    get when bins are narrower than the charge quantum."""
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size < 20:
        return None
    lo = 0.7 * threshold_e
    hi = qmax if qmax is not None else np.percentile(q, 99.7)
    if hi <= lo:
        hi = lo * 4
    lsb = _adc_lsb(q)
    if lsb > 0:
        k = max(1, int(round((hi - lo) / nbins / lsb)))
        w = k * lsb
        start = float(np.min(q)) - 0.5 * lsb        # ADC levels fall at bin centres
        nb = max(int(np.ceil((hi - start) / w)), 1)
        edges = start + w * np.arange(nb + 1)
    else:
        edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(q, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts.astype(float), edges[1] - edges[0], edges


def _gaussian(x, mu, sig, area):
    """Area-normalised Gaussian (the higher-Q shower bump in the template model)."""
    sig = max(float(sig), 1e-6)
    return area * np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * SQRT2PI)


def _guess_shower(centres, counts, threshold_e):
    """Seed (mpv,eta,A_L, mu,sig,A_G): the Landau seeds on the NEAR-THRESHOLD peak (the
    low hump within [T, 2.2T] -- NOT the global argmax, which at high threshold is the
    high-Q bump), the Gaussian on the upper shoulder above it."""
    w = centres[1] - centres[0]
    T = threshold_e
    total = float(counts.sum() * w)
    near = (centres >= 0.9 * T) & (centres <= 2.2 * T)
    if near.any() and counts[near].sum() > 0:
        mpv = float(centres[near][np.argmax(counts[near])])
        aL = float(counts[near].sum() * w)
    else:
        mpv, aL = 1.2 * T, 0.5 * total
    mpv = min(max(mpv, 0.95 * T), 2.0 * T)
    hi = centres > 2.2 * T                             # upper shoulder -> Gaussian bump
    if hi.any() and counts[hi].sum() > 0:
        mu = float(np.average(centres[hi], weights=counts[hi]))
        sig = max(float(np.sqrt(np.average((centres[hi] - mu) ** 2, weights=counts[hi]))), w)
        aG = float(counts[hi].sum() * w)
    else:
        mu, sig, aG = max(3.0 * T, 2.5 * mpv), 1.5 * mpv, 0.3 * total
    return [mpv, 0.15 * T, aL, mu, sig, aG]


def fit_shower_template(q_shower, threshold_e, qmax=None, n_shower=1, noise_e=500.0):
    """Fit the induction-OFF shower single-hit spectrum with a near-threshold LANGAUS
    (Landau (X) Gaussian, sigma FIXED to the electronics noise) + a Gaussian (high-Q bump),
    and return the FROZEN langaus as a reusable template.

    The near-threshold langaus is the induction-immune shape (peripheral collection pads
    whose weak neighbours induce little on them); it is reused in the induction-ON fit and
    its AMPLITUDE is the number the induction extraction hinges on. FIXING the Gaussian
    width to the known noise (not fitting it) gives BOTH: the sub-threshold turn-on smear
    (hits recorded below Q_thr because Q = charge + noise) AND a well-determined amplitude
    (no eta<->sigma degeneracy). Counts are normalised to hits-per-shower (`n_shower`) so
    the frozen amplitude is event-count-independent -- the same template can be used on a
    scan run with a different number of showers. The high-Q Gaussian (core-adjacent,
    induction-shadowed) is NOT reused (it floats in the composite fit)."""
    from scipy.optimize import curve_fit
    h = _hist(q_shower, threshold_e, qmax=qmax)
    if h is None:
        return dict(ok=False, reason="too few shower hits", threshold_e=threshold_e)
    centres, raw, width, edges = h
    counts = raw / max(n_shower, 1)                    # hits per shower
    lg = Langaus(centres[0], centres[-1])
    def model(x, mpv, eta, aL, mu, sig, aG):
        return lg.comp(x, mpv, eta, noise_e, aL) + _gaussian(x, mu, sig, aG)
    T, span, cmax = threshold_e, centres[-1] - centres[0], centres[-1]
    p0 = _guess_shower(centres, counts, threshold_e)
    # eta is the only free width (sigma fixed to noise) -> well-conditioned amplitude.
    # Cap mpv at 2.2*T so the near-threshold langaus can't latch onto the high-Q bump;
    # keep the Gaussian centred clearly above it (mu >= 2.0*T).
    lb = [0.85 * T, 0.02 * T, 0.0,    2.0 * T, 0.05 * T, 0.0]
    ub = [2.2 * T,  0.60 * T, np.inf, cmax,    span,     np.inf]
    p0 = [min(max(v, lb[i] + 1e-9), ub[i] - 1e-9) for i, v in enumerate(p0)]
    sigma = (np.sqrt(raw) + 1.0) / max(n_shower, 1)    # per-shower Poisson errors
    try:
        popt, pcov = curve_fit(model, centres, counts, p0=p0, bounds=(lb, ub),
                               sigma=sigma, absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        return dict(ok=False, reason=str(exc), threshold_e=threshold_e,
                    centres=centres, counts=counts, width=width, edges=edges)
    mpv, eta, aL, mu, sig, aG = popt
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))  # 1-sigma parameter uncertainties
    pred = model(centres, *popt)
    ndf = max(len(centres) - len(popt), 1)
    chi2 = float(np.sum(((counts - pred) / sigma) ** 2))
    return dict(ok=True, threshold_e=float(threshold_e), noise_e=float(noise_e),
                mpv_L=float(mpv), eta_L=float(eta), A_L=float(aL),
                mu_G=float(mu), sig_G=float(sig), A_G=float(aG),
                mpv_L_err=float(perr[0]), eta_L_err=float(perr[1]), A_L_err=float(perr[2]),
                mu_G_err=float(perr[3]), sig_G_err=float(perr[4]),
                popt=popt, perr=perr, model=model, lg=lg, centres=centres, counts=counts,
                width=width, edges=edges, pred=pred, chi2=chi2, ndf=ndf,
                chi2ndf=chi2 / ndf, n_shower=int(n_shower), n_hits=int(raw.sum()))


def fit_shower_plus_induction(q, threshold_e, template, qmax=None, n_shower=1, noise_e=500.0):
    """Composite induction-ON fit: FROZEN shower langaus (from `template`) + a floating
    Gaussian (induction-shadowed high-Q shower bump) + a floating induction langaus.

    All three peaks are Landau (X) Gaussian with sigma FIXED to the electronics noise, so
    the induction turn-on carries its sub-threshold fluctuation hits (recorded Q < Q_thr
    because Q = charge + noise). Freezing the shower langaus -- crucially its AMPLITUDE --
    removes the near-threshold degeneracy: the shower at the turn-on is pinned by the
    induction-off template, so the residual excess is cleanly attributed to induction.
    Counts are hits-per-shower (`n_shower`), matching the template's normalisation so the
    frozen amplitude transfers even when this run has a different shower count. Free params
    [mu, sig, A_G, mpv_i, eta_i, A_i]. Returns a `kind='template'` fit dict."""
    from scipy.optimize import curve_fit
    h = _hist(q, threshold_e, qmax=qmax)
    if h is None:
        return None
    centres, raw, width, edges = h
    counts = raw / max(n_shower, 1)                    # hits per shower
    if not (template and template.get("ok")):
        return dict(ok=False, reason="no shower template", centres=centres, counts=counts,
                    width=width, edges=edges, threshold_e=threshold_e, n_shower=int(n_shower))
    lg = Langaus(centres[0], centres[-1])
    mpv_L, eta_L, A_L = template["mpv_L"], template["eta_L"], template["A_L"]
    # The high-Q tail's SHAPE is frozen too. Measured across the scans, the tail POSITION and
    # WIDTH are induction-blind (mu_G moves 3.6%, sig_G 6.0% from induction x0 -> x1, vs 58%
    # and 38% from threshold) because they are set by collection physics -- exactly what the
    # induction-off template measures. Only the tail AMPLITUDE swings with induction (84%),
    # since induction promotes tail pixels to multi-hit and removes them from the sample. So
    # freeze (mu_G, sig_G) and float only A_G: 6 free parameters -> 4, which sharpens A_ind,
    # the quantity the whole induction extraction rests on.
    mu_G, sig_G = float(template["mu_G"]), float(template["sig_G"])
    frozen_c = lg.comp(centres, mpv_L, eta_L, noise_e, A_L)   # frozen shower langaus (per shower)
    frozen_g = _gaussian(centres, mu_G, sig_G, 1.0)           # frozen tail SHAPE (unit area)
    T, span, cmax = threshold_e, centres[-1] - centres[0], centres[-1]

    def model(x, aG, mpv_i, eta_i, aI):
        return frozen_c + aG * frozen_g + lg.comp(x, mpv_i, eta_i, noise_e, aI)

    total = float(counts.sum() * width)
    p0 = [0.4 * total, 1.1 * T, 0.15 * T, 0.4 * total]
    # Keep the induction langaus a NARROW turn-on spike near threshold (eta_i capped) so it
    # cannot broaden to absorb the frozen shower -- the separation comes from the shower being
    # fixed. (Floating the shower amplitude was tried and made it WORSE: the near-threshold
    # langaus overlap lets the fit pull A_L DOWN to feed the induction component, inflating
    # f_ind. The induction-off template amplitude is the best available anchor.)
    lb = [0.0,    0.9 * T, 0.02 * T, 0.0]
    ub = [np.inf, 2.0 * T, 0.30 * T, np.inf]
    p0 = [min(max(v, lb[i] + 1e-9), ub[i] - 1e-9) for i, v in enumerate(p0)]
    sigma = (np.sqrt(raw) + 1.0) / max(n_shower, 1)    # per-shower Poisson errors
    try:
        popt, pcov = curve_fit(model, centres, counts, p0=p0, bounds=(lb, ub),
                               sigma=sigma, absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        return dict(ok=False, reason=str(exc), centres=centres, counts=counts,
                    width=width, edges=edges, threshold_e=threshold_e, n_shower=int(n_shower))
    aG, mpv_i, eta_i, aI = popt
    pred = model(centres, *popt)
    ndf = max(len(centres) - len(popt), 1)
    chi2 = float(np.sum(((counts - pred) / sigma) ** 2))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    tot_shw = A_L + aG
    f_ind = float(aI / (aI + tot_shw)) if (aI + tot_shw) > 0 else np.nan
    shape = peak_tail_observables(centres, counts, threshold_e)
    return dict(
        ok=True, kind="template", centres=centres, counts=counts, width=width,
        edges=edges, threshold_e=threshold_e, noise_e=float(noise_e), pred=pred, lg=lg,
        frozen=(mpv_L, eta_L, A_L, mu_G, sig_G), popt=popt, perr=perr, n_shower=int(n_shower),
        mpv_ind=float(mpv_i), mpv_ind_err=float(perr[1]),
        mpv_shw=float(mpv_L), mpv_shw_err=0.0,        # frozen -> no fit error
        mu_G=mu_G, sig_G=sig_G, A_G=float(aG), A_G_err=float(perr[0]),
        eta_ind=float(eta_i), area_ind_err=float(perr[3]),
        area_ind=float(aI), area_shw=float(tot_shw), frac_ind=f_ind,
        chi2=chi2, ndf=ndf, chi2ndf=chi2 / ndf, n_single=int(raw.sum()), **shape)


# ===========================================================================
# Aggregation across events
# ===========================================================================
class _silence_device_stdout:
    """Redirect C-level fd 1 to /dev/null so numba-cuda device print()s -- e.g.
    'More ADC values than possible, 30' from dense shower-core pixels (which are
    multi-hit and dropped by the single-hit cut anyway) -- don't spam the log.
    Python-level prints outside the `with` block are unaffected. No-op if stdout
    has no real file descriptor."""
    def __enter__(self):
        try:
            import os as _os
            self._fd = sys.stdout.fileno()
            sys.stdout.flush()
            self._saved = _os.dup(self._fd)
            devnull = _os.open(_os.devnull, _os.O_WRONLY)
            _os.dup2(devnull, self._fd)
            _os.close(devnull)
        except Exception:
            self._saved = None
        return self

    def __exit__(self, *exc):
        if getattr(self, "_saved", None) is not None:
            import os as _os
            sys.stdout.flush()
            _os.dup2(self._saved, self._fd)
            _os.close(self._saved)
        return False


def aggregate_hits(ctx, evs, threshold_e, reset_cycles, induction_scale,
                   seed0, pops=_HIT_POPS):
    """Run the FEE on cached pre-signals and concatenate each multiplicity population.

    Returns {n: (Q, truth, aux)}. `evs` is the list of cached nominal-induction pre-signals
    (used for the threshold and reset scans). For the induction scan, callers pass freshly
    re-induced pre-signals in `evs` (the pre-FEE current depends on induction).

    See `event_fee_hits` for why n=2 is kept alongside n=1: it is where single-hit pixels go
    when threshold or reset promotes them, so the pair carries a migration that the
    single-hit sample cannot show on its own.
    """
    set_periodic_reset(ctx, reset_cycles)
    acc = {n: ([], [], []) for n in pops}
    alla = ([], [], [])
    with _silence_device_stdout():
        for i, ev in enumerate(evs):
            res = event_fee_hits(ctx, ev, threshold_e, seed=seed0 + i, pops=pops)
            for n in pops:
                q, t, a = res[n]
                if q.size:
                    acc[n][0].append(q); acc[n][1].append(t); acc[n][2].append(a)
            qa, ta, na = res["all"]
            if qa.size:
                alla[0].append(qa); alla[1].append(ta); alla[2].append(na)
    out = {}
    for n in pops:
        qs, trs, auxs = acc[n]
        out[n] = (_empty_pop() if not qs else
                  (np.concatenate(qs), np.concatenate(trs),
                   {k: np.concatenate([a[k] for a in auxs]) for k in _AUX_KEYS}))
    out["all"] = (_empty_all() if not alla[0] else
                  (np.concatenate(alla[0]), np.concatenate(alla[1]),
                   np.concatenate(alla[2])))
    return out


def aggregate_singlehits(ctx, evs, threshold_e, reset_cycles, induction_scale,
                         seed0, n_events=None):
    """Single-hit (Q, truth, aux) only -- thin wrapper on `aggregate_hits`."""
    return aggregate_hits(ctx, evs, threshold_e, reset_cycles, induction_scale,
                          seed0, pops=(1,))[1]


def _pixel_category(n_hits, is_coll, q_coll):
    if n_hits == 1:
        return "coll1" if is_coll else "ind1"
    if n_hits > 1:
        return "multi"
    return "charged_nohit" if q_coll > 0 else "quiet"


def _crop_downsample(wave, ts, npts=340, pad=20):
    """Crop a per-tick waveform to its active window (+pad) and downsample to ~npts,
    returning (t_us, current, cumulative_charge_e). Charge = running integral of current."""
    w = np.asarray(wave, float)
    nz = np.nonzero(np.abs(w) > 1e-6)[0]
    if nz.size:
        a, b = max(int(nz[0]) - pad, 0), min(int(nz[-1]) + pad, w.size)
    else:
        a, b = 0, min(w.size, npts)
    idx = np.arange(a, b)
    if idx.size > npts:                                # even stride downsample
        idx = idx[np.linspace(0, idx.size - 1, npts).round().astype(int)]
    cur = w[idx]
    chg = np.cumsum(w[a:b])[idx - a] * ts               # integral over the FULL window
    return (idx * ts).astype(np.float32), cur.astype(np.float32), chg.astype(np.float32)


def capture_waveforms(ctx, evs, threshold_e, reset_cycles, seed0, label,
                      wf_events, max_pix):
    """Save per-pixel WAVEFORMS (induced current + running charge vs time) for a few events,
    for the standalone viewer. Curated across hit categories (single-hit collection, single-hit
    induction, multi-hit, charged-but-no-hit) so the viewer shows *why* each pixel did or
    didn't fire, alongside the truth (charge landed, its arrival time) and the recorded hit(s).
    Returns a list of per-pixel record dicts."""
    import cupy as cp
    if wf_events <= 0:
        return []
    set_periodic_reset(ctx, reset_cycles)
    ts = float(ctx.detector.TIME_SAMPLING)
    quotas = [("coll1", 14), ("ind1", 14), ("multi", 6), ("charged_nohit", 4), ("quiet", 2)]
    recs = []
    with _silence_device_stdout():
        for iev, ev in enumerate(evs[:wf_events]):
            adc, ticks = vd.run_fee(
                ctx, cp.asarray(ev["pixels_signals"]), cp.asarray(ev["pixels_tracks_signals"]),
                cp.asarray(ev["num_backtrack"]), cp.asarray(ev["offset_backtrack"]),
                ev["max_time"], seed=seed0 + iev)
            adc = np.asarray(adc); ticks = np.asarray(ticks)
            nh = (adc > 0).sum(axis=1)
            ps = ev["pixels_signals"]; ids = ev["unique_pix"]
            coll = ev["is_collection"]; qc = ev["q_coll"]
            xs, ys = vd.pixel_xy_from_id(ctx.detector, ids)
            cats = np.array([_pixel_category(int(nh[r]), bool(coll[r]), float(qc[r]))
                             for r in range(len(ids))])
            picked = []
            for cat, quota in quotas:                   # quota sample per category
                rows = np.nonzero(cats == cat)[0]
                if rows.size:
                    take = rows if rows.size <= quota else rows[
                        np.linspace(0, rows.size - 1, quota).round().astype(int)]
                    picked.extend(int(r) for r in take)
                if len(picked) >= max_pix:
                    break
            for r in picked[:max_pix]:
                hitmask = adc[r] > 0
                t_us, cur, chg = _crop_downsample(ps[r], ts)
                recs.append(dict(
                    sample=label, ev=int(iev), pix_id=int(ids[r]),
                    x=float(xs[r]), y=float(ys[r]), cat=str(cats[r]),
                    q_coll=float(qc[r]), t_coll=float(ev["t_coll"][r]),
                    t_near=float(ev["t_near"][r]),
                    sig_t=float(ev["sig_t_coll"][r]), n_hits=int(nh[r]),
                    thr=float(threshold_e), reset=int(reset_cycles),
                    hit_t=np.asarray(ticks[r][hitmask], np.float32),
                    hit_q=np.asarray(vd.adc_to_charge(ctx, adc[r][hitmask]), np.float32),
                    t=t_us, cur=cur, chg=chg))
    return recs


# ===========================================================================
# BURST-MODE (forced continuous readout) study
#   A through-going muon PARALLEL to the pixel plane (theta=0) lays a LINE of
#   collecting pixels at fixed drift depth; the pixels one-or-more pitches
#   TRANSVERSE to that line see a pure, sub-threshold, bipolar induction
#   transient and never self-trigger. "Burst mode" = a DAQ that force-samples
#   those pixels on a fixed cadence with NO discriminator gate, so the induction
#   WAVEFORM SHAPE is recorded. That shape is threshold-independent, so it is a
#   test of the induction model that escapes the threshold-calibration systematic
#   which limits the self-trigger induction RATE. See docs/threshold_induction_study.md.
# ===========================================================================
def muon_neighbor_pixels(ctx, ev, reach_pitches=3, include_collectors=False):
    """Select the BAND of pixels around a through-going muon track: the COLLECTORS the muon's
    charge lands on and/or the transverse NEIGHBOURS at +-1..+-reach pitches perpendicular to it.

    A theta=0 muon lies in the pixel plane at fixed drift depth, so its collectors form a LINE
    (the "column" of the track); pixels one-or-more pitches off that line see only induced
    current and never collect. The track direction is found by PCA on the collector pad centres
    (works at any azimuth, no knowledge of how the track was generated); we step perpendicular to
    it in pitch units, snap to the grid, and keep pixels that received current (have a row in
    `unique_pix`).

      include_collectors=False -> pure-induction neighbours only (dist != 0, is_collection False):
                                  the induction-model / falloff analysis.
      include_collectors=True  -> the FULL band: collectors (dist 0, is_collection True) AND the
                                  neighbours -- so a burst readout captures both the collection
                                  pulse on the track and the induced pulses beside it.

    Returns [{row, pix_id, x, y, dist (signed pitches; 0 = on-track), is_collection, q_coll}, ...];
    `row` indexes ev['pixels_signals'] so the caller reads the analog waveform directly.
    """
    from larndsim.pixels_from_track import pixel2id, id2pixel
    det = ctx.detector
    ids = np.asarray(ev["unique_pix"], np.int64)
    coll = np.asarray(ev["is_collection"], bool)
    qc = np.asarray(ev["q_coll"], float)
    xs, ys = vd.pixel_xy_from_id(det, ids)
    row_of = {int(p): r for r, p in enumerate(ids)}          # pixel id -> row in pixels_signals
    ci = np.nonzero(coll)[0]
    if ci.size < 2:                                          # need a line to define a transverse
        return []
    cx, cy = xs[ci], ys[ci]
    # principal (track) axis from the collector centroid scatter; perpendicular = transverse
    _u, _s, Vt = np.linalg.svd(np.vstack([cx - cx.mean(), cy - cy.mean()]).T,
                               full_matrices=False)
    track_hat = Vt[0]
    perp = np.array([-track_hat[1], track_hat[0]])
    pitch = float(det.PIXEL_PITCH)
    nx, ny = int(det.N_PIXELS[0]), int(det.N_PIXELS[1])
    out, seen = [], set()
    for k in ci:
        _ix, _iy, plane = (int(v) for v in id2pixel(int(ids[k])))
        x0 = float(det.TPC_BORDERS[plane, 0, 0]); y0 = float(det.TPC_BORDERS[plane, 1, 0])
        for step in range(-reach_pitches, reach_pitches + 1):
            if step == 0 and not include_collectors:         # skip the on-track pad unless wanted
                continue
            ixn = int(round((xs[k] + step * pitch * perp[0] - x0) / pitch))
            iyn = int(round((ys[k] + step * pitch * perp[1] - y0) / pitch))
            if not (0 <= ixn < nx and 0 <= iyn < ny):
                continue
            pid = int(pixel2id(ixn, iyn, plane))
            r = row_of.get(pid)
            if r is None or pid in seen:                      # off-grid or already captured
                continue
            is_c = bool(coll[r])
            if is_c and not include_collectors:               # a neighbour that is itself a collector
                continue
            seen.add(pid)
            out.append(dict(row=int(r), pix_id=pid, x=float(xs[r]), y=float(ys[r]),
                            dist=int(step), is_collection=is_c, q_coll=float(qc[r])))
    return out


def burst_readout(ctx, cur, cadence_ticks, noise_e, rng):
    """Burst-mode PIXEL READOUT of one pad's pre-FEE current: the charge a muon-tagged (external
    t0) forced readout records, digitised every `cadence_ticks` ticks with NO discriminator gate.

    The integrator accumulates only the POSITIVE current -- Integral(floor(current, 0)) -- across
    the whole drift; the negative Ramo lobe is NOT integrated. This matters at fsd_cube's granularity:
    the ADC LSB is ~997 e-, larger than a single induction transient (~hundreds of e-), so digitising
    each window independently would quantise the induction to zero. Accumulating the positive charge
    over the many windows of the drift lifts it above a count. Each sample digitises the running
    integrator value (+ readout noise) through the SAME fee.digitize / vd.adc_to_charge calibration
    the real chip uses, so the staircase carries the true ADC granularity. (The raw analog waveform,
    which keeps the negative lobe, is captured separately -- see `_crop_downsample`.)

    Returns (t_us, q_read_e): the digitised integrator value at each readout -- a rising staircase.
    """
    det = ctx.detector
    ts = float(det.TIME_SAMPLING)
    w = np.asarray(cur, float)
    W = max(int(cadence_ticks), 1)
    nwin = w.size // W
    if nwin == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    cum_pos = np.cumsum(np.maximum(w, 0.0)) * ts                # running positive-charge integral (e-)
    idx = np.arange(1, nwin + 1) * W - 1                        # integrator value at each window end
    q_int = cum_pos[idx] + rng.normal(0.0, noise_e, size=nwin)  # readout noise on the sampled integrator
    adc = np.asarray(ctx.fee.digitize(q_int, det.GAIN, det.V_PEDESTAL))
    q_read = np.asarray(vd.adc_to_charge(ctx, adc), float)      # e-, carrying the ~997 e- LSB
    t = (np.arange(1, nwin + 1) * W) * ts
    return t.astype(np.float32), q_read.astype(np.float32)


def capture_burst(ctx, drift_cache, sample, theta, scales, cadence_ticks, reach,
                  noise_e, seed0, collect_frac=0.15):
    """Burst-mode capture on a muon control sample: for each induction SCALE, re-induce the
    events and record the forced burst-mode waveform of every pixel in the BAND around the track
    -- the on-track COLLECTORS and the transverse NEIGHBOURS out to +-reach pitches. This is what
    a muon-tagged (external t0) continuous readout of the muon's column would see: the collection
    pulse on the track pads and the induced sub-threshold pulses beside them, from t0 over the
    whole readout window. The scale axis lets you watch the induced pulses change with the
    induction model (collectors are induction-independent); nominal scale 1.0 is the plain readout.

    Memory-flat: induces ONE event at a time and keeps only the (small) per-pixel waveforms, never
    a list of pre-signal caches. Event seeds depend on the event index ONLY (not the scale), so the
    sole difference between scales is the induction physics; a dedicated rng supplies the readout
    noise. Each record carries `t_near` -- the truth charge-arrival time -- so the ensemble can be
    aligned on arrival (not on a noisy argmax) and the plots can mark when the charge lands.
    """
    recs = []
    bn_rng = np.random.default_rng((int(seed0) & 0xFFFFFFFF) ^ 0x9E3779B9)
    for scale in scales:
        set_induction(ctx, float(scale))
        with _silence_device_stdout():
            for i, dc in enumerate(drift_cache):
                ev = event_induce(ctx, dc[0], dc[1], dc[2], seed=seed0 + i,
                                  collect_frac=collect_frac)
                if ev is None:
                    continue
                t_near = np.asarray(ev["t_near"], float)
                ts = float(ctx.detector.TIME_SAMPLING)
                for nb in muon_neighbor_pixels(ctx, ev, reach_pitches=reach,
                                               include_collectors=True):
                    cur_full = np.asarray(ev["pixels_signals"][nb["row"]], float)
                    t, q = burst_readout(ctx, cur_full, cadence_ticks, noise_e, bn_rng)  # DAQ readout
                    if q.size == 0:
                        continue
                    t_wf, cur_wf, chg_wf = _crop_downsample(cur_full, ts)   # raw analog (keeps -lobe)
                    qpos = float(np.sum(np.maximum(cur_full, 0.0)) * ts)    # total positive induced charge
                    recs.append(dict(sample=str(sample), theta=float(theta), scale=float(scale),
                                     dist=int(abs(nb["dist"])), sdist=int(nb["dist"]),
                                     is_collection=bool(nb["is_collection"]),
                                     ev=int(i), pix_id=int(nb["pix_id"]), q_coll=float(nb["q_coll"]),
                                     t_near=float(t_near[nb["row"]]) if t_near.size else np.nan,
                                     qpos=qpos, qread=float(q[-1]), peak=qpos,   # peak := clean analog amplitude
                                     t=t, q=q, t_wf=t_wf, cur=cur_wf, chg=chg_wf))
    set_induction(ctx, 1.0)
    return recs


def save_burst(recs, path):
    """Save burst-mode band waveforms (from capture_burst) to an .npz for --refit / viewing."""
    if not recs:
        return
    scal = ("sample", "theta", "scale", "dist", "sdist", "is_collection", "ev", "pix_id",
            "q_coll", "t_near", "qpos", "qread", "peak")
    _defaults = {"sdist": 0, "is_collection": False, "t_near": np.nan,
                 "qpos": np.nan, "qread": np.nan}                    # for pre-band records
    kw = {k: np.asarray([r.get(k, _defaults.get(k)) for r in recs]) for k in scal}
    for k in ("t", "q", "t_wf", "cur", "chg"):                       # readout staircase + raw analog
        kw[k] = _obj_array([r.get(k, np.zeros(0, np.float32)) for r in recs])
    np.savez(path, **kw)
    ncoll = int(np.sum([bool(r.get("is_collection")) for r in recs]))
    print(f"  wrote {len(recs)} burst-mode band waveforms to {path} "
          f"({ncoll} on-track, {len(recs) - ncoll} neighbour; readout staircase + raw analog each)")


def load_burst(path):
    """Inverse of save_burst -> list of records (empty list if the file is absent). Tolerant of
    older files that predate the on-track collectors / raw analog (missing fields default in)."""
    if not path or not os.path.exists(path):
        return []
    d = np.load(path, allow_pickle=True)
    if "sample" not in d.files:
        return []
    n = len(d["sample"])
    have = set(d.files)
    def col(name, default, cast):
        return (lambda i: cast(d[name][i])) if name in have else (lambda i: default)
    def arr(name):
        return (lambda i: np.asarray(d[name][i], float)) if name in have \
            else (lambda i: np.zeros(0))
    sdist = col("sdist", 0, int); iscol = col("is_collection", False, bool)
    tnear = col("t_near", np.nan, float); qpos = col("qpos", np.nan, float)
    qread = col("qread", np.nan, float)
    twf, cur, chg = arr("t_wf"), arr("cur"), arr("chg")
    return [dict(sample=str(d["sample"][i]), theta=float(d["theta"][i]),
                 scale=float(d["scale"][i]), dist=int(d["dist"][i]), sdist=sdist(i),
                 is_collection=iscol(i), ev=int(d["ev"][i]), pix_id=int(d["pix_id"][i]),
                 q_coll=float(d["q_coll"][i]), t_near=tnear(i), qpos=qpos(i), qread=qread(i),
                 peak=float(d["peak"][i]), t=np.asarray(d["t"][i], float),
                 q=np.asarray(d["q"][i], float),
                 t_wf=twf(i), cur=cur(i), chg=chg(i)) for i in range(n)]


def _mean_on_arrival(grp, grid, tkey="t", vkey="q", right=0.0):
    """Interpolate each band waveform onto a common time-since-charge-arrival grid (t - t_near)
    and average. Using the TRUTH arrival time (not a noisy argmax) keeps the mean pulse sharp.
    `tkey`/`vkey` pick the readout staircase ('t'/'q') or the raw analog ('t_wf'/'cur' or 'chg').
    `right` is the fill past the last sample (0 for a transient, last value for a staircase).
    Returns (mean, sem, n) or (None, None, 0)."""
    acc = []
    for r in grp:
        t = np.asarray(r.get(tkey, ()), float); v = np.asarray(r.get(vkey, ()), float)
        tn = r.get("t_near", np.nan)
        if not np.isfinite(tn) or t.size < 2 or v.size != t.size:
            continue
        rt = t[-1] if right == "last" else right
        acc.append(np.interp(grid, t - tn, v, left=0.0, right=rt))
    if not acc:
        return None, None, 0
    A = np.vstack(acc)
    return A.mean(axis=0), A.std(axis=0) / sqrt(A.shape[0]), A.shape[0]


def _adc_lsb(recs, default=997.0):
    """Infer the ADC least-significant bit (e-) from the digitised readout: adc_to_charge is
    linear in the integer ADC code, so distinct recorded charges are spaced by exactly one LSB,
    and the smallest positive gap between them is that LSB."""
    vals = [np.asarray(r.get("q", ()), float) for r in recs[:3000]]
    vals = np.concatenate([v for v in vals if v.size]) if vals else np.zeros(0)
    u = np.unique(np.round(vals[np.isfinite(vals)], 2))
    dq = np.diff(u)
    dq = dq[dq > 1.0]
    return float(np.min(dq)) if dq.size else default


def analyze_burst(recs, outdir, noise_e):
    """The burst-mode outputs. First the COLUMN READOUT the user asked for: the mean forced-readout
    waveform of the muon's band -- the on-track collectors and the transverse neighbours -- aligned
    on the charge-arrival time (= tagger t0 + drift time), so you see the collection pulse on the
    track and the induced pulses beside it. Then the neighbour-only analyses: (1) mean induction
    transient by distance, (2) amplitude falloff nominal vs mis-model, (3) model-discrimination
    power vs muon count. All run from the saved ti_burst.npz, so they iterate locally with --refit;
    the per-pixel waveforms are in that file for your own plots too."""
    if not recs:
        return
    import matplotlib.pyplot as plt
    print("\n=== Burst-mode column readout (forced, muon-tagged t0) ===")
    samples = sorted(set(r["sample"] for r in recs))
    scales = sorted(set(r["scale"] for r in recs))
    nom = 1.0 if 1.0 in scales else scales[len(scales) // 2]

    # ---- (0) THE COLUMN READOUT: two views of the band, aligned on charge arrival ----
    #   TOP row  = the RAW analog induced-current waveform (keeps the negative Ramo lobe);
    #   BOTTOM   = the DAQ pixel readout = accumulated Integral(floor(current,0)), digitised.
    #   The bottom row shows why fsd_cube's ~997 e- LSB matters: a single induction transient is
    #   sub-LSB, so only the accumulated positive charge over the drift clears one count.
    lsb = _adc_lsb(recs)
    print(f"  ADC LSB ~ {lsb:.0f} e- (fsd_cube); a single induction transient is sub-LSB, so the "
          f"per-window\n  digitised readout can't see it -- the ACCUMULATED positive integral is "
          f"the observable.")
    dtwf = next((float(r["t_wf"][1] - r["t_wf"][0]) for r in recs
                 if len(r.get("t_wf", ())) > 1), 0.1)
    gwf = np.arange(-12.0, 8.0 + dtwf, dtwf)                      # fine grid for the transient
    dt = next((float(r["t"][1] - r["t"][0]) for r in recs if len(r["t"]) > 1), 1.0)
    grd = np.arange(-40.0, 20.0 + dt, dt)                        # coarse grid for the staircase
    cmap = plt.get_cmap("viridis")
    for sample in samples:
        base = [r for r in recs if r["sample"] == sample and r["scale"] == nom]
        coll = [r for r in base if r.get("is_collection", False)]
        nbrs = [r for r in base if not r.get("is_collection", False)]
        maxd = max((r["dist"] for r in nbrs), default=0)
        if not coll and maxd == 0:
            continue
        fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0))
        # TOP: raw analog current (bipolar, negative lobe)
        mc, sc, nc = _mean_on_arrival(coll, gwf, tkey="t_wf", vkey="cur")
        if mc is not None:
            ax[0, 0].plot(gwf, mc, color=_C_SHW, lw=1.6)
            ax[0, 0].fill_between(gwf, mc - sc, mc + sc, color=_C_SHW, alpha=0.2, lw=0)
        ax[0, 0].set_title(f"on-track collectors — induced current  (N={nc})", fontsize=10)
        ax[0, 0].set_ylabel("mean current (resp. units)")
        for di in range(1, maxd + 1):
            mn, se, nn = _mean_on_arrival([r for r in nbrs if r["dist"] == di], gwf,
                                          tkey="t_wf", vkey="cur")
            if mn is None:
                continue
            c = cmap(0.12 + 0.72 * (di - 1) / max(maxd - 1, 1))
            ax[0, 1].plot(gwf, mn, color=c, lw=1.5, label=f"{di} pitch (N={nn})")
            ax[0, 1].fill_between(gwf, mn - se, mn + se, color=c, alpha=0.15, lw=0)
        ax[0, 1].set_title("neighbours — induced current (bipolar: +lobe leads, −lobe follows)",
                           fontsize=10)
        ax[0, 1].legend(fontsize=8)
        # BOTTOM: DAQ readout = accumulated positive-charge staircase, digitised
        mc, sc, nc = _mean_on_arrival(coll, grd, tkey="t", vkey="q", right="last")
        if mc is not None:
            ax[1, 0].plot(grd, mc, color=_C_SHW, lw=1.6, drawstyle="steps-mid")
        ax[1, 0].set_title("on-track collectors — DAQ readout (∫ floor(I,0))", fontsize=10)
        ax[1, 0].set_ylabel(r"mean recorded charge  ($e^-$)")
        for di in range(1, maxd + 1):
            mn, se, nn = _mean_on_arrival([r for r in nbrs if r["dist"] == di], grd,
                                          tkey="t", vkey="q", right="last")
            if mn is None:
                continue
            c = cmap(0.12 + 0.72 * (di - 1) / max(maxd - 1, 1))
            ax[1, 1].plot(grd, mn, color=c, lw=1.5, drawstyle="steps-mid", label=f"{di} pitch")
        ax[1, 1].axhline(lsb, color="0.4", ls="--", lw=1.0, label="1 ADC LSB (%.0f $e^-$)" % lsb)
        ax[1, 1].set_title("neighbours — DAQ readout (sub-LSB unless accumulated)", fontsize=10)
        ax[1, 1].legend(fontsize=8)
        for a in ax.flat:
            a.axvline(0.0, color="0.5", ls=":", lw=1.0)
            a.grid(alpha=0.25, lw=0.6)
        for a in ax[1]:
            a.set_xlabel(r"time since charge arrival = $t_0$ + drift  ($\mu$s)")
        fig.suptitle(f"Burst-mode column readout — {sample}  (induction $\\times{nom:g}$): "
                     f"raw current (top) vs DAQ readout (bottom)", fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/ti_burst_column_{sample}.png")

    # the induction analyses below are about the NON-collecting neighbours only
    nrecs = [r for r in recs if not r.get("is_collection", False)]
    if not nrecs:
        return
    dists = sorted(set(r["dist"] for r in nrecs))

    def sel(sample, scale, dist):
        return [r for r in nrecs if r["sample"] == sample and r["scale"] == scale
                and r["dist"] == dist]

    # (2) induction-amplitude falloff with transverse distance, one curve per scale
    fig, axes = plt.subplots(1, len(samples), figsize=(4.8 * len(samples), 4.2),
                             squeeze=False)
    for si, sample in enumerate(samples):
        ax = axes[0, si]
        for scale in scales:
            dd, mean, sem = [], [], []
            for dist in dists:
                g = sel(sample, scale, dist)
                if len(g) < 2:
                    continue
                pk = np.array([r["peak"] for r in g])
                dd.append(dist); mean.append(pk.mean()); sem.append(pk.std() / sqrt(len(pk)))
            if dd:
                ax.errorbar(dd, mean, yerr=sem, marker="o", capsize=3, lw=1.5,
                            label=(r"$\times%g$ (nominal)" % scale) if scale == nom
                            else r"$\times%g$" % scale)
        ax.axhspan(-noise_e, noise_e, color="0.85", alpha=0.6, zorder=0)
        ax.set_xlabel("transverse distance (pitches)")
        ax.set_title(sample, fontsize=10.5)
        if si == 0:
            ax.set_ylabel(r"mean positive induced charge $\int\!\lfloor I,0\rfloor$  ($e^-$)")
        ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=8)
    fig.suptitle("Induction amplitude falloff — nominal vs mis-modelled induction", fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_burst_falloff.png")

    # (3) discriminating power: significance of a mis-model vs nominal, and its sqrt(N) growth
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    print("  model-discrimination at the nearest neighbour (dist=1 pitch):")
    print("    %-14s %8s %6s %9s %9s %9s" %
          ("sample", "mismodel", "N_nb", "N_mu", "sigma", "sigma@1e3mu"))
    drew = False
    for sample in samples:
        g0 = [r for r in nrecs if r["sample"] == sample and r["scale"] == nom and r["dist"] == 1]
        p0 = np.array([r["peak"] for r in g0], float)
        for scale in scales:
            if scale == nom:
                continue
            g1 = [r for r in nrecs if r["sample"] == sample and r["scale"] == scale
                  and r["dist"] == 1]
            p1 = np.array([r["peak"] for r in g1], float)
            if p0.size < 3 or p1.size < 3:
                continue
            comb = sqrt(p0.var() / p0.size + p1.var() / p1.size)
            sig = abs(p1.mean() - p0.mean()) / comb if comb > 0 else np.nan
            nmu = len(set(r["ev"] for r in g1)) or 1
            sig_1e3 = sig * sqrt(1000.0 / nmu)
            print("    %-14s %8s %6d %6d %9.2f %9.1f" %
                  (sample, "x%g" % scale, p1.size, nmu, sig, sig_1e3))
            # significance vs muon count by subsampling the neighbour pool (sqrt-N growth)
            ev1 = np.array([r["ev"] for r in g1]); ev0 = np.array([r["ev"] for r in g0])
            uev = np.array(sorted(set(ev1) | set(ev0)))
            if uev.size >= 4:
                ns = np.unique(np.linspace(2, uev.size, 8).round().astype(int))
                xs_, ys_ = [], []
                for k in ns:
                    keep = set(uev[:k])
                    a = p0[np.isin(ev0, list(keep))]; b = p1[np.isin(ev1, list(keep))]
                    if a.size < 3 or b.size < 3:
                        continue
                    c = sqrt(a.var() / a.size + b.var() / b.size)
                    xs_.append(k); ys_.append(abs(b.mean() - a.mean()) / c if c > 0 else np.nan)
                if xs_:
                    ax.plot(xs_, ys_, marker="o", lw=1.5, label=f"{sample}  $\\times{scale:g}$")
                    drew = True
    ax.axhline(3.0, color=_C_SHW, ls="--", lw=1.0, label=r"$3\sigma$")
    ax.set_xlabel("muons used"); ax.set_ylabel(r"separation from nominal ($\sigma$)")
    ax.set_title("Burst-mode discriminating power vs muon count (dist = 1 pitch)")
    ax.grid(alpha=0.25, lw=0.6)
    if drew:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_burst_discrimination.png")


def analyze_energy_scan(escan, meta, outdir):
    """Shower-energy dependence of induction (Part D). For each energy the run recorded the
    induction ON and OFF single-hit and all-multiplicity populations; here we plot the
    pure-induction single-hit rate, the multiplicity-inclusive fired rate ON vs OFF, and the
    mean collected charge, all per shower, vs energy -- the test of whether denser (higher-E)
    cores induce more strongly. Rows: (E, n_ev, q1on,tr1on, qAon,trAon,nAon, q1off,tr1off,
    qAoff,trAoff,nAoff)."""
    if not escan:
        return
    import matplotlib.pyplot as plt
    print("\n=== Shower-energy scan (induction vs shower energy) ===")
    E = np.array([r[0] for r in escan], float)
    o = np.argsort(E); E = E[o]
    esc = [escan[i] for i in o]
    rate_ind, rate_on, rate_off, meanq = [], [], [], []
    print("    %8s %7s %12s %12s %12s %12s" %
          ("E[MeV]", "n_ev", "pureInd/ev", "fired_on/ev", "fired_off/ev", "meanQ_on"))
    for r in esc:
        (_E, nev, q1on, tr1on, qAon, _trAon, _nAon, _q1off, _tr1off, qAoff, _trAoff, _nAoff) = r
        nev = max(int(nev), 1)
        q1on = np.asarray(q1on, float); tr1on = np.asarray(tr1on, bool)
        n_pure = int((~tr1on).sum())
        ri = n_pure / nev
        ron = np.asarray(qAon, float).size / nev
        roff = np.asarray(qAoff, float).size / nev
        mq = float(np.mean(np.asarray(qAon, float))) if np.asarray(qAon, float).size else np.nan
        rate_ind.append(ri); rate_on.append(ron); rate_off.append(roff); meanq.append(mq)
        print("    %8.0f %7d %12.4f %12.3f %12.3f %12.0f" % (_E, nev, ri, ron, roff, mq))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(E, rate_ind, "o-", color=_C_IND, lw=1.6)
    axes[0].set_ylabel("pure-induction single hits / shower")
    axes[0].set_title("Direct induction hits")
    axes[1].plot(E, rate_on, "o-", color=_C_SHW, label="induction ON", lw=1.6)
    axes[1].plot(E, rate_off, "s--", color=_C_SUM, label="induction OFF", lw=1.4)
    axes[1].set_ylabel("fired pixels / shower (all multiplicities)")
    axes[1].set_title("Inclusive fired rate")
    axes[1].legend(fontsize=8)
    axes[2].plot(E, meanq, "o-", color=_C_GRN, lw=1.6)
    axes[2].set_ylabel(r"mean collected charge / pixel  ($e^-$)")
    axes[2].set_title("Charge scale")
    for ax in axes:
        ax.set_xlabel("shower energy (MeV)"); ax.set_xscale("log")
        ax.grid(alpha=0.25, lw=0.6)
    fig.suptitle("Shower-energy dependence of induction", fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_induction_vs_energy.png")


def plot_inclusive_rate(name, sm, n_ev, ts, outdir, suffix):
    """Part B: the MULTIPLICITY-INCLUSIVE fired-pixel rate, induction ON vs OFF, vs threshold and
    reset. The single-hit rate cancels by construction (induction creates pure-induction single
    hits but promotes collection pixels OUT of the single-hit sample); the inclusive rate should
    not, so ON-minus-OFF here is the non-cancelling induction signature the single-hit cut hides."""
    import matplotlib.pyplot as plt
    blocks = []
    for knob, xlabel in (("threshold", r"$Q_{\mathrm{thr}}$  ($10^3\,e^-$)"),
                         ("reset", "reset rate (kHz)")):
        sca = sm.get(f"scan_{knob}_allmult")
        if not sca:
            continue
        vals = np.array([r[0] for r in sca], float)
        x = (reset_rate_khz(vals.astype(int), ts) if knob == "reset" else vals / 1e3)
        ron = np.array([np.asarray(r[1], float).size for r in sca], float) / max(n_ev, 1)
        roff = np.array([np.asarray(r[4], float).size for r in sca], float) / max(n_ev, 1)
        blocks.append((knob, xlabel, x, ron, roff))
    if not blocks:
        return
    fig, axes = plt.subplots(1, len(blocks), figsize=(4.8 * len(blocks), 4.0), squeeze=False)
    for j, (knob, xlabel, x, ron, roff) in enumerate(blocks):
        ax = axes[0, j]; o = np.argsort(x)
        ax.plot(x[o], ron[o], "o-", color=_C_SHW, lw=1.5, label="induction ON")
        ax.plot(x[o], roff[o], "s--", color=_C_SUM, lw=1.3, label="induction OFF")
        ax.set_xlabel(xlabel); ax.set_title(knob, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6)
        if j == 0:
            ax.set_ylabel("fired pixels / event (all multiplicities)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Multiplicity-inclusive fired rate (ON vs OFF) — {name}", fontsize=11)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_inclusive_rate{suffix}.png")


def recommend_operating_point(spectra, sample_results, outdir, noise_e):
    """Part C: synthesise the self-trigger scans and the burst-mode model test into an
    operating-point recommendation -- which (threshold, reset, readout-mode) lets FSD Cube
    measure induction, and with what dominant systematic."""
    import matplotlib.pyplot as plt
    m = spectra["meta"]
    n_main = int(m["n_shower_main"]); base_thr = float(m["base_threshold"])
    print("\n=== Operating-point recommendation (FSD Cube induction measurement) ===")

    # self-trigger: pure-induction single-hit rate vs threshold, shower + each muon topology
    curves = []
    thr_rows = spectra.get("threshold") or []
    if thr_rows:
        v = np.array([r[0] for r in thr_rows], float)
        ri = np.array([int((~np.asarray(r[2], bool)).sum()) for r in thr_rows], float) / max(n_main, 1)
        curves.append(("shower", v, ri))
    for nm, sm in spectra.get("samples", {}).items():
        sc = sm.get("scan_threshold")
        if not sc:
            continue
        ns = int(sm.get("n_scan", sm.get("n", 1))) or 1
        v = np.array([r[0] for r in sc], float)
        ri = np.array([int((~np.asarray(r[2], bool)).sum()) for r in sc], float) / ns
        curves.append((nm, v, ri))

    # burst-mode: best mis-model significance at the nearest neighbour, extrapolated to 1000 muons
    burst = spectra.get("burst") or []
    best = None
    if burst:
        scales = sorted(set(r["scale"] for r in burst))
        noms = 1.0 if 1.0 in scales else scales[len(scales) // 2]
        for sample in sorted(set(r["sample"] for r in burst)):
            p0 = np.array([r["peak"] for r in burst
                           if r["sample"] == sample and r["scale"] == noms and r["dist"] == 1], float)
            for scale in scales:
                if scale == noms:
                    continue
                g1 = [r for r in burst if r["sample"] == sample and r["scale"] == scale
                      and r["dist"] == 1]
                p1 = np.array([r["peak"] for r in g1], float)
                if p0.size < 3 or p1.size < 3:
                    continue
                comb = sqrt(p0.var() / p0.size + p1.var() / p1.size)
                if comb <= 0:
                    continue
                sig = abs(p1.mean() - p0.mean()) / comb
                nmu = len(set(r["ev"] for r in g1)) or 1
                s1e3 = sig * sqrt(1000.0 / nmu)
                if best is None or s1e3 > best[3]:
                    best = (sample, scale, sig, s1e3, nmu)

    if curves:
        print("  self-trigger pure-induction single hits / event vs threshold:")
        for nm, v, ri in curves:
            o = np.argsort(v)
            print(f"    {nm:>13}: " + "  ".join(f"{vv/1e3:.1f}k={rr:.3f}" for vv, rr in zip(v[o], ri[o])))
    if best is not None:
        print(f"  burst-mode model test: best = {best[0]} x{best[1]:g} -> {best[2]:.1f} sigma at "
              f"{best[4]} muons ({best[3]:.0f} sigma at 1000 muons), threshold-INDEPENDENT.")
    print("  RECOMMENDATION:")
    print("    * self-trigger induction is a RATE: needs a LOW threshold (<=2-3 ke-) and per-pixel")
    print("      threshold calibration to ~1%; periodic reset is a weak lever on the spectrum.")
    print("    * BURST-MODE (forced readout) of theta=0 muon transverse neighbours measures the")
    print("      induction WAVEFORM SHAPE -- threshold-independent, the robust configuration.")

    if curves:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for nm, v, ri in curves:
            o = np.argsort(v)
            ax.plot(v[o] / 1e3, ri[o], "o-", lw=1.5, label=nm)
        ax.axvline(base_thr / 1e3, color="0.6", ls=":", lw=1.0, label="nominal thr")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel(r"threshold  ($10^3\,e^-$)")
        ax.set_ylabel("pure-induction single hits / event")
        ttl = "Self-trigger induction yield vs threshold"
        if best is not None:
            ttl += f"\nburst-mode: {best[3]:.0f}$\\sigma$ model test at 1000 muons (threshold-independent)"
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/ti_operating_point.png")


def categorize(q, is_collection):
    """2-way truth split of single-hit pixels from the backtrack collection flag:

      * induction -- drifting charge never landed on the pad (pure induced transient),
      * shower    -- drifting charge collected on the pad (a real deposit).

    `is_collection` is the per-pixel boolean from collection_pixels_from_signals;
    `q` is accepted for a uniform signature but is not used (the label is purely the
    backtrack, so it stays valid on data-free truth)."""
    is_shw = np.asarray(is_collection, bool)
    return ~is_shw, is_shw


def peak_tail_observables(centres, counts, threshold_e):
    """DATA-BLIND handles read straight off the recorded-Q spectrum -- no truth, no fit, and
    no knowledge of the true threshold. Returns a dict.

    POSITIONS (in electrons) -- these track THRESHOLD:
      peak_mpv    -- the near-threshold peak position (mode of the spectrum),
      tail_peak   -- the high-Q tail/shoulder maximum, found fit-free by smoothing the
                     spectrum above 2*peak_mpv and taking its argmax. This is the observable
                     behind "threshold pushes the fat tail to higher Q". NB it is the maximum
                     of the TOTAL spectrum there (the falling langaus tail plus the Gaussian),
                     so it sits systematically BELOW the fit's deconvolved mu_G -- ~25% low on
                     this sample. The two are strongly correlated (both rise ~70% over
                     Q_thr = 4->6 ke-) but are not the same quantity; use tail_peak as the
                     data-blind threshold tracker, mu_G as the template's tail centroid.
      peak_height, tail_height -- amplitudes, kept for continuity.

    SHAPE (dimensionless) -- measured in x = Q / peak_mpv, i.e. charge in units of the
    MEASURED peak position. Rescaling by the peak divides out the threshold, which is what
    makes these nearly threshold-INDEPENDENT and therefore able to isolate reset/induction:
      tail_over_core -- N(x>2) / N(x<1.5), the tail-to-core weight ratio,
      shoulder_frac  -- N(2.5<x<5) / N, the mid-tail shoulder that periodic reset builds up
                        (reset moves weight out of the near-threshold peak into the tail
                        without moving where the tail sits).
    """
    counts = np.asarray(counts, float)
    centres = np.asarray(centres, float)
    nan = dict(peak_mpv=np.nan, peak_height=np.nan, tail_height=np.nan, tail_peak=np.nan,
               tail_over_core=np.nan, shoulder_frac=np.nan)
    if counts.sum() <= 0:
        return nan
    i = int(np.argmax(counts))
    peak_mpv = float(centres[i]); peak_height = float(counts[i])
    tail = (centres > 2.5 * threshold_e) & (counts > 0)
    tail_height = float(np.median(counts[tail])) if tail.any() else 0.0

    # fit-free tail-peak locator: smooth (3-bin) above 2*peak, take the maximum
    hi = centres > 2.0 * peak_mpv
    tail_peak = np.nan
    if hi.sum() >= 3:
        c = counts[hi]
        sm = np.convolve(c, np.ones(3) / 3.0, mode="same")
        tail_peak = float(centres[hi][int(np.argmax(sm))])

    x = centres / max(peak_mpv, 1e-9)          # scale-free charge
    n_tot = counts.sum()
    n_core = counts[x < 1.5].sum()
    tail_over_core = float(counts[x > 2.0].sum() / n_core) if n_core > 0 else np.nan
    shoulder_frac = float(counts[(x > 2.5) & (x < 5.0)].sum() / n_tot)
    return dict(peak_mpv=peak_mpv, peak_height=peak_height, tail_height=tail_height,
                tail_peak=tail_peak, tail_over_core=tail_over_core,
                shoulder_frac=shoulder_frac)


# ===========================================================================
# Shower-template library: fit the induction-OFF langaus over (threshold x reset)
# and look for a closed-form law for each langaus parameter
# ===========================================================================
def reset_rate_khz(cycles, ts):
    """Periodic-reset RATE (kHz) from PERIODIC_RESET_CYCLES; <=0 -> 0 (reset off)."""
    cycles = np.asarray(cycles, float)
    return np.where(cycles > 0, 1.0e3 / (cycles * ts), 0.0)


def aggregate_template_grid(ctx, evs_off, thresholds, resets, seed0,
                            base_thr=None, base_reset=None, mode="cross"):
    """GPU: induction-OFF single-hit spectra over the (threshold x reset) template grid.
    Cheap per node -- NO re-induction, only FEE re-runs on the cached pre-signals; reset
    changes recompile the fee kernel so loop reset OUTER to share each recompile.

    `mode` controls WHICH nodes are computed, and this is the dominant cost knob of the whole
    study (|thr| x |reset| FEE passes over every event):

      "cross" (default) -- only nodes on the cross through the base point: every threshold at
            the base reset, plus every reset at the base threshold. This is EXACTLY the set the
            scans request (the threshold scan asks for (thr_i, base_reset), the reset scan for
            (base_thr, reset_j), the nominal/induction fits for the centre), so no fit loses
            any accuracy. The interior nodes only ever served to over-constrain the diagnostic
            closed-form fit. For a 4x6 grid this is 9 nodes instead of 24 -- a 2.7x saving on
            the study's largest cost.
      "full"  -- the whole Cartesian product, for a better-constrained closed-form fit.

    Returns [(thr, cycles, q, tr, aux), ...]; fit them later with fit_template_grid."""
    thr_arr, rst_arr = np.asarray(thresholds, float), np.asarray(resets, int)
    bt = float(base_thr) if base_thr is not None else float(thr_arr[0])
    br = int(base_reset) if base_reset is not None else int(rst_arr[0])
    out = []
    for ir, rst in enumerate(rst_arr):                    # reset OUTER -> one recompile per rate
        for it, thr in enumerate(thr_arr):
            on_cross = (int(rst) == br) or (abs(float(thr) - bt) < 1e-6)
            if mode == "cross" and not on_cross:
                continue
            q, tr, aux = aggregate_singlehits(ctx, evs_off, float(thr), int(rst), 0.0,
                                              seed0=seed0 + ir * 1000 + it)
            out.append((float(thr), int(rst), q, tr, aux))
    return out


def fit_template_grid(grid_spectra, thresholds, resets, ts, qmax, n_shower, noise_e):
    """CPU: fit fit_shower_template on each induction-OFF spectrum. Returns (grid,
    node_fits): `grid` holds each langaus param as a (n_reset, n_threshold) array plus the
    axes; `node_fits[(thr, cycles)]` is the per-node template dict (diagnostics + the exact
    on-grid template used by the composite fits)."""
    thr_arr = np.asarray(thresholds, float)
    rst_arr = np.asarray(resets, int)
    keys = ("mpv_L", "eta_L", "A_L", "mu_G", "sig_G", "chi2ndf")
    ekeys = ("mpv_L_err", "eta_L_err", "A_L_err", "mu_G_err", "sig_G_err")
    grid = {k: np.full((len(rst_arr), len(thr_arr)), np.nan) for k in keys + ekeys}
    idx = {(float(t), int(r)): (ir, it)
           for ir, r in enumerate(rst_arr) for it, t in enumerate(thr_arr)}
    node_fits = {}
    for (thr, rst, q, tr, _aux) in grid_spectra:
        coll = np.asarray(tr, bool)
        qs = np.asarray(q, float)[coll] if coll.any() else np.asarray(q, float)  # ~all shower
        tmpl = fit_shower_template(qs, float(thr), qmax=qmax, n_shower=n_shower, noise_e=noise_e)
        node_fits[(float(thr), int(rst))] = tmpl
        if tmpl.get("ok") and (float(thr), int(rst)) in idx:
            ir, it = idx[(float(thr), int(rst))]
            for k in keys + ekeys:
                grid[k][ir, it] = tmpl[k]
    grid["thresholds"] = thr_arr
    grid["resets"] = rst_arr
    grid["rates"] = reset_rate_khz(rst_arr, ts)
    return grid, node_fits


def _cf_library():
    """PHYSICALLY-MOTIVATED closed-form candidates for each frozen-template parameter as a
    function of (Q_thr, reset rate f). x = Q_thr/1e3 (10^3 e-), r = f (kHz).

    The template is fit to the induction-OFF single-hit shower spectrum, which is one
    threshold-INDEPENDENT underlying peripheral-collection charge spectrum TRUNCATED at the
    operating threshold. Each parameter is therefore a specific functional of that spectrum,
    which fixes its form (not a free polynomial):

      * mpv_L  -- the recorded peak RIDES the threshold: it sits an ~fixed distance above the
                  turn-on, so MPV = c0 + c1*Q_thr with c1 ~ 1 (affine).
      * A_L    -- the surviving shower yield is the spectrum's tail ABOVE threshold, i.e. the
                  complementary CDF of the peripheral charge spectrum. Its decay form MEASURES
                  that spectrum: exponential (scale Q0) if dN/dQ ~ e^{-Q/Q0}, or power-law if
                  dN/dQ ~ Q^{-a}. Both are offered; AIC picks -> a physics result.
      * eta_L  -- the Landau width tracks the LOCAL charge scale of the surviving slice, so it
                  grows ~linearly from a noise/binning floor: eta = c0 + c1*Q_thr (affine).
      * mu_G   -- the high-Q Gaussian is the core-adjacent shoulder; its centroid SLIDES UP as
                  the threshold eats the shoulder's low-Q side: mu_G = c0 + c1*Q_thr (affine).

    The reset factor (1 + c2*f) is a small linear charge-loss correction (reset periodically
    chops accumulated charge); on this sample c2 comes out ~0 for the position parameters."""
    def affine_r(X, c0, c1, c2):
        x, r = X; return (c0 + c1 * x) * (1.0 + c2 * r)
    def exp_r(X, c0, c1, c2):
        x, r = X; return c0 * np.exp(-x / (abs(c1) + 1e-6)) * (1.0 + c2 * r)
    def power_r(X, c0, c1, c2):
        x, r = X; return c0 * np.power(np.clip(x, 1e-3, None), c1) * (1.0 + c2 * r)

    def seed_affine(y, x, r):
        try:
            c1, c0 = np.polyfit(x, y, 1)
        except Exception:
            c0, c1 = float(np.mean(y)), 0.0
        return [float(c0), float(c1), 0.0]
    def seed_exp(y, x, r):
        ym = np.clip(y, 1e-9, None)
        try:
            b, a = np.polyfit(x, np.log(ym), 1)
            c1 = (-1.0 / b) if b < 0 else float(np.mean(x)); c0 = float(np.exp(a))
        except Exception:
            c0, c1 = float(np.max(y)), float(np.mean(x))
        return [c0, float(abs(c1)), 0.0]
    def seed_power(y, x, r):
        xm = np.clip(x, 1e-3, None); ym = np.clip(np.abs(y), 1e-9, None)
        try:
            c1 = float(np.clip(np.polyfit(np.log(xm), np.log(ym), 1)[0], -6.0, 6.0))
        except Exception:
            c1 = -1.0
        return [float(np.median(y / np.power(xm, c1))), c1, 0.0]

    AFF = (r"$(c_0+c_1 Q_{thr})(1+c_2 f)$", affine_r, seed_affine)
    return dict(
        mpv_L=[("peak rides threshold",) + AFF],
        eta_L=[(r"width $\propto$ charge scale",) + AFF],
        mu_G=[("high-Q centroid slides up",) + AFF],
        sig_G=[(r"width $\propto$ charge scale",) + AFF],
        A_L=[("spectrum tail above thr. (exp.)", r"$c_0\,e^{-Q_{thr}/c_1}(1+c_2 f)$", exp_r, seed_exp),
             ("spectrum tail above thr. (power)", r"$c_0\,Q_{thr}^{c_1}(1+c_2 f)$", power_r, seed_power)],
    )


def _fit_closed_form(x, r, y, yerr=None, forms=None):
    """Fit the physically-motivated candidate form(s) for one parameter over (x=Q_thr/1e3,
    r=rate kHz), WEIGHTED by the per-node fit uncertainties `yerr`. When a parameter has more
    than one motivated form (A_L: exp vs power), AIC selects. Returns the best as a dict
    (note, name, params, r2, red_chi2, aic, func) or None."""
    from scipy.optimize import curve_fit
    if not forms:
        return None
    good = np.isfinite(y) & np.isfinite(x) & np.isfinite(r)
    if yerr is not None:
        good &= np.isfinite(yerr)
    x, r, y = x[good], r[good], y[good]
    if y.size < 3:
        return None
    if yerr is not None:
        # floor tiny/zero errors at 3% of the value so no single node dominates the weight
        w = np.maximum(np.asarray(yerr, float)[good], 0.03 * (np.abs(y) + 1e-9))
    else:
        w = None
    X = np.vstack([x, r])
    sst = float(np.sum((y - y.mean()) ** 2))
    best = None
    for note, name, func, seed in forms:
        k = func.__code__.co_argcount - 1
        if y.size <= k:
            continue
        try:
            popt, _ = curve_fit(func, X, y, p0=seed(y, x, r), sigma=w,
                                absolute_sigma=False, maxfev=30000)
        except Exception:
            continue
        resid = y - func(X, *popt)
        ssr = float(np.sum(resid ** 2))
        if w is not None:
            chi2 = float(np.sum((resid / w) ** 2))
            aic = chi2 + 2 * k                       # Gaussian-likelihood AIC (weighted)
            red_chi2 = chi2 / max(y.size - k, 1)
        else:
            aic = y.size * np.log(ssr / y.size + 1e-30) + 2 * k
            red_chi2 = np.nan
        r2 = 1.0 - ssr / sst if sst > 0 else (1.0 if ssr < 1e-9 else 0.0)
        cand = dict(note=note, name=name, params=[float(p) for p in popt], aic=float(aic),
                    r2=float(r2), red_chi2=float(red_chi2), func=func, k=k)
        if best is None or aic < best["aic"]:
            best = cand
    return best


class TemplateParam:
    """Closed-form (with grid-interpolation fallback) model of the shower langaus
    parameters as functions of (Q_thr, reset rate), fitted from a fit_template_grid
    2-D grid. For each parameter the best closed form is kept; `value()` uses it when its
    R^2 clears `r2_min`, else bilinearly interpolates the grid. `template_at()` returns a
    synthesized template dict for the composite fit at any (on- or off-grid) point."""
    PARAMS = ("mpv_L", "eta_L", "A_L", "mu_G", "sig_G")

    def __init__(self, grid, ts, r2_min=0.9, chi2_cut=3.0):
        self.ts, self.r2_min = ts, r2_min
        self.thr = np.asarray(grid["thresholds"], float)
        rates = np.asarray(grid["rates"], float)
        order = np.argsort(rates)                     # ascending rate (off=0 first)
        self.rates = rates[order]
        self.grid = {k: np.asarray(grid[k], float)[order] for k in self.PARAMS}
        self.err = {k: np.asarray(grid.get(k + "_err", np.full_like(self.grid[k], np.nan)),
                                  float)[order] for k in self.PARAMS}
        self.x = self.thr / 1e3
        xx, rr = np.meshgrid(self.x, self.rates)      # (n_rate, n_thr), aligns with grid[k]
        # Drop nodes whose template fit is poor *relative to the rest of the grid* before
        # fitting the closed form -- a single bad node otherwise drags a clean law's R^2 down.
        # The cut must be RELATIVE: chi2/ndf grows with shower statistics for a fixed model
        # mismatch, so an absolute cut tuned at N=200 rejects every node at N=1000 (which
        # silently killed the whole parametrization -- forms={} -> bilinear fallback). Compare
        # each node to the grid's own median instead, keeping `chi2_cut` only as a floor so a
        # uniformly excellent grid can still reject a genuine outlier.
        chi2 = np.asarray(grid.get("chi2ndf", np.zeros_like(self.grid["mpv_L"])), float)[order]
        fin = np.isfinite(chi2)
        lim = max(float(chi2_cut), 2.0 * float(np.median(chi2[fin]))) if fin.any() else np.inf
        node_ok = fin & (chi2 <= lim)
        if node_ok.sum() < 3 and fin.sum() >= 3:       # never leave the fit with nothing
            node_ok = fin
        lib = _cf_library()
        self.forms = {}
        for k in self.PARAMS:
            yv, ev = self.grid[k].ravel(), self.err[k].ravel()
            m = node_ok.ravel() & np.isfinite(yv)
            self.forms[k] = _fit_closed_form(xx.ravel()[m], rr.ravel()[m], yv[m],
                                             yerr=ev[m], forms=lib.get(k))

    def _interp(self, key, thr, rate):
        """Bilinear interp of grid[key] over (rate, threshold), clamped at the edges.

        A 'cross' grid leaves the interior nodes NaN, so the four bilinear corners are not
        guaranteed to exist. Weight only the finite corners and renormalise; if none is finite
        (an off-cross request with no closed form), fall back to the nearest finite node."""
        g = self.grid[key]
        xi = float(np.interp(thr / 1e3, self.x, np.arange(self.x.size)))
        yi = float(np.interp(rate, self.rates, np.arange(self.rates.size)))
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = min(x0 + 1, self.x.size - 1), min(y0 + 1, self.rates.size - 1)
        fx, fy = xi - x0, yi - y0
        corners = (((1 - fx) * (1 - fy), g[y0, x0]), (fx * (1 - fy), g[y0, x1]),
                   ((1 - fx) * fy, g[y1, x0]), (fx * fy, g[y1, x1]))
        num = sum(w * v for w, v in corners if np.isfinite(v) and w > 0)
        den = sum(w for w, v in corners if np.isfinite(v) and w > 0)
        if den > 0:
            return float(num / den)
        fin = np.isfinite(g)
        if not fin.any():
            return float("nan")
        iy, ix = np.nonzero(fin)                       # nearest finite node in (rate, thr)
        d = ((ix - xi) ** 2 + (iy - yi) ** 2)
        j = int(np.argmin(d))
        return float(g[iy[j], ix[j]])

    def value(self, key, thr, cycles):
        rate = float(reset_rate_khz(cycles, self.ts))
        f = self.forms.get(key)
        if f is not None and f["r2"] >= self.r2_min:
            return float(f["func"](np.array([[thr / 1e3], [rate]]), *f["params"])[0])
        return self._interp(key, thr, rate)

    def template_at(self, thr, cycles):
        """Synthesized template dict (frozen langaus + Gaussian seed) at (thr, cycles)."""
        d = {k: self.value(k, float(thr), int(cycles)) for k in self.PARAMS}
        d.update(ok=True, threshold_e=float(thr), A_G=max(d["A_L"] * 0.3, 1.0), synthesized=True)
        return d


# ===========================================================================
# Plots  -- publication style (Phys-Rev-like: serif/STIX, inward ticks, no grid)
# ===========================================================================
_C_IND = "#3B6FB6"   # induction (blue)
_C_SHW = "#C24A4A"   # shower (red)
_C_SUM = "#1A1A1A"   # sum / total (near-black)
_C_GRN = "#4E9A6B"   # induction fraction
_C_PUR = "#7E6CAD"   # single-hit rate


def _journal_style():
    """Matplotlib rcParams for a clean physics-journal look (call once in main)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9.5,
        "axes.linewidth": 0.9, "lines.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 5.5, "ytick.major.size": 5.5,
        "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7", "legend.fancybox": False, "axes.grid": False,
    })


def _savefig(fig, path):
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  wrote", path)


# theta is measured to the PIXEL PLANE, so the two extremes are physically distinct
# topologies and deserve names rather than angles: 0 deg lies IN the plane (isochronous --
# the whole track's charge arrives together), 90 deg runs along the drift (the "pillar" --
# all charge onto one pixel column, arriving spread over the full drift time).
_MUON_GEOM = {0: ("in-plane", "inplane"), 90: ("along drift", "alongdrift")}


def sample_labels(name, sm):
    """(title, population, slug) for one control sample.

    The dict key (`muon_th45`) is fine for lookups but tells a reader nothing on a plot, and
    the generic plot furniture says "shower" everywhere -- wrong on a muon panel, where the
    collected population is the muon's own ionisation. `population` replaces that word and
    `slug` names the output files.
    """
    if str(sm.get("kind", "")) != "muon":
        return "EM shower", "shower", "shower"
    th = float(sm.get("theta", float("nan")))
    if not np.isfinite(th):
        return "Muon", "muon", str(name)
    key = int(round(th))
    if key in _MUON_GEOM:
        word, slug = _MUON_GEOM[key]
        return f"Muon, {word} ({key}\u00b0 to pixel plane)", "muon", f"muon_{slug}"
    return f"Muon, inclined {key}\u00b0 to pixel plane", "muon", f"muon_incl{key:02d}"


def _draw_charge_dist(ax, q, truth, fit, threshold_e, xlim=None,
                      compact=False, thr_sigma=0.0, logy=True, population="shower"):
    """Draw one FEE-readout single-hit Q spectrum (the noisy Q you trigger on in
    data), bins coloured by the per-pixel backtrack truth into induction vs shower
    (collection) hits (see categorize()), the composite template fit overlaid (frozen
    shower langaus + floating Gaussian as one shower curve, plus the induction langau),
    and the pixel-charge-threshold line with a +/-sigma band: the discriminator fires on
    `q+noise >= threshold + disc_noise`, so the threshold is Gaussian-smeared by
    sigma_disc -- which is why recorded charges land below the nominal line."""
    edges = fit["edges"] / 1e3
    nsh = max(int(fit.get("n_shower", 1)), 1)          # counts -> hits per shower
    is_ind, is_shw = categorize(q, truth)
    ax.hist([q[is_ind] / 1e3, q[is_shw] / 1e3], bins=edges,
            stacked=True, color=[_C_IND, _C_SHW], alpha=0.6, edgecolor="white",
            linewidth=(0.2 if compact else 0.35),
            weights=[np.full(int(is_ind.sum()), 1.0 / nsh), np.full(int(is_shw.sum()), 1.0 / nsh)],
            label=["Induction", population.capitalize()])
    if fit.get("ok") and fit.get("kind") == "template":
        xs = np.linspace(fit["edges"][0], fit["edges"][-1], 700)
        lg = fit["lg"]
        ne = fit.get("noise_e", 500.0)
        mpv_L, eta_L, A_L, mu_G, sig_G = fit["frozen"]
        aG, mpv_i, eta_i, aI = fit["popt"]
        shower = lg.comp(xs, mpv_L, eta_L, ne, A_L) + _gaussian(xs, mu_G, sig_G, aG)
        induction = lg.comp(xs, mpv_i, eta_i, ne, aI)
        lw = 1.2 if compact else 1.6
        ax.plot(xs / 1e3, induction, color=_C_IND, ls="--", lw=lw)
        ax.plot(xs / 1e3, shower, color=_C_SHW, ls="--", lw=lw)
        ax.plot(xs / 1e3, shower + induction, color=_C_SUM, ls="-", lw=lw + 0.5,
                label="Template fit")
    if thr_sigma and thr_sigma > 0:
        ax.axvspan((threshold_e - thr_sigma) / 1e3, (threshold_e + thr_sigma) / 1e3,
                   color="0.45", alpha=0.18, lw=0,
                   label=r"$Q_{\mathrm{thr}}\pm\sigma_{\mathrm{disc}}$")
        ax.axvline(threshold_e / 1e3, color="0.30", ls="--", lw=1.2)
    else:
        ax.axvline(threshold_e / 1e3, color="0.30", ls="--", lw=1.2, label=r"$Q_{\mathrm{thr}}$")
    # Scale the y-axis from the DATA, not from the fit overlay. The template curve is a
    # Landau evaluated across the whole panel and diverges towards x -> 0, so letting
    # matplotlib autoscale on it pushes the top to ~1e6 and squashes every histogram into the
    # bottom decade (visible in every scan-distribution plot before this fix).
    _tot, _ = np.histogram(q, bins=fit["edges"])
    _dtop = float(_tot.max()) / nsh if _tot.size and _tot.max() > 0 else 1.0
    if logy:
        ax.set_yscale("log")
        top = max(_dtop * 6.0, 2.0)
        cur = ax.get_ylim()[1]                 # panels share y: keep a bigger DATA-driven
        if np.isfinite(cur) and top < cur <= 100.0 * top:   # top from a previous panel, but
            top = cur                                       # never the fit-curve runaway
        ax.set_ylim(0.6, top)
    else:
        ax.set_yscale("linear")
        ax.set_ylim(0.0, max(_dtop * 1.35, 1.0))
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif fit.get("edges") is not None:
        ax.set_xlim(fit["edges"][0] / 1e3, fit["edges"][-1] / 1e3)


def plot_headline(ctx, q, truth, fit, outdir, xlim=None, thr_sigma=0.0,
                  suffix="", population="shower", title=None):
    """Nominal single-hit Q spectrum (the noisy Q triggered on, induction/shower
    coloured) + composite template fit (frozen shower langaus + floating Gaussian +
    induction langau) + threshold-with-noise band. Saved both log-y (ti_hist_nominal.png,
    the tail) and linear-y (ti_hist_nominal_lin.png, the peak region)."""
    import matplotlib.pyplot as plt
    for logy, suf in ((True, ""), (False, "_lin")):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        _draw_charge_dist(ax, q, truth, fit, fit["threshold_e"], xlim=xlim,
                          population=population,
                          thr_sigma=thr_sigma, logy=logy)
        if fit.get("ok"):
            stats = "\n".join((
                r"$\mathrm{MPV}_{\mathrm{ind}}=%.1f\times10^{3}\,e^{-}$" % (fit["mpv_ind"] / 1e3),
                r"$\mathrm{MPV}_{\mathrm{shw}}^{\mathrm{L}}=%.1f\times10^{3}\,e^{-}$" % (fit["mpv_shw"] / 1e3),
                r"$\mu_{\mathrm{G}}=%.1f\times10^{3}\,e^{-}$" % (fit.get("mu_G", np.nan) / 1e3),
                r"$f_{\mathrm{ind}}=%.2f$" % fit["frac_ind"],
                r"$\chi^{2}/\mathrm{ndf}=%.2f$" % fit["chi2ndf"]))
            ax.text(0.975, 0.965, stats, transform=ax.transAxes, ha="right", va="top",
                    fontsize=10.5, linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.6", lw=0.8))
        ax.set_xlabel(r"Single-hit pixel charge $Q$ ($10^{3}\,e^{-}$)")
        ax.set_ylabel(f"Single-hit pixels / {population}")
        ax.legend(loc="upper center", fontsize=9.5, handlelength=1.9)
        fig.tight_layout()
        if title:
            ax.set_title(title, fontsize=12)
        _savefig(fig, f"{outdir}/ti_hist_nominal{suffix}{suf}.png")


def plot_scan_distributions(panels, title, fmt_val, col0_header, fname, outdir,
                            xlim=None, thr_sigma=0.0, population="shower"):
    """Grid of readout-Q spectra + fits, one panel per test case (scan value). Each
    panel carries its own fit summary (the two langauss MPVs and the backtrack
    induction/shower pixel yields) in its annotation box -- no separate table -- and
    the grid is sized to exactly the number of test cases (6 -> a clean 2x3), so no
    empty cell and no stray shared x-axis label leak in. `panels` is
    [(knob_disp, fit, q, truth), ...]; `fmt_val` formats each panel's tag.
    `col0_header` is accepted for call-site compatibility but no longer used."""
    import matplotlib.pyplot as plt
    items = [(d, f, q, tr) for (d, f, q, tr) in panels if f and f.get("ok")]
    if not items:
        return
    n = len(items)
    ncol = min(max(n, 1), 3)
    nrow = int(np.ceil(n / ncol))
    base = fname[:-4] if fname.endswith(".png") else fname
    for logy, suf in ((True, ""), (False, "_lin")):   # log (tail) + linear (peak) versions
        # Share y only on the LOG panels; let each LINEAR panel autoscale its own top so a
        # tall induction peak (high induction scale / low threshold) isn't clipped by a
        # neighbour's smaller range -- the linear view is for reading the peak SHAPE.
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                                 sharex=True, sharey=logy, squeeze=False)
        for ax, (d, f, q, tr) in zip(axes.flat, items):
            _draw_charge_dist(ax, q, tr, f, f["threshold_e"], xlim=xlim,
                              compact=True, thr_sigma=thr_sigma, logy=logy,
                              population=population)
            is_ind, is_shw = categorize(q, tr)
            pc = population[0]                      # subscript: i(nduction) vs s(hower)/m(uon)
            stat = "\n".join((
                fmt_val(d),
                r"$\mathrm{MPV}_{i,%s}=%%.1f,\,%%.1f$" % pc % (f["mpv_ind"] / 1e3, f["mpv_shw"] / 1e3),
                r"$N_{i,%s}=%%d,\,%%d$" % pc % (int(is_ind.sum()), int(is_shw.sum()))))
            ax.text(0.95, 0.93, stat, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.0, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
        for ax in axes.flat[n:]:                     # hide any unused cells (label included)
            ax.set_visible(False)
        for c in range(ncol):                        # x-label on the lowest USED cell per column
            used = [r * ncol + c for r in range(nrow) if r * ncol + c < n]
            if used:
                axb = axes.flat[used[-1]]
                axb.set_xlabel(r"$Q$ ($10^{3}\,e^{-}$)")
                axb.tick_params(labelbottom=True)
        for r in range(nrow):
            if r * ncol < n:
                axes[r, 0].set_ylabel(f"Single-hit pixels / {population}")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.06, fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/{base}{suf}.png")


def _paired_hist(q1, q2, threshold_e, qmax=None, nbins=45):
    """Bin the single-hit and two-hit populations on COMMON edges.

    Shared edges are the whole point: only then can the pair be read as a migration
    (counts leaving one histogram reappearing in the other at a similar charge) rather
    than as two unrelated distributions."""
    a1 = np.asarray(q1, float)
    a2 = np.asarray(q2, float)
    h = _hist(np.concatenate([a1, a2]), threshold_e, nbins=nbins, qmax=qmax)
    if h is None:
        return None
    edges = h[3]
    c1, _ = np.histogram(a1[np.isfinite(a1) & (a1 > 0)], bins=edges)
    c2, _ = np.histogram(a2[np.isfinite(a2) & (a2 > 0)], bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), c1.astype(float), c2.astype(float), edges


def plot_multiplicity_dist(panels, title, fmt_val, fname, outdir, n_shower=1,
                           xlim=None, qmax=None, population="shower"):
    """Paired single-hit / two-hit spectra, one panel per scan value.

    Every fit in this study runs on pixels with EXACTLY one ADC sample. That cut is not
    neutral: raising the threshold delays the crossing until a second sample fits, and
    speeding up the periodic reset chops one integration into two -- both PROMOTE pixels out
    of the single-hit sample and into the two-hit one. So part of each knob's apparent
    response is a change in WHICH pixels are selected, not in the spectrum of a fixed set.
    Drawing both populations on common bins separates those two effects by eye.

    `panels` is [(knob_disp, q1, q2, threshold_e), ...]; `fmt_val` formats each panel tag."""
    import matplotlib.pyplot as plt
    if not any(np.size(q2) for (_d, _q1, q2, _t) in panels):
        return                       # spectra file predates the two-hit populations
    items = []
    for (d, q1, q2, thr) in panels:
        h = _paired_hist(q1, q2, thr, qmax=qmax)
        if h is not None:
            items.append((d, h, thr, np.size(q1), np.size(q2)))
    if not items:
        return
    n = len(items)
    ncol = min(max(n, 1), 3)
    nrow = int(np.ceil(n / ncol))
    ns = max(float(n_shower), 1.0)
    base = fname[:-4] if fname.endswith(".png") else fname
    for logy, suf in ((True, ""), (False, "_lin")):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                                 sharex=True, sharey=logy, squeeze=False)
        for ax, (d, (cen, c1, c2, edges), thr, n1, n2) in zip(axes.flat, items):
            x = cen / 1e3
            ax.fill_between(x, c1 / ns, step="mid", color=_C_PUR, alpha=0.35, lw=0,
                            label="1 hit (fitted sample)")
            ax.step(x, c1 / ns, where="mid", color=_C_PUR, lw=1.3)
            ax.step(x, c2 / ns, where="mid", color=_C_GRN, lw=1.6,
                    label="2 hits (total recorded $Q$)")
            ax.axvline(thr / 1e3, color="0.45", ls=":", lw=1.0)
            if logy:
                ax.set_yscale("log")
            if xlim:
                ax.set_xlim(*xlim)
            frac = n2 / max(n1 + n2, 1)
            stat = "\n".join((fmt_val(d),
                               r"$N_{1,2}=%d,\,%d$" % (n1, n2),
                               r"2-hit frac $=%.3f$" % frac))
            ax.text(0.95, 0.93, stat, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.0, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for c in range(ncol):
            used = [r * ncol + c for r in range(nrow) if r * ncol + c < n]
            if used:
                axb = axes.flat[used[-1]]
                axb.set_xlabel(r"$Q$ ($10^{3}\,e^{-}$)")
                axb.tick_params(labelbottom=True)
        for r in range(nrow):
            if r * ncol < n:
                axes[r, 0].set_ylabel(f"Pixels / {population}")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.06, fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/{base}{suf}.png")


def plot_charge_overlay(series, title, fname, outdir, n_shower=1, qmax=None,
                        xlabel=None, note=None, legend_title=None):
    """Overlay one charge distribution per threshold, with NO truth split.

    `series` is [(label, q, threshold_e), ...]. Every curve is binned on COMMON edges (built
    from the pooled sample at the lowest threshold) so they can be read against each other
    bin for bin, rather than each being binned to its own range. Colour runs with the
    threshold and each threshold is marked by a dotted line in its own colour, so the
    turn-on of every curve sits next to where its discriminator actually was.
    """
    import matplotlib.pyplot as plt
    series = [(l, np.asarray(q, float), float(t)) for l, q, t in series if np.size(q)]
    if not series:
        return
    pooled = np.concatenate([q for _l, q, _t in series])
    h = _hist(pooled, min(t for _l, _q, t in series), qmax=qmax)
    if h is None:
        return
    edges = h[3]
    cen = 0.5 * (edges[:-1] + edges[1:])
    ns = max(float(n_shower), 1.0)
    cmap = plt.get_cmap("viridis")
    nser = len(series)
    _one_thr = len({round(t, 6) for _l, _q, t in series}) == 1
    base = fname[:-4] if fname.endswith(".png") else fname
    for logy, suf in ((True, ""), (False, "_lin")):
        fig, ax = plt.subplots(figsize=(7.6, 4.9))
        top = 0.0
        for i, (lab, q, t) in enumerate(series):
            col = cmap(0.06 + 0.86 * i / max(nser - 1, 1))
            cnt, _ = np.histogram(q, bins=edges)
            y = cnt / ns
            top = max(top, float(y.max()) if y.size else 0.0)
            ax.step(cen / 1e3, y, where="mid", color=col, lw=1.7, label=lab)
            if not _one_thr:
                ax.axvline(t / 1e3, color=col, ls=":", lw=1.0, alpha=0.75)
        if _one_thr:                       # reset/induction scans: threshold is fixed
            ax.axvline(series[0][2] / 1e3, color="0.45", ls=":", lw=1.2)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.5 / ns, max(top * 3.0, 1.0 / ns))
        else:
            ax.set_ylim(0.0, max(top * 1.15, 1e-6))
        ax.set_xlim(0.0, edges[-1] / 1e3)
        ax.set_xlabel(xlabel or r"$Q$  ($10^{3}\,e^{-}$)")
        ax.set_ylabel("pixels / shower")
        ax.set_title(title, fontsize=11.5)
        ax.grid(alpha=0.22, lw=0.6)
        ax.legend(title=legend_title or r"$Q_{\mathrm{thr}}$", fontsize=9,
                  title_fontsize=9, loc="upper right", ncol=2)
        if note:
            ax.text(0.015, 0.02, note, transform=ax.transAxes, fontsize=8.0, va="bottom",
                    ha="left", color="0.35")
        fig.tight_layout()
        _savefig(fig, f"{outdir}/{base}{suf}.png")


def plot_multiplicity_rates(cols, outdir, fname="ti_multiplicity_rates.png"):
    """Migration summary: where the single-hit sample gains and loses pixels.

    Row (a) is the yield of each population per shower; row (b) is the two-hit FRACTION,
    n2/(n1+n2). The fraction is the scale-free one -- it divides out how many pixels the
    event lit up at all, so it responds to a knob promoting pixels rather than to overall
    occupancy. It is the observable the single-hit cut was hiding, and the one place the
    periodic reset is expected to show a clean, monotonic response.

    `cols` is [(title, xlabel, x, n1_rate, n2_rate), ...], one entry per knob."""
    import matplotlib.pyplot as plt
    cols = [c for c in cols if np.isfinite(np.asarray(c[4], float)).any()]
    if not cols:
        return
    ncol = len(cols)
    fig, axes = plt.subplots(2, ncol, figsize=(4.0 * ncol, 5.6), squeeze=False)
    ekw = dict(ms=6.0, mfc="white", mew=1.5, capsize=3, elinewidth=1.1, ls="none")
    for j, (title, xlabel, x, r1, r2) in enumerate(cols):
        x = np.asarray(x, float)
        r1 = np.asarray(r1, float)
        r2 = np.asarray(r2, float)
        o = np.argsort(x)
        x, r1, r2 = x[o], r1[o], r2[o]
        a, b = axes[0, j], axes[1, j]
        a.errorbar(x, r1, fmt="o", color=_C_PUR, label="1 hit", **ekw)
        a.errorbar(x, r2, fmt="s", color=_C_GRN, label="2 hits", **ekw)
        a.set_title(title, fontsize=10.5)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = r2 / (r1 + r2)
        b.errorbar(x, frac, fmt="^", color=_C_SUM, **ekw)
        # Anchor the fraction axis at 0. Autoscaling a flat series (the induction scan,
        # which should NOT promote pixels) blows a 1e-4 wobble up to full panel height and
        # reads as a trend; anchored, flat looks flat.
        top = np.nanmax(frac) if np.isfinite(frac).any() else 0.0
        b.set_ylim(0.0, max(1.25 * float(top), 0.02))
        b.set_xlabel(xlabel)
        for ax in (a, b):
            ax.grid(alpha=0.25, lw=0.6)
        if j == 0:
            a.set_ylabel("Pixels / shower")
            b.set_ylabel(r"2-hit fraction  $n_2/(n_1{+}n_2)$")
    axes[0, 0].legend(fontsize=9, loc="best")
    for k, ax in enumerate(axes.flat[:2 * ncol]):
        if k % ncol == 0:
            ax.text(0.02, 0.94, "(%s)" % chr(97 + k // ncol), transform=ax.transAxes,
                    fontsize=11, va="top", ha="left")
    fig.suptitle("Hit-multiplicity migration: what the single-hit cut selects", fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/{fname}")


def _add_multiplicity(scan, items2, n_shower):
    """Attach the two-hit yield to a scan dict, keyed on the scan's own knob values.

    Adds `n2_rate` (two-hit pixels per shower) and `frac_2hit`. Missing entries become NaN
    so a scan without a paired population -- or an older spectra file -- still runs and
    simply shows blanks in the sensitivity matrix."""
    by_val = {float(v): np.size(q) for (v, q, _tr, _a) in (items2 or [])}
    ns = max(float(n_shower), 1.0)
    r2 = np.array([by_val.get(float(k), np.nan) / ns for k in scan["knob"]], float)
    scan["n2_rate"] = r2
    with np.errstate(invalid="ignore", divide="ignore"):
        scan["frac_2hit"] = r2 / (np.asarray(scan["rate"], float) + r2)
    return scan


def plot_scan(ctx, scan, knob_label, fname, outdir, knob_vals_disp=None, mpv_ylim=None,
              population="shower", title=None):
    """The three DATA-BLIND handles vs one knob, in three stacked panels:
    (a) low-Q peak MPV [tracks THRESHOLD], (b) peak height [INDUCTION & reset],
    (c) tail height [periodic RESET & induction]. Points with uncertainties, no
    connecting lines. `mpv_ylim` (10^3 e-) sets panel (a) to the charge range."""
    import matplotlib.pyplot as plt
    x = np.asarray(knob_vals_disp if knob_vals_disp is not None else scan["knob"], float)
    order = np.argsort(x)
    x = x[order]
    col = lambda k: np.asarray(scan[k], float)[order]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.8), sharex=True)
    fig.subplots_adjust(hspace=0.07)
    ekw = dict(ms=6.5, mfc="white", mew=1.5, capsize=3, elinewidth=1.1, ls="none")
    axes[0].errorbar(x, col("peak_mpv") / 1e3, fmt="o", color=_C_IND, **ekw)
    axes[0].set_ylabel(r"Peak MPV  ($10^{3}\,e^{-}$)")
    if mpv_ylim is not None:
        axes[0].set_ylim(*mpv_ylim)
    axes[1].errorbar(x, col("peak_height"), fmt="s", color=_C_SHW, **ekw)
    axes[1].set_ylabel(f"Peak height  (hits/{population})")
    axes[2].errorbar(x, col("tail_height"), fmt="^", color=_C_PUR, **ekw)
    axes[2].set_ylabel(f"Tail height  (hits/{population})")
    axes[2].set_xlabel(knob_label)
    for k, axp in enumerate(axes):
        axp.text(0.018, 0.93, "(%s)" % chr(97 + k), transform=axp.transAxes,
                 fontsize=11, va="top", ha="left")
    if title:
        axes[0].set_title(title, fontsize=11.5)
    fig.align_ylabels(axes)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/{fname}")


def plot_shape_collapse(scans, outdir, fname="ti_shape_collapse.png"):
    """The disentangling money plot: every scan's spectra redrawn in the SCALE-FREE variable
    x = Q / (measured peak position), area-normalised.

    Rescaling charge by the measured peak divides the threshold out of the spectrum. So the
    THRESHOLD panel should COLLAPSE -- its curves land on top of one another, showing that
    threshold only sets the scale and not the shape -- while the RESET and INDUCTION panels
    stay visibly separated, because those change the shape itself (reset moves weight out of
    the near-threshold peak into a mid-tail shoulder; induction piles hits at x~1). That
    separation is what lets the three effects be told apart from the spectrum alone.

    `scans` is [(title, [(label, q, threshold_e), ...]), ...]."""
    import matplotlib.pyplot as plt
    scans = [(t, items) for t, items in scans if items]
    if not scans:
        return
    n = len(scans)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.3), squeeze=False)
    cmap = plt.get_cmap("viridis")
    # x-bin width must hold >= 1 ADC level for EVERY curve, else the LSB quantisation aliases
    # differently per threshold (one LSB is 0.33 in x at Q_thr=4ke- but 0.16 at 6ke-) and the
    # curves look ragged for a purely instrumental reason.
    prepped = []
    xw = 0.0
    for title, items in scans:
        cur = []
        for lab, q, thr in items:
            q = np.asarray(q, float); q = q[np.isfinite(q) & (q > 0)]
            h = _hist(q, float(thr)) if q.size >= 20 else None
            if h is None:
                continue
            c, raw, _w, _e = h
            pk = float(c[int(np.argmax(raw))])            # fit-free peak position
            lsb = _adc_lsb(q)
            if pk > 0 and lsb > 0:
                xw = max(xw, lsb / pk)
            cur.append((lab, q / max(pk, 1e-9)))
        prepped.append((title, cur))
    xw = max(xw, 0.15)
    xb = np.arange(0.0, 6.0 + xw, xw)
    for ax, (title, items) in zip(axes.flat, prepped):
        for j, (lab, xq) in enumerate(items):
            col = cmap(0.12 + 0.76 * (j / max(len(items) - 1, 1)))
            ax.hist(xq, bins=xb, histtype="step", lw=1.9, color=col,
                    density=True, label=lab)
        ax.axvline(1.0, color="0.45", ls=":", lw=1.2)
        ax.set_xlabel(r"$x = Q\;/\;Q_{\mathrm{peak}}$   (charge in units of the measured peak)")
        ax.set_xlim(0, 6)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5)
    axes.flat[0].set_ylabel("normalised single-hit pixels")
    fig.tight_layout()
    _savefig(fig, f"{outdir}/{fname}")


def plot_sample_knob_response(entries, knob_label, fname, outdir, xlog=False,
                              base_x=None, title=None, xfn=None):
    """Per-topology knob response: does lowering this knob buy INDUCTION hits?

    One column per control sample (shower + each muon angle), two rows:
      (a) single-hit pixels per event -- induction ON, induction OFF, and the
          pure-induction subset of ON (pixels where no charge landed, from the spatial
          backtrack). The ON/OFF gap is the hit count that exists only because of induced
          current on neighbouring pads.
      (b) the induction fraction of the ON sample, N(no charge landed)/N.

    The question this exists to answer: fsd_cube records ZERO induction-triggered single
    hits at its nominal 5 ke- threshold, because DISCRIMINATOR_NOISE is ~13x quieter than
    module0's and the marginal crossings module0 gets for free never happen. Whether a
    lower threshold recovers them is not something the shower scans can say -- showers and
    muons expose completely different neighbour geometries -- so it has to be scanned on
    the topology that isolates induction.

    `entries` is [(label, xs, n_on, n_off, n_ind, n_ev), ...] with counts, not rates, and
    `xs` in RAW knob units; `xfn` maps them to the plotted axis (e- -> ke-, reset cycles ->
    kHz) so the console table can keep reporting the knob as it was actually set."""
    import matplotlib.pyplot as plt
    entries = [e for e in entries if np.size(e[1]) and np.isfinite(np.asarray(e[1], float)).any()]
    if not entries:
        return
    ncol = len(entries)
    fig, axes = plt.subplots(2, ncol, figsize=(3.9 * ncol, 5.8), squeeze=False, sharex="col")
    ekw = dict(ms=6.0, mfc="white", mew=1.5, capsize=3, elinewidth=1.1, ls="none")
    for j, (lab, xs, n_on, n_off, n_ind, n_ev) in enumerate(entries):
        x = np.asarray(xs, float)
        o = np.argsort(x)
        x = x[o]
        if xfn is not None:
            x = np.asarray([xfn(v) for v in x], float)
        ne = max(float(n_ev), 1.0)
        on = np.asarray(n_on, float)[o] / ne
        offv = np.asarray(n_off, float)[o] / ne
        ind = np.asarray(n_ind, float)[o] / ne
        a, b = axes[0, j], axes[1, j]
        a.errorbar(x, on, fmt="o", color=_C_SHW, label="induction ON", **ekw)
        a.errorbar(x, offv, fmt="s", color=_C_SUM, label="induction OFF", **ekw)
        a.errorbar(x, ind, fmt="^", color=_C_IND, label="pure induction (ON)", **ekw)
        a.set_yscale("log")
        a.set_title(lab, fontsize=10.5)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(on > 0, ind / on, np.nan)
        b.errorbar(x, frac, fmt="^", color=_C_IND, **ekw)
        top = np.nanmax(frac) if np.isfinite(frac).any() else 0.0
        b.set_ylim(0.0, max(1.25 * float(top), 0.02))
        b.set_xlabel(knob_label)
        for ax in (a, b):
            if xlog:
                ax.set_xscale("log")
            if base_x is not None:
                ax.axvline(base_x, color="0.55", ls=":", lw=1.1)
            ax.grid(alpha=0.25, lw=0.6)
        if j == 0:
            a.set_ylabel("Single-hit pixels / event")
            b.set_ylabel("Induction fraction of single hits")
    axes[0, 0].legend(fontsize=8.5, loc="best")
    for r in range(2):
        axes[r, 0].text(0.02, 0.94, "(%s)" % chr(97 + r), transform=axes[r, 0].transAxes,
                        fontsize=11, va="top", ha="left")
    fig.suptitle(title or "Induction yield vs %s, by topology" % knob_label, fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/{fname}")


def _sample_scan_counts(scan):
    """(vals, n_on, n_off, n_ind) from an aggregate_sample_scan result.

    n_ind counts ON-sample pixels flagged pure-induction by the spatial backtrack (no charge
    landed on the pad), which is the direct truth-level answer -- not an ON-minus-OFF
    subtraction, which mixes in pixels that merely changed multiplicity."""
    vals, n_on, n_off, n_ind = [], [], [], []
    for (v, q_on, tr_on, _q2, _tr2, q_off, _tr_off) in scan:
        tr_on = np.asarray(tr_on, bool)
        vals.append(float(v))
        n_on.append(int(np.size(q_on)))
        n_off.append(int(np.size(q_off)))
        n_ind.append(int((~tr_on).sum()) if tr_on.size else 0)
    return (np.asarray(vals, float), np.asarray(n_on, float),
            np.asarray(n_off, float), np.asarray(n_ind, float))


def plot_sensitivity(matrix, knobs, observables, outdir, suffix="", title=None):
    """The money plot: |fractional response| of each observable to each knob.

    For each knob scan and each observable we compute the end-to-end fractional
    change normalised by the knob's fractional change (a dimensionless
    sensitivity). A near-DIAGONAL pattern -- each knob lighting up a different
    observable -- is the quantitative statement that the three effects are
    separable from the single-hit Q spectrum.
    """
    import matplotlib.pyplot as plt
    # Width tracks the column count -- a fixed 7.0 in collides the tick labels once the
    # migration observable makes it five columns.
    fig, ax = plt.subplots(figsize=(max(7.0, 1.75 * len(observables) + 1.2), 3.4))
    M = np.abs(np.asarray(matrix, float))
    vmax = float(np.nanmax(M)) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="cividis", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(observables)), labels=observables, fontsize=10.5)
    ax.set_yticks(range(len(knobs)), labels=knobs, fontsize=10.5)
    # thin white cell borders for a clean heatmap; no tick marks
    ax.set_xticks(np.arange(-0.5, len(observables)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(knobs)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="both", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v < 0.55 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|\,\Delta\mathrm{obs}/\mathrm{obs}\,|\;/\;|\,\Delta\mathrm{knob}/\mathrm{knob}\,|$",
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title or "Sensitivity of each fit observable to each readout knob",
                 fontsize=11, pad=8)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_sensitivity_matrix{suffix}.png")


def plot_shower_templates(node_fits, thresholds, base_reset, outdir):
    """Diagnostic: induction-OFF shower spectra + langaus+Gaussian template fits across
    thresholds (at base reset). Verifies the hypothesised shape and that the near-
    threshold langaus tracks threshold. Red dashed = langaus, green dashed = Gaussian,
    black = sum. Saved log-y and linear-y."""
    import matplotlib.pyplot as plt
    items = [(thr, node_fits.get((float(thr), int(base_reset)))) for thr in thresholds]
    items = [(thr, f) for thr, f in items if f and f.get("ok")]
    if not items:
        return
    n = len(items); ncol = min(max(n, 1), 3); nrow = int(np.ceil(n / ncol))
    for logy, suf in ((True, ""), (False, "_lin")):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (thr, f) in zip(axes.flat, items):
            edges = f["edges"]
            ax.hist(f["centres"] / 1e3, bins=edges / 1e3, weights=f["counts"],
                    color=_C_SHW, alpha=0.5, edgecolor="white", linewidth=0.2,
                    label="Induction-off shower")
            xs = np.linspace(edges[0], edges[-1], 700)
            mpv, eta, aL, mu, sig, aG = f["popt"]
            ne = f.get("noise_e", 500.0)
            ax.plot(xs / 1e3, f["lg"].comp(xs, mpv, eta, ne, aL), color=_C_SHW, ls="--", lw=1.2)
            ax.plot(xs / 1e3, _gaussian(xs, mu, sig, aG), color=_C_GRN, ls="--", lw=1.2)
            ax.plot(xs / 1e3, f["model"](xs, *f["popt"]), color=_C_SUM, ls="-", lw=1.6,
                    label="langaus + Gaussian")
            ax.axvline(thr / 1e3, color="0.3", ls=":", lw=1.0)
            ax.text(0.95, 0.93, "\n".join((
                r"$Q_{\mathrm{thr}}=%.1f$" % (thr / 1e3),
                r"$\mathrm{MPV}_L=%.1f$" % (mpv / 1e3),
                r"$\chi^{2}/\mathrm{ndf}=%.1f$" % f["chi2ndf"])),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
                linespacing=1.35, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
            ax.set_yscale("log" if logy else "linear")
            ax.set_ylim(bottom=0.6 if logy else 0.0)
            ax.set_xlim(0, 50)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for c in range(ncol):
            used = [r * ncol + c for r in range(nrow) if r * ncol + c < n]
            if used:
                axes.flat[used[-1]].set_xlabel(r"$Q$ ($10^{3}\,e^{-}$)")
                axes.flat[used[-1]].tick_params(labelbottom=True)
        for r in range(nrow):
            if r * ncol < n:
                axes[r, 0].set_ylabel("Single-hit pixels / shower")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(r"Induction-off shower template (langaus + Gaussian) vs threshold",
                     y=1.06, fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/ti_shower_template{suf}.png")


def plot_template_params(grid, tparam, ts, outdir):
    """Diagnostic + deliverable: each frozen-Landau parameter vs threshold, one series
    per reset rate, with the best closed-form fit overlaid (lines) and its formula + R^2
    annotated. Answers 'is there a closed form for the Landau in (threshold, reset)?'
    In the annotated formulae, f is the reset rate (kHz) and Q_thr is in 10^3 e-.
    A_L (top-right) is the amplitude the induction extraction hinges on."""
    import matplotlib.pyplot as plt
    thr = np.asarray(grid["thresholds"], float)
    rates = reset_rate_khz(np.asarray(grid["resets"], int), ts)
    params = [("mpv_L", r"$\mathrm{MPV}_L$ ($10^{3}e^{-}$)", 1e3),
              ("A_L", r"$A_L$ (a.u.)", 1.0),
              ("eta_L", r"$\eta_L$ ($10^{3}e^{-}$)", 1e3),
              ("mu_G", r"$\mu_G$ ($10^{3}e^{-}$)", 1e3)]
    cmap = plt.get_cmap("viridis")
    order = np.argsort(rates)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, ylab, sc) in zip(axes.flat, params):
        g = np.asarray(grid[key], float)               # (n_reset, n_thr)
        ge = np.asarray(grid.get(key + "_err", np.full_like(g, np.nan)), float)
        f = tparam.forms.get(key)
        for ir in order:
            c = cmap(0.12 + 0.76 * (ir / max(len(rates) - 1, 1)))
            lab = "off" if rates[ir] == 0 else "%.0f kHz" % rates[ir]
            yerr = ge[ir] / sc
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            ax.errorbar(thr / 1e3, g[ir] / sc, yerr=yerr, fmt="o", color=c, ms=5,
                        elinewidth=1.0, capsize=2.5, mfc=c, mec=c, label=lab)
            if f is not None:
                xs = np.linspace(thr.min(), thr.max(), 60)
                yhat = f["func"](np.vstack([xs / 1e3, np.full_like(xs, rates[ir])]), *f["params"])
                ax.plot(xs / 1e3, yhat / sc, "-", color=c, lw=1.2, alpha=0.85)
        ax.set_xlabel(r"$Q_{\mathrm{thr}}$ ($10^{3}\,e^{-}$)")
        ax.set_ylabel(ylab)
        if f is not None:
            gof = (r"$R^{2}=%.3f$" % f["r2"] if not np.isfinite(f.get("red_chi2", np.nan))
                   else r"$R^{2}=%.3f,\ \chi^{2}_\nu=%.1f$" % (f["r2"], f["red_chi2"]))
            ax.text(0.03, 0.975, "%s\n%s\n%s" % (f.get("note", ""), f["name"], gof),
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.2, linespacing=1.4,
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.7", lw=0.6))
    axes.flat[0].legend(title="reset rate", fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("Frozen shower-Landau parameters vs threshold and reset "
                 "(points) with closed-form fits (lines)", y=1.0, fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_template_params.png")


def plot_offpeak_anatomy(q, tr, aux, threshold_e, outdir, qmax=None, n_shower=1,
                         suffix="", title=None, population="shower"):
    """(1) WHAT ARE THE TWO PEAKS of the induction-off shower template?

    Splits the induction-OFF single-hit spectrum by the truth quantities that the
    hypothesis is about: how much REAL charge landed on the pad (q_coll) and how
    concentrated in time it arrived (sig_t_coll). Hypothesis under test:
      * near-threshold peak = pads with just enough charge to fire ONCE (low q_coll),
      * high-Q bump        = large, temporally CONCENTRATED deposits (high q_coll, small sig_t).
    """
    import matplotlib.pyplot as plt
    q = np.asarray(q, float); qc = np.asarray(aux.get("q_coll", []), float)
    st = np.asarray(aux.get("sig_t_coll", []), float)
    if qc.size != q.size or q.size < 20:
        return
    T, nsh = float(threshold_e), max(int(n_shower), 1)
    h = _hist(q, T, qmax=qmax)
    if h is None:
        return
    centres, raw, width, edges = h
    bands = [(0.0, 1.0, "$q_{coll}<Q_{thr}$", "#4E79A7"),
             (1.0, 3.0, "$1-3\\,Q_{thr}$", "#F0A73A"),
             (3.0, np.inf, "$>3\\,Q_{thr}$", "#C24A4A")]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    # (a) spectrum stacked by how much real charge landed
    stacks, labels, colors = [], [], []
    for lo, hi, lab, c in bands:
        m = (qc >= lo * T) & (qc < hi * T)
        stacks.append(q[m] / 1e3); labels.append(lab); colors.append(c)
    ax[0].hist(stacks, bins=edges / 1e3, stacked=True, color=colors, label=labels,
               edgecolor="white", linewidth=0.2,
               weights=[np.full(x.size, 1.0 / nsh) for x in stacks])
    ax[0].axvline(T / 1e3, color="0.3", ls="--", lw=1.1)
    ax[0].set_xlabel(r"recorded $Q$ ($10^{3}e^{-}$)")
    ax[0].set_ylabel(f"single-hit pixels / {population}")
    ax[0].set_title("Induction-off spectrum, split by charge that landed", fontsize=11)
    ax[0].set_xlim(0, 40); ax[0].legend(fontsize=9, title="real charge on pad")
    # (b) recorded Q vs collected charge
    ok = np.isfinite(qc) & (qc > 0)
    ax[1].hist2d(q[ok] / 1e3, qc[ok] / 1e3, bins=[45, 45],
                 range=[[0, 40], [0, 40]], cmap="cividis", cmin=1)
    lim = np.array([0, 40])
    ax[1].plot(lim, lim, "-", color="w", lw=1.0, alpha=.7)
    ax[1].axvline(T / 1e3, color="w", ls="--", lw=1.0, alpha=.7)
    ax[1].set_xlabel(r"recorded $Q$ ($10^{3}e^{-}$)")
    ax[1].set_ylabel(r"real charge landed $q_{coll}$ ($10^{3}e^{-}$)")
    ax[1].set_title("Recorded vs actual collected charge", fontsize=11)
    # (c) time concentration of the deposit, for the two recorded-Q regions
    near = q < 2.0 * T
    for m, lab, c in ((near, "near-threshold peak", _C_IND),
                      (~near, "high-$Q$ bump", _C_SHW)):
        v = st[m & np.isfinite(st)]
        if v.size > 5:
            ax[2].hist(v, bins=40, histtype="step", lw=1.8, color=c, density=True, label=lab)
    ax[2].set_xlabel(r"arrival-time spread $\sigma_t$ of the landed charge ($\mu$s)")
    ax[2].set_ylabel("normalised")
    ax[2].set_title("Is the deposit concentrated in time?", fontsize=11)
    if ax[2].get_legend_handles_labels()[0]:      # empty for a pure-induction sample
        ax[2].legend(fontsize=9)
    if title:
        fig.suptitle("Two-peak anatomy: %s" % title, y=1.02, fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_offpeak_anatomy{suffix}.png")


def plot_hit_timing(q, tr, aux, threshold_e, outdir, n_shower=1, n_mad=3.0,
                    suffix="", title=None, population="shower"):
    """(2) CAUSAL shower/induction separation using timing.

    Two reference times are used, because a pure-induction pad collects nothing and so has
    no arrival time of its OWN:

      dt      = t_hit - t_coll  -- vs the charge that landed on THIS pad. Defined only for
                charged pads; a hit with dt < 0 fired BEFORE its own charge arrived, i.e. it
                was induction-TRIGGERED even though the spatial backtrack calls it 'shower'.
                This drives the causal relabelling.
      dt_near = t_hit - t_near  -- vs the charge arriving in the pad's NEIGHBOURHOOD. Defined
                for essentially every pad, so induction hits appear in the comparison at all;
                induction leads the arrival (negative), collection tracks it (~0).

    The collection-timed window is taken from the DATA (median +/- n_mad*MAD of the charged
    population), never hard-coded, since t_coll carries a constant response delay. For a
    near-pure-induction sample (too few charged pads to define a window) the distributions
    are still drawn and the relabelling is simply skipped.
    """
    import matplotlib.pyplot as plt
    q = np.asarray(q, float); spatial = np.asarray(tr, bool)
    dt = np.asarray(aux.get("dt", []), float); qc = np.asarray(aux.get("q_coll", []), float)
    dtn = np.asarray(aux.get("dt_near", []), float)
    if dt.size != q.size or q.size < 20:
        return None
    if dtn.size != q.size:
        dtn = dt                                   # older files without the neighbour clock
    nsh = max(int(n_shower), 1)
    fin = np.isfinite(dt)
    ref = dt[spatial & fin]                       # charge-carrying pads define the window
    have_ref = ref.size >= 10
    if have_ref:
        med = float(np.median(ref))
        # Width from the UPPER half only. The population we are trying to find (hits that
        # fired EARLY) contaminates the lower tail, so a two-sided MAD is inflated by the
        # very outliers it is meant to catch -- which widens the window until nothing is
        # flagged. Late hits have no such mechanism, so the upper half is a clean estimator.
        hi_half = ref[ref >= med]
        mad = float(np.median(np.abs(hi_half - med))) * 1.4826 if hi_half.size >= 5 \
            else float(np.median(np.abs(ref - med))) * 1.4826
        lo, hi = med - n_mad * max(mad, 1e-6), med + n_mad * max(mad, 1e-6)
        causal = spatial & fin & (dt >= lo) & (dt <= hi)
    else:                                          # near-pure induction: no window to define
        med = mad = np.nan
        lo, hi = np.nan, np.nan
        causal = spatial & fin
    flipped = int((spatial & ~causal).sum())

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    # (a) dt_near: hit time vs the NEIGHBOURHOOD arrival -- defined for BOTH populations, so
    #     the induction lead is directly visible.
    fn = np.isfinite(dtn)
    if fn.any():
        b = np.linspace(np.nanpercentile(dtn[fn], 0.5), np.nanpercentile(dtn[fn], 99.5), 70)
        if spatial.any():
            ax[0].hist(dtn[spatial & fn], bins=b, color=_C_SHW, alpha=.65,
                       label=f"charge landed (spatial '{population}')")
        if (~spatial).any():
            ax[0].hist(dtn[~spatial & fn], bins=b, color=_C_IND, alpha=.65,
                       label="no charge (spatial 'induction')")
        ax[0].axvline(0.0, color="0.25", ls="--", lw=1.2)
        for lbl, v, c in (("induction", dtn[~spatial & fn], _C_IND),
                          ("collection", dtn[spatial & fn], _C_SHW)):
            if v.size > 5:
                ax[0].axvline(np.median(v), color=c, ls=":", lw=1.6)
    ax[0].set_xlabel(r"$\Delta t_{near} = t_{hit}-t_{near}$ ($\mu$s)")
    ax[0].set_ylabel("single-hit pixels")
    ax[0].set_title("Hit time vs neighbourhood charge arrival", fontsize=11)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8.5)
    # (b) where the mislabelled hits sit in (Q, dt)
    m = spatial & fin
    ax[1].scatter(q[m & causal] / 1e3, dt[m & causal], s=4, c=_C_SHW, alpha=.4,
                  label="collection-timed")
    ax[1].scatter(q[m & ~causal] / 1e3, dt[m & ~causal], s=6, c=_C_IND, alpha=.6,
                  label="charge landed, but MIS-TIMED")
    if np.isfinite(lo):
        ax[1].axhline(lo, color="0.25", ls="--", lw=1.0)
        ax[1].axhline(hi, color="0.25", ls="--", lw=1.0)
    ax[1].set_xlabel(r"recorded $Q$ ($10^{3}e^{-}$)"); ax[1].set_ylabel(r"$\Delta t$ ($\mu$s)")
    ax[1].set_xlim(0, 30)
    ax[1].set_title("Induction-triggered hits on charged pads", fontsize=11)
    ax[1].legend(fontsize=8.5, markerscale=2)
    # (c) spectrum relabelled: spatial vs causal
    h = _hist(q, float(threshold_e))
    if h is not None:
        edges = h[3]
        w = lambda mask: np.full(int(mask.sum()), 1.0 / nsh)
        ax[2].hist([q[~spatial] / 1e3, q[spatial] / 1e3], bins=edges / 1e3, stacked=True,
                   color=[_C_IND, _C_SHW], alpha=.35, edgecolor="white", linewidth=.2,
                   weights=[w(~spatial), w(spatial)],
                   label=["induction (spatial)", f"{population} (spatial)"])
        ax[2].step(edges[:-1] / 1e3, np.histogram(q[~causal], bins=edges)[0] / nsh,
                   where="post", color=_C_IND, lw=2.0, label="induction (causal, timed)")
        ax[2].axvline(float(threshold_e) / 1e3, color="0.3", ls="--", lw=1.1)
        ax[2].set_xlabel(r"recorded $Q$ ($10^{3}e^{-}$)")
        ax[2].set_ylabel(f"single-hit pixels / {population}")
        ax[2].set_xlim(0, 30); ax[2].legend(fontsize=8.5)
        ax[2].set_title("Induction grows when timing is required", fontsize=11)
    head = ("%s -- " % title) if title else ""
    if have_ref:
        tail = (r"causal shower/induction: window $\Delta t\in[%.2f,%.2f]\,\mu$s, "
                r"$%d/%d$ 'shower' hits induction-triggered" % (lo, hi, flipped, int(spatial.sum())))
    else:
        tail = (r"near-pure induction sample (%d charged pads): timing shown, "
                r"relabelling skipped" % int(spatial.sum()))
    fig.suptitle(head + tail, y=1.03, fontsize=11)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_hit_timing{suffix}.png")
    return dict(lo=lo, hi=hi, med=med, mad=mad, causal=causal, n_flipped=flipped)


# ===========================================================================
# Scans
# ===========================================================================
# fit-dict keys copied into every scan's output arrays (composite template fit)
_SCAN_FIT_KEYS = ("mpv_ind", "mpv_ind_err", "mpv_shw", "mpv_shw_err", "frac_ind",
                  "chi2ndf", "n_single", "peak_mpv", "peak_height", "tail_height",
                  "area_ind", "area_ind_err", "mu_G", "sig_G", "A_G", "A_G_err",
                  # scale-free / fit-free shape handles (see peak_tail_observables)
                  "tail_peak", "tail_over_core", "shoulder_frac")


def _record_fit(out, fit, n_events):
    """Append one fit's observables to the scan-output lists (nan-fill on failure)."""
    if fit and fit.get("ok"):
        for k in _SCAN_FIT_KEYS:
            out[k].append(fit[k])
        out["rate"].append(fit["n_single"] / max(n_events, 1))
    else:
        for k in _SCAN_FIT_KEYS:
            out[k].append(np.nan)
        out["rate"].append(np.nan)


def aggregate_fixed_scan(ctx, evs, knob_vals, set_fn, seed0):
    """GPU: threshold or reset scan spectra on cached nominal-induction pre-signals.
    `set_fn(value)` -> (threshold_e, reset_cycles).

    Returns (items1, items2, items_all): the single-hit scan, the parallel TWO-hit scan, and
    the ALL-multiplicity record [(val, q_sum, tr, n_hits), ...]. All come from the same FEE
    pass, so the extra two are free; items2 makes the migration visible and items_all is the
    only one that can represent 3+ hits (see `event_fee_hits`).
    """
    out, out2, outa = [], [], []
    for j, val in enumerate(knob_vals):
        thr, rst = set_fn(val)
        pops = aggregate_hits(ctx, evs, thr, rst, 1.0, seed0=seed0 + j * 1000)
        out.append((val,) + pops[1])
        out2.append((val,) + pops[2])
        outa.append((float(val),) + pops["all"])
    return out, out2, outa


def aggregate_sample_scan(ctx, evs_on, evs_off, knob_vals, set_fn, seed0):
    """FEE-only knob scan on ONE control sample, mirroring the shower scans.

    At every knob value this records the same three things the shower scans carry: the
    induction-ON single-hit sample (what gets fitted), its TWO-hit counterpart (the
    migration target), and the induction-OFF single-hit sample. The on/off pair is a cheap
    two-point induction axis -- the excess of ON over OFF is the hit count that exists only
    because of induced current on neighbouring pads -- while the spatial-truth flag splits
    ON into collection vs pure-induction pixels.

    Both pre-signal caches already exist, so this costs FEE passes only -- no re-induction,
    which is what makes scanning a muon sample affordable at all.

    `aux` is deliberately NOT recorded: the fits and the shape observables need only (q, tr),
    and the causal-timing diagnostics that do use aux run at the base point. Carrying seven
    aux arrays per knob value per angle would dominate the spectra file for no gain.

    Returns (out, out_all):
      out     = [(val, q_on, tr_on, q2_on, tr2_on, q_off, tr_off), ...]  -- as before;
      out_all = [(val, q_allon, tr_allon, n_allon, q_alloff, tr_alloff, n_alloff), ...]  -- the
                MULTIPLICITY-INCLUSIVE (every fired pixel) population, ON and OFF. The single-hit
                rate cancels by construction -- induction creates pure-induction single hits but
                also promotes collection pixels OUT of the single-hit sample -- so the inclusive
                rate is the observable that does NOT cancel. It is free here (same FEE passes).
    """
    out, out_all = [], []
    for j, val in enumerate(knob_vals):
        thr, rst = set_fn(val)
        pon = aggregate_hits(ctx, evs_on, thr, rst, 1.0, seed0=seed0 + j * 1000)
        pof = aggregate_hits(ctx, evs_off, thr, rst, 0.0, seed0=seed0 + j * 1000 + 500)
        out.append((float(val), pon[1][0], pon[1][1], pon[2][0], pon[2][1],
                    pof[1][0], pof[1][1]))
        out_all.append((float(val),) + pon["all"] + pof["all"])
    return out, out_all


def aggregate_sample_induction_scan(ctx, drift_cache, scales, base_threshold, base_reset,
                                    seed0, evs_off=None, collect_frac=0.15):
    """Induction-response scan on ONE control sample -- the muon analogue of
    `aggregate_induction_scan`.

    This is the ONLY per-sample scan that must RE-INDUCE (the pre-FEE current depends on the
    induction scale), so unlike the threshold/reset scans it is not free: cost is
    |scales| x n_events induce passes per angle, on events that light far more pixels than a
    shower does. That is why it is gated behind its own --muon-induction-events, default 0.

    Returns [(scale, q_on, tr_on, q2_on, tr2_on, q_off, tr_off), ...], the same shape as
    `aggregate_sample_scan`, with the induction-OFF cache reused as the scale-0 reference at
    every point so the on/off column stays meaningful.
    """
    set_periodic_reset(ctx, base_reset)
    ref = (aggregate_hits(ctx, evs_off, base_threshold, base_reset, 0.0, seed0=seed0 + 77)[1]
           if evs_off else (np.empty(0), np.empty(0, bool), None))
    out = []
    for j, s in enumerate(scales):
        if s == 0.0 and evs_off is not None:
            evs = evs_off
        else:
            set_induction(ctx, s)
            evs = [e for e in (event_induce(ctx, dc[0], dc[1], dc[2],
                                            seed=seed0 + j * 777 + i,
                                            collect_frac=collect_frac)
                               for i, dc in enumerate(drift_cache)) if e is not None]
        pon = aggregate_hits(ctx, evs, base_threshold, base_reset, s,
                             seed0=seed0 + j * 1000 + 50000)
        out.append((float(s), pon[1][0], pon[1][1], pon[2][0], pon[2][1], ref[0], ref[1]))
    set_induction(ctx, 1.0)
    return out


def aggregate_induction_scan(ctx, drift_cache, scales, base_threshold, base_reset,
                             seed0, evs_off=None, collect_frac=0.15):
    """GPU: induction-scan spectra -- re-INDUCE (only) the pre-FEE current at each response
    scale on the cached drift/pixel geometry (scale-independent). `evs_off` is reused
    verbatim for the scale=0 point.

    Returns (items1, items2, items_all), as in `aggregate_fixed_scan`."""
    set_periodic_reset(ctx, base_reset)
    out, out2, outa = [], [], []
    for j, s in enumerate(scales):
        if s == 0.0 and evs_off is not None:
            evs = evs_off                              # induction-off cache == scale-0
        else:
            set_induction(ctx, s)
            evs = [event_induce(ctx, dc[0], dc[1], dc[2], seed=seed0 + j * 777 + i,
                                collect_frac=collect_frac) for i, dc in enumerate(drift_cache)]
            evs = [e for e in evs if e is not None]
        pops = aggregate_hits(ctx, evs, base_threshold, base_reset, s,
                              seed0=seed0 + j * 1000 + 50000)
        out.append((float(s),) + pops[1])
        out2.append((float(s),) + pops[2])
        outa.append((float(s),) + pops["all"])
    set_induction(ctx, 1.0)
    return out, out2, outa


def fit_scan(items, n_events, n_shower, qmax, noise_e):
    """CPU: fit each scan point with the composite template model. `items` is
    [(val, q, tr, fit_threshold, template), ...]. Returns (scan_dict, fits) where fits is
    [(val, fit, q, tr)] and scan_dict has the per-point observable arrays (per shower)."""
    out = {k: [] for k in ("knob", "rate") + _SCAN_FIT_KEYS}
    fits = []
    for (val, q, tr, fit_thr, template) in items:
        fit = fit_shower_plus_induction(q, fit_thr, template, qmax=qmax,
                                        n_shower=n_shower, noise_e=noise_e)
        fits.append((val, fit, q, tr))
        out["knob"].append(val)
        _record_fit(out, fit, n_events)
    return {k: np.asarray(v, float) for k, v in out.items()}, fits


def sensitivity_row(scan, observables_keys, knob_axis=None, return_raw=False):
    """End-to-end |frac. obs. change| / |frac. knob change| for one scan.

    `knob_axis` overrides the knob values used for the normalisation -- pass the reset RATE
    (kHz) rather than the raw PERIODIC_RESET_CYCLES, so the reset row is normalised on the
    physical quantity being varied.

    Two things this gets right that the naive version did not:
      * dy and dk are taken at the SAME two points. Endpoints are chosen after sorting by the
        knob, so a scan listed out of order (the reset scan is [-1, 400, 200, ...]) no longer
        measures dy between one pair of points and dk between a different pair.
      * "off" entries (reset < 0, mapped to rate 0) are dropped from BOTH dy and dk. Keeping
        them makes the fractional knob change ~2 by construction, which silently divides the
        reset sensitivity by ~5 relative to the other knobs and hides a real response.
    Returns the sensitivity row, or (row, raw_relative_change) when `return_raw`."""
    knob = np.asarray(scan["knob"], float)
    kax = np.asarray(knob_axis, float) if knob_axis is not None else knob
    use = np.isfinite(kax) & (kax > 0) & np.isfinite(knob)     # drop "off"/non-physical
    if use.sum() < 2:                                          # fall back to whatever exists
        use = np.isfinite(kax)
    order = np.argsort(kax[use])
    idx = np.nonzero(use)[0][order]                            # scan indices, knob-sorted
    k0, k1 = kax[idx[0]], kax[idx[-1]]
    dk = abs(k1 - k0) / (abs(0.5 * (k1 + k0)) + 1e-9)
    row, raw = [], []
    for key in observables_keys:
        y = np.asarray(scan[key], float)[idx]
        good = np.isfinite(y)
        if good.sum() < 2:
            row.append(np.nan); raw.append(np.nan); continue
        y0, y1 = y[good][0], y[good][-1]                       # same endpoints as dk
        dy = abs(y1 - y0) / (abs(0.5 * (y1 + y0)) + 1e-9)
        row.append(dy / dk if dk > 0 else np.nan)
        raw.append(100.0 * (y1 - y0) / (abs(y0) + 1e-9))       # signed % change, un-normalised
    return (row, raw) if return_raw else row


# ===========================================================================
# Driver
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="module0",
                    help="NAMED larnd-sim config, resolved through larndsim/config/config.yaml "
                         "-- the same registry simulate_pixels.py uses, so the detector "
                         "properties / pixel layout / response / sim properties are always a "
                         "self-consistent set (module0, 2x2_no_modvar, fsd, fsd_cube, ndlar, "
                         "...). Use --list-configs to see them. Any of the four explicit flags "
                         "below overrides the corresponding registry entry.")
    ap.add_argument("--list-configs", action="store_true",
                    help="print the available named configs (with their files) and exit")
    # Explicit overrides. Default None -> taken from --config.
    ap.add_argument("--response", default=None)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--pixel-layout", default=None)
    ap.add_argument("--sim-properties", default=None)
    ap.add_argument("--edep-h5", default=None,
                    help="dumpTree.py edep-sim HDF5; use its real 'segments' per "
                         "event instead of the parametric shower generator")
    ap.add_argument("--recenter-showers", action="store_true",
                    help="rigidly translate each --edep-h5 event into THIS config's active "
                         "volume (random transverse position + drift depth). Required to "
                         "re-use a shower file across detector geometries, since edep "
                         "coordinates are tied to the GDML they were generated in. Raw dE/dEdx "
                         "is untouched, so recombination/drift/response are still applied per "
                         "config -- i.e. the same showers read out by a different detector.")
    ap.add_argument("--n-events", type=int, default=30, help="shower events per config")
    ap.add_argument("--shower-energy", type=float, default=300.0, help="MeV")
    ap.add_argument("--n-dep", type=int, default=400, help="deposits per shower")
    # --- muon control samples (run ALONGSIDE the showers in the same job) ---
    ap.add_argument("--muon-thetas", type=float, nargs="*", default=[0.0, 90.0],
                    help="also generate straight MIP MUON samples at these angles to the "
                         "PIXEL PLANE (deg): 0 = in-plane / isochronous (charge arrives "
                         "together), 90 = along the drift axis (charge spread over drift "
                         "time). Default runs both extremes; pass none to skip muons. Muons "
                         "get the truth diagnostics + waveforms, not the full template scan.")
    ap.add_argument("--muon-events", type=int, default=400,
                    help="events per muon sample (a MIP track lights up ~40 pixels, so a few "
                         "hundred is plenty). Capped at --n-events.")
    ap.add_argument("--muon-length", type=float, default=20.0, help="muon track length (cm)")
    ap.add_argument("--muon-scan-events", type=int, default=100,
                    help="events per muon sample used for the THRESHOLD/RESET scans "
                         "(0 disables them). Separate from --muon-events because the scans "
                         "cost 2x|knob values| extra FEE passes per angle: they reuse the "
                         "cached pre-signals, so no re-induction, but a muon lights far more "
                         "pixels per event than a shower does.")
    ap.add_argument("--muon-induction-events", type=int, default=0,
                    help="events per muon sample for a full INDUCTION scan (0 = off, the "
                         "default). Unlike the threshold/reset scans this must RE-INDUCE per "
                         "scale, so it is the expensive one -- turn it on only when you want "
                         "an induction ROW in each topology's sensitivity matrix. The cheap "
                         "two-point induction on/off contrast is recorded either way.")
    ap.add_argument("--muon-thresholds", type=float, nargs="*", default=None,
                    help="thresholds (e-) for the muon scan; defaults to --thresholds. Set "
                         "this LOWER than the shower scan to find where induction starts "
                         "producing single-hit pixels -- in fsd_cube nothing induction-"
                         "triggered survives the nominal 5 ke- cut.")
    ap.add_argument("--collect-frac", type=float, default=0.15,
                    help="a (segment,pad) counts as collection if its net charge > this "
                         "fraction of that segment's peak-collecting pad (backtrack truth "
                         "split; relative, so threshold-independent)")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[3000, 4000, 5000, 6000, 8000, 10000], help="e-")
    ap.add_argument("--resets", type=int, nargs="+",
                    default=[-1, 400, 200, 100, 66, 50],
                    help="PERIODIC_RESET_CYCLES (-1 = off)")
    ap.add_argument("--inductions", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], help="response scale")
    ap.add_argument("--template-grid", choices=["cross", "full"], default="cross",
                    help="which induction-off template nodes to compute. 'cross' (default) does "
                         "every threshold at the base reset + every reset at the base threshold "
                         "-- exactly the nodes the scans request, and the study's biggest cost "
                         "saving (9 vs 24 nodes for a 4x6 grid). 'full' does the whole product, "
                         "which only better-constrains the diagnostic closed-form fit.")
    ap.add_argument("--induction-events", type=int, default=60,
                    help="events for the EXPENSIVE induction scan (it re-runs the pre-FEE "
                         "induction stage per scale). Subsamples the first N of --n-events "
                         "to avoid wall-time timeouts; the cheap threshold/reset/template "
                         "scans still use all --n-events. 0 = use all.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--truth-cut", type=float, default=None,
                    help="(deprecated; ignored -- truth is now a collection backtrack)")
    ap.add_argument("--pretrigger-frac", type=float, default=None,
                    help="(deprecated; ignored -- the 3-way pre-trigger split was removed)")
    ap.add_argument("--noise-e", type=float, default=None,
                    help="fixed Gaussian smear (e-) for the langaus peaks; default = the "
                         "detector UNCORRELATED_NOISE_CHARGE (the recorded-Q electronics noise)")
    ap.add_argument("--refit", default=None,
                    help="skip the GPU sim: load a saved spectra .npz (from --spectra-out) "
                         "and re-run ONLY the fitting + plotting. Runs anywhere (no GPU), so "
                         "you can iterate on the fits locally after copying the file back.")
    ap.add_argument("--spectra-out", default=None,
                    help="write the raw single-hit spectra to this .npz after the sim "
                         "(default <outdir>/ti_spectra.npz) for later --refit.")
    ap.add_argument("--wf-events", type=int, default=6,
                    help="events per sample whose per-pixel waveforms (current + charge vs "
                         "time) are saved to ti_waveforms.npz for the standalone viewer. 0 = off.")
    ap.add_argument("--wf-max-pix", type=int, default=40,
                    help="max pixels saved per waveform event (sampled across hit categories)")
    # --- shower-energy scan (Part D): induction vs shower energy, parametric generator ---
    ap.add_argument("--shower-energies", type=float, nargs="*", default=None,
                    help="run a SHOWER-ENERGY scan at these energies (MeV), e.g. 100 300 1000 "
                         "3000. Each is its own small parametric sub-sample induced ON and OFF; "
                         "n_dep is scaled with energy so denser cores come from geometry, not a "
                         "fake dE/dx. Tests whether higher-energy showers induce more strongly. "
                         "Ignored with --edep-h5 (real showers carry their own energies).")
    ap.add_argument("--energy-scan-events", type=int, default=100,
                    help="showers per energy point for --shower-energies (capped at --n-events)")
    # --- burst-mode sub-threshold induction waveforms (Part A) ---
    ap.add_argument("--burst-thetas", type=float, nargs="*", default=[0.0],
                    help="muon angles (deg to pixel plane) to run the BURST-MODE study on. 0 = "
                         "through-going parallel to the plane, the clean case: collectors form a "
                         "line, their transverse neighbours see pure sub-threshold induction. "
                         "These angles must also be in --muon-thetas.")
    ap.add_argument("--burst-events", type=int, default=60,
                    help="muon events per burst-mode angle (re-induced per --burst-inductions "
                         "scale, so this is the one added cost; 0 = disable burst mode).")
    ap.add_argument("--burst-neighbors", type=int, default=3,
                    help="transverse-neighbour reach in pitches (+-N) whose forced waveforms are "
                         "captured -- the induction falloff profile.")
    ap.add_argument("--burst-cadence", type=float, default=0.0,
                    help="forced-sample period (us). 0 = the ADC integration window "
                         "((3+ADC_HOLD_DELAY)*CLOCK_CYCLE), the natural DAQ sample.")
    ap.add_argument("--burst-inductions", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                    help="induction scales for the burst-mode MODEL test: 1.0 is nominal, the "
                         "others are mis-models whose separation from nominal is the deliverable.")
    ap.add_argument("--burst-only", action="store_true",
                    help="FAST PATH: skip ALL shower work and every template/composite fit; do "
                         "only parallel-muon (--burst-thetas) burst-mode waveforms plus a cheap "
                         "FEE-only self-trigger threshold scan (induction on/off). Minutes, not "
                         "hours, and no big-RAM allocation. The burst waveforms are threshold-"
                         "independent, so the threshold range only scans the self-trigger yield.")
    ap.add_argument("--outdir", default="tistudy")
    return ap.parse_args()


def list_named_configs():
    """Print the named configs from larndsim/config/config.yaml and the files each resolves to,
    flagging the ones this study cannot load (see resolve_config)."""
    from larndsim.config import list_config_keys, get_config
    print("Named larnd-sim configs (larndsim/config/config.yaml):\n")
    for name in sorted(list_config_keys()):
        try:
            cfg = get_config(name)
        except Exception as exc:
            print(f"  {name:26s}  <unreadable: {exc}>"); continue
        multi = [k for k in ("PIXEL_LAYOUT", "RESPONSE")
                 if isinstance(cfg.get(k), list) and len(cfg[k]) > 1]
        tag = "  [MULTI-MODULE - not supported here]" if multi else ""
        print(f"  {name:26s}{tag}")
        for k in ("DET_PROPERTIES", "PIXEL_LAYOUT", "RESPONSE", "SIM_PROPERTIES"):
            v = cfg.get(k)
            v = v if not isinstance(v, list) else "[" + ", ".join(os.path.basename(str(x)) for x in v) + "]"
            print(f"      {k:16s} {os.path.basename(str(v)) if '/' in str(v) else v}")
    print("\n  MULTI-MODULE configs vary the pixel layout / response BETWEEN modules; this study\n"
          "  loads a single consistent set, so use the '_no_modvar' variant instead.")


def resolve_config(args):
    """Fill --detector/--pixel-layout/--response/--sim-properties from the named --config,
    leaving any explicitly-given flag untouched.

    Using the registry (rather than hand-pairing files) matters here: the response table is
    tied to the PIXEL PITCH, so e.g. pairing module0's response_44 (4.434 mm) with the FSD
    cube's 3.72 mm layout would silently mis-model the very induction physics being measured.
    The registry pairings are the ones simulate_pixels.py uses.

    Multi-module configs (2x2, ndlar) list SEVERAL layouts/responses -- one per module -- which
    tests/verify_diffusion.py:load_simulation cannot represent (it loads one of each). Those are
    rejected with a pointer to the single-configuration variant."""
    from larndsim.config import get_config, list_config_keys
    try:
        cfg = get_config(args.config)
    except KeyError:
        sys.exit(f"ERROR: unknown --config '{args.config}'.\n  available: "
                 f"{', '.join(sorted(list_config_keys()))}\n  (--list-configs for details)")

    def pick(key, flag):
        v = cfg.get(key)
        if isinstance(v, list):
            if len(v) != 1:                       # genuine module-to-module variation
                alt = [n for n in list_config_keys() if n.startswith(args.config)
                       and "no_modvar" in n]
                sys.exit(
                    f"ERROR: config '{args.config}' varies {key} between modules "
                    f"({len(v)} entries) and this study loads a single detector "
                    f"configuration.\n  Use "
                    f"{'--config ' + alt[0] if alt else 'a single-module config'} instead, "
                    f"or pass {flag} explicitly.")
            v = v[0]
        return v

    args.detector = args.detector or pick("DET_PROPERTIES", "--detector")
    args.pixel_layout = args.pixel_layout or pick("PIXEL_LAYOUT", "--pixel-layout")
    args.response = args.response or pick("RESPONSE", "--response")
    args.sim_properties = args.sim_properties or pick("SIM_PROPERTIES", "--sim-properties")
    for flag, val in (("--detector", args.detector), ("--pixel-layout", args.pixel_layout),
                      ("--response", args.response), ("--sim-properties", args.sim_properties)):
        if not val or not os.path.exists(val):
            sys.exit(f"ERROR: {flag} resolved to a missing file: {val}")
    print(f"Config '{args.config}':")
    for k, v in (("detector", args.detector), ("pixel_layout", args.pixel_layout),
                 ("response", args.response), ("sim_properties", args.sim_properties)):
        print(f"  {k:15s} {v}")
    return args


def load_edep_events(path, max_events=None):
    """Real edep-sim showers: per-event `tracks` arrays from a dumpTree HDF5.

    dumpTree.py writes a `segments` dataset whose dtype is the SAME as this study's
    SEGMENTS_DTYPE (both copy cli/dumpTree.py:segments_dtype), so each event's rows
    drop straight into the quench->drift->induction->FEE pipeline in place of
    build_shower_event(). Rows are grouped by `event_id`; shared fields are copied
    so the loader is robust to minor dtype differences between larnd-sim versions.
    """
    import h5py
    with h5py.File(path, "r") as f:
        seg = f["segments"][:]
    shared = [n for n in seg.dtype.names if n in vd.SEGMENTS_DTYPE.names]
    ev_ids = np.unique(seg["event_id"])
    if max_events:
        ev_ids = ev_ids[:max_events]
    events = []
    for eid in ev_ids:
        rows = seg[seg["event_id"] == eid]
        tr = vd.blank_tracks(len(rows))
        for name in shared:
            tr[name] = rows[name]
        events.append(tr)
    return events


_POS_FIELDS = (("x", "x_start", "x_end"), ("y", "y_start", "y_end"), ("z", "z_start", "z_end"))


def events_out_of_volume(events, ctx, plane=0, margin=3.0):
    """Fraction of deposits that fall outside `plane`'s active volume (0.0 = all inside).

    Used to catch the silent failure mode where an edep file generated in one detector's
    GDML is handed to a different --config: nothing errors, the events simply drift through
    the wrong volume and the sample comes out empty or clipped.
    """
    det = ctx.detector
    x0, x1, y0, y1 = vd.active_volume(det, plane, margin=margin)
    dmax = abs(det.DRIFT_LENGTH) - margin
    z_anode, _, into = vd.plane_z(det, plane)
    n_out = n_tot = 0
    for tr in events:
        x, y, z = (np.asarray(tr[c], float) for c in ("x", "y", "z"))
        depth = (z - z_anode) * into
        bad = (x < x0) | (x > x1) | (y < y0) | (y > y1) | (depth < 0.0) | (depth > dmax)
        n_out += int(bad.sum()); n_tot += bad.size
    return (n_out / n_tot) if n_tot else 0.0


def recenter_events(events, ctx, rng, plane=0, margin=3.0):
    """Rigidly translate each edep event so it lands inside THIS detector's active volume.

    `load_edep_events` copies segment coordinates verbatim, so a shower file is tied to the
    GDML frame it was generated in (module0 sits at tpc_offsets [0,-21.82,0] with a 4.434 mm
    pitch; fsd_cube at [0,0,0] with 3.72 mm). Re-using it under another --config otherwise
    drifts the showers through the wrong volume.

    A rigid translation is sufficient AND sufficient-only: edep segments carry RAW energy
    deposition (dE, dEdx) in LAr, and every detector-specific stage -- recombination at the
    configured E field, v_drift, lifetime attenuation, diffusion, pixel response -- is applied
    downstream by `quench_and_drift`. So the same shower sample read out by two geometries is a
    controlled comparison: the deposits are identical, only the readout differs.

    Shower structure and orientation relative to the drift axis are preserved; each event is
    dropped at a random transverse position and drift depth (as `build_shower_event` does).
    The target is drawn inset by the event's own half-extent so the shower fits when it can;
    an event larger than the volume is centred instead (and will be clipped by the drift).
    """
    det = ctx.detector
    x0, x1, y0, y1 = vd.active_volume(det, plane, margin=margin)
    dmax = abs(det.DRIFT_LENGTH) - margin
    z_anode, _, into = vd.plane_z(det, plane)
    lo = {"x": x0, "y": y0, "z": 0.5}
    hi = {"x": x1, "y": y1, "z": dmax}
    out = []
    for tr in events:
        tr = tr.copy()
        w = np.maximum(np.asarray(tr["dE"], float), 1e-12)
        cen, half = {}, {}
        for ax, (c, s, e) in zip("xyz", _POS_FIELDS):
            span = np.concatenate([tr[c], tr[s], tr[e]]).astype(float)
            cen[ax] = float(np.average(np.asarray(tr[c], float), weights=w))
            half[ax] = 0.5 * float(span.max() - span.min())
        delta = {}
        for ax in "xyz":
            a, b = lo[ax] + half[ax], hi[ax] - half[ax]
            t = float(rng.uniform(a, b)) if b > a else 0.5 * (lo[ax] + hi[ax])
            # z is chosen as a DRIFT DEPTH, then mapped back to a raw z coordinate.
            delta[ax] = (z_anode + into * t - cen["z"]) if ax == "z" else t - cen[ax]
        for ax, fields in zip("xyz", _POS_FIELDS):
            for f in fields:
                tr[f] = tr[f] + delta[ax]
        out.append(tr)
    return out


def base_config(ctx, args):
    """Nominal operating point = the loaded config's own threshold + reset values."""
    base_thr = float(np.atleast_1d(ctx.detector.DISCRIMINATION_THRESHOLD).ravel()[0])
    base_reset = int(getattr(ctx.detector, "PERIODIC_RESET_CYCLES", -1))
    return base_thr, base_reset


def _obj_array(list_of_arrays):
    """Pack a list of variable-length 1-D arrays into a 1-D object array (avoids numpy's
    'same length -> 2-D' auto-stacking, so save/load round-trips cleanly)."""
    a = np.empty(len(list_of_arrays), dtype=object)
    for i, x in enumerate(list_of_arrays):
        a[i] = np.asarray(x)
    return a


def save_spectra(spectra, path):
    """Save all raw single-hit spectra + metadata to an .npz so the (slow, GPU) sim runs
    ONCE and the fitting can be iterated offline with --refit. q/tr arrays have different
    lengths per config, so they are stored as object arrays (load with allow_pickle)."""
    kw = {}
    for k, v in spectra["meta"].items():
        kw["meta_" + k] = np.asarray(v)
    kw["nominal_q"] = np.asarray(spectra["nominal"][0], float)
    kw["nominal_tr"] = np.asarray(spectra["nominal"][1], bool)
    for k in _AUX_KEYS:
        kw["nominal_aux_" + k] = np.asarray(spectra["nominal"][2][k], float)
    for name in ("threshold", "reset", "induction"):
        items = spectra[name]
        kw[name + "_vals"] = np.asarray([it[0] for it in items], float)
        kw[name + "_q"] = _obj_array([it[1] for it in items])
        kw[name + "_tr"] = _obj_array([it[2] for it in items])
        for k in _AUX_KEYS:
            kw[f"{name}_aux_{k}"] = _obj_array([it[3][k] for it in items])
    # Parallel TWO-hit populations. Written under a distinct `_n2` prefix rather than
    # widening the existing tuples, so every older reader/unpacker keeps working.
    n2 = spectra.get("nominal_n2")
    if n2 is not None:
        kw["nominal_n2_q"] = np.asarray(n2[0], float)
        kw["nominal_n2_tr"] = np.asarray(n2[1], bool)
        for k in _AUX_KEYS:
            kw["nominal_n2_aux_" + k] = np.asarray(n2[2][k], float)
    for name in ("threshold", "reset", "induction"):
        items = spectra.get(name + "_n2")
        if not items:
            continue
        kw[f"{name}_n2_vals"] = np.asarray([it[0] for it in items], float)
        kw[f"{name}_n2_q"] = _obj_array([it[1] for it in items])
        kw[f"{name}_n2_tr"] = _obj_array([it[2] for it in items])
        for k in _AUX_KEYS:
            kw[f"{name}_n2_aux_{k}"] = _obj_array([it[3][k] for it in items])
    # all-multiplicity records: (q_sum, truth, n_hits) per scan point
    na = spectra.get("nominal_all")
    if na is not None:
        kw["nominal_all_q"] = np.asarray(na[0], np.float32)
        kw["nominal_all_tr"] = np.asarray(na[1], bool)
        kw["nominal_all_n"] = np.asarray(na[2], np.int16)
    for name in ("threshold", "reset", "induction"):
        items = spectra.get(name + "_all")
        if not items:
            continue
        kw[f"{name}_all_vals"] = np.asarray([r[0] for r in items], float)
        kw[f"{name}_all_q"] = _obj_array([np.asarray(r[1], np.float32) for r in items])
        kw[f"{name}_all_tr"] = _obj_array([r[2] for r in items])
        kw[f"{name}_all_n"] = _obj_array([np.asarray(r[3], np.int16) for r in items])
    g = spectra["grid"]
    kw["grid_thr"] = np.asarray([it[0] for it in g], float)
    kw["grid_rst"] = np.asarray([it[1] for it in g], int)
    kw["grid_q"] = _obj_array([it[2] for it in g])
    kw["grid_tr"] = _obj_array([it[3] for it in g])
    for k in _AUX_KEYS:
        kw["grid_aux_" + k] = _obj_array([it[4][k] for it in g])
    samples = spectra.get("samples", {})
    kw["samp_names"] = np.asarray(list(samples.keys()))
    for name, s in samples.items():
        kw[f"samp_{name}_kind"] = np.asarray(str(s.get("kind", "")))
        kw[f"samp_{name}_theta"] = np.asarray(float(s.get("theta", np.nan)))
        kw[f"samp_{name}_n"] = np.asarray(int(s.get("n", 0)))
        for which in ("nominal", "off"):
            v = s.get(which)
            if v is None:
                continue
            q, tr, aux = v
            kw[f"samp_{name}_{which}_q"] = np.asarray(q, float)
            kw[f"samp_{name}_{which}_tr"] = np.asarray(tr, bool)
            for k in _AUX_KEYS:
                kw[f"samp_{name}_{which}_aux_{k}"] = np.asarray(aux.get(k, []), float)
        # per-sample knob scans (induction on/off at every point); aux is deliberately not
        # stored -- the causal diagnostics only run at the base point.
        kw[f"samp_{name}_n_scan"] = np.asarray(int(s.get("n_scan", 0)))
        kw[f"samp_{name}_n_ind_scan"] = np.asarray(int(s.get("n_ind_scan", 0)))
        for fld in ("thr_grid", "scan_thresholds"):
            if s.get(fld) is not None:
                kw[f"samp_{name}_{fld}"] = np.asarray(s[fld], float)
        for knob in ("threshold", "reset", "induction"):
            sc = s.get("scan_" + knob)
            if not sc:
                continue
            pre = f"samp_{name}_scan_{knob}"
            kw[pre + "_vals"] = np.asarray([r[0] for r in sc], float)
            for fld, i in (("q_on", 1), ("tr_on", 2), ("q2_on", 3),
                           ("tr2_on", 4), ("q_off", 5), ("tr_off", 6)):
                kw[f"{pre}_{fld}"] = _obj_array([r[i] for r in sc])
            # multiplicity-inclusive (all-hit) ON/OFF companion -- the non-cancelling rate
            sca = s.get(f"scan_{knob}_allmult")
            if sca:
                for fld, i in (("aq_on", 1), ("atr_on", 2), ("an_on", 3),
                               ("aq_off", 4), ("atr_off", 5), ("an_off", 6)):
                    kw[f"{pre}_{fld}"] = _obj_array([r[i] for r in sca])
        gr = s.get("grid")
        if gr:
            pre = f"samp_{name}_grid"
            kw[pre + "_thr"] = np.asarray([it[0] for it in gr], float)
            kw[pre + "_rst"] = np.asarray([it[1] for it in gr], int)
            kw[pre + "_q"] = _obj_array([it[2] for it in gr])
            kw[pre + "_tr"] = _obj_array([it[3] for it in gr])
    # shower-energy scan (Part D): (E, n_ev, q1on,tr1on, qAon,trAon,nAon, q1off,tr1off,
    # qAoff,trAoff,nAoff) per energy point.
    escan = spectra.get("energy_scan") or []
    if escan:
        kw["escan_E"] = np.asarray([r[0] for r in escan], float)
        kw["escan_nev"] = np.asarray([r[1] for r in escan], int)
        for fld, i in (("q1on", 2), ("tr1on", 3), ("qAon", 4), ("trAon", 5), ("nAon", 6),
                       ("q1off", 7), ("tr1off", 8), ("qAoff", 9), ("trAoff", 10), ("nAoff", 11)):
            kw[f"escan_{fld}"] = _obj_array([r[i] for r in escan])
    np.savez(path, **kw)


def save_waveforms(recs, path):
    """Save per-pixel waveform records (from capture_waveforms) to an .npz for the viewer.
    Scalar fields become parallel arrays; the variable-length arrays are object arrays."""
    if not recs:
        return
    scal = ("sample", "ev", "pix_id", "x", "y", "cat", "q_coll", "t_coll", "t_near",
            "sig_t", "n_hits", "thr", "reset")
    vararr = ("hit_t", "hit_q", "t", "cur", "chg")
    kw = {k: np.asarray([r[k] for r in recs]) for k in scal}
    for k in vararr:
        kw[k] = _obj_array([r[k] for r in recs])
    np.savez(path, **kw)
    print(f"  wrote {len(recs)} pixel waveforms to {path}")


def load_spectra(path):
    """Inverse of save_spectra -> the `spectra` dict consumed by analyze_spectra."""
    d = np.load(path, allow_pickle=True)
    meta = {}
    for k in d.files:
        if k.startswith("meta_"):
            v = d[k]
            meta[k[5:]] = v.item() if v.ndim == 0 else v
    has_aux = ("nominal_aux_dt" in d.files)
    def _aux(prefix, i=None):
        # tolerate files written before a given aux key existed (e.g. t_near/dt_near):
        # missing entries come back empty and downstream code falls back gracefully.
        out = {}
        for k in _AUX_KEYS:
            key = f"{prefix}_aux_{k}"
            if not has_aux or key not in d.files:
                out[k] = np.empty(0)
            else:
                out[k] = np.asarray(d[key] if i is None else d[key][i], float)
        return out
    spectra = dict(meta=meta,
                   nominal=(np.asarray(d["nominal_q"], float),
                            np.asarray(d["nominal_tr"], bool), _aux("nominal")))
    for name in ("threshold", "reset", "induction"):
        vals, qs, trs = d[name + "_vals"], d[name + "_q"], d[name + "_tr"]
        spectra[name] = [(float(vals[i]), np.asarray(qs[i], float), np.asarray(trs[i], bool),
                          _aux(name, i)) for i in range(len(vals))]
    # Two-hit populations; absent in files written before the migration study, in which
    # case the analysis simply skips the migration plots.
    spectra["nominal_n2"] = ((np.asarray(d["nominal_n2_q"], float),
                              np.asarray(d["nominal_n2_tr"], bool), _aux("nominal_n2"))
                             if "nominal_n2_q" in d.files else None)
    for name in ("threshold", "reset", "induction"):
        key = name + "_n2_vals"
        if key not in d.files:
            spectra[name + "_n2"] = []
            continue
        vals, qs, trs = d[key], d[name + "_n2_q"], d[name + "_n2_tr"]
        spectra[name + "_n2"] = [(float(vals[i]), np.asarray(qs[i], float),
                                  np.asarray(trs[i], bool), _aux(name + "_n2", i))
                                 for i in range(len(vals))]
    spectra["nominal_all"] = ((np.asarray(d["nominal_all_q"], float),
                               np.asarray(d["nominal_all_tr"], bool),
                               np.asarray(d["nominal_all_n"], int))
                              if "nominal_all_q" in d.files else None)
    for name in ("threshold", "reset", "induction"):
        key = name + "_all_vals"
        if key not in d.files:
            spectra[name + "_all"] = []
            continue
        v = d[key]
        spectra[name + "_all"] = [(float(v[i]), np.asarray(d[name + "_all_q"][i], float),
                                   np.asarray(d[name + "_all_tr"][i], bool),
                                   np.asarray(d[name + "_all_n"][i], int))
                                  for i in range(len(v))]
    gthr, grst, gq, gtr = d["grid_thr"], d["grid_rst"], d["grid_q"], d["grid_tr"]
    spectra["grid"] = [(float(gthr[i]), int(grst[i]), np.asarray(gq[i], float),
                        np.asarray(gtr[i], bool), _aux("grid", i))
                       for i in range(len(gthr))]
    samples = {}
    if "samp_names" in d.files:
        for name in [str(x) for x in np.atleast_1d(d["samp_names"])]:
            s = dict(kind=str(d[f"samp_{name}_kind"]), theta=float(d[f"samp_{name}_theta"]),
                     n=int(d[f"samp_{name}_n"]))
            for which in ("nominal", "off"):
                qk = f"samp_{name}_{which}_q"
                if qk in d.files:
                    aux = {k: (np.asarray(d[f"samp_{name}_{which}_aux_{k}"], float)
                               if f"samp_{name}_{which}_aux_{k}" in d.files else np.empty(0))
                           for k in _AUX_KEYS}     # tolerate pre-t_near files
                    s[which] = (np.asarray(d[qk], float), np.asarray(d[f"samp_{name}_{which}_tr"], bool), aux)
                else:
                    s[which] = None
            for fld in ("n_scan", "n_ind_scan"):
                k = f"samp_{name}_{fld}"
                s[fld] = int(d[k]) if k in d.files else 0
            for fld in ("thr_grid", "scan_thresholds"):
                k = f"samp_{name}_{fld}"
                s[fld] = np.asarray(d[k], float) if k in d.files else None
            for knob in ("threshold", "reset", "induction"):
                pre = f"samp_{name}_scan_{knob}"
                if pre + "_vals" not in d.files:
                    s["scan_" + knob] = []
                    continue
                v = d[pre + "_vals"]
                s["scan_" + knob] = [
                    (float(v[i]), np.asarray(d[pre + "_q_on"][i], float),
                     np.asarray(d[pre + "_tr_on"][i], bool),
                     np.asarray(d[pre + "_q2_on"][i], float),
                     np.asarray(d[pre + "_tr2_on"][i], bool),
                     np.asarray(d[pre + "_q_off"][i], float),
                     np.asarray(d[pre + "_tr_off"][i], bool)) for i in range(len(v))]
                # multiplicity-inclusive companion (tolerate older files without it)
                if pre + "_aq_on" in d.files:
                    s["scan_" + knob + "_allmult"] = [
                        (float(v[i]), np.asarray(d[pre + "_aq_on"][i], float),
                         np.asarray(d[pre + "_atr_on"][i], bool),
                         np.asarray(d[pre + "_an_on"][i], np.int16),
                         np.asarray(d[pre + "_aq_off"][i], float),
                         np.asarray(d[pre + "_atr_off"][i], bool),
                         np.asarray(d[pre + "_an_off"][i], np.int16)) for i in range(len(v))]
                else:
                    s["scan_" + knob + "_allmult"] = []
            gpre = f"samp_{name}_grid"
            if gpre + "_thr" in d.files:
                gt, grr = d[gpre + "_thr"], d[gpre + "_rst"]
                s["grid"] = [(float(gt[i]), int(grr[i]),
                              np.asarray(d[gpre + "_q"][i], float),
                              np.asarray(d[gpre + "_tr"][i], bool),
                              {k: np.empty(0) for k in _AUX_KEYS})
                             for i in range(len(gt))]
            else:
                s["grid"] = []
            samples[name] = s
    spectra["samples"] = samples

    # shower-energy scan (Part D)
    escan = []
    if "escan_E" in d.files:
        Es = np.asarray(d["escan_E"], float)
        for i in range(len(Es)):
            escan.append((float(Es[i]), int(d["escan_nev"][i]),
                          np.asarray(d["escan_q1on"][i], float),
                          np.asarray(d["escan_tr1on"][i], bool),
                          np.asarray(d["escan_qAon"][i], float),
                          np.asarray(d["escan_trAon"][i], bool),
                          np.asarray(d["escan_nAon"][i], np.int16),
                          np.asarray(d["escan_q1off"][i], float),
                          np.asarray(d["escan_tr1off"][i], bool),
                          np.asarray(d["escan_qAoff"][i], float),
                          np.asarray(d["escan_trAoff"][i], bool),
                          np.asarray(d["escan_nAoff"][i], np.int16)))
    spectra["energy_scan"] = escan
    return spectra


def produce_burst_only(ctx, args):
    """GPU, FAST PATH (`--burst-only`): parallel-muon burst-mode waveforms + a cheap FEE-only
    self-trigger threshold scan, and NOTHING else -- no showers, no template grid, no composite
    fits, no induction/reset scans. Built for quick turnaround on the muon burst study.

    For each `--burst-thetas` angle it (1) captures the transverse-neighbour burst waveforms at
    each `--burst-inductions` scale (threshold-independent, so captured once), and (2) runs the
    two-point induction on/off single-/all-multiplicity scan over the thresholds (FEE-only on two
    cached inductions -- the only place a threshold enters). Returns a `spectra` dict with the
    shower-dependent slots left EMPTY, so save/load round-trip and the lean analyzer just skips
    them. Peak memory is one angle's on+off cache at a time -- no shower halo, so no big-RAM node.
    """
    base_thr, base_reset = base_config(ctx, args)
    ts = float(ctx.detector.TIME_SAMPLING)
    noise_e = (float(args.noise_e) if args.noise_e is not None
               else float(np.atleast_1d(getattr(ctx.detector, "UNCORRELATED_NOISE_CHARGE", 500.0)).ravel()[0]))
    rng = np.random.default_rng(args.seed)
    thetas = [float(t) for t in (args.burst_thetas or [0.0])]
    thr_list = list(args.muon_thresholds) if args.muon_thresholds else list(args.thresholds)
    cad_us = (float(args.burst_cadence) if args.burst_cadence and args.burst_cadence > 0
              else (3.0 + float(ctx.detector.ADC_HOLD_DELAY)) * float(ctx.detector.CLOCK_CYCLE))
    cad_ticks = max(int(round(cad_us / ts)), 1)
    print(f"\nBURST-ONLY fast mode: thetas={thetas}, burst {int(args.burst_events)} ev x "
          f"{len(args.burst_inductions)} scales (+-{args.burst_neighbors} pitch), threshold scan "
          f"{thr_list} on {int(args.muon_scan_events)} ev, reset={base_reset}, "
          f"cadence {cad_us:.2f} us ({cad_ticks} ticks); noise sigma={noise_e:.0f} e-")

    samples, burst_recs = {}, []
    for it, theta in enumerate(thetas):
        name = "muon_th%02d" % int(round(theta))
        mu_n = int(args.muon_events)
        print(f"\nMuon '{name}' (theta={theta:.0f} deg to pixel plane): building {mu_n} events...")
        mu_raw = [build_muon_event(ctx, rng, theta_deg=theta, length_cm=args.muon_length)
                  for _ in range(mu_n)]
        mu_drift = [d for d in (event_drift(ctx, r) for r in mu_raw) if d is not None]
        sd = args.seed + 7000 + it * 1000
        print(f"  {len(mu_drift)}/{mu_n} events produced collectable charge")

        nbev = min(int(args.burst_events), len(mu_drift))
        if nbev > 0:
            print(f"  burst capture: {nbev} events x {len(args.burst_inductions)} induction scales...")
            burst_recs += capture_burst(ctx, mu_drift[:nbev], name, theta, args.burst_inductions,
                                        cad_ticks, int(args.burst_neighbors), noise_e,
                                        seed0=sd + 5000, collect_frac=args.collect_frac)

        ns = min(int(args.muon_scan_events), len(mu_drift))   # 0 disables the scan (as documented)
        sc, sca = [], []
        if ns > 0:
            set_induction(ctx, 1.0); set_periodic_reset(ctx, base_reset)
            mu_evs = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=sd + i,
                                               collect_frac=args.collect_frac)
                                  for i, d in enumerate(mu_drift[:ns])) if e is not None]
            set_induction(ctx, 0.0)
            mu_off = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=sd + 500 + i,
                                               collect_frac=args.collect_frac)
                                  for i, d in enumerate(mu_drift[:ns])) if e is not None]
            print(f"  self-trigger threshold scan ({len(thr_list)} pts) on {ns} events, "
                  f"induction on/off...")
            sc, sca = aggregate_sample_scan(ctx, mu_evs, mu_off, thr_list,
                                            lambda v: (float(v), base_reset), seed0=sd + 2000)
            mu_evs = mu_off = None
        samples[name] = dict(kind="muon", theta=theta, n=len(mu_drift), n_scan=ns,
                             scan_threshold=sc, scan_threshold_allmult=sca,
                             scan_thresholds=np.asarray(thr_list, float))
        mu_drift = mu_raw = None
        set_induction(ctx, 1.0)

    det_name, _geom, det_path = vd.detector_identity(ctx)
    meta = dict(base_threshold=base_thr, base_reset=base_reset, ts=ts, noise_e=noise_e,
                thr_sigma=float(np.atleast_1d(ctx.detector.DISCRIMINATOR_NOISE).ravel()[0]),
                qmax=30000.0, thresholds=np.asarray(thr_list, float),
                resets=np.asarray([base_reset], int),
                inductions=np.asarray(args.burst_inductions, float),
                thr_grid=np.asarray(thr_list, float), rst_grid=np.asarray([base_reset], int),
                n_shower_main=0, n_shower_ind=0,
                detector_name=str(det_name), detector_path=str(det_path))
    empty = (np.empty(0), np.empty(0, bool), {k: np.empty(0) for k in _AUX_KEYS})
    return dict(meta=meta, nominal=empty, threshold=[], reset=[], induction=[],
                grid=[], samples=samples, waveforms=[], burst=burst_recs, energy_scan=[])


def analyze_burst_only(spectra, outdir):
    """CPU, FAST PATH: the lean analyzer for `--burst-only` spectra -- the burst-mode plots, the
    per-topology multiplicity-inclusive rate, and the operating-point recommendation. Skips the
    whole shower template/composite/sensitivity machinery (there is no shower data), so it is
    robust on the minimal spectra dict and refits locally with no GPU."""
    m = spectra["meta"]
    noise_e, ts = float(m["noise_e"]), float(m["ts"])
    print("\n=== BURST-ONLY analysis (fast path) ===")
    analyze_burst(spectra.get("burst", []), outdir, noise_e)
    for name, sm in spectra.get("samples", {}).items():
        s_title, _pop, s_slug = sample_labels(name, sm)
        plot_inclusive_rate(s_title, sm, int(sm.get("n_scan", 1)) or 1, ts, outdir, "_" + s_slug)
    recommend_operating_point(spectra, {}, outdir, noise_e)


def produce_spectra(ctx, args):
    """GPU: run the full simulation and return a `spectra` dict of raw single-hit (q, tr)
    for every config (nominal, threshold/reset/induction scans, induction-off template
    grid) + metadata. This is the expensive part; the fitting (analyze_spectra) is split
    off so the sim runs once and fits are iterated offline via --refit."""
    base_thr, base_reset = base_config(ctx, args)
    ts = float(ctx.detector.TIME_SAMPLING)
    thr_sigma = float(np.atleast_1d(ctx.detector.DISCRIMINATOR_NOISE).ravel()[0])
    noise_e = (float(args.noise_e) if args.noise_e is not None
               else float(np.atleast_1d(getattr(ctx.detector, "UNCORRELATED_NOISE_CHARGE", 500.0)).ravel()[0]))
    reset_state = "off" if base_reset <= 0 else f"{base_reset} cycles"
    print(f"\nBase operating point: threshold={base_thr:.0f} e-, periodic_reset={reset_state}, "
          f"induction=1.0;  langaus noise sigma={noise_e:.0f} e-")

    rng = np.random.default_rng(args.seed)
    if args.edep_h5:
        print(f"\nLoading real edep-sim showers from {args.edep_h5} ...")
        raw_events = load_edep_events(args.edep_h5, args.n_events)
        print(f"  loaded {len(raw_events)} events "
              f"({sum(len(e) for e in raw_events)} segments total)")
        f_out = events_out_of_volume(raw_events, ctx)
        if args.recenter_showers:
            raw_events = recenter_events(raw_events, ctx, rng)
            print(f"  re-centred into this config's active volume "
                  f"({100 * f_out:.1f}% of deposits were outside it before)")
        elif f_out > 0.01:
            print(f"  *** WARNING: {100 * f_out:.1f}% of deposits fall OUTSIDE "
                  f"{args.config}'s active volume.\n"
                  f"  *** This edep file was almost certainly generated in a different "
                  f"detector geometry.\n"
                  f"  *** Pass --recenter-showers to place them in this one, or omit "
                  f"--edep-h5 to use parametric showers.")
    else:
        print(f"\nGenerating {args.n_events} parametric shower events "
              f"(E={args.shower_energy:.0f} MeV, {args.n_dep} deposits each)...")
        raw_events = [build_shower_event(ctx, rng, energy_mev=args.shower_energy, n_dep=args.n_dep)
                      for _ in range(args.n_events)]

    print("Computing drift + pixel geometry (cached, once)...")
    drift_cache = [dc for dc in (event_drift(ctx, raw) for raw in raw_events) if dc is not None]
    n_eff = len(drift_cache)
    print(f"  {n_eff}/{args.n_events} events produced collectable charge")
    if n_eff == 0:
        sys.exit("No events produced any pixels -- check shower placement / config.")

    # MEMORY ORDER MATTERS HERE. Each cached pre-signal holds a dense (n_pixels x n_ticks)
    # float32 array, and n_pixels scales with MAX_RADIUS**2 -- which is read from the
    # RESPONSE FILE (int(response.shape[0] * RESPONSE_BIN_SIZE // PIXEL_PITCH)), not from
    # anything this script sets. fsd_cube's dedicated 25x25 response gives MAX_RADIUS=12
    # where the old shared 9x9 table gave 4, i.e. ~3-4x the pixels per event and a
    # correspondingly bigger cache. Holding the induction-ON and induction-OFF caches at
    # full length simultaneously is what OOMs the node.
    #
    # So: build the OFF cache first, let the template grid consume it in full, then keep
    # only the `ind_n` events the induction scan still needs before building the ON cache.
    # Peak residency drops from 2*n_eff to n_eff + ind_n with no change to any result --
    # the seeds are per-event and unchanged, so both caches are bit-identical to before.
    ind_n = min(args.induction_events, n_eff) if args.induction_events else n_eff

    print("Inducing induction-OFF pre-FEE signals (pure shower, for templates)...")
    set_induction(ctx, 0.0)
    set_periodic_reset(ctx, base_reset)
    evs_off = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=args.seed + 200 + i,
                                        collect_frac=args.collect_frac)
                           for i, d in enumerate(drift_cache)) if e is not None]
    print(f"  pre-signal cache: {_cache_mb(evs_off):.1f} MB "
          f"({_cache_mb(evs_off) / max(len(evs_off), 1):.2f} MB/event, "
          f"MAX_RADIUS={ctx.detector.MAX_RADIUS} pix)")

    thr_grid = sorted(set(float(t) for t in args.thresholds) | {base_thr})
    rst_grid = sorted(set(int(r) for r in args.resets) | {int(base_reset)})
    n_nodes = (len(thr_grid) * len(rst_grid) if args.template_grid == "full"
               else len(thr_grid) + len(rst_grid) - 1)
    print(f"Aggregating induction-off template grid ({len(thr_grid)}x{len(rst_grid)}, "
          f"{args.template_grid} -> {n_nodes} nodes)...")
    grid_spectra = aggregate_template_grid(ctx, evs_off, thr_grid, rst_grid,
                                           seed0=args.seed + 6000, base_thr=base_thr,
                                           base_reset=base_reset, mode=args.template_grid)
    if ind_n < len(evs_off):                 # release the tail; only the scan needs it now
        evs_off = evs_off[:ind_n]
        print(f"  released induction-off cache beyond {ind_n} events "
              f"(now {_cache_mb(evs_off):.1f} MB)")

    print("Inducing nominal-induction pre-FEE signals (cached)...")
    set_induction(ctx, 1.0)
    set_periodic_reset(ctx, base_reset)
    evs = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=args.seed + 100 + i,
                                    collect_frac=args.collect_frac)
                       for i, d in enumerate(drift_cache)) if e is not None]
    print(f"  pre-signal cache: {_cache_mb(evs):.1f} MB "
          f"(total resident {_cache_mb(evs) + _cache_mb(evs_off):.1f} MB)")
    if ind_n < len(drift_cache):             # drift geometry is only re-induced for the scan
        drift_cache = drift_cache[:ind_n]

    print("Aggregating nominal single-hit spectrum...")
    nom_pops = aggregate_hits(ctx, evs, base_thr, base_reset, 1.0, seed0=args.seed + 1)
    q0, tr0, aux0 = nom_pops[1]
    nom2 = nom_pops[2]
    qmax = float(min(np.max(q0[q0 > 0]), 50000.0)) if np.any(q0 > 0) else 30000.0

    print("Aggregating threshold + reset scans (single-hit + two-hit populations)...")
    thr_items, thr_items2, thr_all = aggregate_fixed_scan(ctx, evs, args.thresholds,
                                     lambda v: (float(v), base_reset), seed0=args.seed + 2000)
    rst_items, rst_items2, rst_all = aggregate_fixed_scan(ctx, evs, args.resets,
                                     lambda v: (base_thr, int(v)), seed0=args.seed + 3000)
    set_periodic_reset(ctx, base_reset)

    print(f"Aggregating induction scan ({ind_n} of {n_eff} events; re-induces per scale)...")
    ind_items, ind_items2, ind_all = aggregate_induction_scan(
        ctx, drift_cache[:ind_n], args.inductions, base_thr,
        base_reset, seed0=args.seed + 4000,
        evs_off=evs_off[:ind_n], collect_frac=args.collect_frac)

    # --- control-sample DIAGNOSTICS: the shower plus straight-muon topologies, each with a
    #     nominal (induction-on) and an induction-off single-hit sample for the anatomy /
    #     timing checks, plus per-pixel waveforms for the viewer. -----------------------------
    set_induction(ctx, 1.0); set_periodic_reset(ctx, base_reset)
    off_node = next((g for g in grid_spectra
                     if abs(g[0] - base_thr) < 1e-6 and int(g[1]) == base_reset), None)
    samples = {"shower": dict(kind="shower", theta=float("nan"), n=n_eff,
                              nominal=(q0, tr0, aux0),
                              off=(off_node[2], off_node[3], off_node[4]) if off_node else None)}
    waveforms = capture_waveforms(ctx, evs, base_thr, base_reset, args.seed + 9000, "shower",
                                  args.wf_events, args.wf_max_pix)
    # Every shower cache is finished with at this point. Release them before the muon
    # topologies start allocating their own: a through-going muon crosses the whole anode
    # and so lights far more pixels per event than a compact shower does, and at
    # MAX_RADIUS=12 holding both sets at once is what tips the node over.
    if evs or evs_off or drift_cache:
        print(f"  releasing shower caches "
              f"({_cache_mb(evs) + _cache_mb(evs_off):.1f} MB) before the muon samples")
    evs = evs_off = drift_cache = None

    # --- SHOWER-ENERGY SCAN (Part D): does a denser (higher-energy) core induce more? Each
    #     energy is its own small shower sub-sample, induced ON and OFF; we keep the single-hit
    #     and all-multiplicity populations so the analysis can read the pure-induction rate, the
    #     multiplicity-inclusive fired rate (ON vs OFF), and the mean charge vs energy. n_dep is
    #     scaled with energy so the per-tracklet dE/dx stays MIP-like and the core densifies
    #     through GEOMETRY (overlapping tracklets), not through a fake dE/dx that would merely
    #     rescale charge uniformly. -----------------------------------------------------------
    energy_scan = []
    if args.shower_energies and not args.edep_h5:
        es_n = (min(int(args.energy_scan_events), int(args.n_events))
                if args.energy_scan_events else int(args.n_events))
        energies = sorted(float(e) for e in args.shower_energies)
        print(f"\nShower-energy scan: {energies} MeV, {es_n} showers each "
              f"(n_dep ~ energy; higher energy = denser core)...")
        for E in energies:
            ndep = max(int(round(args.n_dep * E / max(args.shower_energy, 1.0))), 50)
            e_raw = [build_shower_event(ctx, rng, energy_mev=E, n_dep=ndep) for _ in range(es_n)]
            e_drift = [d for d in (event_drift(ctx, r) for r in e_raw) if d is not None]
            se = args.seed + 30000 + int(round(E))
            set_induction(ctx, 1.0); set_periodic_reset(ctx, base_reset)
            e_on = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=se + i,
                                             collect_frac=args.collect_frac)
                                for i, d in enumerate(e_drift)) if e is not None]
            pon = aggregate_hits(ctx, e_on, base_thr, base_reset, 1.0, seed0=se + 100)
            set_induction(ctx, 0.0)
            e_off = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=se + 500 + i,
                                              collect_frac=args.collect_frac)
                                 for i, d in enumerate(e_drift)) if e is not None]
            pof = aggregate_hits(ctx, e_off, base_thr, base_reset, 0.0, seed0=se + 600)
            ne = len(e_drift)
            energy_scan.append((float(E), int(ne)) + pon[1][:2] + pon["all"] + pof[1][:2]
                               + pof["all"])
            print(f"  E={E:.0f} MeV: {ne} showers, single-hit {pon[1][0].size} ON / "
                  f"{pof[1][0].size} OFF, fired {pon['all'][0].size} ON / {pof['all'][0].size} OFF")
            e_on = e_off = e_drift = e_raw = None
        set_induction(ctx, 1.0)

    burst_recs = []
    burst_thetas = set(int(round(t)) for t in (args.burst_thetas or []))
    for it, theta in enumerate(args.muon_thetas or []):
        name = "muon_th%02d" % int(round(theta))
        mu_n = min(int(args.muon_events), int(args.n_events))
        print(f"\nMuon sample '{name}' (theta={theta:.0f} deg to pixel plane, {mu_n} events)...")
        mu_raw = [build_muon_event(ctx, rng, theta_deg=float(theta), length_cm=args.muon_length)
                  for _ in range(mu_n)]
        mu_drift = [d for d in (event_drift(ctx, r) for r in mu_raw) if d is not None]
        sd = args.seed + 7000 + it * 1000
        set_induction(ctx, 1.0)
        mu_evs = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=sd + i,
                                           collect_frac=args.collect_frac)
                              for i, d in enumerate(mu_drift)) if e is not None]
        nom = aggregate_singlehits(ctx, mu_evs, base_thr, base_reset, 1.0, seed0=sd + 100)
        set_induction(ctx, 0.0)
        mu_off = [e for e in (event_induce(ctx, d[0], d[1], d[2], seed=sd + 500 + i,
                                           collect_frac=args.collect_frac)
                              for i, d in enumerate(mu_drift)) if e is not None]
        off = aggregate_singlehits(ctx, mu_off, base_thr, base_reset, 0.0, seed0=sd + 600)
        print(f"  pre-signal cache: {_cache_mb(mu_evs) + _cache_mb(mu_off):.1f} MB "
              f"({len(mu_evs)} on + {len(mu_off)} off events)")
        samples[name] = dict(kind="muon", theta=float(theta), n=len(mu_drift),
                             nominal=nom, off=off)
        # Knob scans on THIS topology. The shower scans above cannot answer "does a lower
        # threshold buy induction hits" for a muon: showers and muons expose completely
        # different neighbour geometries (a compact core vs a long pillar), and it is the
        # muon that isolates induction cleanly.
        ns = min(int(args.muon_scan_events), len(mu_evs), len(mu_off))
        if ns > 0:
            mu_thr = list(args.muon_thresholds) if args.muon_thresholds else list(args.thresholds)
            print(f"  scanning threshold ({len(mu_thr)} pts) + reset ({len(args.resets)} pts) "
                  f"on {ns} events, induction on/off...")
            samples[name]["n_scan"] = ns
            (samples[name]["scan_threshold"],
             samples[name]["scan_threshold_allmult"]) = aggregate_sample_scan(
                ctx, mu_evs[:ns], mu_off[:ns], mu_thr,
                lambda v: (float(v), base_reset), seed0=sd + 2000)
            (samples[name]["scan_reset"],
             samples[name]["scan_reset_allmult"]) = aggregate_sample_scan(
                ctx, mu_evs[:ns], mu_off[:ns], args.resets,
                lambda v: (base_thr, int(v)), seed0=sd + 4000)
            set_periodic_reset(ctx, base_reset)
            # This topology's OWN induction-off template grid. The shower grid cannot be
            # reused: it is a template for the EM-shower charge spectrum, and a MIP's is a
            # different distribution entirely, so fitting muon scan points against it would
            # freeze the wrong shape. FEE-only on the cached induction-off pre-signals.
            mu_thr_grid = sorted(set(float(t) for t in mu_thr) | {base_thr})
            print(f"  building this topology's induction-off template grid "
                  f"({len(mu_thr_grid)}x{len(rst_grid)}, {args.template_grid})...")
            samples[name]["grid"] = aggregate_template_grid(
                ctx, mu_off[:ns], mu_thr_grid, rst_grid, seed0=sd + 6000,
                base_thr=base_thr, base_reset=base_reset, mode=args.template_grid)
            samples[name]["thr_grid"] = np.asarray(mu_thr_grid, float)
            samples[name]["scan_thresholds"] = np.asarray(mu_thr, float)
            set_periodic_reset(ctx, base_reset)
            mi_n = min(int(args.muon_induction_events), len(mu_drift))
            if mi_n > 0:
                print(f"  induction scan ({len(args.inductions)} scales; re-induces per "
                      f"scale) on {mi_n} events...")
                samples[name]["n_ind_scan"] = mi_n
                samples[name]["scan_induction"] = aggregate_sample_induction_scan(
                    ctx, mu_drift[:mi_n], args.inductions, base_thr, base_reset,
                    seed0=sd + 8000, evs_off=mu_off[:mi_n],
                    collect_frac=args.collect_frac)
                set_induction(ctx, 1.0); set_periodic_reset(ctx, base_reset)
        # BURST MODE (Part A): for a through-going track PARALLEL to the pixel plane, capture the
        # forced-readout waveforms of the transverse (pure-induction) neighbour pixels at each
        # induction scale -- the threshold-independent induction-model test. Reuses mu_drift; the
        # scan reset above is FEE-only, so pixels_signals is unchanged.
        if int(round(theta)) in burst_thetas and int(args.burst_events) > 0 and mu_drift:
            nbev = min(int(args.burst_events), len(mu_drift))
            cad_us = (float(args.burst_cadence) if args.burst_cadence and args.burst_cadence > 0
                      else (3.0 + float(ctx.detector.ADC_HOLD_DELAY))
                      * float(ctx.detector.CLOCK_CYCLE))
            cad_ticks = max(int(round(cad_us / ts)), 1)
            print(f"  burst-mode capture: {nbev} events x {len(args.burst_inductions)} induction "
                  f"scales, +-{args.burst_neighbors} pitch neighbours, "
                  f"cadence {cad_us:.2f} us ({cad_ticks} ticks)...")
            burst_recs += capture_burst(ctx, mu_drift[:nbev], name, float(theta),
                                        args.burst_inductions, cad_ticks,
                                        int(args.burst_neighbors), noise_e, seed0=sd + 5000,
                                        collect_frac=args.collect_frac)
            set_induction(ctx, 1.0); set_periodic_reset(ctx, base_reset)
        waveforms += capture_waveforms(ctx, mu_evs, base_thr, base_reset, sd + 900, name,
                                       args.wf_events, args.wf_max_pix)
        mu_evs = mu_off = mu_drift = mu_raw = None   # free before the next angle allocates
    set_induction(ctx, 1.0); set_periodic_reset(ctx, base_reset)

    det_name, _geom, det_path = vd.detector_identity(ctx)
    meta = dict(base_threshold=base_thr, base_reset=base_reset, ts=ts, noise_e=noise_e,
                thr_sigma=thr_sigma, qmax=qmax, thresholds=np.asarray(args.thresholds, float),
                resets=np.asarray(args.resets, int), inductions=np.asarray(args.inductions, float),
                thr_grid=np.asarray(thr_grid, float), rst_grid=np.asarray(rst_grid, int),
                n_shower_main=n_eff, n_shower_ind=ind_n,
                detector_name=str(det_name), detector_path=str(det_path))
    return dict(meta=meta, nominal=(q0, tr0, aux0), threshold=thr_items, reset=rst_items,
                induction=ind_items, grid=grid_spectra, samples=samples, waveforms=waveforms,
                # burst-mode neighbour waveforms + the shower-energy scan (Parts A/D)
                burst=burst_recs, energy_scan=energy_scan,
                # parallel TWO-hit populations (same FEE passes): the migration target that
                # makes the threshold/reset response visible -- see event_fee_hits.
                nominal_n2=nom2, threshold_n2=thr_items2, reset_n2=rst_items2,
                induction_n2=ind_items2,
                # every pixel that fired, with its multiplicity -- the only record that can
                # represent 3+ hits (~6 bytes/pixel; the aux arrays are ~74% of this file)
                nominal_all=nom_pops["all"], threshold_all=thr_all,
                reset_all=rst_all, induction_all=ind_all)


def analyze_spectra(spectra, outdir):
    """CPU: fit + plot everything from a `spectra` dict (from produce_spectra or a saved
    --refit file). No GPU needed, so fits can be iterated locally."""
    m = spectra["meta"]
    base_thr, base_reset = float(m["base_threshold"]), int(m["base_reset"])
    ts, noise_e, thr_sigma = float(m["ts"]), float(m["noise_e"]), float(m["thr_sigma"])
    qmax = float(m["qmax"])
    n_main, n_ind = int(m["n_shower_main"]), int(m["n_shower_ind"])
    thresholds = np.asarray(m["thresholds"], float)
    resets = np.asarray(m["resets"], int)
    inductions = np.asarray(m["inductions"], float)
    thr_grid = np.asarray(m["thr_grid"], float)
    rst_grid = np.asarray(m["rst_grid"], int)
    xlim, mpv_ylim = (0.0, 50.0), (0.0, 50.0)

    print("\n=== Shower-template grid fit + closed-form parametrization ===")
    grid, node_fits = fit_template_grid(spectra["grid"], thr_grid, rst_grid, ts, qmax, n_main, noise_e)
    tparam = TemplateParam(grid, ts)
    for k in ("mpv_L", "A_L", "eta_L", "mu_G"):
        f = tparam.forms.get(k)
        if f:
            print(f"  {k:>6}: R2={f['r2']:.3f}  best closed form  {f['name']}")
    plot_shower_templates(node_fits, thr_grid, base_reset, outdir)
    plot_template_params(grid, tparam, ts, outdir)

    def template_fn(thr, cycles):
        t = node_fits.get((float(thr), int(cycles)))
        if t is not None and t.get("ok"):
            return t
        return tparam.template_at(thr, cycles)

    def dist_panels(fits, disp):
        return [(d, f, q, tr) for d, (val, f, q, tr) in zip(disp, fits)]

    def mult_panels(items1, items2, disp, thr_of):
        """Pair each scan point's single-hit and two-hit spectra for the migration plot."""
        d2 = {float(v): q for (v, q, _t, _a) in (items2 or [])}
        return [(d, q, d2.get(float(v), np.empty(0)), float(thr_of(v)))
                for d, (v, q, _t, _a) in zip(disp, items1)]

    print("\n=== Nominal composite template fit ===")
    q0, tr0 = np.asarray(spectra["nominal"][0], float), np.asarray(spectra["nominal"][1], bool)
    aux0 = spectra["nominal"][2] if len(spectra["nominal"]) > 2 else {}
    fit0 = fit_shower_plus_induction(q0, base_thr, template_fn(base_thr, base_reset),
                                     qmax=qmax, n_shower=n_main, noise_e=noise_e)
    if fit0 and fit0.get("ok"):
        print(f"  single-hit pixels: {fit0['n_single']}  Ind.MPV={fit0['mpv_ind']:.0f} e-  "
              f"Shower.MPV_L={fit0['mpv_shw']:.0f} e-  mu_G={fit0['mu_G']:.0f} e-  "
              f"f_ind={fit0['frac_ind']:.2f}  chi2/ndf={fit0['chi2ndf']:.2f}")
        lo = float(np.median(q0[~tr0])) if np.any(~tr0) else np.nan
        hi = float(np.median(q0[tr0])) if np.any(tr0) else np.nan
        print(f"  backtrack medians: induction={lo:.0f} e-  shower={hi:.0f} e-  "
              f"(N_ind={int((~tr0).sum())}, N_shw={int(tr0.sum())}); truth f_ind="
              f"{(~tr0).sum() / max(tr0.size, 1):.2f}")
        plot_headline(None, q0, tr0, fit0, outdir, xlim=xlim, thr_sigma=thr_sigma)
    else:
        print("  !! nominal fit failed:", fit0.get("reason") if fit0 else "no data")

    # --- TRUTH DIAGNOSTICS over the control samples (shower + muon topologies) ---------
    #  (1) plot_offpeak_anatomy: what the two peaks of the induction-off spectrum are, from
    #      the collected-charge truth; (2) plot_hit_timing: how many spatial-"shower" hits
    #      actually fired BEFORE their charge arrived (induction-triggered).
    samples = spectra.get("samples", {})
    if samples:
        print("\n=== Truth diagnostics (control samples) ===")
        for name, sm in samples.items():
            nsh = int(sm.get("n", n_main)) or n_main
            s_title, s_pop, s_slug = sample_labels(name, sm)
            suf = "" if name == "shower" else "_" + s_slug
            lab = s_title
            off, nom = sm.get("off"), sm.get("nominal")
            if off is not None:
                qo, tro, auxo = off
                coll = np.asarray(tro, bool)
                # NB do NOT name this `m` -- that shadows the `m = spectra["meta"]` dict
                # used by the results-npz save at the end of this function.
                # use the collection sub-sample, but fall back to the whole off-sample when
                # there are too few charged pads (e.g. the theta=90 muon, which is ~pure
                # induction) -- otherwise the plot silently never gets made.
                selo = coll if coll.sum() >= 20 else np.ones(qo.size, bool)
                plot_offpeak_anatomy(np.asarray(qo, float)[selo], coll[selo],
                                     _aux_sel(auxo, selo, qo.size),
                                     base_thr, outdir, qmax=qmax, n_shower=nsh,
                                     suffix=suf, title=lab, population=s_pop)
            if nom is not None and np.size(nom[2].get("dt", [])) == np.size(nom[0]):
                qn, trn, auxn = nom
                tinfo = plot_hit_timing(qn, trn, auxn, base_thr, outdir, n_shower=nsh,
                                        suffix=suf, title=lab, population=s_pop)
                if tinfo:
                    n_sp = int(np.asarray(trn, bool).sum())
                    sp_b = np.asarray(trn, bool)
                    dtn = np.asarray(auxn.get("dt_near", []), float)
                    if dtn.size == qn.size:         # the headline induction-lead numbers
                        f = np.isfinite(dtn)
                        mi = np.median(dtn[~sp_b & f]) if (~sp_b & f).any() else np.nan
                        mc = np.median(dtn[sp_b & f]) if (sp_b & f).any() else np.nan
                        print("  %-11s dt_near median: induction %+.2f us, collection %+.2f us "
                              "-> induction leads by %.2f us" % (name, mi, mc, mc - mi))
                    if np.isfinite(tinfo["lo"]):
                        print("  %-11s dt-window [%.2f,%.2f] us | %d/%d spatial-'shower' hits "
                              "induction-triggered -> causal f_ind %.2f (spatial %.2f)"
                              % (name, tinfo["lo"], tinfo["hi"], tinfo["n_flipped"], n_sp,
                                 1.0 - tinfo["causal"].sum() / max(qn.size, 1),
                                 1.0 - n_sp / max(qn.size, 1)))
                    else:
                        print("  %-11s near-pure induction (%d charged pads): relabelling skipped"
                              % (name, n_sp))

    # --- INDUCTION YIELD vs KNOB, per topology --------------------------------------------
    # The shower side costs nothing extra: its induction-OFF counterpart at every scan point
    # is already in the template grid, which was built over exactly these (threshold, reset)
    # points. The muon side comes from aggregate_sample_scan.
    def shower_scan(knob):
        grid = spectra.get("grid") or []
        if knob == "threshold":
            offmap = {float(t): (q, tr) for (t, r, q, tr, _a) in grid if int(r) == base_reset}
            items = spectra.get("threshold") or []
        else:
            offmap = {float(r): (q, tr) for (t, r, q, tr, _a) in grid
                      if abs(float(t) - base_thr) < 1e-6}
            items = spectra.get("reset") or []
        n2map = {float(r[0]): (r[1], r[2]) for r in (spectra.get(knob + "_n2") or [])}
        out = []
        for (v, q, tr, _a) in items:
            qo, tro = offmap.get(float(v), (np.empty(0), np.empty(0, bool)))
            q2, tr2 = n2map.get(float(v), (np.empty(0), np.empty(0, bool)))
            out.append((float(v), q, tr, q2, tr2, qo, tro))
        return out

    def _entry(label, scan, n_ev):
        v, non, noff, nind = _sample_scan_counts(scan)
        return (label, v, non, noff, nind, n_ev)

    thr_entries, rst_entries = [], []
    if spectra.get("grid"):
        thr_entries.append(_entry("shower", shower_scan("threshold"), n_main))
        rst_entries.append(_entry("shower", shower_scan("reset"), n_main))
    for _nm, _sm in samples.items():
        if _nm == "shower":
            continue
        _ns = int(_sm.get("n_scan", 0)) or int(_sm.get("n", 1))
        _lab = sample_labels(_nm, _sm)[0]
        if _sm.get("scan_threshold"):
            thr_entries.append(_entry(_lab, _sm["scan_threshold"], _ns))
        if _sm.get("scan_reset"):
            rst_entries.append(_entry(_lab, _sm["scan_reset"], _ns))

    if thr_entries:
        print("\n=== Induction yield vs knob, by topology ===")
        plot_sample_knob_response(
            thr_entries, r"$Q_{\mathrm{thr}}$  ($10^{3}\,e^{-}$)",
            "ti_induction_vs_threshold.png", outdir, base_x=base_thr / 1e3,
            xfn=lambda v: v / 1e3,
            title="Does a lower threshold buy induction hits?")
        plot_sample_knob_response(
            rst_entries, "periodic-reset rate  (kHz)",
            "ti_induction_vs_reset.png", outdir,
            xfn=lambda v: reset_rate_khz(int(v), ts),
            title="Induction yield vs periodic-reset rate, by topology")
        for tag, ents, scale, unit in (("threshold", thr_entries, 1e3, "ke-"),
                                       ("reset", rst_entries, 1.0, "cycles")):
            if not ents:
                continue
            print(f"  {tag} scan -- PURE-INDUCTION single hits per event "
                  f"(spatial backtrack: no charge landed):")
            # Group by the knob grid actually scanned: --muon-thresholds lets the muon
            # samples use a different (typically lower) list than the showers, and printing
            # those rows under the shower's header would silently mislabel every column.
            groups = {}
            for e in ents:
                key = tuple(np.round(np.sort(np.asarray(e[1], float)), 6))
                groups.setdefault(key, []).append(e)
            w = max(24, max(len(e[0]) for e in ents) + 2)   # fit the descriptive names
            for key, grp in groups.items():
                print("    %-*s" % (w, unit) + "".join("%9.1f" % (v / scale) for v in key))
                for (lab, v, _non, _noff, nind, n_ev) in grp:
                    r = np.asarray(nind, float)[np.argsort(np.asarray(v, float))]
                    print("    %-*s" % (w, lab) + "".join("%9.4f" % y
                                                          for y in r / max(n_ev, 1)))

    # --- FULL SHOWER-PARITY SCAN ANALYSIS PER MUON TOPOLOGY -------------------------------
    # POSITIONS track the threshold; the SCALE-FREE shape ratios (charge in units of the peak
    # position) divide it out, so they isolate reset/induction; frac_2hit catches pixels being
    # promoted OUT of the single-hit sample. Shared so every topology, shower included, is
    # scored on exactly the same observables.
    obs_keys_g = ["peak_mpv", "tail_peak", "tail_over_core", "shoulder_frac", "frac_2hit"]
    obs_lbl_g = ["Peak position", "Tail peak", "Tail/core (x)", "Shoulder frac (x)", "2-hit frac"]

    def analyze_sample(name, sm):
        """Run the shower scan machinery on one control sample: its own template grid, the
        composite fits, the scan/distribution/migration plots and its own sensitivity matrix.

        The template grid must be the SAMPLE'S OWN: the shower grid is a template for the EM
        charge spectrum, and a MIP's is a different distribution, so reusing it would freeze
        the wrong shape into every muon fit."""
        s_title, s_pop, s_slug = sample_labels(name, sm)
        suf = "_" + s_slug
        ns = int(sm.get("n_scan", 0)) or int(sm.get("n", 1))
        grid_sp = sm.get("grid") or []
        sc_thr, sc_rst = sm.get("scan_threshold") or [], sm.get("scan_reset") or []
        sc_ind = sm.get("scan_induction") or []
        if not grid_sp or not sc_thr:
            return None
        nom = sm.get("nominal")
        qn = np.asarray(nom[0], float) if nom else np.empty(0)
        s_qmax = float(min(np.max(qn[qn > 0]), 50000.0)) if np.any(qn > 0) else qmax
        s_thr_grid = np.asarray(sm.get("thr_grid") if sm.get("thr_grid") is not None
                                else thr_grid, float)
        s_scan_thr = np.asarray(sm.get("scan_thresholds") if sm.get("scan_thresholds") is not None
                                else thresholds, float)
        try:
            sgrid, snodes = fit_template_grid(grid_sp, s_thr_grid, rst_grid, ts,
                                              s_qmax, ns, noise_e)
            stp = TemplateParam(sgrid, ts)
        except Exception as exc:                     # a starved topology (e.g. theta=90)
            print(f"  {s_title}: template grid fit failed ({exc}); scans not fitted")
            return None

        def s_template(thr, cycles):
            t = snodes.get((float(thr), int(cycles)))
            return t if (t is not None and t.get("ok")) else stp.template_at(thr, cycles)

        out = {}
        blocks = [("threshold", sc_thr, s_scan_thr / 1e3,
                   lambda v: (float(v), base_reset))]
        if sc_rst:
            blocks.append(("reset", sc_rst,
                           reset_rate_khz(np.asarray([r[0] for r in sc_rst], int), ts),
                           lambda v: (base_thr, int(v))))
        if sc_ind:
            blocks.append(("induction", sc_ind,
                           np.asarray([r[0] for r in sc_ind], float),
                           lambda v: (base_thr, base_reset)))
        for knob, scan, disp, pt in blocks:
            items = [(r[0], r[1], r[2], pt(r[0])[0], s_template(*pt(r[0]))) for r in scan]
            n_ev = int(sm.get("n_ind_scan", ns)) if knob == "induction" else ns
            sc, fits = fit_scan(items, n_ev, n_ev, s_qmax, noise_e)
            _add_multiplicity(sc, [(r[0], r[3], r[4], {}) for r in scan], n_ev)
            out[knob] = sc
            plot_scan(None, sc, {"threshold": r"$Q_{\mathrm{thr}}$  ($10^{3}\,e^{-}$)",
                                 "reset": r"Periodic-reset rate  (kHz)",
                                 "induction": r"Induction-response scale"}[knob],
                      f"ti_scan_{knob}{suf}.png", outdir, knob_vals_disp=disp,
                      population=s_pop, title=s_title)
            plot_scan_distributions(
                [(d, f, q, tr) for d, (v, f, q, tr) in zip(disp, fits)],
                f"{s_title}: readout-$Q$ spectra vs {knob}",
                (lambda d: r"$Q_{\mathrm{thr}}=%.1f$" % d) if knob == "threshold" else
                (lambda d: ("reset off" if d == 0 else r"%.0f kHz" % d)) if knob == "reset" else
                (lambda d: r"induction $\times%.1f$" % d),
                knob, f"ti_dist_{knob}{suf}.png", outdir, xlim=xlim,
                thr_sigma=thr_sigma, population=s_pop)
            plot_multiplicity_dist(
                [(d, r[1], r[3], pt(r[0])[0]) for d, r in zip(disp, scan)],
                f"{s_title}: single-hit vs two-hit pixels vs {knob}",
                (lambda d: r"$Q_{\mathrm{thr}}=%.1f$" % d) if knob == "threshold" else
                (lambda d: ("reset off" if d == 0 else r"%.0f kHz" % d)) if knob == "reset" else
                (lambda d: r"induction $\times%.1f$" % d),
                f"ti_multiplicity_{knob}{suf}.png", outdir, n_shower=n_ev,
                xlim=xlim, qmax=s_qmax, population=s_pop)
        rows, knobs_l = [], []
        rows.append(sensitivity_row(out["threshold"], obs_keys_g, return_raw=True))
        knobs_l.append("Threshold")
        if "reset" in out:
            rr = np.asarray([r[0] for r in sc_rst], int)
            rows.append(sensitivity_row(out["reset"], obs_keys_g,
                                        knob_axis=reset_rate_khz(rr, ts), return_raw=True))
            knobs_l.append("Periodic reset")
        if "induction" in out:
            rows.append(sensitivity_row(out["induction"], obs_keys_g, return_raw=True))
            knobs_l.append("Induction")
        plot_sensitivity([r[0] for r in rows], knobs_l, obs_lbl_g, outdir, suffix=suf,
                         title=f"Sensitivity matrix -- {s_title}")
        # Part B: multiplicity-inclusive fired rate (ON vs OFF) -- the non-cancelling observable
        plot_inclusive_rate(s_title, sm, ns, ts, outdir, suf)
        return dict(scans=out, matrix=[r[0] for r in rows], knobs=knobs_l,
                    title=s_title, slug=s_slug)

    sample_results = {}
    _todo = [(n, s) for n, s in samples.items() if n != "shower" and s.get("scan_threshold")]
    if _todo:
        print("\n=== Per-topology scan analysis (same machinery as the shower scans) ===")
        for _nm, _sm in _todo:
            _r = analyze_sample(_nm, _sm)
            if _r is None:
                continue
            sample_results[_nm] = _r
            print(f"  {_r['title']}: fitted {len(_r['knobs'])} scan(s); "
                  f"sensitivity -> ti_sensitivity_matrix_{_r['slug']}.png")
            print("                 " + "  ".join(f"{l:>19}" for l in obs_lbl_g))
            for kn, row in zip(_r["knobs"], _r["matrix"]):
                cells = [f"{v:7.2f}" if np.isfinite(v) else f"{'--':>7}" for v in row]
                print(f"  {kn:>13}  " + "  ".join(f"{c:>19}" for c in cells))

    print("\n=== Threshold scan ===")
    thr_disp = thresholds / 1e3
    thr_items = [(v, q, tr, float(v), template_fn(float(v), base_reset))
                 for (v, q, tr, _a) in spectra["threshold"]]
    thr_scan, thr_fits = fit_scan(thr_items, n_main, n_main, qmax, noise_e)
    _add_multiplicity(thr_scan, spectra.get("threshold_n2"), n_main)
    plot_scan(None, thr_scan, r"Pixel charge threshold $Q_{\mathrm{thr}}$  ($10^{3}\,e^{-}$)",
              "ti_scan_threshold.png", outdir, knob_vals_disp=thr_disp, mpv_ylim=mpv_ylim)
    plot_scan_distributions(dist_panels(thr_fits, thr_disp),
                            r"Readout-$Q$ spectra vs pixel charge threshold",
                            lambda d: r"$Q_{\mathrm{thr}}=%.1f$" % d, r"$Q_{\mathrm{thr}}$",
                            "ti_dist_threshold.png", outdir, xlim=xlim, thr_sigma=thr_sigma)
    plot_multiplicity_dist(mult_panels(spectra["threshold"], spectra.get("threshold_n2"),
                                       thr_disp, lambda v: v),
                           r"Single-hit vs two-hit pixels vs pixel charge threshold",
                           lambda d: r"$Q_{\mathrm{thr}}=%.1f$" % d,
                           "ti_multiplicity_threshold.png", outdir, n_shower=n_main,
                           xlim=xlim, qmax=qmax)

    # Threshold overlays with NO truth split: the whole recorded population at each
    # threshold, first per HIT (single-hit pixels, one ADC sample each) and then per PIXEL
    # (that pixel's samples summed, so two-hit pixels re-enter at their total charge). The
    # pair shows what the single-hit cut removes and where it puts it back.
    def per_pixel_series(knob, items, labeller, thr_of):
        """Per-pixel summed charge for the overlay, preferring the true all-multiplicity
        record and falling back to 1-hit + 2-hit when a file predates it (in which case the
        3+ pixels are simply missing and the tail is a lower limit)."""
        allrec = {float(r[0]): np.asarray(r[1], float) for r in (spectra.get(knob + "_all") or [])}
        n2 = {float(r[0]): np.asarray(r[1], float) for r in (spectra.get(knob + "_n2") or [])}
        exact = bool(allrec)
        out = []
        for lab, (v, q, _tr, _a) in zip(labeller, items):
            v = float(v)
            qq = allrec[v] if exact else np.concatenate(
                [np.asarray(q, float), n2.get(v, np.empty(0))])
            out.append((lab, qq, float(thr_of(v))))
        return out, exact

    _lab = lambda v: r"%.1f ke$^-$" % (v / 1e3)
    _n2 = {float(r[0]): np.asarray(r[1], float) for r in (spectra.get("threshold_n2") or [])}
    plot_charge_overlay(
        [(_lab(v), q, float(v)) for (v, q, _tr, _a) in spectra["threshold"]],
        "Recorded charge per hit, all single-hit pixels (no truth split)",
        "ti_charge_per_hit_vs_threshold.png", outdir, n_shower=n_main, qmax=qmax,
        xlabel=r"hit charge $Q$  ($10^{3}\,e^{-}$)",
        note="dotted lines mark each threshold")
    _ser, _exact = per_pixel_series("threshold", spectra["threshold"],
                                    [_lab(v) for (v, _q, _t, _a) in spectra["threshold"]],
                                    lambda v: v)
    if _ser and (_exact or _n2):
        plot_charge_overlay(
            _ser,
            ("Total recorded charge per pixel, every multiplicity (no truth split)" if _exact
             else "Total recorded charge per pixel, 1- and 2-hit pixels summed (no truth split)"),
            "ti_charge_per_pixel_vs_threshold.png", outdir, n_shower=n_main, qmax=qmax,
            xlabel=r"summed pixel charge $\Sigma Q$  ($10^{3}\,e^{-}$)",
            note=(None if _exact else
                  "pixels with 3+ hits are not recorded, so the high-$Q$ tail is a lower limit"))

    print("\n=== Periodic-reset scan ===")
    reset_rate = reset_rate_khz(resets, ts)
    rst_items = [(v, q, tr, base_thr, template_fn(base_thr, int(v)))
                 for (v, q, tr, _a) in spectra["reset"]]
    rst_scan, rst_fits = fit_scan(rst_items, n_main, n_main, qmax, noise_e)
    _add_multiplicity(rst_scan, spectra.get("reset_n2"), n_main)
    plot_scan(None, rst_scan, r"Periodic-reset rate  (kHz)",
              "ti_scan_periodic_reset.png", outdir, knob_vals_disp=reset_rate)
    plot_scan_distributions(dist_panels(rst_fits, reset_rate),
                            r"Readout-$Q$ spectra vs periodic-reset rate",
                            lambda d: ("reset off" if d == 0 else r"%.0f kHz" % d), r"kHz",
                            "ti_dist_periodic_reset.png", outdir, xlim=xlim, thr_sigma=thr_sigma)
    plot_multiplicity_dist(mult_panels(spectra["reset"], spectra.get("reset_n2"),
                                       reset_rate, lambda v: base_thr),
                           r"Single-hit vs two-hit pixels vs periodic-reset rate",
                           lambda d: ("reset off" if d == 0 else r"%.0f kHz" % d),
                           "ti_multiplicity_reset.png", outdir, n_shower=n_main,
                           xlim=xlim, qmax=qmax)

    # Same no-truth-split overlays as the threshold scan, but against reset rate. Here the
    # threshold is fixed, so any spread between the curves is the reset genuinely reshaping
    # the spectrum rather than the selection moving.
    _rlab = lambda v, d: ("reset off" if d == 0 else r"%.0f kHz" % d)
    _rn2 = {float(r[0]): np.asarray(r[1], float) for r in (spectra.get("reset_n2") or [])}
    plot_charge_overlay(
        [(_rlab(v, d), q, base_thr)
         for d, (v, q, _tr, _a) in zip(reset_rate, spectra["reset"])],
        "Recorded charge per hit, all single-hit pixels (no truth split)",
        "ti_charge_per_hit_vs_reset.png", outdir, n_shower=n_main, qmax=qmax,
        xlabel=r"hit charge $Q$  ($10^{3}\,e^{-}$)", legend_title="reset rate",
        note="dotted line marks the fixed threshold")
    _rser, _rexact = per_pixel_series("reset", spectra["reset"],
                                      [_rlab(v, d) for d, (v, _q, _t, _a)
                                       in zip(reset_rate, spectra["reset"])],
                                      lambda v: base_thr)
    if _rser and (_rexact or _rn2):
        plot_charge_overlay(
            _rser,
            ("Total recorded charge per pixel, every multiplicity (no truth split)" if _rexact
             else "Total recorded charge per pixel, 1- and 2-hit pixels summed (no truth split)"),
            "ti_charge_per_pixel_vs_reset.png", outdir, n_shower=n_main, qmax=qmax,
            xlabel=r"summed pixel charge $\Sigma Q$  ($10^{3}\,e^{-}$)",
            legend_title="reset rate",
            note=(None if _rexact else
                  "pixels with 3+ hits are not recorded, so the high-$Q$ tail is a lower limit"))

    print("\n=== Induction-response scan ===")
    ind_disp = inductions
    ind_items = [(v, q, tr, base_thr, template_fn(base_thr, base_reset))
                 for (v, q, tr, _a) in spectra["induction"]]
    ind_scan, ind_fits = fit_scan(ind_items, n_ind, n_ind, qmax, noise_e)
    _add_multiplicity(ind_scan, spectra.get("induction_n2"), n_ind)
    plot_scan(None, ind_scan, r"Neighbour-pad induction-response scale",
              "ti_scan_induction.png", outdir, knob_vals_disp=ind_disp)
    plot_scan_distributions(dist_panels(ind_fits, ind_disp),
                            r"Readout-$Q$ spectra vs induction-response scale",
                            lambda d: r"induction $\times%.1f$" % d, r"scale",
                            "ti_dist_induction.png", outdir, xlim=xlim, thr_sigma=thr_sigma)

    print("\n=== Hit-multiplicity migration ===")
    if not any(spectra.get(k) for k in ("threshold_n2", "reset_n2", "induction_n2")):
        print("  (no two-hit populations in this spectra file -- re-run produce_spectra "
              "to record them)")
    plot_multiplicity_rates([
        ("Threshold", r"$Q_{\mathrm{thr}}$  ($10^{3}\,e^{-}$)", thr_disp,
         thr_scan["rate"], thr_scan["n2_rate"]),
        ("Periodic reset", "reset rate  (kHz)", reset_rate,
         rst_scan["rate"], rst_scan["n2_rate"]),
        ("Induction", "induction scale", ind_disp,
         ind_scan["rate"], ind_scan["n2_rate"]),
    ], outdir)
    for lbl, sc, xs in (("threshold", thr_scan, thr_disp),
                        ("reset", rst_scan, reset_rate),
                        ("induction", ind_scan, ind_disp)):
        f2 = np.asarray(sc["frac_2hit"], float)
        if not np.isfinite(f2).any():
            continue
        o = np.argsort(np.asarray(xs, float))
        f2o = f2[o]
        good = np.nonzero(np.isfinite(f2o))[0]
        print("  %-10s 2-hit fraction %.3f -> %.3f over the scan  (1-hit/shower %.2f -> %.2f)"
              % (lbl, f2o[good[0]], f2o[good[-1]],
                 np.asarray(sc["rate"], float)[o][good[0]],
                 np.asarray(sc["rate"], float)[o][good[-1]]))

    print("\n=== Sensitivity matrix ===")
    plot_shape_collapse([
        ("Threshold  (should COLLAPSE)",
         [(r"$Q_{thr}=%.0f$" % v, q, float(v)) for (v, q, tr, _a) in spectra["threshold"]]),
        ("Periodic reset",
         [("off" if v <= 0 else "%.0f kHz" % reset_rate_khz(v, ts), q, base_thr)
          for (v, q, tr, _a) in spectra["reset"]]),
        ("Induction response",
         [(r"$\times%.1f$" % v, q, base_thr) for (v, q, tr, _a) in spectra["induction"]]),
    ], outdir)

    # POSITIONS track the threshold; the SCALE-FREE shape ratios (charge measured in units of
    # the peak position) have the threshold divided out, so they isolate reset/induction.
    obs_keys, obs_lbl = obs_keys_g, obs_lbl_g
    rows = [sensitivity_row(thr_scan, obs_keys, return_raw=True),
            sensitivity_row(rst_scan, obs_keys, knob_axis=reset_rate, return_raw=True),
            sensitivity_row(ind_scan, obs_keys, return_raw=True)]
    matrix = [r[0] for r in rows]; raws = [r[1] for r in rows]
    knobs = ["Threshold", "Periodic reset", "Induction"]
    plot_sensitivity(matrix, knobs, obs_lbl, outdir)
    print("  sensitivity |frac obs / frac knob|  (rows=knobs, cols=observables;")
    print("  reset normalised on RATE with 'off' excluded; raw % change in parentheses):")
    print("                 " + "  ".join(f"{l:>19}" for l in obs_lbl))
    for kn, row, raw in zip(knobs, matrix, raws):
        cells = []
        for v, r in zip(row, raw):
            cells.append(f"{v:7.2f} ({r:+6.0f}%)" if np.isfinite(v) else f"{'--':>19}")
        print(f"  {kn:>13}  " + "  ".join(f"{c:>19}" for c in cells))

    closed_form = {k: dict(name=tparam.forms[k]["name"], r2=tparam.forms[k]["r2"],
                           params=tparam.forms[k]["params"])
                   for k in tparam.PARAMS if tparam.forms.get(k)}
    np.savez(f"{outdir}/threshold_induction_results.npz",
             detector_name=np.array(str(m.get("detector_name", ""))),
             detector_path=np.array(str(m.get("detector_path", ""))),
             base_threshold=base_thr, base_reset=base_reset, noise_e=noise_e,
             n_shower_main=n_main, n_shower_ind=n_ind,
             nominal_q=q0, nominal_truth=tr0,
             **{f"thr_{k}": v for k, v in thr_scan.items()},
             **{f"rst_{k}": v for k, v in rst_scan.items()},
             **{f"ind_{k}": v for k, v in ind_scan.items()},
             template_thresholds=thr_grid, template_resets=rst_grid,
             template_rates=reset_rate_khz(rst_grid, ts),
             **{f"template_{k}": grid[k] for k in
                ("mpv_L", "eta_L", "A_L", "mu_G", "sig_G", "chi2ndf",
                 "mpv_L_err", "eta_L_err", "A_L_err", "mu_G_err", "sig_G_err")},
             template_closed_form=np.array(repr(closed_form)),
             sensitivity=np.array(matrix, float),
             sensitivity_knobs=np.array(knobs),
             sensitivity_observables=np.array(obs_lbl))

    # --- Part D: shower-energy dependence of induction ---
    analyze_energy_scan(spectra.get("energy_scan", []), m, outdir)
    # --- Part A: burst-mode sub-threshold induction waveforms (the threshold-independent test) ---
    analyze_burst(spectra.get("burst", []), outdir, noise_e)
    # --- Part C: operating-point recommendation, from everything above ---
    recommend_operating_point(spectra, sample_results, outdir, noise_e)


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    _journal_style()
    try:
        import scipy  # noqa: F401
    except Exception:
        sys.exit("ERROR: scipy is required (pip install scipy).")
    if args.list_configs:
        list_named_configs(); return
    os.makedirs(args.outdir, exist_ok=True)

    if args.refit:
        print(f"REFIT mode: loading spectra from {args.refit} (no GPU) ...")
        spectra = load_spectra(args.refit)
        # burst-mode waveforms live in a sibling ti_burst.npz (separate, like ti_waveforms.npz)
        spectra["burst"] = load_burst(os.path.join(
            os.path.dirname(os.path.abspath(args.refit)) or ".", "ti_burst.npz"))
        mm = spectra["meta"]
        print(f"  base thr={float(mm['base_threshold']):.0f} e-, reset={int(mm['base_reset'])}, "
              f"noise={float(mm['noise_e']):.0f} e-, showers main/ind="
              f"{int(mm['n_shower_main'])}/{int(mm['n_shower_ind'])}")
        if args.noise_e is not None:                  # allow overriding the langaus smear on refit
            mm["noise_e"] = float(args.noise_e)
            print(f"  overriding noise sigma -> {float(args.noise_e):.0f} e-")
    else:
        from numba import cuda
        if not cuda.is_available():
            sys.exit("ERROR: no CUDA GPU available. Run on a GPU node, or use "
                     "--refit <spectra.npz> to re-fit saved spectra without a GPU.")
        resolve_config(args)          # named config -> the four consistent file paths
        ctx = vd.load_simulation(args)
        vd.print_config(ctx)
        # pristine response + induction mask for the induction knob
        ctx._response0 = ctx.response.copy()
        ctx._induction_mask = induction_mask(ctx)
        ctx.induction_scale = 1.0
        spectra = produce_burst_only(ctx, args) if args.burst_only else produce_spectra(ctx, args)
        spec_path = args.spectra_out or f"{args.outdir}/ti_spectra.npz"
        save_spectra(spectra, spec_path)
        save_waveforms(spectra.get("waveforms", []), f"{args.outdir}/ti_waveforms.npz")
        save_burst(spectra.get("burst", []), f"{args.outdir}/ti_burst.npz")
        print(f"Wrote raw spectra to {spec_path}\n  -> iterate fits locally with:  python "
              f"tests/threshold_induction_study.py --refit {spec_path} --outdir {args.outdir}"
              f"{' --burst-only' if args.burst_only else ''}")

    if args.burst_only:
        analyze_burst_only(spectra, args.outdir)
    else:
        analyze_spectra(spectra, args.outdir)
    print(f"\nWrote results + plots to {args.outdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
