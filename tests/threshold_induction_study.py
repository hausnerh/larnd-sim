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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    """Set ctx.response to a copy with neighbour-pad bins scaled by `scale`.

    scale=1 -> nominal; 0 -> no neighbour induction (collection only); >1 stronger.
    Uses the array module of the pristine response (cupy on a GPU node).
    """
    xp = vd.array_module(ctx) if hasattr(vd, "array_module") else _xp_of(ctx._response0)
    resp = ctx._response0.copy()
    if scale != 1.0:
        m = xp.asarray(ctx._induction_mask)
        resp[m] = resp[m] * scale
    ctx.response = resp
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
    (shower) vs pure-induction hits -- no arbitrary net-charge cut."""
    import cupy as cp
    net_sp = cp.asnumpy(signals.sum(axis=2))          # (nseg, max_neigh) net per (seg,pad)
    neigh_h = cp.asnumpy(neigh)
    seg_peak = np.maximum(net_sp.max(axis=1), 0.0)    # per-segment peak collected net
    landed = (neigh_h >= 0) & (net_sp > 0) & (net_sp > f_collect * seg_peak[:, None])
    return np.unique(neigh_h[landed])


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
    collect_ids = collection_pixels_from_signals(signals, neigh, collect_frac)
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
        truth_e=ps_host.sum(axis=1, dtype=np.float64) * ts,   # net collected charge (e-)
        max_time=ps_host.shape[1] * ts,
    )


def event_presignals(ctx, tracks, seed, collect_frac=0.15):
    """Thin wrapper: event_drift then event_induce (for callers that don't reuse drift)."""
    d = event_drift(ctx, tracks)
    if d is None:
        return None
    return event_induce(ctx, d[0], d[1], d[2], seed, collect_frac)


def event_fee_singlehits(ctx, ev, threshold_e, seed):
    """Run only the FEE on a cached pre-signal; return single-hit (Q, truth_e).

    A pixel is a SINGLE-HIT pixel iff it produced exactly one ADC sample. Its
    recorded charge Q is recovered from the ADC the way a data analysis would
    (vd.adc_to_charge, carrying the ~1-LSB quantization). Returns the measured Q
    (electrons) and the per-pixel truth collection flag for each single-hit pixel.
    """
    import cupy as cp
    ctx.detector.DISCRIMINATION_THRESHOLD = float(threshold_e)
    adc, _ticks = vd.run_fee(
        ctx,
        cp.asarray(ev["pixels_signals"]),
        cp.asarray(ev["pixels_tracks_signals"]),
        cp.asarray(ev["num_backtrack"]),
        cp.asarray(ev["offset_backtrack"]),
        ev["max_time"], seed=seed)
    adc = np.asarray(adc)                            # (npix, MAX_ADC_VALUES)
    n_hits = (adc > 0).sum(axis=1)
    sel = n_hits == 1
    if not sel.any():
        return np.empty(0), np.empty(0, bool)
    single_adc = adc[sel].max(axis=1)               # the one nonzero sample per row
    q = vd.adc_to_charge(ctx, single_adc)           # electrons
    return np.asarray(q, float), ev["is_collection"][sel]


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
    """Seed (mpv,eta,A_L, mu,sig,A_G) for the induction-off shower model
    (bare Landau near threshold + Gaussian bump)."""
    w = centres[1] - centres[0]
    T = threshold_e
    total = float(counts.sum() * w)
    mpv = float(centres[int(np.argmax(counts))])
    mpv = min(max(mpv, 0.95 * T), 2.5 * T)
    hi = centres > 1.6 * mpv                          # upper shoulder -> Gaussian bump
    if hi.any() and counts[hi].sum() > 0:
        mu = float(np.average(centres[hi], weights=counts[hi]))
        sig = max(float(np.sqrt(np.average((centres[hi] - mu) ** 2, weights=counts[hi]))), w)
        aG = float(counts[hi].sum() * w)
    else:
        mu, sig, aG = 2.5 * mpv, mpv, 0.3 * total
    return [mpv, 0.15 * T, 0.6 * total, mu, sig, aG]


def fit_shower_template(q_shower, threshold_e, qmax=None):
    """Fit the induction-OFF shower single-hit spectrum with a bare LANDAU (near-threshold
    peak) + a Gaussian (high-Q bump) and return the FROZEN Landau as a reusable template.

    The near-threshold Landau is the induction-immune shape (peripheral collection pads
    whose weak neighbours induce little on them); it is what we reuse in the induction-ON
    composite fit, and its AMPLITUDE is the number the induction extraction hinges on. We
    use a single Landau (no Gaussian convolution) precisely so that amplitude is well
    determined: the electronics-noise smear is sub-bin here, so a convolved langaus only
    adds an eta<->sigma degeneracy that makes the area scatter. The Gaussian (higher-Q,
    core-adjacent, induction-shadowed) soaks up the upper shoulder so the Landau lands
    correctly -- it is NOT reused (it floats in the composite fit). Returns a 'template'
    dict with the Landau params + fit arrays for diagnostics."""
    from scipy.optimize import curve_fit
    h = _hist(q_shower, threshold_e, qmax=qmax)
    if h is None:
        return dict(ok=False, reason="too few shower hits", threshold_e=threshold_e)
    centres, counts, width, edges = h
    lg = Langaus(centres[0], centres[-1])
    def model(x, mpv, eta, aL, mu, sig, aG):
        return lg.landau(x, mpv, eta, aL) + _gaussian(x, mu, sig, aG)
    T, span, cmax = threshold_e, centres[-1] - centres[0], centres[-1]
    p0 = _guess_shower(centres, counts, threshold_e)
    # One width (eta) for the Landau -> well-conditioned amplitude. Keep the Gaussian
    # centred clearly above the Landau peak so it soaks up the bump, not the turn-on.
    lb = [0.85 * T, 0.02 * T, 0.0,    1.6 * T, 0.05 * T, 0.0]
    ub = [3.0 * T,  0.60 * T, np.inf, cmax,    span,     np.inf]
    p0 = [min(max(v, lb[i] + 1e-9), ub[i] - 1e-9) for i, v in enumerate(p0)]
    sigma = np.sqrt(counts) + 1.0
    try:
        popt, _ = curve_fit(model, centres, counts, p0=p0, bounds=(lb, ub),
                            sigma=sigma, absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        return dict(ok=False, reason=str(exc), threshold_e=threshold_e,
                    centres=centres, counts=counts, width=width, edges=edges)
    mpv, eta, aL, mu, sig, aG = popt
    pred = model(centres, *popt)
    ndf = max(len(centres) - len(popt), 1)
    chi2 = float(np.sum(((counts - pred) / sigma) ** 2))
    return dict(ok=True, threshold_e=float(threshold_e),
                mpv_L=float(mpv), eta_L=float(eta), A_L=float(aL),
                mu_G=float(mu), sig_G=float(sig), A_G=float(aG),
                popt=popt, model=model, lg=lg, centres=centres, counts=counts,
                width=width, edges=edges, pred=pred, chi2=chi2, ndf=ndf,
                chi2ndf=chi2 / ndf, n_shower=int(counts.sum()))


def fit_shower_plus_induction(q, threshold_e, template, qmax=None):
    """Composite induction-ON fit: FROZEN shower Landau (from `template`) + a floating
    Gaussian (induction-shadowed high-Q shower bump) + a floating induction Landau.

    Freezing the shower Landau -- crucially its AMPLITUDE -- removes the near-threshold
    degeneracy that makes a free two-peak fit mis-assign the overlapping populations: the
    shower contribution at the turn-on is pinned by the induction-off template, so the
    residual excess right above threshold is cleanly attributed to induction. Free params
    [mu, sig, A_G, mpv_i, eta_i, A_i]. Returns a `kind='template'` fit dict."""
    from scipy.optimize import curve_fit
    h = _hist(q, threshold_e, qmax=qmax)
    if h is None:
        return None
    centres, counts, width, edges = h
    if not (template and template.get("ok")):
        return dict(ok=False, reason="no shower template", centres=centres, counts=counts,
                    width=width, edges=edges, threshold_e=threshold_e)
    lg = Langaus(centres[0], centres[-1])
    mpv_L, eta_L, A_L = template["mpv_L"], template["eta_L"], template["A_L"]
    frozen_c = lg.landau(centres, mpv_L, eta_L, A_L)   # frozen shower Landau on the bins
    T, span, cmax = threshold_e, centres[-1] - centres[0], centres[-1]

    def model(x, mu, sig, aG, mpv_i, eta_i, aI):
        return frozen_c + _gaussian(x, mu, sig, aG) + lg.landau(x, mpv_i, eta_i, aI)

    total = float(counts.sum() * width)
    p0 = [min(max(template["mu_G"], 1.6 * T), cmax), max(template["sig_G"], width), 0.4 * total,
          1.1 * T, 0.15 * T, 0.4 * total]
    # Keep the induction Landau a NARROW turn-on spike near threshold (eta_i capped) so it
    # cannot broaden to absorb the frozen shower -- the separation is meant to come from
    # the shower being fixed, not from the induction component fattening.
    lb = [1.6 * T, 0.05 * T, 0.0,    0.9 * T, 0.02 * T, 0.0]
    ub = [cmax,    span,     np.inf, 2.0 * T, 0.30 * T, np.inf]
    p0 = [min(max(v, lb[i] + 1e-9), ub[i] - 1e-9) for i, v in enumerate(p0)]
    sigma = np.sqrt(counts) + 1.0
    try:
        popt, pcov = curve_fit(model, centres, counts, p0=p0, bounds=(lb, ub),
                               sigma=sigma, absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        return dict(ok=False, reason=str(exc), centres=centres, counts=counts,
                    width=width, edges=edges, threshold_e=threshold_e)
    mu, sig, aG, mpv_i, eta_i, aI = popt
    pred = model(centres, *popt)
    ndf = max(len(centres) - len(popt), 1)
    chi2 = float(np.sum(((counts - pred) / sigma) ** 2))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    tot_shw = A_L + aG
    f_ind = float(aI / (aI + tot_shw)) if (aI + tot_shw) > 0 else np.nan
    peak_mpv, peak_height, tail_height = peak_tail_observables(centres, counts, threshold_e)
    return dict(
        ok=True, kind="template", centres=centres, counts=counts, width=width,
        edges=edges, threshold_e=threshold_e, pred=pred, lg=lg,
        frozen=(mpv_L, eta_L, A_L), popt=popt, perr=perr,
        mpv_ind=float(mpv_i), mpv_ind_err=float(perr[3]),
        mpv_shw=float(mpv_L), mpv_shw_err=0.0,        # frozen -> no fit error
        mu_G=float(mu), sig_G=float(sig), A_G=float(aG),
        eta_ind=float(eta_i),
        area_ind=float(aI), area_shw=float(tot_shw), frac_ind=f_ind,
        peak_mpv=peak_mpv, peak_height=peak_height, tail_height=tail_height,
        chi2=chi2, ndf=ndf, chi2ndf=chi2 / ndf, n_single=int(np.sum(counts)))


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


def aggregate_singlehits(ctx, evs, threshold_e, reset_cycles, induction_scale,
                         seed0, n_events=None):
    """Run the FEE on cached pre-signals and concatenate single-hit (Q, truth).

    `evs` is the list of cached nominal-induction pre-signals (used for the
    threshold and reset scans). For the induction scan, callers pass freshly
    re-induced pre-signals in `evs` (the pre-FEE current depends on induction).
    """
    set_periodic_reset(ctx, reset_cycles)
    qs, tr = [], []
    with _silence_device_stdout():
        for i, ev in enumerate(evs):
            q, t = event_fee_singlehits(ctx, ev, threshold_e, seed=seed0 + i)
            if q.size:
                qs.append(q); tr.append(t)
    if not qs:
        return np.empty(0), np.empty(0, bool)
    return np.concatenate(qs), np.concatenate(tr)


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
    """Three DATA-BLIND handles read straight off the recorded-Q spectrum (no truth,
    no fit): the low-Q peak position (MPV), its height, and the high-Q tail height.
    Per the disentanglement idea: MPV tracks THRESHOLD; peak height responds to
    INDUCTION (and reset); tail height responds to periodic RESET (and induction)."""
    counts = np.asarray(counts, float)
    if counts.sum() <= 0:
        return np.nan, np.nan, np.nan
    i = int(np.argmax(counts))
    peak_mpv = float(centres[i])
    peak_height = float(counts[i])
    tail = (centres > 2.5 * threshold_e) & (counts > 0)
    tail_height = float(np.median(counts[tail])) if tail.any() else 0.0
    return peak_mpv, peak_height, tail_height


# ===========================================================================
# Shower-template library: fit the induction-OFF langaus over (threshold x reset)
# and look for a closed-form law for each langaus parameter
# ===========================================================================
def reset_rate_khz(cycles, ts):
    """Periodic-reset RATE (kHz) from PERIODIC_RESET_CYCLES; <=0 -> 0 (reset off)."""
    cycles = np.asarray(cycles, float)
    return np.where(cycles > 0, 1.0e3 / (cycles * ts), 0.0)


def build_template_grid(ctx, evs_off, thresholds, resets, ts, qmax, seed0):
    """Fit the shower template on the induction-OFF cache over the full (threshold x
    reset) grid. Cheap: NO re-induction -- only FEE re-runs on the cached pre-signals
    (reset changes recompile the fee kernel, so loop reset OUTER to share each recompile
    across all thresholds). Returns (grid, node_fits): `grid` holds each langaus param as
    a (n_reset, n_threshold) array plus the axes; `node_fits[(thr, cycles)]` is the
    per-node template dict (the diagnostics + exact on-grid template for the fits)."""
    thr_arr = np.asarray(thresholds, float)
    rst_arr = np.asarray(resets, int)
    keys = ("mpv_L", "eta_L", "A_L", "mu_G", "sig_G", "chi2ndf")
    grid = {k: np.full((len(rst_arr), len(thr_arr)), np.nan) for k in keys}
    node_fits = {}
    for ir, rst in enumerate(rst_arr):                # reset OUTER -> one recompile per rate
        for it, thr in enumerate(thr_arr):
            q, tr = aggregate_singlehits(ctx, evs_off, float(thr), int(rst), 0.0,
                                         seed0=seed0 + ir * 1000 + it)
            coll = np.asarray(tr, bool)
            qs = q[coll] if coll.any() else q         # induction-off => essentially all shower
            tmpl = fit_shower_template(qs, float(thr), qmax=qmax)
            node_fits[(float(thr), int(rst))] = tmpl
            if tmpl.get("ok"):
                for k in keys:
                    grid[k][ir, it] = tmpl[k]
    grid["thresholds"] = thr_arr
    grid["resets"] = rst_arr
    grid["rates"] = reset_rate_khz(rst_arr, ts)
    return grid, node_fits


def _cf_forms():
    """Candidate closed forms p(x, r) for a langaus parameter; x = Q_thr/1e3 (10^3 e-),
    r = reset rate (kHz). Physically: p rides threshold (affine in x) and is sculpted by
    reset (a linear/bilinear rate term); A_L instead decays with threshold (exp)."""
    def planar(X, c0, c1, c2):
        x, r = X; return c0 + c1 * x + c2 * r
    def bilinear(X, c0, c1, c2, c3):
        x, r = X; return c0 + c1 * x + c2 * r + c3 * x * r
    def affine_r(X, c0, c1, c2):
        x, r = X; return (c0 + c1 * x) * (1.0 + c2 * r)
    def expdecay_r(X, c0, c1, c2):
        x, r = X; return c0 * np.exp(-x / (abs(c1) + 1e-6)) * (1.0 + c2 * r)
    return [
        (r"$c_0+c_1 Q_{thr}+c_2 f$", planar,
         lambda y, x, r: [float(np.mean(y)), 0.0, 0.0]),
        (r"$c_0+c_1 Q_{thr}+c_2 f+c_3 Q_{thr}f$", bilinear,
         lambda y, x, r: [float(np.mean(y)), 0.0, 0.0, 0.0]),
        (r"$(c_0+c_1 Q_{thr})(1+c_2 f)$", affine_r,
         lambda y, x, r: [float(np.mean(y)), 0.0, 0.0]),
        (r"$c_0 e^{-Q_{thr}/c_1}(1+c_2 f)$", expdecay_r,
         lambda y, x, r: [float(np.max(y)) if y.size else 1.0, float(np.mean(x)) or 1.0, 0.0]),
    ]


def _fit_closed_form(x, r, y):
    """Fit the candidate closed forms to param values y over (x=Q_thr/1e3, r=rate kHz);
    return the lowest-AIC form as a dict (name, params, r2, aic, func) or None."""
    from scipy.optimize import curve_fit
    good = np.isfinite(y) & np.isfinite(x) & np.isfinite(r)
    x, r, y = x[good], r[good], y[good]
    if y.size < 3:
        return None
    X = np.vstack([x, r])
    sst = float(np.sum((y - y.mean()) ** 2))
    best = None
    for name, func, seed in _cf_forms():
        k = func.__code__.co_argcount - 1
        if y.size <= k:
            continue
        try:
            popt, _ = curve_fit(func, X, y, p0=seed(y, x, r), maxfev=20000)
        except Exception:
            continue
        ssr = float(np.sum((y - func(X, *popt)) ** 2))
        aic = y.size * np.log(ssr / y.size + 1e-30) + 2 * k
        r2 = 1.0 - ssr / sst if sst > 0 else (1.0 if ssr < 1e-9 else 0.0)
        cand = dict(name=name, params=[float(p) for p in popt], aic=float(aic),
                    r2=float(r2), func=func, k=k)
        if best is None or aic < best["aic"]:
            best = cand
    return best


class TemplateParam:
    """Closed-form (with grid-interpolation fallback) model of the shower langaus
    parameters as functions of (Q_thr, reset rate), fitted from a build_template_grid
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
        self.x = self.thr / 1e3
        xx, rr = np.meshgrid(self.x, self.rates)      # (n_rate, n_thr), aligns with grid[k]
        # Drop nodes whose template fit was poor (chi2/ndf > chi2_cut) before fitting the
        # closed form -- a single bad node otherwise drags a clean law's R^2 right down.
        chi2 = np.asarray(grid.get("chi2ndf", np.zeros_like(self.grid["mpv_L"])), float)[order]
        node_ok = np.isfinite(chi2) & (chi2 <= chi2_cut)
        self.forms = {}
        for k in self.PARAMS:
            yv = self.grid[k].ravel()
            m = node_ok.ravel() & np.isfinite(yv)
            self.forms[k] = _fit_closed_form(xx.ravel()[m], rr.ravel()[m], yv[m])

    def _interp(self, key, thr, rate):
        """Bilinear interp of grid[key] over (rate, threshold), clamped at the edges."""
        g = self.grid[key]
        xi = float(np.interp(thr / 1e3, self.x, np.arange(self.x.size)))
        yi = float(np.interp(rate, self.rates, np.arange(self.rates.size)))
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = min(x0 + 1, self.x.size - 1), min(y0 + 1, self.rates.size - 1)
        fx, fy = xi - x0, yi - y0
        return float((1 - fx) * (1 - fy) * g[y0, x0] + fx * (1 - fy) * g[y0, x1]
                     + (1 - fx) * fy * g[y1, x0] + fx * fy * g[y1, x1])

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


def _draw_charge_dist(ax, q, truth, fit, threshold_e, xlim=None,
                      compact=False, thr_sigma=0.0, logy=True):
    """Draw one FEE-readout single-hit Q spectrum (the noisy Q you trigger on in
    data), bins coloured by the per-pixel backtrack truth into induction vs shower
    (collection) hits (see categorize()), the composite template fit overlaid (frozen
    shower langaus + floating Gaussian as one shower curve, plus the induction langau),
    and the pixel-charge-threshold line with a +/-sigma band: the discriminator fires on
    `q+noise >= threshold + disc_noise`, so the threshold is Gaussian-smeared by
    sigma_disc -- which is why recorded charges land below the nominal line."""
    edges = fit["edges"] / 1e3
    is_ind, is_shw = categorize(q, truth)
    ax.hist([q[is_ind] / 1e3, q[is_shw] / 1e3], bins=edges,
            stacked=True, color=[_C_IND, _C_SHW], alpha=0.6, edgecolor="white",
            linewidth=(0.2 if compact else 0.35),
            label=["Induction", "Shower"])
    if fit.get("ok") and fit.get("kind") == "template":
        xs = np.linspace(fit["edges"][0], fit["edges"][-1], 700)
        lg = fit["lg"]
        mpv_L, eta_L, A_L = fit["frozen"]
        mu, sig, aG, mpv_i, eta_i, aI = fit["popt"]
        shower = lg.landau(xs, mpv_L, eta_L, A_L) + _gaussian(xs, mu, sig, aG)
        induction = lg.landau(xs, mpv_i, eta_i, aI)
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
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.6)
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0.0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif fit.get("edges") is not None:
        ax.set_xlim(fit["edges"][0] / 1e3, fit["edges"][-1] / 1e3)


def plot_headline(ctx, q, truth, fit, outdir, xlim=None, thr_sigma=0.0):
    """Nominal single-hit Q spectrum (the noisy Q triggered on, induction/shower
    coloured) + composite template fit (frozen shower langaus + floating Gaussian +
    induction langau) + threshold-with-noise band. Saved both log-y (ti_hist_nominal.png,
    the tail) and linear-y (ti_hist_nominal_lin.png, the peak region)."""
    import matplotlib.pyplot as plt
    for logy, suf in ((True, ""), (False, "_lin")):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        _draw_charge_dist(ax, q, truth, fit, fit["threshold_e"], xlim=xlim,
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
        ax.set_ylabel("Single-hit pixels")
        ax.legend(loc="upper center", fontsize=9.5, handlelength=1.9)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/ti_hist_nominal{suf}.png")


def plot_scan_distributions(panels, title, fmt_val, col0_header, fname, outdir,
                            xlim=None, thr_sigma=0.0):
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
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (d, f, q, tr) in zip(axes.flat, items):
            _draw_charge_dist(ax, q, tr, f, f["threshold_e"], xlim=xlim,
                              compact=True, thr_sigma=thr_sigma, logy=logy)
            is_ind, is_shw = categorize(q, tr)
            stat = "\n".join((
                fmt_val(d),
                r"$\mathrm{MPV}_{i,s}=%.1f,\,%.1f$" % (f["mpv_ind"] / 1e3, f["mpv_shw"] / 1e3),
                r"$N_{i,s}=%d,\,%d$" % (int(is_ind.sum()), int(is_shw.sum()))))
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
                axes[r, 0].set_ylabel("Single-hit pixels")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.06, fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/{base}{suf}.png")


def plot_scan(ctx, scan, knob_label, fname, outdir, knob_vals_disp=None, mpv_ylim=None):
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
    axes[1].set_ylabel("Peak height  (pixels/bin)")
    axes[2].errorbar(x, col("tail_height"), fmt="^", color=_C_PUR, **ekw)
    axes[2].set_ylabel("Tail height  (pixels/bin)")
    axes[2].set_xlabel(knob_label)
    for k, axp in enumerate(axes):
        axp.text(0.018, 0.93, "(%s)" % chr(97 + k), transform=axp.transAxes,
                 fontsize=11, va="top", ha="left")
    fig.align_ylabels(axes)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/{fname}")


def plot_sensitivity(matrix, knobs, observables, outdir):
    """The money plot: |fractional response| of each observable to each knob.

    For each knob scan and each observable we compute the end-to-end fractional
    change normalised by the knob's fractional change (a dimensionless
    sensitivity). A near-DIAGONAL pattern -- each knob lighting up a different
    observable -- is the quantitative statement that the three effects are
    separable from the single-hit Q spectrum.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
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
    ax.set_title("Sensitivity of each fit observable to each readout knob",
                 fontsize=11, pad=8)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_sensitivity_matrix.png")


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
            ax.plot(xs / 1e3, f["lg"].landau(xs, mpv, eta, aL), color=_C_SHW, ls="--", lw=1.2)
            ax.plot(xs / 1e3, _gaussian(xs, mu, sig, aG), color=_C_GRN, ls="--", lw=1.2)
            ax.plot(xs / 1e3, f["model"](xs, *f["popt"]), color=_C_SUM, ls="-", lw=1.6,
                    label="Landau + Gaussian")
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
                axes[r, 0].set_ylabel("Single-hit pixels")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(r"Induction-off shower template (Landau + Gaussian) vs threshold",
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
        f = tparam.forms.get(key)
        for ir in order:
            c = cmap(0.12 + 0.76 * (ir / max(len(rates) - 1, 1)))
            lab = "off" if rates[ir] == 0 else "%.0f kHz" % rates[ir]
            ax.plot(thr / 1e3, g[ir] / sc, "o", color=c, ms=5, label=lab)
            if f is not None:
                xs = np.linspace(thr.min(), thr.max(), 60)
                yhat = f["func"](np.vstack([xs / 1e3, np.full_like(xs, rates[ir])]), *f["params"])
                ax.plot(xs / 1e3, yhat / sc, "-", color=c, lw=1.2, alpha=0.85)
        ax.set_xlabel(r"$Q_{\mathrm{thr}}$ ($10^{3}\,e^{-}$)")
        ax.set_ylabel(ylab)
        if f is not None:
            ax.text(0.03, 0.97, "%s\n$R^{2}=%.3f$" % (f["name"], f["r2"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    axes.flat[0].legend(title="reset rate", fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("Frozen shower-Landau parameters vs threshold and reset "
                 "(points) with closed-form fits (lines)", y=1.0, fontsize=12)
    fig.tight_layout()
    _savefig(fig, f"{outdir}/ti_template_params.png")


# ===========================================================================
# Scans
# ===========================================================================
# fit-dict keys copied into every scan's output arrays (composite template fit)
_SCAN_FIT_KEYS = ("mpv_ind", "mpv_ind_err", "mpv_shw", "mpv_shw_err", "frac_ind",
                  "chi2ndf", "n_single", "peak_mpv", "peak_height", "tail_height",
                  "area_ind", "mu_G", "sig_G", "A_G")


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


def run_fixed_induction_scan(ctx, evs, knob_name, knob_vals, base_threshold,
                             base_reset, n_events, seed0, set_fn, template_fn, qmax=None):
    """Threshold or reset scan: reuse cached nominal-induction pre-signals.

    `set_fn(value)` returns (threshold_e, reset_cycles) for that knob value (the other
    is held at base); `template_fn(threshold_e, reset_cycles)` returns the frozen shower
    template for that operating point. Each aggregated spectrum is fit with the composite
    frozen-shower + floating-induction model."""
    out = {k: [] for k in ("knob", "rate") + _SCAN_FIT_KEYS}
    fits = []
    for j, val in enumerate(knob_vals):
        thr, rst = set_fn(val)
        q, tr = aggregate_singlehits(ctx, evs, thr, rst, 1.0, seed0=seed0 + j * 1000)
        fit = fit_shower_plus_induction(q, thr, template_fn(thr, rst), qmax=qmax)
        fits.append((val, fit, q, tr))
        out["knob"].append(val)
        _record_fit(out, fit, n_events)
    return {k: np.asarray(v, float) for k, v in out.items()}, fits


def run_induction_scan(ctx, drift_cache, scales, base_threshold, base_reset,
                       n_events, seed0, template_fn, evs_off=None, qmax=None,
                       collect_frac=0.15):
    """Induction scan: re-INDUCE (only) the pre-FEE current at each response scale, on the
    cached drift/pixel geometry (`drift_cache`, which is induction-scale-independent).
    `evs_off` (the induction-off pre-signals) is reused verbatim for the scale=0 point.
    Threshold and reset stay at base, so one base template is used throughout. Caching
    drift keeps quench/drift/find-pixels out of the per-scale loop (the timeout fix)."""
    set_periodic_reset(ctx, base_reset)
    out = {k: [] for k in ("knob", "rate") + _SCAN_FIT_KEYS}
    fits = []
    template = template_fn(base_threshold, base_reset)
    for j, s in enumerate(scales):
        if s == 0.0 and evs_off is not None:
            evs = evs_off                              # induction-off cache == scale-0
        else:
            set_induction(ctx, s)
            evs = []
            for i, dc in enumerate(drift_cache):
                ev = event_induce(ctx, dc[0], dc[1], dc[2], seed=seed0 + j * 777 + i,
                                  collect_frac=collect_frac)
                if ev is not None:
                    evs.append(ev)
        q, tr = aggregate_singlehits(ctx, evs, base_threshold, base_reset, s,
                                     seed0=seed0 + j * 1000 + 50000)
        fit = fit_shower_plus_induction(q, base_threshold, template, qmax=qmax)
        fits.append((s, fit, q, tr))
        out["knob"].append(s)
        _record_fit(out, fit, n_events)
    set_induction(ctx, 1.0)
    return {k: np.asarray(v, float) for k, v in out.items()}, fits


def sensitivity_row(scan, observables_keys):
    """End-to-end |frac. obs. change / frac. knob change| for one scan."""
    knob = scan["knob"]
    kvals = knob.copy()
    # map "off" reset (-1) to a large finite period so the fractional change is defined
    kvals = np.where(kvals < 0, np.nan, kvals)
    finite = np.isfinite(kvals)
    if finite.sum() >= 2:
        k0, k1 = np.nanmin(kvals[finite]), np.nanmax(kvals[finite])
    else:
        k0, k1 = knob[0], knob[-1]
    dk = abs(k1 - k0) / (abs(0.5 * (k1 + k0)) + 1e-9)
    row = []
    for key in observables_keys:
        y = scan[key]
        good = np.isfinite(y)
        if good.sum() < 2:
            row.append(np.nan); continue
        y0, y1 = y[good][0], y[good][-1]
        dy = abs(y1 - y0) / (abs(0.5 * (y1 + y0)) + 1e-9)
        row.append(dy / dk if dk > 0 else np.nan)
    return row


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
    ap.add_argument("--edep-h5", default=None,
                    help="dumpTree.py edep-sim HDF5; use its real 'segments' per "
                         "event instead of the parametric shower generator")
    ap.add_argument("--n-events", type=int, default=30, help="shower events per config")
    ap.add_argument("--shower-energy", type=float, default=300.0, help="MeV")
    ap.add_argument("--n-dep", type=int, default=400, help="deposits per shower")
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
    ap.add_argument("--outdir", default="tistudy")
    return ap.parse_args()


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


def base_config(ctx, args):
    """Nominal operating point = the loaded config's own threshold + reset values."""
    base_thr = float(np.atleast_1d(ctx.detector.DISCRIMINATION_THRESHOLD).ravel()[0])
    base_reset = int(getattr(ctx.detector, "PERIODIC_RESET_CYCLES", -1))
    return base_thr, base_reset


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    _journal_style()
    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available. Run this on a GPU node.")
    try:
        import scipy  # noqa: F401
    except Exception:
        sys.exit("ERROR: scipy is required (pip install scipy).")

    os.makedirs(args.outdir, exist_ok=True)
    ctx = vd.load_simulation(args)
    vd.print_config(ctx)
    # pristine response + induction mask for the induction knob
    ctx._response0 = ctx.response.copy()
    ctx._induction_mask = induction_mask(ctx)
    ctx.induction_scale = 1.0
    # discriminator-noise sigma -> the +/-sigma band drawn around the threshold line
    thr_sigma = float(np.atleast_1d(ctx.detector.DISCRIMINATOR_NOISE).ravel()[0])

    base_thr, base_reset = base_config(ctx, args)
    reset_state = "off" if base_reset <= 0 else f"{base_reset} cycles"
    print(f"\nBase operating point: threshold={base_thr:.0f} e-, "
          f"periodic_reset={reset_state}, induction=1.0")

    rng = np.random.default_rng(args.seed)
    if args.edep_h5:
        print(f"\nLoading real edep-sim showers from {args.edep_h5} ...")
        raw_events = load_edep_events(args.edep_h5, args.n_events)
        print(f"  loaded {len(raw_events)} events "
              f"({sum(len(e) for e in raw_events)} segments total)")
    else:
        print(f"\nGenerating {args.n_events} parametric shower events "
              f"(E={args.shower_energy:.0f} MeV, {args.n_dep} deposits each)...")
        raw_events = [build_shower_event(ctx, rng, energy_mev=args.shower_energy,
                                         n_dep=args.n_dep)
                      for _ in range(args.n_events)]

    ts = ctx.detector.TIME_SAMPLING

    # --- drift + pixel geometry, computed ONCE per event. Induction-scale-independent,
    #     so it is reused for the nominal pass, the induction-off template pass, and
    #     every induction-scan scale (this is the induction-scan timeout fix).
    print("Computing drift + pixel geometry (cached, once)...")
    drift_cache = []
    for raw in raw_events:
        dc = event_drift(ctx, raw)
        if dc is not None:
            drift_cache.append(dc)
    n_eff = len(drift_cache)
    print(f"  {n_eff}/{args.n_events} events produced collectable charge")
    if n_eff == 0:
        sys.exit("No events produced any pixels -- check shower placement / config.")

    # --- nominal-induction pre-signals (reused by the threshold + reset scans)
    print("Inducing nominal-induction pre-FEE signals (cached)...")
    set_induction(ctx, 1.0)
    set_periodic_reset(ctx, base_reset)
    evs = [event_induce(ctx, d[0], d[1], d[2], seed=args.seed + 100 + i,
                        collect_frac=args.collect_frac) for i, d in enumerate(drift_cache)]
    evs = [e for e in evs if e is not None]

    # --- nominal spectrum -> the common charge range (qmax) shared by every fit/plot.
    #     Bin/fit out to the actual high-Q extent (capped at the 50k display edge) so the
    #     spectrum isn't truncated -- the natural fall-off is the single-hit ceiling.
    print("\n=== Nominal single-hit spectrum ===")
    q0, tr0 = aggregate_singlehits(ctx, evs, base_thr, base_reset, 1.0, seed0=args.seed + 1)
    qmax = float(min(np.max(q0[q0 > 0]), 50000.0)) if np.any(q0 > 0) else 30000.0
    xlim = (0.0, 50.0)
    mpv_ylim = (0.0, 50.0)

    # --- induction-OFF pre-signals: a PURE-shower single-hit sample for the templates.
    print("Inducing induction-OFF pre-FEE signals (pure shower, for templates)...")
    set_induction(ctx, 0.0)
    evs_off = [event_induce(ctx, d[0], d[1], d[2], seed=args.seed + 200 + i,
                            collect_frac=args.collect_frac) for i, d in enumerate(drift_cache)]
    evs_off = [e for e in evs_off if e is not None]

    # --- shower-template library over the FULL (threshold x reset) grid + closed forms
    print("\n=== Shower-template grid (induction-off) + closed-form parametrization ===")
    thr_grid = sorted(set(float(t) for t in args.thresholds) | {base_thr})
    rst_grid = sorted(set(int(r) for r in args.resets) | {int(base_reset)})
    grid, node_fits = build_template_grid(ctx, evs_off, thr_grid, rst_grid, ts, qmax,
                                          seed0=args.seed + 6000)
    tparam = TemplateParam(grid, ts)
    for k in ("mpv_L", "A_L", "eta_L", "mu_G"):
        f = tparam.forms.get(k)
        if f:
            print(f"  {k:>6}: R2={f['r2']:.3f}  best closed form  {f['name']}")
    plot_shower_templates(node_fits, thr_grid, int(base_reset), args.outdir)
    plot_template_params(grid, tparam, ts, args.outdir)

    # restore the nominal operating point for the induction-ON passes
    set_induction(ctx, 1.0)
    set_periodic_reset(ctx, base_reset)

    def template_fn(thr, cycles):
        """Frozen shower template at (thr, reset cycles): the exact induction-off grid
        node when available (all scan points sit on the grid), else the parametrized
        closed-form / interpolated template for off-grid points."""
        t = node_fits.get((float(thr), int(cycles)))
        if t is not None and t.get("ok"):
            return t
        return tparam.template_at(thr, cycles)

    # --- headline: nominal composite fit (frozen shower langaus + Gaussian + induction)
    print("\n=== Nominal composite template fit ===")
    fit0 = fit_shower_plus_induction(q0, base_thr, template_fn(base_thr, base_reset), qmax=qmax)
    if fit0 and fit0.get("ok"):
        coll0 = np.asarray(tr0, bool)
        print(f"  single-hit pixels: {fit0['n_single']}  Ind.MPV={fit0['mpv_ind']:.0f} e-  "
              f"Shower.MPV_L={fit0['mpv_shw']:.0f} e-  mu_G={fit0['mu_G']:.0f} e-  "
              f"f_ind={fit0['frac_ind']:.2f}  chi2/ndf={fit0['chi2ndf']:.2f}")
        lo_med = float(np.median(q0[~coll0])) if np.any(~coll0) else np.nan
        hi_med = float(np.median(q0[coll0])) if np.any(coll0) else np.nan
        print(f"  backtrack medians: induction={lo_med:.0f} e-  shower={hi_med:.0f} e-  "
              f"(N_ind={int((~coll0).sum())}, N_shw={int(coll0.sum())})")
        plot_headline(ctx, q0, tr0, fit0, args.outdir, xlim=xlim, thr_sigma=thr_sigma)
    else:
        print("  !! nominal fit failed:", fit0.get("reason") if fit0 else "no data")

    def dist_panels(fits, disp):
        """[(disp_value, fit, q, truth), ...] for the per-test-case distribution grid."""
        return [(d, f, q, tr) for d, (val, f, q, tr) in zip(disp, fits)]

    # --- threshold scan (cheap: FEE-only on cached nominal signals)
    print("\n=== Threshold scan ===")
    thr_disp = np.asarray(args.thresholds, float) / 1e3
    thr_scan, thr_fits = run_fixed_induction_scan(
        ctx, evs, "threshold", args.thresholds, base_thr, base_reset, n_eff,
        seed0=args.seed + 2000, set_fn=lambda v: (float(v), base_reset),
        template_fn=template_fn, qmax=qmax)
    plot_scan(ctx, thr_scan, r"Pixel charge threshold $Q_{\mathrm{thr}}$  ($10^{3}\,e^{-}$)",
              "ti_scan_threshold.png", args.outdir, knob_vals_disp=thr_disp, mpv_ylim=mpv_ylim)
    plot_scan_distributions(dist_panels(thr_fits, thr_disp),
                            r"Readout-$Q$ spectra vs pixel charge threshold",
                            lambda d: r"$Q_{\mathrm{thr}}=%.1f$" % d, r"$Q_{\mathrm{thr}}$",
                            "ti_dist_threshold.png", args.outdir, xlim=xlim, thr_sigma=thr_sigma)

    # --- periodic-reset scan (recompile per value; FEE-only on cached signals)
    print("\n=== Periodic-reset scan ===")
    rst_scan, rst_fits = run_fixed_induction_scan(
        ctx, evs, "reset", args.resets, base_thr, base_reset, n_eff,
        seed0=args.seed + 3000, set_fn=lambda v: (base_thr, int(v)),
        template_fn=template_fn, qmax=qmax)
    set_periodic_reset(ctx, base_reset)  # restore nominal
    # x-axis = reset RATE (1/period); "off" maps naturally to 0 (no resets)
    reset_rate = reset_rate_khz(np.asarray(args.resets, int), ts)
    plot_scan(ctx, rst_scan, r"Periodic-reset rate  (kHz)",
              "ti_scan_periodic_reset.png", args.outdir, knob_vals_disp=reset_rate)
    plot_scan_distributions(dist_panels(rst_fits, reset_rate),
                            r"Readout-$Q$ spectra vs periodic-reset rate",
                            lambda d: ("reset off" if d == 0 else r"%.0f kHz" % d), r"kHz",
                            "ti_dist_periodic_reset.png", args.outdir, xlim=xlim, thr_sigma=thr_sigma)

    # --- induction scan (re-induce per scale on the drift cache; reuse off-cache at 0).
    #     Subsampled to --induction-events: this scan is the only one that re-runs the
    #     expensive pre-FEE induction per point, so it is the wall-time bottleneck.
    ind_n = min(args.induction_events, n_eff) if args.induction_events else n_eff
    print(f"\n=== Induction-response scan ({ind_n} of {n_eff} events) ===")
    ind_scan, ind_fits = run_induction_scan(
        ctx, drift_cache[:ind_n], args.inductions, base_thr, base_reset, ind_n,
        seed0=args.seed + 4000, template_fn=template_fn, evs_off=evs_off[:ind_n],
        qmax=qmax, collect_frac=args.collect_frac)
    ind_disp = np.asarray(args.inductions, float)
    plot_scan(ctx, ind_scan, r"Neighbour-pad induction-response scale",
              "ti_scan_induction.png", args.outdir, knob_vals_disp=ind_disp)
    plot_scan_distributions(dist_panels(ind_fits, ind_disp),
                            r"Readout-$Q$ spectra vs induction-response scale",
                            lambda d: r"induction $\times%.1f$" % d, r"scale",
                            "ti_dist_induction.png", args.outdir, xlim=xlim, thr_sigma=thr_sigma)

    # --- disentanglement money plot
    print("\n=== Sensitivity matrix ===")
    obs_keys = ["peak_mpv", "peak_height", "tail_height"]
    obs_lbl = ["Peak MPV", "Peak height", "Tail height"]
    matrix = [sensitivity_row(thr_scan, obs_keys),
              sensitivity_row(rst_scan, obs_keys),
              sensitivity_row(ind_scan, obs_keys)]
    knobs = ["Threshold", "Periodic reset", "Induction"]
    plot_sensitivity(matrix, knobs, obs_lbl, args.outdir)
    print("  sensitivity |frac obs / frac knob| (rows=knobs, cols=observables):")
    print("            " + "  ".join(f"{l:>13}" for l in obs_lbl))
    for kn, row in zip(knobs, matrix):
        print(f"  {kn:>13} " + "  ".join(f"{v:13.2f}" if np.isfinite(v) else f"{'--':>13}"
                                         for v in row))

    # --- save everything (incl. the template grid + closed-form coefficients)
    det_name, det_geom, det_path = vd.detector_identity(ctx)
    closed_form = {k: dict(name=tparam.forms[k]["name"], r2=tparam.forms[k]["r2"],
                           params=tparam.forms[k]["params"])
                   for k in tparam.PARAMS if tparam.forms.get(k)}
    np.savez(f"{args.outdir}/threshold_induction_results.npz",
             detector_name=np.array(det_name), detector_path=np.array(det_path),
             base_threshold=base_thr, base_reset=base_reset,
             nominal_q=q0, nominal_truth=tr0,
             **{f"thr_{k}": v for k, v in thr_scan.items()},
             **{f"rst_{k}": v for k, v in rst_scan.items()},
             **{f"ind_{k}": v for k, v in ind_scan.items()},
             template_thresholds=np.asarray(thr_grid, float),
             template_resets=np.asarray(rst_grid, int),
             template_rates=reset_rate_khz(np.asarray(rst_grid, int), ts),
             **{f"template_{k}": grid[k] for k in
                ("mpv_L", "eta_L", "A_L", "mu_G", "sig_G", "chi2ndf")},
             template_closed_form=np.array(repr(closed_form)),
             sensitivity=np.array(matrix, float),
             sensitivity_knobs=np.array(knobs),
             sensitivity_observables=np.array(obs_lbl))
    print(f"\nWrote results + plots to {args.outdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
