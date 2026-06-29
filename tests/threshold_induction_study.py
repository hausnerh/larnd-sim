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

  The Q distribution of single-hit pixels is therefore BIMODAL. We fit it with a
  sum of two Landau*Gauss (langaus) peaks -- one per population -- and ask:

    * Are the fits reasonable? (validated against a per-pixel TRUTH label: the
      net integrated charge tells us which pixels actually collected charge.)
    * Do threshold / periodic-reset / induction-response each move the fitted
      parameters in CLEAR, MEASURABLE, and DISENTANGLED ways?

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
def event_presignals(ctx, tracks, seed):
    """Drift -> get_pixels -> induce -> sum, returned as HOST arrays + truth.

    Everything needed to re-run only the FEE stage later. truth_e[ip] is the NET
    integrated charge on pixel ip (electrons; units.e==1): ~the collected charge
    on a real collection pad, ~0 on an induction-only pad. This is the per-pixel
    TRUTH label used to validate the two-population decomposition.
    """
    drifted = vd.quench_and_drift(ctx, tracks)
    if float(np.sum(drifted["n_electrons"])) <= 0:
        return None
    neigh, radius = vd.find_pixels(ctx, drifted)
    signals = vd.induce_current(ctx, drifted, neigh, seed=seed)
    summed = vd.sum_to_pixels(ctx, signals, neigh, radius, drifted)
    unique_pix, pixels_signals = summed[0], summed[1]
    if pixels_signals is None:
        return None
    ts = ctx.detector.TIME_SAMPLING
    ps_host = vd.to_host(pixels_signals).astype(np.float32)
    npix = ps_host.shape[0]
    # The study uses only the ADC (q_sum), never the per-track backtracking. The
    # backtracking array (pixels_tracks_signals, size nt0*sum(num_backtrack)) is by
    # far the largest per-event object and OOM'd the 200-event cache -- so we DROP
    # it: zero num_backtrack makes get_adc_values skip backtracking entirely while
    # adc_list (built from pixels_signals alone) is bit-for-bit unchanged.
    return dict(
        unique_pix=vd.to_host(unique_pix),
        pixels_signals=ps_host,
        pixels_tracks_signals=np.zeros(1, dtype=np.float64),
        num_backtrack=np.zeros(npix, dtype=np.int64),
        offset_backtrack=np.zeros(npix, dtype=np.int64),
        truth_e=ps_host.sum(axis=1, dtype=np.float64) * ts,   # net collected charge (e-)
        max_time=ps_host.shape[1] * ts,
    )


def event_fee_singlehits(ctx, ev, threshold_e, seed):
    """Run only the FEE on a cached pre-signal; return single-hit (Q, truth_e).

    A pixel is a SINGLE-HIT pixel iff it produced exactly one ADC sample. Its
    recorded charge Q is recovered from the ADC the way a data analysis would
    (vd.adc_to_charge, carrying the ~1-LSB quantization). Returns the measured Q
    (electrons) and the truth net charge for each single-hit pixel.
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
        return np.empty(0), np.empty(0)
    single_adc = adc[sel].max(axis=1)               # the one nonzero sample per row
    q = vd.adc_to_charge(ctx, single_adc)           # electrons
    return np.asarray(q, float), ev["truth_e"][sel]


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

    def two(self, x, m1, e1, s1, A1, m2, e2, s2, A2):
        return (self.comp(x, m1, e1, s1, A1) + self.comp(x, m2, e2, s2, A2))


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


def _guess(centres, counts, threshold_e):
    """Seed (m1,e1,s1,A1, m2,e2,s2,A2) from peak heuristics (truth-free).

    Physical roles: component 1 (induction) is a NARROW peak that rides just above
    threshold; component 2 (shower) is a BROADER langaus at higher Q. The band
    `2*threshold` only splits the seed *amplitudes*, not the fit -- the populations
    may overlap.
    """
    w = centres[1] - centres[0]
    band = 2.0 * threshold_e
    near, far = centres < band, centres >= band
    if near.any() and counts[near].sum() > 0:
        m1 = float(centres[near][np.argmax(counts[near])])
        A1 = float(counts[near].sum() * w)
    else:
        m1, A1 = 1.2 * threshold_e, float(counts.sum() * w * 0.5)
    if far.any() and counts[far].sum() > 0:
        m2 = float(np.average(centres[far], weights=counts[far]))
        A2 = float(counts[far].sum() * w)
    else:
        m2, A2 = max(2.5 * threshold_e, 1.6 * m1), float(counts.sum() * w * 0.3)
    return [m1, 0.15 * threshold_e, 0.12 * threshold_e, A1,
            m2, 0.20 * m2, 0.15 * m2, A2]


def _guess_from_truth(q, truth, truth_cut, threshold_e, w):
    """Seed from the MC truth: MPVs at the truth induction/shower recorded-Q medians,
    areas at the truth sub-sample counts. The fit still optimises on the data -- the
    truth only provides the starting point (and anchors the shower MPV, which the
    heuristic otherwise mis-places out in the sparse Landau tail)."""
    T = threshold_e
    good = np.isfinite(q) & (q > 0)
    qi = q[good & (truth <= truth_cut)]
    qs = q[good & (truth > truth_cut)]
    m1 = float(np.median(qi)) if qi.size else 1.1 * T
    m2 = float(np.median(qs)) if qs.size else 2.5 * T
    m1 = min(max(m1, 0.95 * T), 1.9 * T)
    m2 = max(m2, 1.35 * T, 1.08 * m1)
    return [m1, 0.15 * T, 0.10 * T, float(qi.size * w),
            m2, 0.45 * m2, 0.25 * m2, float(qs.size * w)]


def fit_two_langaus(q, threshold_e, truth=None, truth_cut=1500.0, qmax=None):
    """Fit the single-hit Q spectrum with two langaus peaks (induction + shower).

    If `truth` (per-pixel net charge, aligned with q) is given, the fit is SEEDED
    from the truth sub-sample medians/counts; it still optimises on the data alone,
    so the extracted parameters remain a data measurement. Returns a dict with the
    parameters, derived observables, chi2/ndf, and histogram/edges for plotting.
    """
    from scipy.optimize import curve_fit
    h = _hist(q, threshold_e, qmax=qmax)
    if h is None:
        return None
    centres, counts, width, edges = h
    model = Langaus(centres[0], centres[-1])
    if truth is not None and np.size(truth) == np.size(q):
        p0 = _guess_from_truth(np.asarray(q, float), np.asarray(truth, float),
                               truth_cut, threshold_e, width)
    else:
        p0 = _guess(centres, counts, threshold_e)
    cmax = centres[-1]
    span = centres[-1] - centres[0]
    T = threshold_e
    if truth is not None and np.size(truth) == np.size(q):
        # Anchor the MPVs to a window around the truth-seed medians so the fit
        # can't drift away from the truth-identified populations (the overlap
        # otherwise pushes the broad shower langaus out into the Landau tail).
        m1s, m2s = p0[0], p0[4]
        lb = [max(0.85 * T, 0.70 * m1s), 0.02 * T, 0.02 * T, 0.0,
              max(1.05 * T, 0.60 * m2s), 0.02 * T, 0.02 * T, 0.0]
        ub = [min(2.2 * T, 1.30 * m1s), 0.80 * T, 0.80 * T, np.inf,
              min(cmax,   1.45 * m2s),  span,     span,     np.inf]
    else:
        # No truth: induction is a NARROW peak pinned above threshold, shower a
        # BROADER langaus higher up; overlapping MPV windows so merged populations
        # still fit and it can't collapse to a single broad-flat component.
        lb = [0.9 * T, 0.02 * T, 0.02 * T, 0.0,    1.3 * T, 0.02 * T, 0.02 * T, 0.0]
        ub = [2.0 * T, 0.80 * T, 0.80 * T, np.inf, cmax,    span,     span,     np.inf]
    ub = [max(u, lb[i] * 1.01 + 1e-6) for i, u in enumerate(ub)]   # guard lb < ub
    split = 2.0 * T                                 # kept for the headline annotation
    p0 = [min(max(v, lb[i] + 1e-6), ub[i] - 1e-6) for i, v in enumerate(p0)]
    sigma = np.sqrt(counts) + 1.0                   # Poisson-ish weights
    try:
        popt, pcov = curve_fit(model.two, centres, counts, p0=p0,
                               bounds=(lb, ub), sigma=sigma,
                               absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        return dict(ok=False, reason=str(exc), centres=centres, counts=counts,
                    width=width, edges=edges, threshold_e=threshold_e)
    m1, e1, s1, A1, m2, e2, s2, A2 = popt
    pred = model.two(centres, *popt)
    ndf = max(len(centres) - len(popt), 1)
    chi2 = float(np.sum(((counts - pred) / sigma) ** 2))
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    f_ind = A1 / (A1 + A2) if (A1 + A2) > 0 else np.nan
    return dict(
        ok=True, popt=popt, perr=perr, model=model, centres=centres, edges=edges,
        counts=counts, width=width, pred=pred, split=split, threshold_e=threshold_e,
        mpv_ind=float(m1), mpv_shw=float(m2),
        mpv_ind_err=float(perr[0]), mpv_shw_err=float(perr[4]),
        area_ind=float(A1), area_shw=float(A2), frac_ind=float(f_ind),
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
        return np.empty(0), np.empty(0)
    return np.concatenate(qs), np.concatenate(tr)


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


def _draw_charge_dist(ax, q, truth, fit, threshold_e, truth_cut, xlim=None,
                      compact=False, thr_sigma=0.0, logy=True):
    """Draw one FEE-readout single-hit Q spectrum (the noisy Q you trigger on in
    data), bins coloured by the per-pixel truth into induction (net collected charge
    ~ 0) vs shower (real collection), the two-langauss fit overlaid (each component
    in its population's colour), and the pixel-charge-threshold line with a +/-sigma
    band: the discriminator fires on `q+noise >= threshold + disc_noise`, so the
    threshold itself is Gaussian-smeared by sigma_disc -- which is why recorded
    charges land below the nominal line."""
    edges = fit["edges"] / 1e3
    is_shw = truth > truth_cut
    ax.hist([q[~is_shw] / 1e3, q[is_shw] / 1e3], bins=edges, stacked=True,
            color=[_C_IND, _C_SHW], alpha=0.55, edgecolor="white",
            linewidth=(0.2 if compact else 0.35), label=["Induction", "Shower"])
    if fit.get("ok"):
        xs = np.linspace(fit["edges"][0], fit["edges"][-1], 700)
        m1, e1, s1, A1, m2, e2, s2, A2 = fit["popt"]
        lw = 1.2 if compact else 1.6
        ax.plot(xs / 1e3, fit["model"].comp(xs, m1, e1, s1, A1), color=_C_IND, ls="--", lw=lw)
        ax.plot(xs / 1e3, fit["model"].comp(xs, m2, e2, s2, A2), color=_C_SHW, ls="--", lw=lw)
        ax.plot(xs / 1e3, fit["model"].two(xs, *fit["popt"]), color=_C_SUM,
                ls="-", lw=lw + 0.5, label="Two-Langauss fit")
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


def plot_headline(ctx, q, truth, fit, outdir, truth_cut, xlim=None, thr_sigma=0.0):
    """Nominal single-hit Q spectrum (the noisy Q triggered on, induction/shower
    coloured) + two-langauss fit + threshold-with-noise band. Saved both log-y
    (ti_hist_nominal.png, shows the tail) and linear-y (ti_hist_nominal_lin.png,
    shows the peak region where the two langauss separate)."""
    import matplotlib.pyplot as plt
    for logy, suf in ((True, ""), (False, "_lin")):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        _draw_charge_dist(ax, q, truth, fit, fit["threshold_e"], truth_cut, xlim=xlim,
                          thr_sigma=thr_sigma, logy=logy)
        if fit.get("ok"):
            stats = "\n".join((
                r"$\mathrm{MPV}_{\mathrm{ind}}=%.1f\times10^{3}\,e^{-}$" % (fit["mpv_ind"] / 1e3),
                r"$\mathrm{MPV}_{\mathrm{shw}}=%.1f\times10^{3}\,e^{-}$" % (fit["mpv_shw"] / 1e3),
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
                            truth_cut, xlim=None, thr_sigma=0.0):
    """Grid of readout-Q spectra + fits, one panel per test case (scan value). Any
    leftover grid cell is filled with a table of the fit MPVs and integrals (the
    per-population pixel yields) across the scan. `panels` is
    [(knob_disp, fit, q, truth), ...]; `fmt_val` formats each panel's tag."""
    import matplotlib.pyplot as plt
    items = [(d, f, q, tr) for (d, f, q, tr) in panels if f and f.get("ok")]
    if not items:
        return
    n = len(items)
    ncol = min(max(n, 2), 3)
    nrow = int(np.ceil((n + 1) / ncol))             # +1 so there's a cell for the table
    base = fname[:-4] if fname.endswith(".png") else fname
    for logy, suf in ((True, ""), (False, "_lin")):   # log (tail) + linear (peak) versions
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 2.9 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (d, f, q, tr) in zip(axes.flat, items):
            _draw_charge_dist(ax, q, tr, f, f["threshold_e"], truth_cut, xlim=xlim,
                              compact=True, thr_sigma=thr_sigma, logy=logy)
            ax.text(0.95, 0.93, fmt_val(d), transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
        leftover = list(axes.flat[n:])
        for ax in axes[-1]:
            ax.set_xlabel(r"$Q$ ($10^{3}\,e^{-}$)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Single-hit pixels")
        if leftover:                                # fit-summary table in the first spare cell
            tax = leftover[0]
            # NB: axes share y -- do NOT touch this axis's scale or it flips them all.
            tax.set_frame_on(False); tax.tick_params(left=False, bottom=False,
                                                     labelleft=False, labelbottom=False)
            cols = [col0_header, r"$\mathrm{MPV}_{\mathrm{i}}$", r"$\mathrm{MPV}_{\mathrm{s}}$",
                    r"$\int_{\mathrm{i}}$", r"$\int_{\mathrm{s}}$"]
            rows = [[("%g" % d),
                     "%.1f" % (f["mpv_ind"] / 1e3), "%.1f" % (f["mpv_shw"] / 1e3),
                     "%d" % round(f["area_ind"] / f["width"]),
                     "%d" % round(f["area_shw"] / f["width"])] for (d, f, q, tr) in items]
            tbl = tax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                            bbox=[0.0, 0.0, 1.0, 0.92])
            tbl.auto_set_font_size(False); tbl.set_fontsize(8.0)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("0.8")
                if r == 0:
                    cell.set_text_props(weight="bold")
            tax.set_title(r"MPV ($10^{3}\,e^{-}$), $\int$ = pixels", fontsize=8.5, pad=2)
            for ax in leftover[1:]:
                ax.set_visible(False)
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(title, y=1.06, fontsize=12)
        fig.tight_layout()
        _savefig(fig, f"{outdir}/{base}{suf}.png")


def plot_scan(ctx, scan, knob_label, fname, outdir, knob_vals_disp=None, mpv_ylim=None):
    """Fitted observables vs one knob, in three stacked panels sharing the x-axis:
    (a) induction & shower MPV, (b) induction fraction, (c) single-hit rate. Points
    with uncertainties (no connecting lines). `mpv_ylim` (in 10^3 e-) sets panel (a)
    to the same charge range as the distributions, so the MPV shifts are read in
    context."""
    import matplotlib.pyplot as plt
    x = np.asarray(knob_vals_disp if knob_vals_disp is not None else scan["knob"], float)
    order = np.argsort(x)
    x = x[order]
    col = lambda k: np.asarray(scan[k], float)[order]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.8), sharex=True)
    fig.subplots_adjust(hspace=0.07)
    ekw = dict(ms=6.5, mfc="white", mew=1.5, capsize=3, elinewidth=1.1, ls="none")
    axes[0].errorbar(x, col("mpv_ind") / 1e3, yerr=col("mpv_ind_err") / 1e3,
                     fmt="o", color=_C_IND, label=r"Induction $\mathrm{MPV}$", **ekw)
    axes[0].errorbar(x, col("mpv_shw") / 1e3, yerr=col("mpv_shw_err") / 1e3,
                     fmt="s", color=_C_SHW, label=r"Shower $\mathrm{MPV}$", **ekw)
    axes[0].set_ylabel(r"Fitted MPV  ($10^{3}\,e^{-}$)")
    if mpv_ylim is not None:
        axes[0].set_ylim(*mpv_ylim)
    axes[0].legend(loc="best")
    axes[1].errorbar(x, col("frac_ind"), fmt="D", color=_C_GRN, **ekw)
    axes[1].set_ylabel(r"Induction fraction $f_{\mathrm{ind}}$")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].errorbar(x, col("rate"), fmt="^", color=_C_PUR, **ekw)
    axes[2].set_ylabel("Single-hit pixels / event")
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


# ===========================================================================
# Scans
# ===========================================================================
def run_fixed_induction_scan(ctx, evs, knob_name, knob_vals, base_threshold,
                             base_reset, n_events, seed0, set_fn, qmax=None):
    """Threshold or reset scan: reuse cached nominal-induction pre-signals.

    `set_fn(value)` returns (threshold_e, reset_cycles) for that knob value; the
    other parameter is held at its base. Fits each aggregated spectrum.
    """
    out = {k: [] for k in ("knob", "mpv_ind", "mpv_ind_err", "mpv_shw",
                           "mpv_shw_err", "frac_ind", "rate", "chi2ndf", "n_single")}
    fits = []
    for j, val in enumerate(knob_vals):
        thr, rst = set_fn(val)
        q, tr = aggregate_singlehits(ctx, evs, thr, rst, 1.0,
                                     seed0=seed0 + j * 1000)
        fit = fit_two_langaus(q, thr, truth=tr, qmax=qmax)
        fits.append((val, fit, q, tr))
        out["knob"].append(val)
        if fit and fit.get("ok"):
            for k in ("mpv_ind", "mpv_ind_err", "mpv_shw", "mpv_shw_err",
                      "frac_ind", "chi2ndf", "n_single"):
                out[k].append(fit[k])
            out["rate"].append(fit["n_single"] / max(n_events, 1))
        else:
            for k in ("mpv_ind", "mpv_ind_err", "mpv_shw", "mpv_shw_err",
                      "frac_ind", "chi2ndf", "n_single", "rate"):
                out[k].append(np.nan)
    return {k: np.asarray(v, float) for k, v in out.items()}, fits


def run_induction_scan(ctx, raw_events, scales, base_threshold, base_reset,
                       n_events, seed0, qmax=None):
    """Induction scan: re-induce the pre-FEE current at each response scale.

    This is the expensive scan -- scaling the response changes tracks_current_mc's
    output, so the pre-FEE stage is recomputed for every scale.
    """
    set_periodic_reset(ctx, base_reset)
    out = {k: [] for k in ("knob", "mpv_ind", "mpv_ind_err", "mpv_shw",
                           "mpv_shw_err", "frac_ind", "rate", "chi2ndf", "n_single")}
    fits = []
    for j, s in enumerate(scales):
        set_induction(ctx, s)
        evs = []
        for i, raw in enumerate(raw_events):
            ev = event_presignals(ctx, raw, seed=seed0 + j * 777 + i)
            if ev is not None:
                evs.append(ev)
        q, tr = aggregate_singlehits(ctx, evs, base_threshold, base_reset, s,
                                     seed0=seed0 + j * 1000 + 50000)
        fit = fit_two_langaus(q, base_threshold, truth=tr, qmax=qmax)
        fits.append((s, fit, q, tr))
        out["knob"].append(s)
        if fit and fit.get("ok"):
            for k in ("mpv_ind", "mpv_ind_err", "mpv_shw", "mpv_shw_err",
                      "frac_ind", "chi2ndf", "n_single"):
                out[k].append(fit[k])
            out["rate"].append(fit["n_single"] / max(n_events, 1))
        else:
            for k in ("mpv_ind", "mpv_ind_err", "mpv_shw", "mpv_shw_err",
                      "frac_ind", "chi2ndf", "n_single", "rate"):
                out[k].append(np.nan)
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
    ap.add_argument("--truth-cut", type=float, default=1500.0,
                    help="net collected e- above which a single-hit pixel is 'shower'")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[3000, 4000, 5000, 7000, 10000], help="e-")
    ap.add_argument("--resets", type=int, nargs="+",
                    default=[-1, 400, 200, 100, 50],
                    help="PERIODIC_RESET_CYCLES (-1 = off)")
    ap.add_argument("--inductions", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0], help="response scale")
    ap.add_argument("--seed", type=int, default=12345)
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

    # --- nominal-induction pre-signals (cached, reused by threshold + reset scans)
    print("Computing nominal-induction pre-FEE signals (cached)...")
    set_induction(ctx, 1.0)
    set_periodic_reset(ctx, base_reset)
    evs = []
    for i, raw in enumerate(raw_events):
        ev = event_presignals(ctx, raw, seed=args.seed + 100 + i)
        if ev is not None:
            evs.append(ev)
    n_eff = len(evs)
    print(f"  {n_eff}/{args.n_events} events produced collectable charge")
    if n_eff == 0:
        sys.exit("No events produced any pixels -- check shower placement / config.")

    # --- headline: nominal spectrum + fit. A common charge range (q99.7 of the
    #     nominal data) is used for every fit & plot so the test cases share axes.
    print("\n=== Nominal single-hit spectrum + two-langaus fit ===")
    q0, tr0 = aggregate_singlehits(ctx, evs, base_thr, base_reset, 1.0, seed0=args.seed + 1)
    # Bin/fit out to the tail, but DISPLAY a fixed 0..50 (10^3 e-) window everywhere
    # so the spectra and the MPV panels share one charge axis.
    qmax = float(min(np.percentile(q0[q0 > 0], 99.9), 50000.0)) if np.any(q0 > 0) else 30000.0
    xlim = (0.0, 50.0)
    mpv_ylim = (0.0, 50.0)
    fit0 = fit_two_langaus(q0, base_thr, truth=tr0, qmax=qmax)
    if fit0 and fit0.get("ok"):
        print(f"  single-hit pixels: {fit0['n_single']}  "
              f"Ind.MPV={fit0['mpv_ind']:.0f} e-  Shower.MPV={fit0['mpv_shw']:.0f} e-  "
              f"f_ind={fit0['frac_ind']:.2f}  chi2/ndf={fit0['chi2ndf']:.2f}")
        lo_med = float(np.median(q0[tr0 <= args.truth_cut])) if np.any(tr0 <= args.truth_cut) else np.nan
        hi_med = float(np.median(q0[tr0 > args.truth_cut])) if np.any(tr0 > args.truth_cut) else np.nan
        print(f"  truth medians (seed check): induction={lo_med:.0f} e-  shower={hi_med:.0f} e-")
        plot_headline(ctx, q0, tr0, fit0, args.outdir, args.truth_cut, xlim=xlim,
                      thr_sigma=thr_sigma)
    else:
        print("  !! nominal fit failed:", fit0.get("reason") if fit0 else "no data")

    def dist_panels(fits, disp):
        """[(disp_value, fit, q, truth), ...] for the per-test-case distribution grid."""
        return [(d, f, q, tr) for d, (val, f, q, tr) in zip(disp, fits)]

    # --- threshold scan (cheap: FEE-only on cached signals)
    print("\n=== Threshold scan ===")
    thr_disp = np.asarray(args.thresholds, float) / 1e3
    thr_scan, thr_fits = run_fixed_induction_scan(
        ctx, evs, "threshold", args.thresholds, base_thr, base_reset, n_eff,
        seed0=args.seed + 2000, set_fn=lambda v: (float(v), base_reset), qmax=qmax)
    plot_scan(ctx, thr_scan, r"Pixel charge threshold $Q_{\mathrm{thr}}$  ($10^{3}\,e^{-}$)",
              "ti_scan_threshold.png", args.outdir, knob_vals_disp=thr_disp, mpv_ylim=mpv_ylim)
    plot_scan_distributions(dist_panels(thr_fits, thr_disp),
                            r"Readout-$Q$ spectra vs pixel charge threshold",
                            lambda d: r"$Q_{\mathrm{thr}}=%.1f$" % d, r"$Q_{\mathrm{thr}}$",
                            "ti_dist_threshold.png", args.outdir, args.truth_cut, xlim=xlim, thr_sigma=thr_sigma)

    # --- periodic-reset scan (recompile per value; FEE-only on cached signals)
    print("\n=== Periodic-reset scan ===")
    ts = ctx.detector.TIME_SAMPLING
    rst_scan, rst_fits = run_fixed_induction_scan(
        ctx, evs, "reset", args.resets, base_thr, base_reset, n_eff,
        seed0=args.seed + 3000, set_fn=lambda v: (base_thr, int(v)), qmax=qmax)
    set_periodic_reset(ctx, base_reset)  # restore nominal (off)
    # x-axis = reset RATE (1/period); "off" maps naturally to 0 (no resets)
    reset_rate = np.array([1.0e3 / (v * ts) if v > 0 else 0.0 for v in args.resets])
    plot_scan(ctx, rst_scan, r"Periodic-reset rate  (kHz)",
              "ti_scan_periodic_reset.png", args.outdir, knob_vals_disp=reset_rate)
    plot_scan_distributions(dist_panels(rst_fits, reset_rate),
                            r"Readout-$Q$ spectra vs periodic-reset rate",
                            lambda d: ("reset off" if d == 0 else r"%.0f kHz" % d), r"kHz",
                            "ti_dist_periodic_reset.png", args.outdir, args.truth_cut, xlim=xlim, thr_sigma=thr_sigma)

    # --- induction scan (expensive: re-induce per scale)
    print("\n=== Induction-response scan ===")
    ind_scan, ind_fits = run_induction_scan(ctx, raw_events, args.inductions, base_thr,
                                            base_reset, n_eff, seed0=args.seed + 4000, qmax=qmax)
    ind_disp = np.asarray(args.inductions, float)
    plot_scan(ctx, ind_scan, r"Neighbour-pad induction-response scale",
              "ti_scan_induction.png", args.outdir, knob_vals_disp=ind_disp)
    plot_scan_distributions(dist_panels(ind_fits, ind_disp),
                            r"Readout-$Q$ spectra vs induction-response scale",
                            lambda d: r"induction $\times%.1f$" % d, r"scale",
                            "ti_dist_induction.png", args.outdir, args.truth_cut, xlim=xlim, thr_sigma=thr_sigma)

    # --- disentanglement money plot
    print("\n=== Sensitivity matrix ===")
    obs_keys = ["mpv_ind", "mpv_shw", "frac_ind", "rate"]
    obs_lbl = ["Ind. MPV", "Shower MPV", "Ind. fraction", "Single-hit rate"]
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

    # --- save everything
    det_name, det_geom, det_path = vd.detector_identity(ctx)
    np.savez(f"{args.outdir}/threshold_induction_results.npz",
             detector_name=np.array(det_name), detector_path=np.array(det_path),
             base_threshold=base_thr, base_reset=base_reset,
             nominal_q=q0, nominal_truth=tr0,
             **{f"thr_{k}": v for k, v in thr_scan.items()},
             **{f"rst_{k}": v for k, v in rst_scan.items()},
             **{f"ind_{k}": v for k, v in ind_scan.items()},
             sensitivity=np.array(matrix, float),
             sensitivity_knobs=np.array(knobs),
             sensitivity_observables=np.array(obs_lbl))
    print(f"\nWrote results + plots to {args.outdir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
