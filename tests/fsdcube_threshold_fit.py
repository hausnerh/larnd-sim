#!/usr/bin/env python3
# =============================================================================
# tests/fsdcube_threshold_fit.py -- model the FSDCube DATA hit-Q distribution as a MIXTURE over
# per-pixel discriminator thresholds, built by INTERPOLATING the mono-threshold sim distributions
# in a ti_mult_grid.npz (threshold_induction_study.py --mult-grid).
#
# Everything here is density / shape only (area-normalized), as the question is about the shape of
# the charge spectrum, not absolute rates.
#
#   basis        f_T(Q): the single-hit-pixel charge density the sim produces when EVERY pixel has
#                threshold T. The grid carries it at T in {2.5, 3.75, 5, 6.25, 7.5} ke- (base reset),
#                and its peak tracks T (a clean threshold turn-on), so it is a good mixing basis.
#   interpolate  f_T(Q) linearly (per Q-bin) in T, so ANY threshold between grid nodes has a shape. The
#                grid SHOULD be simulated across the per-pixel map range (THRESHOLDS 1.25..11.25 ke- in
#                shower_grid_array.sh); --extrapolate then CONTINUES the end slopes (a linear spline) only
#                to fill the small residual of pixels below/above the simulated node range.
#   predict      P_known(Q) = < f_{T_pix}(Q) > over the REAL per-pixel threshold map
#                (larndsim/bin/thresholds_fsdcube.npz) -- "what we expect given the known thresholds".
#                Compare to the nominal uniform-threshold sim (should be close) and to the data.
#   fit          w(T) >= 0, sum w = 1 minimizing || sum_T w_T f_T(Q) - D(Q) || for a target density D
#                (NNLS on the probability simplex). D = DATA answers "does a DIFFERENT threshold
#                spread match the data better than the nominal map?"; D = a synthetic mixture / a
#                single node is the self-test (recover the input weights).
#
# Runs in the SAME env as fsdcube_hitq_datamc.py (numpy + matplotlib; h5py only to read FLOW data,
# scipy never needed). Reuses that script's reader / threshold-map / Q-binning so the axis and the
# single-hit definition match the data/MC plots exactly.
#
# Usage:
#   # local validation on a grid, NO data needed (recovers known weights; writes diagnostic plots):
#   python tests/fsdcube_threshold_fit.py --grid 2=grid_prc2/ti_mult_grid.npz --self-test --outdir tfit
#
#   # real-data fit on NERSC (reads FLOW exactly like the plotter), one panel per prc:
#   python tests/fsdcube_threshold_fit.py --outdir tfit --prc 2 16 \
#       --grid 2=$ND_WORK/tistudy_datamc/grid_prc2/ti_mult_grid.npz \
#              16=$ND_WORK/tistudy_datamc/grid_prc16/ti_mult_grid.npz
#
#   # re-fit a data density extracted by an earlier data run (saved as <outdir>/data_density_prc<P>.npz):
#   python tests/fsdcube_threshold_fit.py --grid 2=g2.npz --prc 2 --data-npz tfit/data_density_prc2.npz
# =============================================================================
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the plotter's binning, palette, threshold-map loader, and (on NERSC) the FLOW reader, so the
# Q axis + single-hit selection are byte-identical to the data/MC hit-Q plots we are modelling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsdcube_hitq_datamc import (QBINS, QMIN, QMAX, THR_MAP_DEF, DATA_COLOR, MC_COLORS,
                                 load_thr_map)

BINW = (QMAX - QMIN) / QBINS
EDGES = np.linspace(QMIN, QMAX, QBINS + 1)
CTR = 0.5 * (EDGES[1:] + EDGES[:-1])


def density(x):
    """area-normalized histogram (integral = 1) of x[ke-] on the shared Q axis."""
    h, _ = np.histogram(np.asarray(x, float), bins=EDGES)
    s = h.sum()
    return h / (s * BINW) if s else h.astype(float)


# ---------------------------------------------------------------------------
# basis: mono-threshold sim densities from the grid
# ---------------------------------------------------------------------------
def load_basis(path, observable="single", prc=None, prc_step_us=6.4, reset_per_prc=False):
    """ti_mult_grid.npz -> (T[ke- sorted], F[nT, QBINS] density rows, Y[nT] entries/shower, base_T, rst).

    The basis rows are taken at ONE reset node. By default that is the grid's base reset; with
    reset_per_prc the reset is the one MATCHING this prc (a reset window: cycles = round(prc *
    prc_step_us / ts), e.g. prc2 -> 128 cyc, prc16 -> 1024), so each prc's basis is the sim read out
    at its own reset -- which actually differ. `rst` (the reset used, cycles) is returned for labels.

    observable:
      'single' -- single-hit-pixel charge (node_q where node_n==1). For a single-hit pixel the
                  per-pixel total Q IS its one hit's Q, so this matches the data q[n==1] exactly,
                  and it is the sharpest threshold lever (its peak sits right at T).
      'hit'    -- every hit's charge (node_phq). Matches data q (all per-hit Q).
    """
    g = np.load(path, allow_pickle=True)
    thr = np.asarray(g["node_thr"], float); rst = np.asarray(g["node_rst"], int)
    base_rst = int(g["meta_base_reset"]); base_T = float(g["meta_base_threshold"]) / 1000.0
    ts = float(g["meta_ts"]) if "meta_ts" in g.files else 0.1
    use_rst = int(round(prc * prc_step_us / ts)) if (reset_per_prc and prc is not None) else base_rst
    nsh = max(int(g["meta_n_shower"]), 1)
    Ts = sorted({t for t, r in zip(thr.tolist(), rst.tolist()) if r == use_rst})
    if not Ts:
        raise SystemExit(f"{path}: no nodes at reset {use_rst} cyc (resets present: "
                         f"{sorted(set(int(r) for r in rst))}); check --prc-step-us / --reset-per-prc")
    F, Y = [], []
    for T in Ts:
        i = next(k for k in range(len(thr)) if thr[k] == T and rst[k] == use_rst)
        q = np.asarray(g["node_q"][i], float) / 1000.0
        n = np.asarray(g["node_n"][i], np.int16)
        phq = np.asarray(g["node_phq"][i], float) / 1000.0
        x = q[n == 1] if observable == "single" else phq
        F.append(density(x)); Y.append(x.size / nsh)
    return np.array(Ts) / 1000.0, np.vstack(F), np.array(Y), base_T, use_rst


def _lin_extrap(T, y, Tq):
    """piecewise-linear interpolation of y(T) at Tq with LINEAR EXTRAPOLATION beyond the end knots:
    np.interp clamps the ends flat, so here the first/last segment slope is continued instead (a
    linear spline). Vectorized over Tq."""
    T = np.asarray(T, float); y = np.asarray(y, float); Tq = np.asarray(Tq, float)
    out = np.interp(Tq, T, y)
    lo = Tq < T[0]
    if lo.any():
        out = np.where(lo, y[0] + (y[1] - y[0]) / (T[1] - T[0]) * (Tq - T[0]), out)
    hi = Tq > T[-1]
    if hi.any():
        out = np.where(hi, y[-1] + (y[-1] - y[-2]) / (T[-1] - T[-2]) * (Tq - T[-1]), out)
    return out


def interp_basis(T, F, Tq, extrap=False):
    """per-Q-bin interpolation of the density rows F(T) to query thresholds Tq. extrap=False clips Tq
    to [T_lo, T_hi] (np.interp's flat clamp). extrap=True LINEARLY EXTRAPOLATES each Q-bin's density
    past the grid's end thresholds (a linear spline), so thresholds below T_lo / above T_hi get an
    extrapolated turn-on rather than the edge shape. This is meant to fill only the SMALL residual
    outside the simulated node range -- the bulk should be covered by actual grid nodes (grow the
    grid's THRESHOLDS to [1.25, 11.25] ke-), with extrapolation just catching the thin tails. A
    linear-extrapolated density bin can dip < 0, so rows are floored at 0 and each row renormalized."""
    T = np.asarray(T, float); Tq = np.asarray(Tq, float)
    if extrap:
        G = np.column_stack([_lin_extrap(T, F[:, b], Tq) for b in range(F.shape[1])])  # (nq, QBINS)
    else:
        Tqc = np.clip(Tq, T.min(), T.max())
        G = np.column_stack([np.interp(Tqc, T, F[:, b]) for b in range(F.shape[1])])
    G = np.clip(G, 0.0, None)                         # extrapolation may push a density bin slightly < 0
    a = (G * BINW).sum(1, keepdims=True); a[a == 0] = 1.0
    return G / a


# ---------------------------------------------------------------------------
# known per-pixel threshold map -> prediction
# ---------------------------------------------------------------------------
def map_thresholds(thr_map_path):
    """per-pixel thresholds [ke-] from the real map (one entry per physical pixel)."""
    dense, _default = load_thr_map(thr_map_path)      # ke-, index = pixel id
    return dense


def fine_grid(T, F, n_fine, lo=None, hi=None, extrap=False):
    """A fine threshold axis [lo, hi] (default the basis range [T_lo, T_hi]) + the interpolated density
    rows on it. Fitting on this (not the few native nodes) is what gives a BROAD threshold distribution
    rather than weight piling up on the discrete grid thresholds; with extrap=True and lo/hi beyond the
    grid the wings are linearly extrapolated so the fit spans the full map range."""
    lo = float(T.min()) if lo is None else float(lo)
    hi = float(T.max()) if hi is None else float(hi)
    Tf = np.linspace(lo, hi, int(n_fine))
    return Tf, interp_basis(T, F, Tf, extrap=extrap)


def moments(T, w):
    """(mean, sd) in ke- of a threshold weight vector w over nodes T (w need not be normalized)."""
    w = np.asarray(w, float); s = w.sum()
    if s <= 0:
        return float("nan"), float("nan")
    m = float((w * T).sum() / s)
    return m, float(max((w * (T - m) ** 2).sum() / s, 0.0)) ** 0.5


def predict_known(T, F, vals, extrap=False):
    """P_known(Q): pixel-weighted mean of the interpolated per-pixel shapes f_{T_pix}(Q), area 1.
    extrap=True lets the low-/high-threshold pixels outside the grid contribute their extrapolated
    shape instead of being clamped to the edge node."""
    G = interp_basis(T, F, vals, extrap=extrap)         # (n_pix, QBINS)
    p = G.mean(0)
    s = (p * BINW).sum()
    return p / s if s else p


def seed_from_map(Tf, vals):
    """Seed weights on the fine grid Tf from the known per-pixel threshold map -- i.e. START the fit at
    the SIMULATED threshold distribution. Each pixel is binned to its nearest fine node (pixels outside
    [Tf[0], Tf[-1]] fold into the end node) and the counts are normalized onto the simplex."""
    Tf = np.asarray(Tf, float); vals = np.asarray(vals, float)
    edges = 0.5 * (Tf[1:] + Tf[:-1])                    # midpoints between adjacent fine nodes
    idx = np.clip(np.searchsorted(edges, vals), 0, Tf.size - 1)
    w = np.bincount(idx, minlength=Tf.size).astype(float)
    s = w.sum()
    return w / s if s else np.full(Tf.size, 1.0 / Tf.size)


# ---------------------------------------------------------------------------
# the fit: argmin_{w>=0, sum w=1} || F^T w - D ||^2 , numpy-only
# ---------------------------------------------------------------------------
def project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al. 2008)."""
    v = np.asarray(v, float)
    if v.size == 1:
        return np.ones(1)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, v.size + 1)
    rho = np.nonzero(u - css / idx > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def fit_weights(F, d, reg=0.0, iters=20000, w0=None, prior=0.0):
    """mixture weights w (one per basis row of F) whose density F^T w best matches target density d,
    constrained to the simplex (w>=0, sum=1 -> F^T w is itself a density). `reg` adds a 2nd-difference
    smoothness penalty (useful on a fine threshold grid). `w0` SEEDS the iteration (projected onto the
    simplex; default uniform): because the interpolated basis is low-rank (5 shapes spread over many
    fine nodes) the problem is weakly constrained along smooth re-distributions of weight, so within a
    finite iteration budget w0 shapes which near-equivalent solution is reached -- seeding at the sim
    map yields 'the sim distribution deformed as little as the data allows'. `prior` > 0 makes that a
    PRINCIPLED pull rather than just an initial condition: it adds (prior/2)||w - w0||^2 to the loss,
    so the fit stays near the sim unless the data demands otherwise. Projected gradient; returns w."""
    A = np.asarray(F, float).T               # (QBINS, nT): columns are basis densities
    d = np.asarray(d, float)
    nT = A.shape[1]
    AtA = A.T @ A; Atd = A.T @ d
    if reg > 0 and nT >= 3:
        D2 = (np.diag(np.ones(nT)) * -2.0
              + np.diag(np.ones(nT - 1), 1) + np.diag(np.ones(nT - 1), -1))[1:-1]
        AtA = AtA + reg * (D2.T @ D2)
    if prior > 0 and w0 is not None:         # (prior/2)||w - w0||^2 -> AtA += prior I, Atd += prior w0
        AtA = AtA + prior * np.eye(nT)
        Atd = Atd + prior * np.asarray(w0, float)
    L = float(np.linalg.eigvalsh(AtA)[-1]) or 1.0
    lr = 1.0 / L
    w = project_simplex(np.asarray(w0, float)) if w0 is not None else np.full(nT, 1.0 / nT)
    for _ in range(iters):
        w = project_simplex(w - lr * (AtA @ w - Atd))
    return w


def mixture(F, w):
    """density of a weight vector over the basis rows F, renormalized to area 1."""
    p = np.asarray(w, float) @ np.asarray(F, float)
    s = (p * BINW).sum()
    return p / s if s else p


def fit_joint(Ff_list, d_list, reg=0.0, w0=None, prior=0.0):
    """one SHARED w(T) fit to several (fine-basis, data) pairs at once: stack the per-prc Q-axes and
    solve the SAME simplex NNLS. The per-pixel thresholds are fixed hardware (only the reset window
    differs between prc2 and prc16), so a single threshold distribution must explain both spectra --
    that joint constraint is the added leverage (it breaks degeneracies either spectrum alone allows)."""
    return fit_weights(np.hstack(Ff_list), np.concatenate(d_list), reg=reg, w0=w0, prior=prior)


def fit_stats(T, Tf, Ff, d, w, vals, thr_lo, thr_hi):
    """numbers for the summary text file (these USED to be annotated on the plot). All are mean/width
    of distributions, not peaks: map m/s over the fit range, fitted-w(T) m/s, and the data residual."""
    inr = vals[(vals >= thr_lo) & (vals <= thr_hi)]
    fm, fs = moments(Tf, w)
    return dict(map_m=float(inr.mean()), map_s=float(inr.std()), fit_m=float(fm), fit_s=float(fs),
                resid=float(np.sqrt(((mixture(Ff, w) - d) ** 2).sum()) * BINW),
                grid_lo=float(T.min()), grid_hi=float(T.max()),
                frac_extrap=float(((vals < T.min()) | (vals > T.max())).mean()), n_inr=int(inr.size))


# ---------------------------------------------------------------------------
# target density (data): from FLOW on NERSC, from a saved npz, or self-test pseudo-data
#
# Densities are computed for a LIST of observables from one FLOW read, so the joint fit can later be
# posed over the whole SUITE of Q populations at once (single-hit, all-hit, ... -- one shared w(T)),
# not just one. For now --observable defaults to a single population; passing several opts them all in.
# ---------------------------------------------------------------------------
def _obs_select(q, n, observable):
    """the hit-charge sample for one observable: 'single' = single-hit-pixel Q (n==1), 'hit' = all hits.
    New populations (two-hit n==2, ...) slot in here and everything downstream follows."""
    return q[n == 1] if observable == "single" else q


def data_densities_from_flow(args, prc, observables):
    """DATA charge density per observable, from ONE FLOW read (the plotter's reader). {obs: (density,n)}."""
    from fsdcube_hitq_datamc import accumulate_q, data_files
    files = list(data_files(args.data_base, prc))
    if args.max_files:
        files = files[:args.max_files]
    print(f"prc{prc}: reading {len(files)} data files ...")
    tup = accumulate_q(files, args.adc_min, args.adc_max, args.max_events)
    q, n = np.asarray(tup[0], float), np.asarray(tup[1])
    out = {}
    for obs in observables:
        x = _obs_select(q, n, obs); out[obs] = (density(x), int(x.size))
    return out


def mc_densities_from_flow(args, prc, observables):
    """mkramer's FULL larnd-sim MC charge density per observable -- the 'sim r<R> prc<P>' overlay: the
    end-to-end sim at this prc, already run with the real per-pixel threshold map loaded (the fsd_cube
    config loads PIXEL_THRESHOLDS_FILE). Contrast predict_known(), which RECONSTRUCTS that same nominal
    map from our mono-threshold basis by interpolation; agreement of the two is a basis-closure check.
    Read at ONE readout radius (--sim-r). {obs: density or None}; None when no sim files (non-fatal)."""
    from fsdcube_hitq_datamc import accumulate_q, sim_files
    cfg, files = sim_files(args.sim_base, prc, args.sim_r, args.sim_var)
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        print(f"  prc{prc}: no MC files under '{cfg}' in {args.sim_base}; sim r{args.sim_r} overlay skipped")
        return {obs: None for obs in observables}
    print(f"  prc{prc}: reading {len(files)} MC files ({cfg}) for the sim r{args.sim_r} overlay ...")
    tup = accumulate_q(files, args.adc_min, args.adc_max, args.max_events)
    q, n = np.asarray(tup[0], float), np.asarray(tup[1])
    out = {}
    for obs in observables:
        x = _obs_select(q, n, obs); out[obs] = density(x) if x.size else None
    return out


def save_densities(path, dens, mc, observables, primary, reset):
    """persist per-observable DATA (+ Kramer-sim MC) densities so a later run re-fits fully locally. The
    primary observable is ALSO written under the legacy keys density/n/mc_density (back-compat)."""
    dprim, nprim = dens[primary]
    kw = dict(ctr=CTR, observable=primary, reset=reset, density=dprim, n=nprim,
              mc_density=(np.zeros(0) if mc.get(primary) is None else mc[primary]))
    for obs in observables:
        d, nd = dens[obs]
        kw[f"density_{obs}"] = d; kw[f"n_{obs}"] = nd
        kw[f"mc_density_{obs}"] = (np.zeros(0) if mc.get(obs) is None else mc[obs])
    np.savez(path, **kw)


def load_densities(dd, observables, primary):
    """inverse of save_densities, tolerant of older npz that carry only the primary (legacy bare keys).
    The legacy fallback is used only when the npz's stored observable actually matches, so a single-hit
    npz is never silently re-served as a different observable."""
    stored = str(dd["observable"]) if "observable" in dd.files else None
    dens, mc = {}, {}
    for obs in observables:
        if f"density_{obs}" in dd.files:
            d = np.asarray(dd[f"density_{obs}"], float); nd = int(dd[f"n_{obs}"]); ma = dd.get(f"mc_density_{obs}")
        elif obs == primary and stored in (None, obs):   # legacy npz: primary only, under the bare keys
            d = np.asarray(dd["density"], float); nd = int(dd.get("n", 0)); ma = dd.get("mc_density")
        else:
            print(f"    !! observable '{obs}' not in this npz (older file); skipping it"); continue
        dens[obs] = (d, nd)
        mc[obs] = (np.asarray(ma, float) if ma is not None and np.asarray(ma).size == CTR.size else None)
    return dens, mc


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_basis(T, F, base_T, vals, outpath, observable, use_rst, extrap=False):
    """the mono-threshold basis family + the known-map prediction vs the nominal uniform sim."""
    f, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("plasma")
    for j, Tj in enumerate(T):
        ax[0].step(CTR, F[j], where="mid", lw=1.6,
                   color=cmap(0.06 + 0.82 * j / max(len(T) - 1, 1)),
                   label=f"{Tj:.2f} ke$^-$" + (" (nom)" if abs(Tj - base_T) < 1e-6 else ""))
    ax[0].set_title(f"mono-threshold basis  f$_T$(Q)   [{observable}, reset {use_rst} cyc]")
    ax[0].set_xlabel("hit Q [ke$^-$]"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25, lw=0.6); ax[0].set_xlim(0, 50)

    jn = int(np.argmin(np.abs(T - base_T)))
    pk = predict_known(T, F, vals, extrap=extrap)
    ax[1].step(CTR, F[jn], where="mid", lw=2.0, color=MC_COLORS[0],
               label=f"nominal uniform {T[jn]:.2f} ke$^-$ sim")
    ax[1].step(CTR, pk, where="mid", lw=2.0, color="#0072B2",
               label="known-map mixture (real per-pixel thr)")
    ax[1].set_title("known-map prediction vs nominal")
    ax[1].set_xlabel("hit Q [ke$^-$]"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25, lw=0.6); ax[1].set_xlim(0, 50)
    f.tight_layout(); f.savefig(outpath, dpi=130); plt.close(f)
    print(f"  wrote {outpath}")


def _q_panel(ax, d, pk, mfit, title, pk_sim=None, sim_r=None, prc=None):
    """shared LEFT-panel style: data vs the MC curves. Two of them are the SAME nominal thresholds
    produced two ways -- pk_sim (optional) is mkramer's FULL larnd-sim MC, labeled by its sim config
    'sim r<R> prc<P>' (consistent with the other Q plots); pk is our mono-threshold basis interpolated
    +mixed by that same map (predict_known). Their agreement is a basis-closure check. mfit = the fit.
    The Q axis spans the full [0,50] ke- used by the other Q plots (fsdcube_hitq_datamc.py)."""
    ax.step(CTR, d, where="mid", lw=2.2, color=DATA_COLOR, label="data")
    if pk_sim is not None:
        lab = "MC - sim" + (f" r{sim_r}" if sim_r is not None else "") + (f" prc{prc}" if prc is not None else "")
        ax.step(CTR, pk_sim, where="mid", lw=1.8, color="#009E73", label=lab)
    ax.step(CTR, pk, where="mid", lw=1.8, ls="--", color="#0072B2", label="MC - Nominal Map (interp)")
    ax.step(CTR, mfit, where="mid", lw=2.0, color="#D55E00", label="MC - Threshold Fit")
    ax.set_title(title)
    ax.set_xlabel("hit Q [ke$^-$]"); ax.set_ylabel("density"); ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.set_xlim(0, 50)


def _thr_panel(ax, Tf, w, vals):
    """shared 'Threshold Distributions' panel: the real per-pixel map vs the fitted w(T) as density
    step histograms on the SAME binning -- the map is histogrammed into bins CENTERED ON THE FIT NODES
    Tf (width dTf), so both curves use identical bins (apples-to-apples). No band/extrapolation shading
    (the basis spans the whole plotted range) and no on-plot stats -- those move to the summary txt."""
    dTf = float(Tf[1] - Tf[0]); tlo, thi = float(Tf.min()), float(Tf.max())
    edges = np.concatenate([Tf - dTf / 2.0, [Tf[-1] + dTf / 2.0]])   # bins centered on the fit nodes
    inr = vals[(vals >= edges[0]) & (vals <= edges[-1])]
    hk, _ = np.histogram(inr, bins=edges, density=True)
    ax.step(Tf, hk, where="mid", lw=2.0, color="#0072B2", label="Nominal Map")
    ax.step(Tf, w / dTf, where="mid", lw=2.0, color="#D55E00", label="Threshold Fit")
    ax.set_title("Threshold Distributions")
    ax.set_xlabel("threshold [ke$^-$]"); ax.set_ylabel("pixel density [1/ke$^-$]")
    ax.set_xlim(tlo, thi); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25, lw=0.6)


def plot_fit(T, F, Tf, w_fit, d, vals, prc, outpath, extrap=False, mc_d=None, sim_r=None):
    """per-prc 2-panel fit plot: LEFT 'prc<P>' (data vs the full-sim, nominal-map & threshold-fit
    curves), RIGHT 'Threshold Distributions' (real per-pixel map vs fitted w(T)). Numbers -> summary."""
    f, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    pk = predict_known(T, F, vals, extrap=extrap)
    mfit = mixture(interp_basis(T, F, Tf, extrap=extrap), w_fit)
    _q_panel(ax[0], d, pk, mfit, f"prc{prc}", pk_sim=mc_d, sim_r=sim_r, prc=prc)
    _thr_panel(ax[1], Tf, w_fit, vals)
    f.tight_layout(); f.savefig(outpath, dpi=130); plt.close(f)
    print(f"  wrote {outpath}")


def plot_joint(specs, Tf, vals, w_joint, outpath, extrap=False):
    """joint-fit plot: one Q-density panel per fitted spectrum (data vs nominal map & the SHARED
    threshold-fit mixture), then a final 'Threshold Distributions' panel with the one shared w(T). The
    spectra are whatever the joint fit spanned -- one per prc today, one per (prc x observable) once the
    whole Q-plot suite is opted in -- so this panel row grows with the suite automatically."""
    n = len(specs)
    f, ax = plt.subplots(1, n + 1, figsize=(6.0 * (n + 1), 4.4))
    if n == 0:
        ax = [ax]
    for i, c in enumerate(specs):
        pk = predict_known(c["T"], c["F"], vals, extrap=extrap)
        mfit = mixture(c["Ff"], w_joint)
        _q_panel(ax[i], c["d"], pk, mfit, c["label"], pk_sim=c.get("mc_d"),
                 sim_r=c.get("sim_r"), prc=c.get("prc"))
    _thr_panel(ax[n], Tf, w_joint, vals)
    f.tight_layout(); f.savefig(outpath, dpi=130); plt.close(f)
    print(f"  wrote {outpath}")


def write_summary(path, args, specs, joint, vals):
    """the method explanation + the per-fit numbers that used to be annotated on the plots. `specs` is
    the list of fitted spectra (one per prc today; one per prc x observable once the suite is opted in)."""
    L = []
    Tf0 = specs[0]["Tf"]; n_fit = len(Tf0); binw_fit = float(Tf0[1] - Tf0[0])
    obs_list = list(dict.fromkeys(c["obs"] for c in specs))
    def A(s=""): L.append(s)
    A("FSDCube threshold-mixture fit -- method and results")
    A("=" * 52); A()
    A("WHAT THE FIT DOES")
    A("  The data single-hit-pixel charge spectrum D(Q) is modeled as a mixture over the per-pixel")
    A("  discriminator thresholds:  D(Q) = sum_T w(T) f_T(Q).  f_T(Q) is the single-hit charge")
    A("  density the simulation produces when every pixel has threshold T (its turn-on edge sits at")
    A("  ~T). w(T) >= 0, sum_T w(T) = 1 is the distribution of thresholds across pixels that the data")
    A("  imply. Everything is shape only (area-normalized to 1): this is about the spectrum SHAPE,")
    A("  not absolute rates.")
    A()
    A("BASIS  f_T(Q)")
    A("  Taken from the simulation grid (tests/threshold_induction_study.py --mult-grid) at the reset")
    A("  node matching each prc (prc2 -> 128 cyc, prc16 -> 1024 cyc; prc is a periodic-reset window,")
    A("  which is why the two spectra differ). The grid simulates thresholds at 1.25 ke- spacing;")
    A(f"  intermediate thresholds are linearly interpolated per Q-bin, and the thin residual of pixels")
    A(f"  outside [{args.thr_lo:.2f}, {args.thr_hi:.2f}] ke- (the real map runs to ~20 ke-) is "
      + ("linearly extrapolated." if args.extrapolate else "clamped (extrapolation off)."))
    A()
    A("FITTING TECHNIQUE")
    A("  Non-negative least squares on the probability simplex:")
    A("      argmin_w  || sum_T w(T) f_T(Q) - D(Q) ||^2"
      + ("  +  reg * || 2nd-difference of w ||^2" if args.reg > 0 else ""))
    A("      subject to  w(T) >= 0,  sum_T w(T) = 1.")
    A(f"  Solved by projected-gradient descent -- a gradient step then Euclidean projection onto the")
    A(f"  simplex (Duchi et al. 2008) -- on a grid of {n_fit} threshold nodes ({binw_fit:.3f} ke- apart)")
    A(f"  over [{args.thr_lo:.2f}, {args.thr_hi:.2f}] ke-. The nominal-map histogram uses this SAME binning, so")
    A(f"  the two threshold distributions on the plot are directly comparable.")
    A("  The simplex constraint makes sum_T w(T) f_T(Q) itself a normalized density. Pure numpy (no")
    A("  scipy, no GPU); the fit is a few ms and runs locally from the saved data_density_prc*.npz.")
    if args.reg > 0:
        A(f"  A 2nd-difference penalty (reg={args.reg:g}) lightly smooths w(T).")
    else:
        A("  NO smoothness penalty is applied (reg=0): the real per-pixel map is DAC-quantized and not")
        A("  smooth, so smoothing would fight real structure. Without it the fine-grid fit is under-")
        A("  determined (the few simulated shapes are blended linearly), so the data fix w(T) only")
        A("  where they have leverage; elsewhere w(T) stays at the seed.")
    if args.seed_map:
        A("  The iteration is SEEDED at the known per-pixel threshold map (the simulated distribution),")
        A("  so with no smoothness the fit = the sim map deformed only as much as the data require.")
        if args.prior > 0:
            A(f"  --prior={args.prior:g} additionally adds (prior/2)||w-seed||^2, a real pull toward the sim.")
    else:
        A("  Seeding from the sim map is DISABLED (uniform start).")
    A()
    A("THE MC CURVES ON THE Q PANEL  (two of them are the SAME nominal thresholds, produced two ways)")
    A(f"  MC - sim r{args.sim_r} prc<P>    : mkramer's FULL larnd-sim MC at this prc and readout radius r{args.sim_r},")
    A("                             which already runs with the real per-pixel threshold map loaded.")
    A("                             End-to-end sim of the nominal thresholds -- the ground-truth MC.")
    A("  MC - Nominal Map (interp): the SAME real per-pixel map, but RECONSTRUCTED from our mono-threshold")
    A("                             basis: predict_known(Q) = < f_{T_pix}(Q) > with f_T interpolated in T.")
    A("                             If this matches the full sim, the basis + interpolation faithfully")
    A("                             represent it (a basis-closure check) -- so the fit's departure from")
    A("                             them is a real preference of the DATA, not a basis artifact. (The")
    A("                             self-test has no MC, so only this curve is drawn.)")
    A("  MC - Threshold Fit       : the fitted threshold mixture sum_T w(T) f_T(Q) -- what the data prefer.")
    A()
    A("WHAT THE NUMBERS MEAN  (mean +/- standard deviation of a distribution, NOT its peak/mode)")
    A("  Nominal Map  m +/- s : mean and width of the REAL per-pixel threshold map")
    A("                         (larndsim/bin/thresholds_fsdcube.npz) over the fit range -- the")
    A("                         threshold distribution the simulation actually used.")
    A("  Threshold Fit m +/- s: mean and width of the FITTED distribution w(T) -- what the data prefer.")
    A("  data residual        : L2 norm of (fit mixture - data) over the Q axis; smaller = the mixture")
    A("                         reproduces the measured spectrum better.")
    A()
    A(f"RESULTS -- independent per-spectrum fits  (observable(s): {', '.join(obs_list)})")
    A("-" * 36)
    for c in specs:
        s = c["stats"]
        A(f"  {c['label']}  (reset {c['use_rst']} cyc, n_data={c['ndat']}):")
        A(f"      Nominal Map   : {s['map_m']:.2f} +/- {s['map_s']:.2f} ke-")
        A(f"      Threshold Fit : {s['fit_m']:.2f} +/- {s['fit_s']:.2f} ke-")
        A(f"      data residual (fit vs data)        : {s['resid']:.3e}")
        if "mc_resid" in s:
            A(f"      sim r{c['sim_r']} vs data                  : {s['mc_resid']:.3e}")
            A(f"      basis closure (interp vs sim)      : {s['basis_resid']:.3e}  (small = basis reproduces the full sim)")
        A(f"      pixels outside the simulated nodes [{s['grid_lo']:.2f},{s['grid_hi']:.2f}] "
          f"(extrapolated): {100*s['frac_extrap']:.1f}%")
    A()
    if joint:
        A("RESULTS -- joint fit (ONE shared threshold distribution across all spectra)")
        A("-" * 36)
        A("  The per-pixel thresholds are fixed hardware; only the reset window (prc) changes, and the")
        A("  choice of Q population (observable) is just a different view of the same hits. So a SINGLE")
        A("  w(T) should explain ALL the spectra at once. The joint fit stacks their Q-axes and solves")
        A("  argmin_w sum_k || sum_T w(T) f_T^k(Q) - D_k(Q) ||^2  on the same simplex. Requiring one")
        A("  threshold distribution to match every condition adds leverage and breaks degeneracies that")
        A("  any single spectrum alone would allow. (Today the spectra are the per-prc single-hit Q; the")
        A("  SAME machinery fits the whole suite of Q plots once more observables are opted in.)")
        A(f"      Threshold Fit (joint) : {joint['fit_m']:.2f} +/- {joint['fit_s']:.2f} ke-")
        for lab in joint["labels"]:
            A(f"      data residual {lab}  : {joint['resids'][lab]:.3e}")
        A()
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
def parse_grid(tokens, prcs):
    """'prc=path' tokens (or bare paths aligned with --prc) -> {prc: path}. Paths are expanduser'd so a
    'prc=~/...' token works even though the shell does not expand a mid-word ~ (only a leading one)."""
    exp = os.path.expanduser
    if all("=" in t and t.split("=", 1)[0].strip().lstrip("-").isdigit() for t in tokens):
        m = {int(t.split("=", 1)[0]): exp(t.split("=", 1)[1]) for t in tokens}
        return {p: m[p] for p in prcs if p in m}
    if len(tokens) == 1:
        return {p: exp(tokens[0]) for p in prcs}
    if len(tokens) == len(prcs):
        return {p: exp(tokens[i]) for i, p in enumerate(prcs)}
    raise SystemExit(f"--grid: {len(tokens)} paths for {len(prcs)} prc; pass 1, {len(prcs)}, or 'prc=path'.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs="+", required=True,
                    help="ti_mult_grid.npz basis per prc: 'prc=path' tokens or bare paths aligned with --prc")
    ap.add_argument("--prc", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--observable", choices=["single", "hit"], nargs="+", default=["single"],
                    help="which Q population(s) to fit: single = single-hit-pixel Q (default, sharpest "
                         "turn-on); hit = every hit's Q. Passing SEVERAL opts the whole suite into the "
                         "JOINT fit (one shared w(T) across all prc x observable spectra); the first is "
                         "the 'primary' used for the per-prc 2-panel plots and the legacy npz keys.")
    ap.add_argument("--reset-per-prc", action="store_true",
                    help="take each prc's basis at the reset node MATCHING its reset window (cycles = "
                         "round(prc * --prc-step-us / grid ts); prc2 -> 128, prc16 -> 1024) instead of the "
                         "grid base. prc is a reset window, so this is what makes prc2 and prc16 differ.")
    ap.add_argument("--prc-step-us", type=float, default=6.4,
                    help="microseconds per prc step for --reset-per-prc (default 6.4; prc8 = 51.2 us base)")
    ap.add_argument("--thr-binw", type=float, default=0.25,
                    help="threshold bin width in ke- for BOTH the fit grid and the nominal-map histogram "
                         "(same binning -> apples-to-apples; default 0.25). Sets the fit node spacing.")
    ap.add_argument("--n-fine", type=int, default=None,
                    help="override: use exactly this many fit nodes across [--thr-lo,--thr-hi] instead of "
                         "deriving the count from --thr-binw")
    ap.add_argument("--thr-lo", type=float, default=1.25,
                    help="low edge of the fit's threshold range in ke- (default 1.25 = the bottom SIMULATED "
                         "grid node once the basis is grown to span [1.25,15])")
    ap.add_argument("--thr-hi", type=float, default=15.0,
                    help="high edge of the fit's threshold range in ke- (default 15.0 = the top simulated "
                         "node once the basis is grown to [1.25,15]; the thin tail to 20 ke- extrapolates)")
    ap.add_argument("--no-extrapolate", dest="extrapolate", action="store_false",
                    help="disable the linear-spline extrapolation and clamp thresholds outside the simulated "
                         "node range to the edge shape. With a basis grown to [1.25,15] the fit range is "
                         "all interpolation, so extrapolation then only fills the small <lo / >hi residual.")
    ap.add_argument("--no-seed-map", dest="seed_map", action="store_false",
                    help="start the fit from a uniform threshold distribution instead of seeding it at the "
                         "known per-pixel map (the simulated distribution)")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="strength of a (prior/2)||w-seed||^2 pull toward the sim seed (default 0 = seed is "
                         "only the initial condition; >0 makes the fit stay near the sim unless data demands)")
    ap.add_argument("--reg", type=float, default=0.0,
                    help="2nd-difference smoothness penalty on w(T) (default 0 = OFF: the real per-pixel "
                         "map is DAC-quantized / not smooth, so smoothing would fight real structure)")
    ap.add_argument("--no-sim", dest="use_sim", action="store_false",
                    help="do not read/overlay mkramer's full MC (the 'Kramer sim') when reading FLOW")
    ap.set_defaults(extrapolate=True, seed_map=True, use_sim=True)
    ap.add_argument("--thr-map", default=THR_MAP_DEF, help="per-pixel threshold npz (keys/values/default)")
    ap.add_argument("--outdir", default="tfit")
    ap.add_argument("--self-test", action="store_true",
                    help="no data: validate the machinery (delta + known-mixture recovery) and plot the "
                         "known-map prediction, using each grid as its own basis")
    ap.add_argument("--data-npz", default=None,
                    help="re-fit saved data densities instead of reading FLOW. A FILE is used for the single "
                         "prc; a DIRECTORY is searched for data_density_prc<P>.npz per prc (so the joint fit "
                         "gets all prcs). These are written by a FLOW run, so you can re-fit fully locally.")
    # FLOW reader knobs (NERSC data path), mirrored from fsdcube_hitq_datamc.py
    ap.add_argument("--data-base", default=None)
    ap.add_argument("--sim-base", default=None,
                    help="base dir of mkramer's MC FLOW (the 'Kramer sim' overlay); default = the "
                         "plotter's SIM_BASE_DEF")
    ap.add_argument("--sim-r", type=int, default=3,
                    help="ONE sim readout radius for the Kramer-sim overlay curve (default 3)")
    ap.add_argument("--sim-var", default="", help="sim variant suffix for the Kramer-sim files, e.g. _noFFE")
    ap.add_argument("--adc-min", type=float, default=0.2e6)
    ap.add_argument("--adc-max", type=float, default=0.5e6)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()
    args.outdir = os.path.expanduser(args.outdir)            # tolerate a leading ~ the shell may not expand
    if args.data_npz:
        args.data_npz = os.path.expanduser(args.data_npz)
    if args.sim_base is None:                                # default the Kramer-sim base to the plotter's
        from fsdcube_hitq_datamc import SIM_BASE_DEF
        args.sim_base = SIM_BASE_DEF
    else:
        args.sim_base = os.path.expanduser(args.sim_base)
    # fit node count: explicit --n-fine, else derived from the bin width so map & fit share one binning.
    n_fine = args.n_fine or max(int(round((args.thr_hi - args.thr_lo) / args.thr_binw)) + 1, 2)
    os.makedirs(args.outdir, exist_ok=True)
    grids = parse_grid(args.grid, args.prc)
    vals_all = map_thresholds(args.thr_map)
    print(f"threshold map: {vals_all.size} pixels, median {np.median(vals_all):.2f} ke-, "
          f"in-basis-range fraction set per prc below")

    collected = {}  # label -> spec dict (prc x observable); the joint fit stacks ALL of them
    primary = args.observable[0]
    for prc in args.prc:
        if prc not in grids:
            print(f"prc{prc}: no --grid basis, skipping"); continue

        # basis per requested observable (same grid, sliced per observable); the reset is obs-independent
        bases = {}
        for obs in args.observable:
            T, F, Y, base_T, use_rst = load_basis(grids[prc], obs, prc=prc,
                                                  prc_step_us=args.prc_step_us, reset_per_prc=args.reset_per_prc)
            Tf, Ff = fine_grid(T, F, n_fine, args.thr_lo, args.thr_hi, extrap=args.extrapolate)
            bases[obs] = dict(T=T, F=F, Y=Y, base_T=base_T, use_rst=use_rst, Tf=Tf, Ff=Ff)
        use_rst = bases[primary]["use_rst"]; Tp = bases[primary]["T"]
        frac_in = float(((vals_all >= Tp.min()) & (vals_all <= Tp.max())).mean())
        frac_fit = float(((vals_all >= args.thr_lo) & (vals_all <= args.thr_hi)).mean())
        print(f"\nprc{prc}: basis T={Tp} ke- (base {bases[primary]['base_T']:.2f}), reset {use_rst} cyc, "
              f"obs={'+'.join(args.observable)}; fit range [{args.thr_lo:.1f},{args.thr_hi:.1f}] ke- "
              f"({'extrap' if args.extrapolate else 'clamped'}, seed={'sim-map' if args.seed_map else 'uniform'}, "
              f"reg={args.reg:g}{', prior '+str(args.prior) if args.prior>0 else ''}); "
              f"{100*frac_in:.1f}% of pixels in grid [{Tp.min():.2f},{Tp.max():.2f}], {100*frac_fit:.1f}% in fit range")
        plot_basis(bases[primary]["T"], bases[primary]["F"], bases[primary]["base_T"], vals_all,
                   os.path.join(args.outdir, f"thrfit_basis_prc{prc}.png"), primary, use_rst, extrap=args.extrapolate)

        # data (+ Kramer-sim MC) densities for ALL requested observables -- one FLOW read per prc
        if args.self_test:
            # machinery checks on the primary NATIVE basis: delta + hand-set-mixture recovery
            T, F = bases[primary]["T"], bases[primary]["F"]
            jn = int(np.argmin(np.abs(T - bases[primary]["base_T"])))
            w = fit_weights(F, F[jn])
            print(f"  self-test delta @ {T[jn]:.2f} ke-: recovered weights = {np.round(w, 3)}  "
                  f"(argmax {T[int(np.argmax(w))]:.2f} ke-)")
            k = np.arange(len(T)); w_true = np.exp(-0.5 * ((k - k.mean()) / max(len(T) / 4.0, 1.0)) ** 2)
            w_true = w_true / w_true.sum()                      # a smooth bump over however many nodes
            w_rec = fit_weights(F, mixture(F, w_true))
            print(f"  self-test mixture: true {np.round(w_true,3)} -> recovered {np.round(w_rec,3)}  "
                  f"(max abs err {np.max(np.abs(w_true-w_rec)):.3f})")
            dens = {obs: (predict_known(bases[obs]["T"], bases[obs]["F"], vals_all, extrap=args.extrapolate), 0)
                    for obs in args.observable}
            mc = {obs: None for obs in args.observable}
        elif args.data_npz:
            npz = (os.path.join(args.data_npz, f"data_density_prc{prc}.npz")
                   if os.path.isdir(args.data_npz) else args.data_npz)
            dd = np.load(npz)
            dens, mc = load_densities(dd, args.observable, primary)
            print(f"  loaded data density from {npz} (obs={'+'.join(dens)}, n={dens[primary][1]}"
                  f"{', +sim' if mc.get(primary) is not None else ', no sim in npz'})")
        else:
            if not args.data_base:
                from fsdcube_hitq_datamc import DATA_BASE_DEF
                args.data_base = DATA_BASE_DEF
            dens = data_densities_from_flow(args, prc, args.observable)
            mc = (mc_densities_from_flow(args, prc, args.observable) if args.use_sim
                  else {obs: None for obs in args.observable})
            save_densities(os.path.join(args.outdir, f"data_density_prc{prc}.npz"),
                           dens, mc, args.observable, primary, reset=use_rst)

        # fit each spectrum; the per-prc 2-panel plot is drawn for the PRIMARY observable only
        for obs in args.observable:
            if obs not in dens:
                continue                              # requested obs absent from an older npz (warned above)
            b = bases[obs]; T, F, Tf, Ff = b["T"], b["F"], b["Tf"], b["Ff"]
            w0 = seed_from_map(Tf, vals_all) if args.seed_map else None
            d, ndat = dens[obs]; mc_d = mc.get(obs)
            w_fit = fit_weights(Ff, d, reg=args.reg, w0=w0, prior=args.prior)
            label = f"prc{prc}" if len(args.observable) == 1 else f"prc{prc} {obs}"
            if obs == primary:
                out_png = f"thrfit_selftest_prc{prc}.png" if args.self_test else f"thrfit_prc{prc}.png"
                plot_fit(T, F, Tf, w_fit, d, vals_all, prc, os.path.join(args.outdir, out_png),
                         extrap=args.extrapolate, mc_d=mc_d, sim_r=args.sim_r)
            st = fit_stats(T, Tf, Ff, d, w_fit, vals_all, args.thr_lo, args.thr_hi)
            if mc_d is not None:   # full-sim vs data, and basis-closure (nominal-map interp vs full sim)
                pk = predict_known(T, F, vals_all, extrap=args.extrapolate)
                st["mc_resid"] = float(np.sqrt(((mc_d - d) ** 2).sum()) * BINW)
                st["basis_resid"] = float(np.sqrt(((pk - mc_d) ** 2).sum()) * BINW)
            print(f"  {label} (reset {use_rst} cyc): fit {st['fit_m']:.2f}+/-{st['fit_s']:.2f} ke- vs "
                  f"map {st['map_m']:.2f}+/-{st['map_s']:.2f} (resid {st['resid']:.2e}, n_data={ndat})"
                  + (f"; sim r{args.sim_r} vs data {st['mc_resid']:.2e}, basis-closure {st['basis_resid']:.2e}"
                     if mc_d is not None else ""))
            collected[label] = dict(label=label, prc=prc, obs=obs, sim_r=args.sim_r,
                                    T=T, F=F, Tf=Tf, Ff=Ff, d=d, w_fit=w_fit, use_rst=use_rst,
                                    ndat=ndat, stats=st, mc_d=mc_d, primary=(obs == primary))

    # JOINT fit: one shared w(T) across ALL collected spectra (prc x observable; shared hardware thresholds)
    joint = None
    specs = list(collected.values())
    if len(specs) >= 2:
        Tf = specs[0]["Tf"]
        w0 = seed_from_map(Tf, vals_all) if args.seed_map else None
        w_j = fit_joint([c["Ff"] for c in specs], [c["d"] for c in specs],
                        reg=args.reg, w0=w0, prior=args.prior)
        tag = "thrfit_joint_selftest.png" if args.self_test else "thrfit_joint.png"
        plot_joint(specs, Tf, vals_all, w_j, os.path.join(args.outdir, tag), extrap=args.extrapolate)
        jm, js = moments(Tf, w_j)
        resids = {c["label"]: float(np.sqrt(((mixture(c["Ff"], w_j) - c["d"]) ** 2).sum()) * BINW)
                  for c in specs}
        joint = dict(labels=[c["label"] for c in specs], w=w_j, fit_m=float(jm), fit_s=float(js), resids=resids)
        print(f"  joint fit ({'+'.join(c['label'] for c in specs)}): shared thresholds "
              f"{jm:.2f}+/-{js:.2f} ke- (resid " + ", ".join(f"{c['label']} {resids[c['label']]:.2e}" for c in specs) + ")")

    if specs:
        write_summary(os.path.join(args.outdir, "thrfit_summary.txt"), args, specs, joint, vals_all)
    print(f"\nwrote outputs -> {args.outdir}/")


if __name__ == "__main__":
    main()
