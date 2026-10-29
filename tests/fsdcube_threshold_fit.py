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


# ---------------------------------------------------------------------------
# target density (data): from FLOW on NERSC, from a saved npz, or self-test pseudo-data
# ---------------------------------------------------------------------------
def data_density_from_flow(args, prc):
    """DATA single-hit (or all-hit) charge density, read from FLOW exactly like the plotter."""
    from fsdcube_hitq_datamc import accumulate_q, data_files
    files = list(data_files(args.data_base, prc))
    if args.max_files:
        files = files[:args.max_files]
    print(f"prc{prc}: reading {len(files)} data files ...")
    tup = accumulate_q(files, args.adc_min, args.adc_max, args.max_events)
    q, n = np.asarray(tup[0], float), np.asarray(tup[1])
    x = q[n == 1] if args.observable == "single" else q
    return density(x), int(x.size)


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
    ax[0].grid(alpha=0.25, lw=0.6); ax[0].set_xlim(0, 30)

    jn = int(np.argmin(np.abs(T - base_T)))
    pk = predict_known(T, F, vals, extrap=extrap)
    ax[1].step(CTR, F[jn], where="mid", lw=2.0, color=MC_COLORS[0],
               label=f"nominal uniform {T[jn]:.2f} ke$^-$ sim")
    ax[1].step(CTR, pk, where="mid", lw=2.0, color="#0072B2",
               label="known-map mixture (real per-pixel thr)")
    ax[1].set_title("known-map prediction vs nominal")
    ax[1].set_xlabel("hit Q [ke$^-$]"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25, lw=0.6); ax[1].set_xlim(0, 30)
    f.tight_layout(); f.savefig(outpath, dpi=130); plt.close(f)
    print(f"  wrote {outpath}")


def plot_fit(T, F, Tf, w_fit, d, vals, prc, use_rst, outpath, observable, dlabel="data", extrap=False):
    """LEFT: target density vs nominal / known-map / best-fit mixtures. RIGHT: the per-pixel threshold
    distribution as density STEP histograms -- the real map (broad, all pixels) vs the best fit on a
    fine threshold grid spanning the FULL range (so the fit is a continuous distribution, not weight on
    discrete nodes). The grid-backed sub-range [T_lo, T_hi] is shaded; the wings are extrapolated."""
    f, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    nom = F[int(np.argmin(np.abs(T - 5.0)))]
    pk = predict_known(T, F, vals, extrap=extrap)
    mfit = mixture(interp_basis(T, F, Tf, extrap=extrap), w_fit)
    kind = "single-hit" if observable == "single" else "hit"
    ax[0].step(CTR, d, where="mid", lw=2.2, color=DATA_COLOR, label=dlabel)
    ax[0].step(CTR, nom, where="mid", lw=1.5, ls=":", color=MC_COLORS[0], label="nominal 5 ke$^-$ sim")
    ax[0].step(CTR, pk, where="mid", lw=1.6, ls="--", color="#0072B2", label="known-map mixture")
    ax[0].step(CTR, mfit, where="mid", lw=2.0, color="#D55E00", label="best-fit mixture")
    ax[0].set_title(f"prc{prc}: {kind} Q density [reset {use_rst} cyc]")
    ax[0].set_xlabel("hit Q [ke$^-$]"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25, lw=0.6); ax[0].set_xlim(0, 30)

    # RIGHT: threshold distribution over the FULL fit range, density step histograms
    dTf = float(Tf[1] - Tf[0])
    tlo, thi = float(Tf.min()), float(Tf.max())
    tb = np.arange(np.floor(tlo), np.ceil(thi) + 0.25, 0.25); tc = 0.5 * (tb[1:] + tb[:-1])
    inr = vals[(vals >= tlo) & (vals <= thi)]           # map pixels inside the modelled range
    hk, _ = np.histogram(inr, bins=tb, density=True)
    ax[1].axvspan(float(T.min()), float(T.max()), color="0.85", alpha=0.5, zorder=0,
                  label="grid-backed (interp)")
    ax[1].step(tc, hk, where="mid", lw=2.0, color="#0072B2", label="known map (all pixels)")
    ax[1].step(Tf, w_fit / dTf, where="mid", lw=2.0, color="#D55E00", label=f"best fit to {dlabel}")
    fit_m, fit_s = moments(Tf, w_fit)
    frac_extrap = float(((vals < T.min()) | (vals > T.max())).mean())
    ax[1].set_title(f"prc{prc}: threshold distribution")
    ax[1].set_xlabel("threshold [ke$^-$]"); ax[1].set_ylabel("pixel density [1/ke$^-$]")
    ax[1].set_xlim(tlo, thi); ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=0.25, lw=0.6)
    ax[1].annotate(f"map: {inr.mean():.2f} $\\pm$ {inr.std():.2f} ke$^-$\n"
                   f"best fit: {fit_m:.2f} $\\pm$ {fit_s:.2f} ke$^-$\n"
                   f"{100 * frac_extrap:.0f}% of pixels extrapolated (outside grid)",
                   xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top", fontsize=8,
                   bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    f.tight_layout(); f.savefig(outpath, dpi=130); plt.close(f)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
def parse_grid(tokens, prcs):
    """'prc=path' tokens (or bare paths aligned with --prc) -> {prc: path}."""
    if all("=" in t and t.split("=", 1)[0].strip().lstrip("-").isdigit() for t in tokens):
        m = {int(t.split("=", 1)[0]): t.split("=", 1)[1] for t in tokens}
        return {p: m[p] for p in prcs if p in m}
    if len(tokens) == 1:
        return {p: tokens[0] for p in prcs}
    if len(tokens) == len(prcs):
        return {p: tokens[i] for i, p in enumerate(prcs)}
    raise SystemExit(f"--grid: {len(tokens)} paths for {len(prcs)} prc; pass 1, {len(prcs)}, or 'prc=path'.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", nargs="+", required=True,
                    help="ti_mult_grid.npz basis per prc: 'prc=path' tokens or bare paths aligned with --prc")
    ap.add_argument("--prc", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--observable", choices=["single", "hit"], default="single",
                    help="single = single-hit-pixel Q (default, sharpest turn-on); hit = every hit's Q")
    ap.add_argument("--reset-per-prc", action="store_true",
                    help="take each prc's basis at the reset node MATCHING its reset window (cycles = "
                         "round(prc * --prc-step-us / grid ts); prc2 -> 128, prc16 -> 1024) instead of the "
                         "grid base. prc is a reset window, so this is what makes prc2 and prc16 differ.")
    ap.add_argument("--prc-step-us", type=float, default=6.4,
                    help="microseconds per prc step for --reset-per-prc (default 6.4; prc8 = 51.2 us base)")
    ap.add_argument("--n-fine", type=int, default=81,
                    help="number of fine threshold nodes spanned across [--thr-lo, --thr-hi] for the fit, so "
                         "the fitted threshold distribution is continuous/broad (default 81 -> ~0.125 ke step)")
    ap.add_argument("--thr-lo", type=float, default=1.25,
                    help="low edge of the fit's threshold range in ke- (default 1.25 = the bottom SIMULATED "
                         "grid node once the basis is grown to span [1.25,11.25])")
    ap.add_argument("--thr-hi", type=float, default=11.25,
                    help="high edge of the fit's threshold range in ke- (default 11.25 = the top simulated "
                         "node; covers ~99%% of the map, the thin tail to 20 ke- folds into the top node)")
    ap.add_argument("--no-extrapolate", dest="extrapolate", action="store_false",
                    help="disable the linear-spline extrapolation and clamp thresholds outside the simulated "
                         "node range to the edge shape. With a basis grown to [1.25,11.25] the fit range is "
                         "all interpolation, so extrapolation then only fills the small <lo / >hi residual.")
    ap.add_argument("--no-seed-map", dest="seed_map", action="store_false",
                    help="start the fit from a uniform threshold distribution instead of seeding it at the "
                         "known per-pixel map (the simulated distribution)")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="strength of a (prior/2)||w-seed||^2 pull toward the sim seed (default 0 = seed is "
                         "only the initial condition; >0 makes the fit stay near the sim unless data demands)")
    ap.add_argument("--reg", type=float, default=1e-3,
                    help="2nd-difference smoothness penalty for the fine-grid fit (default 1e-3)")
    ap.set_defaults(extrapolate=True, seed_map=True)
    ap.add_argument("--thr-map", default=THR_MAP_DEF, help="per-pixel threshold npz (keys/values/default)")
    ap.add_argument("--outdir", default="tfit")
    ap.add_argument("--self-test", action="store_true",
                    help="no data: validate the machinery (delta + known-mixture recovery) and plot the "
                         "known-map prediction, using each grid as its own basis")
    ap.add_argument("--data-npz", default=None,
                    help="re-fit a saved data density (CTR + density arrays) instead of reading FLOW")
    # FLOW reader knobs (NERSC data path), mirrored from fsdcube_hitq_datamc.py
    ap.add_argument("--data-base", default=None)
    ap.add_argument("--adc-min", type=float, default=0.2e6)
    ap.add_argument("--adc-max", type=float, default=0.5e6)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    grids = parse_grid(args.grid, args.prc)
    vals_all = map_thresholds(args.thr_map)
    print(f"threshold map: {vals_all.size} pixels, median {np.median(vals_all):.2f} ke-, "
          f"in-basis-range fraction set per prc below")

    for prc in args.prc:
        if prc not in grids:
            print(f"prc{prc}: no --grid basis, skipping"); continue
        T, F, Y, base_T, use_rst = load_basis(grids[prc], args.observable, prc=prc,
                                              prc_step_us=args.prc_step_us, reset_per_prc=args.reset_per_prc)
        Tf, Ff = fine_grid(T, F, args.n_fine, args.thr_lo, args.thr_hi, extrap=args.extrapolate)
        w0 = seed_from_map(Tf, vals_all) if args.seed_map else None
        frac_in = float(((vals_all >= T.min()) & (vals_all <= T.max())).mean())
        frac_fit = float(((vals_all >= args.thr_lo) & (vals_all <= args.thr_hi)).mean())
        print(f"\nprc{prc}: basis T={T} ke- (base {base_T:.2f}), reset {use_rst} cyc, "
              f"yield/shower={np.round(Y,1)}; fit range [{args.thr_lo:.1f},{args.thr_hi:.1f}] ke- "
              f"({'extrap' if args.extrapolate else 'clamped'}, seed={'sim-map' if args.seed_map else 'uniform'}"
              f"{', prior '+str(args.prior) if args.prior>0 else ''}); {100*frac_in:.1f}% of pixels in grid "
              f"[{T.min():.2f},{T.max():.2f}], {100*frac_fit:.1f}% in fit range")
        plot_basis(T, F, base_T, vals_all, os.path.join(args.outdir, f"thrfit_basis_prc{prc}.png"),
                   args.observable, use_rst, extrap=args.extrapolate)

        if args.self_test:
            # (1) delta recovery: pseudo-data = the nominal node -> weights should concentrate there.
            jn = int(np.argmin(np.abs(T - base_T)))
            w = fit_weights(F, F[jn])
            print(f"  self-test delta @ {T[jn]:.2f} ke-: recovered weights = {np.round(w, 3)}  "
                  f"(argmax {T[int(np.argmax(w))]:.2f} ke-)")
            # (2) known-mixture recovery: pseudo-data = a hand-set mixture -> recover its weights.
            w_true = np.array([0.1, 0.2, 0.4, 0.2, 0.1])[:len(T)]; w_true = w_true / w_true.sum()
            w_rec = fit_weights(F, mixture(F, w_true))
            print(f"  self-test mixture: true {np.round(w_true,3)} -> recovered {np.round(w_rec,3)}  "
                  f"(max abs err {np.max(np.abs(w_true-w_rec)):.3f})")
            # (3) the physics closure: fit the known-map prediction on the fine grid -> should recover
            #     the real in-range threshold spread (mean/sd), continuous (not spiked on nodes).
            pk = predict_known(T, F, vals_all, extrap=args.extrapolate)
            w_k = fit_weights(Ff, pk, reg=args.reg, w0=w0, prior=args.prior)
            plot_fit(T, F, Tf, w_k, pk, vals_all, prc, use_rst,
                     os.path.join(args.outdir, f"thrfit_selftest_prc{prc}.png"),
                     args.observable, dlabel="known-map mixture (pseudo-data)", extrap=args.extrapolate)
            inr = vals_all[(vals_all >= args.thr_lo) & (vals_all <= args.thr_hi)]
            print(f"  closure: map in fit range {inr.mean():.2f}+/-{inr.std():.2f} ke- vs "
                  f"fit-of-known {moments(Tf, w_k)[0]:.2f}+/-{moments(Tf, w_k)[1]:.2f} ke-")
            continue

        # real target density: saved npz, or read from FLOW
        if args.data_npz:
            dd = np.load(args.data_npz); d = np.asarray(dd["density"], float); ndat = int(dd.get("n", 0))
            print(f"  loaded data density from {args.data_npz} (n={ndat})")
        else:
            if not args.data_base:
                from fsdcube_hitq_datamc import DATA_BASE_DEF
                args.data_base = DATA_BASE_DEF
            d, ndat = data_density_from_flow(args, prc)
            np.savez(os.path.join(args.outdir, f"data_density_prc{prc}.npz"),
                     ctr=CTR, density=d, n=ndat, observable=args.observable, reset=use_rst)
        w_fit = fit_weights(Ff, d, reg=args.reg, w0=w0, prior=args.prior)
        plot_fit(T, F, Tf, w_fit, d, vals_all, prc, use_rst,
                 os.path.join(args.outdir, f"thrfit_prc{prc}.png"), args.observable, extrap=args.extrapolate)
        inr = vals_all[(vals_all >= args.thr_lo) & (vals_all <= args.thr_hi)]
        print(f"  prc{prc} (reset {use_rst} cyc): best-fit thresholds {moments(Tf, w_fit)[0]:.2f}"
              f"+/-{moments(Tf, w_fit)[1]:.2f} ke- vs known map {inr.mean():.2f}+/-{inr.std():.2f} "
              f"(n_data={ndat})")

    print(f"\nwrote plots -> {args.outdir}/")


if __name__ == "__main__":
    main()
