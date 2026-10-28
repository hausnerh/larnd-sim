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
#   interpolate  f_T(Q) linearly (per Q-bin) in T, so ANY threshold in [T_lo, T_hi] has a shape.
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
def load_basis(path, observable="single"):
    """ti_mult_grid.npz -> (T[ke- sorted], F[nT, QBINS] density rows, Y[nT] entries/shower, base_T).

    observable:
      'single' -- single-hit-pixel charge (node_q where node_n==1). For a single-hit pixel the
                  per-pixel total Q IS its one hit's Q, so this matches the data q[n==1] exactly,
                  and it is the sharpest threshold lever (its peak sits right at T).
      'hit'    -- every hit's charge (node_phq). Matches data q (all per-hit Q).
    """
    g = np.load(path, allow_pickle=True)
    thr = np.asarray(g["node_thr"], float); rst = np.asarray(g["node_rst"], int)
    base_rst = int(g["meta_base_reset"]); base_T = float(g["meta_base_threshold"]) / 1000.0
    nsh = max(int(g["meta_n_shower"]), 1)
    Ts = sorted(set(thr.tolist()))
    F, Y = [], []
    for T in Ts:
        hits = [k for k in range(len(thr)) if thr[k] == T and rst[k] == base_rst]
        if not hits:
            raise SystemExit(f"{path}: no node at threshold {T} e-, reset {base_rst}")
        i = hits[0]
        q = np.asarray(g["node_q"][i], float) / 1000.0
        n = np.asarray(g["node_n"][i], np.int16)
        phq = np.asarray(g["node_phq"][i], float) / 1000.0
        x = q[n == 1] if observable == "single" else phq
        F.append(density(x)); Y.append(x.size / nsh)
    return np.array(Ts) / 1000.0, np.vstack(F), np.array(Y), base_T


def interp_basis(T, F, Tq):
    """linear per-Q-bin interpolation of the density rows F(T) to query thresholds Tq (clipped to
    [T_lo, T_hi]; the ~14% of pixels above the top grid threshold are modelled by the top shape --
    extend the grid's THRESHOLDS to cover them exactly). Each output row is renormalized to area 1."""
    Tq = np.clip(np.asarray(Tq, float), T.min(), T.max())
    G = np.column_stack([np.interp(Tq, T, F[:, b]) for b in range(F.shape[1])])  # (nq, QBINS)
    a = (G * BINW).sum(1, keepdims=True); a[a == 0] = 1.0
    return G / a


# ---------------------------------------------------------------------------
# known per-pixel threshold map -> prediction
# ---------------------------------------------------------------------------
def map_thresholds(thr_map_path):
    """per-pixel thresholds [ke-] from the real map (one entry per physical pixel)."""
    dense, _default = load_thr_map(thr_map_path)      # ke-, index = pixel id
    return dense


def coarse_pdf(vals, T):
    """pixel-weighted pdf of thresholds `vals` onto the coarse basis grid T (nearest-threshold bins)."""
    edges = np.r_[-np.inf, 0.5 * (T[1:] + T[:-1]), np.inf]
    w, _ = np.histogram(np.asarray(vals, float), bins=edges)
    s = w.sum()
    return w / s if s else w.astype(float)


def predict_known(T, F, vals):
    """P_known(Q): pixel-weighted mean of the interpolated per-pixel shapes f_{T_pix}(Q), area 1."""
    G = interp_basis(T, F, vals)         # (n_pix, QBINS)
    p = G.mean(0)
    s = (p * BINW).sum()
    return p / s if s else p


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


def fit_weights(F, d, reg=0.0, iters=20000):
    """mixture weights w (one per basis row of F) whose density F^T w best matches target density d,
    constrained to the simplex (w>=0, sum=1 -> F^T w is itself a density). `reg` adds a 2nd-difference
    smoothness penalty (useful on a fine threshold grid). Projected gradient; returns w."""
    A = np.asarray(F, float).T               # (QBINS, nT): columns are basis densities
    d = np.asarray(d, float)
    nT = A.shape[1]
    AtA = A.T @ A; Atd = A.T @ d
    if reg > 0 and nT >= 3:
        D2 = (np.diag(np.ones(nT)) * -2.0
              + np.diag(np.ones(nT - 1), 1) + np.diag(np.ones(nT - 1), -1))[1:-1]
        AtA = AtA + reg * (D2.T @ D2)
    L = float(np.linalg.eigvalsh(AtA)[-1]) or 1.0
    lr = 1.0 / L
    w = np.full(nT, 1.0 / nT)
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
def plot_basis(T, F, base_T, vals, outpath, observable):
    """the mono-threshold basis family + the known-map prediction vs the nominal uniform sim."""
    f, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("plasma")
    for j, Tj in enumerate(T):
        ax[0].step(CTR, F[j], where="mid", lw=1.6,
                   color=cmap(0.06 + 0.82 * j / max(len(T) - 1, 1)),
                   label=f"{Tj:.2f} ke$^-$" + (" (nom)" if abs(Tj - base_T) < 1e-6 else ""))
    ax[0].set_title(f"mono-threshold basis  f$_T$(Q)   [{observable}]")
    ax[0].set_xlabel("hit Q [ke$^-$]"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25, lw=0.6); ax[0].set_xlim(0, 30)

    jn = int(np.argmin(np.abs(T - base_T)))
    pk = predict_known(T, F, vals)
    ax[1].step(CTR, F[jn], where="mid", lw=2.0, color=MC_COLORS[0],
               label=f"nominal uniform {T[jn]:.2f} ke$^-$ sim")
    ax[1].step(CTR, pk, where="mid", lw=2.0, color="#0072B2",
               label="known-map mixture (real per-pixel thr)")
    ax[1].set_title("what the known threshold SPREAD predicts")
    ax[1].set_xlabel("hit Q [ke$^-$]"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25, lw=0.6); ax[1].set_xlim(0, 30)
    f.tight_layout(); f.savefig(outpath, dpi=130); plt.close(f)
    print(f"  wrote {outpath}")


def plot_fit(T, F, d, w_fit, vals, prc, outpath, observable, dlabel="data"):
    """target density vs nominal / known-map / best-fit mixtures, and the fitted threshold pdf vs map."""
    f, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    jn = int(np.argmin(np.abs(T - np.mean(T))))
    nom = F[int(np.argmin(np.abs(T - 5.0)))]
    pk = predict_known(T, F, vals)
    mfit = mixture(F, w_fit)
    ax[0].step(CTR, d, where="mid", lw=2.2, color=DATA_COLOR, label=dlabel)
    ax[0].step(CTR, nom, where="mid", lw=1.5, ls=":", color=MC_COLORS[0], label="nominal 5 ke$^-$ sim")
    ax[0].step(CTR, pk, where="mid", lw=1.6, ls="--", color="#0072B2", label="known-map mixture")
    ax[0].step(CTR, mfit, where="mid", lw=2.0, color="#D55E00", label="best-fit mixture")
    ax[0].set_title(f"prc{prc}: hit-Q density [{observable}]")
    ax[0].set_xlabel("hit Q [ke$^-$]"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25, lw=0.6); ax[0].set_xlim(0, 30)

    wmap = coarse_pdf(vals, T)
    width = 0.38 * (T[1] - T[0])
    ax[1].bar(T - width / 2, wmap, width, color="#0072B2", alpha=0.8, label="known map")
    ax[1].bar(T + width / 2, w_fit, width, color="#D55E00", alpha=0.8, label="best fit to " + dlabel)
    ax[1].set_title(f"prc{prc}: threshold distribution")
    ax[1].set_xlabel("threshold [ke$^-$]"); ax[1].set_ylabel("fraction of pixels"); ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25, lw=0.6)
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
        T, F, Y, base_T = load_basis(grids[prc], args.observable)
        frac_in = float(((vals_all >= T.min()) & (vals_all <= T.max())).mean())
        print(f"\nprc{prc}: basis T={T} ke- (base {base_T:.2f}), single-hit yield/shower={np.round(Y,1)}, "
              f"{100*frac_in:.1f}% of pixels inside basis range")
        plot_basis(T, F, base_T, vals_all,
                   os.path.join(args.outdir, f"thrfit_basis_prc{prc}.png"), args.observable)

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
            # (3) the physics closure: fit the known-map prediction -> should resemble the map's pdf.
            pk = predict_known(T, F, vals_all)
            w_k = fit_weights(F, pk)
            plot_fit(T, F, pk, w_k, vals_all, prc,
                     os.path.join(args.outdir, f"thrfit_selftest_prc{prc}.png"),
                     args.observable, dlabel="known-map mixture (pseudo-data)")
            print(f"  known-map coarse pdf = {np.round(coarse_pdf(vals_all,T),3)}  "
                  f"fit-of-known      = {np.round(w_k,3)}")
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
                     ctr=CTR, density=d, n=ndat, observable=args.observable)
        w_fit = fit_weights(F, d)
        plot_fit(T, F, d, w_fit, vals_all, prc,
                 os.path.join(args.outdir, f"thrfit_prc{prc}.png"), args.observable)
        print(f"  prc{prc}: known-map coarse pdf = {np.round(coarse_pdf(vals_all,T),3)}")
        print(f"  prc{prc}: best-fit-to-data     = {np.round(w_fit,3)}  (n_data={ndat})")

    print(f"\nwrote plots -> {args.outdir}/")


if __name__ == "__main__":
    main()
