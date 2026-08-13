#!/usr/bin/env python
"""
muon_quickcheck.py -- fast, muon-ONLY induction check for any named detector config.

The muon samples in threshold_induction_study.py are PARAMETRIC (build_muon_event), so they
need no edep-sim input and no detector GDML -- which makes them the cheapest way to look at a
new geometry. This script skips the entire shower pipeline (drift cache, template grid,
threshold/reset/induction scans) and runs only:

    build muons at each angle -> drift -> induce (nominal AND induction-off) -> FEE
      -> truth diagnostics (composition, two-peak anatomy, causal timing)

Typical use -- compare a new geometry against module0 at a few orientations:

    python tests/muon_quickcheck.py --config fsd_cube --thetas 0 45 90 \\
        --muon-events 400 --outdir mucheck_fsdcube

theta is the angle to the PIXEL PLANE: 0 = in-plane / isochronous, 90 = along the drift axis.
Writes the per-angle single-hit spectra to <outdir>/mu_spectra.npz so the plots can be
regenerated offline. GPU required (it runs the real kernels).
"""
import argparse
import os
import sys

import numpy as np

_TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TESTS)
sys.path.insert(1, os.path.dirname(_TESTS))
import threshold_induction_study as tis  # noqa: E402
import verify_diffusion as vd  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="module0",
                    help="named larnd-sim config (module0, fsd_cube, fsd, ndlar, ...)")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--pixel-layout", default=None)
    ap.add_argument("--response", default=None)
    ap.add_argument("--sim-properties", default=None)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.0, 45.0, 90.0],
                    help="muon angles to the pixel plane (deg)")
    ap.add_argument("--muon-events", type=int, default=400, help="events per angle")
    ap.add_argument("--muon-length", type=float, default=20.0, help="track length (cm)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="discriminator threshold (e-); default = the config's own value")
    ap.add_argument("--reset", type=int, default=None,
                    help="PERIODIC_RESET_CYCLES (-1 = off); default = the config's own value. "
                         "NB module0 ships reset OFF but fsd_cube ships 512 cycles, so set "
                         "this explicitly when comparing geometries like-for-like.")
    ap.add_argument("--collect-frac", type=float, default=0.15)
    ap.add_argument("--noise-e", type=float, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default="mucheck")
    return ap.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    tis._journal_style()
    if args.list_configs:
        tis.list_named_configs(); return
    from numba import cuda
    if not cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available -- run this on a GPU node.")
    os.makedirs(args.outdir, exist_ok=True)

    tis.resolve_config(args)
    ctx = vd.load_simulation(args)
    vd.print_config(ctx)
    ctx._response0 = ctx.response.copy()
    ctx._induction_mask = tis.induction_mask(ctx)
    ctx.induction_scale = 1.0

    base_thr, base_reset = tis.base_config(ctx, args)
    if args.threshold is not None:
        base_thr = float(args.threshold)
    if args.reset is not None:
        base_reset = int(args.reset)
    noise_e = (float(args.noise_e) if args.noise_e is not None
               else float(np.atleast_1d(getattr(ctx.detector, "UNCORRELATED_NOISE", 500.0)).ravel()[0]))
    det = ctx.detector
    x0, x1, y0, y1 = vd.active_volume(det, 0, margin=3.0)
    print(f"\nMuon quick-check: config={args.config}  threshold={base_thr:.0f} e-  "
          f"reset={base_reset}  noise={noise_e:.0f} e-")
    print(f"  pitch={det.PIXEL_PITCH:.4f} cm  v_drift={det.V_DRIFT:.5f} cm/us  "
          f"drift_len={abs(det.DRIFT_LENGTH):.2f} cm  1 pitch = "
          f"{det.PIXEL_PITCH / det.V_DRIFT:.2f} us")
    print(f"  active volume (plane 0): x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}] cm")

    rng = np.random.default_rng(args.seed)
    tis.set_periodic_reset(ctx, base_reset)
    samples, rows = {}, []
    for it, theta in enumerate(args.thetas):
        name = "th%02d" % int(round(theta))
        raw = [tis.build_muon_event(ctx, rng, theta_deg=float(theta),
                                    length_cm=args.muon_length)
               for _ in range(int(args.muon_events))]
        raw = [r for r in raw if len(r) >= 2]
        drift = [d for d in (tis.event_drift(ctx, r) for r in raw) if d is not None]
        if not drift:
            print(f"  theta={theta:5.1f}: no events produced collectable charge -- SKIPPED")
            continue
        sd = args.seed + 7000 + it * 1000
        tis.set_induction(ctx, 1.0)
        evs = [e for e in (tis.event_induce(ctx, d[0], d[1], d[2], seed=sd + i,
                                            collect_frac=args.collect_frac)
                           for i, d in enumerate(drift)) if e is not None]
        nom = tis.aggregate_singlehits(ctx, evs, base_thr, base_reset, 1.0, seed0=sd + 100)
        tis.set_induction(ctx, 0.0)
        evs_off = [e for e in (tis.event_induce(ctx, d[0], d[1], d[2], seed=sd + 500 + i,
                                                collect_frac=args.collect_frac)
                               for i, d in enumerate(drift)) if e is not None]
        off = tis.aggregate_singlehits(ctx, evs_off, base_thr, base_reset, 0.0, seed0=sd + 600)
        tis.set_induction(ctx, 1.0)
        n = len(drift)
        samples[name] = dict(kind="muon", theta=float(theta), n=n, nominal=nom, off=off)
        q, tr, aux = nom
        tr = np.asarray(tr, bool)
        dtn = np.asarray(aux["dt_near"], float); f = np.isfinite(dtn)
        med_i = np.median(dtn[~tr & f]) if (~tr & f).any() else np.nan
        med_c = np.median(dtn[tr & f]) if (tr & f).any() else np.nan
        rows.append((theta, n, q.size, q.size / n, (~tr).sum() / max(q.size, 1),
                     off[0].size / n, med_i, med_c))
        print(f"  theta={theta:5.1f}: {n:4d} ev  {q.size:6d} single-hits "
              f"({q.size/n:5.1f}/ev)  f_ind={(~tr).sum()/max(q.size,1):.3f}  "
              f"induction-off {off[0].size:5d} ({off[0].size/n:4.1f}/ev)  "
              f"dt_near med ind={med_i:+7.2f} coll={med_c:+7.2f} us")

    if not samples:
        sys.exit("No muon samples produced any single hits -- check the geometry/threshold.")

    # ---- save + plots -------------------------------------------------------------
    qmax = float(min(np.max(np.concatenate([s["nominal"][0] for s in samples.values()])),
                     50000.0))
    kw = {"meta_config": np.array(args.config), "meta_thr": np.array(base_thr),
          "meta_reset": np.array(base_reset), "meta_noise": np.array(noise_e),
          "meta_qmax": np.array(qmax), "meta_pitch": np.array(det.PIXEL_PITCH),
          "meta_vdrift": np.array(det.V_DRIFT), "meta_names": np.array(list(samples))}
    for nm, s in samples.items():
        kw[f"{nm}_theta"] = np.array(s["theta"]); kw[f"{nm}_n"] = np.array(s["n"])
        for which in ("nominal", "off"):
            q, tr, aux = s[which]
            kw[f"{nm}_{which}_q"] = np.asarray(q, float)
            kw[f"{nm}_{which}_tr"] = np.asarray(tr, bool)
            for k in tis._AUX_KEYS:
                kw[f"{nm}_{which}_aux_{k}"] = np.asarray(aux.get(k, []), float)
    np.savez(f"{args.outdir}/mu_spectra.npz", **kw)
    print(f"\n  wrote {args.outdir}/mu_spectra.npz")

    thr_sigma = float(np.atleast_1d(det.DISCRIMINATOR_NOISE).ravel()[0])
    for nm, s in samples.items():
        lab = r"muon, $\theta=%.0f^{\circ}$" % s["theta"]
        qo, tro, auxo = s["off"]
        coll = np.asarray(tro, bool)
        sel = coll if coll.sum() >= 20 else np.ones(np.size(qo), bool)
        tis.plot_offpeak_anatomy(np.asarray(qo, float)[sel], coll[sel],
                                 tis._aux_sel(auxo, sel, np.size(qo)), base_thr, args.outdir,
                                 qmax=qmax, n_shower=s["n"], suffix="_" + nm, title=lab)
        qn, trn, auxn = s["nominal"]
        tis.plot_hit_timing(qn, trn, auxn, base_thr, args.outdir, n_shower=s["n"],
                            suffix="_" + nm, title=lab)

    # composition overview: one panel per angle, truth-coloured, shared x
    import matplotlib.pyplot as plt
    names = list(samples)
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.2),
                             sharex=True, squeeze=False)
    for ax, nm in zip(axes.flat, names):
        s = samples[nm]; q, tr, _ = s["nominal"]
        q = np.asarray(q, float); tr = np.asarray(tr, bool); n = s["n"]
        h = tis._hist(q, base_thr, qmax=qmax)
        if h is None:
            continue
        edges = h[3] / 1e3
        w = lambda mk: np.full(int(mk.sum()), 1.0 / n)
        ax.hist([q[~tr] / 1e3, q[tr] / 1e3], bins=edges, stacked=True,
                color=[tis._C_IND, tis._C_SHW], alpha=0.65, edgecolor="white", linewidth=0.25,
                weights=[w(~tr), w(tr)], label=["Induction", "Collection"])
        ax.axvspan((base_thr - thr_sigma) / 1e3, (base_thr + thr_sigma) / 1e3,
                   color="0.45", alpha=0.18, lw=0)
        ax.axvline(base_thr / 1e3, color="0.30", ls="--", lw=1.1)
        ax.set_ylim(bottom=0.0); ax.set_xlim(0, 30)
        ax.set_xlabel(r"Single-Hit Pixel Charge $Q$ ($10^{3}\,e^{-}$)")
        ax.text(0.96, 0.95, "\n".join((r"$\theta=%.0f^{\circ}$" % s["theta"],
                r"$N/\mathrm{event}=%.1f$" % (q.size / n),
                r"$f_{\mathrm{ind}}=%.3f$" % ((~tr).sum() / max(q.size, 1)))),
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.7))
    axes.flat[0].set_ylabel("Single-Hit Pixels / Event")
    axes.flat[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    tis._savefig(fig, f"{args.outdir}/mu_composition.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
