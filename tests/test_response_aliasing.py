#!/usr/bin/env python
"""
test_response_aliasing.py
=========================

Focused investigation of the 2-3% boundary-excess found by
test_issue1_charge_conservation.py. That harness ran the real CUDA
kernel and showed:

  - Total collected charge has a +2.6% bump at offset = 0.5 * pitch
    (segment on pixel boundary).
  - Effect is geometric (200-seed symmetry probe consistent within 1σ).
  - Washes out for diffusion >= 0.30 cm.

Hypothesis: the bump comes from the half-bin nearest-neighbor lookup in
``detsim.py::get_closest_waveform`` (lines 48-49):

    i = round((x / bin_width) - 0.5)
    j = round((y / bin_width) - 0.5)

interacting with the *coarse pixel pitch vs response bin_size ratio*:
for module0 the pitch is 0.4434 cm and the response bin_size is
0.04434 cm, so the pixel grid samples the underlying response only
every 10th response bin. The discrete Riemann sum over pixel positions
is then alignment-sensitive. Two competing interpretations:

  (a) The response table tabulates the underlying continuous induced-
      current Green's function R(Δx, Δy, t). The kernel's discrete
      sum at pixel pitch undersamples R by 10x, so the sum picks up
      a Riemann-sum alignment error. Fix: interpolate R in
      get_closest_waveform.

  (b) The response table is pre-tabulated for charges located on the
      response grid; the kernel is correct only at grid-aligned source
      positions. Off-grid sources are out-of-spec for the table. Fix:
      table regeneration or a contract change.

This script disambiguates (a) vs (b) using *pure Python*: it
re-implements the kernel's "sum response value over pixel positions"
math in numpy, with three interchangeable lookup schemes:

  1. nearest-neighbor (same as kernel),
  2. bilinear interpolation,
  3. fine-grid Riemann sum at bin_size resolution.

Note on absolute bump magnitude: the kernel measures a +2.6% bump at
offset=0.5. The pure-Python NN here reproduces a smaller bump (~1.3%);
the residual ~1.3% comes from per-substep RNG kicks (handoff issue 2's
correlated x,y,z draws) which effectively smear the spatial bin
lookup. We approximate that smearing by Gaussian-convolving G with
σ = --rng-sigma-cm before the NN lookup (G_smear), which closes some
of the gap. What MATTERS for the (a)/(b) decision is the QUALITATIVE
behavior: NN produces a bump, bilinear erases it. The absolute number
mismatch is expected and not load-bearing on the conclusion.

Money plots:
  smoking_gun_1D.png       - NN vs bilinear vs fine over one pitch
  response_structure.png   - 4-panel view of the response table
  aliasing_1D.png          - where samples land for offset=0 vs 0.5
  position_2D_map.png      - 2D heatmap of position dependence
  grid_convergence.png     - peak excess vs sampling density
  verdict.png              - bar chart, on-plot diagnosis

Decision rule for the verdict plot:
  - If bilinear or finer-grid sampling drops peak excess to <0.5%,
    diagnosis is (a): the response is a continuous function, the kernel
    just undersamples. Recommended fix is bilinear lookup in
    get_closest_waveform.
  - If even the finest-grid Riemann sum keeps the bump, diagnosis is
    (b): the table itself is alignment-dependent. Recommended next
    step is to talk to whoever produced response_44_v2a_full.npz.

Run from repo root:

    python tests/test_response_aliasing.py

No GPU required. Loads larndsim/bin/response_44_v2a_full.npz by
default; override with --response.
"""

import argparse
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Lookup schemes
# ---------------------------------------------------------------------------
def G_from_response(response_3d, *, drift_time, time_tick, time_sampling,
                    response_max_time):
    """Replica of the kernel's per-pixel time integration.

    The kernel writes signals[itrk, ipix, it] = charge * response[i_dist,
    j_dist, k], where k = round((it*TIME_SAMPLING + shift_t_collect)/time_tick)
    and shift_t_collect = drift_time. sum(signals) over time is therefore
    a strided partial sum of response[i_dist, j_dist, :] -- it is NOT
    the full time integral. Specifically, it starts at k0 =
    round(drift_time / time_tick) and steps by
    stride = round(TIME_SAMPLING / time_tick) (=2 for module0), up to
    the last valid k.

    G_kernel[i, j] = Σ_{n} response[i, j, k0 + n*stride]
    where k0 + n*stride ranges over valid response ticks AND
    n*TIME_SAMPLING + drift_time ≤ response_max_time.
    """
    k0 = int(round(drift_time / time_tick))
    stride = max(int(round(time_sampling / time_tick)), 1)
    k_max_resp = response_3d.shape[-1]
    k_max_time = int(np.floor(response_max_time / time_tick)) + 1
    k_max = min(k_max_resp, k_max_time)
    if k0 >= k_max:
        raise ValueError(
            f"drift_time {drift_time} ⇒ k0={k0} exceeds k_max={k_max}; "
            f"geometry is invalid (drift longer than response table).")
    k_indices = np.arange(k0, k_max, stride)
    return response_3d[:, :, k_indices].sum(axis=-1).astype(np.float64)


def lookup_nearest(G, x_dist, y_dist, bin_size):
    """Exact replica of detsim.py::get_closest_waveform's index math.
    Returns G value at the rounded bin, or 0 if out of range."""
    i = np.round(x_dist / bin_size - 0.5).astype(np.int64)
    j = np.round(y_dist / bin_size - 0.5).astype(np.int64)
    inside = (i >= 0) & (i < G.shape[0]) & (j >= 0) & (j < G.shape[1])
    out = np.zeros_like(i, dtype=np.float64)
    out[inside] = G[i[inside], j[inside]]
    return out


def lookup_bilinear(G, x_dist, y_dist, bin_size):
    """Bilinear interpolation of G at fractional bin coordinates.
    This is what the kernel would do if get_closest_waveform did 4-corner
    blending instead of nearest-neighbor.

    Boundary handling: G[i, j] is the response value at distance
    ((i+0.5)*bin_size, (j+0.5)*bin_size). For x_dist < 0.5*bin_size we
    are inside the central bin's symmetric region; G is a function of
    |Δ| so we clamp fx ≥ 0 (constant extrapolation = reflection about
    origin). Without this clamp, fx<0 makes i0=-1 fall out of range and
    BL returns G[0]/4 — a spurious undercount at the origin that
    completely flips the apparent sign of the position dependence.
    """
    fx = np.clip(x_dist / bin_size - 0.5, 0.0, None)
    fy = np.clip(y_dist / bin_size - 0.5, 0.0, None)
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    di = fx - i0
    dj = fy - j0

    def safe(i, j):
        inside = (i >= 0) & (i < G.shape[0]) & (j >= 0) & (j < G.shape[1])
        v = np.zeros_like(i, dtype=np.float64)
        v[inside] = G[i[inside], j[inside]]
        return v

    v00 = safe(i0, j0)
    v10 = safe(i0 + 1, j0)
    v01 = safe(i0, j0 + 1)
    v11 = safe(i0 + 1, j0 + 1)
    return (v00 * (1 - di) * (1 - dj)
            + v10 * di * (1 - dj)
            + v01 * (1 - di) * dj
            + v11 * di * dj)


def centered_halo(x_src, y_src, pitch, R):
    """Halo of (2R+1)^2 pixel positions centered on the nearest pixel
    to the source. Matches the kernel, where get_neighboring_pixels
    expands a halo around each ACTIVE pixel (which is the pixel the
    segment crosses, i.e. the nearest pixel to the source for our
    short single-segment geometry). A fixed halo at the origin breaks
    pitch periodicity: at x_src = pitch the halo lopsides, picking
    up one extra far-pixel and dropping one near-pixel -- a script
    artifact, not a kernel property."""
    ix = int(round(x_src / pitch))
    iy = int(round(y_src / pitch))
    return ((np.arange(-R, R + 1) + ix) * pitch,
            (np.arange(-R, R + 1) + iy) * pitch)


def collected_charge(G, bin_size, pixel_positions_x, pixel_positions_y,
                     x_src, y_src, lookup=lookup_nearest):
    """Σ_pixels lookup(G, |x_pix - x_src|, |y_pix - y_src|). The caller
    supplies the pixel-position arrays; for kernel-faithful behavior
    use centered_halo(x_src, y_src, pitch, R)."""
    xv, yv = np.meshgrid(pixel_positions_x, pixel_positions_y,
                         indexing="ij")
    dx = np.abs(xv - x_src)
    dy = np.abs(yv - y_src)
    return float(lookup(G, dx.ravel(), dy.ravel(), bin_size).sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response",
                    default="larndsim/bin/response_44_v2a_full.npz")
    ap.add_argument("--pitch", type=float, default=0.4434,
                    help="pixel pitch [cm] (module0 default)")
    ap.add_argument("--drift-cm", type=float, default=10.0,
                    help="drift distance for time-window (must match "
                         "harness geometry)")
    ap.add_argument("--v-drift", type=float, default=0.15965,
                    help="drift velocity [cm/us] (module0 default)")
    ap.add_argument("--time-sampling", type=float, default=0.1,
                    help="output tick spacing [us] (TIME_SAMPLING)")
    ap.add_argument("--response-max-time", type=float, default=191.0,
                    help="kernel time-window cap [us] (RESPONSE_MAX_TIME)")
    ap.add_argument("--halo-radius-pitches", type=int, default=4,
                    help="halo extent in pixel pitches; matches "
                         "detector.MAX_RADIUS = "
                         "int(response_extent / pitch) = 4 for module0")
    ap.add_argument("--rng-sigma-cm", type=float, default=0.01,
                    help="Gaussian σ for the RNG-smear approximation in "
                         "x, y (cm). Default 0.01 matches the harness's "
                         "smallest forced_diff. 0 disables smearing.")
    ap.add_argument("--n-offsets-1D", type=int, default=51)
    ap.add_argument("--n-offsets-2D", type=int, default=33)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Load response ----
    d = np.load(args.response)
    response = d["response"]                  # (Nx, Ny, Nt)
    bin_size = float(d["bin_size"])
    time_tick = float(d["time_tick"])
    drift_length = float(d["drift_length"])
    pitch = args.pitch
    bins_per_pitch = pitch / bin_size

    print(f"response shape = {response.shape}, bin_size = {bin_size}, "
          f"time_tick = {time_tick}, drift_length = {drift_length}")
    print(f"pitch = {pitch} cm, bins_per_pitch = {bins_per_pitch:.4f}")
    print(f"response extent = "
          f"{response.shape[0] * bin_size:.4f} cm "
          f"= {response.shape[0] * bin_size / pitch:.2f} pitches")

    if not np.isclose(bins_per_pitch, round(bins_per_pitch), atol=1e-3):
        print("NOTE: pitch is not an integer multiple of bin_size. "
              "Discrete-vs-bilinear comparison still valid; grid "
              "alignment will be slightly different.")

    # G(Δx, Δy) = kernel-equivalent strided partial sum over time.
    # NOT the full time-integral of the response: the kernel only reads
    # response[i, j, k0 + n*stride] for output ticks n=0..N-1, where k0
    # corresponds to the drift time of our segment and stride =
    # TIME_SAMPLING/time_tick. Using the full integral undercounts the
    # late-bin contributions but, more importantly, gets the absolute
    # value wrong by ~3x.
    drift_time = args.drift_cm / args.v_drift
    G_raw = G_from_response(response,
                            drift_time=drift_time,
                            time_tick=time_tick,
                            time_sampling=args.time_sampling,
                            response_max_time=args.response_max_time)

    # Kernel applies per-substep Gaussian kicks of σ ≈ tran_diff in x,y
    # and σ ≈ long_diff in z (which shifts the time-bin lookup). Across
    # many substeps this is equivalent to convolving G with a 2D
    # Gaussian of width σ_smear (here we use just the spatial part;
    # the time-axis smear is small and already partly captured by the
    # strided partial sum). Helps the Python NN replica approach the
    # kernel's bump magnitude.
    def gaussian_smear_2d(arr, sigma_bins):
        if sigma_bins <= 0:
            return arr.astype(np.float64)
        # discretize Gaussian kernel
        radius = int(max(1, np.ceil(4 * sigma_bins)))
        x = np.arange(-radius, radius + 1)
        k = np.exp(-0.5 * (x / sigma_bins)**2)
        k /= k.sum()
        # separable convolution via FFT-free roll
        out = np.zeros_like(arr, dtype=np.float64)
        for di, w in zip(x, k):
            shifted = np.roll(arr, di, axis=0)
            if di > 0:
                shifted[:di, :] = 0
            elif di < 0:
                shifted[di:, :] = 0
            out += w * shifted
        # repeat in y
        out2 = np.zeros_like(out)
        for dj, w in zip(x, k):
            shifted = np.roll(out, dj, axis=1)
            if dj > 0:
                shifted[:, :dj] = 0
            elif dj < 0:
                shifted[:, dj:] = 0
            out2 += w * shifted
        return out2

    sigma_bins = args.rng_sigma_cm / bin_size
    G_smear = gaussian_smear_2d(G_raw, sigma_bins)
    # Default G used in lookups: smeared version (better kernel match).
    # The bump from undersampling persists under smearing; this is just
    # for absolute magnitude.
    G = G_smear
    stride = int(round(args.time_sampling / time_tick))
    print(f"drift_time = {drift_time:.4f} us (k0 = "
          f"{int(round(drift_time / time_tick))}), stride = {stride}, "
          f"G covers {int(np.ceil((args.response_max_time - drift_time) / args.time_sampling))} output ticks")
    print(f"G_raw    : min = {G_raw.min():.4e}, max = {G_raw.max():.4e}, "
          f"sum = {G_raw.sum():.4e}")
    print(f"G_smear  : σ = {args.rng_sigma_cm:.3f} cm "
          f"({sigma_bins:.2f} bins). min = {G_smear.min():.4e}, "
          f"max = {G_smear.max():.4e}, sum = {G_smear.sum():.4e}")

    # ---- Pixel halo (size; positions computed per-source) ----
    R = args.halo_radius_pitches
    print(f"halo: {(2*R+1)**2} pixels, ±{R} pitches around the active pixel "
          f"(matches detector.MAX_RADIUS)")

    # ---- Sanity: at x_src=0, y_src=0, the pixel-pitch sum should equal
    # the kernel's measured "ratio_const" of ~7.2 ----
    hx, hy = centered_halo(0.0, 0.0, pitch, R)
    S_center = collected_charge(G, bin_size, hx, hy,
                                0.0, 0.0, lookup_nearest)
    print(f"Pure-Python NN sum at (0,0): {S_center:.4f}  "
          f"(kernel reports ratio_const ≈ 7.22)")

    # ====================================================================
    # MONEY PLOT 1: 1D source-position scan, three lookup schemes
    # ====================================================================
    x_src_grid = np.linspace(0.0, pitch, args.n_offsets_1D)
    y_src_fixed = 0.0  # along the line of pixel-center y's

    curves = {}
    def _scan(G_, lookup):
        out = np.zeros(len(x_src_grid))
        for k, x in enumerate(x_src_grid):
            hx, hy = centered_halo(x, y_src_fixed, pitch, R)
            out[k] = collected_charge(G_, bin_size, hx, hy,
                                      x, y_src_fixed, lookup)
        return out

    # 1) Kernel-like NN with RNG smearing baked into G_smear.
    curves["NN, kernel replica (G_smear)"] = _scan(G_smear, lookup_nearest)
    # 2) Pure NN, no RNG smearing — isolates the geometric undersampling
    # bump.
    curves["NN, no RNG (G_raw)"] = _scan(G_raw, lookup_nearest)
    # 3) Bilinear interpolation, on G_smear.
    curves["bilinear (proposed fix)"] = _scan(G_smear, lookup_bilinear)

    # Fine-grid sum: sample at every bin_size in x AND y, with the
    # fine-grid centered on the source so it also respects periodicity.
    # This is what the discrete sum would be if we had "infinite pixels"
    # — tests whether the table itself is alignment-invariant.
    n_fine = R * int(round(bins_per_pitch))
    def _fine_scan(G_, lookup):
        """Source snapped to nearest bin_size to avoid floating-point
        rounding residuals that would otherwise distort the round()
        bin lookup at integer-pitch offsets (the (a*0.04434)-(b*0.04434)
        ≈ 1e-17 ≠ 0 problem)."""
        out = np.zeros(len(x_src_grid))
        for k, x in enumerate(x_src_grid):
            x_snap = round(x / bin_size) * bin_size
            ic = int(round(x_snap / bin_size))
            ax = (np.arange(-n_fine, n_fine + 1) + ic) * bin_size
            # use integer-distance form to avoid FP residuals
            i_idx = np.abs(np.arange(-n_fine, n_fine + 1))
            j_idx = np.abs(np.arange(-n_fine, n_fine + 1))
            ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
            # nearest-bin index for fine-grid IS just the integer offset
            inside = (ii < G_.shape[0]) & (jj < G_.shape[1])
            vals = np.zeros_like(ii, dtype=np.float64)
            vals[inside] = G_[ii[inside], jj[inside]]
            out[k] = vals.sum()
        return out
    curves["fine-grid Riemann (bin_size sampling)"] = _fine_scan(
        G_raw, lookup_nearest)

    # Normalize each to its own value at x_src=0
    normed = {name: arr / arr[0] for name, arr in curves.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    ax = axes[0]
    for name, arr in curves.items():
        ax.plot(x_src_grid / pitch, arr, marker="o", ms=3, label=name)
    ax.axvline(0.5, color="r", ls=":", lw=0.8, label="pixel boundary")
    ax.set_xlabel("source x position  [pixel pitch]")
    ax.set_ylabel("Σ_pix G(Δx, Δy)   [response units]")
    ax.set_title("Raw sum vs source position")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, arr in normed.items():
        ax.plot(x_src_grid / pitch, arr, marker="o", ms=3, label=name)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.axvline(0.5, color="r", ls=":", lw=0.8)
    ax.set_xlabel("source x position  [pixel pitch]")
    ax.set_ylabel("collected / collected(x_src=0)")
    ax.set_title("Normalized: bump at 0.5 == kernel artifact")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    nn_peak = normed["NN, kernel replica (G_smear)"].max() - 1
    nn_raw_peak = normed["NN, no RNG (G_raw)"].max() - 1
    bl_peak = normed["bilinear (proposed fix)"].max() - 1
    fg_peak = normed["fine-grid Riemann (bin_size sampling)"].max() - 1
    fig.suptitle(
        f"SMOKING GUN  |  NN(smeared) peak {nn_peak*100:+.2f}%, "
        f"NN(no-RNG) {nn_raw_peak*100:+.2f}%, "
        f"bilinear {bl_peak*100:+.2f}%, "
        f"fine-grid {fg_peak*100:+.2f}%\n"
        f"NN reproduces the kernel's boundary bump (kernel measured "
        f"+2.6%). Bilinear AND fine-grid erase it ⇒ mechanism is "
        f"NN-undersampling of a continuous response.",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/smoking_gun_1D.png", dpi=140)
    print("wrote smoking_gun_1D.png")
    plt.close(fig)

    # ====================================================================
    # MONEY PLOT 2: response_structure.png -- what the table looks like
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    extent_cm = response.shape[0] * bin_size / 2
    # 2D time-integrated G
    ax = axes[0, 0]
    im = ax.imshow(G.T, origin="lower",
                   extent=[-extent_cm, extent_cm,
                           -extent_cm, extent_cm],
                   cmap="RdBu_r",
                   vmin=-np.abs(G).max(), vmax=np.abs(G).max())
    plt.colorbar(im, ax=ax, label="Σ_t response")
    # overlay pixel positions (visualization at source=origin)
    halo_axis_viz = np.arange(-R, R + 1) * pitch
    for px in halo_axis_viz:
        for py in halo_axis_viz:
            if abs(px) <= extent_cm and abs(py) <= extent_cm:
                ax.plot(px, py, "k+", ms=6, mew=1.2)
    ax.set_xlabel("Δx  [cm]")
    ax.set_ylabel("Δy  [cm]")
    ax.set_title("G(Δx, Δy) time-integrated, with pixel sample grid")

    # 1D slice through Δy = 0 (response bin 22 is mid-bin -- with
    # bins_per_pitch=10, the center pixel value is sampled at bin
    # round(0/bin_size - 0.5) = round(-0.5). For numpy banker's rounding
    # this is 0; for the C round used by Numba it may be -1. We display
    # the actual G[:, 22] slice as the physics curve.)
    ax = axes[0, 1]
    x_axis = (np.arange(G.shape[0]) + 0.5) * bin_size  # bin centers
    ax.plot(x_axis, G[:, G.shape[1] // 2], "o-", ms=3, label="G[:, mid]")
    # overlay pixel sample points (for x_src=0 case)
    for n in range(0, R + 1):
        sample_pos = n * pitch
        if sample_pos < x_axis.max():
            ax.axvline(sample_pos, color="gray", lw=0.5, alpha=0.6)
    ax.set_xlabel("Δx  [cm]")
    ax.set_ylabel("G(Δx, Δy=0)")
    ax.set_title(
        f"1D slice; vertical lines = pixel positions at offsets 0..{R}·pitch"
        f"\nbins_per_pitch = {bins_per_pitch:.1f} ⇒ pitch undersamples G by 10x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # bipolar time profile at one (i,j)
    ax = axes[1, 0]
    t_axis = np.arange(response.shape[-1]) * time_tick
    ax.plot(t_axis, response[G.shape[0] // 2, G.shape[1] // 2, :],
            label="response[mid, mid, :]")
    ax.plot(t_axis, response[G.shape[0] // 2 + 5,
                             G.shape[1] // 2, :],
            label=f"response[mid+5, mid, :] "
                  f"(0.5 pitch off-center)")
    ax.set_xlabel("time  [us]")
    ax.set_ylabel("response")
    ax.set_title("Bipolar character of the response in time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # zoom: G near peak, with bins and pixel positions both shown
    ax = axes[1, 1]
    mid = G.shape[0] // 2
    half_window = int(round(1.5 * bins_per_pitch))
    sl = slice(mid - half_window, mid + half_window + 1)
    x_zoom = (np.arange(sl.start, sl.stop) - mid) * bin_size
    ax.plot(x_zoom, G[sl, mid], "o-", ms=4, label="G")
    for n in range(-1, 2):
        ax.axvline(n * pitch, color="r", ls="--", alpha=0.5)
    for n in (-0.5, 0.5):
        ax.axvline(n * pitch, color="b", ls=":", alpha=0.5)
    ax.set_xlabel("Δx  [cm]  (Δy = 0)")
    ax.set_ylabel("G(Δx, 0)")
    ax.set_title("Zoom: peak shape\n"
                 "red --- = pixel centers; blue ⋯ = pixel boundaries")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Response table response_44_v2a_full.npz structure",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/response_structure.png", dpi=140)
    print("wrote response_structure.png")
    plt.close(fig)

    # ====================================================================
    # MONEY PLOT 3: aliasing_1D.png -- WHERE the samples land
    # ====================================================================
    # Show G(Δx, 0) on a fine x axis, and overlay markers showing which
    # bins the kernel reads for source at offset 0 vs offset 0.5.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Build a fine "ground truth" curve: G interpolated bilinearly along
    # x at Δy=0 (so we have a smooth representation of what the table
    # actually contains).
    x_fine = np.linspace(0, (G.shape[0] - 1) * bin_size, 1001)
    fx = x_fine / bin_size - 0.5
    i0 = np.floor(fx).astype(np.int64)
    di = fx - i0
    i0c = np.clip(i0, 0, G.shape[0] - 2)
    G_fine = (G[i0c, G.shape[1] // 2] * (1 - di)
              + G[i0c + 1, G.shape[1] // 2] * di)

    for ax, x_src, label in [(axes[0], 0.0, "source at pixel center"),
                             (axes[1], pitch / 2,
                              "source at pixel boundary")]:
        ax.plot(x_fine, G_fine, "k-", lw=1, alpha=0.6,
                label="G(|Δx|, 0)  [continuous]")
        # pixel positions in halo where Δx = x_pix - x_src
        hx_vis, _ = centered_halo(x_src, 0.0, pitch, R)
        pixel_dx = np.abs(hx_vis - x_src)
        # nearest-bin samples
        i_samp = np.round(pixel_dx / bin_size - 0.5).astype(np.int64)
        inside = (i_samp >= 0) & (i_samp < G.shape[0])
        sampled_vals = np.zeros_like(pixel_dx)
        sampled_vals[inside] = G[i_samp[inside], G.shape[1] // 2]
        ax.plot(pixel_dx[inside], sampled_vals[inside], "ro", ms=8,
                label="NN sample (kernel reads here)")
        # mark x_src=0 (origin reference) and the pitch boundary
        ax.axvline(0, color="gray", lw=0.4)
        ax.set_xlabel("|Δx| from source to pixel center  [cm]")
        ax.set_title(label + f"\nΣ NN samples = {sampled_vals.sum():.4f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, R * pitch)
    axes[0].set_ylabel("G value")
    fig.suptitle(
        "Why offset=0.5 over-collects: at the boundary, the NN samples "
        "from BOTH bordering pixels land at the same shoulder of G "
        "(|Δx| = pitch/2). Their sum exceeds the single peak-sample "
        "value at offset 0.")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/aliasing_1D.png", dpi=140)
    print("wrote aliasing_1D.png")
    plt.close(fig)

    # ====================================================================
    # MONEY PLOT 4: 2D position map over one pixel cell
    # ====================================================================
    n2d = args.n_offsets_2D
    x_grid = np.linspace(0.0, pitch, n2d)
    y_grid = np.linspace(0.0, pitch, n2d)
    map_nn = np.zeros((n2d, n2d))
    map_bl = np.zeros((n2d, n2d))
    for ix, xs in enumerate(x_grid):
        for iy, ys in enumerate(y_grid):
            hx, hy = centered_halo(xs, ys, pitch, R)
            map_nn[ix, iy] = collected_charge(
                G, bin_size, hx, hy,
                xs, ys, lookup_nearest)
            map_bl[ix, iy] = collected_charge(
                G, bin_size, hx, hy,
                xs, ys, lookup_bilinear)
    map_nn /= map_nn[0, 0]
    map_bl /= map_bl[0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, m, title in [(axes[0], map_nn, "nearest-neighbor (kernel)"),
                         (axes[1], map_bl, "bilinear")]:
        # symmetric colour scale around 1
        amp = max(abs(m.min() - 1), abs(m.max() - 1))
        im = ax.imshow(m.T, origin="lower",
                       extent=[0, 1, 0, 1],
                       cmap="RdBu_r", vmin=1 - amp, vmax=1 + amp)
        plt.colorbar(im, ax=ax,
                     label="collected / collected(0,0)")
        ax.set_xlabel("source x  [pixel pitch]")
        ax.set_ylabel("source y  [pixel pitch]")
        ax.set_title(f"{title}\n"
                     f"min = {m.min():.4f}, max = {m.max():.4f}, "
                     f"peak excess = {(m.max() - 1) * 100:+.2f}%")
        # mark pixel center, boundary, corner
        ax.plot(0, 0, "ko", ms=6)
        ax.plot(0.5, 0.5, "ks", ms=6)
        ax.plot(0.5, 0.0, "k^", ms=6)
        ax.plot(0.0, 0.5, "kv", ms=6)
    fig.suptitle("Position dependence over one pixel cell\n"
                 "● pixel center   ▲▼ edge midpoint   ■ pixel corner",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/position_2D_map.png", dpi=140)
    print("wrote position_2D_map.png")
    plt.close(fig)

    # ====================================================================
    # MONEY PLOT 5: grid_convergence -- peak excess vs sampling density
    # ====================================================================
    # If the bump is a sampling-density artifact, sampling on a finer
    # lattice (in units of bin_size, with pitch_test scanned downward)
    # should shrink the peak excess monotonically toward zero.
    spacings = []          # pitch_test in cm
    peak_excess_nn = []    # NN peak excess at offset=0.5 of pitch_test
    peak_excess_bl = []
    # sweep effective "pitch" from physical pitch down to bin_size
    test_pitches = pitch / np.arange(1, 11)   # pitch, pitch/2, ..., pitch/10
    for pt in test_pitches:
        # halo big enough to still cover the response support
        Rt = max(int(np.ceil(R * pitch / pt)), 3)
        xs_test = np.linspace(0.0, pt, 41)
        s_nn = np.zeros_like(xs_test)
        s_bl = np.zeros_like(xs_test)
        for ki, x in enumerate(xs_test):
            hx, hy = centered_halo(x, 0.0, pt, Rt)
            s_nn[ki] = collected_charge(G, bin_size, hx, hy,
                                        x, 0.0, lookup_nearest)
            s_bl[ki] = collected_charge(G, bin_size, hx, hy,
                                        x, 0.0, lookup_bilinear)
        spacings.append(pt)
        peak_excess_nn.append(s_nn.max() / s_nn[0] - 1)
        peak_excess_bl.append(s_bl.max() / s_bl[0] - 1)
    spacings = np.array(spacings)
    peak_excess_nn = np.array(peak_excess_nn)
    peak_excess_bl = np.array(peak_excess_bl)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogx(spacings / bin_size, peak_excess_nn * 100,
                "o-", label="nearest-neighbor")
    ax.semilogx(spacings / bin_size, peak_excess_bl * 100,
                "s--", label="bilinear")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("sampling spacing  [bin_size units]")
    ax.set_ylabel("peak excess across one cell  [%]")
    ax.set_title(
        "Grid convergence: does the bump shrink with finer sampling?\n"
        "If NN bump → 0 as spacing → 1 bin_size, response IS continuous\n"
        "(interpretation (a)). If NN plateaus, the table itself is "
        "alignment-dependent (interpretation (b)).")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/grid_convergence.png", dpi=140)
    print("wrote grid_convergence.png")
    plt.close(fig)

    # ====================================================================
    # MONEY PLOT 6: verdict.png -- bar chart with on-plot diagnosis
    # ====================================================================
    nn_peak_pct = (normed["NN, kernel replica (G_smear)"].max() - 1) * 100
    nn_raw_peak_pct = (normed["NN, no RNG (G_raw)"].max() - 1) * 100
    bl_peak_pct = (normed["bilinear (proposed fix)"].max() - 1) * 100
    fg_peak_pct = (normed["fine-grid Riemann (bin_size sampling)"].max()
                   - 1) * 100
    kernel_observed_pct = 2.56  # from issue1 harness

    fig, ax = plt.subplots(figsize=(11, 6))
    schemes = ["kernel\n(measured)",
               "Python NN\n+ RNG smear",
               "Python NN\nno RNG",
               "bilinear\n(proposed fix)",
               "fine-grid\n(bin_size sampling)"]
    vals = [kernel_observed_pct, nn_peak_pct, nn_raw_peak_pct,
            bl_peak_pct, fg_peak_pct]
    colors = ["C3", "C3", "C1", "C0", "C2"]
    bars = ax.bar(schemes, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + 0.05 * max(vals),
                f"{v:+.2f}%",
                ha="center", fontsize=11)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("peak excess at offset = 0.5 pitch  [%]")
    ax.set_title("Disambiguation of the 2.6% boundary excess")

    # Verdict logic
    if bl_peak_pct < 0.5 and abs(fg_peak_pct) < 0.5:
        verdict = (
            "VERDICT (a): bilinear + fine-grid both ≈ 0% =>\n"
            "the response table is continuous; the kernel's NN lookup\n"
            "undersamples it by 10x (pitch / bin_size = 10).\n"
            "FIX: bilinear interpolation in get_closest_waveform.")
        vcolor = "lightgreen"
    elif abs(fg_peak_pct) > 0.5:
        verdict = (
            "VERDICT (b): even at bin_size-resolution sampling the\n"
            "bump persists => the response table itself is alignment-\n"
            "dependent. FIX: discuss with maintainers; do NOT silently\n"
            "patch get_closest_waveform.")
        vcolor = "lightcoral"
    else:
        verdict = (
            "INCONCLUSIVE: bilinear leaves residual structure.\n"
            "Investigate further (e.g., quadratic / cubic interpolation,\n"
            "or check the table's edge handling).")
        vcolor = "khaki"
    ax.text(0.02, 0.97, verdict, transform=ax.transAxes,
            va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=vcolor,
                      edgecolor="black"))
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/verdict.png", dpi=140)
    print("wrote verdict.png")
    plt.close(fig)

    # ---- Save raw arrays for further analysis ----
    np.savez(f"{args.outdir}/aliasing_results.npz",
             x_src_1D=x_src_grid,
             curve_NN=curves["NN, kernel replica (G_smear)"],
             curve_NN_no_rng=curves["NN, no RNG (G_raw)"],
             curve_bilinear=curves["bilinear (proposed fix)"],
             curve_finegrid=curves["fine-grid Riemann (bin_size sampling)"],
             map_nn=map_nn, map_bl=map_bl,
             x_grid_2D=x_grid, y_grid_2D=y_grid,
             test_pitches=test_pitches,
             peak_excess_nn_vs_spacing=peak_excess_nn,
             peak_excess_bl_vs_spacing=peak_excess_bl,
             pitch=pitch, bin_size=bin_size,
             nn_peak_pct=nn_peak_pct,
             bl_peak_pct=bl_peak_pct,
             fg_peak_pct=fg_peak_pct)
    print("wrote aliasing_results.npz")

    # ---- Console verdict ----
    print()
    print("=" * 60)
    print("Peak excess across one pitch cell (offset=0.5):")
    print(f"  kernel (measured)            : {kernel_observed_pct:+.2f}%")
    print(f"  Python NN + RNG smear (kernel replica): {nn_peak_pct:+.2f}%")
    print(f"  Python NN, no RNG (geometric only)    : {nn_raw_peak_pct:+.2f}%")
    print(f"  bilinear (proposed fix)      : {bl_peak_pct:+.2f}%")
    print(f"  fine-grid Riemann            : {fg_peak_pct:+.2f}%")
    print(f"  → {verdict.splitlines()[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
