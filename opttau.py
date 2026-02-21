"""
Optimal conformal time (tau) grid for the CMB line-of-sight integral.

The LOS integral is:
    Delta_ell(k) = int_0^tau_0 dtau S(k, tau) j_ell(k (tau_0 - tau))

The integrand has two regimes with very different character:

  1. Recombination (tau ~ tau_*): The visibility function g(tau)
     localises the source to a narrow peak of width delta_tau ~ 40 Mpc.
     Need enough points to resolve this peak and the acoustic structure
     within it.

  2. Late-time ISW (tau >> tau_*): The source is broad and slowly
     varying, but multiplied by a rapidly oscillating Bessel function
     j_ell(k chi). The oscillation period is ~2pi/k in tau, so higher
     k modes need finer sampling here.

The optimal grid is k-dependent. Two options are provided:
  - A single grid optimised for a given k_max (conservative)
  - A function returning per-k grids (more efficient if your code
    can handle non-uniform grids per k-mode)

Uses equidistribution of quadrature error with density ~ |I''|^(1/3).
"""

import numpy as np
from optk import CosmoParams


def _los_weight(
    tau: np.ndarray,
    cosmo: CosmoParams,
    k_max: float,
    isw_weight: float,
) -> np.ndarray:
    """
    Analytic curvature model for the LOS integrand.

    The integrand is S(k,tau) * j_ell(k*chi). The required tau
    resolution comes from different physics in each regime:

    Recombination: The visibility function g(tau) is so narrow
    (~40 Mpc) that its own curvature (1/delta_tau^2) dominates
    over Bessel oscillations for all practical k. The Bessel
    function is nearly constant across the peak. Resolution is
    set by g(tau), independent of k.

    ISW: The source d(Phi+Psi)/dtau is broad, peaked at the
    matter-dark energy transition (z ~ 0.5-1, tau ~ 10000-12000).
    At low k, the Bessel oscillation period is comparable to the
    source width and must be resolved. At high k, the oscillations
    self-cancel (Limber regime) and we just need to capture the
    cancellation, not resolve individual oscillations.
    """
    # --- Recombination source ---
    # Two components:
    #  1. Narrow visibility peak: curvature ~ 1/delta_tau_rec^2
    #  2. Broader source structure: the source functions oscillate
    #     at acoustic frequency k*c_s within the recombination epoch.
    #     This gives curvature ~ (k_max * c_s)^2 over a region of
    #     width ~ r_s around tau_star. This is typically larger than
    #     the visibility curvature and ensures adequate sampling of
    #     the oscillatory source structure.
    g_rec = np.exp(-0.5 * ((tau - cosmo.tau_star) / cosmo.delta_tau_rec) ** 2)
    vis_curv = g_rec / cosmo.delta_tau_rec ** 2

    c_s_rec = 1.0 / np.sqrt(3.0)
    acoustic_curv = (k_max * c_s_rec) ** 2
    g_broad = np.exp(-0.5 * ((tau - cosmo.tau_star) / cosmo.r_s) ** 2)
    source_curv = g_broad * acoustic_curv

    recomb = vis_curv + source_curv

    # --- Reionization ---
    # Broader peak for polarisation, same curvature logic
    g_reion = np.exp(-0.5 * ((tau - cosmo.tau_reion) / cosmo.delta_tau_reion) ** 2)
    reion = g_reion / cosmo.delta_tau_reion ** 2

    # --- Late-time ISW ---
    # The ISW source d(Phi+Psi)/dtau is non-zero during the
    # matter-DE transition. Model as a broad bump peaked at
    # tau ~ 0.85 * tau_0 (z ~ 0.5-1), with width ~ 0.25 * tau_0.
    # This is more physical than 1/chi^2 which diverges at tau_0.
    tau_de = 0.85 * cosmo.tau_0   # matter-DE transition epoch
    sigma_de = 0.25 * cosmo.tau_0  # width of ISW source
    isw_source = np.where(
        tau > cosmo.tau_star + 3 * cosmo.delta_tau_rec,
        np.exp(-0.5 * ((tau - tau_de) / sigma_de) ** 2),
        0.0,
    )

    # k-dependent effective curvature:
    # Low k (k * L_isw << 1): must resolve Bessel oscillations, curvature ~ k^2
    # High k (k * L_isw >> 1): Limber self-cancellation, curvature saturates
    L_isw = 2000.0  # ISW source correlation length (Mpc)
    k_eff_sq = k_max ** 2 / (1.0 + (k_max * L_isw) ** 2)

    # Normalise ISW relative to recombination peak curvature
    # The ISW contribution to C_ell is ~10% of recombination at low ell,
    # so scale its peak curvature to a fraction of the recombination peak.
    isw_curv = isw_source * k_eff_sq
    rec_peak = 1.0 / cosmo.delta_tau_rec ** 2
    isw_peak = np.max(isw_curv) + 1e-30
    isw_curv *= 0.3 * rec_peak / isw_peak

    return recomb + 0.3 * reion + isw_weight * isw_curv


def optimal_tau_grid(
    N: int = 200,
    k_max: float = 0.3,
    cosmo: CosmoParams | None = None,
    isw_weight: float = 0.3,
    tau_min: float = 1.0,
    tau_max: float | None = None,
    n_eval: int = 10000,
) -> np.ndarray:
    """
    Compute an optimal non-uniform tau grid for the LOS integral.

    Returns a single grid suitable for all k <= k_max. For the
    recombination contribution this is nearly k-independent (set
    by the visibility function width). The k-dependence enters
    through the ISW regime where higher k needs finer sampling
    to track Bessel oscillations.

    Parameters
    ----------
    N : int
        Total number of tau points (budget).
    k_max : float
        Maximum k-mode to support (1/Mpc). Higher k_max requires
        denser sampling in the ISW regime.
    cosmo : CosmoParams, optional
        Cosmological parameters. Planck 2018 defaults if None.
    isw_weight : float
        Relative weight for ISW regime (0 to 1).
        Set to 0 to concentrate entirely on recombination.
        Set higher if you need accurate low-ell ISW.
    tau_min : float
        Start of grid (Mpc). Source before recombination is
        negligible, but start early enough for interpolation.
    tau_max : float, optional
        End of grid. Defaults to tau_0.
    n_eval : int
        Internal fine-grid resolution for CDF construction.

    Returns
    -------
    tau_grid : ndarray, shape (N,)
        Optimal tau values in Mpc, sorted.
    """
    if cosmo is None:
        cosmo = CosmoParams()
    if tau_max is None:
        tau_max = cosmo.tau_0

    # Fine evaluation grid
    tau = np.linspace(tau_min, tau_max, n_eval)

    # Compute weight
    weight_raw = _los_weight(tau, cosmo, k_max, isw_weight)

    # Optimal density: |curvature|^(1/3) with floor
    # Floor ensures baseline coverage in the quiet region between
    # recombination and reionization (source is negligible but
    # interpolation still needs some support points).
    floor = 0.002 * np.max(weight_raw)
    density = (weight_raw + floor) ** (1.0 / 3.0)

    # Build and invert CDF
    dtau = tau[1] - tau[0]
    cdf = np.cumsum(density) * dtau
    cdf -= cdf[0]
    cdf /= cdf[-1]

    quantiles = np.linspace(0.0, 1.0, N)
    tau_grid = np.interp(quantiles, cdf, tau)

    tau_grid[0] = tau_min
    tau_grid[-1] = tau_max

    return tau_grid


def optimal_tau_grid_per_k(
    N: int = 200,
    k_values: np.ndarray | None = None,
    cosmo: CosmoParams | None = None,
    isw_weight: float = 0.3,
    tau_min: float = 1.0,
    tau_max: float | None = None,
    n_eval: int = 10000,
) -> dict[float, np.ndarray]:
    """
    Compute per-k optimal tau grids.

    For each k, the Bessel oscillation rate differs, so the optimal
    ISW sampling density changes. The recombination region is nearly
    the same for all k (dominated by the visibility function width).

    Returns a dict mapping k -> tau_grid.
    """
    if cosmo is None:
        cosmo = CosmoParams()
    if k_values is None:
        k_values = np.geomspace(1e-4, 0.3, 10)

    grids = {}
    for k in k_values:
        grids[float(k)] = optimal_tau_grid(
            N=N, k_max=k, cosmo=cosmo, isw_weight=isw_weight,
            tau_min=tau_min, tau_max=tau_max, n_eval=n_eval,
        )
    return grids


def diagnose_tau_grid(
    tau_grid: np.ndarray,
    label: str = "",
    cosmo: CosmoParams | None = None,
    k_ref: float = 0.1,
):
    """Print diagnostic information about a tau grid."""
    if cosmo is None:
        cosmo = CosmoParams()

    N = len(tau_grid)
    dtau = np.diff(tau_grid)

    print(f"=== {label + ' ' if label else ''}(N = {N}) ===")
    print(f"  tau range: [{tau_grid[0]:.1f}, {tau_grid[-1]:.1f}] Mpc")
    print(f"  dtau — min: {dtau.min():.2f}, max: {dtau.max():.2f}, "
          f"ratio: {dtau.max()/dtau.min():.1f}")

    # Points in recombination region
    rec_mask = np.abs(tau_grid - cosmo.tau_star) < 3 * cosmo.delta_tau_rec
    n_rec = np.sum(rec_mask)
    if n_rec > 1:
        dtau_rec = np.median(np.diff(tau_grid[rec_mask]))
        bessel_period_rec = 2 * np.pi / k_ref
        pts_bessel_rec = bessel_period_rec / dtau_rec
        print(f"  Recombination region (tau_* +/- 3*delta):")
        print(f"    {n_rec} points, median dtau = {dtau_rec:.2f} Mpc")
        print(f"    Points per visibility width: "
              f"{cosmo.delta_tau_rec / dtau_rec:.1f}")
        print(f"    Points per Bessel osc at k={k_ref}: "
              f"{pts_bessel_rec:.1f}")

    # ISW region
    isw_mask = tau_grid > cosmo.tau_star + 5 * cosmo.delta_tau_rec
    if np.sum(isw_mask) > 1:
        dtau_isw = np.median(np.diff(tau_grid[isw_mask]))
        bessel_period = 2 * np.pi / k_ref
        pts_per_bessel = bessel_period / dtau_isw
        chi_isw = cosmo.tau_0 - np.median(tau_grid[isw_mask])
        limber_param = k_ref * chi_isw
        print(f"  ISW region (k_ref={k_ref}):")
        print(f"    median dtau: {dtau_isw:.2f} Mpc")
        if limber_param > 10:
            print(f"    Limber regime (k*chi ~ {limber_param:.0f} >> 1): "
                  f"individual oscillations self-cancel")
        else:
            print(f"    Bessel period: {bessel_period:.1f} Mpc, "
                  f"pts/osc: {pts_per_bessel:.1f}")

    # Reionization region
    reion_mask = np.abs(tau_grid - cosmo.tau_reion) < 2 * cosmo.delta_tau_reion
    n_reion = np.sum(reion_mask)
    if n_reion > 1:
        dtau_reion = np.median(np.diff(tau_grid[reion_mask]))
        print(f"  Reionization region:")
        print(f"    {n_reion} points, median dtau = {dtau_reion:.2f} Mpc")
    print()


# ---------------------------------------------------------------
# Demo
# ---------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    cosmo = CosmoParams()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel 1: Node placement for different k_max ---
    ax = axes[0, 0]
    k_values = [0.01, 0.05, 0.2]
    for i, k_max in enumerate(k_values):
        tau = optimal_tau_grid(N=200, k_max=k_max)
        ax.eventplot([tau], lineoffsets=i, linelengths=0.6,
                     colors=f"C{i}", label=f"$k_{{\\max}}={k_max}$")
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f"$k_{{\\max}}={k}$" for k in k_values])
    ax.set_title("Node placement (N=200)")
    ax.legend(loc="upper right")

    # --- Panel 2: Sampling density ---
    ax = axes[0, 1]
    for k_max, color, ls in [(0.01, "C0", "--"), (0.05, "C1", "-"), (0.2, "C2", ":")]:
        tau = optimal_tau_grid(N=200, k_max=k_max)
        density = 1.0 / np.diff(tau)
        tau_mid = 0.5 * (tau[:-1] + tau[1:])
        ax.plot(tau_mid, density, color=color, ls=ls,
                label=f"$k_{{\\max}}={k_max}$")
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_ylabel(r"Node density $dn/d\tau$ [1/Mpc]")
    ax.set_title("Sampling density (N=200)")
    ax.set_yscale("log")
    ax.legend()

    # --- Panel 3: Zoom on recombination ---
    ax = axes[1, 0]
    tau_zoom = [cosmo.tau_star - 4 * cosmo.delta_tau_rec,
                cosmo.tau_star + 4 * cosmo.delta_tau_rec]

    # Visibility function
    tau_fine = np.linspace(tau_zoom[0], tau_zoom[1], 500)
    g = np.exp(-0.5 * ((tau_fine - cosmo.tau_star) / cosmo.delta_tau_rec) ** 2)
    ax2 = ax.twinx()
    ax2.fill_between(tau_fine, g, alpha=0.15, color="grey", label=r"$g(\tau)$")
    ax2.set_ylabel(r"$g(\tau)$ (arb.)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    for k_max, color, ls in [(0.01, "C0", "--"), (0.05, "C1", "-"), (0.2, "C2", ":")]:
        tau = optimal_tau_grid(N=200, k_max=k_max)
        mask = (tau > tau_zoom[0]) & (tau < tau_zoom[1])
        tau_m = tau[mask]
        if len(tau_m) > 1:
            density = 1.0 / np.diff(tau_m)
            tau_mid = 0.5 * (tau_m[:-1] + tau_m[1:])
            ax.plot(tau_mid, density, color=color, ls=ls,
                    label=f"$k_{{\\max}}={k_max}$")
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_ylabel(r"Node density $dn/d\tau$ [1/Mpc]")
    ax.set_xlim(tau_zoom)
    ax.set_title("Zoom: recombination region")
    ax.legend(loc="upper left")

    # --- Panel 4: Effect of budget N ---
    ax = axes[1, 1]
    k_max = 0.1
    for N, color, ls in [(100, "C0", "--"), (200, "C1", "-"), (500, "C2", ":")]:
        tau = optimal_tau_grid(N=N, k_max=k_max)
        density = 1.0 / np.diff(tau)
        tau_mid = 0.5 * (tau[:-1] + tau[1:])
        ax.plot(tau_mid, density, color=color, ls=ls, label=f"N={N}")
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_ylabel(r"Node density $dn/d\tau$ [1/Mpc]")
    ax.set_title(f"Varying N ($k_{{\\max}}={k_max}$)")
    ax.set_yscale("log")
    ax.legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Optimal $\tau$ grid for LOS integral",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/tau_grid_diagnostics.png", dpi=150, bbox_inches="tight")
    print("Saved plots/tau_grid_diagnostics.png\n")

    # Diagnostics
    for k_max in [0.003, 0.01, 0.1, 0.3]:
        diagnose_tau_grid(
            optimal_tau_grid(200, k_max=k_max),
            label=f"LOS (k_max={k_max})",
            k_ref=k_max,
        )

    # Uniform baseline
    diagnose_tau_grid(
        np.linspace(1.0, 14000.0, 200),
        label="Uniform baseline",
        k_ref=0.1,
    )
