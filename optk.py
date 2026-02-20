"""
Optimal k-grid selection for CMB C_ell computation.

Given a budget of N k-modes, places them to minimise quadrature error
in the integral:

    C_ell = int dk k^2 P(k) |Delta_ell(k)|^2

Uses an analytic model for the integrand weight based on known CMB physics:
- Bessel function window selecting k ~ ell/chi_* for each ell
- Acoustic oscillations with period pi/r_s
- Silk damping envelope exp(-k^2/k_D^2)
- Primordial spectrum k^(n_s - 1)

The optimal node density is proportional to |I''|^(1/3), equidistributed
across all ell values in the requested range.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CosmoParams:
    """Standard cosmological parameters relevant to k-grid placement.
    
    Default values are for Planck 2018 best-fit LCDM.
    All distances in Mpc (comoving).
    """
    chi_star: float = 14000.0     # comoving distance to last scattering
    delta_chi: float = 40.0       # width of visibility function
    r_s: float = 145.0            # sound horizon at recombination
    k_D: float = 0.15             # Silk damping scale (1/Mpc)
    n_s: float = 0.965            # scalar spectral index
    chi_reion: float = 4500.0     # rough distance to reionization midpoint
    delta_chi_reion: float = 800.0  # width of reionization visibility


def optimal_k_grid(
    N: int = 100,
    ell_min: int = 2,
    ell_max: int = 2500,
    k_min: float = 1e-5,
    k_max: float = 0.4,
    cosmo: CosmoParams | None = None,
    n_ell_samples: int = 30,
    isw_weight: float = 0.3,
    n_eval: int = 5000,
) -> np.ndarray:
    """
    Compute an optimal non-uniform k-grid for CMB C_ell integration.

    Parameters
    ----------
    N : int
        Total number of k-modes (budget).
    ell_min, ell_max : int
        Range of multipoles to optimise for.
    k_min, k_max : float
        Bounds of the k-grid in 1/Mpc.
    cosmo : CosmoParams, optional
        Cosmological parameters. Uses Planck 2018 defaults if None.
    n_ell_samples : int
        Number of representative ell values to sample for the weight.
    isw_weight : float
        Relative weight given to the ISW/low-k regime (0 to 1).
        Increase if you care about accuracy at ell < 30.
    n_eval : int
        Number of fine-grid points for building the CDF.

    Returns
    -------
    k_grid : ndarray, shape (N,)
        Optimal k-values in 1/Mpc, sorted.
    """
    if cosmo is None:
        cosmo = CosmoParams()

    # Work in x = ln(k) space throughout
    x_min = np.log(k_min)
    x_max = np.log(k_max)
    x = np.linspace(x_min, x_max, n_eval)
    k = np.exp(x)

    # Representative ell values, log-spaced
    ells = np.unique(np.geomspace(ell_min, ell_max, n_ell_samples).astype(int))

    # Primordial + volume element in log-k space: k^3 * k^(n_s-1) = k^(n_s+2)
    primordial = k ** (cosmo.n_s + 2)

    # Damping envelope
    damping = np.exp(-2.0 * (k / cosmo.k_D) ** 2)

    # Build weight as sum over ell of analytic integrand model
    total_weight = np.zeros_like(k)

    for ell in ells:
        # --- Recombination contribution ---
        # Bessel function window: peaks at k ~ ell/chi_*, width ~ 1/delta_chi
        k_peak = ell / cosmo.chi_star
        sigma_k = 1.0 / cosmo.delta_chi

        # Envelope: Gaussian window x acoustic oscillations
        # The second derivative of the oscillating part dominates |I''|
        # For cos^2(k r_s) type oscillations: |I''| ~ (2*pi/Delta_k)^2 * envelope
        # where Delta_k = pi/r_s
        envelope = np.exp(-0.5 * ((k - k_peak) / (3.0 * sigma_k)) ** 2)

        # Acoustic oscillation curvature contribution
        # |I''| ~ (1/r_s)^2 * envelope * primordial * damping
        acoustic_curvature = (1.0 / cosmo.r_s) ** 2 * envelope

        # Smooth (non-oscillatory) envelope curvature from the Gaussian window
        # |I''| ~ sigma_k^2 * envelope * primordial * damping
        smooth_curvature = sigma_k ** 2 * envelope

        # Take the larger of the two — acoustic dominates except very low ell
        recomb = np.maximum(acoustic_curvature, smooth_curvature) * primordial * damping

        # --- ISW contribution (low ell only) ---
        # Broad window at low k, no damping, no acoustic oscillations
        # Source extends over large chi range, so transfer function varies on
        # scale ~ 1/chi_0, giving fine structure at very low k
        if ell < 100:
            k_isw = ell / cosmo.chi_reion
            sigma_isw = 1.0 / cosmo.delta_chi_reion
            isw_envelope = np.exp(-0.5 * ((k - k_isw) / (5.0 * sigma_isw)) ** 2)
            isw_curvature = sigma_isw ** 2 * isw_envelope * primordial
            recomb += isw_weight * isw_curvature

        total_weight += recomb

    # Optimal node density: |I''|^(1/3) with a floor
    floor = 1e-6 * np.max(total_weight)
    density = (total_weight + floor) ** (1.0 / 3.0)

    # Build CDF in x = ln(k) space
    # Use trapezoidal integration
    dx = x[1] - x[0]
    cdf = np.cumsum(density) * dx
    cdf -= cdf[0]
    cdf /= cdf[-1]

    # Invert CDF to place N nodes
    quantiles = np.linspace(0.0, 1.0, N)
    x_optimal = np.interp(quantiles, cdf, x)
    k_grid = np.exp(x_optimal)

    # Ensure endpoints are included
    k_grid[0] = k_min
    k_grid[-1] = k_max

    return k_grid


def diagnose_grid(k_grid: np.ndarray, cosmo: CosmoParams | None = None):
    """
    Print diagnostic information about a k-grid.
    
    Reports spacing statistics and coverage of key physical scales.
    """
    if cosmo is None:
        cosmo = CosmoParams()

    N = len(k_grid)
    dk = np.diff(k_grid)
    dlnk = np.diff(np.log(k_grid))

    print(f"Grid diagnostics (N = {N})")
    print(f"  k range: [{k_grid[0]:.2e}, {k_grid[-1]:.2e}] 1/Mpc")
    print(f"  dk  — min: {dk.min():.2e}, max: {dk.max():.2e}, "
          f"ratio: {dk.max()/dk.min():.1f}")
    print(f"  dlnk — min: {dlnk.min():.4f}, max: {dlnk.max():.4f}, "
          f"ratio: {dlnk.max()/dlnk.min():.1f}")

    # Points per acoustic oscillation in peak region
    acoustic_period = np.pi / cosmo.r_s
    k_acoustic = k_grid[(k_grid > 0.01) & (k_grid < 0.2)]
    if len(k_acoustic) > 1:
        dk_acoustic = np.median(np.diff(k_acoustic))
        points_per_osc = acoustic_period / dk_acoustic
        print(f"  Points per acoustic oscillation (0.01-0.2): {points_per_osc:.1f}")

    # Coverage at key scales
    for label, k_target in [
        ("Horizon (ell~2)", 2.0 / cosmo.chi_star),
        ("First peak (ell~220)", 220.0 / cosmo.chi_star),
        ("Damping (k_D)", cosmo.k_D),
    ]:
        idx = np.argmin(np.abs(k_grid - k_target))
        local_dk = dk[min(idx, len(dk) - 1)]
        print(f"  At {label} (k={k_target:.4f}): local dk = {local_dk:.2e}")


# ---------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: Compare grids for different N ---
    ax = axes[0, 0]
    for N, color in [(50, "C0"), (100, "C1"), (200, "C2")]:
        k = optimal_k_grid(N=N)
        ax.eventplot([np.log10(k)], lineoffsets=N, linelengths=20,
                     colors=color, label=f"N={N}")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel("N")
    ax.set_title("k-mode placement")
    ax.legend()

    # --- Panel 2: Node density ---
    ax = axes[0, 1]
    for N, color, ls in [(50, "C0", "--"), (100, "C1", "-"), (200, "C2", ":")]:
        k = optimal_k_grid(N=N)
        # Empirical density: 1 / dk
        lnk = np.log(k)
        density = 1.0 / np.diff(lnk)
        k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
        ax.plot(np.log10(k_mid), density, color=color, ls=ls, label=f"N={N}")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title("Sampling density")
    ax.legend()

    # --- Panel 3: Compare optimal vs log-uniform vs linear ---
    ax = axes[1, 0]
    N = 100
    k_opt = optimal_k_grid(N=N)
    k_log = np.geomspace(1e-5, 0.4, N)
    k_lin = np.linspace(1e-5, 0.4, N)

    for k_arr, label, color in [
        (k_opt, "Optimal", "C1"),
        (k_log, "Log-uniform", "C3"),
        (k_lin, "Linear", "C4"),
    ]:
        lnk = np.log(k_arr)
        density = 1.0 / np.diff(lnk)
        k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
        ax.plot(np.log10(k_mid), density, label=label, color=color)
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title("Optimal vs naive grids (N=100)")
    ax.legend()

    # --- Panel 4: Effect of ell range ---
    ax = axes[1, 1]
    for ell_max, color in [(100, "C0"), (500, "C1"), (2500, "C2")]:
        k = optimal_k_grid(N=100, ell_min=2, ell_max=ell_max)
        lnk = np.log(k)
        density = 1.0 / np.diff(lnk)
        k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
        ax.plot(np.log10(k_mid), density, color=color,
                label=rf"$\ell_{{\max}}={ell_max}$")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title(r"Effect of $\ell_{\max}$")
    ax.legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Optimal k-grid for CMB power spectrum integration",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/k_grid_diagnostics.png", dpi=150, bbox_inches="tight")
    print("Saved diagnostic plot.")
    print()

    # Print diagnostics
    print("=== Optimal grid (N=100) ===")
    diagnose_grid(optimal_k_grid(100))
    print()
    print("=== Log-uniform grid (N=100) ===")
    diagnose_grid(np.geomspace(1e-5, 0.4, 100))
