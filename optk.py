"""
Optimal k-grid selection for CMB power spectrum computation.

Two modes:

  "cl"  — optimised for the C_ell integral:
           C_ell = int dk k^2 P(k) |Delta_ell(k)|^2
           Weight accounts for Bessel function windowing per ell,
           acoustic oscillations, and Silk damping.

  "ode" — optimised for solving the Boltzmann ODE / interpolating
           source functions S(k, tau) in k:
           Weight is ell-independent, based on acoustic oscillation
           curvature and the damping envelope only.

Uses equidistribution of quadrature error: optimal node density
is proportional to |I''|^(1/3) for trapezoidal rule.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class GridMode(str, Enum):
    CL = "cl"
    ODE = "ode"


@dataclass
class CosmoParams:
    """Standard cosmological parameters relevant to k-grid placement.

    Default values are Planck 2018 best-fit LCDM.
    All distances in comoving Mpc.
    """
    chi_star: float = 14000.0       # comoving distance to last scattering
    delta_chi: float = 40.0         # width of visibility function
    r_s: float = 145.0              # sound horizon at recombination
    k_D: float = 0.15               # Silk damping scale (1/Mpc)
    n_s: float = 0.965              # scalar spectral index
    chi_reion: float = 4500.0       # distance to reionization midpoint
    delta_chi_reion: float = 800.0  # width of reionization visibility
    chi_eq: float = 100.0           # comoving horizon at matter-radiation equality


def _weight_cl(
    k: np.ndarray,
    cosmo: CosmoParams,
    ell_min: int,
    ell_max: int,
    n_ell_samples: int,
    isw_weight: float,
) -> np.ndarray:
    """
    Analytic integrand curvature model for C_ell integration.

    Sums contributions over representative ell values. Each ell
    contributes a Bessel-windowed piece centred on k ~ ell/chi_*,
    modulated by acoustic oscillation curvature and Silk damping.
    """
    ells = np.unique(np.geomspace(ell_min, ell_max, n_ell_samples).astype(int))

    primordial = k ** (cosmo.n_s + 2)
    damping = np.exp(-2.0 * (k / cosmo.k_D) ** 2)

    total = np.zeros_like(k)

    for ell in ells:
        # Recombination: Bessel window at k ~ ell/chi_*
        k_peak = ell / cosmo.chi_star
        sigma_k = 1.0 / cosmo.delta_chi
        envelope = np.exp(-0.5 * ((k - k_peak) / (3.0 * sigma_k)) ** 2)

        # Curvature from acoustic oscillations vs smooth envelope
        acoustic_curv = (1.0 / cosmo.r_s) ** 2 * envelope
        smooth_curv = sigma_k ** 2 * envelope
        recomb = np.maximum(acoustic_curv, smooth_curv) * primordial * damping

        # ISW at low ell
        if ell < 100:
            k_isw = ell / cosmo.chi_reion
            sigma_isw = 1.0 / cosmo.delta_chi_reion
            isw_env = np.exp(-0.5 * ((k - k_isw) / (5.0 * sigma_isw)) ** 2)
            recomb += isw_weight * sigma_isw ** 2 * isw_env * primordial

        total += recomb

    return total


def _weight_ode(k: np.ndarray, cosmo: CosmoParams) -> np.ndarray:
    """
    Curvature model for ODE k-grid (source function interpolation).

    The source oscillates at pi/r_s in k (acoustic), with amplitude set
    by the primordial spectrum and damped by Silk diffusion. Uses
    |I''|^(1/3) equidistribution for the acoustic region, with two
    interpolation-motivated enhancements:

      Low k: the weight ~ k^(n_s+2) drops as ~k^3, starving low k of
      points. But accurate interpolation of S(k) at low k is needed for
      ISW and reionization (low-ell EE). A large density floor ensures
      roughly log-uniform coverage at low k.

      High k: Silk damping kills the amplitude, but S(k) still oscillates
      at pi/r_s. Akima interpolation needs ~5 points per period, so a
      damping floor prevents the spacing from growing too large.
    """
    primordial = k ** (cosmo.n_s + 2)
    damping = np.exp(-2.0 * (k / cosmo.k_D) ** 2)

    # Acoustic oscillation curvature
    acoustic_curv = (1.0 / cosmo.r_s) ** 2

    # Low k: smooth variation on scale ~1/chi_eq
    k_transition = 1.0 / cosmo.r_s
    smooth_curv = (k / k_transition) ** 2 * (1.0 / cosmo.chi_eq) ** 2
    curvature = np.maximum(acoustic_curv, smooth_curv)

    # Damping floor: ensure enough points per oscillation at high k
    weight = curvature * primordial * np.maximum(damping, 0.02)

    return weight


def optimal_k_grid(
    N: int = 100,
    mode: str | GridMode = "cl",
    k_min: float = 1e-5,
    k_max: float = 0.4,
    cosmo: CosmoParams | None = None,
    # C_ell mode parameters
    ell_min: int = 2,
    ell_max: int = 2500,
    n_ell_samples: int = 30,
    isw_weight: float = 0.3,
    # Internal resolution
    n_eval: int = 5000,
) -> np.ndarray:
    """
    Compute an optimal non-uniform k-grid for CMB computation.

    Parameters
    ----------
    N : int
        Total number of k-modes (budget).
    mode : str or GridMode
        "cl"  — optimised for C_ell integration (ell-dependent).
        "ode" — optimised for source function interpolation (ell-independent).
    k_min, k_max : float
        Bounds of the k-grid in 1/Mpc.
    cosmo : CosmoParams, optional
        Cosmological parameters. Planck 2018 defaults if None.
    ell_min, ell_max : int
        Multipole range (only used in "cl" mode).
    n_ell_samples : int
        Number of representative ells to sample (only used in "cl" mode).
    isw_weight : float
        Relative ISW weighting, 0 to 1 (only used in "cl" mode).
    n_eval : int
        Internal fine-grid resolution for CDF construction.

    Returns
    -------
    k_grid : ndarray, shape (N,)
        Optimal k-values in 1/Mpc, sorted.
    """
    if cosmo is None:
        cosmo = CosmoParams()

    mode = GridMode(mode)

    # Work in x = ln(k)
    x_min = np.log(k_min)
    x_max = np.log(k_max)
    x = np.linspace(x_min, x_max, n_eval)
    k = np.exp(x)

    # Compute weight function
    if mode is GridMode.CL:
        raw_weight = _weight_cl(k, cosmo, ell_min, ell_max, n_ell_samples, isw_weight)
    else:
        raw_weight = _weight_ode(k, cosmo)

    # Optimal density: |I''|^(1/3) with floor
    # For ODE mode, a larger floor ensures adequate log-uniform coverage
    # at low k (needed for ISW / reionization source interpolation).
    if mode is GridMode.ODE:
        floor = 0.005 * np.max(raw_weight)
    else:
        floor = 1e-6 * np.max(raw_weight)
    density = (raw_weight + floor) ** (1.0 / 3.0)

    # Build and invert CDF
    dx = x[1] - x[0]
    cdf = np.cumsum(density) * dx
    cdf -= cdf[0]
    cdf /= cdf[-1]

    quantiles = np.linspace(0.0, 1.0, N)
    x_optimal = np.interp(quantiles, cdf, x)
    k_grid = np.exp(x_optimal)

    k_grid[0] = k_min
    k_grid[-1] = k_max

    return k_grid


def diagnose_grid(k_grid: np.ndarray, label: str = "", cosmo: CosmoParams | None = None):
    """Print diagnostic information about a k-grid."""
    if cosmo is None:
        cosmo = CosmoParams()

    N = len(k_grid)
    dk = np.diff(k_grid)
    dlnk = np.diff(np.log(k_grid))

    print(f"=== {label + ' ' if label else ''}(N = {N}) ===")
    print(f"  k range: [{k_grid[0]:.2e}, {k_grid[-1]:.2e}] 1/Mpc")
    print(f"  dk  — min: {dk.min():.2e}, max: {dk.max():.2e}, "
          f"ratio: {dk.max()/dk.min():.1f}")
    print(f"  dlnk — min: {dlnk.min():.4f}, max: {dlnk.max():.4f}, "
          f"ratio: {dlnk.max()/dlnk.min():.1f}")

    acoustic_period = np.pi / cosmo.r_s
    k_acoustic = k_grid[(k_grid > 0.01) & (k_grid < 0.2)]
    if len(k_acoustic) > 1:
        dk_acoustic = np.median(np.diff(k_acoustic))
        pts = acoustic_period / dk_acoustic
        print(f"  Points per acoustic oscillation (0.01–0.2): {pts:.1f}")

    for lbl, k_t in [
        ("Horizon (ell~2)", 2.0 / cosmo.chi_star),
        ("First peak (ell~220)", 220.0 / cosmo.chi_star),
        ("Damping (k_D)", cosmo.k_D),
    ]:
        idx = np.argmin(np.abs(k_grid - k_t))
        local_dk = dk[min(idx, len(dk) - 1)]
        print(f"  At {lbl} (k={k_t:.4f}): local dk = {local_dk:.2e}")
    print()


# ---------------------------------------------------------------
# Demo
# ---------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel 1: CL vs ODE node placement (N=100) ---
    ax = axes[0, 0]
    for mode, color, yoff in [("cl", "C0", 1.5), ("ode", "C1", 0.5)]:
        k = optimal_k_grid(N=100, mode=mode)
        ax.eventplot([np.log10(k)], lineoffsets=yoff, linelengths=0.6,
                     colors=color, label=f"{mode.upper()}")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["ODE", r"$C_\ell$"])
    ax.set_ylim(-0.2, 2.2)
    ax.set_title("Node placement (N=100)")
    ax.legend(loc="upper left")

    # --- Panel 2: Density comparison CL vs ODE ---
    ax = axes[0, 1]
    for mode, color, ls in [("cl", "C0", "-"), ("ode", "C1", "--")]:
        k = optimal_k_grid(N=100, mode=mode)
        lnk = np.log(k)
        density = 1.0 / np.diff(lnk)
        k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
        ax.plot(np.log10(k_mid), density, color=color, ls=ls,
                label=f"{mode.upper()}")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title("Sampling density (N=100)")
    ax.legend()

    # --- Panel 3: ODE grid for different N ---
    ax = axes[1, 0]
    for N, color, ls in [(50, "C0", "--"), (100, "C1", "-"), (200, "C2", ":")]:
        k = optimal_k_grid(N=N, mode="ode")
        lnk = np.log(k)
        density = 1.0 / np.diff(lnk)
        k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
        ax.plot(np.log10(k_mid), density, color=color, ls=ls,
                label=f"N={N}")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title("ODE grid: varying N")
    ax.legend()

    # --- Panel 4: CL grid ell_max dependence ---
    ax = axes[1, 1]
    for ell_max, color in [(100, "C0"), (500, "C1"), (2500, "C2")]:
        k = optimal_k_grid(N=100, mode="cl", ell_min=2, ell_max=ell_max)
        lnk = np.log(k)
        density = 1.0 / np.diff(lnk)
        k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
        ax.plot(np.log10(k_mid), density, color=color,
                label=rf"$\ell_{{\max}}={ell_max}$")
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title(r"$C_\ell$ grid: varying $\ell_{\max}$")
    ax.legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Optimal k-grids: $C_\\ell$ integration vs ODE source functions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/k_grid_diagnostics.png", dpi=150, bbox_inches="tight")
    print("Saved diagnostic plot.\n")

    # Print diagnostics
    for mode in ["cl", "ode"]:
        diagnose_grid(optimal_k_grid(100, mode=mode), label=f"{mode.upper()} optimal")

    diagnose_grid(np.geomspace(1e-5, 0.4, 100), label="Log-uniform baseline")