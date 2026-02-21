"""Compare optimal_k_grid with hand-tuned grids from nanocmb.

Produces two figures:
  1. ODE grid comparison  (optimal ODE mode vs build_k_arr)
  2. Fine grid comparison (optimal CL mode vs k_fine from LOS integration)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanocmb import build_k_arr, optimal_k_grid, diagnose_grid

r_s = 145.0
acoustic_period = np.pi / r_s


def clean_grid(k_arr):
    """Remove near-duplicate points from piecewise-constructed grids."""
    mask = np.concatenate([[True], np.diff(np.log(k_arr)) > 1e-10])
    return k_arr[mask]


def node_density(k_arr):
    """Compute empirical dn/dlnk and midpoint k values."""
    lnk = np.log(k_arr)
    density = 1.0 / np.diff(lnk)
    k_mid = np.exp(0.5 * (lnk[:-1] + lnk[1:]))
    return k_mid, density


def build_k_fine(k_arr):
    """Reproduce the fine k-grid from nanocmb compute_cls."""
    nk_fine = 4000
    k_lin_start = max(0.002, k_arr[0])
    n_log = 80
    return np.unique(np.concatenate([
        np.logspace(np.log10(k_arr[0]), np.log10(k_lin_start), n_log),
        np.linspace(k_lin_start, k_arr[-1], nk_fine - n_log),
    ]))


def plot_comparison(k_hand, k_opt, label_hand, label_opt, title, savename):
    """Four-panel comparison plot for two k-grids."""
    N = len(k_hand)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel 1: Node placement tick marks ---
    ax = axes[0, 0]
    ax.eventplot([np.log10(k_hand)], lineoffsets=1.5, linelengths=0.6,
                 colors="C3", label=label_hand)
    ax.eventplot([np.log10(k_opt)], lineoffsets=0.5, linelengths=0.6,
                 colors="C0", label=label_opt)
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Optimal", "Hand-tuned"])
    ax.set_ylim(-0.2, 2.2)
    ax.set_title(f"Node placement (N={N})")
    ax.legend(loc="upper left", fontsize=9)

    # --- Panel 2: Sampling density ---
    ax = axes[0, 1]
    km_h, d_h = node_density(k_hand)
    km_o, d_o = node_density(k_opt)
    ax.plot(np.log10(km_h), d_h, color="C3", label=label_hand, alpha=0.8)
    ax.plot(np.log10(km_o), d_o, color="C0", label=label_opt, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel(r"Node density $dn/d\ln k$")
    ax.set_title("Sampling density comparison")
    ax.legend(fontsize=9)

    # --- Panel 3: Local dk spacing ---
    ax = axes[1, 0]
    ax.semilogy(k_hand[:-1], np.diff(k_hand), color="C3",
                label=label_hand, alpha=0.8)
    ax.semilogy(k_opt[:-1], np.diff(k_opt), color="C0",
                label=label_opt, alpha=0.8)
    ax.axhline(acoustic_period, color="grey", ls="--", alpha=0.5,
               label=r"$\pi/r_s$")
    ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_title(r"Local spacing $\Delta k$")
    ax.legend(fontsize=9)

    # --- Panel 4: Density ratio ---
    ax = axes[1, 1]
    k_lo = max(k_hand[0], k_opt[0]) * 1.05
    k_hi = min(k_hand[-1], k_opt[-1]) * 0.95
    k_common = np.geomspace(k_lo, k_hi, 500)
    d_h_interp = np.interp(np.log(k_common), np.log(km_h), d_h)
    d_o_interp = np.interp(np.log(k_common), np.log(km_o), d_o)
    ratio = d_o_interp / d_h_interp
    ax.plot(np.log10(k_common), ratio, color="C4")
    ax.axhline(1.0, color="grey", ls="--", alpha=0.5)
    ax.set_xlabel(r"$\log_{10}(k\,[\mathrm{Mpc}^{-1}])$")
    ax.set_ylabel("Density ratio (optimal / hand-tuned)")
    ax.set_title("Where optimal redistributes points")
    ax.fill_between(np.log10(k_common), 1, ratio,
                    where=ratio > 1, alpha=0.15, color="C0",
                    label="Optimal denser")
    ax.fill_between(np.log10(k_common), 1, ratio,
                    where=ratio < 1, alpha=0.15, color="C3",
                    label="Hand-tuned denser")
    ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    print(f"Saved {savename}")
    plt.close(fig)


# ==================================================================
# 1. ODE grid comparison
# ==================================================================
print("=" * 60)
print("ODE GRID COMPARISON")
print("=" * 60)

k_hand_ode = clean_grid(build_k_arr())
N_ode = len(k_hand_ode)
k_opt_ode = optimal_k_grid(N=N_ode, mode="ode",
                           k_min=k_hand_ode[0], k_max=k_hand_ode[-1])

diagnose_grid(k_hand_ode, label="build_k_arr (hand-tuned)")
diagnose_grid(k_opt_ode, label="optimal_k_grid (ODE mode)")

plot_comparison(
    k_hand_ode, k_opt_ode,
    "build_k_arr", "optimal (ODE)",
    f"Optimal ODE grid vs hand-tuned build_k_arr (N={N_ode})",
    "plots/k_grid_comparison.png",
)

# ==================================================================
# 2. Fine (C_ell LOS) grid comparison
# ==================================================================
print("=" * 60)
print("FINE / C_ELL GRID COMPARISON")
print("=" * 60)

k_hand_fine = clean_grid(build_k_fine(build_k_arr()))
N_fine = len(k_hand_fine)
k_opt_fine = optimal_k_grid(N=N_fine, mode="cl",
                            k_min=k_hand_fine[0], k_max=k_hand_fine[-1])

diagnose_grid(k_hand_fine, label="k_fine (hand-tuned)")
diagnose_grid(k_opt_fine, label="optimal_k_grid (CL mode)")

plot_comparison(
    k_hand_fine, k_opt_fine,
    "k_fine", r"optimal ($C_\ell$)",
    f"Optimal $C_\\ell$ grid vs hand-tuned k_fine (N={N_fine})",
    "plots/k_fine_comparison.png",
)
