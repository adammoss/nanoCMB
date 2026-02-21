"""Compare optimal_tau_grid with the hand-tuned build_tau_out from nanocmb."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanocmb import compute_background, compute_thermodynamics, build_tau_out, params
from opttau import optimal_tau_grid, diagnose_tau_grid, CosmoParams


def node_density(grid):
    """Compute empirical dn/dtau and midpoints."""
    dtau = np.diff(grid)
    mid = 0.5 * (grid[:-1] + grid[1:])
    return mid, 1.0 / dtau


def plot_comparison(tau_hand, tau_opt, label_hand, label_opt, title, savename,
                    cosmo=None):
    if cosmo is None:
        cosmo = CosmoParams()

    N = len(tau_hand)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel 1: Node placement ---
    ax = axes[0, 0]
    ax.eventplot([tau_hand], lineoffsets=1.5, linelengths=0.6,
                 colors="C3", label=label_hand)
    ax.eventplot([tau_opt], lineoffsets=0.5, linelengths=0.6,
                 colors="C0", label=label_opt)
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Optimal", "Hand-tuned"])
    ax.set_ylim(-0.2, 2.2)
    ax.set_title(f"Node placement (N={N})")
    ax.legend(loc="upper right", fontsize=9)

    # --- Panel 2: Sampling density ---
    ax = axes[0, 1]
    m_h, d_h = node_density(tau_hand)
    m_o, d_o = node_density(tau_opt)
    ax.semilogy(m_h, d_h, color="C3", label=label_hand, alpha=0.8)
    ax.semilogy(m_o, d_o, color="C0", label=label_opt, alpha=0.8)
    ax.axvline(cosmo.tau_star, color="grey", ls=":", alpha=0.5, label=r"$\tau_*$")
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_ylabel(r"Node density $dn/d\tau$ [1/Mpc]")
    ax.set_title("Sampling density")
    ax.legend(fontsize=9)

    # --- Panel 3: Zoom on recombination ---
    ax = axes[1, 0]
    lo = cosmo.tau_star - 5 * cosmo.delta_tau_rec
    hi = cosmo.tau_star + 10 * cosmo.delta_tau_rec

    # Visibility function
    tau_fine = np.linspace(lo, hi, 500)
    g = np.exp(-0.5 * ((tau_fine - cosmo.tau_star) / cosmo.delta_tau_rec) ** 2)
    ax2 = ax.twinx()
    ax2.fill_between(tau_fine, g, alpha=0.15, color="grey")
    ax2.set_ylabel(r"$g(\tau)$", color="grey", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="grey")

    for grid, color, label in [(tau_hand, "C3", label_hand),
                                (tau_opt, "C0", label_opt)]:
        mask = (grid > lo) & (grid < hi)
        gm = grid[mask]
        if len(gm) > 1:
            m, d = node_density(gm)
            ax.plot(m, d, color=color, label=label, alpha=0.8)
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_ylabel(r"$dn/d\tau$ [1/Mpc]")
    ax.set_xlim(lo, hi)
    ax.set_title("Zoom: recombination")
    ax.legend(fontsize=9, loc="upper left")

    # --- Panel 4: Local spacing ---
    ax = axes[1, 1]
    ax.semilogy(tau_hand[:-1], np.diff(tau_hand), color="C3",
                label=label_hand, alpha=0.8)
    ax.semilogy(tau_opt[:-1], np.diff(tau_opt), color="C0",
                label=label_opt, alpha=0.8)
    ax.axvline(cosmo.tau_star, color="grey", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\tau$ [Mpc]")
    ax.set_ylabel(r"$\Delta\tau$ [Mpc]")
    ax.set_title(r"Local spacing $\Delta\tau$")
    ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    print(f"Saved {savename}")
    plt.close(fig)


if __name__ == "__main__":
    # Get build_tau_out (needs bg/thermo for tau_star, z_reion)
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)
    tau0 = bg['tau0']

    tau_hand = build_tau_out(thermo, tau0)
    N_hand = len(tau_hand)
    print(f"build_tau_out: N={N_hand}, range=[{tau_hand[0]:.1f}, {tau_hand[-1]:.1f}]")

    # Optimal grid matched to same N and range
    tau_opt = optimal_tau_grid(
        N=N_hand, k_max=0.3,
        tau_min=tau_hand[0], tau_max=tau_hand[-1],
    )
    print(f"optimal_tau_grid: N={len(tau_opt)}, range=[{tau_opt[0]:.1f}, {tau_opt[-1]:.1f}]")
    print()

    diagnose_tau_grid(tau_hand, label="build_tau_out (hand-tuned)")
    diagnose_tau_grid(tau_opt, label="optimal_tau_grid")

    plot_comparison(
        tau_hand, tau_opt,
        "build_tau_out", "optimal",
        f"Optimal tau grid vs hand-tuned build_tau_out (N={N_hand})",
        "plots/tau_grid_comparison.png",
    )
