"""
Run nanoCMB with the default hand-tuned k-grid and the optimal ODE k-grid,
then compare the resulting C_ell spectra.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from nanocmb import compute_background, compute_thermodynamics, compute_cls, params, k_grid


def build_k_arr(k_min=4.0e-5, k_max=0.45, n_low=40, n_mid=180, n_mid_hi=70, n_high=50):
    """Hand-tuned ODE k-grid (legacy reference)."""
    k_low = np.logspace(np.log10(k_min), np.log10(0.008), n_low)
    k_mid = np.linspace(0.008, 0.18, n_mid)
    k_mid_hi = np.linspace(0.18, 0.30, n_mid_hi)
    k_high = np.linspace(0.30, k_max, n_high)
    return np.unique(np.concatenate([k_low, k_mid, k_mid_hi, k_high]))


def main():
    # Compute background and thermodynamics (shared)
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)

    # --- Run with default hand-tuned grid ---
    print("\n" + "=" * 60)
    print("RUN 1: Default build_k_arr")
    print("=" * 60)
    t0 = time.time()
    result_default = compute_cls(bg, thermo, params)
    t_default = time.time() - t0
    print(f"Time: {t_default:.1f}s")

    # --- Run with optimal ODE grid (same N, same k range) ---
    k_hand = build_k_arr()
    k_opt = k_grid(N=len(k_hand), mode="ode", bg=bg, thermo=thermo, params=params,
                           k_min=k_hand[0], k_max=k_hand[-1])

    print("\n" + "=" * 60)
    print(f"RUN 2: Optimal ODE grid (N={len(k_opt)})")
    print("=" * 60)
    t0 = time.time()
    result_opt = compute_cls(bg, thermo, params, k_arr=k_opt)
    t_opt = time.time() - t0
    print(f"Time: {t_opt:.1f}s")

    # --- Compare ---
    ells = result_default['ells']

    DlTT_def = result_default['Dl_TT']
    DlEE_def = result_default['Dl_EE']
    DlTE_def = result_default['Dl_TE']

    DlTT_opt = result_opt['Dl_TT']
    DlEE_opt = result_opt['Dl_EE']
    DlTE_opt = result_opt['Dl_TE']

    # Fractional differences
    frac_TT = (DlTT_opt - DlTT_def) / np.maximum(np.abs(DlTT_def), 1e-30)
    frac_EE = (DlEE_opt - DlEE_def) / np.maximum(np.abs(DlEE_def), 1e-30)
    # For TE use absolute difference normalised by sqrt(TT*EE)
    te_norm = np.sqrt(np.abs(DlTT_def) * np.abs(DlEE_def))
    te_norm = np.maximum(te_norm, 1e-30)
    frac_TE = (DlTE_opt - DlTE_def) / te_norm

    print(f"\n{'Spectrum':<8} {'max |frac diff|':<20} {'rms frac diff':<20}")
    print("-" * 50)
    for name, fd in [("TT", frac_TT), ("EE", frac_EE), ("TE", frac_TE)]:
        print(f"{name:<8} {np.max(np.abs(fd)):<20.6f} {np.sqrt(np.mean(fd**2)):<20.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Left column: spectra overlay
    for ax, Dl_def, Dl_opt, label in [
        (axes[0, 0], DlTT_def, DlTT_opt, "TT"),
        (axes[1, 0], DlEE_def, DlEE_opt, "EE"),
        (axes[2, 0], DlTE_def, DlTE_opt, "TE"),
    ]:
        ax.plot(ells, Dl_def, "C3", alpha=0.7, label="Default", lw=1)
        ax.plot(ells, Dl_opt, "C0", alpha=0.7, label="Optimal ODE", lw=1, ls="--")
        ax.set_ylabel(rf"$D_{{\ell}}^{{{label}}}$ [$\mu K^2$]")
        ax.legend(fontsize=9)
        ax.set_title(f"{label} spectrum")

    # Right column: fractional differences
    for ax, fd, label, color in [
        (axes[0, 1], frac_TT, "TT", "C0"),
        (axes[1, 1], frac_EE, "EE", "C1"),
        (axes[2, 1], frac_TE, "TE", "C2"),
    ]:
        ax.plot(ells, fd * 100, color=color, alpha=0.7, lw=0.8)
        ax.axhline(0, color="grey", ls="--", alpha=0.5)
        ax.set_ylabel(f"{label} difference [%]")
        ax.set_title(f"{label}: (optimal - default) / default")
        rms = np.sqrt(np.mean(fd**2)) * 100
        ax.text(0.98, 0.95, f"rms = {rms:.3f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    for ax in axes.flat:
        ax.set_xlabel(r"$\ell$")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Optimal ODE grid (N={len(k_opt)}) vs default build_k_arr (N={len(k_hand)})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/ode_grid_cl_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved plots/ode_grid_cl_comparison.png")


if __name__ == "__main__":
    main()
