"""Generate k-grid density plot and convergence comparison for the paper."""

import numpy as np
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import camb
import nanocmb
from nanocmb import (compute_background, compute_thermodynamics, compute_cls,
                     k_grid, params, c_km_s)


def get_camb_reference():
    """High-accuracy CAMB reference spectra."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['h']*100, ombh2=params['omega_b_h2'],
                       omch2=params['omega_c_h2'], tau=params['tau_reion'],
                       nnu=params['N_eff'], num_massive_neutrinos=0, mnu=0,
                       TCMB=params['T_cmb'], YHe=params['Y_He'])
    pars.set_accuracy(AccuracyBoost=3)
    pars.InitPower.set_params(As=params['A_s'], ns=params['n_s'], r=0)
    pars.set_for_lmax(2600, lens_potential_accuracy=0)
    pars.DoLensing = False
    pars.WantTensors = False
    pars.max_l = 2600
    pars.max_eta_k = 25000
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['unlensed_scalar']
    ells = np.arange(totCL.shape[0])
    return ells, totCL[:, 0]


def compute_rms(ells_n, DlTT_n, ells_c, DlTT_c, ell_lo=30):
    """RMS residual between nanoCMB and CAMB for ell >= ell_lo."""
    ell_max = min(ells_n[-1], len(DlTT_c) - 1)
    mask_c = (ells_c >= ell_lo) & (ells_c <= ell_max)
    mask_n = (ells_n >= ell_lo) & (ells_n <= ell_max)
    return np.sqrt(np.mean((DlTT_n[mask_n] / DlTT_c[mask_c] - 1)**2))


def make_density_plot(bg, thermo):
    """Node density dN/d(ln k) for optimal vs uniform grids."""
    k_ode = k_grid(N=200, mode="ode", bg=bg, thermo=thermo, params=params)
    k_cl = k_grid(N=4000, mode="cl", bg=bg, thermo=thermo, params=params,
                  k_min=k_ode[0], k_max=k_ode[-1])
    k_uniform_ode = np.geomspace(k_ode[0], k_ode[-1], 200)
    k_uniform_cl = np.geomspace(k_cl[0], k_cl[-1], 4000)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for ax, k_opt, k_uni, label, N in [
        (axes[0], k_ode, k_uniform_ode, 'ODE grid', 200),
        (axes[1], k_cl, k_uniform_cl, r'$C_\ell$ grid', 4000),
    ]:
        lnk_opt = np.log(k_opt)
        lnk_uni = np.log(k_uni)
        density_opt = 1.0 / np.diff(lnk_opt)
        density_uni = 1.0 / np.diff(lnk_uni)
        k_mid_opt = np.exp(0.5 * (lnk_opt[:-1] + lnk_opt[1:]))
        k_mid_uni = np.exp(0.5 * (lnk_uni[:-1] + lnk_uni[1:]))

        ax.loglog(k_mid_opt, density_opt, 'C0-', lw=1.2, label='Optimal')
        ax.loglog(k_mid_uni, density_uni, 'C1--', lw=1.0, alpha=0.7,
                  label=r'Uniform in $\ln k$')
        ax.set_ylabel(r'$dN/d\ln k$')
        ax.set_title(f'{label} ($N = {N}$)', fontsize=11)
        ax.legend(fontsize=9)

        for val, lab in [
            (1.0 / thermo['r_s'], r'$1/r_s$'),
            (thermo['k_D'], r'$k_D$'),
        ]:
            ax.axvline(val, color='gray', ls=':', lw=0.7, alpha=0.6)
            ax.text(val * 1.08, ax.get_ylim()[1] * 0.85, lab,
                    fontsize=8, color='gray', va='top')

    axes[1].set_xlabel(r'$k$ [Mpc$^{-1}$]')
    fig.tight_layout()
    fig.savefig('plots/kgrid_density.png', dpi=150)
    fig.savefig('paper/figures/kgrid_density.pdf', dpi=150)
    plt.close(fig)
    print("Saved plots/kgrid_density.png and paper/figures/kgrid_density.pdf")


def make_convergence_plot(bg, thermo):
    """RMS residual vs N_ode for optimal and uniform grids."""
    print("Computing CAMB reference...")
    ells_camb, DlTT_camb = get_camb_reference()

    Ns_test = [50, 75, 100, 150, 200, 300, 400]
    rms_optimal = []
    rms_uniform = []

    original_k_grid = nanocmb.k_grid

    for N_ode in Ns_test:
        for mode_label, use_uniform in [('optimal', False), ('uniform', True)]:
            if use_uniform:
                def uniform_k_grid(N, mode, bg, thermo, params,
                                   k_min=1e-5, k_max=0.5, **kw):
                    return np.geomspace(k_min, k_max, N)
                nanocmb.k_grid = uniform_k_grid
            else:
                nanocmb.k_grid = original_k_grid

            try:
                result = compute_cls(bg, thermo, {**params, 'ell_max': 2500},
                                     N_k_ode=N_ode)
                ells_n = result['ells']
                DlTT_n = result['Dl_TT']
                rms = compute_rms(ells_n, DlTT_n, ells_camb, DlTT_camb)
                print(f"  N_ode={N_ode:4d} {mode_label:8s}: RMS = {rms*100:.3f}%")

                if use_uniform:
                    rms_uniform.append(rms)
                else:
                    rms_optimal.append(rms)
            except Exception as e:
                print(f"  N_ode={N_ode:4d} {mode_label:8s}: FAILED ({e})")
                if use_uniform:
                    rms_uniform.append(np.nan)
                else:
                    rms_optimal.append(np.nan)

    nanocmb.k_grid = original_k_grid

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(Ns_test, np.array(rms_optimal) * 100, 'C0o-', label='Optimal', ms=5)
    ax.semilogy(Ns_test, np.array(rms_uniform) * 100, 'C1s--',
                label=r'Uniform in $\ln k$', ms=5)
    ax.axhline(0.1, color='gray', ls=':', lw=0.7)
    ax.text(Ns_test[-1] * 0.95, 0.11, '0.1%', fontsize=8, color='gray', ha='right')
    ax.set_xlabel('Number of ODE $k$-modes')
    ax.set_ylabel(r'RMS residual vs CAMB [\%]')
    ax.legend(fontsize=9)
    ax.set_xlim(Ns_test[0] - 10, Ns_test[-1] + 30)
    fig.tight_layout()
    fig.savefig('plots/kgrid_convergence.png', dpi=150)
    fig.savefig('paper/figures/kgrid_convergence.pdf', dpi=150)
    plt.close(fig)
    print("Saved plots/kgrid_convergence.png and paper/figures/kgrid_convergence.pdf")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)
    make_density_plot(bg, thermo)
    make_convergence_plot(bg, thermo)
