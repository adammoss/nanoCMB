"""Compare nanoCMB output against CAMB and generate validation plots."""

import numpy as np
import camb
import os

# Same parameters as nanoCMB
pars = camb.CAMBparams()
pars.set_cosmology(
    H0=67.36,
    ombh2=0.02237,
    omch2=0.1200,
    tau=0.0544,
    nnu=3.044,
    num_massive_neutrinos=0,
    mnu=0,
    TCMB=2.7255,
    YHe=0.245,
)
pars.set_accuracy(AccuracyBoost=3)
pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
pars.set_for_lmax(2600, lens_potential_accuracy=0)
pars.DoLensing = False
pars.WantTensors = False
pars.max_l = 2600
pars.max_eta_k = 25000

results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['unlensed_scalar']

ells_camb = np.arange(totCL.shape[0])
DlTT_camb = totCL[:, 0]
DlEE_camb = totCL[:, 1]
DlTE_camb = totCL[:, 3]

# Load nanoCMB
data = np.load('nanocmb_output.npz')
ells_nano = data['ells']
DlTT_nano = data['DlTT']
DlEE_nano = data['DlEE']
DlTE_nano = data['DlTE']

# Restrict to common ℓ range
ell_max = min(ells_nano[-1], len(DlTT_camb) - 1)
ell_min = ells_nano[0]
mask_camb = (ells_camb >= ell_min) & (ells_camb <= ell_max)
mask_nano = (ells_nano >= ell_min) & (ells_nano <= ell_max)

ells = ells_camb[mask_camb]
TT_camb = DlTT_camb[mask_camb]
EE_camb = DlEE_camb[mask_camb]
TE_camb = DlTE_camb[mask_camb]
TT_nano = DlTT_nano[mask_nano]
EE_nano = DlEE_nano[mask_nano]
TE_nano = DlTE_nano[mask_nano]

# Print comparison at key ℓ values
print(f"{'ℓ':>5s}  {'TT_nano':>10s}  {'TT_CAMB':>10s}  {'ratio':>8s}  {'EE_nano':>10s}  {'EE_CAMB':>10s}  {'ratio':>8s}")
for ell in [2, 10, 30, 100, 220, 500, 1000, 1500, 2000]:
    idx = ell - ell_min
    if idx < len(TT_nano) and idx < len(TT_camb):
        rTT = TT_nano[idx] / TT_camb[idx] if TT_camb[idx] != 0 else 0
        rEE = EE_nano[idx] / EE_camb[idx] if EE_camb[idx] != 0 else 0
        print(f"{ell:5d}  {TT_nano[idx]:10.2f}  {TT_camb[idx]:10.2f}  {rTT:8.3f}  {EE_nano[idx]:10.4f}  {EE_camb[idx]:10.4f}  {rEE:8.3f}")

# Summary statistics
for name, nano, ref in [('TT', TT_nano, TT_camb), ('EE', EE_nano, EE_camb)]:
    for lmin, lmax in [(2, 30), (30, 500), (500, 2000), (2000, 2500)]:
        mask = (ells >= lmin) & (ells < lmax) & (ref > 0)
        if mask.sum() > 0:
            ratio = nano[mask] / ref[mask]
            print(f"  {name} ℓ=[{lmin:4d},{lmax:4d}): mean ratio={np.mean(ratio):.3f}, std={np.std(ratio):.3f}")

# Save CAMB reference for plotting
np.savez('camb_reference.npz', ells=ells_camb, DlTT=DlTT_camb, DlEE=DlEE_camb, DlTE=DlTE_camb)

# --- Generate validation plots ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('plots', exist_ok=True)

    # Interpolate CAMB to nanoCMB ℓ values
    TT_camb_interp = np.interp(ells_nano, ells_camb, DlTT_camb)
    EE_camb_interp = np.interp(ells_nano, ells_camb, DlEE_camb)
    TE_camb_interp = np.interp(ells_nano, ells_camb, DlTE_camb)

    # --- TT spectrum + residual ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax_top.plot(ells_nano, TT_camb_interp, 'k-', label='CAMB', alpha=0.8, lw=1)
    ax_top.plot(ells_nano, DlTT_nano, 'r-', label='nanoCMB', alpha=0.7, lw=1)
    ax_top.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
    ax_top.set_title('TT Power Spectrum')
    ax_top.legend()
    ax_top.set_xlim(2, 2500)
    mask_pos = TT_camb_interp > 10
    residual = DlTT_nano[mask_pos] / TT_camb_interp[mask_pos] - 1
    ax_bot.plot(ells_nano[mask_pos], residual * 100, 'r-', alpha=0.6, lw=0.8)
    ax_bot.axhline(0, color='k', ls='--', lw=0.5)
    ax_bot.axhspan(-1, 1, alpha=0.1, color='blue', label=r'$\pm$1%')
    ax_bot.set_xlabel(r'Multipole $\ell$')
    ax_bot.set_ylabel('Residual [%]')
    ax_bot.set_ylim(-3, 3)
    ax_bot.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig('plots/tt_spectrum.png', dpi=150)
    plt.close(fig)
    print("Saved plots/tt_spectrum.png")

    # --- EE spectrum + residual ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax_top.plot(ells_nano, EE_camb_interp, 'k-', label='CAMB', alpha=0.8, lw=1)
    ax_top.plot(ells_nano, DlEE_nano, 'b-', label='nanoCMB', alpha=0.7, lw=1)
    ax_top.set_ylabel(r'$\mathcal{D}_\ell^{EE}$ [$\mu$K$^2$]')
    ax_top.set_title('EE Power Spectrum')
    ax_top.legend()
    ax_top.set_xlim(2, 2500)
    mask_pos = EE_camb_interp > 0.1
    residual = DlEE_nano[mask_pos] / EE_camb_interp[mask_pos] - 1
    ax_bot.plot(ells_nano[mask_pos], residual * 100, 'b-', alpha=0.6, lw=0.8)
    ax_bot.axhline(0, color='k', ls='--', lw=0.5)
    ax_bot.axhspan(-1, 1, alpha=0.1, color='blue', label=r'$\pm$1%')
    ax_bot.set_xlabel(r'Multipole $\ell$')
    ax_bot.set_ylabel('Residual [%]')
    ax_bot.set_ylim(-3, 3)
    ax_bot.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig('plots/ee_spectrum.png', dpi=150)
    plt.close(fig)
    print("Saved plots/ee_spectrum.png")

    # --- TE spectrum + residual ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax_top.plot(ells_nano, TE_camb_interp, 'k-', label='CAMB', alpha=0.8, lw=1)
    ax_top.plot(ells_nano, DlTE_nano, 'g-', label='nanoCMB', alpha=0.7, lw=1)
    ax_top.axhline(0, color='gray', ls=':', lw=0.5)
    ax_top.set_ylabel(r'$\mathcal{D}_\ell^{TE}$ [$\mu$K$^2$]')
    ax_top.set_title('TE Power Spectrum')
    ax_top.legend()
    ax_top.set_xlim(2, 2500)
    mask_pos = np.abs(TE_camb_interp) > 5
    residual = DlTE_nano[mask_pos] / TE_camb_interp[mask_pos] - 1
    ax_bot.plot(ells_nano[mask_pos], residual * 100, 'g-', alpha=0.6, lw=0.8)
    ax_bot.axhline(0, color='k', ls='--', lw=0.5)
    ax_bot.axhspan(-2, 2, alpha=0.1, color='blue', label=r'$\pm$2%')
    ax_bot.set_xlabel(r'Multipole $\ell$')
    ax_bot.set_ylabel('Residual [%]')
    ax_bot.set_ylim(-10, 10)
    ax_bot.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig('plots/te_spectrum.png', dpi=150)
    plt.close(fig)
    print("Saved plots/te_spectrum.png")

    # --- Background checks ---
    from nanocmb import compute_background, hubble, conformal_time, compute_thermodynamics, c_km_s, params
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)

    # CAMB thermo on a common z grid
    z_thermo = np.linspace(0.01, 3000, 4000)
    thermo_camb = results.get_background_redshift_evolution(z_thermo)

    # nanoCMB interpolated to same z grid (z_arr runs high→low, so reverse)
    z_rev = thermo['z_arr'][::-1]
    xe_nano = np.interp(z_thermo, z_rev, thermo['xe'][::-1])
    vis_nano = np.interp(z_thermo, z_rev, thermo['visibility'][::-1])
    opac_nano = np.interp(z_thermo, z_rev, thermo['opacity'][::-1])
    H_nano = np.array([hubble(1.0 / (1 + z), bg) * c_km_s for z in z_thermo])
    H_camb = results.hubble_parameter(z_thermo)

    # --- Background plot: H(z) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(z_thermo, H_camb, 'k-', label='CAMB', alpha=0.8)
    ax.semilogy(z_thermo, H_nano, 'r--', label='nanoCMB', alpha=0.7)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Hubble Parameter')
    ax.legend()

    ax = axes[1]
    ax.plot(z_thermo, H_nano / H_camb, 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title('H(z) Ratio')
    ax.set_ylim(0.998, 1.002)

    fig.tight_layout()
    fig.savefig('plots/background_checks.png', dpi=150)
    plt.close(fig)
    print("Saved plots/background_checks.png")

    # --- Thermodynamics plot: x_e, visibility, opacity ---
    xe_camb = thermo_camb['x_e']
    vis_camb = thermo_camb['visibility']
    vis_camb_norm = vis_camb / np.max(np.abs(vis_camb)) if np.max(np.abs(vis_camb)) > 0 else vis_camb
    vis_nano_norm = vis_nano / np.max(np.abs(vis_nano)) if np.max(np.abs(vis_nano)) > 0 else vis_nano
    mask_rec = (z_thermo > 800) & (z_thermo < 1300)
    opac_camb = thermo_camb['opacity']

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))

    # --- Row 0: x_e (full range) ---
    ax = axes[0, 0]
    ax.semilogy(z_thermo, xe_camb, 'k-', label='CAMB', alpha=0.8)
    ax.semilogy(z_thermo, np.maximum(xe_nano, 1e-5), 'r--', label='nanoCMB', alpha=0.7)
    ax.set_ylabel(r'$x_e$')
    ax.set_title('Ionisation Fraction')
    ax.legend()

    ax = axes[0, 1]
    mask_xe = xe_camb > 1e-4
    ax.plot(z_thermo[mask_xe], xe_nano[mask_xe] / xe_camb[mask_xe], 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title(r'$x_e$ Ratio')
    ax.set_ylim(0.99, 1.01)

    # --- Row 1: Reionisation window (z < 30) with dense z grid ---
    z_reion = np.linspace(0.01, 30, 1000)
    xe_camb_reion = results.get_background_redshift_evolution(z_reion)['x_e']
    xe_nano_reion = np.interp(z_reion, z_rev, thermo['xe'][::-1])

    ax = axes[1, 0]
    ax.plot(z_reion, xe_camb_reion, 'k-', label='CAMB', alpha=0.8)
    ax.plot(z_reion, xe_nano_reion, 'r--', label='nanoCMB', alpha=0.7)
    ax.set_ylabel(r'$x_e$')
    ax.set_title('Reionisation Window')
    ax.legend()

    ax = axes[1, 1]
    mask_reion_ratio = xe_camb_reion > 0.01
    ax.plot(z_reion[mask_reion_ratio], xe_nano_reion[mask_reion_ratio] / xe_camb_reion[mask_reion_ratio], 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title('Reionisation Ratio')
    ax.set_ylim(0.9, 1.1)

    # --- Row 2: Visibility ---
    ax = axes[2, 0]
    ax.plot(z_thermo[mask_rec], vis_camb_norm[mask_rec], 'k-', label='CAMB', alpha=0.8)
    ax.plot(z_thermo[mask_rec], vis_nano_norm[mask_rec], 'r--', label='nanoCMB', alpha=0.7)
    ax.set_ylabel('Visibility (normalised)')
    ax.set_title(f'Visibility Function (z* = {thermo["z_star"]:.1f})')
    ax.legend()

    ax = axes[2, 1]
    mask_vis = mask_rec & (np.abs(vis_camb_norm) > 0.05)
    ax.plot(z_thermo[mask_vis], vis_nano_norm[mask_vis] / vis_camb_norm[mask_vis], 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title('Visibility Ratio')
    ax.set_ylim(0.9, 1.1)

    # --- Row 3: Opacity ---
    ax = axes[3, 0]
    ax.semilogy(z_thermo, np.maximum(opac_camb, 1e-10), 'k-', label='CAMB', alpha=0.8)
    ax.semilogy(z_thermo, np.maximum(opac_nano, 1e-10), 'r--', label='nanoCMB', alpha=0.7)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel(r"$\dot{\kappa}$ [Mpc$^{-1}$]")
    ax.set_title('Opacity')
    ax.legend()

    ax = axes[3, 1]
    mask_opac = opac_camb > 1e-6
    ax.plot(z_thermo[mask_opac], opac_nano[mask_opac] / opac_camb[mask_opac], 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title('Opacity Ratio')
    ax.set_ylim(0.9, 1.1)

    fig.tight_layout()
    fig.savefig('plots/thermo_checks.png', dpi=150)
    plt.close(fig)
    print("Saved plots/thermo_checks.png")

except ImportError as e:
    print(f"matplotlib not available ({e}), skipping plots")

# --- Transfer function comparison ---
try:
    if 'Delta_T' in data and 'k_fine' in data:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Delta_T_nano = data['Delta_T']
        Delta_E_nano = data['Delta_E']
        k_fine_nano = data['k_fine']
        ells_compute = data['ells_compute']

        trans = results.get_cmb_transfer_data()

        chi_max = bg['tau0']  # τ₀ ≈ χ_max
        chi_star = bg['tau0'] - thermo['tau_star']

        # Find ℓ values on both grids (exact match or within 1)
        common_ells = []
        for ell_c in trans.L:
            il_n = np.argmin(np.abs(ells_compute - ell_c))
            if abs(ells_compute[il_n] - ell_c) == 0:
                common_ells.append(int(ell_c))
        common_ells = np.array(common_ells)

        # Pick a representative spread: ~6 values across the range
        targets = [2, 10, 100, 500, 1000, 2000]
        ells_show = []
        for t in targets:
            idx = np.argmin(np.abs(common_ells - t))
            ell_pick = common_ells[idx]
            if ell_pick not in ells_show:
                ells_show.append(ell_pick)
        print(f"Transfer function ℓ values (exact match): {ells_show}")

        for spec, Delta_nano, src_ix, name in [
            ('TT', Delta_T_nano, 0, 'T'),
            ('EE', Delta_E_nano, 1, 'E'),
        ]:
            fig, axes = plt.subplots(2, len(ells_show), figsize=(4*len(ells_show), 7),
                                      gridspec_kw={'height_ratios': [3, 1]})
            for ic, ell in enumerate(ells_show):
                il_nano = np.argmin(np.abs(ells_compute - ell))
                ell_nano = ells_compute[il_nano]
                delta_nano = Delta_nano[il_nano, :]

                il_camb = np.argmin(np.abs(trans.L - ell))
                ell_camb = trans.L[il_camb]
                delta_camb = trans.delta_p_l_k[src_ix, il_camb, :]
                q_camb = trans.q

                # k range where Δ_ℓ(k) is non-negligible
                x_lo = max(0.0, ell - 4.0 * ell**(1.0/3.0))
                k_lo = max(1e-5, x_lo / chi_max * 0.5)
                k_hi = min(0.5, (ell + 1700) / chi_star * 1.2)

                ax = axes[0, ic]
                ax.semilogx(q_camb, delta_camb, 'k-', alpha=0.7, lw=1, label='CAMB')
                ax.semilogx(k_fine_nano, delta_nano, 'r--', alpha=0.7, lw=1, label='nanoCMB')
                ax.set_xlim(k_lo, k_hi)
                ax.set_title(f'{name}  $\\ell={ell}$', fontsize=11)
                ax.set_xticklabels([])
                if ic == 0:
                    ax.legend(fontsize=8)

                ax = axes[1, ic]
                delta_camb_interp = np.interp(k_fine_nano, q_camb, delta_camb)
                diff = delta_nano - delta_camb_interp
                ax.semilogx(k_fine_nano, diff, 'r-', alpha=0.6, lw=0.8)
                ax.axhline(0, color='k', ls='--', lw=0.5)
                ax.set_xlim(k_lo, k_hi)
                ax.set_ylabel('diff', fontsize=9)
                ax.set_xlabel(r'$k$ [Mpc$^{-1}$]')

            fig.suptitle(f'{spec} Transfer Functions $\\Delta_\\ell(k)$', fontsize=14)
            fig.tight_layout()
            fname = f'plots/transfer_{spec}.png'
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"Saved {fname}")
    else:
        print("Transfer functions not in npz — re-run nanocmb.py to generate")

except Exception as e:
    import traceback
    print(f"Transfer function comparison failed: {e}")
    traceback.print_exc()

# --- Perturbation evolution comparison ---
try:
    print("Computing perturbation evolution comparison...")
    from nanocmb import (setup_perturbation_grid, adiabatic_ics,
                            _boltzmann_rhs, _cubic_eval, IX_ETAK, IX_CLXC,
                            IX_CLXB, IX_VB, IX_G, NVAR)
    from scipy import integrate as sp_integrate

    pgrid = setup_perturbation_grid(bg, thermo)
    tau_star_val = thermo['tau_star']

    k_test = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]

    # Output times: from early to past recombination
    tau_pert = np.linspace(1.0, tau_star_val + 200, 500)
    tau_pert = tau_pert[tau_pert > 0.5]

    # Convert tau to z
    a_pert = np.array([float(_cubic_eval(pgrid['sp_a_x'], pgrid['sp_a_c'], t)) for t in tau_pert])
    z_pert = 1.0 / a_pert - 1.0

    # CAMB perturbation evolution (z must be in ascending order)
    z_for_camb = z_pert[::-1]
    pert_camb = results.get_redshift_evolution(
        k_test, z_for_camb,
        ['delta_cdm', 'delta_photon', 'v_baryon_cdm',
            'delta_baryon', 'pi_photon', 'etak'])

    # nanoCMB evolution for each k
    nano_sols = {}
    for k in k_test:
        tau_start = min(0.1 / k, tau_pert[0] * 0.5)
        tau_start = max(tau_start, 0.1)
        y0 = adiabatic_ics(k, tau_start, bg, pgrid)
        sol = sp_integrate.solve_ivp(
            lambda tau, y, _k=k: _boltzmann_rhs(tau, y, _k, pgrid['bg_vec'],
                pgrid['sp_a_x'], pgrid['sp_a_c'],
                pgrid['sp_op_x'], pgrid['sp_op_c']),
            [tau_start, tau_pert[-1]], y0,
            t_eval=tau_pert, method='Radau', rtol=1e-5, atol=1e-8,
            max_step=20.0)
        nano_sols[k] = sol
        print(f"  k={k:.3f}: {'OK' if sol.success else 'FAILED'}")

    var_labels = [r'$\delta_\gamma$', r'$\delta_c$', r'$v_b$',
                    r'$\delta_b$', r'$\pi_\gamma$', r'$\eta k$']
    nano_ix = [IX_G, IX_CLXC, IX_VB, IX_CLXB, IX_G + 2, IX_ETAK]
    camb_ix = [1, 0, 2, 3, 4, 5]

    # --- One figure per k mode: evolution (top) + residual (bottom) ---
    for ik, k in enumerate(k_test):
        sol = nano_sols[k]
        if not sol.success:
            continue

        fig, axes = plt.subplots(2, 6, figsize=(24, 7),
                                    gridspec_kw={'height_ratios': [3, 1]})

        for iv in range(6):
            y_camb = pert_camb[ik, ::-1, camb_ix[iv]]
            y_nano = sol.y[nano_ix[iv]]

            # Top: evolution
            ax = axes[0, iv]
            ax.plot(tau_pert, y_camb, 'k-', alpha=0.7, lw=1, label='CAMB')
            ax.plot(sol.t, y_nano, 'r--', alpha=0.7, lw=1, label='nanoCMB')
            ax.set_title(var_labels[iv], fontsize=12)
            ax.set_xticklabels([])
            if iv == 0:
                ax.legend(fontsize=8)

            # Bottom: residual
            ax = axes[1, iv]
            threshold = np.max(np.abs(y_camb)) * 0.01
            mask = np.abs(y_camb) > threshold
            ratio = np.where(mask, y_nano / y_camb, np.nan)
            ax.plot(tau_pert, ratio, 'r-', alpha=0.6, lw=0.8)
            ax.axhline(1, color='k', ls='--', lw=0.5)
            ax.axhspan(0.99, 1.01, alpha=0.15, color='blue')
            ax.set_ylim(0.95, 1.05)
            ax.set_ylabel('ratio', fontsize=9)
            ax.set_xlabel(r'$\tau$ [Mpc]')

        fig.suptitle(f'Perturbation Evolution  k = {k} Mpc$^{{-1}}$', fontsize=14)
        fig.tight_layout()
        fname = f'plots/perturbations_k{k:.3f}.png'
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")

except Exception as e:
    import traceback
    print(f"Perturbation comparison failed: {e}")
    traceback.print_exc()
