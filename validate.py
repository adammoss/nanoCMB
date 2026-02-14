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
    TCMB=2.7255,
    YHe=0.245,
)
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
    idx_c = ell
    idx_n = ell - ell_min
    if idx_c < len(DlTT_camb) and idx_n < len(TT_nano):
        rTT = TT_nano[idx_n] / TT_camb[idx_c] if TT_camb[idx_c] != 0 else 0
        rEE = EE_nano[idx_n] / EE_camb[idx_c] if EE_camb[idx_c] != 0 else 0
        print(f"{ell:5d}  {TT_nano[idx_n]:10.2f}  {TT_camb[idx_c]:10.2f}  {rTT:8.3f}  {EE_nano[idx_n]:10.4f}  {EE_camb[idx_c]:10.4f}  {rEE:8.3f}")

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

    # --- TT spectrum ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ells_nano, TT_camb_interp, 'k-', label='CAMB', alpha=0.8, lw=1)
    ax.plot(ells_nano, DlTT_nano, 'r-', label='nanoCMB', alpha=0.7, lw=1)
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu$K$^2$]')
    ax.set_title('TT Power Spectrum')
    ax.legend()
    ax.set_xlim(2, 2500)
    fig.tight_layout()
    fig.savefig('plots/tt_spectrum.png', dpi=150)
    plt.close(fig)
    print("Saved plots/tt_spectrum.png")

    # --- EE spectrum ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ells_nano, EE_camb_interp, 'k-', label='CAMB', alpha=0.8, lw=1)
    ax.plot(ells_nano, DlEE_nano, 'b-', label='nanoCMB', alpha=0.7, lw=1)
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$\mathcal{D}_\ell^{EE}$ [$\mu$K$^2$]')
    ax.set_title('EE Power Spectrum')
    ax.legend()
    ax.set_xlim(2, 2500)
    fig.tight_layout()
    fig.savefig('plots/ee_spectrum.png', dpi=150)
    plt.close(fig)
    print("Saved plots/ee_spectrum.png")

    # --- TE spectrum ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ells_nano, TE_camb_interp, 'k-', label='CAMB', alpha=0.8, lw=1)
    ax.plot(ells_nano, DlTE_nano, 'g-', label='nanoCMB', alpha=0.7, lw=1)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$\mathcal{D}_\ell^{TE}$ [$\mu$K$^2$]')
    ax.set_title('TE Power Spectrum')
    ax.legend()
    ax.set_xlim(2, 2500)
    fig.tight_layout()
    fig.savefig('plots/te_spectrum.png', dpi=150)
    plt.close(fig)
    print("Saved plots/te_spectrum.png")

    # --- TT residuals ---
    fig, ax = plt.subplots(figsize=(8, 4))
    mask_pos = TT_camb_interp > 10
    residual = DlTT_nano[mask_pos] / TT_camb_interp[mask_pos] - 1
    ax.plot(ells_nano[mask_pos], residual * 100, 'r-', alpha=0.6, lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axhspan(-5, 5, alpha=0.1, color='green', label=r'$\pm$5%')
    ax.axhspan(-2, 2, alpha=0.1, color='blue', label=r'$\pm$2%')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel('nanoCMB / CAMB $-$ 1  [%]')
    ax.set_title('TT Residual')
    ax.set_xlim(2, 2500)
    ax.set_ylim(-10, 10)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('plots/tt_residuals.png', dpi=150)
    plt.close(fig)
    print("Saved plots/tt_residuals.png")

    # --- EE residuals ---
    fig, ax = plt.subplots(figsize=(8, 4))
    mask_pos = EE_camb_interp > 0.1
    residual = DlEE_nano[mask_pos] / EE_camb_interp[mask_pos] - 1
    ax.plot(ells_nano[mask_pos], residual * 100, 'b-', alpha=0.6, lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axhspan(-5, 5, alpha=0.1, color='green', label=r'$\pm$5%')
    ax.axhspan(-2, 2, alpha=0.1, color='blue', label=r'$\pm$2%')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel('nanoCMB / CAMB $-$ 1  [%]')
    ax.set_title('EE Residual')
    ax.set_xlim(2, 2500)
    ax.set_ylim(-10, 10)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('plots/ee_residuals.png', dpi=150)
    plt.close(fig)
    print("Saved plots/ee_residuals.png")

    # --- TE residuals ---
    fig, ax = plt.subplots(figsize=(8, 4))
    mask_pos = np.abs(TE_camb_interp) > 5
    residual = DlTE_nano[mask_pos] / TE_camb_interp[mask_pos] - 1
    ax.plot(ells_nano[mask_pos], residual * 100, 'g-', alpha=0.6, lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axhspan(-5, 5, alpha=0.1, color='green', label=r'$\pm$5%')
    ax.axhspan(-2, 2, alpha=0.1, color='blue', label=r'$\pm$2%')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel('nanoCMB / CAMB $-$ 1  [%]')
    ax.set_title('TE Residual')
    ax.set_xlim(2, 2500)
    ax.set_ylim(-20, 20)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('plots/te_residuals.png', dpi=150)
    plt.close(fig)
    print("Saved plots/te_residuals.png")

    # --- Background checks ---
    from nanocmb import compute_background, hubble, conformal_time, compute_thermodynamics, c_km_s, params
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)

    # CAMB thermo on a common z grid
    z_thermo = np.linspace(0.01, 1600, 2000)
    thermo_camb = results.get_background_redshift_evolution(z_thermo)

    # nanoCMB interpolated to same z grid (z_arr runs high→low, so reverse)
    z_rev = thermo['z_arr'][::-1]
    xe_nano = np.interp(z_thermo, z_rev, thermo['xe'][::-1])
    vis_nano = np.interp(z_thermo, z_rev, thermo['visibility'][::-1])
    opac_nano = np.interp(z_thermo, z_rev, thermo['opacity'][::-1])
    H_nano = np.array([hubble(1.0 / (1 + z), bg) * c_km_s for z in z_thermo])
    H_camb = results.hubble_parameter(z_thermo)

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))

    # --- Row 0: H(z) ---
    ax = axes[0, 0]
    ax.semilogy(z_thermo, H_camb, 'k-', label='CAMB', alpha=0.8)
    ax.semilogy(z_thermo, H_nano, 'r--', label='nanoCMB', alpha=0.7)
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Hubble Parameter')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(z_thermo, H_nano / H_camb, 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title('H(z) Ratio')
    ax.set_ylim(0.998, 1.002)

    # --- Row 1: x_e ---
    xe_camb = thermo_camb['x_e']

    ax = axes[1, 0]
    ax.semilogy(z_thermo, xe_camb, 'k-', label='CAMB', alpha=0.8)
    ax.semilogy(z_thermo, np.maximum(xe_nano, 1e-5), 'r--', label='nanoCMB', alpha=0.7)
    ax.set_ylabel(r'$x_e$')
    ax.set_title('Ionisation Fraction')
    ax.legend()

    ax = axes[1, 1]
    mask_xe = xe_camb > 1e-4
    ax.plot(z_thermo[mask_xe], xe_nano[mask_xe] / xe_camb[mask_xe], 'r-', alpha=0.6)
    ax.axhline(1, color='k', ls='--', lw=0.5)
    ax.set_ylabel('nanoCMB / CAMB')
    ax.set_title(r'$x_e$ Ratio')
    ax.set_ylim(0.9, 1.1)

    # --- Row 2: Visibility ---
    vis_camb = thermo_camb['visibility']
    vis_camb_norm = vis_camb / np.max(np.abs(vis_camb)) if np.max(np.abs(vis_camb)) > 0 else vis_camb
    vis_nano_norm = vis_nano / np.max(np.abs(vis_nano)) if np.max(np.abs(vis_nano)) > 0 else vis_nano
    mask_rec = (z_thermo > 800) & (z_thermo < 1300)

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
    opac_camb = thermo_camb['opacity']

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
    fig.savefig('plots/background_checks.png', dpi=150)
    plt.close(fig)
    print("Saved plots/background_checks.png")

except ImportError as e:
    print(f"matplotlib not available ({e}), skipping plots")
