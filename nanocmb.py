"""
nanoCMB — A minimal CMB angular power spectrum calculator

Computes TT, EE, and TE angular power spectra for flat ΛCDM cosmologies
in ~1000 lines of readable Python. Think nanoGPT but for cosmology.

Approximations:
  - Flat geometry (K = 0)
  - Massless neutrinos only (no massive species)
  - Cosmological constant (w = -1 exactly)
  - Simplified RECFAST recombination (Peebles equation with fudge factor)
  - No lensing, no tensors, no isocurvature modes
  - First-order tight-coupling approximation

Dependencies: numpy, scipy (nothing else)

Units: distances in Mpc, time in Mpc (c = 1), H in Mpc⁻¹, k in Mpc⁻¹,
       densities as 8πGρ in Mpc⁻² (CAMB convention), temperatures in K
"""

import numpy as np
from scipy import integrate, interpolate, special

# ============================================================
# PHYSICAL CONSTANTS
# Natural units with c = 1, distances in Mpc
# ============================================================

c_km_s = 2.99792458e5               # speed of light (km/s)
k_B = 1.380649e-23                   # Boltzmann constant (J/K)
h_P = 6.62607015e-34                 # Planck constant (J·s)
hbar = h_P / (2 * np.pi)
m_e = 9.1093837015e-31              # electron mass (kg)
m_H = 1.673575e-27                   # hydrogen atom mass (kg)
sigma_T = 6.6524587321e-29           # Thomson cross section (m²)
G = 6.67430e-11                      # gravitational constant (m³/kg/s²)
Mpc_in_m = 3.0856775814913673e22    # 1 Mpc in metres
sigma_SB = 5.670374419e-8           # Stefan-Boltzmann constant (W/m²/K⁴)

# ============================================================
# PARAMETERS
# Default: Planck 2018 best-fit ΛCDM
# ============================================================

params = {
    'omega_b_h2': 0.02237,           # Ω_b h² — baryon density
    'omega_c_h2': 0.1200,            # Ω_c h² — CDM density
    'h': 0.6736,                     # H₀ / (100 km/s/Mpc)
    'n_s': 0.9649,                   # scalar spectral index
    'A_s': 2.1e-9,                   # scalar amplitude at k₀ = 0.05 Mpc⁻¹
    'tau_reion': 0.0544,             # reionisation optical depth
    'N_eff': 3.044,                  # effective number of neutrino species (all massless)
    'T_cmb': 2.7255,                 # CMB temperature today (K)
    'Y_He': 0.245,                   # helium mass fraction
    'k_pivot': 0.05,                 # pivot scale (Mpc⁻¹)
}


# ============================================================
# BACKGROUND COSMOLOGY
# Integrate the Friedmann equation to get H(a), η(a), χ(z)
# ============================================================

def setup_background(p):
    """Precompute background density parameters from cosmological parameters.

    All densities stored in CAMB convention: grho_i such that
    grhoa2 = Σ grho_i × a^(4-n_i) represents 8πGρ_tot a⁴,
    giving H(a) = √(grhoa2/3) / a² and dη/da = √(3/grhoa2).
    """
    h = p['h']
    H0 = 100 * h / c_km_s                          # H₀ in Mpc⁻¹ (c=1)
    grhocrit_h2 = 3 * (100.0 / c_km_s)**2          # 3(H₁₀₀/c)² in Mpc⁻²

    # Photon density: ρ_γ = (4σ_SB/c) T⁴, then Ω_γh² = ρ_γ/ρ_crit,100
    T_cmb = p['T_cmb']
    rho_gamma = 4 * sigma_SB / (c_km_s * 1e3)**3 * T_cmb**4  # J/m³ / c² → kg/m³
    H100_SI = 100 * 1e3 / Mpc_in_m
    rho_crit_100 = 3 * H100_SI**2 / (8 * np.pi * G)
    omega_gamma = rho_gamma / rho_crit_100

    grhog = grhocrit_h2 * omega_gamma                           # photons (∝ a⁻⁴)
    grhornomass = grhog * 7/8 * (4/11)**(4/3) * p['N_eff']     # massless neutrinos
    grhoc = grhocrit_h2 * p['omega_c_h2']                       # CDM (∝ a⁻³)
    grhob = grhocrit_h2 * p['omega_b_h2']                       # baryons (∝ a⁻³)

    # Cosmological constant: Ω_Λ = 1 - Ω_m - Ω_r (flat universe)
    omega_m = (p['omega_c_h2'] + p['omega_b_h2']) / h**2
    omega_r = omega_gamma * (1 + 7/8 * (4/11)**(4/3) * p['N_eff']) / h**2
    grhov = grhocrit_h2 * (1 - omega_m - omega_r) * h**2        # Λ (constant)

    # Thomson scattering: κ̇ = x_e × akthom / a²
    rho_b_SI = p['omega_b_h2'] * rho_crit_100
    n_H_Mpc = (1 - p['Y_He']) * rho_b_SI / m_H * Mpc_in_m**3
    akthom = (sigma_T / Mpc_in_m**2) * n_H_Mpc

    # Helium fraction by number: f_He = Y/(4(1-Y)) = n_He/n_H
    f_He = p['Y_He'] / (4 * (1 - p['Y_He']))

    return {
        'H0': H0, 'h': h,
        'grhog': grhog, 'grhornomass': grhornomass,
        'grhoc': grhoc, 'grhob': grhob, 'grhov': grhov,
        'akthom': akthom, 'f_He': f_He,
        'Y_He': p['Y_He'], 'T_cmb': T_cmb,
    }


def grhoa2(a, bg):
    """Total 8πGρ a⁴ — the key quantity in the Friedmann equation."""
    return (bg['grhog'] + bg['grhornomass']
            + (bg['grhoc'] + bg['grhob']) * a
            + bg['grhov'] * a**4)


def dtauda(a, bg):
    """dη/da = 1/(a²H) = √(3/grhoa2) in Mpc."""
    return np.sqrt(3.0 / grhoa2(a, bg))


def hubble(a, bg):
    """Hubble parameter H(a) = √(grhoa2/3) / a² in Mpc⁻¹ (c=1)."""
    return np.sqrt(grhoa2(a, bg) / 3.0) / a**2


def conformal_time(a, bg):
    """Conformal time η(a) = ∫₀ᵃ da'/(a'²H) in Mpc."""
    if np.isscalar(a):
        result, _ = integrate.quad(dtauda, 0, a, args=(bg,), limit=100, epsrel=1e-8)
        return result
    return np.array([conformal_time(ai, bg) for ai in a])


def sound_speed_squared(a, bg):
    """Photon-baryon sound speed c_s² = 1/(3(1+R)), R = 3ρ_b/(4ρ_γ)."""
    R = 0.75 * bg['grhob'] * a / bg['grhog']
    return 1.0 / (3.0 * (1.0 + R))


def compute_background(p):
    """Compute background quantities: η₀, sound horizon, etc."""
    bg = setup_background(p)
    bg['tau0'] = conformal_time(1.0, bg)
    return bg


# ============================================================
# RECOMBINATION
# Peebles equation for hydrogen ionisation fraction x_e(z),
# with Saha equilibrium at high redshift.
# Then compute optical depth τ(η) and visibility function g(η).
# ============================================================

# Recombination constants
E_ion_H = 13.605693122994  # eV, hydrogen ionisation energy
E_21_H = 10.2              # eV, Lyman-alpha energy (2→1)
lambda_alpha = 1.2157e-7   # Lyman-alpha wavelength (m)
Lambda_2s = 8.2246          # Two-photon decay rate 2s→1s (s⁻¹)
eV_to_K = 11604.518        # 1 eV in Kelvin
eV_to_J = 1.602176634e-19  # 1 eV in Joules
RECFAST_fudge = 1.00       # Fudge factor for Peebles equation (1.14 matches RECFAST; 1.00 matches CAMB's HyRec)


def compute_recombination(bg):
    """Solve for ionisation history x_e(z) using simplified RECFAST.

    Uses Saha equilibrium at high z (x_e > 0.99), then switches to the
    Peebles equation — a rate equation for the effective 3-level hydrogen atom.
    The key physics: recombination to the ground state produces a Lyman-continuum
    photon that immediately reionises another atom, so net recombination only
    proceeds through excited states (Case B) or two-photon decay from 2s.
    """
    T_cmb = bg['T_cmb']
    f_He = bg['f_He']

    # Baryon number density today (m⁻³), needed for recombination rates
    # n_b = n_H + n_He = n_H(1 + f_He)
    # We need n_H in SI for rate calculations
    H100_SI = 100 * 1e3 / Mpc_in_m
    rho_crit_100 = 3 * H100_SI**2 / (8 * np.pi * G)
    n_H_SI = (1 - bg['Y_He']) * (params['omega_b_h2'] * rho_crit_100) / m_H

    # Precompute temperature-independent quantities
    # C_r = (2π m_e k_B / h²) — appears in Saha equation
    C_r = 2 * np.pi * m_e * k_B / h_P**2       # m⁻² K⁻¹

    def saha_xe(z):
        """Saha equilibrium ionisation fraction for hydrogen.
        n_e n_p / n_H = (2πm_e k_B T)^(3/2) / h³ × exp(-E_ion/k_B T) / n_b
        With x_e = n_e/(n_H + n_He) ≈ n_e/n_H for helium fully recombined.
        """
        T = T_cmb * (1 + z)
        n_H = n_H_SI * (1 + z)**3
        # Right-hand side of Saha equation
        rhs = (1.0 / n_H) * (C_r * T)**1.5 * np.exp(-E_ion_H * eV_to_K / T)
        # Solve x²/(1-x) = rhs → x = (-rhs + √(rhs² + 4rhs))/2
        x_H = (-rhs + np.sqrt(rhs**2 + 4 * rhs)) / 2
        return min(x_H, 1.0)

    def peebles_rhs(z, x_H_arr):
        """Right-hand side of the Peebles equation dx_H/dz.

        The Peebles equation describes net recombination through excited states:
        recombination to n=2 competes with photoionisation from n=2, with escape
        of Lyman-alpha photons and two-photon decay providing the bottleneck.
        """
        x_H = float(x_H_arr[0]) if hasattr(x_H_arr, '__len__') else float(x_H_arr)
        x_H = max(x_H, 1e-30)
        T = T_cmb * (1 + z)
        n_H = n_H_SI * (1 + z)**3
        Hz = hubble(1.0 / (1 + z), bg) * c_km_s * 1e3 / Mpc_in_m  # H(z) in s⁻¹

        # Case-B recombination coefficient α_B (Pequignot, Petitjean & Boisson 1991)
        # Fit to recombination rate excluding captures to ground state
        t4 = T / 1e4
        alpha_B = 1e-19 * 4.309 * t4**(-0.6166) / (1 + 0.6703 * t4**0.5300)  # m³/s

        # Photoionisation rate from n=2: β = α_B × (2πm_e k_B T/h²)^(3/2) × exp(-B₂/k_BT)
        # where B₂ = E_ion/4 = 3.4 eV (binding energy of n=2)
        beta = alpha_B * (C_r * T)**1.5 * np.exp(-E_ion_H * eV_to_K / (4 * T))

        # Peebles K factor: K = λ_α³/(8πH(z))
        # This encodes the cosmological redshifting rate of Lyman-alpha photons
        K = lambda_alpha**3 / (8 * np.pi * Hz)

        # Lyman-alpha escape rate: photons escape when cosmological redshift
        # moves them out of the line before reabsorption
        # Λ_α = 1/(n_1s × K) where n_1s = n_H(1-x_H) is the ground-state density
        n_1s = n_H * max(1 - x_H, 1e-30)
        Ly_alpha_rate = 1.0 / (n_1s * K)

        # The Peebles C factor: probability that an atom reaching n=2 decays to
        # ground before being photoionised back to the continuum.
        # C = (Λ_2s + Λ_α) / (Λ_2s + Λ_α + β)
        # Two-photon decay (Λ_2s = 8.22 s⁻¹) and Ly-α escape compete with
        # photoionisation (β) from n=2.
        C_peebles = (Lambda_2s + Ly_alpha_rate) / (Lambda_2s + Ly_alpha_rate + beta)

        # dx_H/dz = C / [H(1+z)] × [x² n α_B f - β(1-x)exp(-E_Lyα/kT)]
        # The fudge factor multiplies α_B to correct for multilevel effects
        dxdz = (C_peebles / (Hz * (1 + z))) * (
            x_H**2 * n_H * alpha_B * RECFAST_fudge
            - beta * (1 - x_H) * np.exp(-E_21_H * eV_to_K / T)
        )
        return [dxdz]

    # Integrate from high to low redshift
    # Start fully ionised at z=1600, use Saha until x_e drops below 0.99
    z_start = 1600
    z_end = 0
    nz = 10000
    z_arr = np.linspace(z_start, z_end, nz + 1)
    xe_arr = np.zeros(nz + 1)

    # Phase 1: Saha equilibrium
    in_saha = True
    saha_switch_idx = 0
    for i, z in enumerate(z_arr):
        if in_saha:
            xe_arr[i] = saha_xe(z)
            if xe_arr[i] < 0.99:
                in_saha = False
                saha_switch_idx = i
                break
        else:
            break

    # Phase 2: Peebles ODE from Saha switch point to z=0
    z_ode = z_arr[saha_switch_idx:]
    x0 = [xe_arr[saha_switch_idx]]

    sol = integrate.solve_ivp(
        peebles_rhs, [z_ode[0], z_ode[-1]], x0,
        t_eval=z_ode, method='Radau', rtol=1e-6, atol=1e-10,
        max_step=2.0,
    )
    xe_arr[saha_switch_idx:] = sol.y[0]

    # Helium: assume fully recombined by z~1600, so x_e = x_H + f_He
    # (helium recombines earlier at z~1800 for He+ and z~6000 for He++)
    # At z > 1600 we add f_He (singly ionised helium), then helium is neutral
    xe_total = xe_arr.copy()
    xe_total[:saha_switch_idx] += f_He  # He contributes one electron per He atom above switch
    xe_total[saha_switch_idx:] += 0     # He recombined by this point

    return z_arr, xe_total


def compute_thermodynamics(bg, p):
    """Build thermodynamic tables: opacity, optical depth, visibility function.

    The visibility function g(η) = κ̇ e^{-τ} tells us the probability that a CMB
    photon last scattered at conformal time η. Its peak defines the surface of
    last scattering, and its width determines the thickness of that surface
    (which causes diffusion damping of small-scale anisotropies).
    """
    z_arr, xe_arr = compute_recombination(bg)

    # Add reionisation: tanh model matching the input τ_reion
    # x_e(z) = (f_re - x_freeze) × (1 + tanh((y_re - y)/Δy)) / 2 + x_freeze
    # where y = (1+z)^1.5, Δy = 1.5√(1+z_re) × Δz_re
    f_He = bg['f_He']
    f_re = 1.0 + f_He  # Full ionisation: H + singly-ionised He

    # Find z_re by bisection to match input τ_reion
    def apply_reionisation(z_re):
        """Apply tanh reionisation model and return modified x_e array."""
        delta_z = 0.5  # Width of reionisation transition
        xe_reion = xe_arr.copy()
        for i, z in enumerate(z_arr):
            y = (1 + z)**1.5
            y_re = (1 + z_re)**1.5
            dy = 1.5 * np.sqrt(1 + z_re) * delta_z
            xod = (y_re - y) / dy
            # Smooth step from freeze-out to fully ionised
            x_reion = f_re * (1 + np.tanh(xod)) / 2
            xe_reion[i] = max(xe_reion[i], x_reion)
        return xe_reion

    def compute_reion_optical_depth(xe_test):
        """Compute optical depth from z=0 to z_max ~ 50.

        This captures the reionisation contribution plus the small residual from
        recombination freeze-out. We don't integrate through the recombination
        epoch itself (τ >> 1 there), since τ_reion only measures the low-z part.
        """
        z_max = 50.0
        tau = 0
        for i in range(len(z_arr) - 1):
            z_mid = 0.5 * (z_arr[i] + z_arr[i+1])
            if z_mid > z_max:
                continue
            a_mid = 1.0 / (1 + z_mid)
            xe_mid = 0.5 * (xe_test[i] + xe_test[i+1])
            dz = abs(z_arr[i] - z_arr[i+1])
            deta = dtauda(a_mid, bg) / (1 + z_mid)**2 * dz
            tau += xe_mid * bg['akthom'] / a_mid**2 * deta
        return tau

    # Bisection to find z_re matching τ_reion
    target_tau = p['tau_reion']
    z_re_low, z_re_high = 2.0, 30.0
    for _ in range(60):
        z_re_mid = 0.5 * (z_re_low + z_re_high)
        xe_test = apply_reionisation(z_re_mid)
        tau_test = compute_reion_optical_depth(xe_test)
        if tau_test > target_tau:
            z_re_high = z_re_mid
        else:
            z_re_low = z_re_mid
    z_re = 0.5 * (z_re_low + z_re_high)
    xe_final = apply_reionisation(z_re)

    # Now build conformal time grid and thermodynamic quantities
    # Convert z grid to conformal time grid
    a_arr = 1.0 / (1 + z_arr)
    tau_arr = conformal_time(a_arr, bg)  # η(a) array, monotonically increasing

    # Opacity: κ̇ = dτ_optical/dη = x_e × n_H × σ_T / a² (in c=1 units)
    # = x_e × akthom / a²
    opacity = xe_final * bg['akthom'] / a_arr**2

    # Optical depth: τ(η) = ∫_η^η₀ κ̇ dη' (integrated from η to today)
    # Integrate from the end (z=0, η=η₀) backwards
    tau_optical = np.zeros_like(tau_arr)
    for i in range(len(tau_arr) - 2, -1, -1):
        deta = tau_arr[i+1] - tau_arr[i]
        tau_optical[i] = tau_optical[i+1] + 0.5 * (opacity[i] + opacity[i+1]) * deta

    # Visibility function: g(η) = κ̇ exp(-τ)
    exptau = np.exp(-tau_optical)
    visibility = opacity * exptau

    # Build interpolators on the conformal time grid
    # (reverse arrays so η is increasing for interpolation — it already is since
    # z_arr goes from high z to 0, so τ_arr goes from small η to large η)
    thermo = {
        'z_arr': z_arr,
        'a_arr': a_arr,
        'tau_arr': tau_arr,
        'xe': xe_final,
        'opacity': opacity,           # κ̇(η)
        'tau_optical': tau_optical,    # optical depth τ(η)
        'exptau': exptau,            # e^{-τ}
        'visibility': visibility,     # g(η) = κ̇ e^{-τ}
        'z_reion': z_re,
    }

    # Cubic spline interpolators for key quantities
    thermo['opacity_interp'] = interpolate.CubicSpline(tau_arr, opacity)
    thermo['exptau_interp'] = interpolate.CubicSpline(tau_arr, exptau)
    thermo['visibility_interp'] = interpolate.CubicSpline(tau_arr, visibility)

    # Find the peak of the visibility function (surface of last scattering)
    peak_idx = np.argmax(visibility)
    thermo['z_star'] = z_arr[peak_idx]
    thermo['tau_star'] = tau_arr[peak_idx]

    return thermo


# ============================================================
# PERTURBATIONS
# Evolve the coupled Einstein-Boltzmann equations in synchronous gauge
# (CDM frame, matching CAMB). The tight-coupling approximation handles
# the stiff photon-baryon coupling at early times; the full Boltzmann
# hierarchy takes over when the photon mean free path ~ wavelength.
# ============================================================

# Hierarchy truncation (increase for higher ℓ_max accuracy)
LMAXG = 15       # photon temperature: Θ₀ ... Θ_LMAXG
LMAXPOL = 15     # photon polarisation: E₂ ... E_LMAXPOL
LMAXNR = 15      # massless neutrinos: N₀ ... N_LMAXNR

# State vector layout (flat arrays for scipy ODE solver)
IX_ETAK = 0
IX_CLXC = 1
IX_CLXB = 2
IX_VB = 3
IX_G = 4                                    # Θ₀ at IX_G, Θ₁ at IX_G+1, etc.
IX_POL = IX_G + LMAXG + 1                   # E₂ at IX_POL, E₃ at IX_POL+1, etc.
IX_R = IX_POL + LMAXPOL - 1                 # N₀ at IX_R, N₁ at IX_R+1, etc.
NVAR = IX_R + LMAXNR + 1


def setup_perturbation_grid(bg, thermo):
    """Precompute a(τ) and background quantities on a fine conformal time grid.

    We need these at arbitrary times during ODE integration, so we build
    spline interpolators covering the full range from the deep radiation era
    (a ~ 10⁻⁹) to today (a = 1).
    """
    # Build τ(a) on a fine log-spaced grid via cumulative integration of dτ/da
    a_grid = np.logspace(-9, 0, 10000)
    dtauda_grid = np.array([dtauda(a, bg) for a in a_grid])
    tau_grid = integrate.cumulative_trapezoid(dtauda_grid, a_grid, initial=0)

    # a(τ) interpolator (the inverse mapping we need during integration)
    a_of_tau = interpolate.CubicSpline(tau_grid, a_grid)

    # Radiation-era expansion rate: ȧ = a²H ≈ √(grho_rad/3) × a  (when a is small)
    grho_rad = bg['grhog'] + bg['grhornomass']
    adotrad = np.sqrt(grho_rad / 3.0)

    # Build extended opacity interpolator covering all times.
    # Before the thermodynamics grid (z > 1600): fully ionised, κ̇ = (1+f_He) × akthom/a²
    # Within the thermodynamics grid: use the computed values
    tau_thermo = thermo['tau_arr']
    opac_thermo = thermo['opacity']

    # Extend to early times (τ < τ_thermo[0])
    tau_early = tau_grid[tau_grid < tau_thermo[0]]
    a_early = a_of_tau(tau_early)
    opac_early = (1.0 + bg['f_He']) * bg['akthom'] / a_early**2

    # Concatenate early + thermodynamics grids
    tau_ext = np.concatenate([tau_early, tau_thermo])
    opac_ext = np.concatenate([opac_early, opac_thermo])

    opacity_interp = interpolate.CubicSpline(tau_ext, opac_ext)

    return {
        'a_of_tau': a_of_tau,
        'opacity_interp': opacity_interp,
        'adotrad': adotrad,
        'grho_rad': grho_rad,
        'tau0': bg['tau0'],
    }


def get_bg_at_tau(tau, bg, pgrid):
    """Compute all background quantities at conformal time τ."""
    a = float(pgrid['a_of_tau'](tau))
    a2 = a * a

    # Species densities (8πGρa⁴ convention, divided by a to get 8πGρa² etc.)
    grhog_t = bg['grhog'] / a2               # photon: grho_g/a²
    grhor_t = bg['grhornomass'] / a2          # massless neutrino: grho_ν/a²
    grhoc_t = bg['grhoc'] / a                 # CDM: grho_c/a
    grhob_t = bg['grhob'] / a                 # baryon: grho_b/a

    # Expansion rate ȧ/a = H
    grho_a2 = grhog_t + grhor_t + grhoc_t + grhob_t + bg['grhov'] * a2
    adotoa = np.sqrt(grho_a2 / 3.0)          # = aH (i.e. ȧ/a in conformal time)

    # Opacity κ̇ = x_e × n_H × σ_T / a²
    opacity = float(pgrid['opacity_interp'](tau))
    opacity = max(opacity, 1e-30)  # prevent division by zero at late times

    # Baryon-photon ratio: R = 3ρ_b/(4ρ_γ) = (3/4)(grho_b a)/(grho_g)
    # In terms of _t quantities: pb43 = (4/3) grhog_t / grhob_t
    # photbar = grhog_t / grhob_t, and pb43 = 4/3 * photbar
    photbar = grhog_t / grhob_t
    pb43 = 4.0 / 3.0 * photbar       # = (4ρ_γ)/(3ρ_b)
    R = 1.0 / pb43                     # = 3ρ_b/(4ρ_γ)

    return {
        'a': a, 'a2': a2,
        'grhog_t': grhog_t, 'grhor_t': grhor_t,
        'grhoc_t': grhoc_t, 'grhob_t': grhob_t,
        'adotoa': adotoa,
        'opacity': opacity,
        'photbar': photbar, 'pb43': pb43, 'R': R,
    }


def adiabatic_ics(k, tau_start, bg, pgrid):
    """Set adiabatic initial conditions deep in the radiation era (kτ ≪ 1).

    These follow CAMB's initial() subroutine exactly. CAMB first populates an
    initv array with mode coefficients, then for adiabatic mode applies a sign
    flip (InitVec = -initv), and finally applies special prefactors when writing
    to the state vector (etak gets a factor of k/2). We combine all steps here.
    """
    tau = tau_start
    x = k * tau
    x2 = x * x

    grho_rad = pgrid['grho_rad']
    # Neutrino fraction of radiation
    Rv = bg['grhornomass'] / grho_rad     # ρ_ν/(ρ_ν + ρ_γ)
    Rg = 1.0 - Rv                          # ρ_γ/(ρ_ν + ρ_γ)
    Rp15 = 4 * Rv + 15                     # convenience combination

    # Matter-radiation ratio parameter (small in radiation era)
    om = (bg['grhob'] + bg['grhoc']) / np.sqrt(3.0 * grho_rad)
    omtau = om * tau

    y0 = np.zeros(NVAR)

    # CAMB's initv coefficients (before sign flip):
    #   initv(i_eta) = -2*(1 - x²/12*(-10/Rp15 + 1))
    #   initv(i_clxg) = -(1/3)*x²*(1 - omtau/5)
    # After sign flip: InitVec = -initv (adiabatic convention)
    # After y application: y(ix_etak) = -InitVec(i_eta)*k/2
    # Combined: etak = -k*(1 - x²/12*(-10/Rp15 + 1))

    # Metric perturbation: etak = k × η_synchronous ≈ -k at leading order
    y0[IX_ETAK] = -k * (1.0 - x2 / 12.0 * (-10.0 / Rp15 + 1.0))

    # Photon monopole and dipole (note: positive sign for adiabatic compression)
    clxg_init = x2 / 3.0 * (1.0 - omtau / 5.0)
    qg_init = x2 * x / 27.0 * (1.0 - omtau / 5.0)
    y0[IX_G] = clxg_init       # δ_γ
    y0[IX_G + 1] = qg_init    # q_γ

    # CDM and baryon density perturbations (3/4 of photon for adiabatic)
    y0[IX_CLXC] = 0.75 * clxg_init     # δ_c = (3/4)δ_γ
    y0[IX_CLXB] = 0.75 * clxg_init     # δ_b = (3/4)δ_γ
    y0[IX_VB] = 0.75 * qg_init         # v_b = (3/4)q_γ

    # Massless neutrinos
    y0[IX_R] = clxg_init                                       # δ_ν = δ_γ (adiabatic)
    y0[IX_R + 1] = (4 * Rv + 23) / Rp15 * x2 * x / 27.0      # q_ν
    y0[IX_R + 2] = -4.0 / 3.0 * x2 / Rp15 * (1.0 + omtau / 4.0 * (4*Rv - 5) / (2*Rv + 15))  # π_ν
    if LMAXNR >= 3:
        y0[IX_R + 3] = -4.0 / 21.0 / Rp15 * x2 * x           # N₃

    # All higher multipoles and polarisation start at zero
    return y0


def boltzmann_derivs(tau, y, k, bg, pgrid, thermo):
    """Right-hand side of the Boltzmann hierarchy: dy/dτ.

    Implements the synchronous gauge equations from CAMB's derivs() subroutine.
    During tight coupling (early times, high opacity), only the photon monopole
    and dipole are evolved; the quadrupole is computed algebraically.
    """
    B = get_bg_at_tau(tau, bg, pgrid)
    a, adotoa = B['a'], B['adotoa']
    opacity = B['opacity']
    grhog_t, grhor_t = B['grhog_t'], B['grhor_t']
    grhoc_t, grhob_t = B['grhoc_t'], B['grhob_t']
    pb43, photbar = B['pb43'], B['photbar']
    k2 = k * k

    # Extract state variables
    etak = y[IX_ETAK]
    clxc = y[IX_CLXC]
    clxb = y[IX_CLXB]
    vb = y[IX_VB]
    clxg = y[IX_G]         # photon monopole δ_γ
    qg = y[IX_G + 1]       # photon dipole
    pig = y[IX_G + 2]      # photon quadrupole
    clxr = y[IX_R]         # neutrino monopole δ_ν
    qr = y[IX_R + 1]       # neutrino dipole
    pir = y[IX_R + 2]      # neutrino quadrupole

    # Determine if tight coupling is active
    tight_coupling = (k / opacity < 0.01) and (1.0 / (opacity * tau) < 0.01)

    # Total density and velocity perturbations (Einstein constraint equations)
    dgrho = grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr
    dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
    dgpi = grhog_t * pig + grhor_t * pir

    # Synchronous gauge auxiliary variables
    # z = ḣ/(2k) — the metric trace perturbation rate
    z = (0.5 * dgrho / k + etak) / adotoa
    # σ — the metric shear (flat: Kf(1) = 1)
    sigma = z + 1.5 * dgq / k2

    # Polter: polarisation source combination Π = pig/10 + 3E₂/5
    E2 = y[IX_POL] if LMAXPOL >= 2 else 0.0
    polter = pig / 10.0 + 9.0 / 15.0 * E2

    # Free-streaming closure: 1/τ term (flat space limit of coth(kτ)/τ)
    cothxor = 1.0 / tau

    dy = np.zeros(NVAR)

    # --- Metric equation ---
    # ėtak = (1/2) dgq  (flat case, from Einstein equations)
    dy[IX_ETAK] = 0.5 * dgq

    # --- CDM: at rest in this gauge, only density evolves ---
    dy[IX_CLXC] = -k * z

    # --- Baryons ---
    dy[IX_CLXB] = -k * (z + vb)

    if tight_coupling:
        # Tight-coupling approximation: photon-baryon fluid is locked together.
        # Compute the photon quadrupole algebraically:
        pig_tc = 32.0 / 45.0 * k / opacity * (sigma + vb)
        # In tight coupling, E₂ = pig/4 (CAMB convention), so
        # polter = pig/10 + 9*(pig/4)/15 = pig/10 + 3pig/20 = pig/4
        polter = pig_tc / 4.0

        # Combined baryon-photon fluid velocity equation (CAMB convention):
        # v̇_b = (-ȧ/a v_b + k/4 pb43 (δ_γ - 2π_γ)) / (1+pb43)
        # Uses the photon quadrupole pig (not polter) — CAMB line 2389.
        vbdot = (-adotoa * vb + k / 4.0 * pb43 * (clxg - 2.0 * pig_tc)) / (1.0 + pb43)
        dy[IX_VB] = vbdot

        # Photon monopole: δ̇_γ = -k(4z/3 + q_g)
        dy[IX_G] = -k * (4.0 / 3.0 * z + qg)

        # Photon dipole: q̇_g derived from baryon equation (tightly coupled)
        # q̇_g = (4/3)(−v̇_b − ȧ/a v_b) / pb43 + k/3 δ_γ − 2k/3 π_γ
        qgdot = 4.0 / 3.0 * (-vbdot - adotoa * vb) / pb43 + k / 3.0 * clxg - 2.0 * k / 3.0 * pig_tc
        dy[IX_G + 1] = qgdot

        # Drive photon quadrupole towards its tight-coupling algebraic value.
        # CAMB treats pig and E₂ as algebraic (not ODE) variables during
        # tight coupling. We use fast relaxation (timescale 1/opacity ≈ photon
        # mean free path) so the ODE solver tracks the correct values.
        # This is essential: sigma, phi, and the source function all depend on
        # pig through dgpi, and the state variable must match pig_tc.
        dy[IX_G + 2] = opacity * (pig_tc - pig)

        if LMAXPOL >= 2:
            dy[IX_POL] = opacity * (pig_tc / 4.0 - E2)

    else:
        # Full Boltzmann hierarchy: evolve all multipoles

        # Baryon velocity with Thomson drag
        # v̇_b = -ȧ/a v_b - κ̇ × (4ρ_γ/3ρ_b)(v_b − 3q_g/4)
        # CAMB form: -adotoa*vb - photbar*opacity*(4/3*vb - qg)
        vbdot = -adotoa * vb - photbar * opacity * (4.0 / 3.0 * vb - qg)
        dy[IX_VB] = vbdot

        # Photon monopole (ℓ=0): δ̇_γ = -k(4z/3 + q_g)
        dy[IX_G] = -k * (4.0 / 3.0 * z + qg)

        # Photon dipole (ℓ=1): q̇_g from tight-coupling-like relation
        # In full hierarchy: q̇_g = (4/3)(-v̇_b - ȧ/a v_b)/pb43 + k/3 δ_γ - 2k/3 π_γ
        # This comes from the combined photon momentum equation with Thomson scattering
        qgdot = 4.0 / 3.0 * (-vbdot - adotoa * vb) / pb43 + k / 3.0 * clxg - 2.0 * k / 3.0 * pig
        dy[IX_G + 1] = qgdot

        # Photon quadrupole (ℓ=2): with scattering source
        # ṗig = (2k/5)q_g - (3k/5)Θ₃ - κ̇(pig - polter) + (8/15)kσ
        Theta3 = y[IX_G + 3] if LMAXG >= 3 else 0.0
        dy[IX_G + 2] = (2.0 * k / 5.0 * qg - 3.0 * k / 5.0 * Theta3
                        - opacity * (pig - polter) + 8.0 / 15.0 * k * sigma)

        # Higher photon multipoles (ℓ = 3 to LMAXG-1)
        for l in range(3, LMAXG):
            dy[IX_G + l] = (k * l / (2*l + 1) * y[IX_G + l - 1]
                            - k * (l + 1) / (2*l + 1) * y[IX_G + l + 1]
                            - opacity * y[IX_G + l])

        # Truncation at ℓ = LMAXG: free-streaming closure
        # Θ̇_ℓmax = k Θ_{ℓmax-1} - (ℓmax+1)/τ Θ_ℓmax - κ̇ Θ_ℓmax
        cothxor = 1.0 / tau   # flat space: coth(kτ) → 1/τ for large arg
        dy[IX_G + LMAXG] = (k * y[IX_G + LMAXG - 1]
                             - (LMAXG + 1) * cothxor * y[IX_G + LMAXG]
                             - opacity * y[IX_G + LMAXG])

        # --- Photon polarisation hierarchy (ℓ = 2 to LMAXPOL) ---
        # E₂: Ė₂ = -κ̇(E₂ - polter) - k/3 E₃
        E3 = y[IX_POL + 1] if LMAXPOL >= 3 else 0.0
        dy[IX_POL] = -opacity * (E2 - polter) - k / 3.0 * E3

        # Higher polarisation ℓ = 3 to LMAXPOL-1
        # Uses spin-2 coupling: polfac(ℓ) = (ℓ+3)(ℓ-1)/(ℓ+1)
        for l in range(3, LMAXPOL):
            idx = IX_POL + l - 2
            polfac_l = (l + 3) * (l - 1) / (l + 1)
            dy[idx] = (-opacity * y[idx]
                       + k * l / (2*l + 1) * y[idx - 1]
                       - polfac_l * k / (2*l + 1) * y[idx + 1])

        # Truncation at ℓ = LMAXPOL: (ℓmax+3)/τ sink for spin-2
        idx_last = IX_POL + LMAXPOL - 2
        polfac_last = (LMAXPOL + 3) * (LMAXPOL - 1) / (LMAXPOL + 1)
        dy[idx_last] = (-opacity * y[idx_last]
                        + k * LMAXPOL / (2*LMAXPOL + 1) * y[idx_last - 1]
                        - (LMAXPOL + 3) * cothxor * y[idx_last])

    # --- Massless neutrinos (no scattering, always full hierarchy) ---
    # N₀: δ̇_ν = -k(4z/3 + q_r)
    dy[IX_R] = -k * (4.0 / 3.0 * z + qr)

    # N₁: q̇_r = k/3(δ_ν - 2π_r)
    dy[IX_R + 1] = k / 3.0 * (clxr - 2.0 * pir)

    # N₂: π̇_r = 2k/5 q_r - 3k/5 N₃ + 8kσ/15
    N3 = y[IX_R + 3] if LMAXNR >= 3 else 0.0
    dy[IX_R + 2] = 2.0 * k / 5.0 * qr - 3.0 * k / 5.0 * N3 + 8.0 / 15.0 * k * sigma

    # Higher neutrino multipoles (ℓ = 3 to LMAXNR-1)
    for l in range(3, LMAXNR):
        dy[IX_R + l] = (k * l / (2*l + 1) * y[IX_R + l - 1]
                        - k * (l + 1) / (2*l + 1) * y[IX_R + l + 1])

    # Truncation at LMAXNR
    dy[IX_R + LMAXNR] = (k * y[IX_R + LMAXNR - 1]
                          - (LMAXNR + 1) * cothxor * y[IX_R + LMAXNR])

    return dy


def compute_source_functions(tau, y, k, bg, pgrid, thermo):
    """Compute CMB source function building blocks at a single (k, τ) point.

    After integration by parts, the temperature source decomposes into three
    Bessel channels: j_ℓ (ISW + monopole + quadrupole), j_ℓ' (Doppler),
    and j_ℓ'' (quadrupole). Returns the coefficients for each channel plus
    the E-mode source.
    """
    B = get_bg_at_tau(tau, bg, pgrid)
    a, adotoa, opacity = B['a'], B['adotoa'], B['opacity']
    grhog_t, grhor_t = B['grhog_t'], B['grhor_t']
    grhoc_t, grhob_t = B['grhoc_t'], B['grhob_t']
    k2 = k * k

    # State variables
    etak = y[IX_ETAK]
    clxb, vb = y[IX_CLXB], y[IX_VB]
    clxg, qg, pig = y[IX_G], y[IX_G + 1], y[IX_G + 2]
    clxr, qr, pir = y[IX_R], y[IX_R + 1], y[IX_R + 2]
    E2 = y[IX_POL] if LMAXPOL >= 2 else 0.0

    # Metric perturbations from Einstein constraint equations
    dgrho = grhob_t * clxb + grhoc_t * y[IX_CLXC] + grhog_t * clxg + grhor_t * clxr
    dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
    dgpi = grhog_t * pig + grhor_t * pir
    z = (0.5 * dgrho / k + etak) / adotoa
    sigma = z + 1.5 * dgq / k2
    phi = -((dgrho + 3.0 * dgq * adotoa / k) + dgpi) / (2.0 * k2)

    # Φ̇ for the ISW effect — compute pigdot, pirdot directly from the
    # Boltzmann hierarchy equations (avoids re-evaluating full RHS)
    polter = pig / 10.0 + 9.0 / 15.0 * E2
    Theta3 = y[IX_G + 3] if LMAXG >= 3 else 0.0
    N3 = y[IX_R + 3] if LMAXNR >= 3 else 0.0
    pigdot = (2*k/5*qg - 3*k/5*Theta3 - opacity*(pig - polter) + 8*k*sigma/15)
    pirdot = (2*k/5*qr - 3*k/5*N3 + 8*k*sigma/15)
    pidot_sum = grhog_t * pigdot + grhor_t * pirdot
    diff_rhopi = pidot_sum - 4.0 * adotoa * dgpi
    gpres_plus_grho = (4.0 / 3.0) * (grhog_t + grhor_t) + grhoc_t + grhob_t
    phidot = 0.5 * (adotoa * (-dgpi - 2.0 * k2 * phi) + dgq * k
                     - diff_rhopi + k * sigma * gpres_plus_grho) / k2

    # Visibility function and optical depth
    tau_min, tau_max = thermo['tau_arr'][0], thermo['tau_arr'][-1]
    if tau < tau_min or tau > tau_max:
        vis = 0.0
        exptau = np.exp(-thermo['tau_optical'][0]) if tau < tau_min else 1.0
    else:
        vis = float(thermo['visibility_interp'](tau))
        exptau = float(thermo['exptau_interp'](tau))

    # Source function building blocks
    ISW = 2.0 * phidot * exptau
    monopole = -etak / k + 2.0 * phi + clxg / 4.0
    chi = pgrid['tau0'] - tau
    source_E = 15.0 / 8.0 * vis * polter / (chi**2 * k2) if chi > 0 else 0.0

    return ISW, monopole, sigma + vb, vis, polter, source_E


def evolve_k(k, bg, thermo, pgrid, tau_out, **kwargs):
    """Evolve perturbations for wavenumber k, return source functions on tau_out grid.

    Start from adiabatic initial conditions deep in radiation domination
    (when kτ ≪ 1), evolve through recombination capturing the acoustic
    oscillations, and extract source functions at each output time.

    Returns (src_j0, src_j1, src_j2, src_E) where the temperature transfer is:
      Δ_ℓ(k) = ∫ [src_j0·j_ℓ(kχ) + src_j1·j_ℓ'(kχ) + src_j2·j_ℓ''(kχ)] dτ

    The three channels correspond to:
      j_ℓ:  ISW + visibility×(monopole + quadrupole)   [Sachs-Wolfe]
      j_ℓ': visibility×(σ+v_b)                         [Doppler]
      j_ℓ'': visibility×Π                               [quadrupole]
    """
    # Starting time: kτ_start = 0.1 (safely in the super-horizon regime)
    tau_start = min(0.1 / k, tau_out[0] * 0.5)
    tau_start = max(tau_start, 0.1)  # don't start before τ = 0.1 Mpc

    # Initial conditions
    y0 = adiabatic_ics(k, tau_start, bg, pgrid)

    # Evolve with Radau (implicit, handles stiffness through recombination)
    ode_rtol = kwargs.get('ode_rtol', 1e-5)
    sol = integrate.solve_ivp(
        lambda tau, y: boltzmann_derivs(tau, y, k, bg, pgrid, thermo),
        [tau_start, tau_out[-1]],
        y0,
        t_eval=tau_out,
        method='Radau',
        rtol=ode_rtol, atol=ode_rtol * 1e-3,
        max_step=20.0,
    )

    ntau = len(tau_out)
    if not sol.success:
        print(f"  Warning: ODE solver failed for k={k:.4e}: {sol.message}")
        z = np.zeros(ntau)
        return z, z.copy(), z.copy(), z.copy()

    # --- Extract source function building blocks at each time step ---
    ISW_arr = np.zeros(ntau)
    monopole_arr = np.zeros(ntau)
    sigma_plus_vb_arr = np.zeros(ntau)
    vis_arr = np.zeros(ntau)
    polter_arr = np.zeros(ntau)
    src_E = np.zeros(ntau)
    for i, tau in enumerate(tau_out):
        (ISW_arr[i], monopole_arr[i], sigma_plus_vb_arr[i],
         vis_arr[i], polter_arr[i], src_E[i]) = \
            compute_source_functions(tau, sol.y[:, i], k, bg, pgrid, thermo)

    # --- Assemble temperature source (multi-channel IBP decomposition) ---
    # After integration by parts on visibility derivatives g' and g'', the
    # temperature transfer integral decomposes into three Bessel channels:
    #   j_ℓ:   ISW + vis×(monopole + (5/8)Π)   [Sachs-Wolfe + quadrupole]
    #   j_ℓ':  vis×(σ+vb)                       [Doppler]
    #   j_ℓ'': (15/8)×vis×Π                     [quadrupole]
    src_j0 = ISW_arr + vis_arr * (monopole_arr + 0.625 * polter_arr)
    src_j1 = vis_arr * sigma_plus_vb_arr
    src_j2 = 1.875 * vis_arr * polter_arr
    return src_j0, src_j1, src_j2, src_E


# ============================================================
# LINE-OF-SIGHT INTEGRATION AND POWER SPECTRA
# Convolve source functions with spherical Bessel functions
# to get transfer functions, then integrate over k for Cℓ.
# ============================================================

def compute_cls(bg, thermo, p, source_cache=None, fast=False):
    """Main pipeline: evolve all k modes, do LOS integration, assemble Cℓ.

    This is the computational core of nanoCMB. For each wavenumber k, we
    evolve the Boltzmann hierarchy and extract source functions. Then for
    each multipole ℓ, we convolve with j_ℓ(k(τ₀−τ)) and integrate over k.

    If source_cache is set, source functions are saved after ODE evolution
    and reloaded on subsequent runs (skipping the expensive ODE step).
    Set source_cache=None to disable caching.

    fast=True uses ~3× fewer k-modes and coarser ℓ sampling (~1 min).
    """
    if source_cache is None:
        source_cache = '_source_cache_fast.npz' if fast else '_source_cache.npz'
    print("Setting up perturbation grid...")
    pgrid = setup_perturbation_grid(bg, thermo)
    tau0 = bg['tau0']

    # --- Output time grid for source functions ---
    tau_star = thermo['tau_star']
    tau_early = np.linspace(1.0, tau_star - 100, 50)
    if fast:
        tau_rec = np.linspace(tau_star - 100, tau_star + 200, 400)
        tau_late = np.linspace(tau_star + 200, tau0 - 10, 50)
    else:
        tau_rec = np.linspace(tau_star - 100, tau_star + 200, 400)
        tau_late = np.linspace(tau_star + 200, tau0 - 10, 100)
    tau_out = np.unique(np.concatenate([tau_early, tau_rec, tau_late]))
    tau_out = tau_out[(tau_out > 0.5) & (tau_out < tau0 - 1)]
    ntau = len(tau_out)
    print(f"  {ntau} output time steps{' (fast)' if fast else ''}")

    # --- k-sampling ---
    k_min = 0.5e-4
    k_max = 0.45

    if fast:
        k_low = np.logspace(np.log10(k_min), np.log10(0.008), 10)
        k_mid = np.linspace(0.008, 0.25, 80)
        k_high = np.linspace(0.25, k_max, 20)
    else:
        # The ODE grid only needs to resolve the acoustic pattern in source functions
        # (period ≈ π/r_s ≈ 0.022 Mpc⁻¹). The finer k-grid for Bessel oscillation
        # resolution is handled by interpolation below. ~200 modes gives >10 pts/oscillation.
        k_low = np.logspace(np.log10(k_min), np.log10(0.008), 20)
        k_mid = np.linspace(0.008, 0.25, 150)
        k_high = np.linspace(0.25, k_max, 40)
    k_arr = np.unique(np.concatenate([k_low, k_mid, k_high]))
    nk = len(k_arr)
    print(f"  {nk} k-modes from {k_arr[0]:.1e} to {k_arr[-1]:.1e} Mpc⁻¹")

    # --- Evolve all k modes and store source functions ---
    _cache_hit = False
    if source_cache is not None:
        try:
            _c = np.load(source_cache)
            if (_c['k_arr'].shape == k_arr.shape and
                    np.allclose(_c['k_arr'], k_arr) and
                    _c['tau_out'].shape == tau_out.shape and
                    np.allclose(_c['tau_out'], tau_out)):
                sources_j0 = _c['sources_j0']
                sources_j1 = _c['sources_j1']
                sources_j2 = _c['sources_j2']
                sources_E = _c['sources_E']
                _cache_hit = True
                print(f"Loaded cached source functions from {source_cache}")
        except (FileNotFoundError, KeyError):
            pass

    if not _cache_hit:
        print("Evolving perturbations...")
        sources_j0 = np.zeros((nk, ntau))
        sources_j1 = np.zeros((nk, ntau))
        sources_j2 = np.zeros((nk, ntau))
        sources_E = np.zeros((nk, ntau))
        for ik, kval in enumerate(k_arr):
            if (ik + 1) % 20 == 0 or ik == 0:
                print(f"  k={kval:.4f} Mpc⁻¹ ({ik+1}/{nk})")
            ode_kw = {'ode_rtol': 1e-5} if fast else {}
            sources_j0[ik], sources_j1[ik], sources_j2[ik], sources_E[ik] = \
                evolve_k(kval, bg, thermo, pgrid, tau_out, **ode_kw)
        if source_cache is not None:
            np.savez(source_cache, k_arr=k_arr, tau_out=tau_out,
                     sources_j0=sources_j0, sources_j1=sources_j1,
                     sources_j2=sources_j2, sources_E=sources_E)
            print(f"Saved source cache to {source_cache}")

    # --- Interpolate source functions to finer k-grid ---
    # Source functions are smooth in k, but the transfer function Δ_ℓ(k)
    # oscillates rapidly due to Bessel function ringing. A fine k-grid is
    # needed for accurate ∫|Δ|² d(ln k) integration (CAMB uses ~3000 k-pts).
    # Interpolate sources from the ODE grid to a ~5× denser grid.
    nk_fine = 3000 if not fast else 600
    # Start dense linear spacing at k=0.002 (covers ℓ>30 peak contributions)
    k_lin_start = 0.002
    n_log = 40 if not fast else 15
    k_fine = np.unique(np.concatenate([
        np.logspace(np.log10(k_arr[0]), np.log10(k_lin_start), n_log),
        np.linspace(k_lin_start, k_arr[-1], nk_fine - n_log if not fast else nk_fine - n_log),
    ]))
    nk_fine = len(k_fine)
    lnk_ode = np.log(k_arr)
    lnk_fine = np.log(k_fine)
    src_fine_j0 = np.zeros((nk_fine, ntau))
    src_fine_j1 = np.zeros((nk_fine, ntau))
    src_fine_j2 = np.zeros((nk_fine, ntau))
    src_fine_E = np.zeros((nk_fine, ntau))
    for it in range(ntau):
        src_fine_j0[:, it] = np.interp(lnk_fine, lnk_ode, sources_j0[:, it])
        src_fine_j1[:, it] = np.interp(lnk_fine, lnk_ode, sources_j1[:, it])
        src_fine_j2[:, it] = np.interp(lnk_fine, lnk_ode, sources_j2[:, it])
        src_fine_E[:, it] = np.interp(lnk_fine, lnk_ode, sources_E[:, it])
    print(f"Interpolated sources: {nk} → {nk_fine} k-modes")

    # --- Line-of-sight integration ---
    print("Computing transfer functions (line-of-sight integration)...")
    ell_max = 2500
    if fast:
        ells_compute = np.unique(np.concatenate([
            np.arange(2, 50, 3),
            np.arange(50, 200, 5),
            np.arange(200, 1200, 10),
            np.arange(1200, 2000, 20),
            np.arange(2000, ell_max + 1, 30),
        ]))
    else:
        ells_compute = np.unique(np.concatenate([
            np.arange(2, 50),
            np.arange(50, 200, 2),
            np.arange(200, 1200, 4),
            np.arange(1200, 2000, 8),
            np.arange(2000, ell_max + 1, 15),
        ]))
    ells_compute = ells_compute[ells_compute <= ell_max]
    nell = len(ells_compute)
    print(f"  {nell} ℓ-values from {ells_compute[0]} to {ells_compute[-1]}")

    chi_arr = tau0 - tau_out   # comoving distance array
    chi_star = tau0 - tau_star
    chi_max = chi_arr.max()

    # Transfer functions: Δ_ℓ^T(k) and Δ_ℓ^E(k) on fine k-grid
    Delta_T = np.zeros((nell, nk_fine))
    Delta_E = np.zeros((nell, nk_fine))

    # x_2d[ik, itau] = k * chi — precompute once (full grid for reference)
    x_2d_full = k_fine[:, None] * chi_arr[None, :]   # shape (nk_fine, ntau)

    for il, ell in enumerate(ells_compute):
        if (il + 1) % 20 == 0 or il == 0:
            print(f"  ℓ={ell} ({il+1}/{nell})")

        # Restrict k-range to where j_ℓ(kχ) is nonzero. The spherical Bessel
        # function j_ℓ(x) is exponentially small for x < ℓ − O(ℓ^{1/3}) and
        # oscillates with decaying amplitude for x > ℓ. Combined with the
        # upper k-cutoff, this typically reduces the array by ~80%.
        x_lo = max(0.0, ell - 4.0 * ell**(1.0/3.0))
        k_lo = x_lo / chi_max if chi_max > 0 else 0
        k_hi = (ell + 1000) / chi_star if chi_star > 0 else k_fine[-1]
        ik_lo = max(0, np.searchsorted(k_fine, k_lo) - 1)
        ik_hi = min(nk_fine, np.searchsorted(k_fine, k_hi) + 1)

        x_2d = x_2d_full[ik_lo:ik_hi, :]
        s_j0 = src_fine_j0[ik_lo:ik_hi, :]
        s_j1 = src_fine_j1[ik_lo:ik_hi, :]
        s_j2 = src_fine_j2[ik_lo:ik_hi, :]
        s_E = src_fine_E[ik_lo:ik_hi, :]

        # Compute spherical Bessel functions using cylindrical Bessel J_{ℓ+½}
        # j_ℓ(x) = √(π/(2x)) × J_{ℓ+½}(x)  — much faster than spherical_jn
        # which must compute all orders 0..ℓ per point (Python-level loop).
        nu = ell + 0.5
        ell_factor = ell * (ell + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            prefac = np.where(x_2d > 1e-30, np.sqrt(np.pi / (2.0 * x_2d)), 0.0)
        Jnu = special.jv(nu, x_2d)
        jl = prefac * Jnu

        # j_ℓ'(x) = (ℓ/x) j_ℓ(x) − j_{ℓ+1}(x)   (recurrence relation)
        Jnu1 = special.jv(nu + 1, x_2d)
        jl_next = prefac * Jnu1
        with np.errstate(divide='ignore', invalid='ignore'):
            jl_d = np.where(x_2d > 1e-30, ell / x_2d * jl - jl_next, 0.0)

        # j_ℓ''(x) from the spherical Bessel ODE: x²j'' + 2xj' + (x²−ℓ(ℓ+1))j = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            jl_dd = np.where(
                x_2d > 1e-30,
                -2.0 * jl_d / x_2d + (ell_factor / x_2d**2 - 1.0) * jl,
                0.0,
            )

        # Temperature: three-channel integral  ∫ dτ [S₀·jₗ + S₁·jₗ' + S₂·jₗ'']
        integrand_T = s_j0 * jl + s_j1 * jl_d + s_j2 * jl_dd
        # E-mode: single jₗ channel
        integrand_E = s_E * jl
        Delta_T[il, ik_lo:ik_hi] = np.trapezoid(integrand_T, tau_out, axis=1)
        Delta_E[il, ik_lo:ik_hi] = np.trapezoid(integrand_E, tau_out, axis=1)

    # --- Power spectrum assembly ---
    # C_ℓ^XY = 4π ∫ d(ln k) P(k) Δ_ℓ^X(k) Δ_ℓ^Y(k)
    print("Assembling power spectra...")
    k_pivot = p['k_pivot']
    A_s = p['A_s']
    n_s = p['n_s']
    # Primordial power spectrum: P(k) = A_s × (k/k_pivot)^(n_s - 1)
    Pk = A_s * (k_fine / k_pivot)**(n_s - 1.0)

    lnk = lnk_fine

    Cl_TT = np.zeros(nell)
    Cl_EE = np.zeros(nell)
    Cl_TE = np.zeros(nell)
    for il, ell in enumerate(ells_compute):
        # E-mode normalisation: ctnorm = (ℓ²−1)(ℓ+2)ℓ
        ctnorm = (ell**2 - 1.0) * (ell + 2) * ell

        # Limit k-integration: k_max = (ℓ + 1000) / chi_star
        # This allows ~160 Bessel oscillations past the peak before cutoff,
        # enough for accurate cancellation but suppresses noise floor.
        k_cut = (ell + 1000) / chi_star
        k_mask = k_fine <= k_cut
        if not k_mask.all():
            lnk_use = lnk[k_mask]
            DT = Delta_T[il, k_mask]
            DE = Delta_E[il, k_mask]
            Pk_use = Pk[k_mask]
        else:
            lnk_use = lnk
            DT = Delta_T[il, :]
            DE = Delta_E[il, :]
            Pk_use = Pk

        # Trapezoidal integration in ln k
        integrand_TT = Pk_use * DT**2
        integrand_EE = Pk_use * DE**2
        integrand_TE = Pk_use * DT * DE
        Cl_TT[il] = np.trapezoid(integrand_TT, lnk_use)
        Cl_EE[il] = np.trapezoid(integrand_EE, lnk_use)
        Cl_TE[il] = np.trapezoid(integrand_TE, lnk_use)

        # Normalise: multiply by 4π × ℓ(ℓ+1)/(2π)
        norm = 4.0 * np.pi  # the 4π from the k-integral
        fac = ell * (ell + 1) / (2.0 * np.pi)
        Cl_TT[il] *= norm * fac
        Cl_EE[il] *= norm * fac * ctnorm
        Cl_TE[il] *= norm * fac * np.sqrt(ctnorm)

    # Convert from dimensionless (ΔT/T)² to μK²
    T0_muK = p['T_cmb'] * 1e6  # CMB temperature in μK
    T0_muK2 = T0_muK**2
    Cl_TT *= T0_muK2
    Cl_EE *= T0_muK2
    Cl_TE *= T0_muK2

    # Interpolate to all integer ℓ using cubic spline (captures peak structure)
    ells_all = np.arange(2, ell_max + 1)
    Dl_TT = interpolate.CubicSpline(ells_compute, Cl_TT)(ells_all)
    Dl_EE = interpolate.CubicSpline(ells_compute, Cl_EE)(ells_all)
    Dl_TE = interpolate.CubicSpline(ells_compute, Cl_TE)(ells_all)

    print("Done!")
    return {
        'ells': ells_all,
        'Dl_TT': Dl_TT,    # D_ℓ^TT = ℓ(ℓ+1)Cℓ^TT/(2π) in μK²
        'Dl_EE': Dl_EE,
        'Dl_TE': Dl_TE,
        'k_arr': k_arr,       # ODE k-grid (coarse, for source functions)
        'k_fine': k_fine,     # Fine k-grid (for transfer functions)
        'ells_compute': ells_compute,
        'Delta_T': Delta_T,   # Transfer functions on k_fine grid
        'Delta_E': Delta_E,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    import sys
    fast = '--fast' in sys.argv

    bg = compute_background(params)
    print("=== nanoCMB Background Cosmology ===")
    print(f"H₀ = {bg['H0'] * c_km_s:.2f} km/s/Mpc")
    print(f"η₀ = {bg['tau0']:.2f} Mpc")

    thermo = compute_thermodynamics(bg, params)
    print(f"\n=== Recombination ===")
    print(f"z* (visibility peak) = {thermo['z_star']:.1f}")
    print(f"η* = {thermo['tau_star']:.2f} Mpc")
    print(f"z_reion = {thermo['z_reion']:.2f}")

    # Compute CMB angular power spectra
    print("\n=== Computing Power Spectra ===")
    result = compute_cls(bg, thermo, params, fast=fast)

    # Print peak values as sanity check
    ells = result['ells']
    DlTT = result['Dl_TT']
    DlEE = result['Dl_EE']
    DlTE = result['Dl_TE']

    peak_idx = np.argmax(DlTT)
    print(f"\nTT first peak: ℓ ≈ {ells[peak_idx]}, D_ℓ ≈ {DlTT[peak_idx]:.1f}")
    print(f"EE max: D_ℓ ≈ {np.max(DlEE):.3f} at ℓ ≈ {ells[np.argmax(DlEE)]}")
    print(f"TE range: [{np.min(DlTE):.3f}, {np.max(DlTE):.3f}]")

    # Save output
    np.savez('nanocmb_output.npz', ells=ells, DlTT=DlTT, DlEE=DlEE, DlTE=DlTE)


if __name__ == '__main__':
    main()
