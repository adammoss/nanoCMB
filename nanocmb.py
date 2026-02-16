"""
nanoCMB — A minimal CMB angular power spectrum calculator

Computes TT, EE, and TE angular power spectra for flat ΛCDM cosmologies
in ~1k lines of readable Python. 1% accuracy in <1 minute runtime.

Features:
  - Full RECFAST recombination (H + He ODEs, matter temperature, Hswitch)

Approximations:
  - Flat geometry (K = 0)
  - Massless neutrinos only (no massive species)
  - Cosmological constant (w = -1 exactly)
  - No lensing, no tensors, no isocurvature modes
  - First-order tight-coupling approximation

Dependencies: numpy, scipy (optional numba for speedup)

Units: distances in Mpc, time in Mpc (c = 1), H in Mpc⁻¹, k in Mpc⁻¹,
       densities as 8πGρ in Mpc⁻² (CAMB convention), temperatures in K
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import integrate, interpolate, special

try:
    from numba import njit
    _jit = njit(cache=False)
except ImportError:
    _jit = lambda f: f

# ============================================================
# PHYSICAL CONSTANTS
# Natural units with c = 1, distances in Mpc
# ============================================================

c_km_s = 2.99792458e5               # speed of light (km/s)
k_B = 1.380649e-23                   # Boltzmann constant (J/K)
h_P = 6.62607015e-34                 # Planck constant (J·s)
m_e = 9.1093837015e-31              # electron mass (kg)
m_H = 1.673575e-27                   # hydrogen atom mass (kg)
m_He4 = 6.646479073e-27              # ⁴He atom mass (kg)
not4 = m_He4 / m_H                   # He/H mass ratio (≈3.9715, not exactly 4)
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
    'ell_max': 2500,                 # maximum multipole
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

    # Helium fraction by number: f_He = Y/(not4*(1-Y)) = n_He/n_H
    f_He = p['Y_He'] / (not4 * (1 - p['Y_He']))

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
    a = np.atleast_1d(a)
    result = np.array([integrate.quad(dtauda, 0, ai, args=(bg,), limit=100, epsrel=1e-8)[0]
                       for ai in a])
    return result.squeeze()


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

c_SI = c_km_s * 1e3                              # speed of light (m/s)

# Atomic transition levels (wavenumber, m⁻¹)
L_H_ion = 1.096787737e7                          # H ionization
L_H_alpha = 8.225916453e6                        # H Lyman-alpha
L_He1_ion = 1.98310772e7                         # HeI ionization
L_He2_ion = 4.389088863e7                        # HeII ionization
L_He_2s = 1.66277434e7                           # HeI 2¹S₀
L_He_2p = 1.71134891e7                           # HeI 2¹P₁

# Decay / transition rates
Lambda_2s1s = 8.2245809                           # H 2s→1s two-photon (s⁻¹)
Lambda_He = 51.3                                  # HeI 2s→1s two-photon (s⁻¹)
A2P_s = 1.798287e9                                # HeI 2¹P₁→1¹S₀ Einstein A (s⁻¹)
A2P_t = 177.58                                    # HeI 2³P₁→1¹S₀ Einstein A (s⁻¹)
L_He_2Pt = 1.690871466e7                          # HeI 2³P₁ (m⁻¹)
L_He_2St = 1.5985597526e7                         # HeI 2³S₁ (m⁻¹)
L_He2St_ion = 3.8454693845e6                      # HeI 2³S₁ ionisation continuum (m⁻¹)
sigma_He_2Ps = 1.436289e-22                       # HeI singlet photoionisation σ (m²)
sigma_He_2Pt = 1.484872e-22                       # HeI triplet photoionisation σ (m²)

# Fudge factor (CAMB default with Hswitch Gaussians: 1.125)
RECFAST_fudge = 1.125

# Hswitch double-Gaussian K correction parameters
AGauss1, AGauss2 = -0.14, 0.079
zGauss1, zGauss2 = 7.28, 6.73
wGauss1, wGauss2 = 0.18, 0.33

# Derived constants
CR = 2 * np.pi * m_e * k_B / h_P**2              # Saha coefficient (m⁻² K⁻¹)
CB1 = h_P * c_SI * L_H_ion / k_B                 # H ionization energy / k_B (K)
CB1_He1 = h_P * c_SI * L_He1_ion / k_B           # HeI ionization / k_B (K)
CB1_He2 = h_P * c_SI * L_He2_ion / k_B           # HeII ionization / k_B (K)
CDB = h_P * c_SI * (L_H_ion - L_H_alpha) / k_B   # H n=2 binding energy / k_B (K)
CDB_He = h_P * c_SI * (L_He1_ion - L_He_2s) / k_B  # HeI (ion−2s) / k_B (K)
CK = (1.0 / L_H_alpha)**3 / (8 * np.pi)          # λ_α³/(8π) (m³)
CK_He = (1.0 / L_He_2p)**3 / (8 * np.pi)         # HeI equivalent (m³)
CL = h_P * c_SI * L_H_alpha / k_B                 # Lyman-alpha energy / k_B (K)
CL_He = h_P * c_SI * L_He_2s / k_B                # HeI 2s energy / k_B (K)
Bfact = h_P * c_SI * (L_He_2p - L_He_2s) / k_B   # He 2P−2S splitting / k_B (K)
CL_PSt = h_P * c_SI * (L_He_2Pt - L_He_2St) / k_B  # He triplet 2³P−2³S splitting / k_B (K)
CB1_He2St = h_P * c_SI * L_He2St_ion / k_B        # He 2³S ionisation energy / k_B (K)
CL_He_2St = h_P * c_SI * L_He_2St / k_B           # He 2³S energy / k_B (K)
a_rad = 4 * sigma_SB / c_SI                       # radiation constant (J/m³/K⁴)
CT = (8.0 / 3.0) * (sigma_T / (m_e * c_SI)) * a_rad  # Compton cooling (s⁻¹ K⁻⁴)


def compute_recombination(bg, p):
    """Solve ionisation history x_e(z) using full RECFAST.

    Three-variable ODE for hydrogen ionisation (x_H), helium ionisation (x_He),
    and matter temperature (T_mat). Uses Saha equilibrium at high z, switching
    to ODEs as each species departs from equilibrium.

    Regime switching (high z → low z):
      z > 8000:  fully ionised (x_H=1, x_He=1, He doubly ionised)
      5000–8000: He++ → He+ Saha
      3500–5000: He singly ionised, H fully ionised
      < 3500:    He+ → He0 Saha until x_He < 0.99, then He ODE
                 H Saha until x_H < 0.99, then H Peebles ODE
    """
    T_cmb = bg['T_cmb']
    f_He = bg['f_He']

    # Present-day hydrogen number density (m⁻³)
    H100_SI = 100 * 1e3 / Mpc_in_m
    rho_crit_100 = 3 * H100_SI**2 / (8 * np.pi * G)
    Nnow = (1 - bg['Y_He']) * (p['omega_b_h2'] * rho_crit_100) / m_H

    # Cosmological parameters for dH/dz in T_mat equation
    H0_SI = bg['H0'] * c_SI / Mpc_in_m
    omega_m = (p['omega_b_h2'] + p['omega_c_h2']) / p['h']**2
    a_eq = (bg['grhog'] + bg['grhornomass']) / (bg['grhoc'] + bg['grhob'])
    z_eq = 1.0 / a_eq - 1.0

    def Hz_SI(z):
        return hubble(1.0 / (1 + z), bg) * c_SI / Mpc_in_m

    # --- Saha equations ---
    def saha_He2(z):
        """He++ → He+ Saha: returns total x_e per H atom."""
        T = T_cmb * (1 + z)
        rhs = (CR * T_cmb / (1 + z))**1.5 * np.exp(-CB1_He2 / T) / Nnow
        return 0.5 * (np.sqrt((rhs - 1 - f_He)**2
                              + 4 * (1 + 2 * f_He) * rhs) - (rhs - 1 - f_He))

    def saha_He1(z):
        """He+ → He0 Saha: returns x_He = n(He+)/n_He."""
        T = T_cmb * (1 + z)
        rhs = 4.0 * (CR * T_cmb / (1 + z))**1.5 * np.exp(-CB1_He1 / T) / Nnow
        x0 = 0.5 * (np.sqrt((rhs - 1)**2 + 4 * (1 + f_He) * rhs) - (rhs - 1))
        return min((x0 - 1.0) / f_He, 1.0)

    def saha_H(z):
        """H Saha: returns x_H (assumes He contribution to n_e negligible)."""
        T = T_cmb * (1 + z)
        rhs = (CR * T_cmb / (1 + z))**1.5 * np.exp(-CB1 / T) / Nnow
        return min(0.5 * (np.sqrt(rhs**2 + 4 * rhs) - rhs), 1.0)

    # --- RECFAST ODE right-hand side ---
    def recfast_rhs(z, y, saha_H_mode=False):
        """dy/dz for y = [x_H, x_He, T_mat]."""
        x_H = max(y[0], 0.0)
        x_He = max(y[1], 0.0)
        T_mat = max(y[2], 0.5)

        if saha_H_mode:
            x_H = saha_H(z)

        x = x_H + f_He * x_He
        T_rad = T_cmb * (1 + z)
        n_H = Nnow * (1 + z)**3
        n_He = f_He * n_H
        Hz = Hz_SI(z)

        # --- f1: Hydrogen Peebles equation ---
        if saha_H_mode or x_H > 0.99:
            f1 = 0.0
        else:
            t4 = T_mat / 1e4
            Rdown = 1e-19 * 4.309 * t4**(-0.6166) / (1 + 0.6703 * t4**0.5300)
            Rup = Rdown * (CR * T_mat)**1.5 * np.exp(-CDB / T_mat)

            K = CK / Hz * (1.0
                + AGauss1 * np.exp(-((np.log(1 + z) - zGauss1) / wGauss1)**2)
                + AGauss2 * np.exp(-((np.log(1 + z) - zGauss2) / wGauss2)**2))
            fu = RECFAST_fudge
            n_1s = n_H * max(1 - x_H, 1e-30)

            if x_H > 0.985:
                # Near-Saha rate (no Peebles bottleneck)
                f1 = (x * x_H * n_H * Rdown
                      - Rup * (1 - x_H) * np.exp(-CL / T_mat)) / (Hz * (1 + z))
            else:
                f1 = ((x * x_H * n_H * Rdown
                       - Rup * (1 - x_H) * np.exp(-CL / T_mat))
                      * (1 + K * Lambda_2s1s * n_1s)
                      / (Hz * (1 + z) * (1.0 / fu + K * Lambda_2s1s * n_1s / fu
                                         + K * Rup * n_1s)))

        # --- f2: Helium singlet ODE ---
        if x_He < 1e-15:
            f2 = 0.0
        else:
            T_0 = 10.0**0.477121   # ~3 K
            T_1 = 10.0**5.114      # ~1.3e5 K
            sq_0 = np.sqrt(T_mat / T_0)
            sq_1 = np.sqrt(T_mat / T_1)
            Rdown_He = 10.0**(-16.744) / (sq_0 * (1 + sq_0)**0.289
                                          * (1 + sq_1)**1.711)
            Rup_He = 4.0 * Rdown_He * (CR * T_mat)**1.5 * np.exp(-CDB_He / T_mat)
            He_Boltz = np.exp(min(Bfact / T_mat, 500.0))

            n_He_ground = n_He * max(1 - x_He, 1e-30)
            tauHe_s = A2P_s * CK_He * 3 * n_He_ground / Hz
            pHe_s = ((1 - np.exp(-tauHe_s)) / tauHe_s
                     if tauHe_s > 1e-7 else 1.0 - tauHe_s / 2.0)

            # Singlet K_He with H continuum opacity (Heflag >= 2)
            if x_H < 0.9999999:
                Doppler_s = c_SI * L_He_2p * np.sqrt(2 * k_B * T_mat / (m_H * not4 * c_SI**2))
                gamma_2Ps = (3 * A2P_s * f_He * (1 - x_He) * c_SI**2
                             / (np.sqrt(np.pi) * sigma_He_2Ps * 8 * np.pi
                                * Doppler_s * max(1 - x_H, 1e-30) * (c_SI * L_He_2p)**2))
                AHcon_s = A2P_s / (1 + 0.36 * gamma_2Ps**0.86)
                K_He = 1.0 / max((A2P_s * pHe_s + AHcon_s) * 3 * n_He_ground, 1e-300)
            else:
                K_He = 1.0 / max(A2P_s * pHe_s * 3 * n_He_ground, 1e-300)

            f2 = ((x * x_He * n_H * Rdown_He
                   - Rup_He * (1 - x_He) * np.exp(-CL_He / T_mat))
                  * (1 + K_He * Lambda_He * n_He_ground * He_Boltz)
                  / (Hz * (1 + z)
                     * (1 + K_He * (Lambda_He + Rup_He)
                        * n_He_ground * He_Boltz)))

            # Triplet channel (Heflag >= 3, gate on x_He as in RECFAST)
            if x_He > 5e-9:
                a_trip = 10.0**(-16.306)
                b_trip = 0.761
                Rdown_trip = a_trip / (sq_0 * (1 + sq_0)**(1.0 - b_trip)
                                       * (1 + sq_1)**(1.0 + b_trip))
                Rup_trip = (4.0 / 3.0) * Rdown_trip * (CR * T_mat)**1.5 * np.exp(-CB1_He2St / T_mat)

                tauHe_t = A2P_t * n_He_ground * 3 / (8 * np.pi * Hz * L_He_2Pt**3)
                pHe_t = ((1 - np.exp(-tauHe_t)) / tauHe_t
                         if tauHe_t > 1e-7 else 1.0 - tauHe_t / 2.0)

                # Triplet C factor with H continuum opacity (Heflag 6)
                if x_H < 0.99999:
                    Doppler_t = c_SI * L_He_2Pt * np.sqrt(2 * k_B * T_mat / (m_H * not4 * c_SI**2))
                    gamma_2Pt = (3 * A2P_t * f_He * (1 - x_He) * c_SI**2
                                 / (np.sqrt(np.pi) * sigma_He_2Pt * 8 * np.pi
                                    * Doppler_t * max(1 - x_H, 1e-30) * (c_SI * L_He_2Pt)**2))
                    AHcon_t = A2P_t / (1 + 0.66 * gamma_2Pt**0.9) / 3.0
                    CfHe_t = (A2P_t * pHe_t + AHcon_t) * np.exp(-CL_PSt / T_mat)
                else:
                    CfHe_t = A2P_t * pHe_t * np.exp(-CL_PSt / T_mat)
                denom = Rup_trip + CfHe_t
                CfHe_t = CfHe_t / denom if denom > 1e-300 else 0.0

                f2 += ((x * x_He * n_H * Rdown_trip
                        - (1 - x_He) * 3 * Rup_trip * np.exp(-CL_He_2St / T_mat))
                       * CfHe_t / (Hz * (1 + z)))

        # --- f3: Matter temperature ---
        x_safe = max(x, 1e-30)
        timeTh = (1.0 / (CT * T_rad**4)) * (1 + x + f_He) / x_safe
        timeH = 2.0 / (3.0 * H0_SI * (1 + z)**1.5)

        if timeTh < 1e-3 * timeH:
            # Tightly coupled: implicit form (T_mat ≈ T_rad + corrections)
            dHdz = (H0_SI**2 / (2 * Hz)) * omega_m * (
                4 * (1 + z)**3 / (1 + z_eq) + 3 * (1 + z)**2)
            epsilon = Hz * (1 + x + f_He) / (CT * T_rad**3 * x_safe)
            f3 = (T_cmb
                  + epsilon * (1 + f_He) / (1 + f_He + x)
                  * (f1 + f_He * f2) / x_safe
                  - epsilon * dHdz / Hz
                  + 3 * epsilon / (1 + z))
        else:
            # Loosely coupled: Compton cooling + adiabatic expansion
            f3 = (CT * T_rad**4 * x_safe / (1 + x + f_He)
                  * (T_mat - T_rad) / (Hz * (1 + z))
                  + 2 * T_mat / (1 + z))

        return [f1, f2, f3]

    # --- Build z grid ---
    z_start = 10000
    z_end = 0
    nz = 20000
    z_arr = np.linspace(z_start, z_end, nz + 1)
    xH_arr = np.ones(nz + 1)
    xHe_arr = np.ones(nz + 1)

    # --- Phase 1: Saha equilibrium ---
    # Scan forward (decreasing z) to find where He and H depart from Saha
    he_ode_idx = None
    for i, z in enumerate(z_arr):
        if z > 8000:
            xH_arr[i] = 1.0
            xHe_arr[i] = 1.0
        elif z > 5000:
            x0 = saha_He2(z)
            xH_arr[i] = 1.0
            xHe_arr[i] = max((x0 - 1.0) / f_He, 1.0) if f_He > 0 else 1.0
        elif z > 3500:
            xH_arr[i] = 1.0
            xHe_arr[i] = 1.0
        elif z > 0:
            x_He = saha_He1(z)
            xHe_arr[i] = x_He
            xH_arr[i] = 1.0
            if x_He < 0.99:
                he_ode_idx = i
                break
        else:
            break

    if he_ode_idx is None:
        he_ode_idx = len(z_arr) - 1

    # Find where H Saha drops below 0.99
    h_ode_z = None
    for z in z_arr[he_ode_idx:]:
        if z <= 0:
            break
        if saha_H(z) < 0.99:
            h_ode_z = z
            break

    # --- Phase 2: He ODE + T_mat ODE, H from Saha ---
    z_phase2_start = z_arr[he_ode_idx]
    z_phase2_end = h_ode_z if h_ode_z is not None else 0.0
    z_phase2 = z_arr[(z_arr <= z_phase2_start) & (z_arr >= z_phase2_end)]

    y0_phase2 = [xH_arr[he_ode_idx], xHe_arr[he_ode_idx],
                 T_cmb * (1 + z_phase2_start)]

    if len(z_phase2) > 1:
        sol2 = integrate.solve_ivp(
            lambda z, y: recfast_rhs(z, y, saha_H_mode=True),
            [z_phase2[0], z_phase2[-1]], y0_phase2,
            t_eval=z_phase2, method='Radau', rtol=1e-6, atol=1e-10,
            max_step=5.0,
        )
        # Override x_H with Saha values
        n_sol2 = sol2.y.shape[1]
        idx_start = np.searchsorted(-z_arr, -z_phase2[0])
        for j in range(n_sol2):
            z_j = sol2.t[j]
            sol2.y[0, j] = saha_H(z_j)
            ii = idx_start + j
            if ii < len(z_arr):
                xH_arr[ii] = sol2.y[0, j]
                xHe_arr[ii] = sol2.y[1, j]

    # --- Phase 3: Full 3-variable ODE ---
    if h_ode_z is not None and h_ode_z > 0:
        h_ode_idx = np.searchsorted(-z_arr, -h_ode_z)
        z_phase3 = z_arr[h_ode_idx:]
        z_phase3 = z_phase3[z_phase3 >= 0]

        if len(z_phase3) > 1:
            # Initial conditions from end of Phase 2
            if len(z_phase2) > 1:
                y0_p3 = [sol2.y[0, -1], sol2.y[1, -1], sol2.y[2, -1]]
            else:
                y0_p3 = [saha_H(h_ode_z), saha_He1(h_ode_z),
                         T_cmb * (1 + h_ode_z)]

            sol3 = integrate.solve_ivp(
                lambda z, y: recfast_rhs(z, y, saha_H_mode=False),
                [z_phase3[0], z_phase3[-1]], y0_p3,
                t_eval=z_phase3, method='Radau', rtol=1e-6, atol=1e-10,
                max_step=2.0,
            )
            n_sol = sol3.y.shape[1]
            for j in range(n_sol):
                ii = h_ode_idx + j
                if ii < len(z_arr):
                    xH_arr[ii] = sol3.y[0, j]
                    xHe_arr[ii] = sol3.y[1, j]
    else:
        # No H ODE needed (shouldn't happen in practice)
        pass

    # Total electron fraction: x_e = x_H + f_He × x_He
    xe_total = xH_arr + f_He * xHe_arr

    # At z > 5000, He is (partially) doubly ionised — add extra electrons
    for i, z in enumerate(z_arr):
        if z > 8000:
            xe_total[i] = 1.0 + 2 * f_He
        elif z > 5000:
            xe_total[i] = saha_He2(z)

    return z_arr, xe_total


def compute_thermodynamics(bg, p):
    """Build thermodynamic tables: opacity, optical depth, visibility function.

    The visibility function g(η) = κ̇ e^{-τ} tells us the probability that a CMB
    photon last scattered at conformal time η. Its peak defines the surface of
    last scattering, and its width determines the thickness of that surface
    (which causes diffusion damping of small-scale anisotropies).
    """
    z_arr, xe_arr = compute_recombination(bg, p)

    # Add reionisation: tanh model matching the input τ_reion
    # x_e(z) = (f_re - x_freeze) × (1 + tanh((y_re - y)/Δy)) / 2 + x_freeze
    # where y = (1+z)^1.5, Δy = 1.5√(1+z_re) × Δz_re
    f_He = bg['f_He']
    f_re = 1.0 + f_He  # Full ionisation: H + singly-ionised He

    # Find z_re by bisection to match input τ_reion
    def apply_reionisation(z_re):
        """Apply tanh reionisation model and return modified x_e array."""
        delta_z = 0.5  # Width of reionisation transition
        y = (1 + z_arr)**1.5
        y_re = (1 + z_re)**1.5
        dy = 1.5 * np.sqrt(1 + z_re) * delta_z
        x_reion = f_re * (1 + np.tanh((y_re - y) / dy)) / 2
        return np.maximum(xe_arr, x_reion)

    def compute_reion_optical_depth(xe_test):
        """Compute optical depth from z=0 to z_max ~ 50.

        This captures the reionisation contribution plus the small residual from
        recombination freeze-out. We don't integrate through the recombination
        epoch itself (τ >> 1 there), since τ_reion only measures the low-z part.
        """
        z_mid = 0.5 * (z_arr[:-1] + z_arr[1:])
        mask = z_mid <= 50.0
        z_mid = z_mid[mask]
        a_mid = 1.0 / (1 + z_mid)
        xe_mid = 0.5 * (xe_test[:-1] + xe_test[1:])[mask]
        dz = np.abs(np.diff(z_arr))[mask]
        dtauda_arr = np.array([dtauda(a, bg) for a in a_mid])
        deta = dtauda_arr / (1 + z_mid)**2 * dz
        return np.sum(xe_mid * bg['akthom'] / a_mid**2 * deta)

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
    tau_optical = -np.flip(integrate.cumulative_trapezoid(
        np.flip(opacity), np.flip(tau_arr), initial=0))

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


# --- Numba-accelerated helpers (fall back to plain Python without numba) ---

@_jit
def _cubic_eval(x_knots, coeffs, t):
    """Evaluate a scipy CubicSpline at point t using binary search + Horner."""
    n = x_knots.shape[0] - 1
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if x_knots[mid + 1] < t:
            lo = mid + 1
        else:
            hi = mid
    dt = t - x_knots[lo]
    return ((coeffs[0, lo] * dt + coeffs[1, lo]) * dt + coeffs[2, lo]) * dt + coeffs[3, lo]


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


def adiabatic_ics(k, tau_start, bg, pgrid):
    """Set adiabatic initial conditions deep in the radiation era (kτ ≪ 1).

    These follow CAMB's initial() subroutine exactly.
    """
    tau = tau_start
    x = k * tau
    x2 = x * x

    grho_rad = pgrid['grho_rad']
    # Neutrino fraction of radiation
    Rv = bg['grhornomass'] / grho_rad     # ρ_ν/(ρ_ν + ρ_γ)
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


@_jit
def _boltzmann_rhs(tau, y, k, bg5, sp_a_x, sp_a_c, sp_op_x, sp_op_c):
    """Right-hand side of the Boltzmann hierarchy: dy/dτ.

    Implements the synchronous gauge equations from CAMB's derivs() subroutine.
    During tight coupling (early times, high opacity), only the photon monopole
    and dipole are evolved; the quadrupole is computed algebraically.
    """
    # Background quantities from spline interpolation
    grhog, grhornomass, grhoc, grhob, grhov = bg5[0], bg5[1], bg5[2], bg5[3], bg5[4]
    a = _cubic_eval(sp_a_x, sp_a_c, tau)
    a2 = a * a
    grhog_t = grhog / a2
    grhor_t = grhornomass / a2
    grhoc_t = grhoc / a
    grhob_t = grhob / a
    grho_a2 = grhog_t + grhor_t + grhoc_t + grhob_t + grhov * a2
    adotoa = np.sqrt(grho_a2 / 3.0)
    opacity = _cubic_eval(sp_op_x, sp_op_c, tau)
    if opacity < 1e-30:
        opacity = 1e-30
    photbar = grhog_t / grhob_t
    pb43 = 4.0 / 3.0 * photbar

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

    # Synchronous gauge: z = ḣ/(2k), σ = metric shear
    z = (0.5 * dgrho / k + etak) / adotoa
    sigma = z + 1.5 * dgq / k2

    # Polter: polarisation source Π = pig/10 + 3E₂/5
    E2 = y[IX_POL] if LMAXPOL >= 2 else 0.0
    polter = pig / 10.0 + 9.0 / 15.0 * E2

    cothxor = 1.0 / tau

    dy = np.zeros(NVAR)

    # --- Metric equation ---
    dy[IX_ETAK] = 0.5 * dgq

    # --- CDM: at rest in this gauge ---
    dy[IX_CLXC] = -k * z

    # --- Baryons ---
    dy[IX_CLXB] = -k * (z + vb)

    if tight_coupling:
        # Tight-coupling: photon-baryon fluid locked together
        pig_tc = 32.0 / 45.0 * k / opacity * (sigma + vb)
        polter = pig_tc / 4.0

        vbdot = (-adotoa * vb + k / 4.0 * pb43 * (clxg - 2.0 * pig_tc)) / (1.0 + pb43)
        dy[IX_VB] = vbdot

        dy[IX_G] = -k * (4.0 / 3.0 * z + qg)
        qgdot = 4.0 / 3.0 * (-vbdot - adotoa * vb) / pb43 + k / 3.0 * clxg - 2.0 * k / 3.0 * pig_tc
        dy[IX_G + 1] = qgdot
        dy[IX_G + 2] = opacity * (pig_tc - pig)

        if LMAXPOL >= 2:
            dy[IX_POL] = opacity * (pig_tc / 4.0 - E2)

    else:
        # Full Boltzmann hierarchy
        vbdot = -adotoa * vb - photbar * opacity * (4.0 / 3.0 * vb - qg)
        dy[IX_VB] = vbdot

        dy[IX_G] = -k * (4.0 / 3.0 * z + qg)
        qgdot = 4.0 / 3.0 * (-vbdot - adotoa * vb) / pb43 + k / 3.0 * clxg - 2.0 * k / 3.0 * pig
        dy[IX_G + 1] = qgdot

        Theta3 = y[IX_G + 3] if LMAXG >= 3 else 0.0
        dy[IX_G + 2] = (2.0 * k / 5.0 * qg - 3.0 * k / 5.0 * Theta3
                        - opacity * (pig - polter) + 8.0 / 15.0 * k * sigma)

        for l in range(3, LMAXG):
            dy[IX_G + l] = (k * l / (2*l + 1) * y[IX_G + l - 1]
                            - k * (l + 1) / (2*l + 1) * y[IX_G + l + 1]
                            - opacity * y[IX_G + l])

        # Truncation: free-streaming closure
        dy[IX_G + LMAXG] = (k * y[IX_G + LMAXG - 1]
                             - (LMAXG + 1) * cothxor * y[IX_G + LMAXG]
                             - opacity * y[IX_G + LMAXG])

        # --- Photon polarisation hierarchy ---
        E3 = y[IX_POL + 1] if LMAXPOL >= 3 else 0.0
        dy[IX_POL] = -opacity * (E2 - polter) - k / 3.0 * E3

        for l in range(3, LMAXPOL):
            idx = IX_POL + l - 2
            polfac_l = (l + 3) * (l - 1) / (l + 1)
            dy[idx] = (-opacity * y[idx]
                       + k * l / (2*l + 1) * y[idx - 1]
                       - polfac_l * k / (2*l + 1) * y[idx + 1])

        idx_last = IX_POL + LMAXPOL - 2
        dy[idx_last] = (-opacity * y[idx_last]
                        + k * LMAXPOL / (2*LMAXPOL + 1) * y[idx_last - 1]
                        - (LMAXPOL + 3) * cothxor * y[idx_last])

    # --- Massless neutrinos ---
    dy[IX_R] = -k * (4.0 / 3.0 * z + qr)
    dy[IX_R + 1] = k / 3.0 * (clxr - 2.0 * pir)

    N3 = y[IX_R + 3] if LMAXNR >= 3 else 0.0
    dy[IX_R + 2] = 2.0 * k / 5.0 * qr - 3.0 * k / 5.0 * N3 + 8.0 / 15.0 * k * sigma

    for l in range(3, LMAXNR):
        dy[IX_R + l] = (k * l / (2*l + 1) * y[IX_R + l - 1]
                        - k * (l + 1) / (2*l + 1) * y[IX_R + l + 1])

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
    a = float(pgrid['a_of_tau'](tau))
    a2 = a * a
    grhog_t = bg['grhog'] / a2
    grhor_t = bg['grhornomass'] / a2
    grhoc_t = bg['grhoc'] / a
    grhob_t = bg['grhob'] / a
    adotoa = np.sqrt((grhog_t + grhor_t + grhoc_t + grhob_t + bg['grhov'] * a2) / 3.0)
    opacity = max(float(pgrid['opacity_interp'](tau)), 1e-30)
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


def evolve_k(k, bg, thermo, pgrid, tau_out):
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
    bg5 = np.array([bg['grhog'], bg['grhornomass'], bg['grhoc'],
                    bg['grhob'], bg['grhov']])
    sp_a_x = pgrid['a_of_tau'].x
    sp_a_c = pgrid['a_of_tau'].c
    sp_op_x = pgrid['opacity_interp'].x
    sp_op_c = pgrid['opacity_interp'].c

    sol = integrate.solve_ivp(
        lambda tau, y: _boltzmann_rhs(tau, y, k, bg5, sp_a_x, sp_a_c, sp_op_x, sp_op_c),
        [tau_start, tau_out[-1]],
        y0,
        t_eval=tau_out,
        method='Radau',
        rtol=1e-5, atol=1e-8,
        max_step=20.0,
    )

    ntau = len(tau_out)
    if not sol.success:
        print(f"  Warning: ODE solver failed for k={k:.4e}: {sol.message}")
        return tuple(np.zeros(ntau) for _ in range(4))

    # --- Extract source function building blocks at each time step ---
    ISW_arr, monopole_arr, sigma_plus_vb_arr, vis_arr, polter_arr, src_E = \
        (np.zeros(ntau) for _ in range(6))
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

# Worker functions for multiprocessing (must be top-level for pickling)
_pool_bg = _pool_thermo = _pool_pgrid = _pool_tau_out = None

def _pool_init(bg, thermo, pgrid, tau_out):
    global _pool_bg, _pool_thermo, _pool_pgrid, _pool_tau_out
    _pool_bg, _pool_thermo, _pool_pgrid, _pool_tau_out = bg, thermo, pgrid, tau_out

def _pool_solve_k(k):
    return evolve_k(k, _pool_bg, _pool_thermo, _pool_pgrid, _pool_tau_out)


def compute_cls(bg, thermo, p):
    """Main pipeline: evolve all k modes, do LOS integration, assemble Cℓ.

    This is the computational core of nanoCMB. For each wavenumber k, we
    evolve the Boltzmann hierarchy and extract source functions. Then for
    each multipole ℓ, we convolve with j_ℓ(k(τ₀−τ)) and integrate over k.
    """
    print("Setting up perturbation grid...")
    pgrid = setup_perturbation_grid(bg, thermo)
    tau0 = bg['tau0']

    # --- Output time grid for source functions ---
    tau_star = thermo['tau_star']
    tau_early = np.linspace(1.0, tau_star - 100, 60)
    tau_rec = np.linspace(tau_star - 100, tau_star + 200, 500)
    tau_late = np.linspace(tau_star + 200, tau0 - 10, 120)
    tau_out = np.unique(np.concatenate([tau_early, tau_rec, tau_late]))
    tau_out = tau_out[(tau_out > 0.5) & (tau_out < tau0 - 1)]
    ntau = len(tau_out)
    print(f"  {ntau} output time steps")

    # --- k-sampling ---
    k_min = 0.5e-4
    k_max = 0.45

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
    print("Evolving perturbations...")
    _args = (bg, thermo, pgrid, tau_out)

    # Warmup JIT (no-op without numba) before forking workers
    _boltzmann_rhs(tau_out[0], np.zeros(NVAR), k_arr[0],
                   np.array([bg['grhog'], bg['grhornomass'], bg['grhoc'],
                             bg['grhob'], bg['grhov']]),
                   pgrid['a_of_tau'].x, pgrid['a_of_tau'].c,
                   pgrid['opacity_interp'].x, pgrid['opacity_interp'].c)

    try:
        from multiprocessing import Pool, cpu_count
        ncpu = cpu_count()
        print(f"  Using {ncpu} cores")
        with Pool(ncpu, initializer=_pool_init, initargs=_args) as pool:
            results = pool.map(_pool_solve_k, k_arr)
    except (ImportError, OSError):
        results = [evolve_k(k, bg, thermo, pgrid, tau_out) for k in k_arr]

    sources_j0 = np.array([r[0] for r in results])
    sources_j1 = np.array([r[1] for r in results])
    sources_j2 = np.array([r[2] for r in results])
    sources_E = np.array([r[3] for r in results])

    # --- Interpolate source functions to finer k-grid ---
    # Source functions are smooth in k, but the transfer function Δ_ℓ(k)
    # oscillates rapidly due to Bessel function ringing. A fine k-grid is
    # needed for accurate ∫|Δ|² d(ln k) integration (CAMB uses ~3000 k-pts).
    # Interpolate sources from the ODE grid to a ~5× denser grid.
    nk_fine = 3000
    # Start dense linear spacing at k=0.002 (covers ℓ>30 peak contributions)
    k_lin_start = 0.002
    n_log = 40
    k_fine = np.unique(np.concatenate([
        np.logspace(np.log10(k_arr[0]), np.log10(k_lin_start), n_log),
        np.linspace(k_lin_start, k_arr[-1], nk_fine - n_log),
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
    ell_max = p['ell_max']
    ells_compute = np.unique(np.concatenate([
        np.arange(2, 16, 1),
        np.arange(17, 39, 2),
        np.arange(40, 95, 5),
        np.array([110, 130, 150, 175, 200]),
        np.arange(200, ell_max + 1, 50),
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

    def _compute_ell_transfer(il, ell):
        """Compute transfer functions for a single ell (thread-safe: writes to distinct row)."""
        # Restrict k-range to where j_ℓ(kχ) is nonzero
        x_lo = max(0.0, ell - 4.0 * ell**(1.0/3.0))
        k_lo = x_lo / chi_max if chi_max > 0 else 0
        k_hi = (ell + 1700) / chi_star if chi_star > 0 else k_fine[-1]
        ik_lo = max(0, np.searchsorted(k_fine, k_lo) - 1)
        ik_hi = min(nk_fine, np.searchsorted(k_fine, k_hi) + 1)

        x_2d = x_2d_full[ik_lo:ik_hi, :]
        s_j0 = src_fine_j0[ik_lo:ik_hi, :]
        s_j1 = src_fine_j1[ik_lo:ik_hi, :]
        s_j2 = src_fine_j2[ik_lo:ik_hi, :]
        s_E = src_fine_E[ik_lo:ik_hi, :]

        nu = ell + 0.5
        ell_factor = ell * (ell + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            prefac = np.where(x_2d > 1e-30, np.sqrt(np.pi / (2.0 * x_2d)), 0.0)
        Jnu = special.jv(nu, x_2d)
        jl = prefac * Jnu

        Jnu1 = special.jv(nu + 1, x_2d)
        jl_next = prefac * Jnu1
        with np.errstate(divide='ignore', invalid='ignore'):
            jl_d = np.where(x_2d > 1e-30, ell / x_2d * jl - jl_next, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            jl_dd = np.where(
                x_2d > 1e-30,
                -2.0 * jl_d / x_2d + (ell_factor / x_2d**2 - 1.0) * jl,
                0.0,
            )

        integrand_T = s_j0 * jl + s_j1 * jl_d + s_j2 * jl_dd
        integrand_E = s_E * jl
        Delta_T[il, ik_lo:ik_hi] = np.trapezoid(integrand_T, tau_out, axis=1)
        Delta_E[il, ik_lo:ik_hi] = np.trapezoid(integrand_E, tau_out, axis=1)

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(_compute_ell_transfer, il, ell)
                   for il, ell in enumerate(ells_compute)]
        for i, f in enumerate(futures):
            f.result()
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  ℓ={ells_compute[i]} ({i+1}/{nell})")

    # --- Power spectrum assembly ---
    # C_ℓ^XY = 4π ∫ d(ln k) P(k) Δ_ℓ^X(k) Δ_ℓ^Y(k)
    print("Assembling power spectra...")
    k_pivot = p['k_pivot']
    A_s = p['A_s']
    n_s = p['n_s']
    # Primordial power spectrum: P(k) = A_s × (k/k_pivot)^(n_s - 1)
    Pk = A_s * (k_fine / k_pivot)**(n_s - 1.0)

    # k-cutoff already applied in LOS step (Delta values are zero beyond cutoff),
    # so we can integrate over the full lnk_fine grid directly.
    Cl_TT, Cl_EE, Cl_TE = [np.trapezoid(Pk * d, lnk_fine, axis=1)
                            for d in (Delta_T**2, Delta_E**2, Delta_T * Delta_E)]

    # Normalise: D_ℓ = ℓ(ℓ+1)C_ℓ/(2π), with 4π from the k-integral
    ells_f = ells_compute.astype(float)
    norm = 4.0 * np.pi * ells_f * (ells_f + 1) / (2.0 * np.pi)
    ctnorm = (ells_f**2 - 1.0) * (ells_f + 2) * ells_f  # E-mode normalisation
    Cl_TT *= norm
    Cl_EE *= norm * ctnorm
    Cl_TE *= norm * np.sqrt(ctnorm)

    # Convert from dimensionless (ΔT/T)² to μK²
    T0_muK2 = (p['T_cmb'] * 1e6)**2
    Cl_TT *= T0_muK2
    Cl_EE *= T0_muK2
    Cl_TE *= T0_muK2

    # Interpolate to all integer ℓ using cubic spline (captures peak structure)
    ells_all = np.arange(2, ell_max + 1)
    Dl_TT, Dl_EE, Dl_TE = [interpolate.CubicSpline(ells_compute, cl)(ells_all)
                            for cl in (Cl_TT, Cl_EE, Cl_TE)]

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
    result = compute_cls(bg, thermo, params)

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
    np.savez('nanocmb_output.npz', ells=ells, DlTT=DlTT, DlEE=DlEE, DlTE=DlTE,
             Delta_T=result['Delta_T'], Delta_E=result['Delta_E'],
             k_fine=result['k_fine'], ells_compute=result['ells_compute'])


if __name__ == '__main__':
    main()
