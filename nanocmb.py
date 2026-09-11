"""
nanoCMB — A minimal CMB angular power spectrum calculator

Computes TT, EE, and TE angular power spectra for flat ΛCDM cosmologies
in ~1600 lines of readable Python.

Features:
  - Full RECFAST recombination (H + He ODEs, matter temperature)

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
from scipy import integrate, interpolate, optimize, special
from concurrent.futures import ThreadPoolExecutor
import os

try:
    from numba import njit
    # Disable disk caching: spawned workers compile as __mp_main__.
    _jit = njit(cache=False, nogil=True)
    NUMBA_AVAILABLE = True
except ImportError:
    _jit = lambda f: f
    NUMBA_AVAILABLE = False

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
# Default: Planck-like parameters, with ALL neutrinos massless
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


def setup_background(params):
    """Precompute background density parameters from cosmological parameters.

    All densities stored in CAMB convention: grho_i such that
    grhoa2 = Σ grho_i × a^(4-n_i) represents 8πGρ_tot a⁴,
    giving H(a) = √(grhoa2/3) / a² and dη/da = √(3/grhoa2).
    """
    h = params['h']
    H0 = 100 * h / c_km_s                          # H₀ in Mpc⁻¹ (c=1)
    grhocrit_h2 = 3 * (100.0 / c_km_s)**2          # 3(H₁₀₀/c)² in Mpc⁻²

    # Photon density: ρ_γ = (4σ_SB/c) T⁴, then Ω_γh² = ρ_γ/ρ_crit,100
    T_cmb = params['T_cmb']
    rho_gamma = 4 * sigma_SB / (c_km_s * 1e3)**3 * T_cmb**4  # J/m³ / c² → kg/m³
    H100_SI = 100 * 1e3 / Mpc_in_m
    rho_crit_100 = 3 * H100_SI**2 / (8 * np.pi * G)
    omega_gamma = rho_gamma / rho_crit_100

    grhog = grhocrit_h2 * omega_gamma                           # photons (∝ a⁻⁴)
    grhornomass = grhog * 7/8 * (4/11)**(4/3) * params['N_eff']  # massless neutrinos
    grhoc = grhocrit_h2 * params['omega_c_h2']                   # CDM (∝ a⁻³)
    grhob = grhocrit_h2 * params['omega_b_h2']                   # baryons (∝ a⁻³)

    # Cosmological constant: Ω_Λ = 1 - Ω_m - Ω_r (flat universe)
    omega_m = (params['omega_c_h2'] + params['omega_b_h2']) / h**2
    omega_r = omega_gamma * (1 + 7/8 * (4/11)**(4/3) * params['N_eff']) / h**2
    grhov = grhocrit_h2 * (1 - omega_m - omega_r) * h**2        # Λ (constant)

    # Thomson scattering: κ̇ = x_e × akthom / a²
    rho_b_SI = params['omega_b_h2'] * rho_crit_100
    n_H_Mpc = (1 - params['Y_He']) * rho_b_SI / m_H * Mpc_in_m**3
    akthom = (sigma_T / Mpc_in_m**2) * n_H_Mpc

    # Helium fraction by number: f_He = Y/(not4*(1-Y)) = n_He/n_H
    f_He = params['Y_He'] / (not4 * (1 - params['Y_He']))

    return {
        'H0': H0, 'h': h,
        'grhog': grhog, 'grhornomass': grhornomass,
        'grhoc': grhoc, 'grhob': grhob, 'grhov': grhov,
        'akthom': akthom, 'f_He': f_He,
        'Y_He': params['Y_He'], 'T_cmb': T_cmb,
    }


def grhoa2(a, bg):
    """Total 8πGρ a⁴ — the key quantity in the Friedmann equation."""
    return bg['grhog'] + bg['grhornomass'] + (bg['grhoc'] + bg['grhob']) * a + bg['grhov'] * a**4


def dtauda(a, bg):
    """dη/da = 1/(a²H) = √(3/grhoa2) in Mpc."""
    return np.sqrt(3.0 / grhoa2(a, bg))


def hubble(a, bg):
    """Hubble parameter H(a) = √(grhoa2/3) / a² in Mpc⁻¹ (c=1)."""
    return np.sqrt(grhoa2(a, bg) / 3.0) / a**2


def conformal_time(a, bg):
    """η(a) in Mpc, using a cached antiderivative of dη/da on 0 <= a <= 1.

    A single smooth background table replaces thousands of separate quadratures.
    The a=0 endpoint is explicit, so every subsystem uses the same time origin.
    Rebuild the background dictionary after changing cosmological parameters.
    """
    values = np.asarray(a, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Scale factors must be finite and non-negative")
    if '_eta_primitive' not in bg:
        knots = np.r_[0.0, np.geomspace(1e-12, 1.0, 10000)]
        bg['_eta_primitive'] = interpolate.CubicSpline(
            knots, dtauda(knots, bg), extrapolate=False).antiderivative()
    # Retain the original API for future scale factors, without extrapolating.
    flat = values.ravel()
    result = np.empty_like(flat)
    inside = flat <= 1.0
    result[inside] = bg['_eta_primitive'](flat[inside])
    for index in np.flatnonzero(~inside):
        result[index] = integrate.quad(dtauda, 0, flat[index], args=(bg,),
                                       epsrel=1e-10, limit=200)[0]
    return result.reshape(values.shape).squeeze()


def sound_horizon(a, bg):
    """Comoving sound horizon r_s(a) = ∫₀ᵃ c_s da'/(a'²H)."""
    def integrand(ap):
        R = 0.75 * bg['grhob'] * ap / bg['grhog']
        return 1.0 / np.sqrt(grhoa2(ap, bg) * (1.0 + R))
    return integrate.quad(integrand, 0, a)[0]


def compute_background(params):
    """Compute background quantities: η₀, sound horizon, etc."""
    bg = setup_background(params)
    bg['tau0'] = conformal_time(1.0, bg)
    a_eq = (bg['grhog'] + bg['grhornomass']) / (bg['grhoc'] + bg['grhob'])
    bg['tau_eq'] = conformal_time(a_eq, bg)
    return bg


# ============================================================
# RECOMBINATION
# Full RECFAST: 3-variable ODE for x_H, x_He, T_mat
# with Saha equilibrium at high redshift.
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
barssc0 = k_B / (m_H * c_SI**2)                      # baryon sound speed prefactor (K⁻¹)


def compute_recombination(bg, params):
    """Solve ionisation history x_e(z) using full RECFAST.

    Three-variable ODE for hydrogen ionisation (x_H), helium ionisation (x_He),
    and matter temperature (T_mat). Uses Saha equilibrium at high z, switching
    to ODEs as each species departs from equilibrium.

    Regime switching (high z → low z):
      z > 8000:  fully ionised (x_H=1, x_He=1, He doubly ionised)
      5000–8000: He++ → He+ Saha
      3500–5000: He singly ionised, H fully ionised
      < 3500:    He+ → He0 Saha until x_He < 0.99, then full
                 3-variable ODE (x_H, x_He, T_mat) to z=0
    """
    T_cmb = bg['T_cmb']
    f_He = bg['f_He']

    # Present-day hydrogen number density (m⁻³)
    H100_SI = 100 * 1e3 / Mpc_in_m
    rho_crit_100 = 3 * H100_SI**2 / (8 * np.pi * G)
    Nnow = (1 - bg['Y_He']) * (params['omega_b_h2'] * rho_crit_100) / m_H

    # Cosmological parameters for dH/dz in T_mat equation
    H0_SI = bg['H0'] * c_SI / Mpc_in_m
    omega_m = (params['omega_b_h2'] + params['omega_c_h2']) / params['h']**2
    a_eq = (bg['grhog'] + bg['grhornomass']) / (bg['grhoc'] + bg['grhob'])
    z_eq = 1.0 / a_eq - 1.0

    def Hz_SI(z):
        return hubble(1.0 / (1 + z), bg) * c_SI / Mpc_in_m

    # --- Saha equations ---
    def saha_He2(z):
        """He++ → He+ Saha: returns total x_e per H atom."""
        T = T_cmb * (1 + z)
        rhs = (CR * T_cmb / (1 + z))**1.5 * np.exp(-CB1_He2 / T) / Nnow
        b = rhs - 1.0 - f_He
        c = (1.0 + 2.0 * f_He) * rhs
        root = np.hypot(b, 2.0 * np.sqrt(c))
        return 2.0 * c / (root + b) if b >= 0 else 0.5 * (root - b)

    def saha_He1(z):
        """He+ → He0 Saha: returns x_He = n(He+)/n_He."""
        T = T_cmb * (1 + z)
        rhs = 4.0 * (CR * T_cmb / (1 + z))**1.5 * np.exp(-CB1_He1 / T) / Nnow
        # Solve directly for x_He, avoiding cancellation in both square-root
        # subtraction and (x_e - 1)/f_He.
        return 2.0 * rhs / (1.0 + rhs + np.hypot(1.0 + rhs, 2.0 * np.sqrt(f_He * rhs)))

    # --- RECFAST ODE right-hand side ---
    def recfast_rhs(z, y):
        """dy/dz for y = [x_H, x_He, T_mat]."""
        x_H = max(y[0], 0.0)
        x_He = max(y[1], 0.0)
        T_mat = max(y[2], 0.5)

        x = x_H + f_He * x_He
        T_rad = T_cmb * (1 + z)
        n_H = Nnow * (1 + z)**3
        n_He = f_He * n_H
        Hz = Hz_SI(z)

        # --- f1: Hydrogen Peebles equation ---
        t4 = T_mat / 1e4
        Rdown = 1e-19 * 4.309 * t4**(-0.6166) / (1 + 0.6703 * t4**0.5300)
        Rup = Rdown * (CR * T_mat)**1.5 * np.exp(-CDB / T_mat)

        K = CK / Hz * (1.0
            + AGauss1 * np.exp(-((np.log(1 + z) - zGauss1) / wGauss1)**2)
            + AGauss2 * np.exp(-((np.log(1 + z) - zGauss2) / wGauss2)**2))
        fu = RECFAST_fudge
        n_1s = n_H * max(1 - x_H, 1e-30)

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

            # Singlet K_He with H continuum opacity
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

            # Triplet channel
            if x_He > 5e-9:
                a_trip = 10.0**(-16.306)
                b_trip = 0.761
                Rdown_trip = a_trip / (sq_0 * (1 + sq_0)**(1.0 - b_trip)
                                       * (1 + sq_1)**(1.0 + b_trip))
                Rup_trip = (4.0 / 3.0) * Rdown_trip * (CR * T_mat)**1.5 * np.exp(-CB1_He2St / T_mat)

                tauHe_t = A2P_t * n_He_ground * 3 / (8 * np.pi * Hz * L_He_2Pt**3)
                pHe_t = ((1 - np.exp(-tauHe_t)) / tauHe_t
                         if tauHe_t > 1e-7 else 1.0 - tauHe_t / 2.0)

                # Triplet C factor with H continuum opacity
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
    xe_total = np.empty(nz + 1)

    # --- Phase 1: Saha equilibrium ---
    # Scan forward (decreasing z) to find where He departs from Saha.
    he_ode_idx = None
    for i, z in enumerate(z_arr):
        if z > 8000.0:
            xH_arr[i] = 1.0
            xHe_arr[i] = 1.0
            xe_total[i] = 1.0 + 2.0 * f_He
        elif z > 5000.0:
            xH_arr[i] = 1.0
            xHe_arr[i] = 1.0
            xe_total[i] = saha_He2(z)
        elif z > 3500.0:
            xH_arr[i] = 1.0
            xHe_arr[i] = 1.0
            xe_total[i] = 1.0 + f_He
        elif z > 0:
            x_He = saha_He1(z)
            xHe_arr[i] = x_He
            xH_arr[i] = 1.0
            xe_total[i] = 1.0 + f_He * x_He
            if x_He < 0.99:
                he_ode_idx = i
                break
        else:
            break

    if he_ode_idx is None:
        he_ode_idx = len(z_arr) - 1

    # --- Phase 2: Full 3-variable ODE from He departure to z=0 ---
    # Stiff solver handles the H transition automatically;
    # no need for separate Saha→ODE handoff for hydrogen.
    z_ode = z_arr[he_ode_idx:]
    z_ode = z_ode[z_ode >= 0]

    # Matter temperature: radiation temperature before ODE, then ODE solution.
    Tmat_arr = T_cmb * (1.0 + z_arr)

    if len(z_ode) > 1:
        y0 = [xH_arr[he_ode_idx], xHe_arr[he_ode_idx],
              T_cmb * (1 + z_ode[0])]
        sol = integrate.solve_ivp(
            recfast_rhs,
            [z_ode[0], z_ode[-1]], y0,
            t_eval=z_ode, method='Radau', rtol=1e-6, atol=1e-10,
            max_step=5.0,
        )
        if not sol.success or sol.y.shape != (3, len(z_ode)) or not np.all(np.isfinite(sol.y)):
            raise RuntimeError(f"RECFAST integration incomplete: {sol.message}")
        xH_arr[he_ode_idx:] = sol.y[0]
        xHe_arr[he_ode_idx:] = sol.y[1]
        Tmat_arr[he_ode_idx:] = sol.y[2]

    # For the recombination phase, x_e = x_H + f_He * x_He.
    xe_total[he_ode_idx:] = xH_arr[he_ode_idx:] + f_He * xHe_arr[he_ode_idx:]

    return z_arr, xe_total, Tmat_arr


def compute_thermodynamics(bg, params):
    """Build thermodynamic tables: opacity, optical depth, visibility function.

    The visibility function g(η) = κ̇ e^{-τ} tells us the probability that a CMB
    photon last scattered at conformal time η. Its peak defines the surface of
    last scattering, and its width determines the thickness of that surface
    (which causes diffusion damping of small-scale anisotropies).
    """
    z_arr, xe_arr, Tmat_rec = compute_recombination(bg, params)

    # Add reionisation: CAMB tanh model.
    f_He = bg['f_He']
    delta_z = 0.5
    he_reion_z = 3.5
    he_reion_dz = 0.4
    he_reion_zstart = he_reion_z + 5 * he_reion_dz
    f_re = 1.0 + f_He  # Full ionisation: H + singly-ionised He
    z_rev = z_arr[::-1]
    xe_rev = xe_arr[::-1]

    def build_reion_xe(z_eval, z_re):
        """Pure reionisation model x_e(z): H tanh + second He reion."""
        if params['tau_reion'] == 0:
            return np.interp(z_eval, z_rev, xe_rev)
        z_reion_start = z_re + 8 * delta_z
        x_e_freeze = np.interp(z_reion_start, z_rev, xe_rev)
        window_var_mid = (1 + z_re)**1.5
        window_var_delta = 1.5 * (1 + z_re)**0.5 * delta_z
        xod = np.clip((window_var_mid - (1 + z_eval)**1.5) / window_var_delta, -100, 100)
        x_H_reion = (f_re - x_e_freeze) * (np.tanh(xod) + 1) / 2 + x_e_freeze
        xod_he = np.clip((he_reion_z - z_eval) / he_reion_dz, -100, 100)
        x_he_extra = np.where(z_eval < he_reion_zstart, f_He * (np.tanh(xod_he) + 1) / 2, 0.0)
        return x_H_reion + x_he_extra

    def compute_reion_optical_depth(z_re):
        """Compute optical depth from z=0 to z_reion_start using quadrature."""
        z_reion_start = z_re + 8 * delta_z
        def integrand(z):
            a = 1.0 / (1 + z)
            return build_reion_xe(z, z_re) * bg['akthom'] * dtauda(a, bg)
        return integrate.quad(integrand, 0, z_reion_start, limit=200)[0]

    # Solve for z_re matching target τ_reion.
    target_tau = params['tau_reion']
    if target_tau < 0 or not np.isfinite(target_tau):
        raise ValueError("tau_reion must be finite and non-negative")
    z_re_low, z_re_high = 0.0, 30.0
    def tau_residual(z_re):
        return compute_reion_optical_depth(z_re) - target_tau
    if target_tau == 0:
        z_re = 0.0
    else:
        f_low, f_high = tau_residual(z_re_low), tau_residual(z_re_high)
        if f_low * f_high > 0:
            raise ValueError("tau_reion is outside the tanh model's supported range "
                             f"[{f_low + target_tau:.6g}, {f_high + target_tau:.6g}]; "
                             "use tau_reion=0 to disable reionisation")
        z_re = optimize.brentq(tau_residual, z_re_low, z_re_high, xtol=1e-8, rtol=1e-8)

    # Refine z_arr in the reionisation region
    z_reion_lo = max(0.01, z_re - 8 * delta_z)
    z_reion_hi = z_re + 8 * delta_z
    z_dense = np.linspace(z_reion_hi, z_reion_lo, 200)
    mask = (z_arr > z_reion_hi) | (z_arr < z_reion_lo)
    z_new = np.sort(np.concatenate([z_arr[mask], z_dense]))[::-1]
    # Interpolate recombination x_e onto refined grid (z_arr is high→low, flip for interp)
    xe_new = np.interp(z_new[::-1], z_arr[::-1], xe_arr[::-1])[::-1]
    Tmat_new = np.interp(z_new[::-1], z_arr[::-1], Tmat_rec[::-1])[::-1]
    z_arr = z_new
    xe_arr = xe_new
    Tmat_rec = Tmat_new

    xe_final = np.maximum(build_reion_xe(z_arr, z_re), xe_arr)

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

    # Build interpolators on the conformal time grid (τ_arr is increasing).
    thermo = {
        'z_arr': z_arr,
        'a_arr': a_arr,
        'tau_arr': tau_arr,
        'xe': xe_final,
        'Tmat': Tmat_rec,
        'opacity': opacity,           # κ̇(η)
        'tau_optical': tau_optical,    # optical depth τ(η)
        'exptau': exptau,            # e^{-τ}
        'visibility': visibility,     # g(η) = κ̇ e^{-τ}
        'z_reion': z_re,
        'reionization': target_tau > 0,
    }

    # Shape-preserving cubics avoid overshoots within the thermodynamics table.
    with np.errstate(over='ignore'):  # harmless reciprocal overflow for subnormal e^-τ
        thermo['opacity_interp'] = interpolate.PchipInterpolator(tau_arr, opacity)
        thermo['exptau_interp'] = interpolate.PchipInterpolator(tau_arr, exptau)
        thermo['visibility_interp'] = interpolate.PchipInterpolator(tau_arr, visibility)

    # Baryon sound speed from thermodynamics (CAMB-style structure).
    # c_s,b^2 = (k_B T_m / m_H c^2) * [1 - (d ln T_m / d ln a)/3], with
    # composition factor in mean molecular weight.
    dlnT_dln_a = np.gradient(np.log(np.maximum(Tmat_rec, 1e-30)), np.log(a_arr))
    barssc = barssc0 * (1.0 - 0.75 * bg['Y_He'] + (1.0 - bg['Y_He']) * xe_arr)
    cs2_b = np.maximum(barssc * Tmat_rec * (1.0 - dlnT_dln_a / 3.0), 0.0)
    thermo['cs2_b'] = cs2_b

    # Find the peak of the visibility function (surface of last scattering)
    peak_idx = np.argmax(visibility)
    thermo['z_star'] = z_arr[peak_idx]
    thermo['tau_star'] = tau_arr[peak_idx]

    # --- Derived quantities for optimal grid construction ---
    a_star = 1.0 / (1.0 + thermo['z_star'])
    fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))

    # Sound horizon
    thermo['r_s'] = sound_horizon(a_star, bg)

    # Silk damping scale (tabulated xe, so use trapezoidal on fine grid)
    a_grid = np.linspace(a_arr[0], a_star, 5000)
    R = 0.75 * bg['grhob'] * a_grid / bg['grhog']
    dtauda_grid = dtauda(a_grid, bg)
    kappa_dot = np.maximum(np.interp(a_grid, a_arr, xe_final) * bg['akthom'] / a_grid**2, 1e-30)
    integrand_D = (R**2 + 16.0*(1.0+R)/15.0) / (6.0*(1.0+R)**2 * kappa_dot) * dtauda_grid
    thermo['k_D'] = 1.0 / np.sqrt(np.trapezoid(integrand_D, a_grid))

    # Visibility function width (Gaussian σ from FWHM)
    half_max = visibility[peak_idx] / 2.0
    tau_left = np.interp(half_max, visibility[:peak_idx+1], tau_arr[:peak_idx+1])
    tau_right = np.interp(half_max, visibility[peak_idx:][::-1], tau_arr[peak_idx:][::-1])
    thermo['delta_tau_rec'] = (tau_right - tau_left) / fwhm_to_sigma

    # Reionization conformal time and width
    z_rev, tau_rev = z_arr[::-1], tau_arr[::-1]
    thermo['tau_reion'] = np.interp(z_re, z_rev, tau_rev)
    thermo['delta_tau_reion'] = abs(
        np.interp(z_re + 6*delta_z, z_rev, tau_rev)
        - np.interp(max(0.01, z_re - 6*delta_z), z_rev, tau_rev)) / fwhm_to_sigma

    return thermo


# ============================================================
# PERTURBATIONS
# Evolve the coupled Einstein-Boltzmann equations in synchronous gauge
# (CDM frame, matching CAMB). The tight-coupling approximation handles
# the stiff photon-baryon coupling at early times; the full Boltzmann
# hierarchy takes over when the photon mean free path ~ wavelength.
# ============================================================

# Hierarchy truncation (increase for higher ℓ_max accuracy)
LMAXG = 15       # photon brightness: F_gamma0 ... F_gamma_LMAXG
LMAXPOL = 15     # photon polarisation: E₂ ... E_LMAXPOL
LMAXNR = 15      # massless-neutrino brightness: F_nu0 ... F_nu_LMAXNR

# State vector layout (flat arrays for scipy ODE solver)
IX_ETAK = 0
IX_CLXC = 1
IX_CLXB = 2
IX_VB = 3
IX_G = 4                                    # F_gamma0 at IX_G, F_gamma1 at IX_G+1, etc.
IX_POL = IX_G + LMAXG + 1                   # E₂ at IX_POL, E₃ at IX_POL+1, etc.
IX_R = IX_POL + LMAXPOL - 1                 # F_nu0 at IX_R, F_nu1 at IX_R+1, etc.
NVAR = IX_R + LMAXNR + 1


# --- Numba-accelerated helpers (fall back to plain Python without numba) ---

@_jit
def _cubic_eval(x_knots, coeffs, t, derivative=False):
    """Evaluate a cubic PPoly (CubicSpline or PCHIP), or its first derivative."""
    n = x_knots.shape[0] - 1
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if x_knots[mid + 1] < t:
            lo = mid + 1
        else:
            hi = mid
    dt = t - x_knots[lo]
    if derivative:
        return (3.0 * coeffs[0, lo] * dt + 2.0 * coeffs[1, lo]) * dt + coeffs[2, lo]
    return ((coeffs[0, lo] * dt + coeffs[1, lo]) * dt + coeffs[2, lo]) * dt + coeffs[3, lo]


def setup_perturbation_grid(bg, thermo):
    """Precompute a(τ) and background quantities on a fine conformal time grid.

    We need these at arbitrary times during ODE integration, so we build
    spline interpolators covering the full range from the deep radiation era
    (a ~ 10⁻⁹) to today (a = 1).
    """
    # Reuse the thermodynamics time origin (include η(a_min), not η=0).
    a_grid = np.logspace(-9, 0, 10000)
    tau_grid = conformal_time(a_grid, bg)

    # a(τ) interpolator (the inverse mapping we need during integration)
    a_of_tau = interpolate.CubicSpline(tau_grid, a_grid)

    # Radiation era: da/dη = a²H ≈ √(grho_rad/3), a constant.
    grho_rad = bg['grhog'] + bg['grhornomass']
    adotrad = np.sqrt(grho_rad / 3.0)

    # Build extended interpolators covering all times.
    # Before the thermodynamics grid (z > 10000): fully ionised, analytic values.
    # Within the thermodynamics grid: use the computed values.
    tau_thermo = thermo['tau_arr']
    tau_early = tau_grid[tau_grid < tau_thermo[0]]
    a_early = a_of_tau(tau_early)
    tau_ext = np.concatenate([tau_early, tau_thermo])

    # Fully ionised helium supplies TWO electrons per He nucleus.
    opac_early = (1.0 + 2.0 * bg['f_He']) * bg['akthom'] / a_early**2
    opacity_interp = interpolate.PchipInterpolator(
        tau_ext, np.concatenate([opac_early, thermo['opacity']]))

    # Baryon sound speed: c_s² = (4/3) k_B T_r / (μ m_H) at early times (T_m = T_r)
    xe_early = 1.0 + 2.0 * bg['f_He']
    barssc_early = barssc0 * (1.0 - 0.75 * bg['Y_He'] + (1.0 - bg['Y_He']) * xe_early)
    cs2_early = (4.0 / 3.0) * barssc_early * bg['T_cmb'] / a_early
    cs2_interp = interpolate.PchipInterpolator(
        tau_ext, np.concatenate([cs2_early, thermo['cs2_b']]))

    return {
        'sp_a_x': a_of_tau.x,
        'sp_a_c': a_of_tau.c,
        'sp_op_x': opacity_interp.x,
        'sp_op_c': opacity_interp.c,
        'sp_cs_x': cs2_interp.x,
        'sp_cs_c': cs2_interp.c,
        'bg_vec': np.array([bg['grhog'], bg['grhornomass'], bg['grhoc'],
                            bg['grhob'], bg['grhov']]),
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
        y0[IX_R + 3] = -4.0 / 21.0 / Rp15 * x2 * x           # F_nu3

    # All higher multipoles and polarisation start at zero
    return y0


@_jit
def _common_terms(tau, y, k, bg_vec, sp_a_x, sp_a_c):
    """Shared background+Einstein terms used by RHS and source construction."""
    grhog, grhornomass, grhoc, grhob, grhov = bg_vec[0], bg_vec[1], bg_vec[2], bg_vec[3], bg_vec[4]
    a = _cubic_eval(sp_a_x, sp_a_c, tau)
    a2 = a * a
    grhog_t = grhog / a2
    grhor_t = grhornomass / a2
    grhoc_t = grhoc / a
    grhob_t = grhob / a
    grho_a2 = grhog_t + grhor_t + grhoc_t + grhob_t + grhov * a2
    adotoa = np.sqrt(grho_a2 / 3.0)

    etak = y[IX_ETAK]
    clxc = y[IX_CLXC]
    clxb = y[IX_CLXB]
    vb = y[IX_VB]
    clxg = y[IX_G]
    qg = y[IX_G + 1]
    pig = y[IX_G + 2]
    clxr = y[IX_R]
    qr = y[IX_R + 1]
    pir = y[IX_R + 2]

    k2 = k * k
    dgrho = grhob_t * clxb + grhoc_t * clxc + grhog_t * clxg + grhor_t * clxr
    dgq = grhob_t * vb + grhog_t * qg + grhor_t * qr
    Z = (0.5 * dgrho / k + etak) / adotoa
    sigma = Z + 1.5 * dgq / k2

    return (a, adotoa, grhog_t, grhor_t, grhoc_t, grhob_t,
            dgrho, dgq, Z, sigma,
            etak, clxc, clxb, vb, clxg, qg, pig, clxr, qr, pir)


@_jit
def _boltzmann_rhs(tau, y, k, bg_vec, sp_a_x, sp_a_c, sp_op_x, sp_op_c, sp_cs_x, sp_cs_c):
    """Right-hand side of the Boltzmann hierarchy: dy/dτ.

    Implements the synchronous gauge equations from CAMB's derivs() subroutine.
    During tight coupling (early times, high opacity), only the photon monopole
    and dipole are evolved; the quadrupole is computed algebraically.
    """
    # Background and Einstein-source terms
    (a, adotoa, grhog_t, grhor_t, grhoc_t, grhob_t,
     dgrho, dgq, Z, sigma,
     etak, clxc, clxb, vb, clxg, qg, pig, clxr, qr, pir) = \
        _common_terms(tau, y, k, bg_vec, sp_a_x, sp_a_c)
    opacity = max(_cubic_eval(sp_op_x, sp_op_c, tau), 1e-30)
    cs2_b = max(_cubic_eval(sp_cs_x, sp_cs_c, tau), 0.0)
    photbar = grhog_t / grhob_t
    pb43 = 4.0 / 3.0 * photbar
    delta_p_b = cs2_b * clxb

    # Determine if tight coupling is active
    tight_coupling = (k / opacity < 0.01) and (1.0 / (opacity * tau) < 0.01)

    E2 = y[IX_POL] if LMAXPOL >= 2 else 0.0

    cothxor = 1.0 / tau

    dy = np.zeros(NVAR)

    # --- Metric equation ---
    dy[IX_ETAK] = 0.5 * dgq

    # --- CDM: at rest in this gauge ---
    dy[IX_CLXC] = -k * Z

    # --- Baryons ---
    dy[IX_CLXB] = -k * (Z + vb)

    if tight_coupling:
        # Tight-coupling: photon-baryon fluid locked together
        pig_tc = 32.0 / 45.0 * k / opacity * (sigma + vb)
        polter = pig_tc / 4.0

        vbdot = (-adotoa * vb + k * delta_p_b + k / 4.0 * pb43 * (clxg - 2.0 * pig_tc)) / (1.0 + pb43)
        # First-order derivative of the slip vb - 3*qg/4 (CAMB derivs()).
        dopacity = _cubic_eval(sp_op_x, sp_op_c, tau, True)
        gpressure = (grhog_t + grhor_t) / 3.0 - bg_vec[4] * a * a
        addot_over_a = 0.5 * (adotoa * adotoa - gpressure)
        clxgdot = -k * (4.0 / 3.0 * Z + qg)
        slipdot = (-(2.0 * adotoa / (1.0 + pb43) + dopacity / opacity)
                   * (vb - 0.75 * qg)
                   + (-addot_over_a * vb - 0.5 * k * adotoa * clxg
                      + k * (cs2_b * dy[IX_CLXB] - clxgdot / 4.0))
                   / (opacity * (1.0 + pb43)))
        vbdot += pb43 / (1.0 + pb43) * slipdot
        dy[IX_VB] = vbdot

        dy[IX_G] = -k * (4.0 / 3.0 * Z + qg)
        qgdot = 4.0 / 3.0 * (-vbdot - adotoa * vb + k * delta_p_b) / pb43 + k / 3.0 * clxg - 2.0 * k / 3.0 * pig_tc
        dy[IX_G + 1] = qgdot
        dy[IX_G + 2] = opacity * (pig_tc - pig)

        if LMAXPOL >= 2:
            dy[IX_POL] = opacity * (pig_tc / 4.0 - E2)

    else:
        # Full Boltzmann hierarchy
        # Polter: polarisation source Π = pig/10 + 3E₂/5
        polter = pig / 10.0 + 9.0 / 15.0 * E2
        vbdot = -adotoa * vb + k * delta_p_b - photbar * opacity * (4.0 / 3.0 * vb - qg)
        dy[IX_VB] = vbdot

        dy[IX_G] = -k * (4.0 / 3.0 * Z + qg)
        qgdot = 4.0 / 3.0 * (-vbdot - adotoa * vb + k * delta_p_b) / pb43 + k / 3.0 * clxg - 2.0 * k / 3.0 * pig
        dy[IX_G + 1] = qgdot

        Fgamma3 = y[IX_G + 3] if LMAXG >= 3 else 0.0
        dy[IX_G + 2] = (2.0 * k / 5.0 * qg - 3.0 * k / 5.0 * Fgamma3
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
    dy[IX_R] = -k * (4.0 / 3.0 * Z + qr)
    dy[IX_R + 1] = k / 3.0 * (clxr - 2.0 * pir)

    Fnu3 = y[IX_R + 3] if LMAXNR >= 3 else 0.0
    dy[IX_R + 2] = 2.0 * k / 5.0 * qr - 3.0 * k / 5.0 * Fnu3 + 8.0 / 15.0 * k * sigma

    for l in range(3, LMAXNR):
        dy[IX_R + l] = (k * l / (2*l + 1) * y[IX_R + l - 1]
                        - k * (l + 1) / (2*l + 1) * y[IX_R + l + 1])

    dy[IX_R + LMAXNR] = (k * y[IX_R + LMAXNR - 1]
                          - (LMAXNR + 1) * cothxor * y[IX_R + LMAXNR])

    return dy


def compute_source_functions(tau, y, k, pgrid, thermo):
    """Compute CMB source function building blocks at a single (k, τ) point.

    After integration by parts, the temperature source decomposes into three
    Bessel channels: j_ℓ (ISW + monopole + quadrupole), j_ℓ' (Doppler),
    and j_ℓ'' (quadrupole). Returns the coefficients for each channel plus
    the E-mode source.
    """
    (a, adotoa, grhog_t, grhor_t, grhoc_t, grhob_t,
     dgrho, dgq, Z, sigma,
     etak, clxc, clxb, vb, clxg, qg, pig, clxr, qr, pir) = _common_terms(
        tau, y, k, pgrid['bg_vec'], pgrid['sp_a_x'], pgrid['sp_a_c']
    )
    opacity = max(float(_cubic_eval(pgrid['sp_op_x'], pgrid['sp_op_c'], tau)), 1e-30)
    k2 = k * k

    E2 = y[IX_POL] if LMAXPOL >= 2 else 0.0

    dgpi = grhog_t * pig + grhor_t * pir
    phi = -((dgrho + 3.0 * dgq * adotoa / k) + dgpi) / (2.0 * k2)

    # Φ̇ for the ISW effect — compute pigdot, pirdot directly from the
    # Boltzmann hierarchy equations (avoids re-evaluating full RHS)
    polter = pig / 10.0 + 9.0 / 15.0 * E2
    Fgamma3 = y[IX_G + 3] if LMAXG >= 3 else 0.0
    Fnu3 = y[IX_R + 3] if LMAXNR >= 3 else 0.0
    pigdot = (2*k/5*qg - 3*k/5*Fgamma3 - opacity*(pig - polter) + 8*k*sigma/15)
    pirdot = (2*k/5*qr - 3*k/5*Fnu3 + 8*k*sigma/15)
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
    # Starting time: kτ_start = 0.01 (safely in the super-horizon regime)
    tau_start = min(0.01 / k, tau_out[0] * 0.5)
    # Stay inside the background table while keeping kη_start <= 0.01.
    tau_start = max(tau_start, float(pgrid['sp_a_x'][0]))
    if tau_start >= tau_out[0] or k * tau_start > 0.01 * (1 + 1e-12):
        raise ValueError("k/tau_out requires initial conditions before the background table")

    # Initial conditions
    y0 = adiabatic_ics(k, tau_start, bg, pgrid)

    # Evolve with LSODA (auto-switches Adams/BDF for non-stiff/stiff regimes)
    bg_vec = pgrid['bg_vec']
    sp_a_x = pgrid['sp_a_x']
    sp_a_c = pgrid['sp_a_c']
    sp_op_x = pgrid['sp_op_x']
    sp_op_c = pgrid['sp_op_c']
    sp_cs_x = pgrid['sp_cs_x']
    sp_cs_c = pgrid['sp_cs_c']

    rhs_args = (k, bg_vec, sp_a_x, sp_a_c, sp_op_x, sp_op_c, sp_cs_x, sp_cs_c)
    sol = integrate.solve_ivp(
        _boltzmann_rhs,
        [tau_start, tau_out[-1]],
        y0,
        t_eval=tau_out,
        method='LSODA', args=rhs_args,
        rtol=pgrid.get('ode_rtol', 1e-5), atol=pgrid.get('ode_atol', 1e-8),
        max_step=pgrid.get('ode_max_step', 20.0),
    )

    ntau = len(tau_out)
    if not sol.success or sol.y.shape != (NVAR, ntau) or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"ODE solver failed/incomplete for k={k:.4e}: {sol.message}")

    # --- Extract source function building blocks at each time step ---
    ISW_arr, monopole_arr, sigma_plus_vb_arr, vis_arr, polter_arr, src_E = \
        (np.zeros(ntau) for _ in range(6))
    for i, tau in enumerate(tau_out):
        (ISW_arr[i], monopole_arr[i], sigma_plus_vb_arr[i],
         vis_arr[i], polter_arr[i], src_E[i]) = \
            compute_source_functions(tau, sol.y[:, i], k, pgrid, thermo)

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
# GRID CONSTRUCTION
# Non-uniform grids in k and τ via equidistribution of
# trapezoidal quadrature error: node density ∝ |f''|^(1/3).
# ============================================================

def _checked_grid(values, name, minimum=2):
    values = np.ascontiguousarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < minimum
            or not np.all(np.isfinite(values)) or np.any(values <= 0)
            or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must contain >= {minimum} finite, positive, strictly increasing points")
    return values


def k_grid(N, mode, bg, thermo, params,
                   k_min=1e-5, k_max=0.5,
                   ell_min=2, ell_max=2500, n_ell_samples=30,
                   n_eval=5000):
    """Compute an optimal non-uniform k-grid for CMB computation.

    mode="cl" optimised for C_ell integration; mode="ode" for source interpolation.
    """
    x = np.linspace(np.log(k_min), np.log(k_max), n_eval)
    k = np.exp(x)

    # Shared quantities
    primordial = k ** (params['n_s'] + 2)
    acoustic_curv = (1.0 / thermo['r_s']) ** 2
    # Silk damping envelope: exp(-(k/k_D)^2)
    damped = primordial * np.exp(-((k / thermo['k_D']) ** 2))

    if mode == "cl":
        sigma_k = 1.0 / thermo['delta_tau_rec']
        chi_star = bg['tau0'] - thermo['tau_star']
        ells = np.unique(np.geomspace(ell_min, ell_max, n_ell_samples).astype(int))
        raw_weight = np.zeros_like(k)
        for ell in ells:
            envelope = np.exp(-0.5 * ((k - ell / chi_star) / (3.0 * sigma_k)) ** 2)
            curv = np.maximum(acoustic_curv, sigma_k ** 2) * envelope
            raw_weight += curv * damped
        floor = 1e-6 * np.max(raw_weight)
    else:
        smooth_curv = (k * thermo['r_s']) ** 2 / bg['tau_eq'] ** 2
        raw_weight = np.maximum(acoustic_curv, smooth_curv) * damped
        floor = 0.005 * np.max(raw_weight)

    density = (raw_weight + floor) ** (1.0 / 3.0)
    dx = x[1] - x[0]
    cdf = np.cumsum(density) * dx
    cdf -= cdf[0]
    cdf /= cdf[-1]

    k_grid = np.exp(np.interp(np.linspace(0, 1, N), cdf, x))
    k_grid[0] = k_min
    k_grid[-1] = k_max
    return k_grid


def tau_grid(N, k_max, bg, thermo,
                     tau_min=1.0, tau_max=None, n_eval=10000):
    """Compute an optimal non-uniform tau grid for the LOS integral."""
    if tau_max is None:
        tau_max = bg['tau0']

    if N < 2 or int(N) != N or n_eval < 2 or not 0 < tau_min < tau_max:
        raise ValueError("Invalid time-grid size or bounds")
    tau = np.linspace(tau_min, tau_max, n_eval)
    tau_star = thermo['tau_star']
    delta_tau_rec = thermo['delta_tau_rec']

    # Recombination: visibility peak + acoustic source structure
    g_rec = np.exp(-0.5 * ((tau - tau_star) / delta_tau_rec) ** 2)
    g_broad = np.exp(-0.5 * ((tau - tau_star) / thermo['r_s']) ** 2)
    weight = g_rec / delta_tau_rec**2 + g_broad * (k_max / np.sqrt(3.0))**2

    # Reionization
    g_reion = np.exp(-0.5 * ((tau - thermo['tau_reion']) / thermo['delta_tau_reion']) ** 2)
    if thermo.get('reionization', True):
        weight += 0.3 * g_reion / thermo['delta_tau_reion'] ** 2

    density = (weight + 0.005 * np.max(weight)) ** (1.0 / 3.0)
    dtau = tau[1] - tau[0]
    cdf = np.cumsum(density) * dtau
    cdf -= cdf[0]
    cdf /= cdf[-1]

    tau_grid = np.interp(np.linspace(0, 1, N), cdf, tau)
    tau_grid[0] = tau_min
    tau_grid[-1] = tau_max
    return tau_grid


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


_bessel_cache = {}

# Round required x_max up to shared cache buckets so nearby cosmologies
# can reuse a single Bessel table rather than rebuilding every run.
BESSEL_CACHE_X_BUCKET = 250.0


def _build_bessel_tables(ells_compute, x_max, dx):
    """Precompute j_l(x) and j_{l+1}(x) on a uniform x-grid for LOS interpolation.

    Results are cached in memory keyed by (ells, x_max, dx). The cache is
    also persisted to disk so that subsequent runs skip the expensive
    scipy.special.jv evaluation entirely.
    """
    # Cache key: ells tuple + quantized x_max + grid parameters.
    x_max_cache = np.ceil(x_max / BESSEL_CACHE_X_BUCKET) * BESSEL_CACHE_X_BUCKET
    x_max_cache = max(x_max_cache, x_max + dx)
    cache_key = (tuple(ells_compute.astype(int)), round(x_max_cache, 2), dx)
    if cache_key in _bessel_cache:
        return _bessel_cache[cache_key]

    # Try loading from disk
    import hashlib, os
    key_str = f"{list(cache_key[0])}_{cache_key[1]}_{cache_key[2]}"
    cache_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
    cache_file = os.path.join(cache_dir, f'bessel_{cache_hash}.npz')

    if os.path.exists(cache_file):
        with np.load(cache_file, allow_pickle=False) as data:
            result = (float(data['x0']), float(data['inv_dx']),
                      int(data['n_x']), data['jl_tab'], data['jl1_tab'])
        _bessel_cache.clear()
        _bessel_cache[cache_key] = result
        return result

    print("Building Bessel tables (first run only, will be cached)...")
    n_x = int(np.ceil(x_max_cache / dx)) + 2
    x_tab = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x_tab > 1e-30, x_tab, 1.0)
    pref = np.where(x_tab > 1e-30, np.sqrt(np.pi / (2.0 * x_safe)), 0.0)

    nell = len(ells_compute)
    jl_tab = np.empty((nell, n_x))
    jl1_tab = np.empty((nell, n_x))
    # Fill final tables directly, reusing adjacent orders at low multipoles.
    for i, ell in enumerate(ells_compute):
        for order, table in ((int(ell), jl_tab), (int(ell) + 1, jl1_tab)):
            if table is jl_tab and i > 0 and ell == ells_compute[i-1] + 1:
                table[i] = jl1_tab[i-1]
                continue
            x_min = max(0.0, order - 4.0 * order**(1.0/3.0))
            i_start = max(0, int(x_min / dx) - 1)
            table[i, :i_start] = 0.0
            table[i, i_start:] = pref[i_start:] * special.jv(order + 0.5, x_tab[i_start:])
            if i_start == 0:
                table[i, 0] = 1.0 if order == 0 else 0.0

    result = (x_tab[0], 1.0 / dx, n_x, jl_tab, jl1_tab)
    _bessel_cache.clear()
    _bessel_cache[cache_key] = result

    # Persist to disk
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file, x0=x_tab[0], inv_dx=1.0/dx, n_x=n_x,
             jl_tab=jl_tab, jl1_tab=jl1_tab)

    return result


def _interp_uniform_table(x, x0, inv_dx, n_x, vals):
    """Linear interpolation for values on a uniform x-grid."""
    u = (x - x0) * inv_dx
    idx = np.floor(u).astype(np.int64)
    idx = np.clip(idx, 0, n_x - 2)
    u = np.clip(u, 0.0, n_x - 1.0)
    frac = u - idx
    return (1.0 - frac) * vals[idx] + frac * vals[idx + 1]


def _akima_columns(x, y, x_new):
    """Batched Akima with INDEPENDENT column thresholds, matching scalar SciPy.

    Simply passing axis=0 to SciPy Akima uses one global slope threshold;
    that couples tiny columns to large columns. Here each time slice keeps
    the same threshold as a separate one-dimensional interpolation.
    """
    m = np.empty((len(x) + 3, y.shape[1]))
    m[2:-2] = np.diff(y, axis=0) / np.diff(x)[:, None]
    m[1] = 2*m[2] - m[3]
    m[0] = 2*m[1] - m[2]
    m[-2] = 2*m[-3] - m[-4]
    m[-1] = 2*m[-2] - m[-3]
    dm = np.abs(np.diff(m, axis=0))
    f1, f2 = dm[2:], dm[:-2]
    denominator = f1 + f2
    slopes = 0.5 * (m[3:] + m[:-3])
    mask = denominator > 1e-9 * np.max(denominator, axis=0, keepdims=True)
    fraction = np.divide(f2, denominator, out=np.zeros_like(f2), where=mask)
    candidate = m[1:-2] + fraction * (m[2:-1] - m[1:-2])
    slopes[mask] = candidate[mask]
    return np.ascontiguousarray(interpolate.CubicHermiteSpline(
        x, y, slopes, axis=0, extrapolate=False)(x_new))


def _trapezoid_weights(x):
    """Positive weights whose dot product equals the non-uniform trapezoid rule."""
    weights = np.empty_like(x)
    weights[0], weights[-1] = (x[1] - x[0])/2, (x[-1] - x[-2])/2
    weights[1:-1] = (x[2:] - x[:-2])/2
    return weights


@_jit
def _small_x_bessel(ell, x):
    """j_l, j_l', j_l'' from a regular series, including x=0 (ell >= 2)."""
    # x^(ell-2)/(2ell+1)!! avoids dividing tiny interpolated j_l by x^2.
    leading = x*0.0 + 1.0 / 15.0
    for order in range(3, ell + 1):
        leading *= x / (2.0 * order + 1.0)
    x2 = x*x
    a = -1.0 / (2.0 * (2.0*ell + 3.0))
    b = -a / (4.0 * (2.0*ell + 5.0))
    c = -b / (6.0 * (2.0*ell + 7.0))
    j = leading*x2*(1.0 + x2*(a + x2*(b + x2*c)))
    d = leading*x*(ell + x2*((ell+2)*a + x2*((ell+4)*b + x2*(ell+6)*c)))
    dd = leading*(ell*(ell-1) + x2*((ell+2)*(ell+1)*a
                   + x2*((ell+4)*(ell+3)*b + x2*(ell+6)*(ell+5)*c)))
    return j, d, dd


@_jit
def _los_integrals(ell, ks, chi, weights, s0, s1, s2, se,
                   x0, inv_dx, nx, jt, jnext):
    """Fused projection: no k-by-time temporary arrays; no fast-math reassociation."""
    temperature = np.zeros(len(ks))
    polarization = np.zeros(len(ks))
    for ik in range(len(ks)):
        total_t, total_e = 0.0, 0.0
        for it in range(len(chi)):
            x = ks[ik]*chi[it]
            if x < 0.1:
                j, d, dd = _small_x_bessel(ell, x)
            else:
                u = min(max((x - x0)*inv_dx, 0.0), nx - 1.0)
                index = min(int(u), nx - 2)
                fraction = u - index
                j = jt[index] + fraction*(jt[index+1] - jt[index])
                jp = jnext[index] + fraction*(jnext[index+1] - jnext[index])
                d = ell/x*j - jp
                dd = -2.0/x*d + (ell*(ell+1)/(x*x) - 1.0)*j
            total_t += weights[it]*(s0[ik,it]*j + s1[ik,it]*d + s2[ik,it]*dd)
            total_e += weights[it]*se[ik,it]*j
        temperature[ik], polarization[ik] = total_t, total_e
    return temperature, polarization


def _los_numpy(ell, ks, chi, weights, s0, s1, s2, se,
                x0, inv_dx, nx, jt, jnext, chunk=64):
    """Bounded-memory vectorized fallback when Numba is not installed."""
    temperature, polarization = np.zeros(len(ks)), np.zeros(len(ks))
    for start in range(0, len(ks), chunk):
        end = min(start+chunk, len(ks))
        x = ks[start:end, None]*chi[None, :]
        j = _interp_uniform_table(x, x0, inv_dx, nx, jt)
        jp = _interp_uniform_table(x, x0, inv_dx, nx, jnext)
        safe = np.maximum(x, 0.1)
        d = ell/safe*j - jp
        dd = -2.0/safe*d + (ell*(ell+1)/safe**2 - 1.0)*j
        small = x < 0.1
        if np.any(small):
            j[small], d[small], dd[small] = _small_x_bessel(ell, x[small])
        temperature[start:end] = np.sum((s0[start:end]*j + s1[start:end]*d
                                        + s2[start:end]*dd)*weights, axis=1)
        polarization[start:end] = np.sum(se[start:end]*j*weights, axis=1)
    return temperature, polarization


def ell_grid(ell_max, step=25):
    """Dense low multipoles, controllable acoustic sampling, exact upper endpoint."""
    if int(ell_max) != ell_max or ell_max < 2 or int(step) != step or step < 1:
        raise ValueError("ell_max >= 2 and step >= 1 must be integers")
    values = np.unique(np.r_[np.arange(2, 40), np.arange(40, 200, 5),
                              np.arange(200, ell_max+1, step), ell_max])
    return values[values <= ell_max].astype(int)


def _interpolate_spectra(ells, tt, ee, te, output):
    """Cubic reconstruction of acoustic features, with an ell_max=2 special case."""
    if len(ells) == 1:
        return tuple(np.full(len(output), value[0]) for value in (tt, ee, te))
    return tuple(interpolate.CubicSpline(ells, value)(output) for value in (tt, ee, te))


def compute_cls(bg, thermo, params, N_k_ode=200, N_k_fine=4000, N_tau=1000,
                k_arr=None, k_fine=None, tau_out=None, *, n_workers=None,
                los_workers=None, ell_step=25, ells_compute=None, bessel_dx=0.03,
                ode_rtol=1e-5, ode_atol=1e-8, ode_max_step=20.0):
    """Main pipeline: evolve all k modes, do LOS integration, assemble Cℓ.

    This is the computational core of nanoCMB. For each wavenumber k, we
    evolve the Boltzmann hierarchy and extract source functions. Then for
    each multipole ℓ, we convolve with j_ℓ(k(τ₀−τ)) and integrate over k.

    Grids k_arr, k_fine, tau_out can be passed directly; otherwise they are
    built from the N_k_ode / N_k_fine / N_tau defaults.

    n_workers controls independent LSODA processes (1 is notebook-friendly).
    los_workers controls projection threads. ell_step sets sampling above
    ell=200; ells_compute can instead specify exact multipoles (endpoints
    are always included). bessel_dx and ode_* expose numerical controls.
    """
    for name, value in (('ode_rtol', ode_rtol), ('ode_atol', ode_atol),
                        ('ode_max_step', ode_max_step), ('bessel_dx', bessel_dx)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if int(params['ell_max']) != params['ell_max'] or params['ell_max'] < 2:
        raise ValueError("ell_max must be an integer >= 2")
    print("Setting up perturbation grid...")
    pgrid = setup_perturbation_grid(bg, thermo)
    pgrid.update(ode_rtol=ode_rtol, ode_atol=ode_atol, ode_max_step=ode_max_step)
    tau0 = bg['tau0']
    tau_star = thermo['tau_star']

    # --- Build grids (use provided arrays or construct defaults) ---
    if k_arr is None:
        k_arr = k_grid(N=N_k_ode, mode="ode", bg=bg, thermo=thermo, params=params)
    k_arr = _checked_grid(k_arr, 'k_arr', minimum=3)
    nk = len(k_arr)
    print(f"  {nk} k-modes from {k_arr[0]:.1e} to {k_arr[-1]:.1e} Mpc⁻¹")
    if k_fine is None:
        k_fine = k_grid(N=N_k_fine, mode="cl", bg=bg, thermo=thermo, params=params,
                        k_min=k_arr[0], k_max=k_arr[-1], ell_max=params['ell_max'])
    k_fine = _checked_grid(k_fine, 'k_fine')
    if k_fine[0] < k_arr[0] or k_fine[-1] > k_arr[-1]:
        raise ValueError("k_fine must lie inside k_arr: source extrapolation is not supported")
    if tau_out is None:
        tau_out = tau_grid(N=N_tau, k_max=k_arr[-1], bg=bg, thermo=thermo, tau_min=1.0, tau_max=tau0 - 1)
    tau_out = _checked_grid(tau_out, 'tau_out')
    if tau_out[-1] >= tau0:
        raise ValueError("tau_out must end before today (the E source contains 1/chi^2)")
    ntau = len(tau_out)
    print(f"  {ntau} output time steps")

    # --- Evolve all k modes and store source functions ---
    print("Evolving perturbations...")
    _args = (bg, thermo, pgrid, tau_out)

    # Fork workers inherit this warmup; spawn workers compile on their first use.
    _boltzmann_rhs(tau_out[0], np.zeros(NVAR), k_arr[0],
                   pgrid['bg_vec'], pgrid['sp_a_x'], pgrid['sp_a_c'],
                   pgrid['sp_op_x'], pgrid['sp_op_c'],
                   pgrid['sp_cs_x'], pgrid['sp_cs_c'])

    import multiprocessing as mp, sys
    n_workers = min(4, mp.cpu_count(), nk) if n_workers is None else n_workers
    if int(n_workers) != n_workers or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")
    # Spawn cannot re-import stdin/REPL entry points; use the serial path there.
    main_file = getattr(sys.modules['__main__'], '__file__', '')
    if mp.get_start_method() == 'spawn' and not (main_file and os.path.exists(main_file)):
        n_workers = 1
    if n_workers == 1:
        results = [evolve_k(k, bg, thermo, pgrid, tau_out) for k in k_arr]
    else:
        try:
            with mp.Pool(min(int(n_workers), nk), initializer=_pool_init, initargs=_args) as pool:
                # High-k modes are slowest: distribute them one at a time.
                results = pool.map(_pool_solve_k, k_arr, chunksize=1)
        except OSError:
            results = [evolve_k(k, bg, thermo, pgrid, tau_out) for k in k_arr]

    sources_j0 = np.array([r[0] for r in results])
    sources_j1 = np.array([r[1] for r in results])
    sources_j2 = np.array([r[2] for r in results])
    sources_E = np.array([r[3] for r in results])

    # --- Interpolate source functions to finer k-grid ---
    # Source functions are smooth in k, but the transfer function Δ_ℓ(k)
    # oscillates rapidly due to Bessel function ringing. A fine k-grid is
    # needed for accurate ∫|Δ|² d(ln k) integration.
    nk_fine = len(k_fine)
    lnk_ode = np.log(k_arr)
    lnk_fine = np.log(k_fine)
    src_fine_j0, src_fine_j1, src_fine_j2, src_fine_E = [
        _akima_columns(lnk_ode, source, lnk_fine)
        for source in (sources_j0, sources_j1, sources_j2, sources_E)]
    if not all(np.all(np.isfinite(v)) for v in
               (src_fine_j0, src_fine_j1, src_fine_j2, src_fine_E)):
        raise RuntimeError("Non-finite interpolated source functions")
    print(f"Interpolated sources: {nk} → {nk_fine} k-modes")

    # --- Line-of-sight integration with precomputed Bessel tables ---
    print("Computing transfer functions (line-of-sight integration)...")
    ell_max = params['ell_max']
    if ells_compute is None:
        ells_compute = ell_grid(ell_max, ell_step)
    else:
        requested = np.asarray(ells_compute)
        if (requested.ndim != 1 or not np.all(np.isfinite(requested))
                or np.any(requested != np.floor(requested))
                or np.any(requested < 2) or np.any(requested > ell_max)):
            raise ValueError("ells_compute must contain integer multipoles in [2, ell_max]")
        ells_compute = np.unique(np.r_[2, requested, ell_max]).astype(int)
    nell = len(ells_compute)
    print(f"  {nell} ℓ-values from {ells_compute[0]} to {ells_compute[-1]}")

    chi_arr = tau0 - tau_out   # comoving distance array
    chi_star = tau0 - tau_star
    chi_max = chi_arr.max()

    # Transfer functions: Δ_ℓ^T(k) and Δ_ℓ^E(k) on fine k-grid
    Delta_T = np.zeros((nell, nk_fine))
    Delta_E = np.zeros((nell, nk_fine))

    # Fixed tables plus O(n_k) outputs; no O(n_k*n_tau) scratch per thread.
    x0_tab, inv_dx_tab, n_x_tab, jl_tab, jl1_tab = _build_bessel_tables(
        ells_compute, float(k_fine[-1] * chi_max) + 2.0, bessel_dx)
    weights = _trapezoid_weights(tau_out)

    def _compute_ell_transfer(il, ell):
        x_lo = max(0.0, ell - 4.0 * ell**(1.0/3.0))
        ik_lo = max(0, np.searchsorted(k_fine, x_lo / chi_max) - 1)
        k_hi = (ell + 2500) / chi_star if chi_star > 0 else k_fine[-1]
        ik_hi = min(nk_fine, np.searchsorted(k_fine, k_hi) + 1)
        args = (ell, k_fine[ik_lo:ik_hi], chi_arr, weights,
                src_fine_j0[ik_lo:ik_hi], src_fine_j1[ik_lo:ik_hi],
                src_fine_j2[ik_lo:ik_hi], src_fine_E[ik_lo:ik_hi],
                x0_tab, inv_dx_tab, n_x_tab, jl_tab[il], jl1_tab[il])
        if NUMBA_AVAILABLE:
            dt, de = _los_integrals(*args)
        else:
            dt, de = _los_numpy(*args)
        Delta_T[il, ik_lo:ik_hi], Delta_E[il, ik_lo:ik_hi] = dt, de

    los_workers = min(4, os.cpu_count() or 1, nell) if los_workers is None else los_workers
    if int(los_workers) != los_workers or los_workers < 1:
        raise ValueError("los_workers must be a positive integer")
    # Compile once, outside worker threads; subsequent calls release the GIL.
    _compute_ell_transfer(0, int(ells_compute[0]))
    if nell > 1:
        with ThreadPoolExecutor(max_workers=int(los_workers)) as pool:
            futures = [pool.submit(_compute_ell_transfer, il, int(ells_compute[il]))
                       for il in range(1, nell)]
            for future in futures:
                future.result()

    # --- Power spectrum assembly ---
    # C_ℓ^XY = 4π ∫ d(ln k) P(k) Δ_ℓ^X(k) Δ_ℓ^Y(k)
    print("Assembling power spectra...")
    k_pivot = params['k_pivot']
    A_s = params['A_s']
    n_s = params['n_s']
    # Primordial power spectrum: P(k) = A_s × (k/k_pivot)^(n_s - 1)
    Pk = A_s * (k_fine / k_pivot)**(n_s - 1.0)

    # The LOS step zeros transfer functions outside the low/high-k cutoffs.
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
    T0_muK2 = (params['T_cmb'] * 1e6)**2
    Cl_TT *= T0_muK2
    Cl_EE *= T0_muK2
    Cl_TE *= T0_muK2

    # Cubic interpolation resolves acoustic features between sampled multipoles.
    ells_all = np.arange(2, ell_max + 1)
    Dl_TT, Dl_EE, Dl_TE = _interpolate_spectra(
        ells_compute, Cl_TT, Cl_EE, Cl_TE, ells_all)

    print("Done!")
    return {
        'ells': ells_all,
        'Dl_TT': Dl_TT,    # D_ℓ^TT = ℓ(ℓ+1)Cℓ^TT/(2π) in μK²
        'Dl_EE': Dl_EE,
        'Dl_TE': Dl_TE,
        'k_fine': k_fine,     # Fine k-grid (for transfer functions)
        'ells_compute': ells_compute,
        'Delta_T': Delta_T,   # Transfer functions on k_fine grid
        'Delta_E': Delta_E,
        'Dl_TT_compute': Cl_TT, 'Dl_EE_compute': Cl_EE, 'Dl_TE_compute': Cl_TE,
        'settings': dict(N_k_ode=nk, N_k_fine=nk_fine, N_tau=ntau,
                         ode_rtol=ode_rtol, ode_atol=ode_atol,
                         ode_max_step=ode_max_step, bessel_dx=bessel_dx),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    bg = compute_background(params)
    print("=== Background ===")
    print(f"H₀ = {bg['H0'] * c_km_s:.2f} km/s/Mpc")
    print(f"τ₀ = {bg['tau0']:.2f} Mpc")
    print(f"τ_eq = {bg['tau_eq']:.2f} Mpc")

    thermo = compute_thermodynamics(bg, params)
    print(f"\n=== Thermodynamics ===")
    print(f"z* = {thermo['z_star']:.1f}")
    print(f"τ* = {thermo['tau_star']:.2f} Mpc")
    print(f"r_s = {thermo['r_s']:.2f} Mpc")
    print(f"k_D = {thermo['k_D']:.4f} Mpc⁻¹")
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
