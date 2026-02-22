"""Benchmark nanoCMB accuracy across a Latin hypercube of cosmologies."""

import numpy as np
import time
import camb
from nanocmb import (compute_background, compute_thermodynamics, compute_cls,
                     c_km_s, params as default_params)

# Planck 2018 TT,TE,EE+lowE best-fit ± 1σ (Table 1 of arXiv:1807.06209)
param_ranges = {
    'omega_b_h2':   (0.02237, 0.00015),
    'omega_c_h2':   (0.1200,  0.0012),
    'h':            (0.6736,  0.0054),
    'n_s':          (0.9649,  0.0042),
    'ln10e10As':    (3.044,   0.014),    # sample in log-space, convert to A_s
    'tau_reion':    (0.0544,  0.0073),
}

N_SIGMA = 3
N_SAMPLES = 50
SEED = 42


def latin_hypercube(n_samples, n_dims, rng):
    """Generate a Latin hypercube sample in [0, 1]^n_dims."""
    result = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        result[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples
    return result


def make_param_sets(n_samples, seed):
    """Generate n_samples cosmologies via Latin hypercube over ±3σ."""
    rng = np.random.default_rng(seed)
    names = list(param_ranges.keys())
    n_dims = len(names)

    lhs = latin_hypercube(n_samples, n_dims, rng)

    param_sets = []
    for i in range(n_samples):
        p = dict(default_params)
        for j, name in enumerate(names):
            mean, sigma = param_ranges[name]
            # Map [0, 1] -> [-N_SIGMA, +N_SIGMA] sigma
            val = mean + (2 * lhs[i, j] - 1) * N_SIGMA * sigma
            if name == 'ln10e10As':
                p['A_s'] = np.exp(val) * 1e-10
            else:
                p[name] = val
        # Ensure physical
        p['tau_reion'] = max(p['tau_reion'], 0.01)
        p['A_s'] = max(p['A_s'], 1e-10)
        param_sets.append(p)
    return param_sets


def run_camb(p):
    """Run CAMB for a given parameter set, return D_ell arrays."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=p['h'] * 100,
        ombh2=p['omega_b_h2'],
        omch2=p['omega_c_h2'],
        tau=p['tau_reion'],
        nnu=p['N_eff'],
        num_massive_neutrinos=0,
        mnu=0,
        TCMB=p['T_cmb'],
        YHe=p['Y_He'],
    )
    pars.set_accuracy(AccuracyBoost=3)
    pars.InitPower.set_params(As=p['A_s'], ns=p['n_s'], r=0)
    pars.set_for_lmax(p['ell_max'] + 100, lens_potential_accuracy=0)
    pars.DoLensing = False
    pars.WantTensors = False
    pars.max_l = p['ell_max'] + 100
    pars.max_eta_k = 25000

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['unlensed_scalar']
    ells = np.arange(totCL.shape[0])
    return ells, totCL[:, 0], totCL[:, 1], totCL[:, 3]  # TT, EE, TE


def run_nanocmb(p):
    """Run nanoCMB for a given parameter set, return D_ell arrays."""
    bg = compute_background(p)
    thermo = compute_thermodynamics(bg, p)
    result = compute_cls(bg, thermo, p)
    return result['ells'], result['Dl_TT'], result['Dl_EE'], result['Dl_TE']


def accuracy_stats(ells, nano, ref, ell_ranges):
    """Compute mean ratio and std by ell range."""
    stats = {}
    for lmin, lmax in ell_ranges:
        mask = (ells >= lmin) & (ells < lmax) & (ref > 0)
        if mask.sum() > 0:
            ratio = nano[mask] / ref[mask]
            stats[(lmin, lmax)] = {
                'mean': np.mean(ratio),
                'std': np.std(ratio),
                'max_resid': np.max(np.abs(ratio - 1)),
            }
    return stats


def main():
    print(f"Generating {N_SAMPLES} cosmologies (±{N_SIGMA}σ Latin hypercube)...")
    param_sets = make_param_sets(N_SAMPLES, SEED)

    ell_ranges = [(2, 30), (30, 500), (500, 2000), (2000, 2501)]

    all_stats_TT = []
    all_stats_EE = []
    timings_nano = []
    timings_camb = []

    for i, p in enumerate(param_sets):
        label = (f"[{i+1:2d}/{N_SAMPLES}] ωb={p['omega_b_h2']:.5f} ωc={p['omega_c_h2']:.4f} "
                 f"h={p['h']:.4f} ns={p['n_s']:.4f} τ={p['tau_reion']:.4f}")
        print(f"\n{label}")

        # CAMB
        t0 = time.time()
        try:
            ells_c, TT_c, EE_c, TE_c = run_camb(p)
        except Exception as e:
            print(f"  CAMB FAILED: {e}")
            continue
        t_camb = time.time() - t0
        timings_camb.append(t_camb)

        # nanoCMB
        t0 = time.time()
        try:
            ells_n, TT_n, EE_n, TE_n = run_nanocmb(p)
        except Exception as e:
            print(f"  nanoCMB FAILED: {e}")
            continue
        t_nano = time.time() - t0
        timings_nano.append(t_nano)

        # Match ell ranges
        ell_max = min(ells_n[-1], len(TT_c) - 1)
        ell_min = ells_n[0]
        mask_c = (ells_c >= ell_min) & (ells_c <= ell_max)
        mask_n = (ells_n >= ell_min) & (ells_n <= ell_max)
        ells = ells_c[mask_c]

        stats_TT = accuracy_stats(ells, TT_n[mask_n], TT_c[mask_c], ell_ranges)
        stats_EE = accuracy_stats(ells, EE_n[mask_n], EE_c[mask_c], ell_ranges)
        all_stats_TT.append(stats_TT)
        all_stats_EE.append(stats_EE)

        # Print per-sample summary
        worst_TT = max((s['max_resid'] for s in stats_TT.values()), default=0)
        worst_EE = max((s['max_resid'] for s in stats_EE.values()), default=0)
        print(f"  CAMB: {t_camb:.1f}s  nanoCMB: {t_nano:.1f}s  "
              f"worst TT: {worst_TT:.4f}  worst EE: {worst_EE:.4f}")

    # === Summary ===
    n_ok = len(all_stats_TT)
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_ok}/{N_SAMPLES} cosmologies completed successfully")
    print(f"{'='*70}")

    # Timing
    print(f"\nTiming (nanoCMB): median {np.median(timings_nano):.1f}s, "
          f"mean {np.mean(timings_nano):.1f}s, range [{np.min(timings_nano):.1f}, {np.max(timings_nano):.1f}]s")
    print(f"Timing (CAMB):    median {np.median(timings_camb):.1f}s, "
          f"mean {np.mean(timings_camb):.1f}s, range [{np.min(timings_camb):.1f}, {np.max(timings_camb):.1f}]s")

    # Accuracy by ell range
    for spec, all_stats in [('TT', all_stats_TT), ('EE', all_stats_EE)]:
        print(f"\n{spec} accuracy across {n_ok} cosmologies:")
        print(f"  {'ell range':>15s}  {'median |resid|':>14s}  {'95th %ile':>10s}  {'worst case':>10s}  {'median std':>10s}")
        for lmin, lmax in ell_ranges:
            max_resids = [s[(lmin, lmax)]['max_resid'] for s in all_stats if (lmin, lmax) in s]
            stds = [s[(lmin, lmax)]['std'] for s in all_stats if (lmin, lmax) in s]
            if max_resids:
                print(f"  [{lmin:4d}, {lmax:4d})     "
                      f"{np.median(max_resids):12.4f}  "
                      f"{np.percentile(max_resids, 95):10.4f}  "
                      f"{np.max(max_resids):10.4f}  "
                      f"{np.median(stds):10.4f}")

    # Overall worst case
    all_worst_TT = [max(s['max_resid'] for s in st.values()) for st in all_stats_TT]
    all_worst_EE = [max(s['max_resid'] for s in st.values()) for st in all_stats_EE]
    print(f"\nOverall worst-case max|residual|:")
    print(f"  TT: {np.max(all_worst_TT):.4f} ({np.max(all_worst_TT)*100:.2f}%)")
    print(f"  EE: {np.max(all_worst_EE):.4f} ({np.max(all_worst_EE)*100:.2f}%)")
    print(f"  TT median: {np.median(all_worst_TT):.4f} ({np.median(all_worst_TT)*100:.2f}%)")
    print(f"  EE median: {np.median(all_worst_EE):.4f} ({np.median(all_worst_EE)*100:.2f}%)")

    # Save results
    np.savez('benchmark_results.npz',
             param_sets=[{k: v for k, v in p.items()} for p in param_sets[:n_ok]],
             timings_nano=timings_nano,
             timings_camb=timings_camb,
             worst_TT=all_worst_TT,
             worst_EE=all_worst_EE)
    print("\nSaved benchmark_results.npz")


if __name__ == '__main__':
    main()
