"""Benchmark optimal tau grid: speed and accuracy vs CAMB."""

import numpy as np
import time

from nanocmb import (compute_background, compute_thermodynamics, compute_cls,
                     build_k_arr, build_tau_out, params,
                     optimal_tau_grid, cosmo_params_from_nanocmb)


def run(bg, thermo, camb_data, tau_out=None, label=""):
    t0 = time.time()
    result = compute_cls(bg, thermo, params, tau_out=tau_out)
    elapsed = time.time() - t0

    ells = result['ells']
    stats = {'time': elapsed, 'label': label}

    for spec, key in [('TT', 'Dl_TT'), ('EE', 'Dl_EE'), ('TE', 'Dl_TE')]:
        nano = result[key]
        camb_interp = np.interp(ells, camb_data['ells'], camb_data[f'Dl_{spec}'])
        for lmin, lmax in [(2, 30), (30, 500), (500, 2500)]:
            mask = (ells >= lmin) & (ells < lmax)
            if spec in ('TT', 'EE'):
                ref = camb_interp[mask]
                pos = ref > (0.01 * np.max(ref))
                if pos.sum() > 0:
                    ratio = nano[mask][pos] / ref[pos]
                    stats[f'{spec}_{lmin}_{lmax}_mean'] = np.mean(ratio)
                    stats[f'{spec}_{lmin}_{lmax}_std'] = np.std(ratio)
            elif spec == 'TE':
                ref = camb_interp[mask]
                sig = np.abs(ref) > 5.0
                if sig.sum() > 0:
                    ratio = nano[mask][sig] / ref[sig]
                    stats[f'{spec}_{lmin}_{lmax}_mean'] = np.mean(ratio)
                    stats[f'{spec}_{lmin}_{lmax}_std'] = np.std(ratio)
    return stats


def fmt_ratio(s, spec, lmin, lmax):
    mk = f'{spec}_{lmin}_{lmax}_mean'
    sk = f'{spec}_{lmin}_{lmax}_std'
    if mk in s:
        return f"{s[mk]:.4f}+/-{s[sk]:.4f}"
    return "    ---    "


def main():
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)

    # CAMB reference
    print("Computing CAMB reference...")
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544,
                       nnu=3.044, num_massive_neutrinos=0, mnu=0,
                       TCMB=2.7255, YHe=0.245)
    pars.set_accuracy(AccuracyBoost=3)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    pars.set_for_lmax(2600, lens_potential_accuracy=0)
    pars.DoLensing = False
    pars.WantTensors = False
    camb_results = camb.get_results(pars)
    powers = camb_results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['unlensed_scalar']
    ells_camb = np.arange(totCL.shape[0])
    camb_data = {
        'ells': ells_camb,
        'Dl_TT': totCL[:, 0],
        'Dl_EE': totCL[:, 1],
        'Dl_TE': totCL[:, 3],
    }
    print("Done.\n")

    tau_default = build_tau_out(thermo, bg['tau0'])
    k_default = build_k_arr()
    N_default = len(tau_default)
    cosmo = cosmo_params_from_nanocmb(bg, thermo, params)

    all_results = []

    # Default baseline
    print("=" * 60)
    print(f"Default tau (N={N_default})")
    print("=" * 60)
    s = run(bg, thermo, camb_data, label=f"default tau N={N_default}")
    all_results.append(s)
    print(f"  Time: {s['time']:.1f}s\n")

    # Optimal tau at same N
    print("=" * 60)
    print(f"Optimal tau (N={N_default})")
    print("=" * 60)
    tau_opt = optimal_tau_grid(N=N_default, k_max=k_default[-1],
                              tau_min=tau_default[0], tau_max=tau_default[-1],
                              cosmo=cosmo)
    s = run(bg, thermo, camb_data, tau_out=tau_opt,
            label=f"optimal tau N={N_default}")
    all_results.append(s)
    print(f"  Time: {s['time']:.1f}s\n")

    # Reduced optimal tau
    for N in [1000, 700, 500, 350, 200]:
        print("=" * 60)
        print(f"Optimal tau (N={N})")
        print("=" * 60)
        tau_opt = optimal_tau_grid(N=N, k_max=k_default[-1],
                                  tau_min=tau_default[0], tau_max=tau_default[-1],
                                  cosmo=cosmo)
        s = run(bg, thermo, camb_data, tau_out=tau_opt,
                label=f"optimal tau N={N}")
        all_results.append(s)
        print(f"  Time: {s['time']:.1f}s\n")

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    header = (f"{'Configuration':<30s} {'Time':>6s}  "
              f"{'TT 2-30':>16s} {'TT 30-500':>16s} {'TT 500+':>16s}  "
              f"{'EE 2-30':>16s} {'EE 30-500':>16s} {'EE 500+':>16s}")
    print(header)
    print("-" * 130)

    for s in all_results:
        line = f"{s['label']:<30s} {s['time']:5.1f}s  "
        for spec in ('TT', 'EE'):
            for lmin, lmax in [(2, 30), (30, 500), (500, 2500)]:
                line += f" {fmt_ratio(s, spec, lmin, lmax):>16s}"
            line += "  "
        print(line)


if __name__ == "__main__":
    main()
