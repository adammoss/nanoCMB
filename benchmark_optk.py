"""
Benchmark: how does reducing ODE and fine k-grid sizes with optimal
sampling affect speed and accuracy (vs CAMB)?
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanocmb import (compute_background, compute_thermodynamics, compute_cls,
                     build_k_arr, params, optimal_k_grid)


def run_and_validate(bg, thermo, camb_data, k_arr=None, k_fine=None, label=""):
    """Run compute_cls and return timing + accuracy stats vs CAMB."""
    t0 = time.time()
    result = compute_cls(bg, thermo, params, k_arr=k_arr, k_fine=k_fine)
    elapsed = time.time() - t0

    ells = result['ells']
    ell_min = ells[0]

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


def main():
    # Compute shared background/thermo
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

    k_default = build_k_arr()
    k_min, k_max = k_default[0], k_default[-1]
    N_default = len(k_default)

    all_results = []

    # --- 1. Default baseline ---
    print("=" * 60)
    print(f"Default (ODE={N_default}, fine=4000)")
    print("=" * 60)
    s = run_and_validate(bg, thermo, camb_data, label=f"default ODE={N_default} fine=4000")
    all_results.append(s)
    print(f"  Time: {s['time']:.1f}s\n")

    # --- 2. Vary ODE grid size (fine grid = default 4000) ---
    for N_ode in [150, 200, 250, 338]:
        print("=" * 60)
        print(f"Optimal ODE N={N_ode}, default fine=4000")
        print("=" * 60)
        k_ode = optimal_k_grid(N=N_ode, mode="ode", k_min=k_min, k_max=k_max)
        s = run_and_validate(bg, thermo, camb_data, k_arr=k_ode,
                             label=f"optODE={N_ode} fine=4000")
        all_results.append(s)
        print(f"  Time: {s['time']:.1f}s\n")

    # --- 3. Vary fine grid size (ODE = default) ---
    for N_fine in [1000, 2000, 3000, 4000]:
        print("=" * 60)
        print(f"Default ODE={N_default}, optimal fine N={N_fine}")
        print("=" * 60)
        k_fine = optimal_k_grid(N=N_fine, mode="cl", k_min=k_min, k_max=k_max)
        s = run_and_validate(bg, thermo, camb_data, k_fine=k_fine,
                             label=f"ODE={N_default} optFine={N_fine}")
        all_results.append(s)
        print(f"  Time: {s['time']:.1f}s\n")

    # --- 4. Both optimal, reduced ---
    for N_ode, N_fine in [(200, 2000), (250, 2000), (250, 3000), (338, 4000)]:
        print("=" * 60)
        print(f"Both optimal: ODE={N_ode}, fine={N_fine}")
        print("=" * 60)
        k_ode = optimal_k_grid(N=N_ode, mode="ode", k_min=k_min, k_max=k_max)
        k_fine = optimal_k_grid(N=N_fine, mode="cl", k_min=k_min, k_max=k_max)
        s = run_and_validate(bg, thermo, camb_data, k_arr=k_ode, k_fine=k_fine,
                             label=f"optODE={N_ode} optFine={N_fine}")
        all_results.append(s)
        print(f"  Time: {s['time']:.1f}s\n")

    # --- Summary table ---
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    header = (f"{'Configuration':<35s} {'Time':>6s}  "
              f"{'TT 2-30':>10s} {'TT 30-500':>10s} {'TT 500+':>10s}  "
              f"{'EE 2-30':>10s} {'EE 30-500':>10s} {'EE 500+':>10s}")
    print(header)
    print("-" * 120)

    def fmt_ratio(s, spec, lmin, lmax):
        mk = f'{spec}_{lmin}_{lmax}_mean'
        sk = f'{spec}_{lmin}_{lmax}_std'
        if mk in s:
            return f"{s[mk]:.4f}+/-{s[sk]:.4f}"
        return "    ---    "

    for s in all_results:
        line = f"{s['label']:<35s} {s['time']:5.1f}s  "
        for spec in ('TT', 'EE'):
            for lmin, lmax in [(2, 30), (30, 500), (500, 2500)]:
                line += f" {fmt_ratio(s, spec, lmin, lmax):>10s}"
            line += "  "
        print(line)

    # --- Plot: accuracy vs speed ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, spec, lmin, lmax, title in [
        (axes[0], 'TT', 30, 500, 'TT ell=30-500'),
        (axes[1], 'TT', 500, 2500, 'TT ell=500-2500'),
        (axes[2], 'EE', 500, 2500, 'EE ell=500-2500'),
    ]:
        mk = f'{spec}_{lmin}_{lmax}_mean'
        sk = f'{spec}_{lmin}_{lmax}_std'
        for s in all_results:
            if mk in s and sk in s:
                err = abs(s[mk] - 1.0) * 100
                std = s[sk] * 100
                ax.errorbar(s['time'], err, yerr=std, fmt='o', markersize=6,
                            capsize=3, alpha=0.7)
                ax.annotate(s['label'].replace(' ', '\n'), (s['time'], err),
                            fontsize=5, ha='left', va='bottom',
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("|mean ratio - 1| [%]")
        ax.set_title(title)
        ax.axhspan(0, 0.1, alpha=0.1, color="green", label="<0.1%")
        ax.axhspan(0.1, 0.5, alpha=0.1, color="orange", label="0.1-0.5%")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle("Speed vs accuracy trade-off with optimal k-grids",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/optk_benchmark.png", dpi=150, bbox_inches="tight")
    print("\nSaved plots/optk_benchmark.png")


if __name__ == "__main__":
    main()
