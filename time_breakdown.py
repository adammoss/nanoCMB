"""Time each stage of compute_cls separately."""

import numpy as np
import time
from nanocmb import (compute_background, compute_thermodynamics, params,
                     build_k_arr, build_tau_out, setup_perturbation_grid,
                     evolve_k, _pool_init, _pool_solve_k, _boltzmann_rhs,
                     NVAR)
from scipy import interpolate
from optk import optimal_k_grid


def time_pipeline(bg, thermo, k_arr, label=""):
    pgrid = setup_perturbation_grid(bg, thermo)
    tau0 = bg['tau0']
    tau_out = build_tau_out(thermo, tau0)
    nk = len(k_arr)

    # Stage 1: ODE evolution
    _args = (bg, thermo, pgrid, tau_out)
    _boltzmann_rhs(tau_out[0], np.zeros(NVAR), k_arr[0],
                   pgrid['bg_vec'], pgrid['sp_a_x'], pgrid['sp_a_c'],
                   pgrid['sp_op_x'], pgrid['sp_op_c'])

    t0 = time.time()
    from multiprocessing import Pool, cpu_count
    ncpu = cpu_count()
    with Pool(ncpu, initializer=_pool_init, initargs=_args) as pool:
        results = pool.map(_pool_solve_k, k_arr)
    t_ode = time.time() - t0

    sources_j0 = np.array([r[0] for r in results])
    sources_j1 = np.array([r[1] for r in results])
    sources_j2 = np.array([r[2] for r in results])
    sources_E = np.array([r[3] for r in results])

    # Stage 2: Interpolation to fine grid
    t0 = time.time()
    nk_fine = 4000
    k_lin_start = max(0.002, k_arr[0])
    n_log = 80
    k_fine = np.unique(np.concatenate([
        np.logspace(np.log10(k_arr[0]), np.log10(k_lin_start), n_log),
        np.linspace(k_lin_start, k_arr[-1], nk_fine - n_log),
    ]))
    nk_fine = len(k_fine)
    lnk_ode = np.log(k_arr)
    lnk_fine = np.log(k_fine)
    ntau = len(tau_out)
    src_fine_j0 = np.zeros((nk_fine, ntau))
    src_fine_j1 = np.zeros((nk_fine, ntau))
    src_fine_j2 = np.zeros((nk_fine, ntau))
    src_fine_E = np.zeros((nk_fine, ntau))
    for it in range(ntau):
        src_fine_j0[:, it] = interpolate.Akima1DInterpolator(lnk_ode, sources_j0[:, it])(lnk_fine)
        src_fine_j1[:, it] = interpolate.Akima1DInterpolator(lnk_ode, sources_j1[:, it])(lnk_fine)
        src_fine_j2[:, it] = interpolate.Akima1DInterpolator(lnk_ode, sources_j2[:, it])(lnk_fine)
        src_fine_E[:, it] = interpolate.Akima1DInterpolator(lnk_ode, sources_E[:, it])(lnk_fine)
    t_interp = time.time() - t0

    print(f"{label:<35s}  ODE({nk:3d}k): {t_ode:5.1f}s  "
          f"Interp: {t_interp:5.1f}s  "
          f"Total ODE+Interp: {t_ode+t_interp:5.1f}s  "
          f"(ODE per batch: {t_ode/np.ceil(nk/ncpu):.2f}s x {int(np.ceil(nk/ncpu))} batches)")


def main():
    bg = compute_background(params)
    thermo = compute_thermodynamics(bg, params)

    k_default = build_k_arr()
    k_min, k_max = k_default[0], k_default[-1]

    print(f"CPU count: 12\n")
    print(f"{'Config':<35s}  {'ODE':>12s}  {'Interp':>8s}  {'Total':>16s}  {'Detail'}")
    print("-" * 110)

    time_pipeline(bg, thermo, k_default, "default (338)")

    for N in [100, 150, 200, 250, 338]:
        k_opt = optimal_k_grid(N=N, mode="ode", k_min=k_min, k_max=k_max)
        time_pipeline(bg, thermo, k_opt, f"optimal ODE N={N}")


if __name__ == "__main__":
    main()
