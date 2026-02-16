# nanoCMB

A minimal CMB angular power spectrum calculator in ~1k lines of Python. 

Computes the TT, EE, and TE angular power spectra for flat LCDM cosmologies from first principles: Friedmann equation, recombination, Boltzmann hierarchy, line-of-sight integration. Matches [CAMB](https://github.com/cmbant/CAMB) to sub-percent accuracy across the acoustic peaks.

## Quick start

```bash
python nanocmb.py
```

Runs in under a minute on a modern multi-core machine (thread-parallel LOS integration + multiprocess ODE evolution).

Output is saved to `nanocmb_output.npz` with arrays `ells`, `DlTT`, `DlEE`, `DlTE` (D_l in muK^2).

## Validation

Compare against CAMB and generate plots:

```bash
pip install camb matplotlib
python validate.py
```

This produces `plots/{tt,ee,te}_spectrum.png`, `plots/{tt,ee,te}_residuals.png`, and `plots/background_checks.png`.

## Accuracy

Validated against CAMB with Planck 2018 best-fit parameters:

| l range | TT (mean ratio) | TT (std) | EE (mean ratio) | EE (std) |
|---------|:---:|:---:|:---:|:---:|
| 2-30 | 0.999 | 0.5% | - | - |
| 30-500 | 1.004 | 0.3% | 0.995 | 0.4% |
| 500-2000 | 1.001 | 0.2% | 0.994 | 0.2% |
| 2000-2500 | 1.000 | 0.1% | 0.997 | 0.3% |

## What's inside

The entire calculation lives in `nanocmb.py`, structured as a top-to-bottom narrative:

1. **Background cosmology** -- Friedmann equation, conformal time, sound speed
2. **Recombination** -- Full RECFAST recombination (H + He ODEs, matter temperature, Hswitch), visibility function
3. **Perturbations** -- Boltzmann hierarchy in synchronous gauge (CDM frame) with tight-coupling approximation
4. **Line-of-sight integration** -- Multi-channel IBP decomposition (j_l, j_l', j_l'' channels) with jv-based Bessel functions
5. **Power spectrum assembly** -- Primordial spectrum, k-integration, l-interpolation

## Approximations

- Flat geometry (K = 0)
- Massless neutrinos only
- Cosmological constant (w = -1)
- No lensing, no tensors, no isocurvature modes
- First-order tight-coupling approximation

## Dependencies

- numpy
- scipy
- numba (optional for speedup)

That's it. CAMB and matplotlib are only needed for `validate.py`.

## Default parameters

Planck 2018 best-fit LCDM: H0 = 67.36, omega_b h^2 = 0.02237, omega_c h^2 = 0.1200, tau = 0.0544, n_s = 0.9649, A_s = 2.1e-9, N_eff = 3.044.
