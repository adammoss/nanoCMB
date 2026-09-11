<div align="center">
<img width="400" alt="nanoCMB" src="assets/nanocmb.png">
<h1>nanoCMB</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)

**A minimal CMB angular power spectrum calculator in ~1600 lines of Python.**

Designed for students learning CMB theory, researchers prototyping new physics, or anyone who wants to understand what a Boltzmann solver actually does. The entire calculation lives in a single readable Python file.

Computes the unlensed TT, EE, and TE angular power spectra for flat LCDM cosmologies from first principles: Friedmann equation, RECFAST recombination, Boltzmann hierarchy in synchronous gauge, line-of-sight integration with precomputed Bessel tables, and optimally constructed non-uniform grids. Matches [CAMB](https://github.com/cmbant/CAMB) to sub-percent accuracy on the unlensed spectra. Since lensing is not included, the output should not be compared directly to observed CMB data.

Contributions welcome — whether it's improving speed, accuracy, conciseness, or adding new physics. **Pull requests encouraged!**
</div>

* [Quick start](#quick-start)
* [Validation](#validation)
* [Accuracy](#accuracy)
* [What's inside](#whats-inside)
* [Approximations](#approximations)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [Citation](#citation)

## Spectra

![TT Power Spectrum](assets/tt_spectrum.png)

![EE Power Spectrum](assets/ee_spectrum.png)

![TE Power Spectrum](assets/te_spectrum.png)

TE residuals use the covariance normalization described in [Accuracy](#accuracy).

## Quick start

```bash
python nanocmb.py
```

The default calculation takes about 7s with optional Numba JIT and warm caches on the benchmark machine; runtime depends on hardware. The first run takes longer as it compiles the numerical kernels and builds spherical Bessel function tables. Subsequent runs reuse the Bessel cache.

Against the previous implementation (`8c6191a`), median compute time fell from **10.51s to 7.08s (33% less time)** over three warm runs per version, using the default grids through l=2500. The timer includes background, thermodynamics, and spectrum calculation. This comparison used 12 logical CPUs, Python 3.12.12, NumPy 2.3.5, SciPy 1.17.0, and Numba 0.63.1.

Output is saved to `nanocmb_output.npz` with arrays `ells`, `DlTT`, `DlEE`, `DlTE` (D_l in muK^2).

For notebooks, `compute_cls(..., n_workers=1)` runs the ODEs serially. Script callers using multiple processes should put the calculation under `if __name__ == '__main__':`. The solver automatically falls back to serial execution for stdin/REPL entry points that spawned workers cannot import.

`compute_cls` also exposes `los_workers`, `ell_step`, `ells_compute`, `bessel_dx`, and `ode_rtol`/`ode_atol`/`ode_max_step` for numerical checks. Rebuild the background and thermodynamics dictionaries after changing cosmological parameters.

## Validation

Compare against CAMB and generate plots:

```bash
pip install camb matplotlib
python scripts/validate.py
```

This produces comparison plots in `plots/` with residual panels.

To regenerate the tracked spectrum figures embedded in this README:

```bash
python nanocmb.py
python scripts/validate.py --update-assets
```

Run the numerical regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Accuracy

Validated against CAMB (AccuracyBoost=3) with Planck 2018 best-fit parameters:

| l range | TT (mean ratio) | TT (std) | EE (mean ratio) | EE (std) |
|---------|:---:|:---:|:---:|:---:|
| 2-29 | 0.9994 | 0.02% | 1.0003 | 0.21% |
| 30-499 | 0.9994 | 0.06% | 1.0000 | 0.14% |
| 500-1999 | 0.9996 | 0.06% | 0.9998 | 0.05% |
| 2000-2500 | 0.9985 | 0.04% | 0.9988 | 0.06% |

Over l=2–2500, default-cosmology RMS residuals are 0.096% TT, 0.098% EE, and 0.052% TE. TE residuals are normalized by sqrt(TT_CAMB * EE_CAMB), avoiding divisions at TE zero crossings.

The updated solver was checked at the default cosmology, six Latin-hypercube cosmologies spanning +/-3 sigma of the Planck 2018 posterior, and a zero-reionization case. Across these eight cases, the largest absolute TT and EE residuals were 0.30% and 0.81%, respectively. The calculation uses massless neutrinos and matched unlensed CAMB spectra; these checks do not establish accuracy outside the tested parameter range.

## What's inside

The entire calculation lives in `nanocmb.py`, structured as a top-to-bottom pipeline:

1. **Background cosmology** -- Friedmann equation, conformal time, sound horizon
2. **Recombination** -- Full RECFAST (H + He ODEs, matter temperature, Hswitch corrections), reionisation, visibility function
3. **Grid construction** -- Optimal non-uniform grids in k and tau via error equidistribution
4. **Perturbations** -- Boltzmann hierarchy in synchronous gauge (CDM frame) with tight-coupling approximation
5. **Source functions** -- Multi-channel IBP decomposition with ISW, Doppler, and quadrupole terms
6. **Line-of-sight integration** -- Precomputed Bessel tables with dead-zone skipping and recurrence derivatives
7. **Power spectrum assembly** -- Primordial spectrum, k-integration on fine grid, l-interpolation

## Approximations

- Flat geometry (K = 0)
- Massless neutrinos only
- Cosmological constant (w = -1)
- No lensing, no tensors, no isocurvature modes
- First-order tight-coupling approximation

## Dependencies

- numpy
- scipy
- numba (optional; accelerates the perturbation and line-of-sight kernels)

That's it. CAMB and matplotlib are only needed for `validate.py`.

## Default parameters

Planck 2018 best-fit flat LCDM:

| Parameter | Symbol | Value |
|-----------|:------:|------:|
| Hubble parameter | H0 | 67.36 km/s/Mpc |
| Baryon density | omega_b h^2 | 0.02237 |
| CDM density | omega_c h^2 | 0.1200 |
| Optical depth | tau | 0.0544 |
| Scalar spectral index | n_s | 0.9649 |
| Scalar amplitude | A_s | 2.1e-9 |
| Effective neutrino number | N_eff | 3.044 |

## Contributing

Contributions are welcome! In particular:

- **Performance improvements** -- Faster ODE integration, better parallelisation, reduced memory usage, or other optimisations that cut runtime without sacrificing accuracy.
- **Accuracy improvements** -- Better tight-coupling schemes, higher-order corrections, improved grid strategies, or other changes that reduce residuals vs CAMB.
- **New physics** -- Extensions such as gravitational lensing, tensor modes, massive neutrinos, spatial curvature, or non-standard dark energy. The modular structure is designed to make these additions straightforward.
- **Bug fixes** -- If you find a discrepancy or numerical issue, please open an issue or submit a fix.

Please keep contributions consistent with the project philosophy: everything in a single readable Python file, minimal dependencies, and every approximation made explicit.

## Citation

If you use nanoCMB in your research, please cite the accompanying paper (submitted to Astronomy and Computing).

## License

MIT
