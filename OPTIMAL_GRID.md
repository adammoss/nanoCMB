# Optimal $k$-grid selection for CMB power spectrum computation

## Principle

The angular power spectrum $C_\ell$ requires evaluating a nested integral: an
inner line-of-sight integral over conformal time $\tau$ to obtain the transfer
function $\Delta_\ell(k)$, followed by an outer integral over wavenumber $k$
weighted by the primordial power spectrum. The accuracy of the final $C_\ell$
depends critically on how the discrete $k$-grid points are distributed. We use
the principle of quadrature error equidistribution to place $k$-nodes optimally
given a fixed computational budget of $N$ modes.

For the trapezoidal rule on a non-uniform grid $\{k_i\}$ in $x = \ln k$ space,
the local quadrature error on the interval $[x_i, x_{i+1}]$ scales as
$h_i^3 |I''(\xi_i)|$, where $h_i = x_{i+1} - x_i$ is the local spacing and
$I(x)$ is the integrand expressed in log-$k$. Minimising the total error subject
to the constraint that the grid spans a fixed range with $N$ points yields the
optimal local spacing $h_i \propto |I''(x_i)|^{-1/3}$, or equivalently an
optimal node density $\rho(x) \propto |I''(x)|^{1/3}$. The grid is constructed
by building the cumulative distribution function
$\Phi(x) = \int_{x_\mathrm{min}}^{x} \rho(x') \, dx'$ and placing nodes at
equal quantiles of $\Phi$.

Since the true integrand is not known before the computation, we use an analytic
model for $|I''|$ based on the known physical structure of the CMB transfer
functions. The cosmological parameters entering this model (sound horizon $r_s$,
damping scale $k_D$, recombination width $\Delta\tau_\mathrm{rec}$, etc.) are
taken from the pre-computed background and thermodynamics rather than hardcoded,
so the grid adapts to the specific cosmology.

## $C_\ell$ integration grid (`mode="cl"`)

For the $C_\ell$ integral the integrand (in log-$k$ space) is
$I(x) = k^{n_s+2} \, |\Delta_\ell(k)|^2$, combining the primordial spectrum,
the volume element, and the squared transfer function. The curvature of this
integrand is modelled as a sum over representative $\ell$-values (log-spaced
from $\ell_\mathrm{min}$ to $\ell_\mathrm{max}$), each contributing:

- A **Bessel window**: for each $\ell$, the spherical Bessel function
  $j_\ell(k\chi_*)$ selects wavenumbers around $k \sim \ell/\chi_*$, where
  $\chi_* = \tau_0 - \tau_*$ is the comoving distance to last scattering. This
  is modelled as a Gaussian envelope centred at $k_\mathrm{peak} = \ell/\chi_*$
  with width $\sigma_k = 1/\Delta\tau_\mathrm{rec}$, broadened by a factor of 3
  to approximate the non-Gaussian tails of the true Bessel window.

- **Acoustic oscillation curvature**: within each Bessel window, the transfer
  function oscillates with period $\Delta k \sim \pi/r_s$. The second derivative
  from these oscillations contributes $\sim r_s^{-2}$ times the envelope.

- **Smooth curvature**: for low $\ell$ where the acoustic oscillation period
  exceeds the Bessel window width, the curvature is instead set by the envelope
  shape, contributing $\sim \sigma_k^2$ times the envelope. The weight takes the
  larger of the acoustic and smooth contributions.

- **ISW contribution** ($\ell < 100$): an additional broad Gaussian window
  centred at $k \sim \ell/\chi_\mathrm{reion}$ with width set by the
  reionization duration, weighted by an adjustable `isw_weight` parameter.

All contributions are multiplied by the primordial factor $k^{n_s+2}$ and a
damping envelope $e^{-2k^2/k_D^2}$, floored at 0.02 to ensure the grid retains
some coverage in the damping tail beyond $k_D$ where the true damping factor
becomes very small but the power spectrum is still measurable.

## ODE source function grid (`mode="ode"`)

When the $k$-grid is used for solving the Boltzmann hierarchy and storing the
source functions $S(k, \tau)$, the optimisation target changes. The grid must
support accurate interpolation of $S$ in $k$ at any fixed $\tau$, independent of
$\ell$. The Bessel function windowing that makes the $C_\ell$ grid
$\ell$-dependent is absent here.

The variation of $S(k, \tau)$ with $k$ arises from the acoustic oscillations
(period $\pi/r_s$) modulated by the primordial spectrum and Silk damping. The
curvature model is:

- **Acoustic regime** ($k \gtrsim 1/r_s$): curvature $\sim r_s^{-2}$, constant
  across all $k$ in this range since the oscillation frequency is set by $r_s$.

- **Low-$k$ regime** ($k \lesssim 1/r_s$): the source is smooth and slowly
  varying. The curvature transitions to $\sim (k/k_\mathrm{tr})^2 \,
  \tau_\mathrm{eq}^{-2}$, where $k_\mathrm{tr} = 1/r_s$ and $\tau_\mathrm{eq}$
  is the conformal time at matter-radiation equality. This reflects the turn in
  the matter transfer function at the equality scale.

The weight takes the larger of the two regimes, multiplied by $k^{n_s+2}$ and
the floored damping envelope. The ODE mode uses a higher floor (0.5% of peak vs
$10^{-6}$ for $C_\ell$) to ensure more uniform coverage, since the source
functions at all $k$ contribute to some $\ell$ and the grid cannot be
$\ell$-selective.

The resulting ODE grid distributes nodes more uniformly across the acoustic
regime than the $C_\ell$ grid, without the concentration at high $k$ driven by
high-$\ell$ Bessel windows, and typically requires fewer points for the same
interpolation accuracy.
